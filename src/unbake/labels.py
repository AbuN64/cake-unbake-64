"""Reconstruct an older build's labels from the address tables its link left, by the current build's table for the same
switch or pointer array.

An older address table (see `pointertype`) is a run of addresses of the older build. The current build holds the same
table when the run has the same entry count, the same zero entries and the same offsets between entries. Then every
label of the run is a place the current image names: a function and the offset inside it, or the data symbol before the
target and the offset after it. The older address of that owner is the owner's current address plus one delta, the
difference of the run and the table. Evidence rules:

- A run is matched only against current tables of at least `MIN_ENTRIES` words, whose every entry is an address of the
  linked image; a shorter run is listed, never matched, because any short run fits many tables.
- All labels of a run agree on one delta, or the run is refused with the best table's measured disagreement (labels
  that follow the most common delta, labels that do not, distinct deltas).
- A run is matched in its own version first, then in the project's other versions in order. Every table of the first
  version that has any must give the same symbols; tables that give different symbols refuse the run as ambiguous and
  list the symbols that differ. The three copies of one array a build holds give the same symbols and do not.
- Two runs of one version may not give one name two addresses: both are refused with the conflict. Two names for one
  address are aliases (the build holds two copies of the data) and both stay.
- A word that is an offset without the segment bits (24 bits) is no address of the image: it is listed with the reason
  and never matched.

The names are `g_earlier_<owner>` for the owner's start and `g_earlier_<owner>_<offset in hex>` for a label inside it,
the way splat names a label after its function; the comment of a row says which current table the row came from."""

from __future__ import annotations

import bisect
import re
import struct
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unbake import config, windows
from unbake.contracts import Config, Finding, Json, Refusal
from unbake.pointertype import RAM

MIN_ENTRIES = 4  # the fewest words of a run (and of a current table) that identify a table
SEGMENTLESS = 0x00100000  # a word below RAM and from here up is an offset with the segment bits dropped
_SHOWN = 4  # the most tables and symbols listed for one refusal
PREFIX = "g_earlier_"
_SYMBOL = re.compile(r"^(\w+)\s*=\s*0x([0-9A-Fa-f]+);")


@dataclass(frozen=True)
class Owner:
    """What a current address belongs to: a function (its extent is known) or the data symbol before it."""

    name: str
    start: int
    function: bool


@dataclass(frozen=True)
class Table:
    """A run of words in an image: addresses (0x80 form) and zero entries; `owners` has one Owner per address."""

    version: str
    at: int  # address of the first word
    words: tuple[int, ...]
    owners: tuple[Owner | None, ...]
    name: str  # the symbol the table starts at, or the data symbol before it


@dataclass(frozen=True)
class Run:
    """A run of the older ROM: `rom` of the first word and the words (zero or an address of the 0x80 form)."""

    rom: int
    words: tuple[int, ...]


class Owners:
    """Owner lookup of one image: functions by extent, anything else by the nearest symbol before in its section."""

    def __init__(self, symbols: Iterable[tuple[str, int, int, bool]], sections: Sequence[tuple[int, int]]) -> None:
        funcs = sorted((a, s, n) for n, a, s, f in symbols if f and s)
        self.funcs = [Owner(n, a, True) for a, _, n in funcs]
        self.fstarts = [a for a, _, _ in funcs]
        self.fends = [a + s for a, s, _ in funcs]
        data = sorted({(a, n) for n, a, s, f in symbols if not f})
        self.data = [Owner(n, a, False) for a, n in data]
        self.dstarts = [a for a, _ in data]
        self.sections = sorted(sections)

    def inside(self, addr: int) -> bool:
        return any(a <= addr < b for a, b in self.sections)

    def of(self, addr: int) -> Owner | None:
        i = bisect.bisect_right(self.fstarts, addr) - 1
        if i >= 0 and addr < self.fends[i]:
            return self.funcs[i]
        j = bisect.bisect_right(self.dstarts, addr) - 1
        if j >= 0 and self.inside(self.dstarts[j]) and self.sections_of(self.dstarts[j]) == self.sections_of(addr):
            return self.data[j]
        return None

    def sections_of(self, addr: int) -> tuple[int, int] | None:
        return next(((a, b) for a, b in self.sections if a <= addr < b), None)


def word_kind(word: int) -> str:
    """'zero', 'address' (RAM window), 'segmentless' (an offset with the segment bits dropped) or 'other'."""
    if word == 0:
        return "zero"
    if RAM[0] <= word < RAM[1]:
        return "address"
    return "segmentless" if SEGMENTLESS <= word < RAM[1] - RAM[0] else "other"


def runs_of(words: Sequence[int], first: int, segmentless: bool = False) -> tuple[list[Run], list[Json]]:
    """(runs, segmentless): maximal runs of address and zero words (a run starts and ends on an address; a zero joins
    only between two addresses), and the segment-less words with their ROM offsets. With `segmentless` such a word
    is an address of the 0x80 form in the run; without it, it breaks the run."""
    runs: list[Run] = []
    odd: list[Json] = []
    kinds = [word_kind(w) for w in words]
    if segmentless:
        words = [w | RAM[0] if k == "segmentless" else w for w, k in zip(words, kinds, strict=True)]
        kinds = ["address" if k == "segmentless" else k for k in kinds]
    else:
        odd = [
            {"rom": first + 4 * i, "word": f"0x{w:08X}"}
            for i, (w, k) in enumerate(zip(words, kinds, strict=True))
            if k == "segmentless"
        ]
    start: int | None = None
    for i, kind in enumerate([*kinds, "other"]):
        if kind == "address" and start is None:
            start = i
        if start is not None and kind not in ("address", "zero"):
            end = i
            while kinds[end - 1] != "address":
                end -= 1
            runs.append(Run(first + 4 * start, tuple(words[start:end])))
            start = None
    return runs, odd


def scan(
    version: str, sections: Iterable[tuple[int, bytes]], owners: Owners, names: Mapping[int, str] | None = None
) -> list[Table]:
    """The tables of an image: in each section, the maximal runs of address words whose every address the image owns
    (zero words join between two addresses), of at least MIN_ENTRIES words. Name = a symbol at the run's start."""
    out: list[Table] = []
    for base, data in sections:
        words = struct.unpack(f">{len(data) // 4}I", data[: len(data) // 4 * 4])
        words = tuple(w | RAM[0] if word_kind(w) == "segmentless" else w for w in words)
        runs, _ = runs_of([w if w == 0 or owners.of(w) else 1 for w in words], base)
        for run in runs:
            if len(run.words) < MIN_ENTRIES:
                continue
            at = run.rom
            seen = [owners.of(w) if w else None for w in run.words]
            near = owners.of(at)
            out.append(Table(version, at, run.words, tuple(seen), (names or {}).get(at) or (near.name if near else "")))
    return out


def _agreement(run: Run, table: Table, shift: int) -> tuple[Counter[int], bool]:
    """(deltas of the labels, same zero pattern) for the run laid on the table from word `shift`."""
    deltas: Counter[int] = Counter()
    for i, word in enumerate(run.words):
        cur = table.words[shift + i]
        if (word == 0) != (cur == 0):
            return Counter(), False
        if word:
            deltas[word - cur] += 1
    return deltas, True


def candidates(run: Run, tables: Sequence[Table]) -> tuple[list[tuple[Table, int, int]], Json | None]:
    """((table, shift, delta) of every table the run fits with one delta, the closest miss as a measured row)."""
    fits: list[tuple[Table, int, int]] = []
    best: tuple[int, Json] | None = None
    for table in tables:
        for shift in range(len(table.words) - len(run.words) + 1):
            deltas, same = _agreement(run, table, shift)
            if not same or not deltas:
                continue
            ((delta, most),) = deltas.most_common(1)
            if len(deltas) == 1:
                fits.append((table, shift, delta))
            elif best is None or most > best[0]:
                best = (
                    most,
                    {
                        "table": table.name,
                        "table_at": f"0x{table.at + 4 * shift:08X}",
                        "labels": sum(deltas.values()),
                        "agree": most,
                        "differ": sum(deltas.values()) - most,
                        "deltas": len(deltas),
                    },
                )
    return fits, None if best is None else best[1]


def claim(run: Run, table: Table, shift: int, delta: int) -> dict[str, int]:
    """name -> older address of every label of the run and the start of every function it lies in."""
    out: dict[str, int] = {}
    for i, word in enumerate(run.words):
        owner = table.owners[shift + i]
        if not word or owner is None:
            continue
        cur = table.words[shift + i]
        out[_name(owner, cur - owner.start)] = word
        if owner.function:
            out[_name(owner, 0)] = owner.start + delta
    return out


def _name(owner: Owner, offset: int) -> str:
    return PREFIX + owner.name + (f"_{offset:X}" if offset else "")


def _known(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        match = _SYMBOL.match(line.strip())
        if match:
            out.setdefault(match.group(1), int(match.group(2), 16))
    return out


def decide(run: Run, order: Sequence[str], tables: Mapping[str, Sequence[Table]]) -> tuple[Json, dict[str, int] | None]:
    """(row, claim): the run matched, or the row of its refusal. The first version in `order` with a fitting table
    decides; its tables must all give the same symbols."""
    miss: dict[str, Any] | None = None
    for version in order:
        fits, near = candidates(run, tables.get(version, ()))
        if near and (miss is None or near["agree"] > miss["agree"]):
            miss = {**near, "version": version}
        if not fits:
            continue
        claims = [claim(run, t, s, d) for t, s, d in fits]
        first = claims[0]
        apart = sorted({k for c in claims[1:] for k in c.keys() | first.keys() if c.get(k) != first.get(k)})
        if apart:
            return {
                "why": "ambiguous: tables of one version give different symbols",
                "version": version,
                "tables": len(fits),
                "differ": apart[:_SHOWN],
            }, None
        table, shift, delta = fits[0]
        return {
            "version": version,
            "table": table.name,
            "table_at": f"0x{table.at + 4 * shift:08X}",
            "copies": len(fits),
            "delta": delta,
        }, first
    row: dict[str, Any] = {"why": "no current table has these entries with one delta"}
    if miss:
        row["closest"] = miss
    return row, None


def resolve(matched: Sequence[tuple[Run, Json, dict[str, int]]]) -> tuple[list[int], list[Json]]:
    """(indices accepted, conflicts): runs giving one name two addresses are refused; aliases of one address stay."""
    by_name: dict[str, dict[int, set[int]]] = {}
    for i, (_, _, got) in enumerate(matched):
        for name, addr in got.items():
            by_name.setdefault(name, {}).setdefault(addr, set()).add(i)
    bad: dict[int, Json] = {}
    for name, where in by_name.items():
        if len(where) > 1:
            for ids in where.values():
                for i in ids:
                    bad.setdefault(
                        i,
                        {
                            "why": "conflict: one name has two addresses",
                            "name": name,
                            "addresses": sorted(f"0x{a:08X}" for a in where)[:_SHOWN],
                        },
                    )
    return [i for i in range(len(matched)) if i not in bad], [{"run": i, **row} for i, row in sorted(bad.items())]


def image(root: Path, project: str, version: str) -> tuple[Owners, list[tuple[int, bytes]]] | None:
    """The owners and the loaded sections of `build/<version>/<project>.elf`, without the windows of an older build."""
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection

    path = root / "build" / version / f"{project}.elf"
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        elf = ELFFile(handle)
        loaded = [
            s
            for s in elf.iter_sections()
            if s["sh_type"] == "SHT_PROGBITS"
            and s["sh_flags"] & 2
            and s["sh_addr"] >= RAM[0]
            and not windows.is_window(s.name)
        ]
        syms: list[tuple[str, int, int, bool]] = []
        for section in elf.iter_sections():
            if isinstance(section, SymbolTableSection):
                for y in section.iter_symbols():
                    kind = y["st_info"]["type"]
                    if y.name and kind in ("STT_FUNC", "STT_NOTYPE", "STT_OBJECT") and y["st_value"] >= RAM[0]:
                        syms.append((y.name, y["st_value"], y["st_size"], kind == "STT_FUNC"))
        owners = Owners(syms, [(s["sh_addr"], s["sh_addr"] + s["sh_size"]) for s in loaded])
        return owners, [(s["sh_addr"], s.data()) for s in loaded]


def command(cfg: Config, params: Json) -> Json:
    """`reconstruct-labels --spec V:LOW:HIGH:DELTA ...`: match each address run of the spans against the current
    builds' tables, write the older-build symbols of every accepted run to its version's earlier_build_symbols.txt
    (not with --check) and list every run not accepted with its measured disagreement."""
    specs = []
    for text in params.get("spec") or ():
        v, low, high, delta = text.split(":")
        specs.append((v, int(low, 0), int(high, 0), int(delta, 0)))
    if not specs:
        raise Refusal(Finding("reconstruct-labels.spec", "give --spec VERSION:LOW:HIGH:DELTA (hex ok) at least once"))
    root = cfg.project.root
    versions = list(cfg.project.versions)
    tables: dict[str, list[Table]] = {}
    for v in versions:
        built = image(root, cfg.project.name, v)
        if built is not None:
            tables[v] = scan(v, built[1], built[0])
    if not tables:
        raise Refusal(
            Finding("reconstruct-labels.image", "no build/<version>/<project>.elf: build before reconstructing")
        )
    rows: list[Json] = []
    unfit: list[Json] = []
    matched: list[tuple[Run, Json, dict[str, int]]] = []
    owner_of: list[str] = []
    for v, low, high, delta in specs:
        rom = config.rom(cfg.project, v).read_bytes()
        a, b = low & ~3, (high + 3) & ~3
        words = struct.unpack(f">{(b - a) // 4}I", rom[a:b])
        runs, odd = runs_of(words, a)
        unfit += [
            {"version": v, **o, "why": "an offset without the segment bits is no pointer the linker can place"}
            for o in odd
        ]
        order = [v, *[x for x in versions if x != v]]
        for run in runs:
            where = {"version": v, "rom": run.rom, "end": run.rom + 4 * len(run.words), "entries": len(run.words)}
            if len(run.words) < MIN_ENTRIES:
                unfit.append({**where, "why": f"fewer than {MIN_ENTRIES} entries identify no table"})
                continue
            row, got = decide(run, order, tables)
            if got is None:
                unfit.append({**where, **row})
            else:
                matched.append((run, {**where, **row, "delta_old": delta}, got))
                owner_of.append(v)
    keep: list[int] = []
    conflicts: list[Json] = []
    for v in sorted(set(owner_of)):
        at = [i for i, o in enumerate(owner_of) if o == v]
        kept, bad = resolve([matched[i] for i in at])
        keep += [at[j] for j in kept]
        conflicts += [{**c, "run": at[c["run"]]} for c in bad]
    keep.sort()
    for c in conflicts:
        _, row, _ = matched[c["run"]]
        unfit.append(
            {k: v for k, v in row.items() if k in ("version", "rom", "end", "entries")}
            | {k: v for k, v in c.items() if k != "run"}
        )
    added: dict[str, dict[str, tuple[int, str]]] = {}
    for i in keep:
        run, row, got = matched[i]
        rows.append({k: v for k, v in row.items() if k != "delta_old"} | {"symbols": len(got)})
        for name, addr in got.items():
            added.setdefault(owner_of[i], {})[name] = (addr, f"from {row['version']} {row['table']}@{row['table_at']}")
    if not params.get("check"):
        for v, syms in added.items():
            path = root / windows.symbols_path(v)
            have = path.read_text().splitlines() if path.exists() else []
            known = _known("\n".join(have))
            add = [
                f"{n} = 0x{a:08X}; // {why}"
                for n, (a, why) in sorted(syms.items(), key=lambda kv: kv[1][0])
                if n not in known
            ]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join([*have, *add]) + "\n")
    return {
        "tables": rows,
        "unfit": unfit,
        "symbols": {v: len(s) for v, s in sorted(added.items())},
        "labels": sum(r["symbols"] for r in rows),
    }
