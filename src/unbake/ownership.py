"""Data ownership derived from the compile: a C unit owns the ROM ranges its object emits as rodata and data."""

from __future__ import annotations

import io
import mmap
import re
from bisect import bisect_left, bisect_right, insort
from collections import OrderedDict, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, cast

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from elftools.elf.enums import ENUM_RELOC_TYPE_MIPS
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection

from unbake import config, effort, layout, native, pool, recipes, store, versions, view
from unbake.contracts import Claim, Finding, Placement, Recipe, Refusal, Snapshot, UnitSpec, code_identity, digest

_WORD = 0xFFFFFFFF
_MANY = 16  # more equal ranges than this are ambiguous without counting them all
_Row = tuple[int, int, str, str, int]  # rom start, rom end, member, section, vram
Located = dict[str, list[list[int]] | dict[str, str]]  # per emitted section its ROM ranges, or why it has none
Emitted = dict[str, tuple[bytes, dict[int, bool], list[tuple[int, int]]]]  # per output section bytes, relocs, refs
Built = tuple[Emitted, tuple[int, int] | None]  # a holder's emitted sections and its text's (vram, ROM offset)


class _Index(NamedTuple):
    rom: mmap.mmap
    rows: list[_Row]  # the data rows by ROM start
    row_of: dict[int, int]  # ROM start -> position in rows
    first: dict[int, list[int]]  # first word -> the row starts that hold it
    digest: str
    by_vram: list[tuple[int, int, int, int]]  # vram start, vram end, rom start, rom end
    segments: tuple[tuple[str, int, int, int], ...]


def _classes() -> dict[str, str]:
    """Input section name -> output section, for every allocated non-zero section but text (units.toml)."""
    rows = config.sections()
    return {
        name: "." + key
        for key, row in rows.items()
        if key != "text" and not row["zero"]
        for name in row["inputs"]
        if not name.endswith("*")
    }


def emitted(blob: bytes) -> Emitted:
    """Per output section the ld-concatenated bytes, relocated word offsets (True = to .text), .text low-half refs."""
    elf, classes = ELFFile(io.BytesIO(blob)), _classes()
    sections = list(elf.iter_sections())
    relocated: dict[str, dict[int, bool]] = defaultdict(dict)
    lows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    text = next((s.data() for s in sections if s.name == ".text"), b"")
    for section in sections:
        if isinstance(section, RelocationSection):
            symbols = sections[section["sh_link"]]
            assert isinstance(symbols, SymbolTableSection)
            for r in section.iter_relocations():
                symbol = symbols.get_symbol(r["r_info_sym"])
                index = symbol["st_shndx"]
                target = sections[section["sh_info"]].name
                if isinstance(index, int) and target in classes:
                    relocated[target][r["r_offset"]] = sections[index].name.startswith(".text")
                elif (
                    target == ".text"
                    and r["r_info_type"] == ENUM_RELOC_TYPE_MIPS["R_MIPS_LO16"]
                    and isinstance(index, int)
                ):
                    imm = int.from_bytes(text[r["r_offset"] + 2 : r["r_offset"] + 4], "big", signed=True)
                    imm += symbol["st_value"]
                    lows[sections[index].name].append((r["r_offset"], imm))
    out: dict[str, tuple[bytearray, dict[int, bool], list[Any]]] = {}
    for section in sections:
        if section.name in classes and section["sh_size"]:
            data, words, refs = out.setdefault(classes[section.name], (bytearray(), {}, []))
            data.extend(bytes(-len(data) % max(section["sh_addralign"], 1)))
            words.update({len(data) + o: t for o, t in relocated[section.name].items()})
            refs.extend((o, len(data) + imm) for o, imm in lows[section.name])
            data.extend(section.data())
    return {c: (bytes(d), w, r) for c, (d, w, r) in out.items()}


def _index(snapshot: Snapshot, version: str) -> _Index:
    def build() -> _Index:
        rom = versions.rom_view(snapshot.versions[version])
        rows = sorted(
            (p.rom_start, p.rom_end, m.name, p.section, p.vram)
            for m in snapshot.layout.members.values()
            if m.kind != "function" and sum(q.version == version for q in m.placements) == 1  # a name per row
            for p in m.placements
            if p.version == version and p.section in (".rodata", ".data") and p.rom_end > p.rom_start
        )
        first: dict[int, list[int]] = defaultdict(list)
        for start, *_ in rows:
            first[int.from_bytes(rom[start : start + 4], "big")].append(start)
        return _Index(
            rom,
            rows,
            {r[0]: i for i, r in enumerate(rows)},
            first,
            digest([(r[0], r[1], r[3], r[4]) for r in rows]),  # where rows lie, not their names: a placement reads
            # no name, so a move that renames or re-homes a data row keeps every unit's key
            sorted((r[4], r[4] + r[1] - r[0], r[0], r[1]) for r in rows),
            snapshot.versions[version].segments,
        )

    return effort.memo(("index", snapshot.digest, version), build)


def _same(rom: mmap.mmap, start: int, data: bytes, words: dict[int, bool], text: int | None) -> bool:
    if not words:
        return rom[start : start + len(data)] == data
    return all(
        rom[start + o : start + o + 4] == data[o : o + 4]
        if o not in words
        else not words[o]
        or text is None
        or (int.from_bytes(rom[start + o : start + o + 4], "big") - int.from_bytes(data[o : o + 4], "big") - text)
        & _WORD
        == 0
        for o in range(0, len(data), 4)
    )


def _joins(a: _Row, b: _Row) -> bool:  # b follows a in the ROM and in memory
    return b[0] == a[1] and b[4] == a[4] + a[1] - a[0]


def _whole(rows: list[_Row], start: int, end: int) -> bool:
    """Whether the rows hold [start, end) without a gap, following each other in the ROM and in memory."""
    at = bisect_right(rows, (start, _WORD)) - 1
    if at < 0 or rows[at][1] <= start:
        return False
    while rows[at][1] < end:
        at += 1
        if at >= len(rows) or not _joins(rows[at - 1], rows[at]):
            return False
    return True


def _loaded(rom: mmap.mmap, base: int, offset: int) -> int | None:
    """The address the ROM's code forms at a %lo instruction: its low half joined to the lui that feeds its base."""
    word = int.from_bytes(rom[base + offset : base + offset + 4], "big")
    low = (word & 0xFFFF) - ((word & 0x8000) << 1)
    for at in range(offset - 4, max(offset - 132, -4), -4):
        earlier = int.from_bytes(rom[base + at : base + at + 4], "big")
        if earlier >> 26 == 0x0F and (earlier >> 16) & 31 == (word >> 21) & 31:
            return (((earlier & 0xFFFF) << 16) + low) & _WORD
    return None


def _why(rom: mmap.mmap, data: bytes, words: dict[int, bool], runs: int) -> dict[str, str]:
    """Why bytes that nothing in the ROM matches cannot be placed, and where they do appear."""
    if runs > 1:
        return {"why": f"the code places the bytes in {runs} separate runs of the ROM, not one", "at": ""}
    if words and all(words.values()) and len(words) * 4 >= len(data):
        return {"why": "label table absent from the ROM (every word is a jump into the unit's own text)", "at": ""}
    if not words and (at := rom.find(data)) >= 0:
        return {"why": "bytes found outside the data rows", "at": f"0x{at:X}"}
    return {"why": "bytes found nowhere in the ROM at the unit's text", "at": ""}


def _placed(
    index: _Index,
    data: bytes,
    words: dict[int, bool],
    refs: list[tuple[int, int]],
    text: tuple[int, int],
    loose: bool = False,
) -> int | None:
    """The ROM offset the unit's code puts its bytes at, if the ROM holds them there (loose: where it forms it)."""
    rom, by_vram = index.rom, index.by_vram
    formed = {(a - at) & _WORD for o, at in refs if (a := _loaded(rom, text[1], o)) is not None}
    if len(formed) != 1:
        return None
    start = next(iter(formed))
    i = bisect_right(by_vram, (start, _WORD)) - 1
    if i < 0 or not by_vram[i][0] <= start < by_vram[i][1]:
        return None
    at = by_vram[i][2] + start - by_vram[i][0]
    return at if loose or _same(rom, at, data, words, text[0]) else None


def _content(index: _Index, data: bytes, words: dict[int, bool], vram: int | None) -> list[int]:
    """Every ROM offset holding the bytes: at a row start when relocated words are involved, else anywhere in data."""
    rom, rows, first = index.rom, index.rows, index.first
    if not data or not rows:
        return []
    if words:
        word = int.from_bytes(data[:4], "big")
        if 0 in words:  # a leading relocated word is predictable only for a jump table into the unit's own text
            starts = first.get((word + vram) & _WORD, []) if words[0] and vram is not None else []
        else:
            starts = first.get(word, [])
        return [s for s in starts if _same(rom, s, data, words, vram)]
    found: list[int] = []
    at = 0
    low = rows[0][0]
    high = max(r[1] for r in rows)
    while len(found) < _MANY and (at := rom.find(data, max(at, low), high)) >= 0:
        found.append(at)
        at += 1
    if len(found) == _MANY:  # the count was cut short: the places a row starts at still count
        rows_at = first.get(int.from_bytes(data[:4], "big"), [])
        found = sorted({*found, *(s for s in rows_at if rom[s : s + len(data)] == data)})
    return found


def borrowed(
    snapshot: Snapshot, unit: UnitSpec, version: str, blob: bytes, ranges: Sequence[Placement]
) -> tuple[Placement, ...]:
    """Place an emitted constant pool where the ROM code forms its address, without changing its owner. Bytes that
    differ there still link it there (the code then reads the target's addresses, not 0): native.measure compares the
    linked pool with the ROM and a difference keeps the proof inexact."""
    text = next((p for p in ranges if p.section == ".text"), None)
    if text is None or any(p.section == ".rodata" for p in ranges):
        return ()
    data, words, refs = emitted(blob).get(".rodata", (b"", {}, []))
    if not data or not refs:
        return ()
    index = _index(snapshot, version)
    refs = _aligned(index.rom, text.rom_start, _text_words(blob), refs)
    at = _placed(index, data, words, refs, (text.vram, text.rom_start), loose=True)
    if at is None or not _whole(index.rows, at, at + len(data)):
        return ()
    return (Placement(version, ".rodata", at, at + len(data), _vram_of(index, at)),)


def _text_words(blob: bytes) -> bytes:
    """The object's .text bytes; none for a blob that is no ELF object (nothing to align then)."""
    try:
        section = ELFFile(io.BytesIO(blob)).get_section_by_name(".text")
    except ELFError:
        return b""
    return section.data() if section is not None else b""


_NEAR = 16  # words a candidate's %lo may sit from the target's: the inserted or dropped instructions before it


def _aligned(rom: mmap.mmap, base: int, code: bytes, refs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """The refs at the target's own %lo instructions: a candidate a few words longer or shorter than the target has
    its %lo at another offset, so the ROM word at the candidate's offset forms nothing. Each moves to the nearest
    offset whose ROM word has the candidate's opcode and base register and forms an address."""
    out = []
    for offset, at in refs:
        mine = int.from_bytes(code[offset : offset + 4], "big") if offset + 4 <= len(code) else None
        near = sorted(range(offset - 4 * _NEAR, offset + 4 * _NEAR + 4, 4), key=lambda o: abs(o - offset))
        for o in near if mine is not None else (offset,):
            word = int.from_bytes(rom[base + o : base + o + 4], "big") if o >= 0 and base + o + 4 <= len(rom) else 0
            if (mine is None or word >> 21 == mine >> 21) and _loaded(rom, base, o) is not None:
                out.append((o, at))
                break
    return out


def locate(index: _Index, blob: bytes, text: tuple[int, int] | None, loose: bool = False) -> Located:
    """Per emitted section the ROM ranges [start, end) holding its bytes (relocated .text words shifted by the unit's
    text vram), or the reason instead. The place the unit's own code names decides; else every holder, narrowed by
    the addresses the code forms."""
    return place(index, emitted(blob), text, loose)


def place(index: _Index, sections: Emitted, text: tuple[int, int] | None, loose: bool = False) -> Located:
    """`locate` on sections already read out of the object: a built object is cached apart from the rows it lands on."""
    rom, segments = index.rom, index.segments
    vram, base = text or (None, 0)
    out: Located = {}
    for section, (data, words, refs) in sections.items():
        at = _placed(index, data, words, refs, text, loose) if text else None
        starts = [at] if at is not None else _content(index, data, words, vram)
        if len(starts) > 1 and text:
            formed = [(_loaded(rom, base, o), at) for o, at in refs]
            starts = [
                s for s in starts if all(a is None or a == _vram_of(index, s) + at & _WORD for a, at in formed)
            ] or starts
        if len(starts) > 1 and text:  # data lies in the segment of the code that uses it
            home = (
                next((lo, hi) for _, lo, hi, _ in segments if lo <= base < hi)
                if any(lo <= base < hi for _, lo, hi, _ in segments)
                else None
            )
            starts = [s for s in starts if home is None or home[0] <= s < home[1]] or starts
        if len(starts) > 1 and text:  # a pool sits nearer to its code than an equal one of other code does
            near = sorted(abs(s - base) for s in starts)
            starts = [s for s in starts if abs(s - base) == near[0]] if near[0] < near[1] else starts
        runs = len({(a - at) & _WORD for o, at in refs if (a := _loaded(rom, base, o)) is not None}) if text else 0
        out[section] = [[s, s + len(data)] for s in starts] or _why(rom, data, words, runs)
    return out


def _vram_of(index: _Index, offset: int) -> int:
    """The address the rows give a ROM offset (the nearest row at or before it)."""
    rows = index.rows
    i = bisect_right(rows, (offset, _WORD)) - 1
    return rows[i][4] + offset - rows[i][0]


def _job_key(item: tuple[Any, ...]) -> str:
    """What the unit's placements read: its builds in every holder (their inputs, tools and producing
    code), the rows they can land on, its own text range and the code that follows an object to the ROM. Any other
    edit of the tool leaves every placement warm."""
    snapshot, unit, only, loose, recipe = item
    held = sorted(snapshot.layout.held(unit.members) & ({only} if only else set(snapshot.versions)))
    return digest(
        (
            [_built_key(snapshot, unit, v, recipe) for v in held],
            [_index(snapshot, v).digest for v in held],
            loose,
            code_identity(place),
        )
    )


_job_key.coded = True  # type: ignore[attr-defined]  # the key names the code it depends on: pool adds none


def _built_key(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe | None) -> str:
    """What one holder's object says about where it lands, without the rows: an edit of any data row of the version
    (a relabel, a split) moves the job key of every unit there, while their objects stay built."""
    return digest((native.stamp(snapshot, unit, version, recipe, pinned=True, code=False), code_identity(_built)))


def _built(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe) -> Built | str:
    """The holder's emitted sections and the place of its text, or why it has none (a refusal replays as text)."""
    cfg = snapshot.config
    code = replace(unit, members=tuple(n for n in unit.members if snapshot.layout.members[n].kind == "function"))
    try:
        with store.work(cfg) as work:
            obj = native.objects(snapshot, unit, version, recipe, work)[0]
            blob = obj.read_bytes()
            unresolved = versions.resolve(snapshot.versions[version], versions.undefined(obj), unit.path)
        if unresolved:  # the link would fail: the holder is withheld like any compile failure
            raise Refusal(
                replace(
                    unresolved[0],
                    reason=f"{unresolved[0].missing[0]} has no address in {version} ({len(unresolved)} unresolved)",
                )
            )
        held = next((p for p in native.sections(snapshot, code, version) if p.section == ".text"), None)
    except Refusal as refusal:
        if any(f.key == "journal.changed" for f in refusal.findings):
            raise
        return f"{refusal.findings[0].key}: {refusal.findings[0].reason}"
    return emitted(blob), (held.vram, held.rom_start) if held else None


def _job(item: tuple[Any, ...]) -> dict[str, Located | str]:
    snapshot, unit, only, loose, recipe = item
    cfg = snapshot.config
    out: dict[str, Located | str] = {}
    recipe = recipe or recipes.resolve(cfg, unit, {})
    for version in sorted(snapshot.layout.held(unit.members)):
        if only not in (None, version):
            continue
        built = store.cached_object(
            cfg,
            "ownership.built",
            _built_key(snapshot, unit, version, item[4]),
            partial(_built, snapshot, unit, version, recipe),
        )
        out[version] = built if isinstance(built, str) else place(_index(snapshot, version), *built, loose)
    return out


def _range(hits: list[list[int]], here: list[tuple[int, int]], fixed: bool = False) -> tuple[list[int] | None, str]:
    """The unit's range; data stays within its own rows, while a function's emitted pool can move."""
    if fixed:
        hits = [h for h in hits if sum(max(0, min(b, h[1]) - max(a, h[0])) for a, b in here) == h[1] - h[0]]
    owned = (
        hits
        if len(hits) == 1
        else [h for h in hits if (h[0], h[1]) in here]
        or [h for h in hits if any(a < h[1] and h[0] < b for a, b in here)]
    )
    if len(owned) == 1:
        return owned[0], ""
    shown = ", ".join(f"0x{s:X}-0x{e:X}" for s, e in hits[:4])
    return None, f"ambiguous among {len(hits)} ranges {shown}" if hits else "bytes outside owned rows"


def _named(
    unit: UnitSpec,
    picked: dict[str, list[tuple[str, int, int]]],
    indexes: Mapping[str, _Index],
    names_from: str,
    members: Mapping[str, Any],
) -> list[Claim]:
    """The claims of the unit, each named by the unit-owned member starting there, else by section, unit and address."""
    out: list[Claim] = []
    for section in sorted({s for rows in picked.values() for s, _, _ in rows}):
        held = {v: (a, b) for v, rows in picked.items() for s, a, b in rows if s == section}
        ref = names_from if names_from in held else min(held)
        name = f"{section[1:]}/{Path(unit.path).stem}/{_vram_of(indexes[ref], held[ref][0]):08X}"
        name = next(
            (
                m
                for m in unit.members
                if members[m].kind != "function"
                and any(
                    p.version == ref and p.section == section and p.rom_start == held[ref][0]
                    for p in members[m].placements
                )
            ),
            name,
        )
        out.extend(Claim(unit.path, v, section, a, b, (name,)) for v, (a, b) in sorted(held.items()))
    return out


def _resolve(
    unit: UnitSpec,
    members: Mapping[str, Any],
    result: Mapping[str, Located | str],
    indexes: Mapping[str, _Index],
    names_from: str,
) -> tuple[list[str], set[str], list[Claim], list[str]]:
    """The unit's code and bss members, holders, claims and debt; a holder whose data cannot be placed is left out."""
    keep = [
        n
        for n in unit.members
        if members[n].kind == "function" or all(p.section == ".bss" for p in members[n].placements)
    ]
    resolved: set[str] = set()
    picked: dict[str, list[tuple[str, int, int]]] = {}
    debt: list[str] = []
    for version, got in result.items():
        if isinstance(got, str):
            state = "withheld compile failed" if version in unit.withheld else "compile failed"
            debt.append(f"{state} ({got.split(':')[0]}): {unit.path} {version} {got.split(': ', 1)[1]}")
            continue
        index = indexes[version]
        before = [
            (p.rom_start, p.rom_end, p.section)
            for n in unit.members
            if n not in keep
            for p in members[n].placements
            if p.version == version
        ]
        mine: list[tuple[str, int, int]] = []
        for section, hits in got.items():
            here = [(a, b) for a, b, held in before if held == section]
            chosen, why = _range(hits, here, layout.is_datum(unit)) if isinstance(hits, list) else (None, "")
            if chosen is None:
                reason, at = (
                    (hits["why"], f" @{hits['at']}" if hits["at"] else "")
                    if isinstance(hits, dict)
                    else (("ambiguous", why[len("ambiguous") :]) if why.startswith("ambiguous") else (why, ""))
                )
                debt.append(f"{reason}: {unit.path} {version} {section}{at}")
                break
            start, end = chosen
            if not _whole(index.rows, start, end):
                debt.append(
                    f"bytes are not whole rows that follow each other in memory: {unit.path} {version} "
                    f"{section} @0x{start:X}-0x{end:X}"
                )
                break
            mine.append((section, start, end))
        else:
            resolved.add(version)
            picked[version] = mine
    return keep, resolved, _named(unit, picked, indexes, names_from, members), debt


def _clash(taken: list[tuple[int, int, str]], start: int, end: int) -> str | None:
    """The unit that holds some of [start, end) in taken (sorted, disjoint), if any."""
    at = bisect_left(taken, (start,))
    return next((o for s, e, o in taken[max(at - 1, 0) : at + 1] if s < end and start < e), None)


def _occupied(taken: list[tuple[int, int, str]], start: int, end: int) -> bool:
    """Whether other owners' disjoint claims cover every byte, including adjacent claims from several owners."""
    at = max(0, bisect_right(taken, (start, float("inf"))) - 1)
    while at < len(taken) and taken[at][0] <= start:
        start = max(start, taken[at][1])
        if start >= end:
            return True
        at += 1
    return False


def _own(
    unit: UnitSpec,
    members: Mapping[str, Any],
    keep: list[str],
    resolved: set[str],
    found: list[Claim],
    taken: dict[str, list[tuple[int, int, str]]],
    debt: list[str],
) -> tuple[UnitSpec, list[Claim]]:
    """The unit with the rows its claims become, the holders it is withheld in and the claims kept."""
    holders = {v for n in unit.members for v in members[n].holders()}
    for version in sorted(resolved):
        clashes = (_clash(taken[version], c.start, c.end) for c in found if c.version == version)
        if other := next(filter(None, clashes), None):
            debt.append(f"row owned by another unit: {unit.path} {version} (claimed by {other})")
            resolved.discard(version)
    kept = [c for c in found if c.version in resolved]
    for c in kept:
        insort(taken[c.version], (c.start, c.end, unit.path))
    names = sorted({n for c in kept for n in c.rows})
    return replace(unit, members=(*keep, *names), withheld=tuple(sorted(holders - resolved))), kept


def _findings(debt: dict[str, list[str]], action: str) -> list[Finding]:
    """One finding per reason: a previously built holder failing to compile refuses; withheld failures stay debt."""
    out = []
    for why, rows in sorted(debt.items()):
        broken = why.startswith("compile failed (") and "symbols.unknown" not in why
        key = why[len("compile failed (") :].split(")")[0] if broken else "layout.ownership"
        out.append(
            Finding(
                key,
                f"{len(rows)} unit holders {'do not compile' if broken else 'are withheld'}: {why}",
                missing=tuple(rows),
                blocking=broken,
                action=action if broken else "",
            )
        )
    return out


def _residuals(
    snapshot: Snapshot, units: Sequence[UnitSpec], claims: Sequence[Claim], indexes: Mapping[str, _Index]
) -> dict[str, tuple[str, ...]]:
    """The unclaimed rows left inside displaced providers. Use claim_rows' partition, so a retained source names
    exactly the prefix and suffix rows the split edits create. Withheld sources leave those bytes in the base ROM."""
    places: dict[str, list[tuple[str, Placement]]] = defaultdict(list)
    for unit in units:
        for name in unit.members:
            for p in snapshot.layout.members[name].placements:
                places[p.version].append((unit.path, p))
    remaining: dict[str, list[str]] = defaultdict(list)
    for version, held in places.items():
        # Only the row names are needed: section classes give the same partition as the split's raw/dotted kinds.
        rows: list[list[Any]] = [[s, e, at, n, section, n, True] for s, e, n, section, at in indexes[version].rows]
        named = {r[3] for r in rows}
        mine = sorted((c for c in claims if c.version == version), key=lambda c: c.start)
        for c in mine:
            layout._claim_row(snapshot.versions[version], rows, named, c)
        claimed = {n for c in mine for n in c.rows}
        for start, end, _, name, *_ in rows:
            if name not in claimed:
                for path, p in held:
                    if p.rom_start <= start and end <= p.rom_end:
                        remaining[path].append(name)
    return {path: tuple(dict.fromkeys(names)) for path, names in remaining.items()}


class LocatedBatch(NamedTuple):
    """One completed location pass, bound to its producer state; an older context never follows a newer memo slot."""

    snapshot: Snapshot
    version: str | None
    source_tree: str | None
    located: tuple[tuple[UnitSpec, Mapping[str, Located | str]], ...]
    invocation: str


def context(snapshot: Snapshot, version: str | None = None) -> LocatedBatch | None:
    """The latest whole-map pass of this snapshot, if this command actually ran one. Capture before another pass."""
    return cast(LocatedBatch | None, effort.peek(("ownership.located", snapshot.digest, version)))


def claims(
    snapshot: Snapshot,
    version: str | None = None,
    units: Mapping[str, UnitSpec] | None = None,
    *,
    reuse: tuple[LocatedBatch, Collection[str]] | None = None,
    source_tree: str | None = None,
) -> tuple[dict[str, UnitSpec], tuple[Claim, ...], list[Finding]]:
    """The one place that decides who owns which ROM bytes: each compiled unit's object is followed to the ROM ranges
    its data occupies in each holder. Returns the units with their claims' rows and withheld holders, the claims,
    and findings. `reuse` = (a captured earlier batch, the units its writes reach): unchanged units keep that batch's
    placements. Inference supplies its already captured source-tree fingerprint; moves supply their write reach."""
    with effort.stage("layout.ownership") as span:
        wanted = dict(snapshot.layout.units if units is None else units)
        phases = config.kinds()
        todo = sorted((u for u in wanted.values() if "compile" in phases[u.kind]["phases"]), key=_unowned)
        earlier = _earlier(snapshot, version, reuse, source_tree) if reuse and units is None else {}
        fresh = [u for u in todo if u.path not in earlier or replace(earlier[u.path][0], withheld=u.withheld) != u]
        jobs = [(snapshot, u, version, False, None) for u in fresh]
        located = dict(
            zip(
                (u.path for u in fresh),
                pool.map(snapshot.config, "layout.ownership", _job, jobs, _job_key),
                strict=True,
            )
        )
        results = [located[u.path] if u.path in located else earlier[u.path][1] for u in todo]
        if units is None:  # replace the slot, never the captured batch: the map digest alone excludes live C/headers
            batch = LocatedBatch(
                snapshot, version, source_tree, tuple(zip(todo, results, strict=True)), effort.invocation()
            )
            effort.memo_valid(
                ("ownership.located", snapshot.digest, version), lambda: batch, lambda kept: kept is batch
            )
        if reuse:
            with effort.stage("layout.ownership.reused") as taken_over:  # units this pass took from the earlier one
                taken_over.add(items=len(todo) - len(fresh))
        indexes = {v: _index(snapshot, v) for v in snapshot.versions}
        taken: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        debt: dict[str, list[str]] = defaultdict(list)
        out, found = dict(wanted), []
        displaced = {}
        for unit, result in zip(todo, results, strict=True):
            keep, resolved, mine, left = _resolve(
                unit, snapshot.layout.members, result, indexes, snapshot.config.project.names_from
            )
            # A provider may yield only after every byte it successfully emits in every holder is owned
            # elsewhere. Keep failed or unplaced sources attached to their old rows; an empty unit loses both
            # its compile regression and the version in which policy must read it on the next pass.
            holders = snapshot.layout.held(unit.members)
            retired = (
                layout.is_datum(unit)
                and not keep
                and bool(mine)
                and not left
                and resolved == holders
                and all(_occupied(taken[c.version], c.start, c.end) for c in mine)
                and all(
                    p.size > 0 and _occupied(taken[p.version], p.rom_start, p.rom_end)
                    for n in unit.members
                    for p in snapshot.layout.members[n].placements
                )
            )
            if retired:
                del out[unit.path]
                continue
            owned, kept = _own(unit, snapshot.layout.members, keep, resolved, mine, taken, left)
            out[unit.path] = (
                owned if owned.members else replace(unit, withheld=tuple(sorted(holders | set(unit.withheld))))
            )
            if layout.is_datum(unit) and any(line.startswith("row owned by another unit:") for line in left):
                displaced[unit.path] = owned
            found.extend(kept)
            for line in left:
                debt[line.split(":")[0]].append(line.split(": ", 1)[1])
        residuals = _residuals(snapshot, [wanted[p] for p in displaced], found, indexes)
        stranded = []
        for path, owned in displaced.items():
            names = tuple(dict.fromkeys((*owned.members, *residuals.get(path, ()))))
            if names:
                out[path] = replace(owned, members=names)
            else:
                stranded.append(path)
        findings = _findings(debt, "fix the source of each listed unit")
        if stranded:
            findings.append(
                Finding(
                    "layout.ownership",
                    "data providers cannot yield until all their bytes have other owners",
                    missing=tuple(stranded),
                    action="resolve the overlapping data ownership before retiring these sources",
                )
            )
        span.add(items=len(todo), findings=findings)
        return out, tuple(found), findings


def _earlier(
    after: Snapshot, version: str | None, reuse: tuple[LocatedBatch, Collection[str]], source_tree: str | None
) -> dict[str, tuple[UnitSpec, Mapping[str, Located | str]]]:
    """The earlier pass's (unit, placements) of every unit whose placements cannot differ in `after`: not reached by
    the writes, the same spec, members and version facts, and every data row index it locates against unchanged
    (row places only, so renames keep it). Anything else that differs gives nothing back."""
    batch, reached = reuse
    before = batch.snapshot
    if (
        batch.invocation != effort.invocation()
        or before.config != after.config
        or batch.version != version
        or batch.source_tree != source_tree
    ):
        return {}
    unchanged = {v for v, new in after.versions.items() if v in before.versions and before.versions[v] == new}
    same = {
        v
        for v, new in after.versions.items()
        if v in before.versions
        and (v in unchanged or replace(before.versions[v], symbols=new.symbols, code=new.code) == new)
        and _index(before, v).digest == _index(after, v).digest
    }

    def facts(unit: UnitSpec) -> bool:
        for v in after.layout.held(unit.members) - unchanged:
            names = view.names(before, unit, v)
            if versions.facts_of(before.versions[v], names) != versions.facts_of(after.versions[v], names):
                return False
        return True

    old, new = before.layout.members, after.layout.members
    return {
        unit.path: (unit, located)
        for unit, located in batch.located
        if unit.path not in reached
        and all(old.get(n) == new.get(n) for n in unit.members)
        and after.layout.held(unit.members) <= same
        and facts(unit)
    }


def _unowned(unit: UnitSpec) -> bool:
    """A data unit of rows no function has claimed: it gives them up to the first function whose object emits them."""
    return bool(unit.members) and all(versions.unowned(n) for n in unit.members)


def _covered(snapshot: Snapshot, unit: UnitSpec, claims: Sequence[Claim], held: Collection[str] = ()) -> bool:
    """An independent data provider yields when the compile reproduces all its bytes in every holder (or every version
    the claims reach when they are its own rows by name): the unit owns them. A row holding the claimed bytes yields
    too (the claim splits it). Function units keep their ownership and a compiler-generated pool returns to the unit
    that emits it."""
    if not layout.is_datum(unit):
        return False
    places = [p for n in unit.members for p in snapshot.layout.members[n].placements]

    def whole(version: str | None) -> bool:
        return all(
            p.size > 0
            and any(
                c.version == p.version
                and (c.section == p.section or {c.section, p.section} <= {".data", ".rodata"})
                and ((c.start <= p.rom_start and p.rom_end <= c.end) or (p.rom_start <= c.start and c.end <= p.rom_end))
                for c in claims
            )
            for p in places
            if version in (None, p.version)
        )

    if bool(places) and whole(None):
        return True
    named = set(unit.members) <= {name for c in claims for name in c.rows}
    return (
        named
        and bool(places)
        and {p.version for p in places} <= set(held)
        and all(whole(v) for v in {c.version for c in claims})
    )


def _emits_data(snapshot: Snapshot, unit: UnitSpec) -> bool:
    """Whether the unit is compiled from functions, so what it emits is derived at all."""
    return "compile" in config.kind(unit.kind)["phases"] and any(
        snapshot.layout.members[n].kind == "function" for n in unit.members
    )


def derive(
    snapshot: Snapshot, unit: UnitSpec, loose: bool = False, recipe: Recipe | None = None
) -> tuple[Snapshot, UnitSpec]:
    """The snapshot and unit with the data rows the source emits, per holder, where the ROM holds them exactly (private
    copy). Loose (a score, never a landing) places data where the ROM's code reads it even when bytes differ. A trial
    recipe (--flag, --toolchain) is the one whose object is followed, so its literal pools are placed too."""
    return derive_many(snapshot, [unit], loose, recipe)[0]


def derive_many(
    snapshot: Snapshot,
    units: Sequence[UnitSpec],
    loose: bool = False,
    recipe: Recipe | Sequence[Recipe | None] | None = None,
    keep: bool = True,
) -> list[tuple[Snapshot, UnitSpec]]:
    """derive for each unit: every (unit, holder) compile is placed in one pool dispatch, so the versions of one unit
    build side by side (cached ones dispatch nothing). A recipe sequence names the trial recipe of each unit."""
    return derive_pairs([(snapshot, u) for u in units], loose, recipe, keep)


_KEPT = 512  # derived tree states one command keeps: enough for a bulk chunk and its consumers, bounded in memory


def _derivations() -> OrderedDict[str, tuple[Snapshot, UnitSpec]]:
    """The derivations of the running command by input key (dropped when it ends), least recently used first."""
    return effort.memo(("ownership.derive",), OrderedDict)


def _derive_key(snapshot: Snapshot, unit: UnitSpec, loose: bool, recipe: Recipe | None) -> str | None:
    """A derivation is a pure function of the tree state (its digest), the unit, looseness and the trial recipe."""
    tree = getattr(snapshot, "digest", None)
    return digest((tree, unit, loose, recipe)) if isinstance(tree, str) else None


def derive_pairs(
    pairs: Sequence[tuple[Snapshot, UnitSpec]],
    loose: bool = False,
    recipe: Recipe | Sequence[Recipe | None] | None = None,
    keep: bool = True,
) -> list[tuple[Snapshot, UnitSpec]]:
    """derive for each (snapshot, unit) pair, any number of tree states in one pool dispatch. A tree state already
    derived in this command (an input seen before, or a derive's own output, which already holds its rows) is taken
    as it is, so callers along one command never derive the same state twice and workers only get derived pairs.
    Not `keep`: the derivations are not kept for later callers (a proof job's derived state dies with the job)."""
    trials: list[Recipe | None] = (
        list(recipe) if isinstance(recipe, list | tuple) else [cast("Recipe | None", recipe)] * len(pairs)
    )
    kept = _derivations()
    out: list[tuple[Snapshot, UnitSpec] | None] = [None] * len(pairs)
    keys: list[str | None] = []
    todo: list[int] = []
    for n, ((snapshot, u), r) in enumerate(zip(pairs, trials, strict=True)):
        key = _derive_key(snapshot, u, loose, r)
        keys.append(key)
        if key is not None and key in kept:
            kept.move_to_end(key)
            out[n] = kept[key]
        elif _emits_data(snapshot, u):
            todo.append(n)
        else:
            out[n] = (snapshot, u)
    holders = {n: sorted(pairs[n][0].layout.held(pairs[n][1].members)) for n in todo}
    jobs = [(pairs[n][0], pairs[n][1], v, loose, trials[n]) for n in todo for v in holders[n]]
    found = iter(pool.map(pairs[todo[0]][0].config, "ownership.derive", _job, jobs, _job_key) if jobs else ())
    merged = {n: {v: next(found)[v] for v in holders[n]} for n in todo}
    for n in todo:
        snapshot, u = pairs[n]
        done = out[n] = _derived(snapshot, u, merged[n])
        for key in (keys[n], _derive_key(done[0], done[1], loose, trials[n])):  # the output is a fixed point
            if key is not None and keep:
                kept[key] = done
                kept.move_to_end(key)
    while len(kept) > _KEPT:
        kept.popitem(last=False)
    return [pair for pair in out if pair is not None]


class _Rows(NamedTuple):
    """What every derive of one layout reads, listed once per layout and not once per derived unit."""

    taken: Mapping[str, Sequence[tuple[int, int, str]]]  # the ROM rows other units' data occupy, sorted per version
    spans: Mapping[str, Sequence[tuple[int, int, str]]]  # every placement of every unit, sorted per version
    longest: Mapping[str, int]  # the longest placement per version, how far back an overlap search looks
    owners: Mapping[str, Sequence[str]]  # the units naming each member


def _rows(snapshot: Snapshot) -> _Rows:
    units, members = snapshot.layout.units, snapshot.layout.members

    def build() -> tuple[object, object, _Rows]:
        taken: defaultdict[str, list[tuple[int, int, str]]] = defaultdict(list)
        spans: defaultdict[str, list[tuple[int, int, str]]] = defaultdict(list)
        owners: defaultdict[str, list[str]] = defaultdict(list)
        for other in units.values():
            free = _unowned(other)
            for n in other.members:
                owners[n].append(other.path)
                for p in members[n].placements if n in members else ():
                    spans[p.version].append((p.rom_start, p.rom_end, other.path))
                    if not free and members[n].kind != "function" and p.rom_end > p.rom_start:
                        taken[p.version].append((p.rom_start, p.rom_end, other.path))
        longest = {v: max(e - s for s, e, _ in rows) for v, rows in spans.items()}
        sort = {v: sorted(rows) for v, rows in taken.items()}
        # the mappings are held with the rows, so their ids name them for as long as the memo lives
        return units, members, _Rows(sort, {v: sorted(rows) for v, rows in spans.items()}, longest, dict(owners))

    return effort.memo(("ownership.rows", id(units), id(members)), build)[2]


def _near(rows: _Rows, claims: Sequence[Claim]) -> set[str]:
    """The units a claim can cover: one naming a claimed row, or one with a placement touching a claim's bytes.
    `_covered` holds for no other unit, so only these are tested."""
    near = {path for c in claims for name in c.rows for path in rows.owners.get(name, ())}
    for c in claims:
        spans = rows.spans.get(c.version, ())
        at = bisect_right(spans, (c.end, float("inf")))
        reach = c.start - rows.longest.get(c.version, 0)
        while at > 0 and spans[at - 1][0] >= reach:
            at -= 1
            start, end, path = spans[at]
            if start <= c.end and c.start <= end:
                near.add(path)
    return near


def _derived(snapshot: Snapshot, unit: UnitSpec, result: Mapping[str, Located | str]) -> tuple[Snapshot, UnitSpec]:
    from unbake import layout  # the rows come from the layout, which in turn reads ownership

    members = snapshot.layout.members
    indexes = {v: _index(snapshot, v) for v in snapshot.versions}
    keep, resolved, mine, debt = _resolve(unit, members, result, indexes, snapshot.config.project.names_from)
    held = {v for n in unit.members for v in members[n].holders()}
    rows = _rows(snapshot)
    units = snapshot.layout.units
    yielding = {p for p in _near(rows, mine) if p in units and _covered(snapshot, units[p], mine, held)}
    skip = {unit.path, *yielding}
    taken: defaultdict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for v in resolved:  # the only versions _own reads; the base rows are sorted already
        taken[v] = [r for r in rows.taken.get(v, ()) if r[2] not in skip]
    owned, kept = _own(unit, members, keep, resolved, mine, taken, debt)
    claimed = layout.claim_rows(snapshot, kept) if kept else {}  # no claim, no rows to write
    if claimed:
        spans = [(c.version, c.start, c.end) for c in kept]
        drop = {
            path
            for path in _near(rows, kept)
            if path in units
            and (
                _covered(snapshot, units[path], kept, held)
                or (
                    _unowned(units[path])
                    and any(
                        p.version == v and p.rom_start < e and s < p.rom_end
                        for n in units[path].members
                        for p in members[n].placements
                        for v, s, e in spans
                    )
                )
            )
        }
        units = {path: u for path, u in units.items() if path not in drop}
        units[owned.path] = owned
        snapshot = layout.overlay(
            snapshot, {**claimed, "layout.toml": layout.dump_map(replace(snapshot.layout, units=units))}
        )
    return snapshot, owned


def _stamp_key(item: tuple[Snapshot, UnitSpec, str]) -> str:
    return native.stamp(*item, pinned=True)


def _exact_job(item: tuple[Snapshot, UnitSpec, str]) -> str:
    proofs = native.prove_job(item)
    gap = next((m for p in proofs for m in p.missing), "no measurement")
    return "" if proofs and all(p.exact for p in proofs) else gap


def exact(snapshot: Snapshot, units: dict[str, UnitSpec]) -> tuple[dict[str, UnitSpec], list[Finding]]:
    """A unit is built only where its bytes reproduce the ROM; other holders are withheld, so `make check` is clean."""
    with effort.stage("layout.exactness") as span:
        phases = config.kinds()
        trial = replace(snapshot, layout=replace(snapshot.layout, units=units), digest=digest((snapshot.digest, units)))
        jobs = [
            (trial, u, v)
            for u in units.values()
            if "compile" in phases[u.kind]["phases"]
            for v in sorted(snapshot.layout.held(u.members) - set(u.withheld))
        ]
        gaps = pool.map(snapshot.config, "layout.exactness", _exact_job, jobs, _stamp_key)
        lost: dict[str, set[str]] = defaultdict(set)
        debt: dict[str, list[str]] = defaultdict(list)
        for (_, unit, version), gap in zip(jobs, gaps, strict=True):
            if gap:
                lost[unit.path].add(version)
                why = re.sub(r"\d+ bytes differ at \+0x[0-9A-Fa-f]+ ", "bytes differ in ", gap.split(": ", 1)[-1])
                debt[why[:80]].append(f"{unit.path} {version}")
        findings = [
            Finding(
                "layout.ownership",
                f"{len(rows)} unit holders are not exact and are withheld: {why}",
                missing=tuple(rows),
                blocking=False,
            )
            for why, rows in sorted(debt.items())
        ]
        span.add(items=len(jobs), findings=findings)
        return {
            path: replace(u, withheld=tuple(sorted({*u.withheld, *lost.get(path, ())}))) for path, u in units.items()
        }, findings
