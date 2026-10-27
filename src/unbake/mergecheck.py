"""Which merges must stay separate files: a module map (unit, module) read against the link order and the built objects.

A module of several units can become one source file only when one object can take the units' place in the link
without moving a byte. `verdicts` is the pure rule over plain facts; `facts_of` reads them from the link script and the
objects of each version. The hard findings (the module must stay separate files):
  flags       the units do not share one toolchain and flag set (a translation-unit boundary)
  sections    one class of the module sits in several output sections
  gap         two entries of one class are not adjacent in the link order (another module's bytes sit between)
  order       the units are not linked in the map's order
  lma         one merged object would not fill the span the units fill in an output section (their data and read-only
              pieces, each aligned as linked, packed again): the next section's load address would overlap or leave a
              gap
The soft findings (the merge can go ahead, and the source must reproduce them):
  align       an ALIGN directive pads between two entries of the module
  statics     two units define a local symbol of the same name (the merged file needs one renamed)
  literals    two or more units each hold read-only data (a merged file lets the compiler pool and reorder it)"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

Row = dict[str, Any]
CLASS_OF = {".text": "text", ".data": "data", ".rodata": "rodata", ".rdata": "rodata", ".bss": "bss", ".sbss": "bss"}
RODATA = "rodata"
PADDED = ("data", "rodata")
_ENTRY = re.compile(r"^\s*(\S+?)\.o\(([^)]*)\);")
_ALIGN = re.compile(r"^\s*\.\s*=\s*ALIGN\((\d+)\)")
_OUTSEC = re.compile(r"^\s*(\.[A-Za-z0-9_.]+)\s+(?:0x[0-9A-Fa-f]+\s*)?(?:\(NOLOAD\)\s*)?:")


def read_modules(text: str) -> list[tuple[str, str]]:
    """(unit, module) rows of a TSV with a header row holding `unit` and `module` columns."""
    rows = list(csv.reader(text.splitlines(), delimiter="\t"))
    if not rows or "unit" not in rows[0] or "module" not in rows[0]:
        raise ValueError("a module map needs a header row with `unit` and `module` columns")
    unit, module = rows[0].index("unit"), rows[0].index("module")
    return [(r[unit], r[module]) for r in rows[1:] if r and not r[0].startswith("#") and len(r) > max(unit, module)]


def parse_link(text: str) -> list[Row]:
    """The link script rows in order: {"o": output section, "k": "entry"|"align", "u": unit, "cls": classes, "a": n}."""
    rows: list[Row] = []
    outsec = ""
    for line in text.splitlines():
        found = _OUTSEC.match(line)
        if found and "{" not in line:
            outsec = found.group(1)
            continue
        align = _ALIGN.match(line)
        if align:
            rows.append({"o": outsec, "k": "align", "a": int(align.group(1))})
            continue
        entry = _ENTRY.match(line)
        if entry:
            classes = tuple(sorted({CLASS_OF[s] for s in entry.group(2).split() if s in CLASS_OF}))
            rows.append({"o": outsec, "k": "entry", "u": entry.group(1), "cls": classes})
    return rows


def sections_of(path: Path) -> tuple[dict[str, int], set[str]]:
    """Byte size per class of one object, and the names of its local (static) symbols."""
    sizes: dict[str, int] = defaultdict(int)
    local: set[str] = set()
    with path.open("rb") as handle:
        elf = ELFFile(handle)
        for section in elf.iter_sections():
            if section.name in CLASS_OF:
                sizes[CLASS_OF[section.name]] += section["sh_size"]
        table = elf.get_section_by_name(".symtab")
        if isinstance(table, SymbolTableSection):
            for symbol in table.iter_symbols():
                kind = symbol["st_info"]["type"]
                if symbol["st_info"]["bind"] == "STB_LOCAL" and kind in ("STT_FUNC", "STT_OBJECT") and symbol.name:
                    local.add(symbol.name)
    return dict(sizes), local


def pieces_of(path: Path) -> dict[str, list[tuple[int, int]]]:
    """(size, alignment) of each non-empty data and read-only input section of one object, per class."""
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open("rb") as handle:
        for section in ELFFile(handle).iter_sections():
            cls = CLASS_OF.get(section.name)
            if cls in PADDED and section["sh_size"]:
                out[cls].append((section["sh_size"], max(section["sh_addralign"], 1)))
    return dict(out)


def fills(path: Path, symbol: str) -> bool | None:
    """Whether the symbol's bytes only fill its section out to the section alignment (the section rounded up is the
    same with the symbol cut out); None when the object has no such symbol."""
    with path.open("rb") as handle:
        elf = ELFFile(handle)
        table = elf.get_section_by_name(".symtab")
        if not isinstance(table, SymbolTableSection):
            return None
        for found in table.get_symbol_by_name(symbol) or ():
            index = found["st_shndx"]
            if not isinstance(index, int):
                continue
            section = elf.get_section(index)
            size, align = section["sh_size"], max(section["sh_addralign"], 1)
            return _up(size - found["st_size"], align) == size
    return None


def _up(position: int, align: int) -> int:
    return -(-position // align) * align


def spans(seq: Sequence[Row], units: set[str]) -> list[tuple[str, int, int]]:
    """Per output section holding entries of `units`: (section, bytes the linked entries span, bytes one merged object
    spans). The linked span walks the section from the module's first entry to its last (other units' pieces and ALIGN
    directives included); the merged span packs the module's pieces of each class again, class after class."""
    out: list[tuple[str, int, int]] = []
    for section in dict.fromkeys(e["o"] for e in seq if e["k"] == "entry" and e["u"] in units):
        rows = [(i, e) for i, e in enumerate(seq) if e["o"] == section]
        mine = [i for i, e in rows if e["k"] == "entry" and e["u"] in units]
        position, classes = 0, defaultdict(list)
        for i, row in rows:
            if i < mine[0] or i > mine[-1]:
                continue
            if row["k"] == "align":
                position = _up(position, row["a"])
                continue
            for cls in row.get("parts", {}):
                for size, align in row["parts"][cls]:
                    position = _up(position, align) + size
                    if row["u"] in units:
                        classes[cls].append((size, align))
        merged = 0
        for pieces in classes.values():
            merged = _up(merged, max(a for _, a in pieces))
            for size, align in pieces:
                merged = _up(merged, align) + size
        out.append((section, position, merged))
    return out


def facts_of(
    root: Path, versions: Sequence[str], ld: str, obj: str
) -> tuple[dict[str, list[Row]], dict[str, set[str]]]:
    """Per version the link rows with each entry's byte count, and per unit its local symbol names."""
    seqs: dict[str, list[Row]] = {}
    statics: dict[str, set[str]] = {}
    cache: dict[tuple[str, str], dict[str, int] | None] = {}
    pieces: dict[tuple[str, str], dict[str, list[tuple[int, int]]]] = {}
    for version in versions:
        seq = parse_link((root / ld.format(version=version)).read_text())
        for row in seq:
            if row["k"] != "entry":
                continue
            key = (version, row["u"])
            if key not in cache:
                path = root / obj.format(version=version, unit=row["u"])
                if path.is_file():
                    cache[key], local = sections_of(path)
                    pieces[key] = pieces_of(path)
                    statics.setdefault(row["u"], set()).update(local)
                else:
                    cache[key] = None
            sizes = cache[key] or {}
            row["n"] = sum(sizes.get(c, 0) for c in row["cls"])
            row["parts"] = {c: pieces.get(key, {}).get(c, []) for c in row["cls"] if c in PADDED}
            row["cls"] = "+".join(row["cls"])
        seqs[version] = seq
    return seqs, statics


def verdicts(
    rows: Sequence[tuple[str, str]],
    seqs: Mapping[str, Sequence[Row]],
    statics: Mapping[str, set[str]],
    identity: Callable[[str], Any],
) -> list[Row]:
    """One verdict per module of two or more units: {"module", "units", "verdict": merge|separate, "hard", "soft"}."""
    modules: dict[str, list[str]] = defaultdict(list)
    order: dict[str, int] = {}
    for unit, module in rows:
        modules[module].append(unit)
        order[unit] = len(order)
    multi = {m: us for m, us in modules.items() if len(us) > 1}
    owner = {u: m for m, us in multi.items() for u in us}
    hard: dict[str, list[str]] = defaultdict(list)
    soft: dict[str, list[str]] = defaultdict(list)
    for module, units in multi.items():
        if len({identity(u) for u in units}) > 1:
            hard[module].append("flags: the units differ in toolchain or flags")
        names: dict[str, str] = {}
        for unit in units:
            for name in statics.get(unit, ()):
                if name in names and names[name] != unit:
                    soft[module].append(f"statics: {name} is local to {names[name]} and {unit}")
                names.setdefault(name, unit)
    for version, seq in seqs.items():
        live = [i for i, e in enumerate(seq) if e["k"] == "entry" and e.get("n", 0) > 0]
        rodata = {e["u"] for e in seq if e["k"] == "entry" and e["cls"] == RODATA and e["n"] > 0}
        following = dict(pairwise(live))
        where: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
        for i in live:
            home = owner.get(seq[i]["u"])
            if home:
                where[(home, seq[i]["cls"])].append((i, seq[i]["o"], seq[i]["u"]))
        for (module, cls), hits in where.items():
            if len({o for _, o, _ in hits}) > 1:
                hard[module].append(f"sections: {version} {cls} sits in several output sections")
                continue
            for (i, _, one), (j, _, two) in pairwise(hits):
                if following.get(i) != j:
                    between = [seq[k]["u"] for k in live if i < k < j and owner.get(seq[k]["u"]) != module]
                    if between:
                        hard[module].append(f"gap: {version} {cls} {one} .. {two} has {', '.join(between[:3])} between")
                if order[one] > order[two]:
                    hard[module].append(f"order: {version} {cls} {one} is linked before {two}")
                for k in range(i + 1, j):
                    if seq[k]["k"] == "align":
                        soft[module].append(f"align: {version} {cls} ALIGN({seq[k]['a']}) before {two}")
        for module, units in multi.items():
            for section, linked, merged in spans(seq, set(units)):
                if merged != linked:
                    effect = "overlaps the next section" if merged > linked else "leaves a gap before the next section"
                    shape = f"{version} {section} merged {merged} B against linked {linked} B"
                    hard[module].append(f"lma: {shape}: {effect}")
            held = [u for u in units if u in rodata]
            if len(held) > 1:
                soft[module].append(f"literals: {version} rodata in {len(held)} units")
    return [
        {
            "module": m,
            "units": len(us),
            "verdict": "separate" if hard[m] else "merge",
            "hard": sorted(set(hard[m])),
            "soft": sorted(set(soft[m])),
        }
        for m, us in multi.items()
    ]
