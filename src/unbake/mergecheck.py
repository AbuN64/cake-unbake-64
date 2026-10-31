"""Which merges must stay separate files: a module map (unit, module) read against the link order and the built objects.

A module of several units can become one source file only when one object can take the units' place in the link
without moving a byte. `verdicts` is the pure rule over plain facts; `facts_of` reads them from the link script and the
objects of each version. The hard findings (the module must stay separate files):
  flags       the units do not share one toolchain and flag set (a translation-unit boundary)
  sections    one class of the module sits in several output sections; the parts (units that share the section in every
              version where both are placed) are printed, each with its section per version, so a map row can name one
  gap         two entries of one class are not adjacent in the link order (another module's bytes sit between)
  lma         one merged object would be larger than the span the units fill in an output section (their pieces, each
              aligned as linked, packed again): it would overlap the next piece
The soft findings (the merge can go ahead, and the source must reproduce them):
  order       a version links some units of a section class elsewhere than the base order (the versions' orders
              merged): merge emits each moved unit's chunks again under the guard of the versions that move it
  fill        one merged object is smaller than that span: merge regenerates the linker fill record for the difference
  align       an ALIGN directive pads between two entries of the module
  statics     two units define a local symbol of the same name (the merged file needs one renamed)
  literals    two or more units each hold read-only data (a merged file lets the compiler pool and reorder it)"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from unbake import classwise

Row = dict[str, Any]
CLASS_OF = {".text": "text", ".data": "data", ".rodata": "rodata", ".rdata": "rodata", ".bss": "bss", ".sbss": "bss"}
RODATA = "rodata"
CLASSES = ("text", "data", "rodata", "bss")
_ENTRY = re.compile(r"^\s*(\S+?)\.o\(([^)]*)\);")
_ALIGN = re.compile(r"^\s*\.\s*=\s*ALIGN\((0x[0-9A-Fa-f]+|\d+)\)")
_MINIMUM = re.compile(r"^\s*\.\s*=\s*MAX\(\s*\.\s*,\s*(0x[0-9A-Fa-f]+|\d+)\s*\)")
_CONDITIONAL_ALIGN = re.compile(r"^\s*\.\s*=\s*DEFINED\((\w+)\)\s*\?\s*ALIGN\((\d+)\)\s*:\s*\.")
_OUTSEC = re.compile(r"^\s*(\.[A-Za-z0-9_.]+)\s+(?:0x[0-9A-Fa-f]+\s*)?(?:\(NOLOAD\)\s*)?:")


def _section_class(name: str) -> str | None:
    return CLASS_OF.get("." + name.split(".", 2)[1]) if name.startswith(".") else None


def read_modules(text: str) -> list[tuple[str, str]]:
    """(unit, module) rows of a TSV with a header row holding `unit` and `module` columns."""
    rows = list(csv.reader(text.splitlines(), delimiter="\t"))
    if not rows or "unit" not in rows[0] or "module" not in rows[0]:
        raise ValueError("a module map needs a header row with `unit` and `module` columns")
    unit, module = rows[0].index("unit"), rows[0].index("module")
    return [(r[unit], r[module]) for r in rows[1:] if r and not r[0].startswith("#") and len(r) > max(unit, module)]


def read_targets(text: str) -> dict[str, str]:
    """module -> the path of its merged file, from the optional `target` column of a module map; a module without one
    sits at `merged_path`."""
    rows = list(csv.reader(text.splitlines(), delimiter="\t"))
    if not rows or "target" not in rows[0] or "module" not in rows[0]:
        return {}
    module, target = rows[0].index("module"), rows[0].index("target")
    found: dict[str, str] = {}
    for r in rows[1:]:
        if not r or r[0].startswith("#") or len(r) <= max(module, target) or not r[target]:
            continue
        if found.setdefault(r[module], r[target]) != r[target]:
            raise ValueError(f"module {r[module]} names two targets: {found[r[module]]} and {r[target]}")
    return found


def merged_path(module: str, target: str | None = None) -> str:
    """Where a module's merged file sits: the `target` the module map names, else in the folder named by the module,
    under the module file name, the way every unit sits in the folder of its module."""
    if target:
        return target
    path = PurePosixPath(module)
    return (path.with_suffix("") / path.name).as_posix()


def merged_modules(text: str, mapping: Mapping[str, str]) -> str:
    """Rewrite all selected units in one pass, keeping one row per destination and all other rows/comments."""
    lines = text.splitlines(keepends=True)
    col = lines[0].rstrip("\r\n").split("\t").index("unit")
    out = [lines[0]]
    done: set[str] = set()
    for line in lines[1:]:
        cells = line.rstrip("\r\n").split("\t")
        if not line.startswith("#") and len(cells) > col and cells[col] in mapping:
            merged = mapping[cells[col]]
            if merged not in done:
                cells[col] = merged
                out.append("\t".join(cells) + "\n")
                done.add(merged)
            continue
        out.append(line)
    return "".join(out)


def split_modules(text: str, unit: str, pieces: Mapping[str, str]) -> str:
    """A module map with the row of `unit` replaced by one row per piece (target -> path): the unit column holds the
    piece's path and the module column its target; every other cell and row stays."""
    lines = text.splitlines(keepends=True)
    head = lines[0].rstrip("\r\n").split("\t")
    col, module = head.index("unit"), head.index("module") if "module" in head else -1
    out = [lines[0]]
    for line in lines[1:]:
        cells = line.rstrip("\r\n").split("\t")
        if line.startswith("#") or len(cells) <= col or cells[col] != unit:
            out.append(line)
            continue
        for target, path in pieces.items():
            row = list(cells)
            row[col] = path
            if 0 <= module < len(row):
                row[module] = target
            out.append("\t".join(row) + "\n")
    return "".join(out)


def parse_link(text: str) -> list[Row]:
    """The ordered entries, alignment and minimum output offsets emitted by the linker writer."""
    rows: list[Row] = []
    outsec = ""
    subalign = 0
    for line in text.splitlines():
        found = _OUTSEC.match(line)
        if found and "{" not in line:
            outsec = found.group(1)
            sub = re.search(r"SUBALIGN\((\d+)\)", line)
            subalign = int(sub.group(1)) if sub else 0
            continue
        align = _ALIGN.match(line)
        if align:
            rows.append({"o": outsec, "k": "align", "a": int(align.group(1), 0)})
            continue
        minimum = _MINIMUM.match(line)
        if minimum:
            rows.append({"o": outsec, "k": "minimum", "offset": int(minimum.group(1), 0)})
            continue
        conditional = _CONDITIONAL_ALIGN.match(line)
        if conditional:
            rows.append({"o": outsec, "k": "align", "a": int(conditional.group(2)), "symbol": conditional.group(1)})
            continue
        entry = _ENTRY.match(line)
        if entry:
            sections = tuple(s.strip('"') for s in entry.group(2).split())
            classes = tuple(sorted({c for s in sections if (c := _section_class(s)) is not None}))
            rows.append(
                {
                    "o": outsec,
                    "k": "entry",
                    "u": entry.group(1),
                    "cls": classes,
                    "sections": sections,
                    "subalign": subalign,
                }
            )
    return rows


def object_facts(path: Path) -> tuple[dict[str, int], set[str], dict[str, list[tuple[int, int]]], set[str]]:
    """Read exact section sizes, local names, pieces and defined alignment markers once."""
    sizes: dict[str, int] = defaultdict(int)
    local: set[str] = set()
    pieces: dict[str, list[tuple[int, int]]] = defaultdict(list)
    defined: set[str] = set()
    with path.open("rb") as handle:
        elf = ELFFile(handle)
        for section in elf.iter_sections():
            cls = _section_class(section.name)
            if cls is not None:
                sizes[section.name] += section["sh_size"]
            if cls is not None and section["sh_size"]:
                pieces[section.name].append((section["sh_size"], max(section["sh_addralign"], 1)))
        table = elf.get_section_by_name(".symtab")
        if isinstance(table, SymbolTableSection):
            for symbol in table.iter_symbols():
                if symbol["st_shndx"] != "SHN_UNDEF":
                    defined.add(symbol.name)
                kind = symbol["st_info"]["type"]
                if symbol["st_info"]["bind"] == "STB_LOCAL" and kind in ("STT_FUNC", "STT_OBJECT") and symbol.name:
                    local.add(symbol.name)
    return dict(sizes), local, dict(pieces), defined


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


def spans(seq: Sequence[Row], units: set[str]) -> list[tuple[str, int, int, int]]:
    """Per output section holding entries of `units`: (section, bytes the linked entries span, bytes one merged object
    spans, offset in the section where the module starts). The linked span walks the section from the module's first
    entry to its last (the module's pieces aligned as linked, ALIGN directives and `MAX(., offset)` pins between
    them); the bytes of other modules between two entries are the `gap` finding's, so they are not walked and the
    pins after them are not read. The merged span packs the module's pieces of each class again, class after class,
    each as aligned in the link."""
    out: list[tuple[str, int, int, int]] = []
    for section in dict.fromkeys(e["o"] for e in seq if e["k"] == "entry" and e["u"] in units):
        rows = [(i, e) for i, e in enumerate(seq) if e["o"] == section]
        mine = [i for i, e in rows if e["k"] == "entry" and e["u"] in units]
        position, start, foreign, classes = 0, 0, False, defaultdict(list)
        for i, row in rows:
            if i > mine[-1]:
                break
            if i == mine[0]:
                start = position
            if row["k"] == "entry" and row["u"] not in units:
                foreign = foreign or (i > mine[0] and any(row.get("parts", {}).values()))
                if i < mine[0]:
                    for cls in row.get("parts", {}):
                        for size, align in row["parts"][cls]:
                            position = _up(position, row.get("subalign") or align) + size
                continue
            if row["k"] == "align":
                position = _up(position, row["a"])
                continue
            if row["k"] == "minimum":
                position = position if foreign else max(position, row["offset"])
                continue
            for cls in row.get("parts", {}):
                for size, align in row["parts"][cls]:
                    position = _up(position, row.get("subalign") or align) + size
                    if i >= mine[0]:
                        classes[cls].append((size, row.get("subalign") or align))
        merged = 0
        for pieces in classes.values():
            merged = _up(merged, max(a for _, a in pieces))
            for size, align in pieces:
                merged = _up(merged, align) + size
        out.append((section, position - start, merged, start))
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
        defined: set[str] = set()
        for row in seq:
            if row["k"] != "entry":
                continue
            key = (version, row["u"])
            if key not in cache:
                path = root / obj.format(version=version, unit=row["u"])
                if path.is_file():
                    cache[key], local, pieces[key], names = object_facts(path)
                    defined.update(names)
                    statics.setdefault(row["u"], set()).update(local)
                else:
                    cache[key] = None
            sizes = cache[key] or {}
            row["n"] = sum(sizes.get(s, 0) for s in row["sections"])
            parts: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for section in row["sections"]:
                if (cls := _section_class(section)) is not None:
                    parts[cls].extend(pieces.get(key, {}).get(section, ()))
            row["parts"] = dict(parts)
            row["missing"] = cache[key] is None
            row["cls"] = "+".join(row["cls"])
        seqs[version] = [r for r in seq if "symbol" not in r or r["symbol"] in defined]
    return seqs, statics


def _linked(units: Sequence[str], seq: Sequence[Row], cls: str) -> list[str]:
    """The units of `units` that hold bytes of `cls` in one version, in the order that version links them."""
    wanted = set(units)
    return list(
        dict.fromkeys(
            row["u"]
            for row in seq
            if row["k"] == "entry" and row["u"] in wanted and row.get("n", 0) > 0 and cls in row["cls"].split("+")
        )
    )


def code_order(units: Sequence[str], seqs: Mapping[str, Sequence[Row]], cls: str = "text") -> list[str]:
    """The units in the base order of a class: the versions' orders merged (a unit only a later version places goes
    after its neighbour there); a unit without that class follows the others, in the order given."""
    base = classwise.base_order(_linked(units, seq, cls) for seq in seqs.values())
    return base + [u for u in units if u not in set(base)]


def class_orders(units: Sequence[str], seqs: Mapping[str, Sequence[Row]]) -> dict[str, dict[str, list[str]]]:
    """Per section class, per version, the units that hold bytes of it in the order that version links them."""
    out: dict[str, dict[str, list[str]]] = {}
    for cls in CLASSES:
        per = {version: order for version, seq in seqs.items() if (order := _linked(units, seq, cls))}
        if per:
            out[cls] = per
    return out


def object_classes(path: Path) -> dict[str, str]:
    """symbol -> section class of every symbol an object defines in a section of a known class."""
    out: dict[str, str] = {}
    with path.open("rb") as handle:
        elf = ELFFile(handle)
        table = elf.get_section_by_name(".symtab")
        if isinstance(table, SymbolTableSection):
            for symbol in table.iter_symbols():
                index = symbol["st_shndx"]
                if symbol.name and isinstance(index, int) and (cls := _section_class(elf.get_section(index).name)):
                    out[symbol.name] = cls
    return out


def section_parts(placed: Mapping[str, Mapping[str, str]], order: Sequence[str]) -> list[Row]:
    """The parts of one class of a module: units that share their output section in every version where both are
    placed (and are placed together in one), joined through such pairs. `placed` is version -> unit -> section; each
    part is {"sections": {version: section}, "units": [...]}, units in `order`, parts in the order of their first unit."""
    units = [u for u in order if any(u in per for per in placed.values())]
    parent = {u: u for u in units}

    def find(u: str) -> str:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for i, one in enumerate(units):
        for two in units[i + 1 :]:
            common = [per for per in placed.values() if one in per and two in per]
            if common and all(per[one] == per[two] for per in common):
                parent[find(two)] = find(one)
    groups: dict[str, list[str]] = {}
    for u in units:
        groups.setdefault(find(u), []).append(u)
    return [
        {"sections": {v: per[members[0]] for v, per in placed.items() if members[0] in per}, "units": members}
        for members in groups.values()
    ]


def _part_label(sections: Mapping[str, str]) -> str:
    """`.text` when every version links the part in that section, else `us .text, eu .boot`."""
    if len(set(sections.values())) == 1:
        return next(iter(sections.values()))
    return ", ".join(f"{v} {s}" for v, s in sections.items())


def verdicts(
    rows: Sequence[tuple[str, str]],
    seqs: Mapping[str, Sequence[Row]],
    statics: Mapping[str, set[str]],
    identity: Callable[[str], Any],
) -> list[Row]:
    """One verdict per module of two or more units: {"module", "units", "verdict": merge|separate, "hard", "soft"}."""
    modules: dict[str, list[str]] = defaultdict(list)
    for unit, module in rows:
        modules[module].append(unit)
    multi = {m: us for m, us in modules.items() if len(us) > 1}
    owner = {u: m for m, us in multi.items() for u in us}
    hard: dict[str, list[str]] = defaultdict(list)
    soft: dict[str, list[str]] = defaultdict(list)
    placed = {row["u"] for seq in seqs.values() for row in seq if row["k"] == "entry"}
    for module, units in multi.items():
        for unit in units:
            if unit not in placed:
                hard[module].append(f"unplaced: {unit} has no linker input in any version")
        if len({identity(u) for u in units}) > 1:
            hard[module].append("flags: the units differ in toolchain or flags")
        names: dict[str, str] = {}
        for unit in units:
            for name in statics.get(unit, ()):
                if name in names and names[name] != unit:
                    soft[module].append(f"statics: {name} is local to {names[name]} and {unit}")
                names.setdefault(name, unit)
    layout: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(dict)
    placed_in: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    split: set[tuple[str, str]] = set()
    for version, seq in seqs.items():
        for row in seq:
            if row.get("missing") and (missing_owner := owner.get(row["u"])):
                hard[missing_owner].append(f"object: {version} {row['u']} has no built object")
        live = [i for i, e in enumerate(seq) if e["k"] == "entry" and e.get("n", 0) > 0]
        rodata = {e["u"] for e in seq if e["k"] == "entry" and e["cls"] == RODATA and e["n"] > 0}
        following = dict(pairwise(live))
        where: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
        for i in live:
            if home := owner.get(seq[i]["u"]):
                for part in seq[i]["cls"].split("+"):
                    where[(home, part)].append((i, seq[i]["o"], seq[i]["u"]))
        for (module, cls), hits in where.items():
            for _, o, u in hits:
                placed_in[module, cls][version][u] = o
            if len({o for _, o, _ in hits}) > 1:
                split.add((module, cls))
                hard[module].append(f"sections: {version} {cls} sits in several output sections")
            layout[module, cls][version] = list(dict.fromkeys(u for _, _, u in hits))
            for (i, _, one), (j, _, two) in pairwise(hits):
                if following.get(i) != j:
                    between = [seq[k]["u"] for k in live if i < k < j and owner.get(seq[k]["u"]) != module]
                    if between:
                        hard[module].append(f"gap: {version} {cls} {one} .. {two} has {', '.join(between[:3])} between")
                for k in range(i + 1, j):
                    if seq[k]["k"] == "align":
                        soft[module].append(f"align: {version} {cls} ALIGN({seq[k]['a']}) before {two}")
        for module, units in multi.items():
            for section, linked, merged, _ in spans(seq, set(units)):
                shape = f"{version} {section} merged {merged} B against linked {linked} B"
                if merged > linked:
                    hard[module].append(f"lma: {shape}: overlaps the next section")
                elif merged < linked:
                    soft[module].append(f"fill: {shape}: the link fill of the {linked - merged} B is regenerated")
            held = [u for u in units if u in rodata]
            if len(held) > 1:
                soft[module].append(f"literals: {version} rodata in {len(held)} units")
    parts: dict[str, dict[str, list[Row]]] = defaultdict(dict)
    for module, cls in sorted(split):
        base = classwise.base_order(layout[module, cls].values())
        parts[module][cls] = found = section_parts(placed_in[module, cls], base)
        shown = "; ".join(f"{_part_label(p['sections'])} [{', '.join(p['units'])}]" for p in found)
        hard[module].append(f"sections: {cls} parts: {len(found)}: {shown}")
    for (module, cls), per in layout.items():
        base = classwise.base_order(per.values())
        for version, seen in per.items():
            if gone := classwise.moved(base, seen):
                soft[module].append(f"order: {version} {cls} links {', '.join(gone)} elsewhere than the other versions")
    return [
        {
            "module": m,
            "units": len(us),
            "verdict": "separate" if hard[m] else "merge",
            "hard": sorted(set(hard[m])),
            "soft": sorted(set(soft[m])),
            "parts": parts.get(m, {}),
        }
        for m, us in multi.items()
    ]
