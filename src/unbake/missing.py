"""Symbols the assembly names that no symbol fact defines: classified by provenance, reported, and written.

splat names an address with no symbol `D_auto_<ADDR>` / `func_auto_<ADDR>`; a stand-in or a C file spells such a
place the same way. Each undefined name is one of
  a  a RAM address (a hardware register is reported, never written): gets a real `D_`/`func_` fact;
  b  a segment-relative or ROM offset (`D_00xxxxxx`, no KSEG bit): a constant the code builds, kept as itself;
  c  an address another fact already names: the fact is the answer, nothing is added;
  d  anything else, reported with the file and line that names it."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from unbake import symbols
from unbake.contracts import Config, Json

_NAME = re.compile(r"^(?:(D|func|jtbl)_)(?:auto_)?([0-9A-Fa-f]{8})$")
_AUTO = re.compile(r"\b((?:D|func)_auto_[0-9A-Fa-f]{8})\b")
_REFERENCE = re.compile(r"(?:(%(?:hi|lo|gp_rel)\()|\b(jal|j)\s+|(\.word)\s+)([A-Za-z_]\w*)")
_LABEL = re.compile(r"^\s*(?:glabel|dlabel|jlabel|[A-Za-z_]\w*:)\s*(\S*)")
RAM_END = 0x80800000  # the end of expansion-pak RAM: data below it and outside every loaded segment is bss
_LOCAL = (".L", "L8", "L0")
_SIZES = {
    "lw": 4,
    "sw": 4,
    "lwc1": 4,
    "swc1": 4,
    "lh": 2,
    "lhu": 2,
    "sh": 2,
    "lb": 1,
    "lbu": 1,
    "sb": 1,
    "ld": 8,
    "sd": 8,
}


def references(config: Config, version: str) -> dict[str, list[tuple[str, int, str]]]:
    """name -> [(project-relative file, line, how)] over the version's extracted assembly; how is `call` (jal/j),
    `reloc` (%hi/%lo/%gp_rel) or `word`. Names the assembly defines itself (a label) are left out."""
    root = config.project.root
    seen: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    defined: set[str] = set()
    for file in sorted((root / "asm" / version).rglob("*.s")):
        rel = file.relative_to(root).as_posix()
        for number, line in enumerate(file.read_text(errors="replace").splitlines(), 1):
            label = re.match(r"^\s*(?:glabel|dlabel|jlabel)\s+(\S+)", line)
            if label:
                defined.add(label[1])
                continue
            for hi, call, _word, name in _REFERENCE.findall(line):
                if not name.startswith(_LOCAL):
                    seen[name].append((rel, number, "reloc" if hi else "call" if call else "word"))
    spelled = _spelled_in_c(config)
    return {n: [*rows, *spelled.get(n, [])] for n, rows in seen.items() if n not in defined}


def _spelled_in_c(config: Config) -> dict[str, list[tuple[str, int, str]]]:
    """The splat names (`D_auto_`, `func_auto_`) the C sources and headers spell: C shared by every version links to
    them whichever version it is built for, so a real name for the place must leave them defined."""
    root = config.project.root
    found: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for folder in ("src", "include"):
        for file in sorted((root / folder).rglob("*")):
            if file.is_file() and file.suffix.lower() in (".c", ".cpp", ".h"):
                rel = file.relative_to(root).as_posix()
                for number, text in enumerate(file.read_text(errors="replace").splitlines(), 1):
                    for name in _AUTO.findall(text):
                        found[name].append((rel, number, "source"))
    return found


def _ram(config: Config, version: str, address: int) -> tuple[int | None, str | None]:
    """(rom, segment) of a RAM address in a loaded segment of the version, else (None, None)."""
    for low, high, rom in symbols.spans(config, version):
        if low <= address < high:
            return rom + address - low, symbols.segment_at(config, version, rom + address - low)
    return None, None


def _classify(config: Config, version: str, name: str, rows: list[tuple[str, int, str]], table: Json) -> Json | None:
    found = _NAME.match(name)
    if not found:
        return None  # a name that spells no address is not this report's to place
    address = int(found[2], 16)
    wants_code = {how for _, _, how in rows if how != "source"} == {"call"}
    row: Json = {
        "name": name,
        "version": version,
        "address": address,
        "refs": len(rows),
        "in_c": any(how == "source" for _, _, how in rows),
        "at": [f"{file}:{line}" for file, line, _ in rows[:3]],
    }
    named = sorted(n for n, (vram, *_rest) in table.items() if vram == address)
    if named:
        return {**row, "class": "c", "symbol": named[0]}
    if address < 0x80000000:
        rom, segment = _ram(config, version, address)
        ram = None
        for low, high, start in symbols.spans(config, version):
            if start <= address < start + high - low:
                ram = low + address - start
        return {**row, "class": "b", "ram": ram, "segment": segment, "proposed": f"D_{address:08X}_{_tag(version)}"}
    if address >= 0xA0000000:  # an uncached (KSEG1) or hardware register address: no segment holds it
        return {**row, "class": "a", "kind": "data", "proposed": f"D_{address:08X}", "register": True}
    rom, segment = _ram(config, version, address)
    if rom is None and (wants_code or address >= RAM_END):
        return {**row, "class": "d", "reason": "no loaded segment of the version holds this address"}
    kind = "func" if wants_code else "data"
    size = max((_SIZES.get(op, 0) for op in _ops(config, rows)), default=0) or None
    return {
        **row,
        "class": "a",
        "kind": kind,
        "rom": rom,  # None for bss: no ROM bytes hold it
        "segment": segment,
        "size": None if wants_code else size,
        "proposed": f"{'func' if wants_code else 'D'}_{address:08X}_{_tag(version)}",
    }


def _ops(config: Config, rows: list[tuple[str, int, str]]) -> list[str]:
    ops = []
    for file, number, how in rows[:8]:
        if how == "reloc":
            line = (config.project.root / file).read_text(errors="replace").splitlines()[number - 1]
            op = re.match(r"\s*/\*.*?\*/\s*(\w+)", line)
            ops += [op[1]] if op else []
    return ops


def _tag(version: str) -> str:
    return version.replace("-", "_")


def report(config: Config, versions: list[str]) -> dict[str, Any]:
    """Every undefined referenced name per version, classified, as {"versions": {v: {count, classes, rows}}}."""
    out: dict[str, Any] = {}
    for version in versions:
        table = symbols.load(config, version)
        rows = []
        for name, refs in sorted(references(config, version).items()):
            if name in table:
                continue
            row = _classify(config, version, name, refs, table)
            if row is not None:
                rows.append(row)
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["class"]] += 1
        out[version] = {"count": len(rows), "classes": dict(sorted(counts.items())), "rows": rows}
    return {"versions": out}


def line(row: Json) -> str:
    """The symbol_addrs.txt row for a class a or b finding."""
    attrs = []
    if row["class"] == "a" and row["kind"] == "func":
        attrs.append("type:func")
    if row.get("rom") is not None:
        attrs.append(f"rom:0x{row['rom']:X}")
    if row.get("size"):
        attrs.append(f"size:0x{row['size']:X}")
    return f"{row['proposed']} = 0x{row['address']:08X}; // {' '.join(attrs)}".rstrip() + "\n"


def write(config: Config, document: Json) -> Json:
    """Append the class a and b facts to versions/<v>/symbol_addrs.txt; a name the file already holds is left alone.
    A hardware register stays reported only: a fact at its address makes the policy refuse every C spelling it.
    A name the C spells (it is shared by every version, and the real name differs per version) stays, in
    versions/<v>/aliases.txt, a second name of the same address: the C links as it did."""
    written: dict[str, list[str]] = {}
    for version, body in document["versions"].items():
        add = [r for r in body["rows"] if r["class"] in ("a", "b") and not r.get("register")]
        written[version] = _append(config, symbols.path(version), add, "proposed", lambda r: line(r))
        kept = [r for r in add if r["in_c"]]
        _append(
            config,
            symbols.aliases_path(version),
            kept,
            "name",
            lambda r: line({**r, "proposed": r["name"], "rom": None, "size": None}),
        )
    return written


def _append(config: Config, where: str, rows: list[Json], key: str, render: Any) -> list[str]:
    file: Path = config.project.root / where
    text = file.read_text() if file.exists() else ""
    have = symbols.parse(text, where)
    add = [r for r in rows if r[key] not in have]
    if add:
        sep = "" if not text or text.endswith("\n") else "\n"
        file.write_text(text + sep + "".join(render(r) for r in add))
    return [r[key] for r in add]


def command(config: Config, params: Json) -> Json:
    versions: Any = [params["version"]] if params.get("version") else list(config.project.versions)
    document: dict[str, Any] = report(config, versions)
    if params.get("write"):
        document["written"] = write(config, document)
    return {"missing": document}
