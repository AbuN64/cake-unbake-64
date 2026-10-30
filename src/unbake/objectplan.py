"""Which targets of a recovered object plan may become one source file: the object the linker built, not the unit.

The plan is three TSVs of a source plan: FILES.tsv (target, object indices per version), OBJECTS.tsv (every recovered
object per version and segment, with its start and end) and ASSIGN.tsv (every function, data symbol, variable and
linker pad atom with its address, size, object, target and linkage). `judge` lays the merged object of each target out
the way the linker places it and compares the layout with the ROM addresses:
  split      one object's code spans two code segments
  gap        another file's object sits between two runs of the target in one data or code region, or bytes of no atom
             fill a hole larger than the alignment of the next atom, or a pad atom is not linker fill
  lma        two atoms overlap, a function is not 4-aligned, or a data atom is not reachable by its own alignment
  plan       FILES.tsv and OBJECTS.tsv name different objects for the target
Pad atoms (linkage drop, under 16 bytes) are the object's alignment fill and are absorbed. A target whose runs are not
adjacent but whose first run passes every check settles: the first run keeps the target, later runs keep the module
named by their functions (the settlement of the plan), so the verdict is `settle`; `merge` is a single clean run."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Row = dict[str, Any]
VERSIONS = ("de", "eu", "eu-x", "us", "us-rev1")
ABSENT = "-"
MAX_ALIGN = 16
CODE = "function"
PAD = "pad"
_DATA_SEGMENT = re.compile(r"^bss$|data|^\.rdata", re.IGNORECASE)


@dataclass(frozen=True)
class Atom:
    address: int
    size: int
    kind: str
    linkage: str
    object: int
    target: str
    symbol: str


@dataclass
class Plan:
    files: list[Row] = field(default_factory=list)
    objects: dict[tuple[str, str], set[int]] = field(default_factory=lambda: defaultdict(set))
    segments: dict[tuple[str, int], dict[str, tuple[int, int]]] = field(default_factory=lambda: defaultdict(dict))
    atoms: dict[tuple[str, str], list[Atom]] = field(default_factory=lambda: defaultdict(list))


def _rows(text: str, columns: Sequence[str], name: str) -> list[Row]:
    rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    for column in columns:
        if not rows or column not in rows[0]:
            raise ValueError(f"{name} needs a header row with a `{column}` column")
    return rows


def read_plan(files: str, objects: str, assign: str) -> Plan:
    """Parse FILES.tsv, OBJECTS.tsv and ASSIGN.tsv text into one plan."""
    plan = Plan(files=_rows(files, ["target", *(f"obj_{v}" for v in VERSIONS)], "FILES.tsv"))
    for row in _rows(objects, ["version", "object", "target", "segment", "start", "end"], "OBJECTS.tsv"):
        plan.objects[(row["version"], row["target"])].add(int(row["object"]))
        plan.segments[(row["version"], int(row["object"]))][row["segment"]] = (
            int(row["start"], 16),
            int(row["end"], 16),
        )
    columns = ["version", "symbol", "kind", "segment", "address", "size", "object", "target", "linkage"]
    for row in _rows(assign, columns, "ASSIGN.tsv"):
        atom = Atom(
            int(row["address"], 16),
            int(row["size"]),
            row["kind"],
            row["linkage"],
            int(row["object"]),
            row["target"],
            row["symbol"],
        )
        plan.atoms[(row["version"], row["segment"])].append(atom)
    for atoms in plan.atoms.values():
        atoms.sort(key=lambda a: (a.address, a.size))
    return plan


def _up(position: int, align: int) -> int:
    return -(-position // align) * align


def reachable(cursor: int, address: int) -> bool:
    """Whether some alignment up to 16 places a symbol that follows `cursor` exactly at `address`."""
    return any(_up(cursor, 1 << p) == address for p in range(MAX_ALIGN.bit_length()))


def regions(atoms: Sequence[Atom]) -> list[list[Atom]]:
    """A segment's atoms in address order cut into the regions in which the objects appear in object order: a region
    ends where the object index falls."""
    out: list[list[Atom]] = []
    last = -1
    for atom in atoms:
        if not out or atom.object < last:
            out.append([])
        out[-1].append(atom)
        last = atom.object
    return out


def runs(region: Sequence[Atom]) -> list[list[Atom]]:
    """The maximal stretches of one target in a region."""
    out: list[list[Atom]] = []
    for atom in region:
        if not out or out[-1][-1].target != atom.target:
            out.append([])
        out[-1].append(atom)
    return out


def is_code(atom: Atom, segment: str) -> bool:
    """A function is code only in a code segment: a named literal pool the extraction typed as a function but placed in
    a data segment is data."""
    return atom.kind == CODE and not _DATA_SEGMENT.search(segment)


def lay_out(run: Sequence[Atom], segment: str) -> tuple[list[str], int, int, int]:
    """Place one stretch's atoms one after the other from the first atom's address with the alignment the linker gives
    them; the findings are the ROM addresses the layout cannot reproduce. A hole between two atoms of one object is
    that object's own unnamed bytes; a hole between two objects is linker fill and must be under one alignment.
    Returns (findings, pad bytes, align bytes, unnamed bytes)."""
    findings: list[str] = []
    pad = fill = unnamed = 0
    cursor = run[0].address
    before = run[0]
    for atom in run:
        code = is_code(atom, segment)
        if atom.kind == PAD:
            if atom.linkage != "drop" or atom.size >= MAX_ALIGN:
                findings.append(f"gap: {atom.symbol} {atom.size} B at {atom.address:08X} is not linker fill")
            else:
                pad += atom.size
        elif code and atom.address % 4:
            findings.append(f"lma: function {atom.symbol} at {atom.address:08X} is not 4-aligned")
        if atom.address < cursor:
            findings.append(
                f"lma: {atom.symbol} at {atom.address:08X} overlaps the atom before it by {cursor - atom.address} B"
            )
        elif atom.address > cursor:
            hole = atom.address - cursor
            if code and atom.address - cursor:
                findings.append(f"gap: {hole} B of no atom before function {atom.symbol} at {atom.address:08X}")
            elif atom.object == before.object:
                unnamed += hole
            elif hole < MAX_ALIGN and reachable(cursor, atom.address):
                fill += hole
            else:
                findings.append(
                    f"gap: {hole} B of no atom between objects {before.object} and {atom.object} "
                    f"before {atom.symbol} at {atom.address:08X}"
                )
        cursor = max(cursor, atom.address + atom.size)
        before = atom
    return findings, pad, fill, unnamed


@dataclass
class Stretch:
    """One target's adjacent atoms in one region of a segment, and the other files' targets that follow it up to the
    target's next stretch in the same region."""

    segment: str
    region: int
    atoms: list[Atom]


def stretches(plan: Plan) -> dict[tuple[str, str], list[Stretch]]:
    """Every stretch of every (version, target), in address order."""
    out: dict[tuple[str, str], list[Stretch]] = defaultdict(list)
    for (version, segment), atoms in sorted(plan.atoms.items()):
        for number, region in enumerate(regions(atoms)):
            for run in runs(region):
                out[(version, run[0].target)].append(Stretch(segment, number, run))
    return out


def judge(plan: Plan) -> list[Row]:
    """One verdict per (version, target) of FILES.tsv: {"version", "target", "verdict", "reason"}, the verdict one of
    merge, settle, refuse, absent."""
    placed = stretches(plan)
    out: list[Row] = []
    for row in plan.files:
        target = row["target"]
        for version in VERSIONS:
            listed = {int(o) for o in row[f"obj_{version}"].split(",") if o and o != ABSENT}
            verdict = (
                _verdict(plan, version, target, listed, placed.get((version, target), []))
                if listed
                else {"verdict": "absent", "reason": "no object in this version"}
            )
            out.append({"version": version, "target": target, **verdict})
    return out


def _verdict(plan: Plan, version: str, target: str, listed: set[int], found: Sequence[Stretch]) -> Row:
    refuse: list[str] = []
    named = plan.objects.get((version, target), set())
    if named != listed:
        refuse.append(f"plan: FILES.tsv names objects {sorted(listed)} and OBJECTS.tsv {sorted(named)}")
    code = sorted({s.segment for s in found if any(is_code(a, s.segment) for a in s.atoms)})
    if len(code) > 1:
        refuse.append(f"split: the code of one object spans {' and '.join(code)}")
    functions = data = pads = fills = holes = 0
    later: dict[tuple[str, int], list[Stretch]] = defaultdict(list)
    for stretch in found:
        findings, pad, fill, unnamed = lay_out(stretch.atoms, stretch.segment)
        refuse.extend(f"{stretch.segment}: {f}" for f in findings[:2])
        pads, fills, holes = pads + pad, fills + fill, holes + unnamed
        functions += sum(1 for a in stretch.atoms if is_code(a, stretch.segment))
        data += sum(1 for a in stretch.atoms if a.kind != PAD and not is_code(a, stretch.segment))
        later[(stretch.segment, stretch.region)].append(stretch)
    objects = ",".join(str(o) for o in sorted(listed))
    if refuse:
        return {"verdict": "refuse", "reason": "; ".join(refuse[:4])}
    measured = (
        f"objects {objects}: {functions} functions, {data} data, {pads} B pad absorbed, {fills} B align fill, "
        f"{holes} B unnamed inside objects"
    )
    split = {k: v for k, v in later.items() if len(v) > 1}
    if not split:
        return {"verdict": "merge", "reason": measured}
    notes = []
    for (segment, _), parts in sorted(split.items()):
        kept = sorted({a.object for a in parts[0].atoms})
        moved = sorted({a.object for part in parts[1:] for a in part.atoms})
        notes.append(f"{segment} run {kept} keeps the target, run {moved} is not adjacent")
    return {"verdict": "settle", "reason": f"{measured}; " + "; ".join(notes)}


def tsv(rows: Sequence[Row]) -> str:
    """mergecheck.tsv: version, target, verdict, measured reason."""
    lines = ["version\ttarget\tverdict\treason"]
    lines.extend(f"{r['version']}\t{r['target']}\t{r['verdict']}\t{r['reason']}" for r in rows)
    return "\n".join(lines) + "\n"


def read_dir(files: Path) -> Plan:
    """The plan whose FILES.tsv is `files`, with OBJECTS.tsv and ASSIGN.tsv beside it."""
    return read_plan(
        files.read_text(), (files.parent / "OBJECTS.tsv").read_text(), (files.parent / "ASSIGN.tsv").read_text()
    )
