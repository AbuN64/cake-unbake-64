"""The audit of the linker's fill records: a record sits in a gap of its own, and only in a gap the object plan has.

A fill record (`linkfill`) stands for stale bytes the original link left between two objects. Two things make it a lie:
it overlaps a symbol (the bytes are somebody's code or data, not fill), or the object plan has no pad atom of that
object, address and size (the plan never saw a gap there). `unbake audit` refuses both, one line each, from the record
files, `versions/<v>/symbol_addrs.txt` and the plan: no ROM, no build."""

from __future__ import annotations

import itertools
import tomllib
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

from unbake import linkfill, objectplan, versions
from unbake.contracts import Finding
from unbake.linkfill import Fill

Pad = tuple[str, str, int, int]  # object, section, address, size of a pad atom of the plan


def pads(plan: objectplan.Plan, version: str) -> frozenset[Pad]:
    """The plan's alignment pads of one version."""
    return frozenset(
        (str(a.object), section, a.address, a.size)
        for (v, section), atoms in plan.atoms.items()
        if v == version
        for a in atoms
        if a.kind == linkfill.PAD_KIND
    )


def _spans(symbols: Sequence[versions.Row], fills: Sequence[Fill]) -> list[tuple[int, int, str]]:
    """Each symbol's (start, end, name) by address. A symbol with no stated size ends at the next symbol or fill row,
    as in the link; one with none after it holds its first byte."""
    starts = sorted({s.vram for s in symbols} | {f.address for f in fills})
    after = dict(itertools.pairwise(starts))
    return [
        (s.vram, s.vram + (s.size if s.size is not None else after.get(s.vram, s.vram + 1) - s.vram), s.name)
        for s in symbols
        if s.size != 0
    ]


def findings(
    version: str, fills: Sequence[Fill], symbols: Sequence[versions.Row], plan: Collection[Pad]
) -> list[Finding]:
    """The refusals of one version's records, in ROM order, each one line: a record that overlaps a symbol
    (`fill.overlap`) and a record whose gap the plan does not have (`fill.gap`)."""
    spans = _spans(symbols, fills)
    found = []
    for f in sorted(fills, key=lambda f: f.rom):
        at = f"{version}: fill at 0x{f.rom:X} ({f.size} bytes, object {f.object})"
        if hit := next((n for a, b, n in spans if a < f.address + f.size and f.address < b), None):
            found.append(
                Finding(
                    "fill.overlap",
                    f"{at} overlaps symbol {hit}; the bytes are the symbol's, not fill",
                    path=linkfill.path(version),
                    unit=hit,
                    versions=(version,),
                    action="drop the record, or fix the symbol's size",
                )
            )
        if (f.object, f.section, f.address, f.size) not in plan:
            found.append(
                Finding(
                    "fill.gap",
                    f"{at} is in {f.section} at 0x{f.address:08X}, a gap the object plan does not have",
                    path=linkfill.path(version),
                    versions=(version,),
                    action="regenerate with `unbake layout fill --plan FILES.tsv`",
                )
            )
    return found


def audit(root: Path, plan_files: str | None) -> list[Finding]:
    """The refusals of every version's fill records. Records but no plan to judge them by is itself a refusal."""
    config = tomllib.loads((root / "config.toml").read_text()) if (root / "config.toml").is_file() else {}
    held = {v: linkfill.read(root, v) for v in config.get("version", {})}
    held = {v: fills for v, fills in held.items() if fills}
    if not held:
        return []
    if plan_files is None:
        return [
            Finding(
                "fill.plan",
                f"{', '.join(held)} hold fill records but the audit has no --plan FILES.tsv to judge their gaps by",
                action="rerun `unbake audit --against REV --plan FILES.tsv`",
                versions=tuple(held),
            )
        ]
    plan = objectplan.read_dir(Path(plan_files))
    out: list[Finding] = []
    for v, fills in held.items():
        file = root / versions.symbol_file(v)
        rows: Iterable[versions.Row] = (
            versions.parse_symbol_addrs(file.read_bytes(), str(file)) if file.is_file() else ()
        )
        out += findings(v, fills, tuple(rows), pads(plan, v))
    return out
