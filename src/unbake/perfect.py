"""The perfect metric: a module (`[[group]]`) is perfect when it is one original object with the right flags and every
member is exact. Four axes per code module:

- B boundary: both edges are proven cuts (edges.py, an SDK module change, a project evidence row or a flag proof), no
  proven cut lies inside, no edge sits at a proven join, and the module is laid as one unit
- F flags: "default" (no options, every function exact) or "proven" (a recorded proof); never "conflict" or "open"
- M members: every member exact in every version holding it
- N name: the source path comes from evidence (SDK, library, project evidence row), else "assigned"

perfect = B and F and M; N is counted apart."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from typing import Any

from unbake import config as configuration
from unbake import edges as edge_classes
from unbake import effort, infer, layout
from unbake.contracts import Json, Snapshot, UnitSpec

PROVEN = frozenset({"pad", "segment", "gap", "version", "sdk", "evidence", "flags"})
_LABEL = {"pad": "pad", "seg": "segment", "gap": "gap"}
_RANK = {"join": 0, "open": 1}  # worst first: a defect, then unproven; every proven label ranks after


def evidence_rows(snapshot: Snapshot) -> tuple[Json, ...]:
    """Project evidence rows (`[[evidence.object]]` in config.toml): first, last, source, origin."""
    return tuple(snapshot.config.project.evidence_objects)


def _edge(
    e: edge_classes.Edges,
    version: str,
    left: str | None,
    right: str | None,
    sdk: Mapping[str, str],
    marked: Mapping[tuple[str, str], str],
) -> str:
    """The label of the edge between two functions in one version; None on a side is a segment end."""
    if left is None or right is None:
        return "segment"
    own = e.classes.get((left, right), {}).get(version, "")
    kind = e.verdicts.get((left, right), "open")
    if kind == "cut":
        return _LABEL.get(own, "version")
    if kind == "join":
        return "join"
    if (left, right) in marked:  # a stored proof decided an open edge
        return marked[left, right]
    a, b = sdk.get(left, ""), sdk.get(right, "")
    if a and b:
        return "sdk" if a != b else "join"
    return "open"  # al0 everywhere, or a contradiction


def _stem(path: str) -> str:
    """A file's name without folder or suffix, lower case: TURRETAI.CPP and src/game/ai/enemies/turretai.c agree."""
    return path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()


def _worst(labels: Iterable[str]) -> str:
    seen = list(labels)
    return min(seen, key=lambda label: _RANK.get(label, 2)) if seen else "open"


def _built(snapshot: Snapshot, unit: UnitSpec | None) -> bool:
    """Whether a unit is matched code in every holder: decompiled source, or assembly in a library directory."""
    if unit is None or unit.withheld:
        return False
    kind = configuration.kind(unit.kind)
    return bool(
        kind["decompiled"] or (kind["credit"] == "code" and configuration.library(snapshot.config.project, unit.path))
    )


def _flags(options: bool, built: bool, edges: Iterable[str], mixed: bool = False) -> str:
    """default: no options and every function compiles exact; proven: a flag set every function compiles exact under;
    conflict: units of one module under two sets, or a flag set on a module that sits inside an object (an edge at a
    proven join): one object cannot have two sets, so this C is not the original; open: not every function compiles
    yet."""
    if mixed or (options and "join" in edges):
        return "conflict"
    if not built:
        return "open"
    return "proven" if options else "default"


def _code_modules(snapshot: Snapshot) -> dict[str, list[str]]:
    lay = snapshot.layout
    return {
        g.name: functions
        for g in lay.groups.values()
        if (functions := [n for n in g.members if n in lay.members and lay.members[n].kind == "function"])
    }


def objects(snapshot: Snapshot) -> dict[str, Json]:
    """The derived `object` record of every code module, once per snapshot."""
    return effort.memo(("perfect.objects", snapshot.digest), lambda: _objects(snapshot))


def _marked(snapshot: Snapshot) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """(edges a project evidence row proves, member -> the row's source)."""
    marked: dict[tuple[str, str], str] = {}
    named: dict[str, str] = {}
    lay = snapshot.layout
    for row in evidence_rows(snapshot):
        first, last = row.get("first", ""), row.get("last", "")
        if first in lay.members and last in lay.members:
            for v in lay.members[first].holders():
                e = edge_classes.of(snapshot)
                before, after = e.before.get(v, {}).get(first), e.after.get(v, {}).get(last)
                if before:
                    marked[before, first] = "evidence"
                if after:
                    marked[last, after] = "evidence"
            named[first] = row.get("source", "")
    return marked, named


def _objects(snapshot: Snapshot) -> dict[str, Json]:
    with effort.stage("perfect.objects"):
        e = edge_classes.of(snapshot)
        lay, sdk_hits = snapshot.layout, infer.sdk(snapshot)
        sdk = {n: h.module for n, h in dict(sdk_hits).items() if h.module}
        marked, named = _marked(snapshot)
        boundaries = set(lay.boundaries)
        out: dict[str, Json] = {}
        for name, functions in _code_modules(snapshot).items():
            group = lay.groups[name]
            starts, ends, cut_inside, apart = [], [], False, False
            for v in sorted(lay.held(functions)):
                before, after = e.before.get(v, {}), e.after.get(v, {})
                order = sorted(
                    (p.rom_start, n)
                    for n in functions
                    for p in lay.members[n].placements
                    if p.version == v and p.section == ".text"
                )
                held = [n for _, n in order]
                first, last = held[0], held[-1]
                starts.append(_edge(e, v, before.get(first), first, sdk, marked))
                ends.append(_edge(e, v, last, after.get(last), sdk, marked))
                for left, right in pairwise(held):
                    if after.get(left) != right:
                        apart = True  # another module's function sits between: two objects at least
                    elif _edge(e, v, left, right, sdk, marked) in PROVEN - {"evidence", "flags"}:
                        cut_inside = True
            units = {u.path if (u := layout.unit_of(snapshot, n)) else n for n in functions}
            debt = sum(1 for n in functions[1:] if n in boundaries)
            sets = {
                json.dumps(group.options or u.options, sort_keys=True)
                for p in units
                if (u := lay.units.get(p)) is not None
            }
            options = any(group.options.values()) or any(
                any(u.options.values()) for p in units if (u := lay.units.get(p)) is not None
            )
            built = all(_built(snapshot, layout.unit_of(snapshot, n)) for n in functions)
            hits = {sdk.get(n, "") for n in functions}
            if _stem(named.get(functions[0], "")) and _stem(named[functions[0]]) == _stem(group.source):
                label = "evidence"  # the module starts where the row does and its file is the one the row names
            elif len(hits) == 1 and "" not in hits:
                label = "sdk"
            elif group.library:
                label = "library"
            else:
                label = "assigned"
            record = {
                "start": _worst(starts),
                "end": _worst(ends),
                "inside": "cut" if cut_inside or apart else "joined",
                "units": len(units),
                "flags": _flags(options, built, starts + ends, len(sets) > 1),
                "name": label,
            }
            record["boundary"] = (
                record["start"] in PROVEN
                and record["end"] in PROVEN
                and record["inside"] == "joined"
                and record["units"] == 1
                and not debt
            )
            record["split_debt"] = debt
            out[name] = record
        return out


def census(snapshot: Snapshot, credits: Sequence[Any]) -> tuple[dict[str, Json], frozenset[str]]:
    """(per version the perfect census, the perfect modules). `credits` are report.credits rows: a module's members
    are exact when every row of every member is."""
    return effort.memo(("perfect.census", snapshot.digest), lambda: _census(snapshot, credits))


def _census(snapshot: Snapshot, credits: Sequence[Any]) -> tuple[dict[str, Json], frozenset[str]]:
    with effort.stage("perfect.census"):
        records = objects(snapshot)
        e = edge_classes.of(snapshot)
        lay = snapshot.layout
        exact: dict[str, bool] = defaultdict(lambda: True)
        for row in credits:
            exact[row.name] &= bool(row.exact)
        perfect: dict[str, bool] = {}
        members: dict[str, bool] = {}
        for name, record in records.items():
            members[name] = all(exact[n] for n in lay.groups[name].members if n in lay.members)
            perfect[name] = bool(record["boundary"] and record["flags"] in ("default", "proven") and members[name])
        totals: dict[str, Counter[str]] = defaultdict(Counter)
        for row in credits:
            size = row.code + row.data
            totals[row.version]["total"] += size
            if row.member.group and perfect.get(row.member.group):
                totals[row.version]["bytes"] += size
        contradictions = Counter(v for c in e.contradictions() for v in c["classes"])
        out: dict[str, Json] = {}
        for v in snapshot.config.project.versions:
            held = [n for n, functions in _code_modules(snapshot).items() if lay.held(functions) & {v}]
            module_cuts: Counter[str] = Counter()
            for n in held:
                module_cuts[records[n]["start"]] += 1
            pairs = e.census(v)
            out[v] = {
                "bytes": totals[v]["bytes"],
                "total": totals[v]["total"],
                "modules": len(held),
                "perfect_modules": sum(perfect[n] for n in held),
                "boundary": sum(records[n]["boundary"] for n in held),
                "flags": sum(records[n]["flags"] in ("default", "proven") for n in held),
                "members": sum(members[n] for n in held),
                "named": sum(records[n]["name"] != "assigned" for n in held),
                "module_starts": dict(sorted(module_cuts.items())),
                "open_edges": pairs["open"],
                "contradictions": contradictions[v],
                "objects_estimate": pairs["objects_estimate"],
            }
        return out, frozenset(n for n, ok in perfect.items() if ok)
