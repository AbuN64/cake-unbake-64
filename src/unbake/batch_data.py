"""Atomic publication of independent extracted data, with one snapshot and pool dispatch."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from unbake import (
    compare,
    effort,
    journal,
    layout,
    lift,
    native,
    policy,
    pool,
    publish,
    recipes,
    repo,
    store,
    types,
    view,
)
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Receipt, Refusal, Snapshot, UnitSpec, digest


def _manifest(path: Path) -> list[dict[str, str]]:
    try:
        rows = json.loads(path.read_text())
        if (
            not isinstance(rows, list)
            or not rows
            or any(
                not isinstance(r, dict)
                or set(r) != {"member", "file"}
                or any(not isinstance(v, str) or not v for v in r.values())
                for r in rows
            )
            or len({r["member"] for r in rows}) != len(rows)
        ):
            raise ValueError("expected a nonempty list of unique {member, file} records")
        return [{"member": r["member"], "file": str((path.parent / r["file"]).resolve())} for r in rows]
    except (OSError, ValueError) as error:
        raise Refusal(Finding("land.request", f"Invalid data manifest: {error}", path=str(path))) from error


def _prepare(snapshot: Snapshot, rows: list[dict[str, str]]) -> tuple[Snapshot, tuple[UnitSpec, ...]]:
    """Retype all rows and serialize their combined map once, without touching the working tree."""
    units, groups = dict(snapshot.layout.units), dict(snapshot.layout.groups)
    sections = configuration.sections()
    raw = frozenset(t for s in sections.values() for t in s["types"] if not t.startswith("."))
    writes: dict[str, Any] = {}
    added = []
    # Decode each split once; every member below uses this same baseline.
    split_rows = {
        v: {n: state for n, state, _ in layout.split_states(snapshot, version)}
        for v, version in snapshot.versions.items()
    }
    for row in rows:
        member = snapshot.layout.members.get(row["member"])
        if member is None or member.kind == "function" or layout.unit_of(snapshot, row["member"]) is not None:
            raise Refusal(
                Finding("land.request", "Batch data requires an unpublished data member.", unit=row["member"])
            )
        source = Path(row["file"]).read_bytes()
        text, declared = lift.lift(snapshot, member, source.decode())
        # the address words become named symbols, each undeclared one declared once at the top
        source = ("".join(f"{d}\n" for d in declared.values()) + ("\n" if declared else "") + text).encode()
        for holder in member.holders():
            version = snapshot.versions[holder]
            state = split_rows[holder][member.name]
            if state not in raw:
                raise Refusal(Finding("layout.member", "Batch member is not extracted data.", unit=member.name))
            section = next(name for name, s in sections.items() if state in s["types"])
            writes[version.split] = layout.edit_split(
                writes.get(version.split, snapshot.read(version.split)), member.name, state=f".{section}", types=raw
            )
        group, new = (member.group, None) if member.group else layout.data_group(snapshot, member.name)
        if new:
            groups[group] = new
        stem = member.name.removesuffix(".c").removeprefix("src/").replace("/", ".")
        if len(stem.encode()) > 180:
            stem = "data_" + digest(member.name)[:24]
        unit = UnitSpec(
            f"src/{stem}.c", "data", group, (member.name,), snapshot.config.project.toolchain, {"add": [], "omit": []}
        )
        if unit.path in units:
            raise Refusal(Finding("land.request", "Batch source path collides with an existing unit.", unit=unit.path))
        units[unit.path], writes[unit.path] = unit, source
        added.append(unit)
    writes["layout.toml"] = layout.dump_map(replace(snapshot.layout, units=units, groups=groups))
    return layout.overlay(snapshot, writes), tuple(added)


def _prove(item: tuple[Any, ...]) -> tuple[Any, ...]:
    snapshot, unit, version, recipe, work, baseline = item
    text = snapshot.read(unit.path).decode()
    sdk = snapshot.layout.groups[unit.group].sdk
    source_view = view.get(snapshot, unit, version, recipe)
    publish.new_findings(baseline, unit.path, policy.evaluate(snapshot, unit.path, text, source_view, sdk), sdk)
    proofs = native.prove_args(item[:5])
    publish.refuse_blocking(
        publish.view_findings(
            unit, text, baseline.peek(unit.path), native.warnings(snapshot, unit, version, recipe, work / "warnings")
        )
    )
    return proofs


def run(config: Config, params: Json) -> Json:
    if params["files"] or params["function"]:
        raise Refusal(Finding("land.request", "--batch-data cannot be combined with files or --function."))
    rows = _manifest(Path(params["batch_data"]).resolve())
    with ExitStack() as held:
        held.enter_context(effort.stage("publish.batch_data"))
        if not held.enter_context(store.exclusive(config, "land", wait=False)):
            raise Refusal(Finding("layout.busy", "A landing is running; retry the data batch when idle."))
        held.enter_context(store.shared(config, "structure"))  # the drain's order: land, then structure
        snapshot = layout.capture(config)
        proposed, units = _prepare(snapshot, rows)
        # Scope policy against the actual baseline, before any compiler work.
        before: list[Any] = []
        after: list[Any] = []
        for unit in units:
            sdk = proposed.layout.groups[unit.group].sdk
            old = snapshot.peek(unit.path)
            if old is not None:
                before.extend(policy.evaluate(snapshot, unit.path, old.decode(), None, sdk))
            after.extend(policy.evaluate(proposed, unit.path, proposed.read(unit.path).decode(), None, sdk))
        blocking, debt = policy.scope(before, after, proposed.overlays)
        publish.refuse_blocking(blocking)
        with store.work(config) as work:
            jobs = [
                (proposed, u, v, recipes.resolve(config, u, u.options), work / str(i) / v, snapshot)
                for i, u in enumerate(units)
                for v in compare.holders(proposed, u)
            ]
            batches = pool.map(config, "publish.batch_data.prove", _prove, jobs)
        proofs = tuple(p for batch in batches for p in batch)
        if params["withhold"]:
            changed = []
            for unit in units:
                exact = {
                    v
                    for v in compare.holders(proposed, unit)
                    if any(p.unit == unit.path and p.version == v for p in proofs)
                    and all(p.exact for p in proofs if p.unit == unit.path and p.version == v)
                }
                if not exact:
                    publish.refuse_blocking(compare.gaps(proposed, unit, proofs))
                    raise Refusal(Finding("land.request", "No exact version of a batch member.", unit=unit.path))
                changed.append(replace(unit, withheld=tuple(sorted(set(compare.holders(proposed, unit)) - exact))))
            units = tuple(changed)
            mapped = {**proposed.layout.units, **{u.path: u for u in units}}
            proposed = layout.overlay(
                proposed, {"layout.toml": layout.dump_map(replace(proposed.layout, units=mapped))}
            )
        gaps = tuple(
            replace(f, versions=left)
            for u in units
            for f in compare.gaps(proposed, u, proofs)
            if (left := tuple(v for v in f.versions if v not in u.withheld))
        )
        publish.refuse_blocking(gaps)
        proofs = tuple(p for p in proofs if p.version not in proposed.layout.units[p.unit].withheld)
        if not proofs or not all(p.exact for p in proofs):
            raise Refusal(Finding("land.request", "The batch did not produce complete exact proofs."))
        writes = dict(proposed.overlays)
        scanned = types.scan(proposed)
        if scanned != snapshot.peek("types.toml"):
            writes["types.toml"] = scanned
        rendered = repo.files(layout.overlay(snapshot, writes))
        writes.update({p: b for p, b in rendered.items() if b != snapshot.peek(p)})
        plan = publish.make_plan(
            "publish", snapshot, writes, tuple(u.path for u in units), debt, f"publish {len(units)} data members"
        )
        publish.credit_guard(snapshot, plan.writes)
        files = journal.apply(config, plan, snapshot)
        landed, submissions = [], []
        for row in rows:
            identity = digest((plan.digest, row["member"]))
            body = {
                "id": identity,
                "member": row["member"],
                "operation": "publish",
                "files": list(files),
                "message": plan.message,
            }
            receipt = Receipt(
                "publish",
                files,
                plan.message,
                plan.digest,
                tuple(p for p in proofs if p.member == row["member"]),
                len(debt),
                effort.invocation(),
            )
            store.log(config, "receipt", {**body, **asdict(receipt)})
            landed.append(body)
            submissions.append(
                {
                    "file": row["file"],
                    "files": [row["file"]],
                    "member": row["member"],
                    "id": identity,
                    "operation": "publish",
                    "exact": True,
                    "score": 1.0,
                    "findings": [],
                    "consumers": {"count": 0, "wall": 0.0},
                }
            )
        return {"submissions": submissions, "drain": {"running": False, "landed": landed, "refused": []}}
