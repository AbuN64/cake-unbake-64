"""Admission, submission landing and withdrawal."""

import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from unbake import (
    compare,
    effort,
    headers,
    journal,
    layout,
    moved,
    native,
    ownership,
    policy,
    pool,
    present,
    recipes,
    repo,
    report,
    rescore,
    store,
    symptoms,
    types,
    versions,
    view,
)
from unbake import config as configuration
from unbake.contracts import (
    Config,
    Finding,
    Json,
    Plan,
    Proof,
    Receipt,
    Refusal,
    Snapshot,
    Submission,
    UnitSpec,
    digest,
)


def _objects(item: tuple[Any, ...]) -> Path:
    return native.objects(*item)[0]


def _measure(item: tuple[Any, ...]) -> tuple[Proof, ...]:
    return native.measure(*item)


def view_findings(unit: UnitSpec, text: str, old: bytes | None, warnings: Sequence[str]) -> tuple[Finding, ...]:
    """The refused warnings whose line is new or changed text of the candidate; its landed debt never blocks."""
    lines, before, found = text.splitlines(), set((old or b"").decode().splitlines()), []
    for warning in warnings:
        at = re.search(r"([^\s:,]+)[:,] ?(?:line )?(\d+)\b", warning)
        if (
            at
            and Path(at[1]).name == Path(unit.path).name
            and 0 < int(at[2]) <= len(lines)
            and lines[int(at[2]) - 1] not in before
        ):
            found.append(
                Finding(
                    "land.view-conversion",
                    warning,
                    path=unit.path,
                    line=int(at[2]),
                    action="type the variable as the pointer it holds",
                )
            )
    return tuple(found)


def refuse_blocking(findings: Sequence[Finding]) -> None:
    findings = tuple(f for f in findings if f.blocking)
    if findings:
        raise Refusal(*findings)


def new_findings(snapshot: Snapshot, path: str, findings: Sequence[Finding], sdk: bool) -> None:
    refuse_blocking(policy.fresh(snapshot, path, findings, sdk))


def make_plan(
    operation: str,
    snapshot: Snapshot,
    writes: Mapping[str, bytes | None],
    affected: tuple[str, ...],
    debt: tuple[Finding, ...],
    message: str,
) -> Plan:
    writes = {**writes, **progress_block(snapshot, writes)}
    body = (operation, snapshot.digest, writes, affected, (), debt, message)
    return Plan(*body, digest(body))


def _credits(path: str) -> bool:
    """Whether a write can move the credit a report shows: the map (units, fuzzy rows, splits) or a source's text,
    which decides its unclean findings."""
    return path == "layout.toml" or path.endswith((".c", ".h"))


def progress_block(snapshot: Snapshot, writes: Mapping[str, bytes | None]) -> dict[str, bytes]:
    """The README progress block and the presentation files (progress.json, memory maps, symbol lines) of the tree a
    plan leaves, from the report of that tree. Every plan that can move credit carries them (land, fuzzy, move,
    repair, generate), so what a lane commits with its change always equals a fresh report: no separate step to
    remember. {} when the plan writes the README itself, writes nothing that credits or the project has no README."""
    if "README.md" in writes or not any(map(_credits, writes)) or snapshot.peek("README.md") is None:
        return {}
    after = layout.overlay(snapshot, writes)
    text = after.read("README.md")  # read through the snapshot: the journal refuses a README changed under the plan
    current = report.progress(after)
    made = {"README.md": report.readme(text.decode(), after, current).encode(), **present.files(after, current)}
    return {path: data for path, data in made.items() if data != after.peek(path)}


def _stamp_key(job: tuple[Any, ...]) -> str:
    """A proof is named by what it reads, so one already made never travels to a worker."""
    snapshot, unit, version, recipe, _ = job
    return native.stamp(snapshot, unit, version, recipe)


_PROOF_CHUNK = 256  # units derived and proved in the parent at once: each derived unit holds its own layout (~5 MB)


def _proofs(snapshot: Snapshot, affected: Sequence[str]) -> tuple[tuple[Proof, ...], tuple[Finding, ...]]:
    """A withheld version is not built, so it is neither proved nor a gap. Up to a chunk of units derive in one
    dispatch and prove in one more. More units go to one dispatch of one job per unit that derives, proves and judges
    it where it runs: the derived tree state (~5 MB) never reaches the parent, and the pool stays full to the end
    instead of draining at two barriers per chunk."""
    units = [snapshot.layout.units[path] for path in affected]
    settle(_read_versions(snapshot, [(u, v) for u in units for v in compare.holders(snapshot, u)]))
    with store.work(snapshot.config) as work:
        if len(units) > _PROOF_CHUNK:
            done = pool.map(snapshot.config, "publish.units", _prove_unit, [(snapshot, u, work) for u in units])
            return tuple(p for made, _ in done for p in made), tuple(g for _, gaps in done for g in gaps)
        jobs: list[Any] = []
        try:
            derived = ownership.derive_many(snapshot, units)
        except Refusal:  # one unit's derive refused: each unit derives alone, so the refusal is that unit's gap
            done = [_prove_unit((snapshot, u, work)) for u in units]
            return tuple(p for made, _ in done for p in made), tuple(g for _, gaps in done for g in gaps)
        for base, (own, unit) in zip(units, derived, strict=True):
            recipe = recipes.resolve(snapshot.config, unit, unit.options)
            for version in sorted(set(compare.holders(own, unit)) - set(base.withheld)):
                jobs.append((own, unit, version, recipe, work / f"{len(jobs)}"))
        batches = pool.map(snapshot.config, "publish.prove", native.prove_args, jobs, _stamp_key)
        made = tuple(p for batch in batches for p in batch)
        return made, tuple(g for b, (o, u) in zip(units, derived, strict=True) for g in _judged(o, u, made, b.withheld))


def _judged(own: Snapshot, unit: UnitSpec, made: Sequence[Proof], withheld: Collection[str]) -> list[Finding]:
    """The unit's gaps in the versions it is not withheld from."""
    skip = set(withheld)
    return [
        replace(f, versions=left)
        for f in compare.gaps(own, unit, made)
        if (left := tuple(v for v in f.versions if v not in skip))
    ]


def _prove_unit(item: tuple[Snapshot, UnitSpec, Path]) -> tuple[tuple[Proof, ...], list[Finding]]:
    """One unit derived, proved in each version holding it and judged, in one job: its versions run in turn here (the
    pool is busy with the other units), and the derived tree state is not kept past the job."""
    snapshot, base, work = item
    try:
        own, unit = ownership.derive_many(snapshot, [base], keep=False)[0]
    except Refusal as refusal:
        return (), _refused(snapshot, base, refusal)
    recipe = recipes.resolve(snapshot.config, unit, unit.options)
    versions = sorted(set(compare.holders(own, unit)) - set(base.withheld))
    stem = re.sub(r"\W", "_", base.path)
    jobs = [(own, unit, v, recipe, work / f"{stem}.{v}") for v in versions]
    batches = pool.map(snapshot.config, "publish.prove", native.prove_args, jobs, _stamp_key)
    made = tuple(p for batch in batches for p in batch)
    return made, _judged(own, unit, made, base.withheld)


def _refused(snapshot: Snapshot, base: UnitSpec, refusal: Refusal) -> list[Finding]:
    """A derive that refuses (a row it cannot write, a claim outside the rows) is the unit's gap in every version
    holding it, named by its first member as a proof gap is: the caller keeps or refuses that unit alone, never the
    whole proof. What the refusal named stays in the gap's symptoms."""
    versions = tuple(sorted(set(compare.holders(snapshot, base)) - set(base.withheld)))
    member = base.members[0] if base.members else base.path
    return [
        replace(
            f,
            unit=member,
            versions=f.versions or versions,
            missing=f.missing or (f.reason,),
            symptoms={**f.symptoms, "refused": f.unit},
        )
        for f in refusal.findings
    ]


def _regressions(
    snapshot: Snapshot, gaps: tuple[Finding, ...], own: Collection[str], candidate: str
) -> tuple[Finding, ...]:
    """Another member's gap blocks only where HEAD proves it exact, named as the candidate breaking a landed file."""
    heads = {g.unit: layout.unit_of(snapshot, g.unit) for g in gaps if g.unit not in own}
    paths = sorted({u.path for u in heads.values() if u is not None})
    exact = {(p.member, p.version) for p in _proofs(snapshot, paths)[0] if p.exact} if paths else set()
    kept = []
    for g in gaps:
        head = heads.get(g.unit)
        if g.unit in own or head is None:
            kept.append(g)
        elif any((g.unit, v) in exact for v in g.versions):
            kept.append(
                Finding(
                    "land.breaks_dependent",
                    f"{candidate} breaks {head.path}: {g.missing[0].splitlines()[0]}",
                    path=head.path,
                    unit=g.unit,
                    versions=g.versions,
                    missing=g.missing,
                    symptoms=g.symptoms,
                    action=f"make the declaration agree with {head.path}, or change both",
                )
            )
    return tuple(kept)


def _blocking(findings: Sequence[Finding]) -> tuple[Finding, ...]:
    return tuple(f for f in findings if f.blocking)


def admit(snapshot: Snapshot, request: Json, partial: bool = False) -> tuple[UnitSpec, Snapshot, tuple[Proof, ...]]:
    """Every check runs that the ones before it leave runnable, and one refusal names every finding they make, so
    one submit lists all a candidate must fix: source rules, unresolved symbols, view warnings and inexact bytes."""
    with effort.stage("publish.admit"):
        found: list[Finding] = []  # in order, each finding once (the view rules repeat the text rules)

        def add(findings: Sequence[Finding]) -> None:
            found.extend(f for f in findings if f not in found)

        def refuse() -> None:
            if found:
                raise Refusal(*found)

        with effort.stage("publish.admit.1"):
            file = Path(request["file"])
            file = file if file.is_absolute() else snapshot.config.project.root / file
            rewritten: list[str] = []
            unit, snap = compare.bind(snapshot, file, request["function"], rewritten)
            if rewritten:  # a move landed after the candidate was written: say what was respelled
                moved.note(snapshot.config, rewritten, str(request["file"]))
            for member in unit.members:
                owner = layout.unit_of(snapshot, member)
                if owner is not None and owner.path != unit.path and snapshot.layout.members[member].state == unit.kind:
                    raise Refusal(Finding("land.request", f"Already published in {owner.path}.", unit=member))
            text = snap.read(unit.path).decode()
            group = snap.layout.groups[unit.group]
        with effort.stage("publish.admit.3"):
            fresh = policy.fresh(
                snapshot, unit.path, policy.evaluate(snap, unit.path, text, None, group.sdk), group.sdk
            )
            add(_blocking(fresh))
        with effort.stage("publish.admit.4"):
            recipe = recipes.resolve(snap.config, unit, request["overrides"])
            unit = replace(
                unit,
                toolchain=recipe.toolchain,
                options={k: v for k, v in request["overrides"].items() if k != "toolchain"},
            )
            try:
                snap, unit = ownership.derive(snap, unit)  # the data its source emits is proved with its code
                holders = compare.holders(snap, unit)
                views = view.all_versions(snap, unit, recipe, holders)
            except Refusal as refusal:
                add(refusal.findings)
                raise Refusal(*found) from None
        with effort.stage("publish.admit.5"):
            evaluated = policy.evaluate(snap, unit.path, text, views[holders[0]], group.sdk)
            add(_blocking(policy.fresh(snapshot, unit.path, evaluated, group.sdk)))
        with store.work(snap.config) as work:
            with effort.stage("publish.admit.6"):
                jobs = [(snap, unit, v, recipe, work / f"{i}") for i, v in enumerate(holders)]
                try:
                    objects = pool.map(snap.config, "publish.objects", _objects, jobs)
                except Refusal as refusal:
                    add(refusal.findings)
                    raise Refusal(*found) from None
            with effort.stage("publish.admit.7"):
                unresolved = tuple(
                    f
                    for v, obj in zip(holders, objects, strict=True)
                    for f in versions.resolve(snap.versions[v], versions.undefined(obj), unit.path)
                )
                add(unresolved)
                warned = view_findings(
                    unit,
                    text,
                    snapshot.peek(unit.path),
                    native.warnings(snap, unit, holders[0], recipe, work / "warnings"),
                )
                add(_blocking(warned))
                if unresolved:  # what does not link measures nothing more
                    refuse()
            with effort.stage("publish.admit.8"):
                measure_jobs = [
                    (snap, unit, v, recipe, obj, work / f"{i}")
                    for i, (v, obj) in enumerate(zip(holders, objects, strict=True))
                ]
                proofs = tuple(
                    p for batch in pool.map(snap.config, "publish.measure", _measure, measure_jobs) for p in batch
                )
                gaps = compare.gaps(snap, unit, proofs)
                bad = {v for g in gaps for v in g.versions}
                if gaps and (not partial or bad >= set(holders)):
                    add(gaps)
                refuse()
                if tuple(sorted(bad)) != unit.withheld:  # a partial landing withholds exactly the inexact versions
                    unit = replace(unit, withheld=tuple(sorted(bad)))
        return unit, snap, proofs


def _append(old: bytes, folded: bytes | None) -> bytes:
    assert folded is not None  # the source being appended was folded
    existing, incoming = old.decode().splitlines(), folded.decode().splitlines()
    includes = [line for line in incoming if line.lstrip().startswith("#include") and line not in existing]
    position = max((i + 1 for i, line in enumerate(existing) if line.lstrip().startswith("#include")), default=0)
    existing[position:position] = list(dict.fromkeys(includes))
    remaining = [line for line in incoming if not line.lstrip().startswith("#include")]
    return ("\n".join(existing).rstrip() + "\n\n" + "\n".join(remaining).strip() + "\n").encode()


def plans(snapshot: Snapshot, unit: UnitSpec, proposed: Snapshot, extras: Mapping[str, bytes]) -> Iterator[Plan]:
    """The publication layout options, each planned only when the caller asks for it: the first option that proves
    lands, so the others cost nothing. `extras` are a change set's other files, written in every plan."""
    with effort.stage("publish.plans"):
        recipe = recipes.resolve(proposed.config, unit, unit.options)
        first = compare.holders(proposed, unit)[0]
        owner = layout.unit_of(snapshot, unit.members[0])
        state = snapshot.layout.members[unit.members[0]].state
        expanded = (
            owner is not None
            and owner.path == unit.path
            and any(proposed.layout.members[n].kind == "function" and n not in owner.members for n in unit.members)
        )
        again = owner is not None and owner.path == unit.path and (state == unit.kind or state.startswith("."))
        folded: dict[str, bytes | None]
        conflicts: tuple[Finding, ...]
        if again:  # its declarations were folded when it first landed: a new text keeps them where they are
            folded, conflicts = {unit.path: proposed.read(unit.path)}, ()
        else:
            folded, conflicts = headers.fold(proposed, unit, view.get(proposed, unit, first, recipe))
        if conflicts:
            raise Refusal(*conflicts)
        source = folded[unit.path]
        assert source is not None  # the unit's own text is always written
        claims = {p: b for p, b in proposed.overlays.items() if p.endswith(".yaml") and b != snapshot.peek(p)}
        if expanded:
            options = [(unit, claims)]
        elif owner is not None and again:
            options = [(owner, {})]
        else:  # whichever unit takes the source, it owns the data rows its compile claimed
            options = [
                (
                    replace(option, members=(*option.members, *(m for m in unit.members if m not in option.members))),
                    {**writes, **claims},
                )
                for option, writes in layout.unit_options(snapshot, unit.members[0], source)
            ]
    return _option_plans(snapshot, unit, proposed, (folded, again, extras), options)


def _option_plans(
    snapshot: Snapshot,
    unit: UnitSpec,
    proposed: Snapshot,
    shared: tuple[Any, ...],
    options: Sequence[tuple[UnitSpec, dict[str, bytes | None]]],
) -> Iterator[Plan]:
    produced = False
    last: tuple[Finding, ...] = ()
    for option, writes in options:
        try:
            with effort.stage("publish.option"):
                made = _option_plan(snapshot, unit, proposed, shared, option, writes)
        except Refusal as error:  # one option failing outright leaves the others to try
            last = error.findings
            continue
        if isinstance(made, Plan):
            produced = True
            yield made
        else:
            last = made
    if not produced:
        raise Refusal(*last or (Finding("land.request", "No publication layout option is available.", path=unit.path),))


def _clash_names(clashes: Sequence[Finding]) -> dict[tuple[str, str], set[str]]:
    """Per (rule, name) of header clashes the spellings they name."""
    named: dict[tuple[str, str], set[str]] = {}
    for f in clashes:
        named.setdefault((f.key, f.reason.partition(":")[0]), set()).update(re.findall(r"'([^']*)'", f.reason))
    return named


def _new_clashes(
    snapshot: Snapshot, written: Mapping[str, bytes | None], clashes: Sequence[Finding]
) -> tuple[Finding, ...]:
    """The header clashes the candidate adds: one its base tree already has (same rule, name and spellings, judged
    once per tree state over the same headers) is landed debt and never refuses; a new name or spelling always does."""
    if not clashes:
        return ()
    paths = tuple(sorted(p for p, b in written.items() if b is not None and snapshot.peek(p) is not None))
    base = effort.memo(
        ("publish.base_clashes", snapshot.digest, paths),
        lambda: _clash_names(headers.reconcile(snapshot, snapshot, {p: snapshot.read(p) for p in paths})[1]),
    )
    return tuple(f for f in clashes if not all(s <= base.get(k, set()) for k, s in _clash_names([f]).items()))


def _option_plan(
    snapshot: Snapshot,
    unit: UnitSpec,
    proposed: Snapshot,
    shared: tuple[Any, ...],
    option: UnitSpec,
    writes: dict[str, bytes | None],
) -> Plan | tuple[Finding, ...]:
    folded, again, extras = shared
    writes = {**extras, **writes}
    writes[option.path] = (
        _append(snapshot.read(option.path), folded[unit.path])
        if option.path in snapshot.layout.units and not again
        else folded[unit.path]
    )
    writes.update((p, b) for p, b in folded.items() if p != unit.path)
    with effort.stage("publish.reconcile"):
        written = {p: b for p, b in writes.items() if p.endswith(".h")}
        # a name is declared once: a copy in a written header goes to its owner
        writes, reconciled = headers.reconcile(snapshot, _overlay(snapshot, written), writes)
        clashes = _new_clashes(snapshot, written, reconciled)
    if clashes:
        return clashes
    # Claims rename split rows and retire their old data providers together. Read the option's layout before
    # applying those split edits, so there is never a snapshot with a provider pointing at a vanished row.
    splits = {v.split for v in snapshot.versions.values()}
    overlay = _overlay(snapshot, {p: b for p, b in writes.items() if p not in splits})
    configured = replace(option, toolchain=unit.toolchain, options=unit.options, withheld=unit.withheld)
    gone = {
        k
        for k in snapshot.layout.units.keys() - proposed.layout.units.keys()  # data it now owns
        if layout.is_datum(snapshot.layout.units[k]) or all(map(versions.unowned, snapshot.layout.units[k].members))
    }
    units = {k: u for k, u in overlay.layout.units.items() if k not in gone}
    units[option.path] = configured
    writes.update(dict.fromkeys(gone))
    code = {n for n in unit.members if proposed.layout.members[n].kind == "function"}
    for member in code & snapshot.layout.fuzzy.keys():
        writes[snapshot.layout.fuzzy[member]["path"]] = None
    fuzzy = {k: v for k, v in overlay.layout.fuzzy.items() if k not in code}
    added = [n for n in unit.members if n in code and (layout.unit_of(snapshot, n) is None)]
    member = ", ".join(added) if added else unit.members[0]
    writes["layout.toml"] = layout.dump_map(replace(overlay.layout, units=units, fuzzy=fuzzy))
    overlay = _overlay(snapshot, writes)  # the type map and generated files follow once per drain (generate)
    group = proposed.layout.groups[unit.group]
    affected = {option.path, *extras.keys() & overlay.layout.units.keys()}
    edited = [p for p, data in writes.items() if p.endswith(".h") and data != snapshot.peek(p)]
    with effort.stage("publish.consumers"):
        found = headers.consumers(overlay, edited)
        known = [p for p in found if p in snapshot.layout.units]
        affected.update(p for p in found if p not in snapshot.layout.units)
        dead = headers.dangling(snapshot, overlay, edited)  # a header naming a dropped declaration breaks its includers
        if dead:
            affected.update(p for p in headers.consumers(overlay, dead) if p in snapshot.layout.units)
        known = list(headers.reaching(snapshot, overlay, edited, known))  # the others read what they read before
        jobs = [(overlay, snapshot, p) for p in known]
        affected.update(
            p
            for p, moved in zip(
                known, pool.map(snapshot.config, "publish.moved", _moved, jobs, _moved_key), strict=True
            )
            if moved
        )
    before: list[Any] = []
    after: list[Any] = []
    with effort.stage("publish.policy"):
        _evaluate(snapshot, overlay, group, writes, before, after)
    blocking, debt = policy.scope(before, after, writes)
    if blocking:
        return blocking
    kept = sorted(set(compare.holders(proposed, unit)) - set(unit.withheld))
    where = f"{', '.join(kept)} exact; {', '.join(unit.withheld)} withheld" if unit.withheld else unit.path
    with effort.stage("publish.plan"):
        return make_plan("publish", snapshot, writes, tuple(sorted(affected)), debt, f"publish {member} ({where})")


def _overlay(snapshot: Snapshot, writes: Mapping[str, bytes | None]) -> Snapshot:
    with effort.stage("layout.overlay"):
        return layout.overlay(snapshot, dict(writes))


def _evaluate(
    snapshot: Snapshot,
    overlay: Snapshot,
    group: Any,
    writes: Mapping[str, bytes | None],
    before: list[Any],
    after: list[Any],
) -> None:
    """The policy findings of every written source, before and after the write."""
    for path, data in writes.items():
        if Path(path).suffix not in (".c", ".h"):
            continue
        owner = overlay.layout.units.get(path)
        sdk = overlay.layout.groups[owner.group].sdk if owner else group.sdk
        old = snapshot.peek(path)
        if old is not None:
            before.extend(policy.evaluate(snapshot, path, old.decode(), None, sdk))
        if data is not None:
            after.extend(policy.evaluate(overlay, path, data.decode(), None, sdk))


def _reads(job: tuple[Snapshot, Snapshot, str]) -> list[list[tuple[Any, ...]]]:
    """What a proof of each version of the consumer reads, in the overlay and in the snapshot."""
    overlay, snapshot, path = job
    sides = []
    for side in (overlay, snapshot):
        unit = side.layout.units[path]
        recipe = recipes.resolve(side.config, unit, unit.options)
        sides.append([view.reads(side, unit, v, recipe) for v in compare.holders(overlay, overlay.layout.units[path])])
    return sides


def _moved(job: tuple[Snapshot, Snapshot, str]) -> bool:
    new, old = _reads(job)
    return new != old


def _moved_key(job: tuple[Snapshot, Snapshot, str]) -> str:
    """A consumer's answer is named by the files it reads on both sides, so an entry that left them alone reuses it."""
    overlay, snapshot, path = job
    return digest(
        [
            [
                (side.layout.units[path].options, view.closure(side, side.layout.units[path], v))
                for v in compare.holders(overlay, overlay.layout.units[path])
            ]
            for side in (overlay, snapshot)
        ]
    )


def _changed(snapshot: Snapshot, extras: Mapping[str, bytes]) -> Snapshot:
    """The snapshot a change set lands on: its files in place, and the type map read from them, so a definition the
    set rewrites is the type its callers are judged against."""
    if not extras:
        return snapshot
    edited = layout.overlay(snapshot, dict(extras))
    scanned = types.scan(edited)
    return edited if scanned == edited.peek("types.toml") else layout.overlay(edited, {"types.toml": scanned})


_stale: dict[str, Config] = {}  # versions whose split rows a landing changed: extracted once per drain
_changed_names: dict[str, set[str]] = {}  # per stale version, the members its landings turned from asm into C


def _read_versions(snapshot: Snapshot, jobs: Sequence[tuple[UnitSpec, str]]) -> set[str]:
    """The stale versions a proof can tell from their settled self. A proof reads a version's generated facts by the
    names its source mentions, and a landing changes only the rows of the members it moved to C, so a stale version is
    extracted before a proof only when one of those members is named by a unit proved in it. The drain end settles
    the rest."""
    words: dict[str, set[str]] = {}
    out = set()
    for unit, version in jobs:
        if version in _stale and version not in out:
            if version not in _changed_names:
                out.add(version)
                continue
            if unit.path not in words:
                words[unit.path] = set(re.findall(r"[A-Za-z_]\w*", snapshot.read(unit.path).decode(errors="replace")))
            if _changed_names[version] & words[unit.path]:
                out.add(version)
    return out


def settle(only: Collection[str] | None = None) -> None:
    """Extracts each stale version once, however many entries changed it: those a proof is about to read, or all."""
    jobs = [(config, v) for v, config in sorted(_stale.items()) if only is None or v in only]
    if jobs:
        for _, v in jobs:
            del _stale[v]
            _changed_names.pop(v, None)
        pool.map(jobs[0][0], "repo.extract", repo.extract_job, jobs)
        effort.forget("generated", "versions.read", "pin", "includes", "closure", "asm")


def accept(
    snapshot: Snapshot, request: Json, extras: Mapping[str, bytes], partial: bool
) -> tuple[Plan, tuple[Proof, ...], tuple[str, ...]]:
    """The one acceptance check of a publish: submit runs it at admission, land again when HEAD moved. It returns the
    first layout plan whose proofs hold, or raises the Refusal of the last."""
    member = request["function"]
    unit = layout.unit_of(snapshot, member)
    file = Path(request["file"])
    if (
        unit is not None
        and configuration.kinds()[unit.kind]["decompiled"]
        and snapshot.read(unit.path) == file.read_bytes()
        and not extras
    ):
        raise Refusal(Finding("land.duplicate", f"{member} is already landed in {unit.path}", unit=member))
    unit, proposed, _ = admit(_changed(snapshot, extras), request, partial)
    touched = (
        *unit.members,
        *(m for p in extras if p in snapshot.layout.units for m in snapshot.layout.units[p].members),
    )
    gaps: tuple[Finding, ...] = ()
    for plan in plans(snapshot, unit, proposed, extras):
        proofs, gaps = _proofs(layout.overlay(snapshot, plan.writes), plan.affected)
        gaps = _regressions(snapshot, gaps, unit.members, unit.path)  # the change set's other units: no regression
        if not gaps:
            return plan, proofs, touched
    raise Refusal(*gaps)


def land(config: Config, submission: Submission) -> Receipt:
    with effort.stage("publish.land"):
        snapshot = layout.capture(config)
        if submission.operation != "publish":
            raise Refusal(Finding("land.request", "Unknown submission operation.", unit=submission.member))
        extras = {path: (config.project.root / copy).read_bytes() for path, copy in submission.extras.items()}
        request = {
            "file": str(config.project.root / submission.source),
            "function": submission.function or submission.member,
            "overrides": submission.overrides,
            "note": submission.note,
        }
        plan, proofs, touched = accept(snapshot, request, extras, bool(submission.withheld))
        changed = [
            (config, v.id)
            for v in snapshot.versions.values()
            if v.split in plan.writes and plan.writes[v.split] != snapshot.peek(v.split)
        ]
        files = journal.apply(config, plan, snapshot)
        _stale.update({v: config for _, v in changed})
        for _, v in changed:
            _changed_names.setdefault(v, set()).update(touched)
        return Receipt("publish", files, plan.message, plan.digest, proofs, len(plan.debt), effort.invocation())


_GENERATED = ".unbake/generated.stamp"  # what the last generate derived from: the same inputs derive nothing new


def _inputs(snapshot: Snapshot) -> str:
    """Everything the generated files derive from: the snapshot (layout, versions, code) and the text of every
    header, source and the type map, which a live tree's snapshot digest does not cover."""
    paths = [*headers.sources(snapshot), *headers.sources(snapshot, "src", ".c"), "types.toml", "layout.toml"]
    return digest((snapshot.digest, [(p, snapshot.peek(p)) for p in paths]))


def generate(config: Config) -> tuple[str, ...]:
    """The type map and the generated files, derived once from what a drain landed (never per entry). A drain over a
    tree whose inputs are those the last generate derived from (one that landed nothing) derives nothing again."""
    with effort.stage("publish.generate"):
        snapshot = layout.capture(config)
        stamp = config.project.root / _GENERATED
        if stamp.is_file() and stamp.read_text() == _inputs(snapshot):
            return ()
        writes: dict[str, bytes | None] = dict(rescore.writes(snapshot))  # moved fuzzy rows scored once, saved
        scanned = types.scan(snapshot)
        if scanned != snapshot.peek("types.toml"):
            writes["types.toml"] = scanned
        writes.update({p: d for p, d in repo.files(_overlay(snapshot, writes)).items() if d != snapshot.peek(p)})
        files = (
            journal.apply(config, make_plan("generate", snapshot, writes, (), (), "generated files"), snapshot)
            if writes
            else ()
        )
        store.write(stamp, _inputs(layout.capture(config) if writes else snapshot).encode())
        return files


def _fuzzy(config: Config, snapshot: Snapshot, submission: Any) -> tuple[str, bytes, dict[str, float], dict[str, str]]:
    """One fuzzy candidate measured against the snapshot: its source path, text and per-version scores, or a refusal."""
    member, inbox = submission.member, config.project.root / submission.source
    rewritten: list[str] = []
    unit, bound = compare.bind(snapshot, inbox, member, rewritten)
    if rewritten:
        moved.note(config, rewritten, submission.source)
    group = bound.layout.groups[unit.group]
    text = bound.read(unit.path).decode()
    new_findings(snapshot, unit.path, policy.evaluate(bound, unit.path, text, None, group.sdk), group.sdk)
    holders = compare.holders(bound, unit)
    views = view.all_versions(bound, unit, recipes.resolve(config, unit, submission.overrides), holders)
    new_findings(snapshot, unit.path, policy.evaluate(bound, unit.path, text, views[holders[0]], group.sdk), group.sdk)
    proofs = compare.measure(bound, unit, submission.overrides, None)
    failed = tuple(
        p for p in proofs if any(p.symptoms.get(k) for k in ("compile_failed", "unresolved", "no_measurement"))
    )
    if failed:
        raise Refusal(
            Finding(
                "land.not_exact",
                "The fuzzy candidate could not be measured.",
                unit=member,
                missing=tuple(m for p in failed for m in p.missing),
                symptoms=symptoms.merge([p.symptoms for p in failed]),
            )
        )
    old = min(snapshot.layout.fuzzy[member]["scores"].values()) if member in snapshot.layout.fuzzy else 0.0
    if min(p.score for p in proofs) - old < configuration.fuzzy()["min_gain"]:
        raise Refusal(Finding("fuzzy.no_gain", "The fuzzy candidate has insufficient gain.", unit=member))
    return (
        f"src/fuzzy/{store.stem(member).removesuffix('.c')}.c",
        inbox.read_bytes(),
        {p.version: p.score for p in proofs if p.member == member},
        compare.fuzzy_keys(config, [(bound, unit)])[0],
    )


def land_fuzzy(config: Config, submissions: Sequence[Any]) -> tuple[dict[str, Receipt], dict[str, tuple[Finding, ...]]]:
    """Fuzzy submissions landed as one write set: each admitted alone (a refusal leaves the others), one plan."""
    with effort.stage("publish.land_fuzzy"):
        snapshot = layout.capture(config)
        made: dict[str, tuple[str, bytes, dict[str, float], dict[str, str]]] = {}
        refused: dict[str, tuple[Finding, ...]] = {}
        for submission in submissions:
            try:
                made[submission.id] = _fuzzy(config, snapshot, submission)
            except Refusal as error:
                refused[submission.id] = error.findings
            except Exception as error:  # one crash must not take the others with it
                refused[submission.id] = (
                    Finding("internal.error", f"{type(error).__name__}: {error}", unit=submission.member),
                )
        if not made:
            return {}, refused
        fuzzy, writes, shown = dict(snapshot.layout.fuzzy), {}, []
        for submission in submissions:
            if submission.id in made:
                path, data, scores, key = made[submission.id]
                fuzzy[submission.member] = {"path": path, "scores": scores, "key": key}
                writes[path] = data
                shown.append(f"{submission.member} {min(scores.values()) * 100:.1f}%")
        writes["layout.toml"] = layout.dump_map(replace(snapshot.layout, fuzzy=fuzzy))
        writes.update(report.files(layout.overlay(snapshot, writes)))
        message = f"fuzzy {shown[0]}" if len(shown) == 1 else f"fuzzy {len(shown)} members: {', '.join(shown)}"
        plan = make_plan("fuzzy", snapshot, writes, (), (), message)
        files = journal.apply(config, plan, snapshot)
        return {i: Receipt("fuzzy", files, message, plan.digest, (), 0, effort.invocation()) for i in made}, refused
