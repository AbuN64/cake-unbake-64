"""Submission intake and the single project landing drain."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from unbake import (
    build,
    declarations,
    effort,
    headers,
    journal,
    layout,
    native,
    policy,
    pool,
    process,
    publish,
    recipes,
    store,
    view,
)
from unbake import config as configuration
from unbake.contracts import (
    Config,
    Finding,
    Json,
    Proof,
    Receipt,
    Refusal,
    Snapshot,
    Submission,
    UnitSpec,
    digest,
    proof_from,
    sha256,
)

_ATTEMPTS = (
    3  # a write that finds its files changed is planned again, up to this many times; the last holds the journal
)


def _read(path: Path) -> Submission:
    value = json.loads(path.read_text(encoding="utf-8"))
    if "extra_bases" not in value:
        configuration.validate("submission", {**value, "extra_bases": {}}, str(path))
        raise Refusal(
            Finding(
                "land.rebase",
                "Submission lacks measured extra-input provenance; resubmit against the current tree.",
                path=str(path),
            )
        )
    configuration.validate("submission", value, str(path))
    value.pop("refusal", None)
    value["proofs"] = tuple(proof_from(p) for p in value["proofs"])
    value["withheld"] = tuple(value["withheld"])
    return Submission(**value)


def _enqueue(
    config: Config,
    request: Json,
    member: str,
    operation: str,
    source: bytes,
    proofs: Sequence[Proof],
    origin: str,
    base: str,
    extras: Mapping[str, bytes],
    extra_bases: Mapping[str, bytes | None],
    withheld: Sequence[str],
) -> Submission:
    source_hash = sha256(source)
    identity = digest(
        (
            member,
            source_hash,
            request["overrides"],
            operation,
            *([sorted((p, sha256(b)) for p, b in extras.items())] if extras else []),
            *([sorted((p, sha256(b) if b is not None else None) for p, b in extra_bases.items())] if extras else []),
            *([tuple(withheld)] if withheld else []),
        )
    )
    inbox_path = config.project.root / ".unbake/inbox"
    for directory in (inbox_path, inbox_path / "done"):  # a refused entry is tried again: its refusal may be stale
        path = directory / f"{identity}.json"
        if path.is_file():
            return _read(path)
    source_path = f".unbake/inbox/{identity}.c"
    entry = Submission(
        identity,
        operation,
        member,
        request["function"],
        source_path,
        source_hash,
        request["overrides"],
        base,
        tuple(proofs),
        origin,
        request["note"],
        effort.invocation(),
        {path: f".unbake/inbox/{identity}.x{n}" for n, path in enumerate(sorted(extras))},
        {
            path: f".unbake/inbox/{identity}.b{n}" if data is not None else None
            for n, (path, data) in enumerate(sorted(extra_bases.items()))
        },
        tuple(withheld),
    )
    value = json.loads(json.dumps(asdict(entry)))
    configuration.validate("submission", value, f".unbake/inbox/{identity}.json")
    store.write(config.project.root / source_path, source)
    for extra, copy in entry.extras.items():
        store.write(config.project.root / copy, extras[extra])
    for extra, copy in entry.extra_bases.items():
        data = extra_bases[extra]
        if copy is not None and data is not None:
            store.write(config.project.root / copy, data)
    store.write(inbox_path / f"{identity}.json", json.dumps(value, sort_keys=True).encode())
    return entry


def submit(
    config: Config,
    request: Json,
    unit: UnitSpec,
    proofs: Sequence[Proof],
    source: bytes,
    origin: str,
    extras: Mapping[str, bytes] | None = None,
    withhold: bool = False,
    snapshot: Snapshot | None = None,
) -> Submission | None:
    """One submission. `extras` (path -> new text) makes it a change set landed as one plan, every touched unit exact
    or the whole set refused; only exact sets are submitted. `withhold` lands the exact versions and withholds the rest
    (with none exact it measures as without it). `snapshot` is the capture the caller measured against."""
    with effort.stage("land.submit"):
        intake = _intake(config, request, unit, proofs, source, extras, withhold, snapshot)
        if intake is None:
            return None
        if intake.operation == "publish":  # refusal
            publish.accept(intake.snapshot, intake.check, intake.extras, bool(intake.withheld), present=False)
        return _entered(config, request, source, proofs, origin, intake)


def submit_many(config: Config, calls: Sequence[Mapping[str, Any]]) -> list[Submission | Refusal | None]:
    """Many submissions (each call holds `submit`'s arguments by name), their publish acceptance checks in ONE
    dispatch across every call, never one call's checks after another's. Each call keeps its own verdict: its
    Refusal is its result and blocks no other call; the accepted ones enter the queue in order."""
    with effort.stage("land.submit_many"):
        intakes = [
            _intake(
                config,
                c["request"],
                c["unit"],
                c["proofs"],
                c["source"],
                c.get("extras"),
                c.get("withhold", False),
                c.get("snapshot"),
            )
            for c in calls
        ]
        checked = [n for n, i in enumerate(intakes) if i is not None and i.operation == "publish"]
        verdicts = dict(
            zip(
                checked,
                publish.accept_each(
                    config,
                    [
                        (i.snapshot, i.check, i.extras, bool(i.withheld))
                        for i in (intakes[n] for n in checked)
                        if i is not None
                    ],
                ),
                strict=True,
            )
        )
        out: list[Submission | Refusal | None] = []
        for n, (call, intake) in enumerate(zip(calls, intakes, strict=True)):
            if intake is None:
                out.append(None)
            elif verdicts.get(n):
                out.append(Refusal(*verdicts[n]))
            else:
                try:
                    out.append(
                        _entered(config, call["request"], call["source"], call["proofs"], call["origin"], intake)
                    )
                except Refusal as refusal:
                    out.append(refusal)
        return out


@dataclass(frozen=True)
class _Intake:
    """What a submission enters as, decided before its acceptance check."""

    member: str
    operation: str
    withheld: tuple[str, ...]
    snapshot: Snapshot
    extras: Mapping[str, bytes]
    extra_bases: Mapping[str, bytes | None]
    check: Json  # the request the publish acceptance check reads


def _intake(
    config: Config,
    request: Json,
    unit: UnitSpec,
    proofs: Sequence[Proof],
    source: bytes,
    extras: Mapping[str, bytes] | None,
    withhold: bool,
    snapshot: Snapshot | None,
) -> _Intake | None:
    """The operation a submission enters as, or None when there is nothing to submit."""
    member = request["function"] or unit.members[0]
    snapshot = snapshot or layout.capture(config)
    current = snapshot.layout.units.get(unit.path)
    kinds = configuration.kinds()
    path = config.project.root / unit.path
    extras = extras or {}
    extra_bases = {p: snapshot.peek(p) for p in extras}
    if current and kinds[current.kind]["decompiled"] and path.is_file() and path.read_bytes() == source and not extras:
        return None
    exact = {p.version for p in proofs if all(q.exact for q in proofs if q.version == p.version)}
    withheld = tuple(sorted({p.version for p in proofs} - exact)) if withhold and exact else ()
    check = {**request, "function": member}
    if proofs and (all(p.exact for p in proofs) or withheld):
        return _Intake(member, "publish", withheld, snapshot, extras, extra_bases, check)
    if extras:
        return None
    scores = snapshot.layout.fuzzy[member]["scores"] if member in snapshot.layout.fuzzy else {}
    before = min(scores.values()) if scores else 0.0
    gain = configuration.fuzzy()["min_gain"]
    if (
        not proofs
        or any(any(key in p.symptoms for key in ("compile_failed", "unresolved", "no_measurement")) for p in proofs)
        or min(p.score for p in proofs) - before < gain
    ):
        return None
    return _Intake(member, "fuzzy", withheld, snapshot, extras, extra_bases, check)


def _entered(
    config: Config, request: Json, source: bytes, proofs: Sequence[Proof], origin: str, intake: _Intake
) -> Submission:
    with ExitStack() as stack:
        if intake.extras:
            stack.enter_context(journal.held(config))
        if changed := journal.changed(intake.snapshot, tuple(intake.extras)):
            raise Refusal(
                Finding(
                    "journal.changed", "Extra inputs changed during admission.", missing=changed, action="submit again"
                )
            )
        return _enqueue(
            config,
            request,
            intake.member,
            intake.operation,
            source,
            proofs,
            origin,
            intake.snapshot.commit,
            intake.extras,
            intake.extra_bases,
            intake.withheld,
        )


def _move(config: Config, identity: str, destination: str) -> None:
    root = config.project.root / ".unbake/inbox"
    target = root / destination
    target.mkdir(parents=True, exist_ok=True)
    for path in root.glob(f"{identity}.*"):
        if path.suffix != ".lock":
            os.replace(path, target / path.name)


def inbox(config: Config, only: str | None = None) -> list[Submission]:
    with effort.stage("land.inbox"):
        entries, stamped = [], []
        for path in (config.project.root / ".unbake/inbox").glob("*.json"):
            if only is not None and path.stem != only:
                continue
            with suppress(FileNotFoundError):  # another drain landed it between the listing and now
                stamped.append((path.stat().st_mtime_ns, path.name, path))
        for _, _, path in sorted(stamped):
            try:
                entries.append(_read(path))
            except FileNotFoundError:
                continue
            except (Refusal, ValueError, TypeError, KeyError, OSError) as error:
                if isinstance(error, Refusal) and any(f.key == "land.rebase" for f in error.findings):
                    raise  # unsupported provenance stays intact for explicit review and resubmission
                finding = Finding("inbox.corrupt", str(error), path=str(path.relative_to(config.project.root)))
                _move(config, path.stem, "refused")
                store.log(config, "refusal", {"id": path.stem, "findings": [asdict(finding)]})
        return entries


def _retry[T](config: Config, attempt_land: Callable[[], T]) -> T:
    """Planning takes no lock, only the write: a file changed under it means plan again; the last holds the journal."""
    for _ in range(_ATTEMPTS - 1):
        try:
            return attempt_land()
        except Refusal as error:
            if not any(f.key == "journal.changed" for f in error.findings):
                raise
    with journal.held(config):
        return attempt_land()


def _land(config: Config, entry: Submission) -> Receipt:
    return _retry(config, lambda: publish.land(config, entry))


def _findings(error: Exception, unit: str = "") -> tuple[Finding, ...]:
    """What a failed landing refuses with: a refusal's own findings, else one internal error."""
    return (
        error.findings
        if isinstance(error, Refusal)
        else (Finding("internal.error", f"{type(error).__name__}: {error}", unit=unit),)
    )


_PENDING = ".unbake/generate.pending"  # a landing whose generated files are not derived yet (a drain stopped between)


def _record_landed(config: Config, entry: Submission, receipt: Receipt, landed: list[Any]) -> None:
    store.write(config.project.root / _PENDING, entry.id.encode())
    body = {"id": entry.id, "member": entry.member}
    landed.append({**body, "operation": receipt.operation, "files": list(receipt.files), "message": receipt.message})
    store.log(config, "receipt", {**body, **asdict(receipt)})
    _move(config, entry.id, "done")


def _record_refused(config: Config, entry: Submission, findings: Sequence[Finding], refused: list[Any]) -> None:
    body = {"id": entry.id, "member": entry.member, "findings": [asdict(f) for f in findings]}
    refused.append(body)
    store.log(config, "refusal", body)
    _move(config, entry.id, "refused")
    kept = config.project.root / f".unbake/inbox/refused/{entry.id}.json"
    with suppress(OSError):  # the refusal is read from the entry itself, not only from the ledger
        store.write(kept, json.dumps({**json.loads(kept.read_text()), "refusal": body["findings"]}).encode())


def _fuzzies(config: Config, entries: Sequence[Submission], landed: list[Any], refused: list[Any]) -> None:
    """The claimed fuzzy entries landed as one write set; a refused one is recorded and the rest land."""
    try:
        receipts, refusals = _retry(config, lambda: publish.land_fuzzy(config, entries))
    except Exception as error:
        receipts, refusals = {}, {e.id: _findings(error) for e in entries}
    for entry in entries:
        if entry.id in receipts:
            _record_landed(config, entry, receipts[entry.id], landed)
        else:
            _record_refused(config, entry, refusals[entry.id], refused)


def drain(config: Config, only: str | None = None) -> Json:
    """Lands the inbox, one lock per entry so drains run side by side; `only` lands the one entry of that id."""
    with effort.stage("land.drain"):
        landed: list[Any] = []
        refused: list[Any] = []
        busy = False
        with store.exclusive(config, "land", wait=False) as free:
            busy = not free
        while not busy and (entries := inbox(config, only)):
            claimed = False
            fuzzy = [e for e in entries if e.operation == "fuzzy"][: configuration.fuzzy()["chunk"]]
            with ExitStack() as locks:
                mine_fuzzy = [
                    e
                    for e in fuzzy
                    if locks.enter_context(store.exclusive(config, f"inbox/{e.id}", wait=False))
                    and (config.project.root / f".unbake/inbox/{e.id}.json").exists()
                ]
                if mine_fuzzy:
                    claimed = True
                    with effort.stage("land.fuzzy"), store.shared(config, "structure"):
                        _fuzzies(config, mine_fuzzy, landed, refused)
            for e in mine_fuzzy:
                (config.project.root / f".unbake/inbox/{e.id}.lock").unlink(missing_ok=True)
            for entry in entries:
                if entry in fuzzy:
                    continue
                with store.exclusive(config, f"inbox/{entry.id}", wait=False) as mine:
                    if not mine or not (config.project.root / f".unbake/inbox/{entry.id}.json").exists():
                        continue
                    claimed = True
                    with effort.stage("land.entry"), store.shared(config, "structure"), effort.scoped():
                        try:
                            receipt = _land(config, entry)
                        except Exception as error:  # a crash must not leave the entry to fail again unseen
                            _record_refused(config, entry, _findings(error, entry.member), refused)
                        else:
                            _record_landed(config, entry, receipt, landed)
                (config.project.root / f".unbake/inbox/{entry.id}.lock").unlink(missing_ok=True)
            busy = not claimed  # what is left belongs to other drains
        publish.settle()
        generated: list[str] = []
        pending = config.project.root / _PENDING
        if landed or pending.is_file():  # the type map, report.json and generated files, once per drain that landed
            with store.shared(config, "structure"):  # a drain that landed nothing writes nothing
                generated = list(_retry(config, lambda: publish.generate(config)))
            pending.unlink(missing_ok=True)
            if generated:  # a receipt row like any entry's: lanes commit the paths their ledger lists
                store.log(
                    config, "receipt", {"id": "generate", "member": "", "operation": "generate", "files": generated}
                )
        return {"running": busy, "landed": landed, "refused": refused, "generated": generated}


def land_command(config: Config, params: Json) -> Json:
    return drain(config)


_INCLUDE = re.compile(r'#\s*include\s*[<"]([^>"]+)[>"]')
_GUARD = re.compile(r"#\s*ifndef\s+(\w+)$")


def _shape(rows: Sequence[tuple[frozenset[str], frozenset[str], str]]) -> tuple[str | None, list[str], list[str]]:
    """A header's directives: (its include guard, every other non-include directive in order, its include names). The
    guard only wraps the file, so a header that gained one (a new header) differs in nothing its includers read."""
    texts = [t for names, _, t in rows if not names]
    includes = [m[1] for t in texts if (m := _INCLUDE.match(t))]
    other = [t for t in texts if not _INCLUDE.match(t)]
    guard = _GUARD.match(other[0]) if len(other) >= 2 and re.match(r"#\s*endif\b", other[-1]) else None
    defined = guard is not None and any(guard[1] in names for names, _, t in rows if t.startswith("#"))
    return (guard[1], other[1:-1], includes) if guard and defined else (None, other, includes)


def _reached(snapshot: Snapshot, path: str, names: Collection[str]) -> set[str] | None:
    """The project headers an include of `names` in `path` reaches, through their own includes; None when one of them
    is found nowhere (what it brings in cannot be told)."""

    def find(base: str, name: str) -> str | None:
        options = (os.path.normpath(os.path.join(d, name)) for d in (os.path.dirname(base), "include", "src"))
        return next((o for o in options if snapshot.peek(o) is not None), None)

    found: set[str] = set()
    todo = [(path, n) for n in names]
    while todo:
        base, name = todo.pop()
        target = find(base, name)
        if target is None:
            return None
        if target not in found:
            found.add(target)
            todo.extend((target, raw.decode()) for _, raw in view.directives(snapshot.peek(target) or b""))
    return found


def declaration_users(before: Snapshot, after: Snapshot, affected: set[str], changed: Sequence[str]) -> set[str]:
    """The `affected` units that use a declaration the changed headers alter (all of them when that cannot be told).
    An include a changed header gains or drops alters only the names the headers it reaches declare: a unit naming none
    of them (and including no other header that declares one) reads the same declarations as before."""
    roots: set[Any] = set()
    reached: dict[bool, set[str]] = {True: set(), False: set()}  # after (gained includes), before (dropped ones)
    dead = headers.dangling(before, after, changed)
    # a header naming a dropped declaration no longer parses: each of its includers breaks, used or not
    broken = affected & {*dead, *headers.consumers(after, dead)}
    for path in changed:
        if configuration.is_header(path):
            old, new = (declarations.dependencies((s.peek(path) or b"").decode()) for s in (before, after))
            if old is None or new is None:
                return affected
            (old_guard, old_rest, old_includes), (new_guard, new_rest, new_includes) = _shape(old), _shape(new)
            if old_rest != new_rest or (before.peek(path) is not None and old_guard != new_guard):
                return affected  # conditional and other directives can affect every consumer
            for gained, snap, moved in (
                (True, after, set(new_includes) - set(old_includes)),
                (False, before, set(old_includes) - set(new_includes)),
            ):
                found = _reached(snap, path, moved) if moved else set()
                if found is None:
                    return affected
                reached[gained] |= found
            tables: list[defaultdict[str, set[str]]] = [defaultdict(set), defaultdict(set)]
            for table, group in zip(tables, (old, new), strict=True):
                for names, _, text in group:
                    for name in names:
                        table[name].add(text)
            roots.update(n for n in tables[0].keys() | tables[1].keys() if tables[0].get(n) != tables[1].get(n))
    if not roots and not reached[True] and not reached[False]:
        return broken
    texts = sorted(
        {s.read(p).decode() for s in (before, after) for p in headers.sources(s)}
        | {(s.peek(p) or b"").decode() for gained, s in ((True, after), (False, before)) for p in reached[gained]}
    )
    parsed = pool.map(after.config, "land.declarations", declarations.dependencies, texts, digest)
    if any(row is None for row in parsed):
        return affected
    rows_of = dict(zip(texts, parsed, strict=True))
    # a name a gained or dropped include brings is read only where it is named; a macro also reaches other headers
    seen: set[str] = set()
    for gained, snap in ((True, after), (False, before)):
        for p in reached[gained]:
            rows = rows_of[(snap.peek(p) or b"").decode()] or []
            if _shape(rows)[0] is None and any(names for names, _, _ in rows):
                return affected  # an unguarded header read twice redefines what it declares, named or not
            for names, _, text in rows:
                (roots if text.startswith("#") else seen).update(names)
    users: defaultdict[str, set[str]] = defaultdict(set)
    for group in parsed:
        for names, words, _ in group or ():
            for word in words:
                users[word].update(names)
    pending = list(roots)
    while pending:
        extra = users[pending.pop()] - roots
        roots.update(extra)
        pending.extend(extra)
    # another header declaring a newly reached name (or testing a reached macro) may now clash with it unnamed
    every = {*reached[True], *reached[False]}
    clash = [
        p
        for p in headers.sources(after)
        if p not in every
        and any(
            names & seen or (not names and words & roots) for names, words, _ in rows_of[after.read(p).decode()] or ()
        )
    ]
    broken |= affected & set(headers.consumers(after, clash))
    wanted = roots | seen

    def used(snap: Snapshot, path: str) -> set[str]:
        masks = policy.scopes(snap.read(path).decode())
        directives = re.sub(r"(?m)^\s*#\s*include[^\n]*", "", masks["directives"])
        return wanted & set(re.findall(r"\w+", masks["code"] + directives))

    return broken | {p for p in affected if any(used(s, p) for s in (before, after))}


def _consumers(before: Snapshot, after: Snapshot, affected: set[str], changed: Sequence[str]) -> list[str]:
    """Prove changed units and users whose reached declarations changed, not every header includer."""
    own = affected & (set(changed) | (after.layout.units.keys() - before.layout.units.keys()))
    return sorted(own | declaration_users(before, after, affected - own, changed))


def _project_units(before: Snapshot, after: Snapshot) -> set[str]:
    old, new = before.config.project, after.config.project
    if old.digest == new.digest:
        return set()
    units = {**before.layout.units, **after.layout.units}
    if replace(old, version_differences=(), origins={}, digest="") != replace(
        new, version_differences=(), origins={}, digest=""
    ):
        return set(units)
    rows = [{digest(row): row["function"] for row in p.version_differences} for p in (old, new)]
    names = {rows[0].get(key, rows[1].get(key)) for key in rows[0].keys() ^ rows[1].keys()}
    return {path for path, unit in units.items() if names.intersection(unit.members)}


def check_command(config: Config, params: Json) -> Json:
    with effort.stage("land.check_command"):
        if params.get("staged"):
            commit = process.git(config, "rev-parse", "HEAD").stdout.decode().strip()
            index = process.git(config, "write-tree").stdout.decode().strip()
            snapshot = layout.tree(config, commit, commit)
            staged = layout.tree(config, index, commit)
            diffed = [
                p
                for p in process.git(
                    config,
                    "diff",
                    "--cached",
                    "--name-only",
                    "--no-renames",
                    "-z",
                    "--diff-filter=ACMTD",
                    snapshot.commit,
                    "--",
                    *params.get("paths", ()),
                )
                .stdout.decode()
                .split("\0")
                if configuration.is_text(p)
            ]
            paths = sorted(set(diffed) | _project_units(snapshot, staged))
            changes = {p: ((snapshot.peek(p) or b"").decode(), (staged.peek(p) or b"").decode()) for p in paths}
            findings, _ = policy.census(staged, changes=changes, baseline=snapshot)
            blocking = tuple(f for f in findings if f.blocking)
            duplicate = [
                f for p, (_, text) in changes.items() if configuration.is_header(p) for f in headers.duplicates(p, text)
            ]
            if paths:
                affected = set(headers.consumers(staged, paths)) | set(headers.consumers(snapshot, paths))
                affected.update(p for p in paths if p in staged.layout.units)
                affected.intersection_update(staged.layout.units)
                proved = _consumers(snapshot, staged, affected, paths)
                if proved:
                    _, gaps = publish._proofs(staged, proved)
                    gaps = tuple(
                        replace(f, versions=held)
                        for f in gaps
                        if (member := snapshot.layout.members.get(f.unit)) is not None
                        and (held := tuple(v for v in f.versions if v in member.holders()))
                    )
                    own = {n for p in diffed if p in staged.layout.units for n in staged.layout.units[p].members}
                    publish.refuse_blocking(
                        (*publish._regressions(snapshot, gaps, own, "staged declarations"), *duplicate, *blocking)
                    )
            publish.refuse_blocking((*duplicate, *blocking))
            return {"commit": snapshot.commit, "counts": policy.counts(findings)}
        snapshot = layout.capture(config)
        repaired, clashes = _repair(snapshot) if params["repair"] else ({}, [])
        disagree = [f for f in clashes if not f.blocking]  # debt the census reports beside its own
        clashes = [f for f in clashes if f.blocking]
        scoped = _scope(snapshot, params.get("paths", ()))
        findings, (compiled,) = policy.census(
            snapshot,
            [  # the compiles fill the workers the census leaves idle
                (
                    "land.compile",
                    build.check_unit,
                    build.compile_jobs(snapshot, list(snapshot.layout.units if scoped is None else scoped)),
                    build.check_key,
                )
            ],
            only=scoped,
        )
        broken = _merged(compiled)
        debt, findings = findings, (*findings, *disagree, *(f for f in broken if not f.blocking))
        broken = [f for f in broken if f.blocking]
        counts = policy.counts(findings)
        if scoped is None:  # the file is the whole project's census: a scoped run must not replace it
            store.write(
                config.project.root / ".unbake/check.json",
                json.dumps(
                    {
                        "commit": snapshot.commit,
                        "digest": snapshot.digest,
                        "counts": counts,
                        "findings": [asdict(f) for f in findings],
                        "failing": sorted({f.unit for f in broken}),
                    },
                    sort_keys=True,
                ).encode(),
            )
        if broken:
            raise Refusal(*broken)
        if params["strict"] and debt:
            raise Refusal(
                Finding(
                    "check.debt",
                    f"{len(debt)} findings",
                    missing=tuple(sorted({f.key for f in debt})),
                    action="fix each finding in the source",
                )
            )
        if clashes:
            raise Refusal(*clashes)
        withholding, rows = _located(snapshot, findings)
        return {
            "commit": snapshot.commit,
            "counts": counts,
            "withholding": policy.counts(withholding),
            **({"findings": rows} if scoped is not None else {}),
            **({"repaired": sorted(repaired)} if repaired else {}),
        }


def _located(snapshot: Snapshot, findings: Sequence[Finding]) -> tuple[list[Finding], list[Json]]:
    """The findings that withhold credit (an unclean rule in a unit's own text; one in a header it includes only
    counts) and every finding as a row with its path:line, the withholding ones first."""
    unclean = set(configuration.load_resource("rules.toml").get("unclean", ()))
    held = [f for f in findings if f.key.removeprefix("source.") in unclean and f.path in snapshot.layout.units]
    mine = {id(f) for f in held}
    rows: list[Json] = [
        {"at": f"{f.path}:{f.line}", "key": f.key, "reason": f.reason, "withholding": id(f) in mine}
        for f in sorted(findings, key=lambda f: (id(f) not in mine, f.path, f.line, f.key))
    ]
    return held, rows


def _relative(snapshot: Snapshot, given: str) -> str:
    """`given` as a project-relative path: an absolute path under the project root loses the root."""
    root = snapshot.config.project.root
    path = Path(given)
    return path.relative_to(root).as_posix() if path.is_absolute() and path.is_relative_to(root) else path.as_posix()


def _scope(snapshot: Snapshot, paths: Sequence[str]) -> frozenset[str] | None:
    """The unit paths a path-limited check judges: each given unit, and for a header every unit that reads it."""
    if not paths:
        return None
    units, found = snapshot.layout.units, set[str]()
    for given in map(lambda p: _relative(snapshot, p), paths):
        if given not in units and snapshot.peek(given) is None:
            raise Refusal(Finding("check.path", f"{given} is no unit or header of this project", action="unbake check"))
        found.update(p for p in ((given,) if given in units else headers.consumers(snapshot, [given])) if p in units)
    return frozenset(found)


def _merged(results: Sequence[Sequence[Finding]]) -> list[Finding]:
    merged: dict[tuple[str, str, str], Finding] = {}
    for result in results:
        for f in result:
            old = merged.get(k := (f.key, f.unit, f.reason), f)
            merged[k] = replace(old, versions=tuple(sorted({*old.versions, *f.versions})))
    return list(merged.values())


def failing(
    snapshot: Snapshot, paths: Sequence[str]
) -> list[Finding]:  # units among `paths` that do not compile in some version
    return _merged(pool.map(snapshot.config, "land.compile", native.compiles, build.compile_jobs(snapshot, paths)))


def _preprocessed(
    snapshot: Snapshot, changed: Mapping[str, bytes | None]
) -> None:  # nothing is written unless every consumer still compiles
    trial = layout.overlay(snapshot, changed)
    if found := failing(trial, headers.consumers(trial, changed)):
        raise Refusal(*found)


def _live_words(snapshot: Snapshot) -> set[str]:
    """Every word of the live text of every decompiled unit in every version it holds."""
    words: set[str] = set()
    for unit in snapshot.layout.units.values():
        if not configuration.is_source(unit.path) or not configuration.kind(unit.kind)["decompiled"]:
            continue
        for version in sorted(snapshot.layout.held(unit.members)):
            source = view.get(snapshot, unit, version, recipes.resolve(snapshot.config, unit, {}), lines=False)
            text = declarations.live(
                source.text,
                source.unit,
                lambda key, produce: store.cached(snapshot.config, "declarations", key, produce),
            )
            words.update(re.findall(r"[A-Za-z_]\w*", text))
    return words


def _prototype_users(trial: Snapshot, paths: Sequence[str]) -> set[str]:
    """The units that read one of the written files: the sources among them and the includers of the headers."""
    return {
        p
        for p in (*paths, *headers.consumers(trial, [p for p in paths if configuration.is_header(p)]))
        if p in trial.layout.units
    }


def _prototypes(snapshot: Snapshot) -> tuple[dict[str, bytes | None], list[Finding]]:
    """Gives every name one prototype in the working tree (headers.prototypes). A change that keeps every spelling a
    unit reads (a repeat dropped) must still compile. A change of a spelling is also proved: a unit that proves exact
    at HEAD stays exact. A name whose change breaks a unit, raises its view-conversion findings or loses a proof is
    left as it was; the census then still reports it. The findings are the names whose declarations disagree."""
    skip: set[str] = set()
    while True:
        items, writes, conflicts = headers.prototypes(snapshot, skip)
        if not writes:
            return {}, conflicts
        trial = layout.overlay(snapshot, writes)
        users = {n: _prototype_users(trial, sorted(paths)) for n, (_, paths) in items.items()}
        every = sorted(set().union(*users.values()))
        before, after = (Counter((f.unit or f.path, f.key) for f in failing(snap, every)) for snap in (snapshot, trial))
        worse = {unit for (unit, key), n in after.items() if n > before[(unit, key)]}
        typed = sorted(set().union(*(users[n] for n, (t, _) in items.items() if t)) - worse)
        if typed and not worse:
            _, gaps = publish._proofs(trial, typed)
            for f in publish._regressions(snapshot, gaps, set(), "prototype repair"):
                holder = layout.unit_of(trial, f.unit)
                worse.add(f.path or (holder.path if holder else ""))
        bad = {n for n, units in users.items() if units & worse}
        if not bad:
            if worse:
                return {}, conflicts  # a regression no name owns: nothing is written
            plan = publish.make_plan("repair", snapshot, writes, (), (), "repair prototypes")
            journal.apply(snapshot.config, plan, snapshot)
            return dict(plan.writes), conflicts
        skip |= bad


def _repair(snapshot: Snapshot) -> tuple[dict[str, bytes | None], list[Finding]]:
    """Write the working tree with every header's repeated declaration folded into its owner and the version_differences
    rows its guards earn (never commits)."""
    protos, conflicts = _prototypes(snapshot)
    if protos:
        snapshot = layout.capture(snapshot.config)
    texts: dict[str, bytes | None] = {p: snapshot.read(p) for p in headers.sources(snapshot)}
    fixed, clashes = headers.reconcile(snapshot, snapshot, texts)
    current = {p: (fixed.get(p, d) or b"").decode() for p, d in texts.items() if d is not None}
    trimmed = declarations.dead(current, _live_words(snapshot))
    fixed = {**fixed, **{p: t.encode() for p, t in trimmed.items()}}
    changed = {p: d for p, d in fixed.items() if d != texts[p]}
    if (rows := policy.version_config(snapshot)) != snapshot.read("config.toml"):
        changed["config.toml"] = rows
    if changed:
        _preprocessed(snapshot, {p: d for p, d in changed.items() if p != "config.toml"})
        plan = publish.make_plan(
            "repair", snapshot, changed, (), (), "repair header declarations and version differences"
        )
        journal.apply(snapshot.config, plan, snapshot)
        changed = dict(plan.writes)
    return {**protos, **changed}, [*clashes, *conflicts]
