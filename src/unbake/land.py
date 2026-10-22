"""Submission intake and the single project landing drain."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Collection, Mapping, Sequence
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from unbake import api, effort, foreign, journal, policy, process, publish, report, rescore, store, tree
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Proof, Receipt, Refusal, Submission, digest, proof_from, sha256


def _read(path: Path) -> Submission:
    value = json.loads(path.read_text(encoding="utf-8"))
    if "extra_bases" not in value:
        configuration.validate("submission", {**value, "extra_bases": {}}, str(path))
        raise Refusal(
            Finding(
                "land.rebase",
                "Submission lacks measured extra-input provenance; resubmit against the current tree.",
                path=str(path),
                action=f"recipe try {value.get('member', 'ITEM')}",
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
            try:
                return _read(path)
            except Refusal as stale:
                if not any(f.key == "land.rebase" for f in stale.findings):
                    raise
                _move(config, identity, "stale")  # a pre-provenance entry: measured again below, not refused forever
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
        os.environ.get("RECIPE_SESSION", ""),
        time.time(),
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
    proofs: Sequence[Proof],
    source: bytes,
    origin: str,
    extras: Mapping[str, bytes] | None = None,
) -> Submission | None:
    """One submission. `extras` (path -> new text) makes it a change set landed as one plan, every touched version
    building to its base sha1 or the whole set refused; only exact sets are submitted."""
    with effort.stage("land.submit"):
        intake = _intake(config, request, proofs, extras)
        if intake is None:
            return None
        publish.accept(config, intake.check, intake.extras, intake.operation == "fuzzy")
        return _entered(config, request, source, proofs, origin, intake)


def submit_many(config: Config, calls: Sequence[Mapping[str, Any]]) -> list[Submission | Refusal | None]:
    """Many submissions (each call holds `submit`'s arguments by name), their acceptance checks together. Each call
    keeps its own verdict: its Refusal is its result and blocks no other call; the accepted ones enter the queue in
    order."""
    with effort.stage("land.submit_many"):
        intakes = [_intake(config, c["request"], c["proofs"], c.get("extras")) for c in calls]
        checked = [n for n, i in enumerate(intakes) if i is not None]
        verdicts = dict(
            zip(
                checked,
                publish.accept_each(
                    config,
                    [
                        (i.check, i.extras, i.operation == "fuzzy")
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
    base: str
    extras: Mapping[str, bytes]
    extra_bases: Mapping[str, bytes | None]
    check: Json  # the request the acceptance check reads


def _peek(config: Config, path: str) -> bytes | None:
    try:
        return (config.project.root / path).read_bytes()
    except FileNotFoundError:
        return None


def _intake(
    config: Config, request: Json, proofs: Sequence[Proof], extras: Mapping[str, bytes] | None
) -> _Intake | None:
    """The operation a submission enters as, or None when there is nothing to submit: a candidate that is not exact
    must score above the function's cached score by `fuzzy.min_gain`."""
    member = request["function"]
    extras = extras or {}
    extra_bases = {p: _peek(config, p) for p in extras}
    base = process.git(config, "rev-parse", "HEAD").stdout.decode().strip()
    if proofs and all(p.exact for p in proofs):
        return _Intake(member, "publish", base, extras, extra_bases, request)
    if extras or not proofs:
        return None
    if any(any(key in p.symptoms for key in ("compile_failed", "unresolved", "no_measurement")) for p in proofs):
        return None
    if min(p.score for p in proofs) - publish.before_score(config, member) < configuration.fuzzy()["min_gain"]:
        return None
    return _Intake(member, "fuzzy", base, extras, extra_bases, request)


def _entered(
    config: Config, request: Json, source: bytes, proofs: Sequence[Proof], origin: str, intake: _Intake
) -> Submission:
    with ExitStack() as stack:
        if intake.extras:
            stack.enter_context(journal.held(config))
        if stale := tuple(p for p, b in intake.extra_bases.items() if _peek(config, p) != b):
            raise Refusal(
                Finding(
                    "journal.changed", "Extra inputs changed during admission.", missing=stale, action="submit again"
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
            intake.base,
            intake.extras,
            intake.extra_bases,
            (),
        )


def _move(config: Config, identity: str, destination: str) -> None:
    root = config.project.root / ".unbake/inbox"
    target = root / destination
    target.mkdir(parents=True, exist_ok=True)
    for path in root.glob(f"{identity}.*"):
        if path.suffix != ".lock":
            os.replace(path, target / path.name)


_STALE_SECONDS = 3 * 3600  # an entry nobody drained in this long belongs to a lane that is gone


def _abandoned(session: str, entered: float) -> bool:
    """An entry whose numeric session (a process id) no longer exists, or that outlived `_STALE_SECONDS`."""
    if entered and time.time() - entered > _STALE_SECONDS:
        return True
    try:
        os.kill(int(session), 0)
    except ProcessLookupError:
        return True
    except (ValueError, OSError):
        return False
    return False


def expire(config: Config) -> list[str]:
    """Moves each abandoned entry (dead session, or older than `_STALE_SECONDS`) to inbox/attic; returns their ids."""
    gone = []
    for path in sorted((config.project.root / ".unbake/inbox").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict) and _abandoned(
            str(value.get("session", "")), float(value.get("entered") or path.stat().st_mtime)
        ):
            _move(config, path.stem, "attic")
            gone.append(path.stem)
    return gone


def inbox(config: Config, only: Collection[str] | str | None = None) -> list[Submission]:
    with effort.stage("land.inbox"):
        entries, stamped = [], []
        wanted = {only} if isinstance(only, str) else only
        for path in (config.project.root / ".unbake/inbox").glob("*.json"):
            if wanted is not None and path.stem not in wanted:
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


def _findings(error: Exception, unit: str = "") -> tuple[Finding, ...]:
    """What a failed landing refuses with: a refusal's own findings, else one internal error."""
    return (
        error.findings
        if isinstance(error, Refusal)
        else (Finding("internal.error", f"{type(error).__name__}: {error}", unit=unit),)
    )


def _record_landed(config: Config, entry: Submission, receipt: Receipt, landed: list[Any]) -> None:
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


def drain(config: Config, only: Collection[str] | str | None = None) -> Json:
    """Lands the inbox, one lock per entry so drains run side by side; `only` lands the entries of those ids and
    leaves every other lane's entry where it is. Abandoned entries go to inbox/attic first."""
    with effort.stage("land.drain"):
        expire(config)
        landed: list[Any] = []
        refused: list[Any] = []
        busy = False
        with store.exclusive(config, "land", wait=False) as free:
            busy = not free
        while not busy and (entries := inbox(config, only)):
            claimed = False
            for entry in entries:
                with store.exclusive(config, f"inbox/{entry.id}", wait=False) as mine:
                    if not mine or not (config.project.root / f".unbake/inbox/{entry.id}.json").exists():
                        continue
                    claimed = True
                    with effort.stage("land.entry"), store.shared(config, "structure"), effort.scoped():
                        try:
                            receipt = publish.land(config, entry)
                        except Exception as error:  # a crash must not leave the entry to fail again unseen
                            _record_refused(config, entry, _findings(error, entry.member), refused)
                        else:
                            _record_landed(config, entry, receipt, landed)
                (config.project.root / f".unbake/inbox/{entry.id}.lock").unlink(missing_ok=True)
            busy = not claimed  # what is left belongs to other drains
        return {"running": busy, "landed": landed, "refused": refused}


def land_command(config: Config, params: Json) -> Json:
    return drain(config, tuple(params.get("only") or ()) or None)


def _bodies(config: Config, paths: Sequence[str]) -> tuple[Finding, ...]:
    """The foreign-body rule over the NON_MATCHING bodies of `paths`. A body is measured once per content (the score
    cache); a version whose ROM is missing is named and left unmeasured."""
    held = tree.capture(config)
    modules = [m for m in held.modules if m.path in set(paths) and (m.fuzzy or m.unnamed)]
    try:
        scores = rescore.rows(config, SimpleNamespace(digest=held.digest, modules=modules), report.facts(config))
    except Refusal as refusal:
        return tuple(replace(f, blocking=False) for f in refusal.findings)
    return foreign.findings(config, modules, scores)


def check_command(config: Config, params: Json) -> Json:
    """The source rules over the files that own names (or the given paths); a blocking finding refuses."""
    with effort.stage("land.check_command"):
        given = list(params.get("paths", ()))
        paths = sorted(given or {m.path for m in tree.capture(config).modules})
        snapshot = api.snapshot(config)
        findings: list[Finding] = []
        for path in paths:
            if (text := _peek(config, path)) is not None:
                findings.extend(policy.evaluate(snapshot, path, text.decode(), None, False))
        findings.extend(_bodies(config, paths))
        publish.refuse_blocking(findings)
        report.record(config, tree.capture(config), commit=snapshot.commit, counts=policy.counts(findings))
        if params.get("strict") and findings:
            raise Refusal(
                Finding(
                    "check.debt",
                    f"{len(findings)} findings",
                    missing=tuple(sorted({f.key for f in findings})),
                    action="fix each finding in the source",
                )
            )
        return {"commit": snapshot.commit, "counts": policy.counts(findings)}
