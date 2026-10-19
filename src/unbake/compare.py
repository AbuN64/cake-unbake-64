"""Measure a candidate function in every version that holds it and report the gaps against the ROMs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from unbake import alloc, effort, land, native, policy, pool, publish, sched, store, symptoms, tree, versions
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Proof, Refusal

_LOCATION = re.compile(r"([\w./-]+\.\w+):(\d+): (?!previous)")  # PATH:LINE: of a diagnostic, not of its note
_ROOT_ERROR = "the file {} does not exist"


# ---------------------------------------------------------------- measuring
def _measure_job(item: tuple[Any, ...]) -> Proof:
    config, version, name, path, text, overrides, overlays = item
    return native.measure(config, version, name, path, text, overrides, overlays)


def measure_versions(
    config: Config,
    held: Sequence[str],
    name: str,
    path: str,
    text: str,
    overrides: Json | None = None,
    overlays: Mapping[str, bytes] | None = None,
) -> tuple[Proof, ...]:
    """One proof per holding version, measured side by side."""
    jobs = [(config, v, name, path, text, overrides, overlays) for v in held]
    return tuple(pool.map(config, "compare.measure", _measure_job, jobs))


def score_functions(config: Config, path: str, names: Sequence[str], version: str) -> dict[str, float]:
    """The score of each of `names` as the module `path` of the working tree holds it (its NON_MATCHING body linked
    alone at the function's address in `version`); what the fuzzy scores of the tree are made of. The module is
    compiled once for all of them."""
    text = (config.project.root / path).read_text(encoding="utf-8")
    return {n: p.score for n, p in native.measure_names(config, version, names, path, text).items()}


def gaps(name: str, held: Sequence[str], proofs: Sequence[Proof]) -> tuple[Finding, ...]:
    found = {p.version: p for p in proofs}
    bad: list[str] = []
    lost: list[str] = []
    wrong: list[Proof] = []
    for version in held:
        proof = found.get(version)
        if proof is None:
            bad.append(version)
            lost.append(f"version {version}: no measurement")
        elif not proof.exact:
            bad.append(version)
            wrong.append(proof)
            lost.extend(proof.missing)
    if not lost:
        return ()
    facts = symptoms.merge([p.symptoms for p in wrong] + [symptoms.from_missing(lost)])
    return (
        Finding(
            "land.not_exact",
            reason=f"{name} does not match the ROM in every holding version",
            unit=name,
            versions=tuple(bad),
            missing=tuple(lost),
            symptoms=facts,
        ),
    )


# ---------------------------------------------------------------- the candidate
def _overrides(params: Json) -> dict[str, Any]:
    return {
        "add": list(params.get("flag") or ()),
        "omit": list(params.get("omit_flag") or ()),
        **({"toolchain": params["toolchain"]} if params.get("toolchain") else {}),
    }


def _source(file: Path) -> str:
    if not file.is_file():
        raise Refusal(
            Finding(
                "land.request", reason=_ROOT_ERROR.format(file), path=str(file), action="pass an existing source file"
            )
        )
    return file.read_text(encoding="utf-8")


def _extras(config: Config, specs: Sequence[Path]) -> dict[str, bytes]:
    """The change-set files `PATH=FILE`: the project path each replaces and its new text."""
    root, out = config.project.root.resolve(), {}
    for spec in specs:
        target, equals, source = str(spec).partition("=")
        path = (root / target).resolve()
        if configuration.is_source(target) and not path.is_file():
            raise Refusal(
                Finding(
                    "land.request",
                    f"{target} would be a new source; a change-set entry creates headers only.",
                    path=target,
                    action="land the source as a candidate",
                )
            )
        if not equals or not path.is_relative_to(root) or not (path.is_file() or configuration.is_header(target)):
            raise Refusal(
                Finding(
                    "land.request",
                    f"{spec} is not PATH=FILE naming a project file or a new header.",
                    path=str(spec),
                    action="pass PATH=FILE: the project file left of =, its new text right of it",
                )
            )
        if not Path(source).is_file():
            raise Refusal(
                Finding("land.request", _ROOT_ERROR.format(source) + ".", path=source, action="pass the new text")
            )
        out[path.relative_to(root).as_posix()] = Path(source).read_bytes()
    return out


def _member(captured: tree.Tree, versions_: Sequence[str], source: str, function: str | None) -> str:
    """The function a candidate implements: --function, else the first function it defines that a file owns."""
    if function:
        return function
    defined = [i.name for i in publish.items(source) if i.kind == "function"]
    for name in defined:
        if any(captured.owner(v, name) for v in versions_):
            return name
    raise Refusal(
        Finding(
            "land.request",
            reason="the source defines no function a file of the tree owns: it defines "
            + (", ".join(defined) or "no function"),
            action="define a function that has a stand-in line or pass --function NAME",
        )
    )


def _replace(current: str, name: str, source: str) -> str:
    """`current` with the definition of `name` it already holds replaced by the candidate's (a landed function)."""
    declared, body = publish.candidate(source, name)
    parts = [p for p in publish.items(current) if p.kind == "function" and p.name == name]
    if len(parts) != 1:
        raise Refusal(Finding("land.request", f"{name} has no single definition to replace.", unit=name))
    new = (declared + "\n\n" if declared and declared not in current else "") + body.strip()
    return current[: parts[0].start] + new + current[parts[0].end :]


def _text(current: str, segment: str, name: str, source: str) -> str:
    """The module with the candidate in place of the stand-in of `name`, or of its landed definition."""
    try:
        return publish.splice(current, segment, name, source, False)
    except Refusal as refusal:
        if any(f.key == "land.duplicate" for f in refusal.findings):
            return _replace(current, name, source)
        raise


def _resolve(config: Config, file: Path, function: str | None) -> tuple[str, str, str, tuple[str, ...], str, str]:
    """(name, module path, candidate source, holding versions, segment, new module text) of one candidate."""
    source = _source(file)
    captured = tree.capture(config)
    name = _member(captured, config.project.versions, source, function)
    held, path, segment = publish.holders(captured, name)
    current = (config.project.root / path).read_text(encoding="utf-8")
    return name, path, source, held, segment, _text(current, segment, name, source)


def _request(name: str, file: Path, note: str | None) -> dict[str, Any]:
    return {"file": str(file), "function": name, "overrides": {"add": [], "omit": []}, "note": note or ""}


# ---------------------------------------------------------------- compare and submit
def _attempt(config: Config, files: Sequence[Path], params: Json, origin: str, out: bool = True) -> dict[str, Any]:
    """Measure one candidate in every holding version and enter it in the inbox when it is exact (or a fuzzy gain)."""
    extras = _extras(config, files[1:])
    name, path, source, held, _, text = _resolve(config, files[0], params.get("function"))
    proofs = measure_versions(config, held, name, path, text, None, extras)
    found = list(gaps(name, held, proofs))
    entry = None
    try:
        entry = land.submit(
            config, _request(name, files[0], params.get("note")), proofs, source.encode(), origin, extras
        )
    except Refusal as refusal:
        found.extend(refusal.findings)
    drain = land.drain(config) if entry and out else None
    return {
        "unit": path,
        "member": name,
        "proofs": proofs,
        "found": found,
        "submission": entry,
        "consumers": {"count": 0, "wall": 0.0},
        "rewritten": [],
        "drain": drain,
    }


def run(config: Config, params: Json) -> Json:
    with effort.stage("compare.run"):
        done = _attempt(config, [Path(params["file"])], params, "compare")
        proofs, found, entry = done["proofs"], done["found"], done["submission"]
        score = {v: min(p.score for p in proofs if p.version == v) for v in sorted({p.version for p in proofs})}
        return {
            "unit": done["unit"],
            "members": [done["member"]],
            "exact": all(p.exact for p in proofs) and not found,
            "score": score,
            "proofs": [asdict(p) for p in proofs],
            "gaps": [asdict(f) for f in found],
            "submitted": entry.id if entry else None,
            "consumers": done["consumers"],
            "rewritten": done["rewritten"],
            "drain": done["drain"],
        }


def _row(files: Sequence[Path], done: dict[str, Any]) -> dict[str, Any]:
    proofs, entry = done["proofs"], done["submission"]
    return {
        "file": str(files[0]),
        "files": [str(f) for f in files],
        "member": done["member"],
        "id": entry.id if entry else None,
        "operation": entry.operation if entry else "none",
        "exact": all(p.exact for p in proofs),
        "score": min((p.score for p in proofs), default=0.0),
        "findings": [asdict(f) for f in done["found"]],
        "consumers": done["consumers"],
    }


def submit_command(config: Config, params: Json) -> Json:
    with effort.stage("compare.submit_command"):
        if params.get("batch_data"):
            from unbake import batch_data

            return batch_data.run(config, params)
        if not params["files"]:
            raise Refusal(Finding("land.request", "Pass source files or --batch-data MANIFEST."))
        files = [Path(f) for f in params["files"]]
        if params["bulk"]:
            return _bulk(config, files, params)
        done = _attempt(config, files, params, "submit")
        return {"submissions": [_row(files, done)], "drain": done["drain"]}


def _bulk(config: Config, files: Sequence[Path], params: Json) -> Json:
    """Each source, with the PATH=FILE files after it, is one candidate: all are measured in one dispatch, entered
    independently (a refusal is its row and blocks no other), and landed once."""
    groups: list[list[Path]] = []
    for file in files:
        if groups and "=" in str(file) and not file.exists():
            groups[-1].append(file)
        else:
            groups.append([file])
    resolved: dict[int, Any] = {}
    for n, group in enumerate(groups):
        try:
            resolved[n] = (*_resolve(config, group[0], None), _extras(config, group[1:]))
        except Refusal as refusal:
            resolved[n] = refusal
    jobs = [
        (config, v, r[0], r[1], r[5], None, r[6]) for r in resolved.values() if not isinstance(r, Refusal) for v in r[3]
    ]
    measured = iter(pool.map(config, "compare.measure", _measure_job, jobs) if jobs else [])
    rows: list[dict[str, Any]] = []
    calls: dict[int, dict[str, Any]] = {}
    proofs: dict[int, tuple[Proof, ...]] = {}
    for n, r in resolved.items():
        if isinstance(r, Refusal):
            continue
        proofs[n] = tuple(next(measured) for _ in r[3])
        calls[n] = {
            "request": _request(r[0], groups[n][0], params.get("note")),
            "proofs": proofs[n],
            "source": r[2].encode(),
            "origin": "submit",
            "extras": r[6],
        }
    entered = dict(zip(calls, land.submit_many(config, list(calls.values())), strict=True))
    for n, group in enumerate(groups):
        r = resolved[n]
        row: dict[str, Any] = {
            "file": str(group[0]),
            "files": [str(f) for f in group],
            "member": group[0].stem,
            "id": None,
            "operation": "none",
            "exact": False,
            "score": 0.0,
            "findings": [],
            "consumers": {"count": 0, "wall": 0.0},
        }
        if isinstance(r, Refusal):
            row.update(findings=[asdict(f) for f in r.findings], refused=True)
        else:
            result = entered[n]
            found = list(gaps(r[0], r[3], proofs[n]))
            if isinstance(result, Refusal):
                found.extend(result.findings)
            entry = None if isinstance(result, Refusal) else result
            row.update(
                member=r[0],
                id=entry.id if entry else None,
                operation=entry.operation if entry else "none",
                exact=all(p.exact for p in proofs[n]),
                score=min((p.score for p in proofs[n]), default=0.0),
                findings=[asdict(f) for f in found],
                refused=bool(found),
            )
        rows.append(row)
    drain = land.drain(config) if any(row["id"] for row in rows) else None
    return {"submissions": rows, "drain": drain}


# ---------------------------------------------------------------- score
def _compact(diff: Sequence[str], limit: int) -> list[str]:
    """Every hunk of a unified diff, reduced to its header and changed lines, in at most `limit` lines."""
    rows = [line for line in diff if line.startswith(("@@", "-", "+")) and not line.startswith(("---", "+++"))]
    return rows[:limit]


def score_command(config: Config, params: Json) -> Json:
    """Score candidates in one version and write nothing (no journal, inbox, landing or artifacts): the ordered
    match against the ROM with the differing lines, where they part (diagnostic), the source rules broken, and with
    --alloc the registers and the instruction order. --variants scores a JSON list of {file, overlays} in one process,
    side by side."""
    with effort.stage("compare.score"):
        if params.get("variants"):
            return _score_variants(config, params)
        if not params.get("file"):
            raise Refusal(Finding("land.request", reason="score needs a C source", action="pass FILE or --variants"))
        return _score_one(config, params)


def _variant_params(params: Json, row: Any) -> Json:
    if not isinstance(row, dict) or not isinstance(row.get("file"), str) or not row["file"]:
        raise Refusal(
            Finding(
                "land.request",
                reason=f"--variants rows must be objects with a nonempty file path, got {str(row)[:80]}",
                action="pass a JSON list of {file, overlays}",
            )
        )
    unknown = sorted(set(row) - {"file", "overlays", "flags", "omit_flags", "function"})
    if unknown:  # a misspelt key would score the variant without what it names
        raise Refusal(
            Finding(
                "land.request",
                reason=f"--variants rows take file, overlays, flags, omit_flags, function: not {', '.join(unknown)}",
                action="pass a JSON list of {file, overlays, flags, omit_flags, function}",
            )
        )
    for key in ("flags", "omit_flags"):  # a row's own flag set: one call scores a whole flag lattice
        value = row.get(key)
        if value is not None and (not isinstance(value, list) or any(not isinstance(f, str) or not f for f in value)):
            raise Refusal(
                Finding(
                    "land.request",
                    reason=f"--variants {key} must be a list of nonempty compiler options or null",
                    action='pass e.g. "flags": ["-ffast-math"]',
                )
            )
    if row.get("function") is not None and (not isinstance(row["function"], str) or not row["function"]):
        raise Refusal(Finding("land.request", reason="--variants function must be a member name or null"))
    overlays = row.get("overlays")
    if overlays is not None and (
        not isinstance(overlays, list) or any(not isinstance(o, str) or not o for o in overlays)
    ):
        raise Refusal(
            Finding(
                "land.request",
                reason="--variants overlays must be a list of nonempty path strings or null",
                action="pass overlays as a JSON list of file paths or PATH=FILE strings",
            )
        )
    own = {
        **({"flag": list(row["flags"])} if row.get("flags") is not None else {}),
        **({"omit_flag": list(row["omit_flags"])} if row.get("omit_flags") is not None else {}),
        **({"function": row["function"]} if row.get("function") is not None else {}),
    }
    return {**params, **own, "file": Path(row["file"]), "extras": [Path(o) for o in overlays or ()]}


def _variant_job(item: tuple[Config, Json]) -> Json:
    config, params = item
    flags = {"flags": list(params.get("flag") or ()), "omit_flags": list(params.get("omit_flag") or ())}
    try:
        result = _score_one(config, params)
    except Refusal as refusal:
        return {"file": str(params["file"]), **flags, "ok": False, "findings": [asdict(f) for f in refusal.findings]}
    return {
        "file": str(params["file"]),
        **flags,
        "ok": True,
        "result": {k: v for k, v in result.items() if k != "proofs"},
    }


def _score_variants(config: Config, params: Json) -> Json:
    """Validate the whole manifest first, then dispatch each variant as one complete score."""
    manifest = Path(params["variants"])
    try:
        rows = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Refusal(
            Finding(
                "land.request",
                reason=f"--variants is not JSON at line {error.lineno}, column {error.colno}: {error.msg}",
                path=str(manifest),
                line=error.lineno,
                action="fix the JSON in the variants input file",
            )
        ) from error
    except (OSError, UnicodeDecodeError) as error:
        raise Refusal(
            Finding(
                "land.request",
                reason=f"cannot read --variants: {error}",
                path=str(manifest),
                action="pass a readable UTF-8 JSON file",
            )
        ) from error
    if not isinstance(rows, list):
        raise Refusal(
            Finding(
                "land.request",
                reason="--variants must hold a JSON list",
                path=str(manifest),
                action="pass {file, overlays} rows",
            )
        )
    variants = []
    for index, row in enumerate(rows, 1):
        try:
            variants.append(_variant_params(params, row))
        except Refusal as error:
            raise Refusal(
                *(replace(f, path=str(manifest), reason=f"row {index}: {f.reason}") for f in error.findings)
            ) from error
    return {"variants": pool.map(config, "compare.variants", _variant_job, [(config, v) for v in variants])}


def _score_one(config: Config, params: Json) -> Json:
    name, path, _, held, _, text = _resolve(config, Path(params["file"]), params.get("function"))
    extras = _extras(config, [Path(s) for s in params.get("extras") or ()])
    overrides = _overrides(params)
    preferred = config.project.names_from
    version = params.get("version") or (preferred if preferred in held else held[0])
    if version not in held:
        raise Refusal(
            Finding(
                "land.request",
                reason=f"{name} is not held by version {version}",
                versions=held,
                action="pass --version one of the holding versions",
            )
        )
    proof = native.measure(config, version, name, path, text, overrides, extras)
    proofs = [native.measure(config, v, name, path, text, overrides, extras) for v in held if v != version]
    proofs.insert(0, proof)
    facts = native.symbol_facts(config, version, extras)[name]
    built = store.get(config, "bytes", proof.built_sha256) or b""
    size = len(built) or (facts[2] or 0)
    target = (
        versions.rom_bytes(native.version_records(config, extras)[version], facts[1] or 0, (facts[1] or 0) + size)
        if size
        else b""
    )
    if not built and not proof.exact:
        reason = "; ".join(ln for m in proof.missing for ln in m.splitlines() if ln.strip())
        where = _LOCATION.search(reason)  # the first place the compiler names; a conflict's other one is in the reason
        raise Refusal(
            Finding(
                "score.compile",
                reason=reason or "no instructions were built",
                path=where[1] if where else "",
                line=int(where[2]) if where else 0,
                unit=path,
                versions=(version,),
                action="fix the compile error shown",
            )
        )
    everything = symptoms.diff(built, target, facts[0]).splitlines() if built else []
    limit = params.get("lines") or 80
    shown = _compact(everything, params.get("lines") or 40) if params.get("short") else everything[:limit]
    matched = round(symptoms.score(built, target) * len(target) / 4) if built and target else 0
    snapshot, unit, recipe, _ = native.candidate(config, version, path, text, overrides, extras)
    source_findings = policy.evaluate(snapshot, path, text, None, False)
    result: dict[str, Any] = {
        "unit": path,
        "members": [name],
        "version": version,
        "versions": list(held),
        "exact": all(p.exact for p in proofs),
        "score": proof.score,
        "words": {"built": len(built) // 4, "target": len(target) // 4, "matched": matched},
        "shift": symptoms.shift(built, target),
        "source": [asdict(f) for f in source_findings],
        "diff": shown,
        "proofs": [asdict(p) for p in proofs],
        "rewritten": [],
    }
    if params.get("alloc"):
        with store.work(config) as work:
            greg, lreg, asm, passes = native.dumps(snapshot, unit, version, recipe, work / "alloc")
        result["alloc"] = alloc.report(name, greg, lreg, asm, symptoms.register_votes(built, target))
        result["sched"] = sched.report(name, passes, symptoms.listing(built, facts[0]), everything)
    return result
