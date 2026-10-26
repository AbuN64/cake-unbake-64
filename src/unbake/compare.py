"""Measure a candidate function in every version that holds it and report the gaps against the ROMs."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from unbake import (
    alloc,
    api,
    effort,
    land,
    native,
    permute,
    policy,
    pool,
    publish,
    sched,
    store,
    symbols,
    symptoms,
    tree,
    versions,
    view,
)
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Proof, Refusal

_LOCATION = re.compile(r"([\w./-]+\.\w+):(\d+): (?!previous)")  # PATH:LINE: of a diagnostic, not of its note
_ROOT_ERROR = "the file {} does not exist"


# ---------------------------------------------------------------- measuring
def _measure_job(item: tuple[Any, ...]) -> Proof:
    config, version, name, path, text, overrides, overlays = item
    return native.measure(config, version, name, path, text, overrides, overlays)


@dataclass(frozen=True)
class Placed:
    """The module text one owner of the name is measured as: the candidate spliced in, and where each line of it came
    from (a line of the file on disk, or of the candidate)."""

    owner: publish.Owner
    text: str
    origins: tuple[publish.Origin, ...]


@dataclass(frozen=True)
class Resolved:
    """A candidate placed in every file that owns its function: one text per owner, measured in its versions."""

    name: str
    file: Path
    source: str
    placed: tuple[Placed, ...]

    @property
    def held(self) -> tuple[str, ...]:
        return tuple(sorted(v for p in self.placed for v in p.owner.versions))

    def at(self, version: str) -> Placed:
        return next(p for p in self.placed if version in p.owner.versions)


_SITE = re.compile(r"([\w./-]+\.\w+):(\d+)")


def _sited(proof: Proof, resolved: Resolved) -> Proof:
    """The proof with every line the compiler names in the module's new text said where a reader can open it: a line
    of the candidate is `CANDIDATE:LINE`, a line of the file the file's own line. The new text is not on disk, so its
    line numbers mean nothing to anyone else (every line after the candidate is shifted by what was added)."""
    placed = resolved.at(proof.version)

    def fix(match: re.Match[str]) -> str:
        line = int(match[2])
        if match[1] != placed.owner.path or not 0 < line <= len(placed.origins):
            return match[0]
        kind, number = placed.origins[line - 1]
        return f"{resolved.file}:{number}" if kind == "candidate" else f"{match[1]}:{number}"

    return replace(proof, missing=tuple(_SITE.sub(fix, m) for m in proof.missing))


def _siter(config: Config, placed: Placed, resolved: Resolved) -> Callable[[int], str]:
    """A line of the compiled module as `file:line  text`: of the candidate, or of the file it was spliced into."""
    cache: dict[str, list[str]] = {}

    def lines(path: str) -> list[str]:
        if path not in cache:
            source = resolved.source if Path(path) == resolved.file else None
            try:
                cache[path] = (source if source is not None else (config.project.root / path).read_text("utf-8")).split(
                    "\n"
                )
            except OSError:
                cache[path] = []
        return cache[path]

    def site(line: int) -> str:
        if not 0 < line <= len(placed.origins):
            return ""
        kind, number = placed.origins[line - 1]
        path = str(resolved.file) if kind == "candidate" else placed.owner.path
        rows = lines(str(resolved.file) if kind == "candidate" else placed.owner.path)
        text = rows[number - 1].strip() if 0 < number <= len(rows) else ""
        return f"{path}:{number}  `{text[:100]}`"

    return site


def _measure(
    config: Config, resolved: Resolved, overrides: Json | None = None, overlays: Mapping[str, bytes] | None = None
) -> tuple[Proof, ...]:
    """One proof per holding version, each measured as the text of its own owner, side by side."""
    jobs = [
        (config, v, resolved.name, p.owner.path, p.text, overrides, overlays)
        for p in resolved.placed
        for v in p.owner.versions
    ]
    return tuple(_sited(proof, resolved) for proof in pool.map(config, "compare.measure", _measure_job, jobs))


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


def _extras(config: Config, specs: Sequence[Path], file: Path | None = None) -> dict[str, bytes]:
    """The change-set files `PATH=FILE`: the project path each replaces and its new text. A candidate `X.c` brings the
    files under `X.overlay/` (each at its project path) as well: the headers and rows it was proved with."""
    root, out = config.project.root.resolve(), {}
    beside = file.with_suffix(".overlay") if file else None
    companions = (
        [Path(f"{p.relative_to(beside)}={p}") for p in sorted(beside.rglob("*")) if p.is_file()] if beside else []
    )
    for spec in [*specs, *companions]:
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


def _member(
    captured: tree.Tree,
    versions_: Sequence[str],
    source: str,
    function: str | None,
    read: Callable[[str], str | None] = lambda path: None,
) -> str:
    """The function a candidate implements: --function, else the one function it defines that a file owns, or of
    several the one whose text differs from the owning file's (`read` gives a file's current text). A candidate that
    changes two or more owned functions is refused: one number would hide the others (10-20: the first function
    scored exact while the edited one was wrong)."""
    if function:
        return function
    defined = [(i.name, source[i.start : i.end]) for i in publish.items(source) if i.kind == "function"]
    owned = [(name, body) for name, body in defined if any(captured.owner(v, name) for v in versions_)]
    if not owned:
        raise Refusal(
            Finding(
                "land.request",
                reason="the source defines no function a file of the tree owns: it defines "
                + (", ".join(name for name, _ in defined) or "no function"),
                action="define a function that has a stand-in line or pass --function NAME",
            )
        )
    if len(owned) == 1:
        return owned[0][0]
    changed = []
    for name, body in owned:
        owners = {captured.owner(v, name) for v in versions_} - {None}
        held = [read(path) for path in sorted(p for p in owners if p)]
        bodies = {
            t[i.start : i.end] for t in held if t for i in publish.items(t) if i.kind == "function" and i.name == name
        }
        if " ".join(body.split()) not in {" ".join(b.split()) for b in bodies}:
            changed.append(name)
    if len(changed) > 1:
        raise Refusal(
            Finding(
                "land.request",
                reason=f"the candidate changes {len(changed)} owned functions ({', '.join(changed)}): one score "
                "would stand for all of them",
                action="pass --function NAME, once per function",
            )
        )
    return changed[0] if changed else owned[0][0]


def _text(current: str, segment: str, name: str, source: str) -> tuple[str, list[publish.Origin]]:
    """The module with the candidate in place of the stand-in of `name`, or of its landed definition, and where each
    of its lines came from. A stand-in that only builds without NON_MATCHING (`#ifndef NON_MATCHING`), whose draft
    the file holds in a NON_MATCHING block of its own, is not where a measure reads the candidate: the candidate
    takes the draft's place, among the types and declarations the draft is written against."""
    if publish.per_version_copies(source, name):  # one copy per version guard, all in the file: the file is the unit
        return publish.whole_traced(source)
    if publish.whole_unit(source, name):  # a vendored library unit: the file is the unit, the member is found by symbol
        return publish.whole_traced(source)
    if publish.measured_apart(current, segment, name):
        return publish.replace_traced(current, name, source)
    try:
        return publish.splice_traced(current, segment, name, source, False)
    except Refusal as refusal:
        if any(f.key == "land.duplicate" for f in refusal.findings):
            return publish.replace_traced(current, name, source)
        raise


def _resolve(
    config: Config,
    file: Path,
    function: str | None,
    version: str | None = None,
    extras: Mapping[str, bytes] | None = None,
) -> Resolved:
    """One candidate placed in every file that owns its function (a named version selects the file that owns it
    there); a change-set file that replaces an owner is the text the candidate is spliced into."""
    source = _source(file)
    captured = tree.capture(config)

    def read(path: str) -> str | None:
        given = (extras or {}).get(path)
        if given is not None:
            return given.decode()
        try:
            return (config.project.root / path).read_text(encoding="utf-8")
        except OSError:
            return None

    name = _member(captured, config.project.versions, source, function, read)
    placed = []
    for owner in publish.owners(captured, name, version):
        given = (extras or {}).get(owner.path)
        current = (
            given.decode() if given is not None else (config.project.root / owner.path).read_text(encoding="utf-8")
        )
        text, origins = _text(current, owner.segment, name, source)
        placed.append(Placed(owner, text, tuple(origins)))
    return Resolved(name, file, source, tuple(placed))


def _request(name: str, file: Path, note: str | None) -> dict[str, Any]:
    return {"file": str(file), "function": name, "overrides": {"add": [], "omit": []}, "note": note or ""}


# ---------------------------------------------------------------- compare and submit
def _attempt(config: Config, files: Sequence[Path], params: Json, origin: str, enter: bool) -> dict[str, Any]:
    """Measure one candidate in every holding version; with `enter`, put it in the inbox when it is exact (or a fuzzy
    gain) and land the inbox. Without it nothing is submitted, journalled or written."""
    extras = _extras(config, files[1:], files[0])
    resolved = _resolve(config, files[0], params.get("function"), None, extras)
    name, source, held = resolved.name, resolved.source, resolved.held
    proofs = _measure(config, resolved, None, extras)
    found = list(gaps(name, held, proofs))
    entry = None
    if enter:
        try:
            entry = land.submit(
                config, _request(name, files[0], params.get("note")), proofs, source.encode(), origin, extras
            )
        except Refusal as refusal:
            found.extend(refusal.findings)
    drain = land.drain(config, entry.id) if entry else None
    return {
        "unit": ", ".join(p.owner.path for p in resolved.placed),
        "member": name,
        "proofs": proofs,
        "found": found,
        "submission": entry,
        "consumers": {"count": 0, "wall": 0.0},
        "rewritten": [],
        "drain": drain,
    }


def run(config: Config, params: Json) -> Json:
    if params.get("widen"):
        from unbake import widen

        with effort.stage("compare.widen"):
            return widen.run(config, params)
    with effort.stage("compare.run"):
        done = _attempt(config, [Path(params["file"])], params, "compare", False)
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
        if params.get("boundary"):
            from unbake import boundary

            return boundary.run(config, params)
        if params.get("pad_split"):
            from unbake import padsplit

            return padsplit.run(config, params)
        if not params["files"]:
            raise Refusal(Finding("land.request", "Pass source files or --batch-data MANIFEST."))
        files = [Path(f) for f in params["files"]]
        if params["bulk"]:
            return _bulk(config, files, params)
        done = _attempt(config, files, params, "submit", True)
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
            given = _extras(config, group[1:], group[0])
            resolved[n] = (_resolve(config, group[0], None, None, given), given)
        except Refusal as refusal:
            resolved[n] = refusal
    jobs = [
        (config, v, r[0].name, p.owner.path, p.text, None, r[1])
        for r in resolved.values()
        if not isinstance(r, Refusal)
        for p in r[0].placed
        for v in p.owner.versions
    ]
    measured = iter(pool.map(config, "compare.measure", _measure_job, jobs) if jobs else [])
    rows: list[dict[str, Any]] = []
    calls: dict[int, dict[str, Any]] = {}
    proofs: dict[int, tuple[Proof, ...]] = {}
    for n, r in resolved.items():
        if isinstance(r, Refusal):
            continue
        proofs[n] = tuple(_sited(next(measured), r[0]) for _ in r[0].held)
        calls[n] = {
            "request": _request(r[0].name, groups[n][0], params.get("note")),
            "proofs": proofs[n],
            "source": r[0].source.encode(),
            "origin": "submit",
            "extras": r[1],
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
            found = list(gaps(r[0].name, r[0].held, proofs[n]))
            if isinstance(result, Refusal):
                found.extend(result.findings)
            entry = None if isinstance(result, Refusal) else result
            row.update(
                member=r[0].name,
                id=entry.id if entry else None,
                operation=entry.operation if entry else "none",
                exact=all(p.exact for p in proofs[n]),
                score=min((p.score for p in proofs[n]), default=0.0),
                findings=[asdict(f) for f in found],
                refused=bool(found),
            )
        rows.append(row)
    ids = {row["id"] for row in rows if row["id"]}
    drain = land.drain(config, ids) if ids else None
    return {"submissions": rows, "drain": drain}


# ---------------------------------------------------------------- score
def _compact(diff: Sequence[str], limit: int) -> list[str]:
    """Every hunk of a unified diff, reduced to its header and changed lines, in at most `limit` lines."""
    rows = [line for line in diff if line.startswith(("@@", "-", "+")) and not line.startswith(("---", "+++"))]
    return rows[:limit]


def score_command(config: Config, params: Json) -> Json:
    """Score candidates in one version and write nothing (no journal, inbox, landing or artifacts): the ordered
    match against the ROM with the differing lines, where they part (diagnostic), the source rules broken, and with
    --alloc the registers (a table of every pseudo with its priority, the target's register and who took it) and the
    instruction order; --quick drops the source rules to answer in seconds. --variants scores a JSON list of
    {file, overlays} in one process, side by side."""
    with effort.stage("compare.score"):
        if params.get("variants"):
            return _score_variants(config, params)
        if not params.get("file"):
            raise Refusal(Finding("land.request", reason="score needs a C source", action="pass FILE or --variants"))
        if params.get("permute"):
            return _permute(config, params)
        return _score_one(config, params)


def _permute(config: Config, params: Json) -> Json:
    """score --permute SECONDS: the permuter on the candidate's function in one version, its best body spliced into a
    copy of the candidate file (--out, default FILE.permuted.c) that `recipe try` takes; scores are the permuter's."""
    extras = _extras(
        config, [Path(s) for s in params.get("extras") or ()], Path(params["file"]) if params.get("file") else None
    )
    file = Path(params["file"])
    resolved = _resolve(config, file, params.get("function"), params.get("version"), extras)
    held, preferred = resolved.held, config.project.names_from
    version = params.get("version") or (preferred if preferred in held else held[0])
    ctx = api.build_context(config, resolved.name, version, resolved.at(version).text.encode())
    with store.work(config) as work:
        found = permute.search(config, resolved.name, ctx, params, work)
    given = file.read_text(encoding="utf-8", errors="replace")
    text = permute.candidate(given, resolved.name, found) if permute.span(given, resolved.name) else None
    out = Path(params.get("out") or f"{file}.permuted.c")
    if text is not None:
        store.write(out, text.encode())
    return {"permute": {**found, "version": version, "member": resolved.name, "candidate": str(out) if text else ""}}


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


_MAX_VARIANTS = 50  # a variant costs about 5 CPU s (a compile of its module): 50 stay under the 5 CPU-minute cap


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
    if len(rows) > _MAX_VARIANTS:
        raise Refusal(
            Finding(
                "land.request",
                reason=f"--variants holds {len(rows)} rows: at about 5 CPU s each, over {_MAX_VARIANTS} passes the cap",
                path=str(manifest),
                action=f"split the list into runs of at most {_MAX_VARIANTS} rows",
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


def _inactive(
    config: Config, version: str, name: str, placed: Placed, overrides: Json, extras: Mapping[str, bytes]
) -> str:
    """Why the candidate's definition of `name` is not compiled in `version` (the reason; "" when it is). A stand-in
    can sit in a version guard of its module (`#if VERSION_EU`) while another version defines the function in an
    `#else` side: the measure then reads that other definition, not the candidate. The line judged is the one that
    names the function in each top-level definition the candidate brings (a comment or directive above it is no part
    of it); one active definition is enough, as when the candidate defines it once per version guard."""
    text, called = placed.text, re.compile(rf"\b{re.escape(name)}\s*\(")
    heads = []
    for item in publish.items(text):
        if item.kind == "function" and item.name == name and (at := called.search(text, item.start, item.end)):
            number = text.count("\n", 0, at.start()) + 1
            if number <= len(placed.origins) and placed.origins[number - 1][0] == "candidate":
                heads.append(number)
    if not heads:
        return ""
    snapshot, unit, recipe, _ = native.candidate(
        config, version, placed.owner.path, placed.text, overrides, extras, [name]
    )
    active = view.active_lines(view.get(snapshot, unit, version, recipe), placed.owner.path)
    if any(number in active for number in heads):
        return ""
    return (
        f"version {version}: the candidate's {name} (line {heads[0]} of {placed.owner.path} as measured) sits in an "
        "inactive #if region there, so the module's own definition is what a measure would read"
    )


_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_NOT_CALLS = {"if", "while", "for", "switch", "return", "sizeof", "do", "else", "case", "goto"}


def _implicit_calls(text: str, name: str) -> list[str]:
    """The names the preprocessed unit calls in `name`'s body and never spells anywhere else: no prototype, no
    definition, no variable. C compiles such a call as an implicit `int f()`, which reorders registers and drops a
    float return, so a measure of it proves nothing about the function as the project declares it (THINK-80289970:
    a splice lost DirectionDot)."""
    bodies = [i for i in publish.items(text) if i.kind == "function" and i.name == name]
    found: list[str] = []
    for body in bodies:
        head = text.find("{", body.start, body.end)
        inside = text[head : body.end] if head >= 0 else ""
        for called in dict.fromkeys(m[1] for m in _CALL.finditer(inside)):
            if called in _NOT_CALLS or called in found:
                continue
            spelled = len(re.findall(rf"\b{re.escape(called)}\b", text))
            if spelled == len(re.findall(rf"\b{re.escape(called)}\s*\(", inside)):
                found.append(called)
    return found


def _words(built: bytes, target: bytes) -> dict[str, int]:
    matched = round(symptoms.score(built, target) * len(target) / 4) if built and target else 0
    return {"built": len(built) // 4, "target": len(target) // 4, "matched": matched}


def _verdict(
    config: Config, version: str, name: str, path: str, proof: Proof, extras: Mapping[str, bytes], params: Json
) -> tuple[list[str], bytes, bytes, list[str]]:
    """(the diff lines shown, the bytes built, the ROM's bytes of the function, the whole diff) of one measured proof:
    what score and probe both print, so a probe agrees with a score. A proof that built nothing is its compile error."""
    table = native.symbol_facts(config, version, extras)
    facts = table[name]
    built = store.get(config, "bytes", proof.built_sha256) or b""
    size = native.extent(table, name) or len(built)  # not the candidate's size
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
    return shown, built, target, everything


def _score_one(config: Config, params: Json) -> Json:
    extras = _extras(
        config, [Path(s) for s in params.get("extras") or ()], Path(params["file"]) if params.get("file") else None
    )
    resolved = _resolve(config, Path(params["file"]), params.get("function"), params.get("version"), extras)
    name, held = resolved.name, resolved.held
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

    def measured(v: str) -> Proof:
        placed = resolved.at(v)
        proof = _sited(native.measure(config, v, name, placed.owner.path, placed.text, overrides, extras), resolved)
        if not (why := _inactive(config, v, name, placed, overrides, extras)):
            return proof
        if v == version:  # the score asked for is of something else than the candidate: no number is better than that
            raise Refusal(
                Finding(
                    "score.inactive",
                    reason=why,
                    unit=placed.owner.path,
                    versions=(v,),
                    action="put the candidate where this version compiles it, or score another version",
                )
            )
        return replace(proof, exact=False, score=0.0, missing=(why,), symptoms=symptoms.from_missing((why,)))

    order = [version] if params.get("version") else [version, *(v for v in held if v != version)]
    for version in order:
        try:  # an unasked-for version where the candidate sits in an inactive #if: the first version it is active in
            proof = measured(version)
        except Refusal as inactive:
            if params.get("version") or inactive.findings[0].key != "score.inactive" or version == held[-1]:
                raise
            continue
        break
    path, text = resolved.at(version).owner.path, resolved.at(version).text
    # --quick --version V (a search loop: one candidate after another): V alone is measured, not every version that
    # holds the function (5 compiles a variant), and `exact` speaks for V (A16, 10-25)
    alone = bool(params.get("quick") and params.get("version"))
    proofs = [] if alone else [measured(v) for v in held if v != version]
    proofs.insert(0, proof)
    shown, built, target, everything = _verdict(config, version, name, path, proof, extras, params)
    snapshot, unit, recipe, _ = native.candidate(config, version, path, text, overrides, extras, [name])
    source_findings = [] if params.get("quick") else policy.evaluate(snapshot, path, text, None, False)
    result: dict[str, Any] = {
        "unit": path,
        "members": [name],
        "version": version,
        "versions": list(held),
        "exact": all(p.exact for p in proofs),
        "score": proof.score,
        "words": _words(built, target),
        "shift": {
            **symptoms.shift(built, target),
            "frame": symptoms.frame_hint(built, target),
            "folded": symptoms.folded_hint(built, target),
        },
        "source": [asdict(f) for f in source_findings],
        "diff": shown,
        "proofs": [asdict(p) for p in proofs],
        "rewritten": [],
    }
    if result["exact"]:
        placed = resolved.at(version)
        snap, unit, recipe, _ = native.candidate(
            config, version, placed.owner.path, placed.text, overrides, extras, [name]
        )
        if undeclared := _implicit_calls(view.get(snap, unit, version, recipe, lines=False).text, name):
            raise Refusal(
                Finding(
                    "score.implicit",
                    f"{name} calls {', '.join(undeclared)} with no declaration in {path}: the measure compiled an "
                    "implicit int function, so its exact score proves nothing about the function as declared",
                    path=path,
                    unit=name,
                    versions=(version,),
                    action=f"declare {undeclared[0]} (include its header or the helper the splice dropped)",
                )
            )
    given = Path(params["file"]).read_text(encoding="utf-8", errors="replace")  # as the lane wrote it, not spliced
    if result["exact"] and publish.non_matching_only(given, name):
        result["notes"] = [
            asdict(
                Finding(
                    "score.non-matching-only",
                    f"exact but inactive: {name} is defined only inside a NON_MATCHING side of {path}, so the build "
                    "still compiles its stand-in and this match ships nothing until it lands",
                    path=path,
                    unit=name,
                    action=f"unbake submit the candidate: land puts {name} where the build compiles it",
                    blocking=False,
                )
            )
        ]
    if params.get("alloc") or params.get("keep_dumps"):
        with store.work(config) as work:
            greg, lreg, asm, passes = native.dumps(snapshot, unit, version, recipe, work / "alloc")
            if params.get("keep_dumps"):  # the scratch goes when the command ends; the lane reads its copy
                kept = Path(params["keep_dumps"]).resolve()
                shutil.copytree(work / "alloc", kept, dirs_exist_ok=True)
                result["dumps"] = sorted(str(kept / f.name) for f in (work / "alloc").iterdir())
        site = _siter(config, resolved.at(version), resolved)
        result["alloc"] = alloc.report(name, greg, lreg, asm, symptoms.register_votes(built, target), site)
        result["rotations"] = alloc.rotations(result["alloc"], site)
        named, pairs = symptoms.operand_pairs(built, target)
        final = passes.get("sched2") or greg
        result["pseudos"] = alloc.table(
            name,
            greg,
            lreg,
            alloc.pseudo_votes(name, lreg, final, alloc.placement(greg, lreg, name), named, pairs),
            site,
        )
        vram = native.symbol_facts(config, version, extras)[name][0]
        result["sched"] = sched.report(name, passes, symptoms.listing(built, vram), everything)
    return result


# ---------------------------------------------------------------- probe
def _quick_owner(config: Config, version: str, name: str, extras: Mapping[str, bytes]) -> publish.Owner | None:
    """The file that owns `name` in `version`, found without capturing the tree: the sources that spell the name,
    each parsed alone in that version (the parse is cached per file). A C body wins over a stand-in elsewhere, as the
    tree decides; None when no file or more than one claims it (the caller asks the whole tree, which names why)."""
    fact = native.symbol_facts(config, version, extras).get(name)
    if fact is None or fact[1] is None:
        return None
    root, word = config.project.root, name.encode()
    defined: list[str] = []
    standing: list[str] = []
    for folder, _, names in os.walk(root / "src"):
        for file in names:
            if not file.lower().endswith(tree.SOURCE_SUFFIXES):
                continue
            path = (Path(folder) / file).relative_to(root).as_posix()
            data = extras.get(path) or (root / path).read_bytes()
            if word not in data:
                continue
            defs, ins, _ = tree.parse(config, path, data, version, None, extras)
            (defined if name in defs else standing if name in ins else []).append(path)
    found = defined or standing
    segment = symbols.segment_at(config, version, fact[1])
    if len(found) != 1 or segment is None:
        return None
    return publish.Owner(found[0], segment, (version,))


def _probe_dumps(
    config: Config, version: str, name: str, placed: Placed, params: Json, extras: Mapping[str, bytes], everything: str
) -> list[str]:
    """Every pass dump of the probed compile (-da where the toolchain takes it, else the allocation and scheduler
    dumps), the -g assembly, the preprocessed unit and the instruction diff, copied into --dumps DIR."""
    kept = Path(params["dumps"]).resolve()
    overrides = _overrides(params)
    every = {**overrides, "add": [*overrides["add"], "-da"]}
    path, text = placed.owner.path, placed.text
    try:
        snapshot, unit, recipe, _ = native.candidate(config, version, path, text, every, extras, [name])
    except Refusal:  # a toolchain without -da: the dumps native.dumps always asks for
        snapshot, unit, recipe, _ = native.candidate(config, version, path, text, overrides, extras, [name])
    with store.work(config) as work:
        native.dumps(snapshot, unit, version, recipe, work / "dumps")
        shutil.copytree(work / "dumps", kept, dirs_exist_ok=True)
        files = sorted(f.name for f in (work / "dumps").iterdir())
    (kept / f"{name}.diff").write_text(everything + "\n", encoding="utf-8")
    return [str(kept / f) for f in [*files, f"{name}.diff"]]


def probe_placement(config: Config, params: Json) -> tuple[dict[str, bytes], str, str, Placed, Resolved]:
    """(the change-set files, the version, the function, the module text the candidate is measured as, the candidate
    placed in it) of a probe-style command: the one file that owns the function, found without the tree capture."""
    extras = _extras(
        config, [Path(s) for s in params.get("extras") or ()], Path(params["file"]) if params.get("file") else None
    )
    file = Path(params["file"])
    source = _source(file)
    version = params.get("version") or config.project.names_from
    if version not in config.project.versions:
        raise Refusal(
            Finding(
                "land.request",
                reason=f"{version} is not a version of the project",
                versions=tuple(config.project.versions),
                action="pass --version one of the project's versions",
            )
        )
    name = params.get("function") or next((i.name for i in publish.items(source) if i.kind == "function"), "")
    owner = _quick_owner(config, version, name, extras) if name else None
    if owner is None:  # the whole tree names the owner, or why there is none
        resolved = _resolve(config, file, params.get("function"), version, extras)
        name, owner = resolved.name, next(p.owner for p in resolved.placed if version in p.owner.versions)
    given = extras.get(owner.path)
    current = given.decode() if given is not None else (config.project.root / owner.path).read_text("utf-8")
    text, origins = _text(current, owner.segment, name, source)
    placed = Placed(publish.Owner(owner.path, owner.segment, (version,)), text, tuple(origins))
    return extras, version, name, placed, Resolved(name, file, source, (placed,))


def probe_overrides(params: Json) -> dict[str, Any]:
    """The toolchain and flag overrides a probe-style command's options name."""
    return _overrides(params)


def probe_diff(
    config: Config, version: str, name: str, path: str, proof: Proof, extras: Mapping[str, bytes], params: Json
) -> tuple[list[str], dict[str, int]]:
    """(the instruction diff lines shown, the word counts) of one measured proof, as probe prints them."""
    shown, built, target, _ = _verdict(config, version, name, path, proof, extras, params)
    return shown, _words(built, target)


def probe_command(config: Config, params: Json) -> Json:
    """Compile only the module that owns the function, the candidate in place of its stand-in, in one version, and
    diff that function against the ROM: the measure score makes, without the tree capture, the source rules, the
    other versions or the landing checks. Writes nothing but --dumps DIR."""
    with effort.stage("compare.probe"):
        extras, version, name, placed, resolved = probe_placement(config, params)
        owner, text = placed.owner, placed.text
        proof = _sited(native.measure(config, version, name, owner.path, text, _overrides(params), extras), resolved)
        shown, built, target, everything = _verdict(config, version, name, owner.path, proof, extras, params)
        result: dict[str, Any] = {
            "unit": owner.path,
            "members": [name],
            "version": version,
            "exact": proof.exact,
            "score": proof.score,
            "words": _words(built, target),
            "shift": symptoms.shift(built, target),
            "diff": shown,
            "missing": list(proof.missing),
        }
        if params.get("dumps"):
            result["dumps"] = _probe_dumps(config, version, name, placed, params, extras, "\n".join(everything))
        return result
