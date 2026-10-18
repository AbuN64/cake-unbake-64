"""Landing: a candidate replaces its stand-in line in the file that owns the name, and the builds must still match."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from unbake import api, effort, journal, native, policy, process, report, tree
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Plan, Proof, Receipt, Refusal, Submission, digest, sha1, sha256

_KINDS = "INCLUDE_(?:ASM|DATA|RODATA)"


# ---------------------------------------------------------------- the C scanner (top-level items only)
@dataclass(frozen=True)
class Item:
    start: int
    end: int
    kind: str  # "directive" | "comment" | "function" | "decl"
    name: str = ""  # a function's name


def _skip(text: str, i: int) -> int:
    """The index after the comment, string or character literal that starts at i; i itself when none does."""
    if text.startswith("/*", i):
        j = text.find("*/", i + 2)
        return len(text) if j < 0 else j + 2
    if text.startswith("//", i):
        j = text.find("\n", i)
        return len(text) if j < 0 else j
    if text[i] in "\"'":
        j = i + 1
        while j < len(text) and text[j] != text[i]:
            j += 2 if text[j] == "\\" else 1
        return min(j + 1, len(text))
    return i


def _function(header: str) -> str | None:
    """The name of the function a declaration head defines, or None when it is not a function definition."""
    bare = re.sub(r"/\*.*?\*/|//[^\n]*", "", header, flags=re.S).strip()
    if "=" in bare or bare.startswith("typedef") or not bare.endswith(")"):
        return None
    named = re.search(r"(\w+)\s*\(", bare)
    return named[1] if named else None


def items(text: str) -> list[Item]:
    """The top-level items of C text: directives, comments, function definitions and every other declaration."""
    found: list[Item] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        start = i
        if text[i] == "#":
            while i < n and (text[i] != "\n" or text[i - 1] == "\\"):
                i += 1
            found.append(Item(start, i, "directive"))
            continue
        if text.startswith(("/*", "//"), i):
            i = _skip(text, i)
            found.append(Item(start, i, "comment"))
            continue
        depth = paren = 0
        name: str | None = None
        while i < n:
            if (j := _skip(text, i)) != i:
                i = j
                continue
            c = text[i]
            i += 1
            if c == "(":
                paren += 1
            elif c == ")":
                paren -= 1
            elif c == "{":
                if depth == 0:
                    name = _function(text[start : i - 1])
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and name is not None:
                    break
            elif c == ";" and depth == 0 and paren == 0:
                name = None
                break
        found.append(Item(start, i, "function" if name else "decl", name or ""))
    return found


def _norm(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------- splicing
def _stand_in(segment: str, name: str) -> str:
    return rf'^[ \t]*{_KINDS}\([ \t]*"{re.escape(segment)}"[ \t]*,[ \t]*{re.escape(name)}[ \t]*\)[ \t]*;[ \t]*\n?'


def _block(segment: str, name: str) -> re.Pattern[str]:
    """A fuzzy candidate's block: the NON_MATCHING body over the stand-in it falls back to."""
    return re.compile(
        r"^#ifdef NON_MATCHING[ \t]*\n(?:(?!^#ifdef NON_MATCHING).)*?^#else[ \t]*\n"
        + _stand_in(segment, name)
        + r"^#endif[^\n]*(?:\n|$)",
        re.M | re.S,
    )


def wrap(segment: str, name: str, body: str) -> str:
    """A fuzzy body behind NON_MATCHING, falling back to the stand-in."""
    return f'#ifdef NON_MATCHING\n{body.rstrip()}\n#else\nINCLUDE_ASM("{segment}", {name});\n#endif\n'


def candidate(source: str, name: str | None) -> tuple[str, str]:
    """(declarations, body): the candidate's definition of `name`, and every other top-level item it brings. A data
    candidate (no `name`) is all body."""
    if name is None:
        return "", source.strip() + "\n"
    parts = items(source)
    mine = [k for k, p in enumerate(parts) if p.kind == "function" and p.name == name]
    if len(mine) != 1:
        raise Refusal(Finding("land.request", f"The candidate must define {name} exactly once.", unit=name))
    k = mine[0]
    first = k
    while first > 0 and parts[first - 1].kind == "comment":
        first -= 1  # the comments written above the function travel with it
    body = source[parts[first].start : parts[k].end].strip() + "\n"
    rest = [p for j, p in enumerate(parts) if not first <= j <= k and p.kind != "comment"]
    texts = [source[p.start : p.end].strip() for p in rest]
    opened = sum(bool(re.match(r"#\s*if", t)) for t in texts)
    closed = sum(bool(re.match(r"#\s*endif", t)) for t in texts)
    if opened != closed:
        raise Refusal(Finding("land.request", f"The candidate wraps {name} in a conditional.", unit=name))
    return "\n".join(texts), body


def splice(text: str, segment: str, name: str, source: str, fuzzy: bool, kind: str = "function") -> str:
    """`text` with the stand-in line (or earlier fuzzy block) of `name` replaced by the candidate, and the
    candidate's other top-level items the file lacks added right above it, after every include that precedes it."""
    declared, body = candidate(source, name if kind == "function" else None)
    new = wrap(segment, name, body) if fuzzy else body
    found = _block(segment, name).search(text) or re.compile(_stand_in(segment, name), re.M).search(text)
    if found is None:
        raise Refusal(
            Finding(
                "land.duplicate" if re.search(rf"\b{re.escape(name)}\b", text) else "land.request",
                f"{name} has no stand-in line in its file; it is already landed or the file does not own it.",
                unit=name,
            )
        )
    before, after = text[: found.start()], text[found.end() :]
    have = _norm(before + new + after)
    wanted = [d for d in re.split(r"\n(?=\S)", declared) if d.strip() and _norm(d) not in have]
    return before + "".join(d + "\n" for d in wanted) + "\n" * bool(wanted) + new + after


# ---------------------------------------------------------------- planning
@dataclass
class Reads:
    """What a plan read, by sha: the journal refuses the write when any of it changed."""

    config: Config
    reads: dict[str, str | None]


@dataclass(frozen=True)
class Landing:
    plan: Plan
    reads: Reads
    versions: tuple[str, ...]
    name: str
    path: str
    segment: str
    old: Mapping[str, bytes | None]


def refuse_blocking(findings: Sequence[Finding]) -> None:
    findings = tuple(f for f in findings if f.blocking)
    if findings:
        raise Refusal(*findings)


def make_plan(
    operation: str,
    base: str,
    writes: Mapping[str, bytes | None],
    affected: tuple[str, ...],
    debt: tuple[Finding, ...],
    message: str,
) -> Plan:
    body = (operation, base, writes, affected, (), debt, message)
    return Plan(*body, digest(body))


def read_file(reads: Reads, path: str) -> bytes | None:
    try:
        data: bytes | None = (reads.config.project.root / path).read_bytes()
    except FileNotFoundError:
        data = None
    reads.reads[path] = None if data is None else sha256(data)
    return data


def holders(captured: tree.Tree, name: str) -> tuple[tuple[str, ...], str, str]:
    """(versions holding `name`, the one file owning it in all of them, its segment)."""
    held = sorted({v for m in captured.modules for v, names in m.owns.items() if name in names})
    if not held:
        raise Refusal(Finding("land.request", f"{name} is owned by no file in any version.", unit=name))
    owners = {captured.owner(v, name) for v in held}
    if len(owners) != 1 or None in owners:
        raise Refusal(
            Finding(
                "land.request",
                f"{name} is owned by different files in different versions.",
                unit=name,
                versions=tuple(held),
                missing=tuple(sorted(o or "" for o in owners)),
            )
        )
    path = next(iter(owners)) or ""
    module = next(m for m in captured.modules if m.path == path)
    return tuple(held), path, module.placed.get(held[0], {}).get(name, module.segment)


def judged(
    config: Config, writes: Mapping[str, bytes | None], old: Mapping[str, bytes | None]
) -> tuple[tuple[Finding, ...], tuple[Finding, ...]]:
    snapshot = api.snapshot(config)
    before: list[Finding] = []
    after: list[Finding] = []
    for path, data in writes.items():
        if not configuration.is_text(path):
            continue
        if (was := old.get(path)) is not None:
            before.extend(policy.evaluate(snapshot, path, was.decode(), None, False))
        if data is not None:
            after.extend(policy.evaluate(snapshot, path, data.decode(), None, False))
    return policy.scope(before, after, set(writes))


def prepare(
    config: Config, request: Json, extras: Mapping[str, bytes], fuzzy: bool, message: str | None = None
) -> Landing:
    """The write that lands one candidate, judged by policy but not yet built."""
    with effort.stage("publish.prepare"):
        name = request["function"]
        source = Path(request["file"]).read_text(encoding="utf-8")
        captured = tree.capture(config)
        held, path, segment = holders(captured, name)
        reads = Reads(config, {})
        old: dict[str, bytes | None] = {p: read_file(reads, p) for p in (path, *extras)}
        current = old[path]
        if current is None:
            raise Refusal(Finding("land.request", f"{path} is missing.", path=path, unit=name))
        writes: dict[str, bytes | None] = {
            **extras,
            path: splice(current.decode(), segment, name, source, fuzzy).encode(),
        }
        blocking, debt = judged(config, writes, old)
        refuse_blocking(blocking)
        operation = "fuzzy" if fuzzy else "publish"
        plan = make_plan(operation, captured.digest, writes, (path,), debt, message or f"{operation} {name}")
        return Landing(plan, reads, held, name, path, segment, old)


def accept(config: Config, request: Json, extras: Mapping[str, bytes], fuzzy: bool) -> Landing:
    """The acceptance check of a submission: the same plan land makes, so a candidate land would refuse is refused
    here. Nothing is written and nothing is built."""
    return prepare(config, request, extras, fuzzy)


def accept_each(config: Config, checks: Sequence[tuple[Json, Mapping[str, bytes], bool]]) -> list[tuple[Finding, ...]]:
    """Per (request, extras, fuzzy) the findings of its Refusal, empty when accepted: one refusal blocks no other."""
    verdicts: list[tuple[Finding, ...]] = []
    for request, extras, fuzzy in checks:
        try:
            accept(config, request, extras, fuzzy)
        except Refusal as refusal:
            verdicts.append(tuple(refusal.findings))
        else:
            verdicts.append(())
    return verdicts


# ---------------------------------------------------------------- building
def build(config: Config, versions: Sequence[str]) -> None:
    """Builds the ROM of each given version (one make, each target once) and refuses unless every sha1 matches."""
    with effort.stage("publish.build"):
        roms = {v: f"build/{v}/{config.project.name}.z64" for v in versions}
        process.make(config, list(roms.values()), config.host.slots or config.host.workers)
        bad = []
        for v, rom in roms.items():
            built = config.project.root / rom
            if not built.is_file() or sha1(built.read_bytes()) != config.project.version[v].sha1:
                bad.append(v)
        if bad:
            raise Refusal(
                Finding(
                    "land.not_exact",
                    "The built ROM differs from the base ROM.",
                    versions=tuple(bad),
                    missing=tuple(f"{v}: sha1" for v in bad),
                    action="make check",
                )
            )
        report.record(config, tree.capture(config), exact=sorted(roms), failing=[])


def apply_checked(config: Config, landing: Landing) -> tuple[str, ...]:
    """Writes the plan through the journal, builds the affected versions and, when a build does not match, puts the old
    bytes back through the journal: a refused landing leaves the files as they were."""
    files = journal.apply(config, landing.plan, landing.reads)
    try:
        build(config, landing.versions)
    except BaseException:
        undo = make_plan("restore", landing.plan.base, landing.old, (), (), "restore")
        journal.apply(config, undo, landing.reads)
        raise
    return files


# ---------------------------------------------------------------- scores
def scores(config: Config, landing: Landing) -> dict[str, float]:
    """The candidate's score per version: the NON_MATCHING variant measured, linked alone at the function's address."""
    text = (landing.plan.writes[landing.path] or b"").decode()
    out: dict[str, float] = {}
    failed: list[Proof] = []
    for version in landing.versions:
        proof = cast(Callable[..., Proof], native.measure)(config, version, landing.name, landing.path, text)
        if any(proof.symptoms.get(k) for k in ("compile_failed", "unresolved", "no_measurement")):
            failed.append(proof)
        out[version] = proof.score
    if failed:
        raise Refusal(
            Finding(
                "land.not_exact",
                "The fuzzy candidate could not be measured.",
                unit=landing.name,
                missing=tuple(m for p in failed for m in p.missing),
            )
        )
    return out


def before_score(config: Config, name: str) -> float:
    """The minimum score over versions the function has now (0.0 when it has none)."""
    held = api.score(config, name)
    return min(held.values()) if held else 0.0


def gained(config: Config, name: str, now: Mapping[str, float]) -> None:
    if min(now.values()) - before_score(config, name) < configuration.fuzzy()["min_gain"]:
        raise Refusal(Finding("fuzzy.no_gain", "The fuzzy candidate has insufficient gain.", unit=name))


# ---------------------------------------------------------------- landing
def land(config: Config, submission: Submission) -> Receipt:
    with effort.stage("publish.land"):
        if submission.operation not in ("publish", "fuzzy"):
            raise Refusal(Finding("land.request", "Unknown submission operation.", unit=submission.member))
        if submission.extra_bases.keys() != submission.extras.keys():
            raise Refusal(
                Finding(
                    "land.rebase",
                    "Submission lacks measured extra-input provenance; resubmit against the current tree.",
                )
            )
        root = config.project.root
        try:
            extras = {path: (root / copy).read_bytes() for path, copy in submission.extras.items()}
        except OSError as error:
            raise Refusal(Finding("land.rebase", f"Measured extra-input is unavailable: {error}")) from None
        request = {
            "file": str(root / submission.source),
            "function": submission.function or submission.member,
            "overrides": submission.overrides,
            "note": submission.note,
        }
        fuzzy = submission.operation == "fuzzy"
        with journal.held(config):
            landing = prepare(config, request, extras, fuzzy)
            if fuzzy:
                now = scores(config, landing)
                gained(config, landing.name, now)
                shown = f"{landing.name} {min(now.values()) * 100:.1f}%"
                landing = prepare(config, request, extras, fuzzy, f"fuzzy {shown}")
            files = apply_checked(config, landing)
        plan = landing.plan
        return Receipt(
            plan.operation, files, plan.message, plan.digest, submission.proofs, len(plan.debt), effort.invocation()
        )
