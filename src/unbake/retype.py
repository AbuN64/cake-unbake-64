"""Retype void-pointer and integer parameters that every use casts to one struct pointer: the parameter takes the cast
type, the casts go, and the definition, every declaration and the callers follow as one atomic group."""

from __future__ import annotations

import posixpath
import re
import time
from dataclasses import asdict
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from unbake import compare, effort, headers, land, layout, policy, pool, store
from unbake.contracts import Config, Finding, Json, Refusal, Snapshot, digest
from unbake.families import typedebt

_KEYWORDS = frozenset(["return", "else", "case", "goto", "do", "if", "while", "for", "switch", "sizeof", "typedef"])
_PREFIX = re.compile(r"\s*(?:[A-Za-z_]\w*[\s*]+)+")
_DECLARATION = r"(?P<type>(?:\b(?:const|volatile|struct|union|unsigned|signed|register)\s+)*[A-Za-z_]\w*)"
_QUALIFIED = r"\(\s*(?:(?:const|volatile|struct|union|unsigned|signed)\s+)*"
_PRIMITIVES = "|".join(sorted(typedebt._PRIMITIVE))
_CAST_BEFORE = rf"{_QUALIFIED}(?:\w+\s*\*+|(?:{_PRIMITIVES}))\s*\)\s*$"  # a pointer or primitive cast's type
_INTEGER = typedebt._INTEGER | {"float", "f32", "double", "f64", "bool"}


def _balanced(code: str, start: int) -> int:
    """The index of the parenthesis closing the one at start, or -1."""
    depth = 0
    for index in range(start, len(code)):
        depth += (code[index] == "(") - (code[index] == ")")
        if depth == 0:
            return index
    return -1


def _split(text: str) -> list[tuple[int, str]]:
    """The (offset, text) of each top-level comma-separated part."""
    parts, begin, depth = [], 0, 0
    for index, char in enumerate(text):
        depth += (char in "([{") - (char in ")]}")
        if char == "," and not depth:
            parts.append((begin, text[begin:index]))
            begin = index + 1
    parts.append((begin, text[begin:]))
    return parts


def _retyped(old: str, spelled: str) -> str | None:
    """One parameter's text with the new type and the old name, or None when the old text is no void pointer or
    integer."""
    tokens = re.findall(r"\w+|\*", old)
    lead = old[: len(old) - len(old.lstrip())]
    if "*" in tokens:
        if tokens.count("*") != 1 or not set(tokens[: tokens.index("*")]) <= typedebt._PRIMITIVE:
            return None
        after = tokens[tokens.index("*") + 1 :]
        name = after[-1] if after else ""
    else:
        if not set(tokens) <= _INTEGER | {"long", "short"} and not (
            len(tokens) > 1 and set(tokens[:-1]) <= typedebt._INTEGER and tokens[-1] not in typedebt._PRIMITIVE
        ):
            return None
        name = tokens[-1] if len(tokens) > 1 and tokens[-1] not in typedebt._PRIMITIVE else ""
    return lead + spelled + name


def _identifier(code: str, name: str, at: int) -> tuple[str, int] | None:
    """The type word and star count of the nearest declaration of name before at, or None."""
    found = None
    pattern = rf"{_DECLARATION}\s*(?P<stars>\*+)?\s*\b{re.escape(name)}\b\s*(?P<after>[;,=)\[])"
    for match in re.finditer(pattern, code[:at]):
        before = code[: match.start()].rstrip()[-1:]
        word = match["type"].split()[-1]
        if word in _KEYWORDS or before not in ("", ";", "{", "}", "(", ","):
            continue
        found = word, len(match["stars"] or "") + (match["after"] == "[")
    return found


def _argument(code: str, text: str, at: int, target: str) -> str:
    """How a call's argument meets a parameter of pointer type to target: ok, integer, pointer or unknown."""
    arg = text.strip()
    if re.fullmatch(r"0+|NULL|\(\s*void\s*\*\s*\)\s*0", arg):
        return "ok"
    if re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|\d+)[uUlL]*", arg):
        return "integer"
    cast = re.match(r"\(\s*((?:\w+\s+)*\w+)\s*(\*+)?\s*\)\s*\S", arg)
    if cast and not arg.startswith("(("):
        if not cast[2]:
            return "integer" if set(cast[1].split()) <= _INTEGER else "unknown"
        word = cast[1].split()[-1]
        return "ok" if word in ("void", target) else "pointer"
    if re.fullmatch(r"[A-Za-z_]\w*", arg):
        known = _identifier(code, arg, at)
        if known is None:
            return "unknown"
        word, stars = known
        if stars:
            return "ok" if word in ("void", target) else "pointer"
        return "integer" if word in _INTEGER else "unknown"
    return "unknown"


def _bare(code: str, at: int, name: str) -> tuple[str, int] | None:
    """How a use of a parameter outside its casts is harmless once the parameter is a struct pointer: a comparison
    with zero gives ("", -1), a bare call argument gives the callee and the argument's index, anything else None."""
    end = at + len(name)
    before, after = code[:at].rstrip(), code[end:].lstrip()
    if re.search(r"(?:!|[=!]=\s*(?:0|NULL))$", before) or re.match(r"[=!]=\s*(?:0|NULL)\b", after):
        return "", -1
    if before[-1:] not in ("(", ",") or after[:1] not in (",", ")"):
        return None
    depth = commas = 0
    for index in range(len(before) - 1, -1, -1):
        char = before[index]
        depth += (char in ")]}") - (char in "([{")
        if depth < 0:
            callee = re.search(r"([A-Za-z_]\w*)\s*$", before[:index])
            if char != "(" or not callee or callee[1] in _KEYWORDS:
                return None
            return (callee[1], commas) if commas or before[index + 1 :].strip() == "" else None
        commas += char == "," and depth == 0
    return None


def _accepts(
    known: Mapping[str, list[dict[str, Any]]],
    codes: Mapping[str, str],
    callee: str,
    index: int,
    pointer: bool,
    path: str,
) -> bool:
    """Whether the callee's definition and the declarations in the calling file take a void pointer (or an integer,
    from a pointer) at index."""
    seen = 0
    for site in known.get(callee, ()):
        if site["kind"] == "call" or (site["kind"] == "prototype" and site["path"] != path):
            continue
        args = _split(codes[site["path"]][slice(*site["args"])])
        if index >= len(args):
            return False
        old = args[index][1]
        seen += 1
        tokens = re.findall(r"\w+|\*", old)
        stars = tokens.count("*")
        pointer_to_void = stars == 1 and "void" in tokens[: tokens.index("*")]
        integer = not stars and bool(tokens) and set(tokens[: max(1, len(tokens) - 1)]) <= typedebt._INTEGER
        if not (pointer_to_void or (pointer and integer)):
            return False
    return seen > 0


def _sites(codes: Mapping[str, str], names: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Per function name its prototypes and its calls across every source and header."""
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\b\s*\(")
    for path, code in codes.items():
        for match in pattern.finditer(code):
            close = _balanced(code, match.end() - 1)
            if close < 0:
                continue
            tail = code[close + 1 :].lstrip()[:1]
            if tail == "{":
                found[match[1]].append(
                    {"path": path, "kind": "definition", "args": (match.end(), close), "at": match.start()}
                )
                continue
            edge = max(code.rfind(c, 0, match.start()) for c in ";{}")
            prefix = code[edge + 1 : match.start()]
            words = set(re.findall(r"\w+", prefix))
            prototype = bool(_PREFIX.fullmatch(prefix)) and not words & _KEYWORDS and tail in (";", ",", ")")
            found[match[1]].append(
                {
                    "path": path,
                    "kind": "prototype" if prototype else "call",
                    "args": (match.end(), close),
                    "at": match.start(),
                }
            )
    return found


def _reaches(texts: Mapping[str, str]) -> Callable[[str, str], bool]:
    """Whether one file includes another, directly or through others."""

    def reaches(start: str, target: str) -> bool:
        seen, pending = set(), [start]
        while pending:
            path = pending.pop()
            if path == target:
                return True
            if path not in seen:
                seen.add(path)
                for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"', texts.get(path, ""), re.M):
                    pending += [posixpath.normpath(f"{base}/{name}") for base in ("include", posixpath.dirname(path))]
        return False

    return reaches


def _tag(owner: str, name: str, tagged: bool) -> str:
    """The struct tag that names the record `name` in its declaring header's text, or "" when it has none."""
    if tagged:
        return name
    alias = re.search(rf"\btypedef\s+struct\s+(\w+)\s*\**\s*\b{re.escape(name)}\s*;", owner)
    if alias:
        return alias[1]
    for head in re.finditer(r"\btypedef\s+struct\s+(\w+)\s*\{", owner):
        depth, end = 0, len(owner)
        for index in range(head.end() - 1, len(owner)):
            depth += (owner[index] == "{") - (owner[index] == "}")
            if depth == 0:
                end = index
                break
        if re.match(rf"\}}\s*{re.escape(name)}\s*;", owner[end:]):
            return head[1]
    return ""


def _visible(code: str, name: str, tagged: bool) -> bool:
    """Whether code declares the type name (a struct tag when tagged) at file scope."""
    if tagged:
        return _declared_tag(code, name)
    return bool(re.search(rf"\btypedef\b[^;{{}}]*\b{re.escape(name)}\s*;|\}}\s*{re.escape(name)}\s*;", code))


def _declared_tag(code: str, tag: str) -> bool:
    """Whether code declares struct tag at file scope: a forward declaration, a definition or a typedef of it."""
    pattern = rf"^[ \t]*(?:typedef\s+|extern\s+)?struct\s+{re.escape(tag)}\b\s*(?:;|\{{|\**\s*\w+\s*[;\[])"
    return any(_depth(code, m.start()) == 0 for m in re.finditer(pattern, code, re.M))  # a field declares nothing


def _depth(code: str, at: int) -> int:
    """The brace depth at an offset of comment-free code."""
    return code.count("{", 0, at) - code.count("}", 0, at)


def plan(
    texts: Mapping[str, str],
    declared: Mapping[str, str],
    reaches: Callable[[str, str], bool],
    exclude: Collection[tuple[str, int]] = (),
) -> dict[str, Any]:
    """The writes that retype every eligible parameter, with the debt counted before and after and each skipped
    parameter under its reason. texts maps every source and header path to its text, declared maps each type name
    to the header declaring it, and reaches says whether one file includes another. `exclude` names the (function,
    parameter index) pairs not to retype; `touches` gives per retyped pair the files its writes reach."""
    with effort.stage("retype.scopes"):
        codes = {p: policy.scopes(t)["code"] for p, t in texts.items()}
    units = [p for p in codes if p.startswith("src/") and p.endswith(".c")]
    landed = [p for p in units if not p.startswith("src/fuzzy/")]
    rows: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    with effort.stage("retype.parameters"):
        for path in landed:
            for row in typedebt.parameters(codes[path]):
                rows[row["function"]].append((path, row))
    skipped: Counter[str] = Counter()
    reasons: dict[str, list[str]] = defaultdict(list)

    def skip(reason: str, function: str, name: str) -> None:
        skipped[reason] += 1
        reasons[reason].append(f"{function}({name})")

    wanted: dict[tuple[str, int], dict[str, Any]] = {}
    for function, entries in rows.items():
        for path, row in entries:
            targets = {c["target"] for c in row["casts"] if c["target"] not in typedebt._PRIMITIVE}
            if not targets:
                continue
            label = (function, row["name"])
            if len(targets) > 1:
                skip("several cast targets", *label)
                continue
            if len({where for where, _ in entries}) > 1:
                skip("name defined in several sources", *label)
                continue
            base = row["body"][0]
            covered = {c["span"][1] - len(row["name"]) for c in row["casts"]}
            uses = re.finditer(rf"(?<![.\w])(?<!->){re.escape(row['name'])}\b", codes[path][slice(*row["body"])])
            loose = [_bare(codes[path], base + m.start(), row["name"]) for m in uses if base + m.start() not in covered]
            if None in loose:
                skip("parameter used outside the cast", *label)
                continue
            passed = [c for c in loose if c and c[1] >= 0]
            chosen = [c for c in row["casts"] if c["target"] in targets]
            if len({c["spelled"] for c in row["casts"]}) > 1:
                skip("casts spell the type differently", *label)
                continue
            if row["pointer"] and {c["qualifiers"] for c in chosen} != {row["qualifiers"]}:
                skip("qualifiers differ from the parameter", *label)
                continue
            if any(c["open"] and not re.match(r"\s*\)", codes[path][c["span"][1] :]) for c in row["casts"]):
                skip("cast operand is more than the parameter", *label)
                continue
            if (function, row["index"]) in exclude:
                skip("its files would carry new debt", *label)
                continue
            spellings = {c["spelled"] for c in chosen}
            wanted[(function, row["index"])] = {
                "path": path,
                "row": row,
                "spelled": next(iter(spellings)),
                "passed": passed,
            }
    callees = {c for item in wanted.values() for c, _ in item["passed"]}
    with effort.stage("retype.sites"):
        sites = _sites(codes, {f for f, _ in wanted} | callees) if wanted else {}
    for (function, index), item in sorted(wanted.items()):
        for callee, at in item["passed"]:
            other = wanted.get((callee, at))
            if other is not None and other["spelled"] != item["spelled"]:
                bad = "passes the parameter to a retyped callee of another type"
            elif other is None and not _accepts(sites, codes, callee, at, item["row"]["pointer"], item["path"]):
                bad = "parameter used outside the cast"
            else:
                continue
            skip(bad, function, item["row"]["name"])
            del wanted[(function, index)]
            break
    edits: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    includes: dict[str, set[str]] = defaultdict(set)
    added: set[tuple[str, str]] = set()  # the include edges this plan already adds
    forwards: dict[tuple[str, str], int] = {}  # per header and tag the earliest prototype that needs its declaration

    def linked(start: str, goal: str, used: frozenset[tuple[str, str]] = frozenset()) -> bool:
        """Whether start includes goal once the planned includes are in: a planned edge joins two reaches."""
        if reaches(start, goal):
            return True
        return any(
            (start == src or reaches(start, src)) and linked(dst, goal, used | {(src, dst)})
            for src, dst in added - used
        )

    retyped = 0
    touches: dict[tuple[str, int], list[str]] = {}
    for (function, index), item in sorted(wanted.items()):
        path, row, spelled = item["path"], item["row"], item["spelled"]
        label = (function, row["name"])
        target = re.findall(r"\w+", spelled)[-1]
        key = "struct " + target if spelled.split()[0] in ("struct", "const") and "struct" in spelled else target
        owner = declared.get(key)
        tag = _tag(texts.get(owner or "", ""), target, key != target)
        plan_edits: list[tuple[str, int, int, str]] = []
        needs: set[str] = set()
        declares: dict[str, int] = {}
        failure = ""
        for site in sites.get(function, ()):
            where, (begin, end) = site["path"], site["args"]
            if site["kind"] == "definition":
                continue
            if site["kind"] == "prototype":
                args = _split(codes[where][begin:end])
                if len(args) == 1 and args[0][1].strip() in ("", "void"):
                    continue
                if index >= len(args):
                    failure = "declaration has no such parameter"
                    break
                offset, old = args[index]
                if owner is None:
                    blind = False
                elif where == owner:  # its own header: the type must come before the declaration
                    blind = not _visible(codes[where][: site["at"]], target, key != target)
                else:
                    blind = not linked(where, owner)
                # a file-scope declaration that cannot see the type names its tag, declared just before: an include
                # would add an edge to the include graph (a cycle, or another header's conflicting declarations)
                tagged = blind and bool(tag) and _depth(codes[where], site["at"]) == 0
                if typedebt.fake_name(target if not tagged else tag) and not re.search(
                    rf"\b{re.escape(tag if tagged else target)}\b", codes[where]
                ):  # an invented record name spreads to a file that never named it: new debt
                    failure = "declaration would name an invented type"
                    break
                if blind and not tagged and where == owner:
                    failure = "declaration comes before its type"
                    break
                named = rf"(?:\bstruct\s+)?\b{re.escape(target)}\b"
                local = re.sub(named, f"struct {tag}", spelled) if tagged else spelled
                new = _retyped(old, local)
                if new is None:
                    if re.sub(r"\s+", "", old).startswith(re.sub(r"\s+", "", spelled)):
                        continue
                    failure = "declaration spells another type"
                    break
                plan_edits.append((where, begin + offset, begin + offset + len(old), new))
                if tagged:
                    if not _declared_tag(codes[where][: site["at"]], tag):
                        code = codes[where]
                        edge = max(code.rfind(c, 0, site["at"]) for c in ";{}") + 1
                        lead = edge + len(code[edge:]) - len(code[edge:].lstrip())
                        line = code.rfind("\n", 0, lead) + 1  # the line the declaration statement starts on
                        declares[where] = min(declares.get(where, line), line)
                else:
                    needs.add(where)
            else:
                args = _split(codes[where][begin:end])
                if index < len(args):
                    verdict = _argument(codes[where], args[index][1], site["at"], target)
                    if verdict in ("integer", "pointer"):
                        failure = f"caller passes {'an integer' if verdict == 'integer' else 'another pointer type'}"
                        break
        if failure:
            skip(failure, *label)
            continue
        for casting in row["casts"]:
            first, last = casting["span"]
            if casting["open"]:  # the parenthesis around the operand goes with the cast
                last += len(re.match(r"\s*\)", codes[path][last:])[0])  # type: ignore[index]
            code = codes[path]
            outer = len(code[:first].rstrip())
            if (
                code[:first].rstrip().endswith("(")
                and (
                    not re.search(r"[)\]]\s*$|\b(?!(?:return|else|case)\b)\w+\s*$", code[: outer - 1])
                    or re.search(_CAST_BEFORE, code[: outer - 1])  # (U *)((T *)p)->f is (U *)p->f
                )
                and (close := re.match(r"\s*\)(?=\s*(?:->|\[|\.))", code[last:]))
            ):  # ((T *)p)->f is p->f
                first, last = outer - 1, last + close.end()
            plan_edits.append((path, first, last, row["name"]))
        plan_edits.append(
            (path, row["span"][0], row["span"][1], _retyped(texts[path][slice(*row["span"])], spelled) or "")
        )
        needs.discard(path)
        extra: set[tuple[str, str]] = set()
        for where in sorted(needs):
            if owner is None:
                failure = "type is not declared in a header"
            elif where != owner and not linked(where, owner):
                if linked(owner, where):  # with every include planned so far: a new include must close no cycle
                    failure = "declaring header includes the file that needs the type"
                else:
                    extra.add((where, owner))
                    added.add((where, owner))
        if failure:
            added.difference_update(extra)
            skip(failure, *label)
            continue
        retyped += 1
        touches[(function, index)] = sorted({e[0] for e in plan_edits} | {w for w, _ in extra} | set(declares))
        for where, first, last, new in plan_edits:
            edits[where].append((first, last, new))
        for where, _ in extra:
            includes[where].add(owner or "")
        for where, line in declares.items():
            forwards[(where, tag)] = min(forwards.get((where, tag), line), line)
    for (where, tag), line in sorted(forwards.items()):
        edits[where].append((line, line, f"struct {tag};\n"))
    writes: dict[str, str] = {}
    for path, spans in edits.items():
        text = texts[path]
        taken = -1
        for first, last, new in sorted(spans, reverse=True):
            if last > taken >= 0:
                continue
            text = text[:first] + new + text[last:]
            taken = first
        writes[path] = text
    for path, owners in includes.items():
        writes[path] = headers.included(writes.get(path, texts[path]), [o.removeprefix("include/") for o in owners])
    with effort.stage("retype.count"):
        before = sum(len(typedebt.param_cast(codes[p])) for p in landed)
        after = sum(len(typedebt.param_cast(policy.scopes(writes.get(p, texts[p]))["code"])) for p in landed)
    return {
        "writes": writes,
        "touches": touches,
        "retyped": retyped,
        "param_cast": {"before": before, "after": after},
        "skipped": dict(skipped),
        "examples": {r: sorted(v)[:5] for r, v in reasons.items()},
    }


def _debt_job(item: tuple[Snapshot, Snapshot, str]) -> tuple[tuple[Finding, ...], tuple[Finding, ...]]:
    """The policy findings of one written file before and after the write (publish judges the same pair)."""
    snapshot, overlay, path = item
    unit = overlay.layout.units.get(path)
    sdk = bool(unit and overlay.layout.groups[unit.group].sdk)
    old, new = snapshot.peek(path), overlay.peek(path)
    before = policy.evaluate(snapshot, path, old.decode(), None, sdk) if old is not None else ()
    after = policy.evaluate(overlay, path, new.decode(), None, sdk) if new is not None else ()
    return tuple(before), tuple(after)


def new_debt(config: Config, snapshot: Snapshot, writes: Mapping[str, str]) -> tuple[Finding, ...]:
    """The blocking rows the writes add, judged as a submission of them would judge them, on all cores."""
    if not writes:
        return ()
    overlay = layout.overlay(snapshot, {p: t.encode() for p, t in writes.items()})
    jobs = [(snapshot, overlay, p) for p in sorted(writes)]
    with effort.stage("retype.debt"):
        judged = pool.map(config, "retype.debt", _debt_job, jobs)
    before = [f for b, _ in judged for f in b]
    after = [f for _, a in judged for f in a]
    return policy.scope(before, after, writes)[0]


def converge(
    planner: Callable[[set[tuple[str, int]]], dict[str, Any]],
    judge: Callable[[Mapping[str, str]], Collection[Finding]],
    exclude: set[tuple[str, int]],
) -> tuple[dict[str, Any], list[Json]]:
    """Plan, judge the writes, and drop every retype whose writes reach a file with a blocking row, until the rest
    carries none: one row then costs only the retypes that touch its file, never the whole group."""
    rounds: list[Json] = []
    while True:
        result = planner(exclude)
        rows = judge(result["writes"])
        bad = {f.path for f in rows}
        drop = {key for key, paths in result["touches"].items() if bad & set(paths)}
        rounds.append({"rows": len(rows), "dropped": len(drop), "paths": sorted(bad)[:20]})
        if not rows or not drop:
            return result, rounds
        exclude |= drop


def blamed(findings: Sequence[Json], touches: Mapping[tuple[str, int], Sequence[str]]) -> set[tuple[str, int]]:
    """The retypes a refusal names: those whose writes reach a file a finding sits in, is about, or names as the
    file that breaks it ("src/a.c breaks src/b.c: ...")."""
    written = {p for files in touches.values() for p in files}
    named: set[str] = set()
    for f in findings:
        named |= {f.get("path", ""), f.get("unit", "")}
        named |= {p for p in re.findall(r"[\w./-]+\.[ch]\b", f.get("reason", "")) if p in written}
    return {key for key, files in touches.items() if named & set(files)}


def run(config: Config, params: Json) -> Json:
    """Plan the retype over the live tree, read-only; with submit, prove every affected unit in all versions. A
    retype whose files carry a new blocking row, or that a refused submission names, is dropped and the rest planned
    and submitted again, so the run converges on what lands."""
    start = time.monotonic()
    out = Path(params.get("out") or Path("/tmp") / ("unbake-retype-" + digest(str(config.project.root))[:16]))
    out = out.resolve()
    if out.is_relative_to(config.project.root.resolve()):
        raise Refusal(Finding("land.request", "Retype output must be outside the live project.", path=str(out)))
    snapshot = layout.capture(config)
    paths = [*headers.sources(snapshot, "src", ".c"), *headers.sources(snapshot)]
    texts = {p: snapshot.read(p).decode(errors="replace") for p in paths}
    declared = {name: row[0] for name, row in headers.catalog(snapshot).items() if row[0].endswith(".h")}
    reaches = _reaches(texts)
    exclude: set[tuple[str, int]] = set()
    rounds: list[Json] = []
    submissions, drain = None, None
    while True:
        result, judged = converge(
            lambda skip: plan(texts, declared, reaches, skip),
            lambda writes: new_debt(config, snapshot, writes),
            exclude,
        )
        rounds += judged
        writes, touches = result.pop("writes"), result.pop("touches")
        sources = sorted(p for p in writes if p.endswith(".c"))
        if sources:
            for path, text in writes.items():
                store.write(out / path, text.encode())
        if not (params.get("submit") and sources):
            break
        try:
            submissions, drain = _submit(config, snapshot, writes, sources, out)
            refused = [] if drain else [f for row in submissions["submissions"] for f in row.get("findings", ())]
        except Refusal as refusal:
            refused = [asdict(f) for f in refusal.findings]
            if not blamed(refused, touches):
                raise
        drop = blamed(refused, touches)
        rounds.append({"rows": len(refused), "dropped": len(drop), "refused": bool(refused)})
        if not drop:
            break
        exclude |= drop
    return {
        **result,
        "files": sorted(writes),
        "out": str(out),
        "rounds": rounds,
        "submissions": submissions,
        "drain": {
            "landed": [r for d in drain or () for r in d.get("landed", ())],
            "refused": [r for d in drain or () for r in d.get("refused", ())],
        }
        if drain is not None
        else None,
        "seconds": round(time.monotonic() - start, 1),
    }


def _submit(
    config: Config, snapshot: Snapshot, writes: Mapping[str, str], sources: list[str], out: Path
) -> tuple[Json, list[Json]]:
    """Prove and land the writes as one atomic group."""
    pinned = {p: digest(snapshot.read(p)) for p in writes}
    if any(digest((config.project.root / p).read_bytes()) != pin for p, pin in pinned.items()):
        raise Refusal(Finding("journal.changed", "Retype input changed during analysis."))
    primary = sources[0]
    group = [out / primary, *(Path(p + "=" + str(out / p)) for p in sorted(writes) if p != primary)]
    submissions = compare.submit_command(
        config,
        {
            "files": group,
            "bulk": True,
            "function": None,
            "toolchain": None,
            "flag": (),
            "omit_flag": (),
            "withhold": False,
            "note": "retype parameters to the struct their casts name",
            "batch_data": None,
            "prove_extras": True,
        },
    )
    admitted = [row["id"] for row in submissions["submissions"] if row["id"] and row["exact"]]
    return submissions, [land.drain(config, identity) for identity in admitted]
