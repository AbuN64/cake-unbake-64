"""Retype void-pointer and integer parameters that every use casts to one struct pointer: the parameter takes the cast
type, the casts go, and the definition, every declaration and the callers follow as one atomic group."""

from __future__ import annotations

import posixpath
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from unbake import compare, effort, headers, land, layout, policy, store
from unbake.contracts import Config, Finding, Json, Refusal, digest
from unbake.families import typedebt

_KEYWORDS = frozenset(["return", "else", "case", "goto", "do", "if", "while", "for", "switch", "sizeof", "typedef"])
_PREFIX = re.compile(r"\s*(?:[A-Za-z_]\w*[\s*]+)+")
_DECLARATION = r"(?P<type>(?:\b(?:const|volatile|struct|union|unsigned|signed|register)\s+)*[A-Za-z_]\w*)"
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


def plan(texts: Mapping[str, str], declared: Mapping[str, str], reaches: Callable[[str, str], bool]) -> dict[str, Any]:
    """The writes that retype every eligible parameter, with the debt counted before and after and each skipped
    parameter under its reason. texts maps every source and header path to its text, declared maps each type name
    to the header declaring it, and reaches says whether one file includes another."""
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
    retyped = 0
    for (function, index), item in sorted(wanted.items()):
        path, row, spelled = item["path"], item["row"], item["spelled"]
        label = (function, row["name"])
        target = re.findall(r"\w+", spelled)[-1]
        key = "struct " + target if spelled.split()[0] in ("struct", "const") and "struct" in spelled else target
        owner = declared.get(key)
        plan_edits: list[tuple[str, int, int, str]] = []
        needs: set[str] = set()
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
                new = _retyped(old, spelled)
                if new is None:
                    if re.sub(r"\s+", "", old).startswith(re.sub(r"\s+", "", spelled)):
                        continue
                    failure = "declaration spells another type"
                    break
                plan_edits.append((where, begin + offset, begin + offset + len(old), new))
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
                and not re.search(r"[)\]]\s*$|\b(?!(?:return|else|case)\b)\w+\s*$", code[: outer - 1])
                and (close := re.match(r"\s*\)(?=\s*(?:->|\[|\.))", code[last:]))
            ):  # ((T *)p)->f is p->f
                first, last = outer - 1, last + close.end()
            plan_edits.append((path, first, last, row["name"]))
        plan_edits.append(
            (path, row["span"][0], row["span"][1], _retyped(texts[path][slice(*row["span"])], spelled) or "")
        )
        needs.discard(path)
        extra: set[str] = set()
        for where in needs:
            if owner is None:
                failure = "type is not declared in a header"
            elif where != owner and not reaches(where, owner):
                if reaches(owner, where):
                    failure = "declaring header includes the file that needs the type"
                else:
                    extra.add(where)
        if failure:
            skip(failure, *label)
            continue
        retyped += 1
        for where, first, last, new in plan_edits:
            edits[where].append((first, last, new))
        for where in extra:
            includes[where].add(owner or "")
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
        "retyped": retyped,
        "param_cast": {"before": before, "after": after},
        "skipped": dict(skipped),
        "examples": {r: sorted(v)[:5] for r, v in reasons.items()},
    }


def run(config: Config, params: Json) -> Json:
    """Plan the retype over the live tree, read-only; with submit, prove every affected unit in all versions."""
    start = time.monotonic()
    out = Path(params.get("out") or Path("/tmp") / ("unbake-retype-" + digest(str(config.project.root))[:16]))
    out = out.resolve()
    if out.is_relative_to(config.project.root.resolve()):
        raise Refusal(Finding("land.request", "Retype output must be outside the live project.", path=str(out)))
    snapshot = layout.capture(config)
    paths = [*headers.sources(snapshot, "src", ".c"), *headers.sources(snapshot)]
    texts = {p: snapshot.read(p).decode(errors="replace") for p in paths}
    declared = {name: row[0] for name, row in headers.catalog(snapshot).items() if row[0].endswith(".h")}
    result = plan(texts, declared, _reaches(texts))
    writes = result.pop("writes")
    submissions, drain = None, None
    sources = sorted(p for p in writes if p.endswith(".c"))
    if sources:
        for path, text in writes.items():
            store.write(out / path, text.encode())
    if params.get("submit") and sources:
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
        drain = [land.drain(config, identity) for identity in admitted]
    return {
        **result,
        "files": sorted(writes),
        "out": str(out),
        "submissions": submissions,
        "drain": {
            "landed": [r for d in drain or () for r in d.get("landed", ())],
            "refused": [r for d in drain or () for r in d.get("refused", ())],
        }
        if drain is not None
        else None,
        "seconds": round(time.monotonic() - start, 1),
    }
