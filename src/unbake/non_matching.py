"""Classify conditional C and ratchet host overrides of ROM-side C definitions.

A fallback owns a function only on the NON_MATCHING side and replaces an asm stand-in.
An override changes a function which also has a C definition on the ROM side, including
file-scope macros used by that definition. No compilation or ROM is needed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from unbake import config, naming_baseline, store
from unbake.contracts import Finding, Json, Refusal, sha256

KEY = "source.non_matching_override"
PATH = "config/non-matching-baseline.json"
_CONDITION = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", re.S)
_DEFINE = re.compile(r"^\s*#\s*define\s+(\w+)(.*)", re.M)
_WORD = re.compile(r"\b[A-Za-z_]\w*\b")


def _directives(text: str) -> list[tuple[int, int, str, str]]:
    from unbake.policy import scopes

    lines = scopes(text)["directives"].splitlines(keepends=True)
    out: list[tuple[int, int, str, str]] = []
    at = 0
    for line in lines:
        if out and out[-1][3].rstrip().endswith("\\"):
            start, _, kind, rest = out.pop()
            out.append((start, at + len(line), kind, rest.rstrip()[:-1] + line))
        elif match := _CONDITION.match(line):
            out.append((at, at + len(line), match[1], match[2]))
        at += len(line)
    return out


def _blocks(directives: Sequence[tuple[int, int, str, str]]) -> list[tuple[int, int]]:
    stack: list[tuple[int, bool]] = []
    out = []
    for start, end, kind, rest in directives:
        if kind in ("if", "ifdef", "ifndef"):
            stack.append((start, "NON_MATCHING" in _WORD.findall(rest)))
        elif kind == "elif" and stack:
            opened, named = stack.pop()
            stack.append((opened, named or "NON_MATCHING" in _WORD.findall(rest)))
        elif kind == "endif" and stack:
            opened, named = stack.pop()
            if named:
                out.append((opened, end))
    return sorted(out)


def _condition(kind: str, rest: str, on: bool, version: str) -> bool | None:
    from unbake.cheat import _condition as condition
    from unbake.tree import macro

    if kind in ("ifdef", "ifndef"):
        rest = f"{'!' if kind == 'ifndef' else ''}defined({rest.strip()})"
    known = {"NON_MATCHING": on}
    if version:
        known.update({name: name == macro(version) for name in _WORD.findall(rest) if name.startswith("VERSION_")})
    rest = re.sub(
        r"\bdefined\s*(?:\(\s*(\w+)\s*\)|(\w+))",
        lambda m: str(int(known[m[1] or m[2]])) if (m[1] or m[2]) in known else m[0],
        rest,
    )
    rest = _WORD.sub(lambda m: str(int(known[m[0]])) if m[0] in known else m[0], rest)
    try:
        return condition("if", rest.strip(), on)
    except (ValueError, IndexError):
        return None


def _project(text: str, directives: Sequence[tuple[int, int, str, str]], on: bool, version: str) -> str:
    from unbake.cheat import _and, _or

    def negate(value: bool | None) -> bool | None:
        return None if value is None else not value

    def blank(value: str) -> str:
        return re.sub(r"[^\n]", " ", value)

    active: bool | None = True
    stack: list[tuple[bool | None, bool | None]] = []
    parts = []
    at = 0
    for start, end, kind, rest in directives:
        parts.append(blank(text[at:start]) if active is False else text[at:start])
        parts.append(blank(text[start:end]))
        if kind in ("if", "ifdef", "ifndef"):
            value = _condition(kind, rest, on, version)
            stack.append((active, value))
            active = _and(active, value)
        elif kind in ("elif", "else") and stack:
            parent, seen = stack[-1]
            value = True if kind == "else" else _condition("if", rest, on, version)
            active = _and(parent, _and(negate(seen), value))
            stack[-1] = parent, _or(seen, value)
        elif kind == "endif" and stack:
            active, _ = stack.pop()
        at = end
    parts.append(blank(text[at:]) if active is False else text[at:])
    return "".join(parts)


@lru_cache(maxsize=2048)
def _tokens(text: str) -> tuple[str, ...]:
    from unbake.tree import _TOKEN

    return tuple(m[0] for m in _TOKEN.finditer(text) if not m[0].startswith(("/*", "//")))


def _functions(text: str) -> dict[str, list[tuple[int, int, str]]]:
    from unbake.policy import scopes
    from unbake.publish import items

    result: dict[str, list[tuple[int, int, str]]] = {}
    for item in items(scopes(text)["code"]):
        if item.kind == "function":
            aliases: dict[str, str] = {}
            for match in re.finditer(r"^\s*#\s*(define|undef)\s+(\w+)([^\n]*)", text[: item.start], re.M):
                if match[1] == "undef":
                    aliases.pop(match[2], None)
                elif re.fullmatch(r"\s+\w+\s*", match[3]):
                    aliases[match[2]] = match[3].strip()
            name, seen = item.name, set()
            while name in aliases and name not in seen:
                seen.add(name)
                name = aliases[name]
            result.setdefault(name, []).append((item.start, item.end, text[item.start : item.end]))
    return result


def scan(
    path: str,
    text: str,
    versions: Sequence[str] = (),
    assembled: Mapping[str, Collection[str]] | None = None,
    matching: Mapping[str, Collection[str]] | None = None,
) -> list[dict[str, Any]]:
    """One row per conditional group, including inverted and elif guards; line numbers are 1-based.

    Unknown non-version conditions retain both possible arms. Versions are the project's macro
    names, not inferred ROM matches: a B row means ROM-side C, whose bytes make check still proves.
    """
    directives = _directives(text)
    blocks = _blocks(directives)
    if not blocks:
        return []
    rows: list[dict[str, Any]] = [
        {
            "file": path,
            "line": text.count("\n", 0, start) + 1,
            "end_line": text.count("\n", 0, end),
            "guard": text[start : text.find("\n", start)].strip(),
            "fingerprint": sha256(repr(_tokens(text[start:end])).encode()),
            "class": "C",
            "functions": {},
        }
        for start, end in blocks
    ]
    for version in versions or ("",):
        off, on = (_project(text, directives, flag, version) for flag in (False, True))
        original, host = _functions(off), _functions(on)
        definitions = [{m[1]: m[2] for m in _DEFINE.finditer(side)} for side in (off, on)]
        for row, (start, end) in zip(rows, blocks, strict=True):
            changed = {
                m[1]
                for m in _DEFINE.finditer(text[start:end])
                if _tokens(definitions[0].get(m[1], "")) != _tokens(definitions[1].get(m[1], ""))
            }
            # Follow local macro aliases: a function may use WRAP while this block changes WAIT.
            while True:
                more = {
                    name
                    for defs in definitions
                    for name, value in defs.items()
                    if changed.intersection(_WORD.findall(value))
                } - changed
                if not more:
                    break
                changed.update(more)
            for name in sorted(original.keys() | host.keys()):
                before, after = original.get(name, []), host.get(name, [])
                inside = any(start < b and a < end for a, b, _ in before + after)
                macro = any(changed.intersection(_WORD.findall(body)) for _, _, body in before + after)
                if not (inside or macro):
                    continue
                differs = [_tokens(body) for _, _, body in before] != [_tokens(body) for _, _, body in after]
                if (before and (differs or macro)) or (
                    after and not before and name in (matching or {}).get(version, ())
                ):
                    kind = "B"
                elif (
                    after
                    and not before
                    and (
                        name in (assembled or {}).get(version, ())
                        or re.search(rf"\bINCLUDE_ASM\s*\([^;]*\b{re.escape(name)}\s*\)", off)
                    )
                ):
                    kind = "A"
                else:
                    kind = "C"
                detail = row["functions"].setdefault(name, {"A": [], "B": [], "C": []})
                detail[kind].append(version)
                if kind == "B" or (kind == "A" and row["class"] == "C"):
                    row["class"] = kind
    return rows


def findings(
    path: str, text: str, versions: Sequence[str] = (), matching: Mapping[str, Collection[str]] | None = None
) -> tuple[Finding, ...]:
    return tuple(
        Finding(
            KEY,
            f"NON_MATCHING changes ROM-side C: {name} [{row['fingerprint']}]",
            path=path,
            line=row["line"],
            unit=name,
            versions=tuple(v for v in classes["B"] if v),
            action="restore the original C and put host adaptation in the shim",
        )
        for row in scan(path, text, versions, matching=matching)
        for name, classes in row["functions"].items()
        if classes["B"]
    )


def read(root: Path) -> Json | None:
    try:
        document: Json = json.loads((root / PATH).read_bytes())
    except FileNotFoundError:
        return None
    except (ValueError, UnicodeError) as error:
        raise Refusal(Finding("config.schema", str(error), path=PATH)) from error
    config.validate("non-matching-baseline", document, PATH)
    return document


def check(
    root: Path,
    files: Mapping[str, str],
    versions: Sequence[str] = (),
    matching: Mapping[str, Mapping[str, Collection[str]]] | None = None,
) -> tuple[tuple[Finding, ...], Json | None]:
    baseline = read(root)
    blocked, debt = naming_baseline.split(
        [f for path, text in sorted(files.items()) for f in findings(path, text, versions, (matching or {}).get(path))],
        baseline,
    )
    return (*blocked, *debt), baseline


def shrink(root: Path, baseline: Json | None, findings: Sequence[Finding]) -> None:
    """A successful full check removes resolved entries; a partial check never rewrites the baseline."""
    if baseline is not None:
        debt = [f for f in findings if f.key == KEY and not f.blocking]
        store.write(
            root / PATH, (json.dumps({**baseline, "files": naming_baseline.by_file(debt)}, indent=2) + "\n").encode()
        )
