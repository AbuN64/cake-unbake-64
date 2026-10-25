"""FAKEMATCH: a statement that exists only for its effect on the compiler (a dead store, a dead clamp, a forced register
form), written with a `/* FAKEMATCH: reason */` note. One detection for the two places that read it:

- the policy gate (`policy`): a waivable finding is accepted when a FAKEMATCH note sits beside it (`WAIVER`);
- the published score (`report`, the maps): a function that builds exact only through a FAKEMATCH, its own or one
  inherited from an inline function or a macro it uses, scores `SCORE` (below 100), not 100 (`tainted`).

A note counts in a version only on the lines that version compiles (`active`: VERSION_<V> defined, every other name
undefined, so NON_MATCHING sides are off). A note inside a function taints the function; a note directly above a
function or a `#define` (only comments and blank lines between) taints that definition. An inline function or a macro
is pasted into its users, so every function whose body names a tainted one is tainted too, transitively; a called
function that is not inline is its own code and taints nobody."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from unbake import publish, tree

WAIVER = re.compile(r"/\*\s*FAKEMATCH:((?:[^*]|\*(?!/))*)\*/")
SCORE = 0.99  # what an exact build through a FAKEMATCH earns: fuzzy, just below exact
_DIRECTIVE = re.compile(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)$")
_DEFINE = re.compile(r"#\s*define\s+([A-Za-z_]\w*)")
_INLINE = re.compile(r"\binline\b|\b__inline(?:__)?\b")


def _truth(expr: str, defined: frozenset[str]) -> bool:
    """An `#if` expression with DEFINED defined and every other name undefined; an unreadable one counts as true."""
    expr = re.sub(r"/\*.*?\*/|//.*", "", expr).strip()
    expr = re.sub(
        r"defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)", lambda m: str((m.group(1) or m.group(2)) in defined), expr
    )
    expr = re.sub(r"\b(?!True\b|False\b)[A-Za-z_]\w*", "False", expr)
    expr = expr.replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"!(?!=)", " not ", expr)
    try:
        return bool(eval(expr, {"__builtins__": {}}, {"True": True, "False": False}))
    except Exception:
        return True


def active(text: str, version: str) -> str:
    """TEXT with every line the matching build of VERSION does not compile blanked (offsets kept)."""
    if "#" not in text:
        return text
    defined = frozenset({tree.macro(version)})
    out: list[str] = []
    stack: list[list[bool]] = []  # [parent on, this branch on, a branch taken]
    for line in text.split("\n"):
        m = _DIRECTIVE.match(line)
        if m:
            d, rest = m.groups()
            top = stack[-1] if stack else None
            parent = top[0] and top[1] if top else True
            if d in ("if", "ifdef", "ifndef"):
                name = (rest.strip().split() or [""])[0]
                on = _truth(rest, defined) if d == "if" else (name in defined) == (d == "ifdef")
                stack.append([parent, on, on])
            elif d == "elif" and top:
                on = (not top[2]) and _truth(rest, defined)
                top[1], top[2] = on, top[2] or on
            elif d == "else" and top:
                top[1], top[2] = not top[2], True
            elif d == "endif" and top:
                stack.pop()
            out.append(" " * len(line))
            continue
        out.append(line if all(e[0] and e[1] for e in stack) else " " * len(line))
    return "\n".join(out)


def _defines(items: list[publish.Item], text: str) -> list[tuple[publish.Item, str, bool]]:
    """(item, name, pasted) of each function and `#define` of TEXT: pasted when users get its body (inline, macro)."""
    out = []
    for item in items:
        if item.kind == "function":
            head = text[item.start : text.find("{", item.start, item.end)]
            out.append((item, item.name, bool(_INLINE.search(head))))
        elif item.kind == "directive" and (m := _DEFINE.match(text, item.start)):
            out.append((item, m.group(1), True))
    return out


def tainted(texts: Mapping[str, str], version: str) -> frozenset[str]:
    """Names (functions, inline functions, macros) whose definition in VERSION carries a FAKEMATCH note, or that use a
    tainted inline function or macro, transitively. TEXTS: C sources and headers by path."""
    parsed: dict[str, tuple[str, list[tuple[publish.Item, str, bool]]]] = {}

    def read(path: str) -> tuple[str, list[tuple[publish.Item, str, bool]]]:
        if path not in parsed:
            act = active(texts[path], version)
            items = publish.items(act)
            parsed[path] = (act, _defines(items, act))
        return parsed[path]

    seeds: set[str] = set()
    pasted: set[str] = set()
    for path, raw in texts.items():
        if "FAKEMATCH" not in raw:
            continue
        act, defs = read(path)
        notes = [m.start() for m in WAIVER.finditer(act)]
        if not notes:
            continue
        for item, name, paste in defs:
            inside = any(item.start <= at < item.end for at in notes)
            above = any(at < item.start and not act[act.find("*/", at) + 2 : item.start].strip(" \t\n") for at in notes)
            if inside or above:
                seeds.add(name)
                if paste:
                    pasted.add(name)
    done = set(seeds)
    spread = set(pasted)
    while spread:
        words = re.compile(r"(?<![\w$])(?:" + "|".join(map(re.escape, sorted(spread))) + r")(?![\w$])")
        found: set[str] = set()
        for path, raw in texts.items():
            if not any(n in raw for n in spread):
                continue
            act, defs = read(path)
            for item, name, paste in defs:
                if name in done:
                    continue
                body = act[item.start : item.end]
                if item.kind == "directive":
                    body = body[_DEFINE.match(body).end() :]  # type: ignore[union-attr]
                if words.search(body):
                    found.add(name)
                    if paste:
                        pasted.add(name)
        done |= found
        spread = {n for n in found if n in pasted}
    return frozenset(done)


def by_version(texts: Mapping[str, str], versions: Iterable[str]) -> dict[str, frozenset[str]]:
    """`tainted` for every version."""
    return {v: tainted(texts, v) for v in versions}
