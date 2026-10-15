"""Dead header declarations: what no live word reaches through the declarations' own references."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Mapping

from unbake import declarations, headers, policy

_WORD = re.compile(r"[A-Za-z_]\w*")


def _words(text: str, scope: str) -> set[str]:
    return set(_WORD.findall(policy.scopes(text)[scope]))


def trim(texts: Mapping[str, str], live: Collection[str], tool_comment: re.Pattern[str]) -> dict[str, str]:
    """The header texts with every pure declaration unreachable from the roots cut out. The roots are `live` (the words
    of every C source in every version), every word of every directive and every word of a statement that is not one
    pure declaration. A group of declarations that only name each other is unreachable, so it goes together."""
    clean = {p: tool_comment.sub("", t) for p, t in texts.items()}
    roots = set(live)
    rows: list[tuple[str, int, int, set[str], set[str]]] = []
    declares: dict[str, list[int]] = defaultdict(list)
    for path, text in clean.items():
        roots |= _words(text, "directives")
        for start, end, statement in headers.statements(text):
            if not (names := declarations.pure_names(statement)):
                roots |= _words(statement, "code")
                continue
            for name in names:
                declares[name].append(len(rows))
            rows.append((path, start, end, names, _words(statement, "code")))
    reached: set[int] = set()
    work = list(roots)
    seen = set(roots)
    while work:
        for n in declares.get(work.pop(), ()):
            if n not in reached:
                reached.add(n)
                new = rows[n][4] - seen
                seen |= new
                work.extend(new)
    cuts: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for n, (path, start, end, _, _) in enumerate(rows):
        if n not in reached:
            cuts[path].append((start, end))
    out = {}
    for path, text in clean.items():
        for start, end in sorted(cuts.get(path, ()), reverse=True):
            a = text.rfind("\n", 0, start) + 1
            b = text.find("\n", end)
            b = len(text) if b < 0 else b + 1
            if text[a:start].strip() or text[end:b].strip():
                a, b = start, end
            text = text[:a] + text[b:]
        if text != texts[path]:
            out[path] = text
    return out
