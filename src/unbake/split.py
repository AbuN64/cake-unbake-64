"""One unit into one file per target: the pure text rule behind `unbake layout split`.

The reverse of `merge`. `split` takes the text of one unit and the target of each symbol it defines, and returns the
text of one file per target:
  a function or variable goes to the file of its target; so does the macro call that places a symbol (an included
  data or code block), by the symbols named in its arguments, and the prototype of a `static` function with it,
  everything else (includes, typedefs, macros, prototypes, externs) is kept in every file,
  a symbol the targets do not name goes to the target that holds most of the unit (and is reported),
  a symbol one file defines and another uses is declared in the user (`static` is dropped from the definition),
  an `#if` block left empty is removed.
The text is read as top-level declarations like `merge` reads it; it needs no compiler and knows no project."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from unbake import merge as unitmerge

Row = dict[str, Any]
_WORD = re.compile(r"[A-Za-z_]\w*")
_STATIC = re.compile(r"\bstatic\b[ \t]*")
_INCLUDE = re.compile(r"^[ \t]*#[ \t]*include\b", re.M)
_EMPTY_IF = re.compile(r"^[ \t]*#[ \t]*if[^\n]*\n(?:[ \t]*\n)*[ \t]*#[ \t]*endif[^\n]*\n?", re.M)


@dataclass
class Statement:
    """A top-level statement with the comment that leads it, the target that owns it (None: kept in every file) and
    the names it defines."""

    start: int
    end: int
    lead: int  # offset where the comment attached to the statement begins
    kind: str
    names: list[str]
    static: bool
    owner: str | None = None
    declares: set[str] = field(default_factory=set)


def _lead(text: str, masked: str, floor: int, start: int) -> int:
    """Where the comment lines directly above the statement begin: the part of the gap after its last blank line, when
    that part is comments only."""
    gap = masked[floor:start]
    at = max(gap.rfind("\n\n"), gap.rfind("\n \n"))
    begin = floor + (at + 2 if at >= 0 else 0)
    wanted = text[begin:start]
    return begin if wanted.strip() and not masked[begin:start].strip() else start


def _statements(text: str, unit: str) -> list[Statement]:
    masked = unitmerge.mask(text)
    found: dict[tuple[int, int], Statement] = {}
    for d in unitmerge.decls(text, unit):
        one = found.setdefault((d.start, d.end), Statement(d.start, d.end, d.start, d.kind, [], d.static))
        if d.kind in ("def", "var", "proto") and d.name:
            one.names.append(d.name)
    out = sorted(found.values(), key=lambda s: s.start)
    floor = 0
    for s in out:
        s.lead = _lead(text, masked, floor, s.start)
        floor = s.end
    return out


def _head(text: str, masked: str, s: Statement) -> str:
    """The declaration part of a statement: up to its body or initialiser."""
    body = masked[s.start : s.end]
    cut = min((body.find(c) for c in "{=" if body.find(c) >= 0), default=len(body.rstrip(";")))
    return text[s.start : s.start + cut]


def _declaration(text: str, masked: str, s: Statement) -> str:
    head = _STATIC.sub("", _head(text, masked, s), count=1).strip()
    return f"{head};" if s.kind == "def" else f"extern {head};"


def split(unit: str, text: str, owner: Mapping[str, str]) -> Row:
    """{"pieces": {target: text}, "unassigned": [names], "unstatic": [names], "declared": {target: [names]},
    "conflicts": [reason]} for the unit `text` whose symbols the targets `owner` (symbol -> target) name."""
    masked = unitmerge.mask(text)
    statements = _statements(text, unit)
    conflicts: list[str] = []
    defined = {n: s for s in statements if s.kind in ("def", "var") for n in s.names}
    for s in statements:
        if s.kind in ("def", "var"):
            targets = {owner[n] for n in s.names if n in owner}
            if len(targets) > 1:
                conflicts.append(
                    f"{', '.join(s.names)} are declared together and belong to {', '.join(sorted(targets))}"
                )
            s.owner = min(targets) if targets else None
        elif s.kind == "proto" and s.names:
            statement = masked[s.start : s.end]
            if (call := _WORD.match(statement.lstrip())) and call.group(0) == s.names[0]:
                places = {owner[w] for w in _WORD.findall(statement[statement.find("(") :]) if w in owner}
                if len(places) > 1:
                    conflicts.append(f"{s.names[0]} places symbols of {', '.join(sorted(places))}")
                s.owner = min(places) if places else None
    held = Counter(s.owner for s in statements if s.owner)
    unassigned = sorted(n for s in statements if s.kind in ("def", "var") and s.owner is None for n in s.names)
    if unassigned:
        if not held:
            conflicts.append("no symbol of the unit has a target")
        else:
            fallback = max(sorted(held), key=lambda t: held[t])
            for s in statements:
                if s.kind in ("def", "var") and s.owner is None:
                    s.owner = fallback
    # a static prototype follows its definition
    for s in statements:
        if s.kind == "proto" and s.names and s.names[0] in defined and defined[s.names[0]].static and s.owner is None:
            s.owner = defined[s.names[0]].owner
    files = sorted({s.owner for s in statements if s.owner})
    shared = {n for s in statements if s.owner is None and s.kind in ("proto", "extern") for n in s.names}
    unstatic: set[str] = set()
    declared: dict[str, list[str]] = {t: [] for t in files}
    for t in files:
        own = " ".join(masked[s.start : s.end] for s in statements if s.owner == t)
        words = set(_WORD.findall(own))
        for name, s in defined.items():
            if s.owner and s.owner != t and name in words and name not in shared:
                declared[t].append(name)
                if s.static:
                    unstatic.add(name)
    pieces: dict[str, str] = {}
    for t in files:
        out: list[str] = []
        at = 0
        for s in statements:
            out.append(text[at : s.lead])
            at = s.end
            if s.owner is None or s.owner == t:
                chunk = text[s.lead : s.end]
                if s.owner == t and s.static and unstatic.intersection(s.names):
                    chunk = chunk[: s.start - s.lead] + _STATIC.sub("", chunk[s.start - s.lead :], count=1)
                out.append(chunk)
                if text[s.end : s.end + 1] == "\n":
                    at = s.end + 1
                    out.append("\n")
            elif text[s.end : s.end + 1] == "\n":
                at = s.end + 1
        out.append(text[at:])
        body = "".join(out)
        while (cleaned := _EMPTY_IF.sub("", body)) != body:
            body = cleaned
        if declared[t]:
            lines = "\n".join(_declaration(text, masked, defined[n]) for n in declared[t])
            includes = list(_INCLUDE.finditer(body))
            where = body.find("\n", includes[-1].end()) + 1 if includes else 0
            if includes and where == 0:
                where = len(body)
            body = f"{body[:where]}{lines}\n{body[where:]}"
        pieces[t] = re.sub(r"\n{3,}", "\n\n", body).strip("\n") + "\n"
    return {
        "pieces": pieces,
        "unassigned": unassigned,
        "unstatic": sorted(unstatic),
        "declared": declared,
        "conflicts": conflicts,
    }
