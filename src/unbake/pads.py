"""Zero pads a version carries in C, left to the layout: the pure text rule behind `unbake layout pads`.

A source file that serves several versions holds, per version, the all-zero data its symbol facts place between two
tables (`#if defined(VERSION_X)` around `unsigned int pad[1] = { 0 };`). Those bytes are no source: the link places
every file's data at the ROM offset of its first remaining name; the gap is filled with zeros (`build._items`), so the
block says nothing the ROM does not. `pads` takes the text of one file and
  reads the top-level `#if` blocks that hold nothing but non-static data definitions with an initializer (the blocks
    `collapse.blocks` reads),
  takes those whose every initializer is zero and whose every name is a symbol fact with a ROM address in each version
    the guard names (the name is then a row the layout keeps),
  removes them, and
  holds a block, and lists it, when another definition active in one of its versions lies on both sides of it (the
    zeros would sit inside one object's data, which the link cannot place apart) or when other text names it.
The rule reads no compiler and knows no project."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

from unbake import collapse

_ZERO = re.compile(r"(?:0[xX]0*|0+)[uUlL]*")
_COMMENT_OR_STRING = re.compile(r"(\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*')|/\*.*?\*/|//[^\n]*", re.S)
_DIRECTIVE = re.compile(r"[ \t]*#")


def inside(table: Mapping[str, tuple[int, int | None, int | None, str]]) -> set[str]:
    """The names of one version whose bytes need no owner: the nearest fact before them in ROM order, in the same
    mapping, is data (code before them would leave the bytes to a stand-in of their own, `repo.sync`)."""
    out: set[str] = set()
    before: tuple[int, str] | None = None  # (rom - vram, type) of the fact before
    for rom, name in sorted((fact[1], n) for n, fact in table.items() if fact[1] is not None):
        vram, kind = table[name][0], table[name][3]
        if before is not None and before[0] == rom - vram and before[1] != "func":
            out.add(name)
        before = (rom - vram, kind)
    return out


@dataclass
class Pads:
    text: str
    removed: list[dict[str, object]] = field(default_factory=list)
    held: list[dict[str, object]] = field(default_factory=list)


def clean(text: str) -> str:
    """The text with comments blanked (newlines kept, so lines stay lines) and each string literal as `""`."""
    return _COMMENT_OR_STRING.sub(
        lambda m: ('""' if m[1][0] == '"' else "''") if m[1] else re.sub(r"[^\n]", " ", m[0]), text
    )


def zeros(body: str) -> bool:
    """Whether every statement of `body` (data definitions with an initializer) initializes nothing but zeros."""
    statements = [s for s in collapse.mask(body).split(";") if s.strip()]
    if not statements:
        return False
    for statement in statements:
        _, eq, init = statement.partition("=")
        tokens = [t for t in re.sub(r"[{}\s]", "", init).split(",") if t]
        if not eq or not tokens or not all(_ZERO.fullmatch(t) for t in tokens):
            return False
    return True


def _active(lines: Sequence[str], versions: Sequence[str]) -> list[frozenset[str]]:
    """Per line, the versions the line is compiled in (every version where a guard is anything but version
    definedness; an `#else` takes the versions its `#if` leaves)."""
    everyone = frozenset(versions)
    out: list[frozenset[str]] = []
    stack: list[tuple[frozenset[str], frozenset[str] | None]] = []  # (versions active outside, the branch's own)
    current = everyone
    for line in lines:
        found = collapse.OPEN.match(line)
        if found:
            guard = collapse.guard_versions(found[2], found[1], versions)
            stack.append((current, guard))
            current = current & (guard if guard is not None else everyone)
        elif collapse.MID.match(line) and stack:
            outer, guard = stack[-1]
            if line.split("#", 1)[1].split(None, 1)[0].startswith("else"):
                current = outer & (everyone - guard if guard is not None else everyone)
            else:
                current = outer  # an #elif: its own test is unread, so it is active wherever the #if's outside is
        elif collapse.END.match(line) and stack:
            current = stack.pop()[0]
        out.append(current)
    return out


def _content(lines: Sequence[str], skip: Collection[int]) -> list[int]:
    """The lines (outside `skip`) that start a data definition with an initializer at file scope or a static inside a
    function: bytes in the object's data that stay."""
    out: list[int] = []
    depth = 0
    saved: list[int] = []
    for i, line in enumerate(lines):
        if collapse.OPEN.match(line):
            saved.append(depth)
        elif collapse.MID.match(line) and saved:
            depth = saved[-1]
        elif collapse.END.match(line) and saved:
            saved.pop()
        if _DIRECTIVE.match(line):
            continue
        text = collapse.mask(line)
        if i not in skip and (
            (depth == 0 and re.search(r"(?<![=!<>])=(?!=)", text)) or (depth > 0 and re.search(r"\bstatic\b", text))
        ):
            out.append(i)
        depth = max(0, depth + text.count("{") - text.count("}"))
    return out


def pads(text: str, versions: Sequence[str], known: Mapping[str, Collection[str]]) -> Pads:
    """The text of one file without its removable zero pads; `known` is, per version, the names the layout keeps (`inside`)."""
    source = clean(text)
    lines = text.split("\n")
    cleaned = source.split("\n")
    result = Pads(text)
    found = [b for b in collapse.blocks(cleaned, versions) if zeros("\n".join(b.body))]
    if not found:
        return result
    active = _active(cleaned, versions)
    skip = {i for b in found for i in range(b.start, b.end)}
    content = _content(cleaned, skip)
    drop: list[collapse.Block] = []
    for b in found:
        where: dict[str, object] = {"line": b.start + 1, "names": b.names, "versions": sorted(b.versions)}
        lost = [v for v in sorted(b.versions) if not all(n in known.get(v, ()) for n in b.names)]
        inner = [i for i in range(b.start, b.end)]
        named = any(
            re.search(rf"\b{re.escape(n)}\b", "\n".join(cleaned[i] for i in range(len(cleaned)) if i not in inner))
            for n in b.names
        )
        sandwiched = [
            v
            for v in sorted(b.versions)
            if any(i < b.start and v in active[i] for i in content)
            and any(i >= b.end and v in active[i] for i in content)
        ]
        if lost:
            result.held.append({**where, "why": f"no data fact in {', '.join(lost)}"})
        elif named:
            result.held.append({**where, "why": "named by other text of the file"})
        elif sandwiched:
            result.held.append({**where, "why": f"between other data in {', '.join(sandwiched)}"})
        else:
            drop.append(b)
            result.removed.append(where)
    gone = {i for b in drop for i in range(b.start, b.end)}
    kept = [line for i, line in enumerate(lines) if i not in gone]
    out: list[str] = []
    for i, line in enumerate(lines):
        if i in gone:
            continue
        # one blank line of the gap goes with the block it separated
        if not line.strip() and out and not out[-1].strip() and i > 0 and (i - 1) in gone:
            continue
        out.append(line)
    result.text = "\n".join(out if drop else kept)
    return result
