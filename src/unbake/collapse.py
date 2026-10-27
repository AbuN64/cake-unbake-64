"""Version branches that differ only by name, into one definition: the pure text rule behind `unbake layout collapse`.

A source file that serves several versions holds a data table once per version, each in its own `#if defined(VERSION_X)`
block, each under the name its version's symbol facts give it (the name spells the table's address there). When the
blocks are the same text once the names they define are set aside, the table is one fact and one definition. `collapse`
takes the text of one file and
  reads the top-level `#if` blocks that hold nothing but non-static data definitions with an initializer (guards are
    `defined(VERSION_X)` joined by `!`, `&&`, `||` and parentheses; any other test, an `#else`, a nested `#if` or a
    block inside braces is left alone),
  groups the blocks whose text is equal with the defined names replaced by their ordinal (comments ignored),
  replaces each group by its first block (the table named by `names_from` when the group has it): one block guarded by
    the union of the group's versions, no guard when the union is every version, and
  returns, per version, the (name, original name) pairs the version's alias file must now carry: the one definition
    links in a version whose fact spells the table under another name as a second name of the same address.
A group stays as it was, and is listed, when the blocks' versions overlap, a name has no fact in its version, the
chosen name is a fact of a version at another address, or a definition of another block that is active in one of the
group's versions stands between the blocks (the move would reorder that version's data). Real regional content differs
in text and never groups. The rule reads no compiler and knows no project."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

_TOKEN = re.compile(r"(\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*')|/\*.*?\*/|//[^\n]*", re.S)
OPEN = re.compile(r"[ \t]*#[ \t]*(if|ifdef|ifndef)\b(.*)")
MID = re.compile(r"[ \t]*#[ \t]*(else|elif)\b")
END = re.compile(r"[ \t]*#[ \t]*endif\b")
_GUARD = re.compile(r"\s*(defined\s*\(\s*VERSION_(\w+)\s*\)|defined\s+VERSION_(\w+)|[()!]|&&|\|\|)")
_NAME = re.compile(r"[A-Za-z_]\w*")
_TOKEN_OR_NAME = re.compile(r"(\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*')|[A-Za-z_]\w*")


@dataclass
class Block:
    start: int  # line index of the #if
    end: int  # line index after the #endif
    versions: frozenset[str]
    body: list[str]
    names: list[str]
    shape: str


@dataclass
class Collapse:
    text: str
    groups: list[dict[str, object]] = field(default_factory=list)
    held: list[dict[str, object]] = field(default_factory=list)
    aliases: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    @property
    def removed(self) -> int:
        return sum(int(g["blocks"]) - 1 for g in self.groups)  # type: ignore[call-overload]


def version_key(token: str) -> str:
    """VERSION_US_REV1 -> us-rev1."""
    return token.lower().replace("_", "-")


def guard_versions(test: str, kind: str, versions: Sequence[str]) -> frozenset[str] | None:
    """The versions an `#if` test is true for, or None when the test is anything but version definedness."""
    if kind == "ifdef":
        found = re.fullmatch(r"\s*VERSION_(\w+)\s*", test)
        return frozenset([version_key(found[1])]) if found and version_key(found[1]) in versions else None
    if kind == "ifndef":
        found = re.fullmatch(r"\s*VERSION_(\w+)\s*", test)
        return frozenset(versions) - {version_key(found[1])} if found else None
    tokens: list[str] = []
    pos = 0
    test = test.strip()
    while pos < len(test):
        found = _GUARD.match(test, pos)
        if found is None:
            return None
        pos = found.end()
        word = found[2] or found[3]
        if word is not None:
            if version_key(word) not in versions:
                return None
            tokens.append(f"({version_key(word)!r} in v)")
        else:
            tokens.append({"&&": " and ", "||": " or ", "!": " not "}.get(found[1], found[1]))
    try:
        return frozenset(v for v in versions if eval("".join(tokens), {"__builtins__": {}}, {"v": {v}}))
    except SyntaxError:
        return None


def strip(text: str) -> str:
    """The text with its comments removed; string and character literals stay."""
    return _TOKEN.sub(lambda m: m[1] or " ", text)


def mask(text: str) -> str:
    """The text with comments removed and strings blanked (no braces or semicolons are left in them)."""
    return _TOKEN.sub(lambda m: '""' if m[1] and m[1][0] == '"' else "''" if m[1] else " ", text)


def definitions(body: str) -> list[str] | None:
    """The names the statements of `body` define, in order, when every statement is a non-static data definition with
    an initializer; None for anything else."""
    clean = mask(body)
    names: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(clean):
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
            if depth < 0:
                return None
        elif ch == ";" and depth == 0:
            statement = clean[start:i].strip()
            start = i + 1
            head, eq, _ = statement.partition("=")
            if not eq or "(" in head or "{" in head:
                return None
            words = head.split()
            if not words or "static" in words or "extern" in words or "typedef" in words:
                return None
            declarator = re.sub(r"(\[[^\]]*\]\s*)+$", "", head.strip()).rstrip()
            found = _NAME.findall(declarator.split()[-1] if declarator.split() else "")
            if (
                not found
                or not declarator.endswith(found[-1])
                or (len(declarator.split()) < 2 and "*" not in declarator)
            ):
                return None
            names.append(found[-1])
    if depth != 0 or clean[start:].strip():
        return None
    return names or None


def shape(body: str, names: Sequence[str]) -> str:
    """The comment-free text of a block with each defined name (outside string literals) replaced by its ordinal."""
    ordinal = {name: f"\x00{n}\x00" for n, name in enumerate(names)}
    text = _TOKEN_OR_NAME.sub(lambda m: m[0] if m[1] else ordinal.get(m[0], m[0]), strip(body))
    return " ".join(text.split())


def blocks(lines: Sequence[str], versions: Sequence[str]) -> list[Block]:
    """The top-level collapsible blocks of a file, in order."""
    out: list[Block] = []
    braces = 0
    i = 0
    clean = [mask(line) for line in lines]  # block comments spanning lines are rare in these files; masked per line
    while i < len(lines):
        found = OPEN.match(lines[i])
        if found is None or braces != 0:
            braces += clean[i].count("{") - clean[i].count("}")
            i += 1
            continue
        depth = 1
        j = i + 1
        simple = True
        while j < len(lines) and depth:
            if OPEN.match(lines[j]):
                depth += 1
                simple = False
            elif END.match(lines[j]):
                depth -= 1
            elif MID.match(lines[j]) and depth == 1:
                simple = False
            j += 1
        if depth:
            break
        guard = guard_versions(found[2], found[1], versions) if simple else None
        body = list(lines[i + 1 : j - 1])
        names = definitions("\n".join(body)) if guard else None
        if guard and names:
            out.append(Block(i, j, guard, body, names, shape("\n".join(body), names)))
        i = j
    return out


def _between(lines: Sequence[str], found: Sequence[Block], first: Block, other: Block, members: set[int]) -> bool:
    """Whether anything active in a version of `other` stands between `first` and `other` outside the group."""
    owner = {n: b for b in found for n in range(b.start, b.end)}
    for n in range(first.end, other.start):
        block = owner.get(n)
        if block is None:
            if mask(lines[n]).strip():
                return True
        elif id(block) not in members and block.versions & other.versions:
            return True
    return False


def _one(
    lines: list[str], versions: Sequence[str], facts: Mapping[str, Mapping[str, int]], names_from: str, skip: set[str]
) -> tuple[list[str], dict[str, object] | None, list[dict[str, object]], dict[str, list[tuple[str, str]]]]:
    found = blocks(lines, versions)
    groups: dict[tuple[str, int], list[Block]] = {}
    for block in found:
        groups.setdefault((block.shape, len(block.names)), []).append(block)
    held: list[dict[str, object]] = []
    for (_, _), group in groups.items():
        if len(group) < 2:
            continue
        key = group[0].names[0] + "@" + ",".join(sorted(group[0].versions))
        if key in skip:
            continue
        why = _refusal(lines, found, group, facts, names_from)
        if why:
            skip.add(key)
            held.append(
                {"names": group[0].names, "versions": sorted(set().union(*(b.versions for b in group))), "why": why}
            )
            continue
        chosen = next((b for b in group if names_from in b.versions), group[0])
        union = frozenset().union(*(b.versions for b in group))
        aliases: dict[str, list[tuple[str, str]]] = {}
        for block in group:
            for n, name in enumerate(block.names):
                if name != chosen.names[n]:
                    for v in sorted(block.versions):
                        aliases.setdefault(v, []).append((chosen.names[n], name))
        body = list(chosen.body)
        first = min(group, key=lambda b: b.start)
        head = []
        tail = []
        if union != frozenset(versions):
            test = " || ".join(f"defined(VERSION_{v.upper().replace('-', '_')})" for v in versions if v in union)
            head, tail = [f"#if {test}"], ["#endif"]
        drop = {n for b in group for n in range(b.start, b.end)}
        out: list[str] = []
        for n, line in enumerate(lines):
            if n == first.start:
                out += head + body + tail
            if n in drop:
                continue
            out.append(line)
        # a removed block leaves its separating blank line behind: keep one blank line between what remains
        tidy: list[str] = []
        for line in out:
            if not line.strip() and len(tidy) > 1 and not tidy[-1].strip():
                continue
            tidy.append(line)
        done = {"names": chosen.names, "blocks": len(group), "versions": sorted(union)}
        return tidy, done, held, aliases
    return lines, None, held, {}


def _refusal(
    lines: Sequence[str],
    found: Sequence[Block],
    group: Sequence[Block],
    facts: Mapping[str, Mapping[str, int]],
    names_from: str,
) -> str:
    seen: set[str] = set()
    for block in group:
        if seen & block.versions:
            return "versions overlap"
        seen |= block.versions
    chosen = next((b for b in group if names_from in b.versions), group[0])
    for block in group:
        for v in block.versions:
            if v not in facts:
                return f"no facts for {v}"
            for name in block.names:
                if name not in facts[v]:
                    return f"{name} has no fact in {v}"
    for block in group:
        for n, name in enumerate(block.names):
            want = chosen.names[n]
            for v in block.versions:
                if want != name and want in facts[v] and facts[v][want] != facts[v][name]:
                    return f"{want} is another address in {v}"
    members = {id(b) for b in group}
    first = min(group, key=lambda b: b.start)
    for block in group:
        if block is not first and _between(lines, found, first, block, members):
            return "other data of a version stands between the blocks"
    return ""


def collapse(text: str, versions: Sequence[str], facts: Mapping[str, Mapping[str, int]], names_from: str) -> Collapse:
    """The collapsed text of one file, the groups collapsed, the groups held and the aliases each version now needs."""
    lines = text.split("\n")
    result = Collapse(text)
    skip: set[str] = set()
    while True:
        lines, done, held, aliases = _one(lines, versions, facts, names_from, skip)
        result.held += [h for h in held if h not in result.held]
        if done is None:
            break
        result.groups.append(done)
        for v, rows in aliases.items():
            result.aliases.setdefault(v, []).extend(rows)
    result.text = "\n".join(lines)
    return result
