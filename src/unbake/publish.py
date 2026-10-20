"""Landing: a candidate replaces its stand-in line in the file that owns the name, and the builds must still match."""

import re
from collections.abc import Callable, Collection, Mapping, Sequence
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
def measured_apart(text: str, segment: str, name: str) -> bool:
    """Whether the stand-in of `name` only builds without NON_MATCHING (`#ifndef NON_MATCHING`) while the file holds
    one definition of it elsewhere, its draft in a NON_MATCHING block of its own: a measure (NON_MATCHING defined)
    reads the candidate in the draft's place, among the types the draft is written against, not at the stand-in."""
    found = re.search(_stand_in(segment, name), text, re.M)
    if found is None or _block(text, found.start()) is not None:
        return False  # the else side of the draft's own NON_MATCHING block: splice puts the candidate in its place
    if len([i for i in items(text) if i.kind == "function" and i.name == name]) != 1:
        return False
    stack: list[str | None] = []  # the condition of each side open at the stand-in; None when not judged (#elif)
    for m in _DIRECTIVE.finditer(text, 0, found.start()):
        kind, rest = m[1], m[0][m.end(1) - m.start() :].strip()
        if kind in ("if", "ifdef", "ifndef"):
            stack.append(rest if kind == "if" else f"{'!' if kind == 'ifndef' else ''}defined({rest})")
        elif kind == "else" and stack:
            stack[-1] = None if stack[-1] is None else f"!({stack[-1]})"
        elif kind == "elif" and stack:
            stack[-1] = None
        elif kind == "endif" and stack:
            stack.pop()
    # off with NON_MATCHING and on without it: its negation opens a NON_MATCHING side (tree.guard)
    return any(e is not None and tree.guard(f"#if !({e})\n") is not None for e in stack)


def _stand_in(segment: str, name: str) -> str:
    return rf'^[ \t]*{_KINDS}\([ \t]*"{re.escape(segment)}"[ \t]*,[ \t]*{re.escape(name)}[ \t]*\)[ \t]*;[ \t]*\n?'


_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|else|elif|endif)\b[^\n]*(?:\n|$)", re.M)


def _block(text: str, at: int) -> tuple[int, int, int, int] | None:
    """(start of the `#ifdef NON_MATCHING` line, start of its `#else` body, start of its `#endif` line, end of that
    line) of the innermost NON_MATCHING conditional whose `#else` side holds offset `at`, whatever the layout and
    whatever conditionals nest inside either side."""
    stack: list[list[int]] = []  # [start, else body start or -1, is NON_MATCHING]
    best: tuple[int, int, int, int] | None = None
    for m in _DIRECTIVE.finditer(text):
        kind = m[1]
        if kind in ("if", "ifdef", "ifndef"):
            stack.append([m.start(), -1, int(tree.guard(m[0]) is not None)])
        elif kind == "else" and stack and stack[-1][1] < 0:
            stack[-1][1] = m.end()
        elif kind == "endif" and stack:
            start, other, flagged = stack.pop()
            if flagged and 0 <= other <= at < m.start() and (best is None or start > best[0]):
                best = (start, other, m.start(), m.end())
    return best


def wrap(segment: str, name: str, body: str) -> str:
    """A fuzzy body behind NON_MATCHING, falling back to the stand-in."""
    return f'#ifdef NON_MATCHING\n{body.rstrip()}\n#else\nINCLUDE_ASM("{segment}", {name});\n#endif\n'


def _wrap_over(body: str, fallback: str) -> str:
    """A fuzzy body behind NON_MATCHING over the else side `fallback`, which keeps its stand-ins."""
    return f"#ifdef NON_MATCHING\n{body.rstrip()}\n#else\n{fallback}#endif\n"


def _visible(before: str) -> str:
    """The lines of `before` a declaration inserted right after it can rely on: those at the same preprocessor
    nesting, in the same branch of every conditional still open. A closed conditional, another branch or anything
    after the insertion point is not visible."""
    stack: list[int] = []
    seen = 0
    marked: list[tuple[tuple[int, ...], str]] = []
    for line in before.splitlines(keepends=True):
        kind = _DIRECTIVE.match(line)
        word = kind[1] if kind else ""
        if word in ("else", "elif", "endif") and stack:
            stack.pop()
        marked.append((tuple(stack), line))
        if word in ("if", "ifdef", "ifndef", "else", "elif"):
            seen += 1
            stack.append(seen)
    return "".join(line for frames, line in marked if frames == tuple(stack[: len(frames)]))


def _declarations(declared: str) -> list[str]:
    """The top-level items of the declarations, each whole: a function's closing brace is never an item of its own."""
    return [declared[i.start : i.end].strip() for i in items(declared)]


def _wrapper(source: str, parts: Sequence[Item], k: int) -> set[int]:
    """The indexes of the items that are only the NON_MATCHING wrapper of the candidate's function `parts[k]`: the
    `#if`/`#else`/`#endif` lines of every NON_MATCHING conditional (any spelling of its guard) whose first side holds
    it, and its whole `#else` side, which is the stand-in the candidate replaces. The function's own `#include` and
    declarations in the first side are not wrapper and stay."""
    stack: list[list[int]] = []  # [index of the opening directive, index of its #else or -1]
    dropped: set[int] = set()
    for j, part in enumerate(parts):
        if part.kind != "directive":
            continue
        word = re.match(r"#[ \t]*(\w+)", source[part.start : part.end])
        kind = word[1] if word else ""
        if kind in ("if", "ifdef", "ifndef"):
            stack.append([j, -1])
        elif kind in ("else", "elif") and stack and stack[-1][1] < 0:
            stack[-1][1] = j
        elif kind == "endif" and stack:
            opened, other = stack.pop()
            if tree.guard(source[parts[opened].start : parts[opened].end] + "\n") and opened < k < (
                other if other >= 0 else j
            ):
                dropped |= {opened, j, *(range(other, j) if other >= 0 else ())}
    return dropped


def _groups(text: str, parts: Sequence[Item], indexes: Sequence[int]) -> list[list[int]]:
    """The `indexes` of `parts` (comments left out) cut into top-level groups: one item outside any conditional, or a
    whole conditional from its `#if` to its `#endif` with everything inside."""
    groups: list[list[int]] = []
    depth = 0
    for j in indexes:
        if parts[j].kind == "comment":
            continue
        word = re.match(r"#[ \t]*(\w+)", text[parts[j].start : parts[j].end]) if parts[j].kind == "directive" else None
        kind = word[1] if word else ""
        if depth == 0:
            groups.append([])
        groups[-1].append(j)
        depth += 1 if kind in ("if", "ifdef", "ifndef") else -1 if kind == "endif" and depth else 0
    return groups


def _group_text(text: str, parts: Sequence[Item], group: Sequence[int]) -> str:
    return _norm("\n".join(text[parts[j].start : parts[j].end] for j in group))


def _held(text: str) -> frozenset[str]:
    """Every group of `text` (see `_groups`) at every depth, whitespace-normalized: the top-level ones and, inside
    each conditional, the groups of its branches. What a whole-file candidate that carries the rest of its file brings
    again, also when the file holds it only behind a guard (a NON_MATCHING block shared by several functions)."""
    parts = items(text)
    found: set[str] = set()

    def walk(indexes: Sequence[int]) -> None:
        for group in _groups(text, parts, indexes):
            found.add(_group_text(text, parts, group))
            if len(group) > 2:
                walk(group[1:-1])

    walk(range(len(parts)))
    return frozenset(found)


def _directive(text: str, part: Item) -> str:
    """The word of a directive item (`if`, `ifdef`, `else`, `endif`, `include`, ...), "" for any other item."""
    word = re.match(r"#[ \t]*(\w+)", text[part.start : part.end]) if part.kind == "directive" else None
    return word[1] if word else ""


def _visible_groups(before: str) -> frozenset[str]:
    """Each whole conditional (`#if` to its `#endif`) of `before` that text inserted right after it sees, whitespace-
    normalized: one that sits at top level or in a branch still open at the end of `before`. A conditional inside a
    closed one, or in a branch that ended, is not seen there."""
    parts = [p for p in items(before) if p.kind != "comment"]
    stack: list[tuple[int, int]] = []  # per open conditional: (index of its `#if`, the branch it sits in)
    branch = seen = 0
    closed: list[tuple[int, int, int]] = []  # (the branch it sits in, index of its `#if`, index of its `#endif`)
    for j, part in enumerate(parts):
        word = _directive(before, part)
        if word in ("if", "ifdef", "ifndef"):
            stack.append((j, branch))
            seen += 1
            branch = seen
        elif word in ("else", "elif") and stack:
            seen += 1
            branch = seen
        elif word == "endif" and stack:
            opened, branch = stack.pop()
            closed.append((branch, opened, j))
    open_ = {branch, *(b for _, b in stack)}
    return frozenset(
        _norm("\n".join(before[p.start : p.end] for p in parts[lo : hi + 1])) for b, lo, hi in closed if b in open_
    )


def _wanted(groups: Sequence[Sequence[tuple[str, int]]], before: str) -> list[tuple[str, int]]:
    """The items of the candidate's groups that text inserted right after `before` still needs: a single item already
    visible there (same nesting, same branch) or a whole conditional visible there is not brought again. A conditional
    is kept or dropped whole, never one of its directives alone (an `#endif` the file also has must not go)."""
    have = {_norm(d) for d in _declarations(_visible(before))}
    whole = _visible_groups(before)
    out: list[tuple[str, int]] = []
    for group in groups:
        if len(group) == 1 and _norm(group[0][0]) in have:
            continue
        if len(group) > 1 and _norm("\n".join(t for t, _ in group)) in whole:
            continue
        out += group
    return out


def _pieces(source: str, name: str, known: Collection[str] = ()) -> tuple[list[list[tuple[str, int]]], str, int]:
    """([[(item, its first line in `source`)] per group], body, the body's first line): the candidate's definition of
    `name`, and every other top-level item it brings, each with where `source` has it, cut into the groups of
    `_groups` (a conditional is one group). A whole group after the definition that the file already holds (`known`,
    from `_held`) is the file's own later text, not something the candidate brings: a whole-file candidate does not
    duplicate the functions and stand-ins after it."""
    parts = items(source)
    mine = [k for k, p in enumerate(parts) if p.kind == "function" and p.name == name]
    if len(mine) != 1:
        raise Refusal(Finding("land.request", f"The candidate must define {name} exactly once.", unit=name))
    k = mine[0]
    dropped = _wrapper(source, parts, k)
    first = k
    while first > 0 and parts[first - 1].kind == "comment":
        first -= 1  # the comments written above the function travel with it
    body = source[parts[first].start : parts[k].end].strip() + "\n"
    after = [j for j in range(k + 1, len(parts)) if j not in dropped]
    again = {j for g in _groups(source, parts, after) if _group_text(source, parts, g) in known for j in g}
    rest = [j for j, p in enumerate(parts) if not first <= j <= k and j not in dropped | again and p.kind != "comment"]
    texts = [(source[parts[j].start : parts[j].end].strip(), source.count("\n", 0, parts[j].start) + 1) for j in rest]
    opened = sum(bool(re.match(r"#\s*if", t)) for t, _ in texts)
    closed = sum(bool(re.match(r"#\s*endif", t)) for t, _ in texts)
    if opened != closed:
        raise Refusal(Finding("land.request", f"The candidate wraps {name} in a conditional.", unit=name))
    line = dict(zip(rest, texts, strict=True))
    groups = [[line[j] for j in group] for group in _groups(source, parts, rest)]
    return groups, body, source.count("\n", 0, parts[first].start) + 1


def candidate(source: str, name: str | None) -> tuple[str, str]:
    """(declarations, body): the candidate's definition of `name`, and every other top-level item it brings. A data
    candidate (no `name`) is all body."""
    if name is None:
        return "", source.strip() + "\n"
    groups, body, _ = _pieces(source, name)
    return "\n".join(t for group in groups for t, _ in group), body


Origin = tuple[str, int]  # ("disk" | "candidate", the line of that text the output line came from)
Piece = tuple[str, str, int]  # (origin, text, the line of that origin the text starts at)


def _traced(pieces: Sequence[Piece]) -> tuple[str, list[Origin]]:
    """The joined text of (origin, text, first line) pieces, and per output line the piece and line it came from."""
    out = "".join(t for _, t, _ in pieces)
    starts, offset = [], 0
    for _, t, _ in pieces:
        starts.append(offset)
        offset += len(t)
    origins: list[Origin] = []
    k, at = 0, 0
    while at < len(out):
        while k + 1 < len(pieces) and starts[k + 1] <= at:
            k += 1
        kind, t, first = pieces[k]
        origins.append((kind, first + t.count("\n", 0, at - starts[k])))
        at = out.find("\n", at) + 1 or len(out)
    return out, origins


def _definition(text: str, name: str) -> Item:
    """The one top-level definition with an initializer that `text` holds for the data `name`."""
    own = re.compile(rf"\b{re.escape(name)}\b\s*(?:\[[^\]]*\]\s*)*=")
    found = [i for i in items(text) if i.kind == "decl" and own.search(text[i.start : i.end])]
    if len(found) != 1:
        raise Refusal(
            Finding(
                "land.duplicate" if re.search(rf"\b{re.escape(name)}\b", text) else "land.request",
                f"{name} has no stand-in line in its file and no single definition to replace.",
                unit=name,
            )
        )
    return found[0]


def splice_traced(
    text: str, segment: str, name: str, source: str, fuzzy: bool, kind: str = "function"
) -> tuple[str, list[Origin]]:
    """`splice`, and per line of the new text where it came from: a line of `text` (the file as it is on disk) or a
    line of the candidate `source`, so a diagnostic naming a line of the new text names a place a reader can open.
    The candidate's other items come along only where the text before the insertion point does not already give them,
    each conditional whole; what the file holds anywhere after the replaced text is not brought again."""
    if kind == "function":
        _pieces(source, name)  # a candidate that does not define `name` once is refused before anything else
    found = re.compile(_stand_in(segment, name), re.M).search(text)
    if found is None and kind == "function":
        raise Refusal(
            Finding(
                "land.duplicate" if re.search(rf"\b{re.escape(name)}\b", text) else "land.request",
                f"{name} has no stand-in line in its file; it is already landed or the file does not own it.",
                unit=name,
            )
        )

    def disk(lo: int, hi: int) -> Piece:
        return "disk", text[lo:hi], text.count("\n", 0, lo) + 1

    def cand(t: str, line: int) -> Piece:
        return "candidate", t, line

    head: list[Piece]
    tail: list[Piece]
    enclosing = None
    if found is None:
        # data already defined in C: its definition is the text the candidate replaces; the build proves the bytes
        item = _definition(text, name)
        start, end = item.start, item.end
    elif enclosing := _block(text, found.start()):
        start, end = enclosing[0], enclosing[3]
    else:
        start, end = found.start(), found.end()
    if kind == "function":
        groups, body, body_line = _pieces(source, name, _held(text[:start] + text[end:]))
    else:
        groups, body, body_line = [], source.strip() + "\n", 1
    if found is None:
        head, tail = [cand(body.rstrip("\n"), body_line)], []
    else:
        head, tail = [cand(body, body_line)], []
        if fuzzy:
            head = [cand("#ifdef NON_MATCHING\n" + body.rstrip() + "\n#else\n", body_line)]
            tail = [cand("INCLUDE_ASM(" + f'"{segment}", {name});\n#endif\n', body_line)]
        if enclosing:
            # the NON_MATCHING body is the old candidate: the new one takes its place, and the else side (its nested
            # version guards and the other stand-ins) stays; the stand-in itself goes unless the candidate is only fuzzy
            _, other, close, _ = enclosing
            if fuzzy:
                tail = [disk(other, close), cand("#endif\n", body_line)]
            else:
                head, tail = [cand(body, body_line)], [disk(other, found.start()), disk(found.end(), close)]
    wanted = _wanted(groups, text[:start])
    pieces = [disk(0, start)]
    pieces += [cand(d + "\n", line) for d, line in wanted]
    pieces += [cand("\n", wanted[0][1])] if wanted else []
    pieces += head + tail + [disk(end, len(text))]
    return _traced([p for p in pieces if p[1]])


def replace_traced(current: str, name: str, source: str) -> tuple[str, list[Origin]]:
    """`current` with the definition of `name` it already holds replaced by the candidate's (a landed function), and
    per line of the new text where it came from. The candidate's other items come along as `splice_traced` brings
    them: a whole-file candidate does not bring its file again. When the candidate's definition carries the comment
    written above it, that comment replaces the one above the landed definition."""
    _, lead, _ = _pieces(source, name)  # a candidate that does not define `name` once is refused first
    found = items(current)
    mine = [k for k, p in enumerate(found) if p.kind == "function" and p.name == name]
    if len(mine) != 1:
        raise Refusal(Finding("land.request", f"{name} has no single definition to replace.", unit=name))
    top = mine[0]
    if lead.startswith(("/*", "//")):  # the body starts with the comment written above the candidate's definition
        while top > 0 and found[top - 1].kind == "comment":
            top -= 1
    first, last = found[top].start, found[mine[0]].end
    groups, body, body_line = _pieces(source, name, _held(current[:first] + current[last:]))
    wanted = _wanted(groups, current[:first])
    pieces: list[Piece] = [("disk", current[:first], 1)]
    pieces += [("candidate", t + ("\n" if k < len(wanted) - 1 else "\n\n"), n) for k, (t, n) in enumerate(wanted)]
    pieces += [("candidate", body.strip(), body_line), ("disk", current[last:], current.count("\n", 0, last) + 1)]
    return _traced([p for p in pieces if p[1]])


def splice(text: str, segment: str, name: str, source: str, fuzzy: bool, kind: str = "function") -> str:
    """`text` with the stand-in line (or earlier fuzzy block) of `name` replaced by the candidate, and the
    candidate's other top-level items the file lacks added right above it, after every include that precedes it. Data
    that the file defines in C has that definition replaced."""
    return splice_traced(text, segment, name, source, fuzzy, kind)[0]


# ---------------------------------------------------------------- planning
@dataclass
class Reads:
    """What a plan read, by sha: the journal refuses the write when any of it changed."""

    config: Config
    reads: dict[str, str | None]


@dataclass(frozen=True)
class Owner:
    """One file that owns a name, the segment it places it in, and the versions it owns it in."""

    path: str
    segment: str
    versions: tuple[str, ...]


@dataclass(frozen=True)
class Landing:
    plan: Plan
    reads: Reads
    versions: tuple[str, ...]
    name: str
    owners: tuple[Owner, ...]
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


def owners(captured: tree.Tree, name: str, version: str | None = None) -> tuple[Owner, ...]:
    """The files that own `name`, each with the versions it owns it in and its segment. A version is landed in the file
    that owns the name there, so versions owned by different files are different owners of one request, each given the
    same candidate. With `version`, only the file that owns it in that version, and the versions it holds it in: the
    other owners are another context, not part of a request that names the version."""
    held = sorted({v for m in captured.modules for v, names in m.owns.items() if name in names})
    if not held:
        raise Refusal(Finding("land.request", f"{name} is owned by no file in any version.", unit=name))
    if version in held:
        chosen = captured.owner(version, name)
        held = [v for v in held if captured.owner(v, name) == chosen]
    paths: dict[str, list[str]] = {}
    for v in held:
        paths.setdefault(captured.owner(v, name) or "", []).append(v)
    if "" in paths:
        raise Refusal(
            Finding(
                "land.request",
                f"{name} is owned by no single file in {', '.join(paths[''])}.",
                unit=name,
                versions=tuple(paths[""]),
            )
        )
    found = []
    for path, vs in paths.items():
        module = next(m for m in captured.modules if m.path == path)
        found.append(Owner(path, module.placed.get(vs[0], {}).get(name, module.segment), tuple(vs)))
    return tuple(found)


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
        found = owners(captured, name)
        reads = Reads(config, {})
        old: dict[str, bytes | None] = {p: read_file(reads, p) for p in (*(o.path for o in found), *extras)}
        writes: dict[str, bytes | None] = {**extras}
        for owner in found:
            current = old[owner.path]
            if current is None:
                raise Refusal(Finding("land.request", f"{owner.path} is missing.", path=owner.path, unit=name))
            writes[owner.path] = splice(current.decode(), owner.segment, name, source, fuzzy).encode()
        blocking, debt = judged(config, writes, old)
        refuse_blocking(blocking)
        operation = "fuzzy" if fuzzy else "publish"
        affected = tuple(o.path for o in found)
        plan = make_plan(operation, captured.digest, writes, affected, debt, message or f"{operation} {name}")
        return Landing(plan, reads, tuple(sorted({v for o in found for v in o.versions})), name, found, old)


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
    out: dict[str, float] = {}
    failed: list[Proof] = []
    for owner, version in ((o, v) for o in landing.owners for v in o.versions):
        text = (landing.plan.writes[owner.path] or b"").decode()
        proof = cast(Callable[..., Proof], native.measure)(config, version, landing.name, owner.path, text)
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
