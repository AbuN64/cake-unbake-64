"""The part of a preprocessed unit that its object can depend on.

A header declaration that only names a type, a prototype or an extern is read through the names it declares: when no
code of the unit and no declaration that code reaches mentions one of them, it cannot change the object, so it is left
out. Everything that emits code or data (function bodies, definitions, pragmas, the unit's own text) always stays. A
text that does not split cleanly stays whole. The result keys a compile and the proofs built from it, so an edit of a
declaration the unit never reaches is a cache hit for that unit and an edit of one it reaches is not."""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable, Collection, Mapping
from typing import Any

from unbake.contracts import digest

_MARKER = re.compile(r'^# (\d+) "([^"]*)"')
_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\d\w*|\S')
_IDENT = re.compile(r"[A-Za-z_]\w*$")
_TAGGED = frozenset(("struct", "union", "enum"))
_WORDS = frozenset(
    (
        "typedef",
        "extern",
        "static",
        "register",
        "inline",
        "const",
        "volatile",
        "signed",
        "unsigned",
        "int",
        "char",
        "short",
        "long",
        "float",
        "double",
        "void",
        "struct",
        "union",
        "enum",
        "auto",
        "restrict",
        "_Bool",
        "__extension__",
        "__inline",
        "__inline__",
        "__const",
        "__volatile__",
        "__signed__",
        "__restrict",
        "__attribute__",
        "__asm__",
        "__asm",
        "asm",
    )
)
_ATTRIBUTES = frozenset(("__attribute__", "__asm__", "__asm", "asm"))
_WORD = re.compile(r"[A-Za-z_]\w*")
Shelf = Callable[[str, Callable[[], bytes]], bytes]


def dependencies(text: str) -> list[tuple[frozenset[str], frozenset[str], str]] | None:
    """Declared names, referenced words and canonical text, including macros and inline function bodies."""
    text = re.sub(r"\\\r?\n", " ", text)
    chunks = _split(re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S))
    if chunks is None:
        return None
    rows = []
    for raw, tokens in chunks:
        if tokens:
            names, words, canonical = _declared(tokens), set(_WORD.findall(" ".join(tokens))), " ".join(tokens)
        else:
            macro = re.match(r"#\s*define\s+(\w+)", raw)
            names, words, canonical = {macro[1]} if macro else set(), set(_WORD.findall(raw)), " ".join(raw.split())
        rows.append((frozenset(names), frozenset(words), canonical))
    return rows


def _split(text: str) -> list[tuple[str, list[str]]] | None:
    """The top-level declarations and definitions of one file's text in order (lines, tokens), or None when it does
    not balance."""
    chunks: list[tuple[str, list[str]]] = []
    lines: list[str] = []
    tokens: list[str] = []
    parens = braces = 0
    body, last = False, ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0] == "#" and not lines:  # a pragma or any directive the preprocessor passed on stands alone
            chunks.append((line, []))
            continue
        lines.append(line)
        for token in _TOKEN.findall(line):
            tokens.append(token)
            c = token[0]
            if c in "([":
                parens += 1
            elif c in ")]":
                parens -= 1
            elif c == "{":
                if parens == 0 and braces == 0:
                    body = last == ")"
                braces += 1
            elif c == "}":
                braces -= 1
            last = token
            if parens < 0 or braces < 0:
                return None
            if parens == 0 and braces == 0 and ((c == ";") or (c == "}" and body)):
                chunks.append(("\n".join(lines), tokens))
                lines, tokens, body = [], [], False
    return None if lines or parens or braces else chunks


def _bare(tokens: list[str]) -> list[str]:
    """The tokens without attribute and asm groups, whose words name no declaration."""
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] in _ATTRIBUTES and i + 1 < len(tokens) and tokens[i + 1] == "(":
            depth, i = 0, i + 1
            while i < len(tokens):
                depth += (tokens[i] == "(") - (tokens[i] == ")")
                i += 1
                if depth == 0:
                    break
            continue
        out.append(tokens[i])
        i += 1
    return out


def _head(tokens: list[str]) -> list[str]:
    """The tokens outside brace bodies."""
    out, depth = [], 0
    for token in tokens:
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
        elif depth == 0:
            out.append(token)
    return out


def _enumerators(tokens: list[str]) -> set[str]:
    """The constants of every enum body in the tokens: the word that starts each element."""
    names: set[str] = set()
    for i, token in enumerate(tokens):
        if token != "enum":
            continue
        j = i + 1 + (i + 1 < len(tokens) and bool(_IDENT.match(tokens[i + 1])))
        if j >= len(tokens) or tokens[j] != "{":
            continue
        depth, expect = 0, True
        for t in tokens[j:]:
            if t in "{([":
                depth += 1
            elif t in "})]":
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1 and t == ",":
                expect = True
            elif depth == 1 and expect:
                if _IDENT.match(t):
                    names.add(t)
                expect = False
    return names


def _declarator(segment: list[str]) -> str | None:
    """The one name a declarator declares."""
    while segment and segment[-1] == "]":  # arrays: drop the trailing [..] groups
        depth = 0
        while segment:
            t = segment.pop()
            depth += (t == "]") - (t == "[")
            if depth == 0:
                break
    for i, t in enumerate(segment):
        if t == "(":
            if i + 1 < len(segment) and segment[i + 1] == "*":  # a pointer to a function or an array
                return next((w for w in segment[i + 2 :] if _IDENT.match(w) and w not in _WORDS), None)
            return next((w for w in reversed(segment[:i]) if _IDENT.match(w) and w not in _WORDS), None)  # a prototype
    return next((w for w in reversed(segment) if _IDENT.match(w) and w not in _WORDS), None)


def _declared(tokens: list[str]) -> set[str]:
    """Every name the declaration gives (tags, enumerators and the names of its declarators), a superset when unsure."""
    clean = _bare(tokens)
    names = _enumerators(clean)
    names.update(clean[i + 1] for i, t in enumerate(clean[:-1]) if t in _TAGGED and _IDENT.match(clean[i + 1]))
    segments = []
    current: list[Any] = []
    depth = 0
    for token in _head(clean):
        depth += (token in "([") - (token in ")]")
        if token == "," and depth == 0:
            segments.append(current)
            current = []
        elif token != ";":
            current.append(token)
    segments.append(current)
    names.update(n for n in (_declarator(s) for s in segments if s) if n)
    return names


def _pure(tokens: list[str]) -> bool:
    """Whether the chunk only declares: a typedef, an extern, a tag, a prototype; never an object or a body."""
    if not tokens or tokens[-1] != ";" or "=" in _head(tokens):
        return False
    first = next((t for t in tokens if t != "__extension__"), "")
    if first in ("typedef", "extern"):
        return True
    if first in _TAGGED:
        return tokens[-2] == "}" or len(tokens) - tokens.index(first) == 3
    if "{" in tokens or "(" not in tokens:
        return False
    after = tokens.index("(") + 1
    return tokens[after] != "*" and tokens[-2] in (")",)


def _segments(text: str, source: str) -> list[tuple[bool, str]]:
    """The text cut where the file it comes from changes: (the unit's own, the lines)."""
    out: list[tuple[bool, str]] = []
    here, own, lines = "", False, []
    for line in text.split("\n"):
        marker = _MARKER.match(line)
        if marker is None:
            lines.append(line)
            continue
        flags = line[marker.end() :].split()
        if marker[2] != here or "1" in flags or "2" in flags:
            out.append((own, "\n".join(lines)))
            here, own, lines = marker[2], marker[2] == source or marker[2].endswith("/" + source), []
    out.append((own, "\n".join(lines)))
    return out


def headers(
    text: str, source: str
) -> str:  # every file's text but the unit's own: it must compile for unreached declarations to be harmless
    return "\n".join(body for own, body in _segments(text, source) if not own and body.strip()) + "\n"


Summary = tuple[tuple[bool, str, frozenset[str], tuple[str, ...]], ...]  # per chunk: (read always, text, words, names)
_SEEN: dict[str, Summary | None] = {}  # the summaries of header texts parsed by this process


def _summary(text: str) -> Summary | None:
    """What liveness needs of one header file's text; None when it does not split. Every unit that includes the file
    shares it, so it is parsed once per text, not once per unit."""
    chunks = _split(text)
    if chunks is None:
        return None
    out = []
    for lines, tokens in chunks:
        words = frozenset(t for t in tokens if _IDENT.match(t)) if tokens else frozenset(_WORD.findall(lines))
        names = tuple(sorted(_declared(tokens))) if tokens and _pure(tokens) else ()
        out.append((not names, lines, words, names))
    return tuple(out)


def _summarized(text: str, shelf: Shelf | None) -> Summary | None:
    key = digest(text)
    if key not in _SEEN:
        if len(_SEEN) > 4096:
            _SEEN.clear()
        if shelf is None:
            _SEEN[key] = _summary(text)
        else:
            _SEEN[key] = pickle.loads(shelf(key, lambda: pickle.dumps(_summary(text), protocol=5)))
    return _SEEN[key]


def live(text: str, source: str, shelf: Shelf | None = None) -> str:
    """The text a unit's object depends on: line markers and header declarations nothing reachable names are dropped;
    the whole text when it does not split. `shelf(key, produce)` keeps a header's parse between commands."""
    chunks: list[tuple[bool, str, frozenset[str], tuple[str, ...]]] = []
    for own, body in _segments(text, source):
        if not body.strip():
            continue
        if own:
            chunks.append((True, body.strip(), frozenset(_WORD.findall(body)), ()))
        elif (summary := _summarized(body, shelf)) is None:
            return text
        else:
            chunks.extend(summary)
    kept = [root for root, *_ in chunks]
    declares: dict[str, list[int]] = {}
    for n, (_, _, _, names) in enumerate(chunks):
        for name in names:
            declares.setdefault(name, []).append(n)
    seen: set[str] = set()
    work: list[str] = []
    for root, _, words, _ in chunks:
        if root:
            work.extend(words - seen)
            seen.update(words)
    while work:
        for n in declares.get(work.pop(), ()):
            if not kept[n]:
                kept[n] = True
                work.extend(chunks[n][2] - seen)
                seen.update(chunks[n][2])
    return "\n".join(chunk[1] for chunk, keep in zip(chunks, kept, strict=True) if keep) + "\n"


_TOOL_COMMENT = re.compile(
    r"/\*(?:(?!\*/)[\s\S])*?\b(?:m2c|unbake|TODO: decomp|permuter|Generated by)\b[\s\S]*?\*/[ \t]*\n?"
)


def pure_names(statement: str) -> set[str]:
    """The names one statement declares when it is a single pure declaration, else none."""
    chunks = _split(statement)
    tokens = chunks[0][1] if chunks and len(chunks) == 1 else []
    return _declared(tokens) if tokens and _pure(tokens) else set()


def dead(texts: Mapping[str, str], live: Collection[str]) -> dict[str, str]:
    """The header texts without the declarations no live word reaches, and without tool comments."""
    from unbake import dead as _dead

    return _dead.trim(texts, live, _TOOL_COMMENT)
