"""One header per source file: the exported prototypes and externs of a translation unit, in the header of that unit.

`unbake layout header --file F.CPP` reads F.CPP and every other tracked source file as text, then:
- the exports of F are its non-static function definitions (`extern <signature>;`) and its non-static data definitions
  (`extern <declaration>;`, initializer dropped); a function F only pulls in with INCLUDE_ASM is exported when every
  declaration of it in the tree is the same prototype;
- they go to F.H (the header F already includes under its own name, else a new one next to the headers of the
  directory's module) once each, with the `#include`s that give their types; an export whose type nothing F includes
  provides, or that F.H already declares differently, is held and listed;
- every unit-local declaration of an export that spells the same prototype is deleted and the unit gets
  `#include "F.H"`; a unit that declares it differently, or defines a type of the same name as one F.H brings, keeps its
  text and is listed.
- a header that only F includes and that holds nothing but includes and declarations of F's exports (the header of a
  unit that was merged into F) is folded into F.H: F's include of it becomes the include of F.H and the file is deleted,
  unless an export of it is held.
The tool edits no other header than F.H and those; declarations of an export in other headers are listed for their
owner."""

from __future__ import annotations

import posixpath
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})
HEADER_SUFFIXES = frozenset({".h", ".hh", ".hpp"})
KEYWORDS = frozenset(
    [
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "int",
        "long",
        "register",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
        "inline",
    ]
)
_PREFIX = frozenset({"struct", "union", "enum", "unsigned", "signed", "long", "short", "const", "volatile", "register"})
_WORD = re.compile(r"[A-Za-z_]\w*")
_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|\S")
_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"\n]+)[>"]', re.M)
_MACRO_HEADER = re.compile(r"#\s*define\s+INCLUDE_ASM\b")
_ASM = re.compile(r"\bINCLUDE_ASM\s*\([^;]*?,\s*(\w+)\s*\)\s*;")
_DEFINED = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)", re.M)
_TAG = re.compile(r"\b(?:struct|union|enum)\s+(\w+)\s*\{")


def blank(text: str) -> str:
    """The text with comments and string and character literals replaced by spaces: same length, same newlines."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        two = text[i : i + 2]
        if two == "/*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
        elif two == "//":
            end = text.find("\n", i)
            end = n if end < 0 else end
        elif text[i] in "\"'":
            end = i + 1
            while end < n and text[end] != text[i] and text[end] != "\n":
                end += 2 if text[end] == "\\" else 1
            end = min(end + 1, n)
        else:
            i += 1
            continue
        for k in range(i, end):
            if out[k] != "\n":
                out[k] = " "
        i = end
    return "".join(out)


@dataclass(frozen=True)
class Stmt:
    start: int
    end: int
    text: str  # blanked, whitespace collapsed
    pp: bool  # a preprocessor line
    cond: int  # conditional-compilation depth around it
    body: bool  # a function definition


def statements(clean: str) -> list[Stmt]:
    """The top-level statements of blanked text: preprocessor lines, function definitions (to their closing brace) and
    everything else to its `;`."""
    out: list[Stmt] = []
    i, n, cond = 0, len(clean), 0
    while i < n:
        if clean[i].isspace():
            i += 1
            continue
        start = i
        if clean[i] == "#":
            while i < n and clean[i] != "\n":
                i += 2 if clean[i] == "\\" and i + 1 < n else 1
            i = min(i, n)
            word = clean[start + 1 : i].split()[:1]
            kind = word[0] if word else ""
            if kind in ("endif",):
                cond -= 1
            out.append(Stmt(start, i, " ".join(clean[start:i].split()), True, cond, False))
            if kind in ("if", "ifdef", "ifndef"):
                cond += 1
            continue
        paren = brace = 0
        function = False
        prev = ""
        while i < n:
            c = clean[i]
            if c in "([":
                paren += 1
            elif c in ")]":
                paren -= 1
            elif c == "{":
                if brace == 0 and paren == 0 and prev == ")":
                    function = True
                brace += 1
            elif c == "}":
                brace -= 1
                if brace == 0 and function:
                    i += 1
                    break
            elif c == ";" and brace == 0 and paren == 0:
                i += 1
                break
            if not c.isspace():
                prev = c
            i += 1
        out.append(Stmt(start, i, " ".join(clean[start:i].split()), False, cond, function))
    return out


def without_guard(stmts: list[Stmt]) -> list[Stmt]:
    """Statements of a header, its include guard (`#ifndef G`, `#define G`, ... `#endif`) not counted as a condition."""
    pps = [s for s in stmts if s.pp]
    if (
        len(pps) >= 3
        and pps[0].text.startswith("#ifndef")
        and pps[1].text.startswith("#define")
        and pps[-1].text == "#endif"
    ):
        return [
            Stmt(s.start, s.end, s.text, s.pp, s.cond - 1, s.body) if s.start > pps[0].start and s is not pps[-1] else s
            for s in stmts
        ]
    return stmts


def _tidy(tokens: Sequence[str]) -> str:
    out = ""
    for t in tokens:
        word_after_word = (t[0].isalnum() or t[0] == "_") and (out[-1].isalnum() or out[-1] in "_*,") if out else False
        if word_after_word or (out and t == "*" and out[-1].isalnum()):
            out += " "
        out += t
    return out


def _split_params(tokens: list[str]) -> list[list[str]]:
    params: list[list[str]] = [[]]
    depth = 0
    for t in tokens:
        if t in "([":
            depth += 1
        elif t in ")]":
            depth -= 1
        if t == "," and depth == 0:
            params.append([])
        else:
            params[-1].append(t)
    return params


def _unname(param: list[str]) -> list[str]:
    tail: list[str] = []
    while param and param[-1] == "]":
        k = len(param) - 1 - param[::-1].index("[")
        tail = param[k:] + tail
        param = param[:k]
    named = len(param) >= 2 and _WORD.fullmatch(param[-1]) and param[-1] not in KEYWORDS and param[-2] not in _PREFIX
    return (param[:-1] if named else param) + tail


@dataclass(frozen=True)
class Decl:
    name: str
    shape: str  # the canonical form: `extern`, parameter names and the initializer gone
    text: str  # the declaration as written, `extern ...;`
    kind: str  # function | data
    types: frozenset[str]
    static: bool
    defined: bool
    cond: int


def describe(stmt: Stmt) -> Decl | None:
    """A function definition or prototype, or a data definition or extern; None for anything else (types, macros)."""
    if stmt.pp:
        return None
    tokens = _TOKEN.findall(stmt.text)
    if not tokens or tokens[0] in ("typedef", "INCLUDE_ASM") or tokens[-1] not in (";", "}"):
        return None
    if tokens[0] in ("struct", "union", "enum") and "{" in tokens and tokens.index("{") < 4:
        return None
    static = "static" in tokens or "inline" in tokens
    head = [t for t in tokens if t not in ("extern", "static", "inline")]
    if len(head) == 3 and head[0] in ("struct", "union", "enum"):
        return None  # `struct Tag;` names a type, it is no object
    if stmt.body:
        head = head[: head.index("{")]
    elif "=" in head:
        head = head[: head.index("=")]
    else:
        head = head[:-1]  # the ';'
    defined = stmt.body or "=" in tokens or (tokens[0] != "extern" and "(" not in head)
    if "(" in head:
        open_at = head.index("(")
        name = head[open_at - 1] if open_at else ""
        if name == ")" or not _WORD.fullmatch(name) or head[open_at + 1 : open_at + 2] == ["*"]:
            return None  # a function-pointer variable: left alone
        if ")" not in head:
            return None  # a statement the splitter cut inside a macro call
        close = len(head) - 1 - head[::-1].index(")")
        params = [_unname(p) for p in _split_params(head[open_at + 1 : close])]
        shape = _tidy([*head[:open_at], "("]) + ",".join(_tidy(p) for p in params) + ")"
        types = {t for t in head[: open_at - 1] if _WORD.fullmatch(t)} | {
            t for p in params for t in p if _WORD.fullmatch(t)
        }
        kind = "function"
    else:
        words = [t for t in head if _WORD.fullmatch(t) and t not in KEYWORDS]
        if not words or head[-1] not in {*words, "]"}:
            return None
        name = next(t for t in reversed(head) if _WORD.fullmatch(t))
        shape = _tidy(head)
        types = {t for t in head if _WORD.fullmatch(t)}
        kind = "data"
        defined = defined and not stmt.body
        if "{" in head or "}" in head:
            return None
    written = "extern " + _tidy_text(head) + ";"
    return Decl(name, shape, written, kind, frozenset(types - KEYWORDS - {name}), static, defined, stmt.cond)


def _tidy_text(head: Sequence[str]) -> str:
    """Tokens joined the way C is written: `s32 *x`, `f32 f(f32, CVector3 *)`."""
    out = ""
    for t in head:
        if not out:
            out = t
        elif t in (",",):
            out += ","
        elif out[-1] == "," or (_WORD.fullmatch(t) and (out[-1].isalnum() or out[-1] in "_*)")) or t == "*":
            out += " " + t if not (out[-1] == "*" and t != "*") else t
            if out[-1:] == "*" and False:
                pass
        else:
            out += t
    return out


def provided(text: str) -> set[str]:
    """The names a header's own text defines: typedefs, struct/union/enum tags, macros."""
    clean = blank(text)
    names = {m.group(1) for m in _DEFINED.finditer(clean)} | {m.group(1) for m in _TAG.finditer(clean)}
    for s in statements(clean):
        if not s.pp and s.text.startswith("typedef "):
            tokens = _TOKEN.findall(s.text)
            if "(" in tokens and tokens.index("(") + 2 < len(tokens) and tokens[tokens.index("(") + 1] == "*":
                names.add(tokens[tokens.index("(") + 2])
            elif len(tokens) >= 2:
                names.add(next(t for t in reversed(tokens[:-1]) if _WORD.fullmatch(t)))
    return names


@dataclass
class Hoist:
    header: str
    created: bool
    files: dict[str, str] = field(default_factory=dict)
    exported: list[str] = field(default_factory=list)
    removed: dict[str, int] = field(default_factory=dict)
    included: list[str] = field(default_factory=list)
    held: list[dict[str, Any]] = field(default_factory=list)
    elsewhere: list[dict[str, str]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    externs_before: int = 0
    externs_after: int = 0


def include_root(texts: Mapping[str, str], headers: Sequence[str]) -> str:
    """The directory includes are written against: the one under which most `#include "X"` name a tracked header."""
    votes: Counter[str] = Counter()
    names = {h for h in headers}
    for text in texts.values():
        for name in {m.group(1) for m in _INCLUDE.finditer(text)}:
            for h in names:
                if h.endswith("/" + name):
                    votes[h[: len(h) - len(name)]] += 1
    return votes.most_common(1)[0][0] if votes else ""


def _closure(
    include_names: Sequence[str], headers: Mapping[str, str], root: str, seen: set[str] | None = None
) -> dict[str, set[str]]:
    """For each include name, everything its header and the headers it includes define."""
    seen = set() if seen is None else seen
    out: dict[str, set[str]] = {}
    for name in include_names:
        path = root + name
        if path not in headers or path in seen:
            out[name] = set()
            continue
        seen.add(path)
        if _MACRO_HEADER.search(headers[path]):
            out[name] = set()  # the INCLUDE_ASM header is no home for a prototype's types
            continue
        names = provided(headers[path])
        for sub in _closure([m.group(1) for m in _INCLUDE.finditer(headers[path])], headers, root, seen).values():
            names |= sub
        out[name] = names
    return out


def _header_path(source: str, texts: Mapping[str, str], headers: Sequence[str], root: str) -> str:
    stem = posixpath.splitext(posixpath.basename(source))[0].lower()
    for name in (m.group(1) for m in _INCLUDE.finditer(texts[source])):
        if root + name in headers and posixpath.splitext(posixpath.basename(name))[0].lower() == stem:
            return root + name
    upper = posixpath.splitext(source)[1].isupper()
    suffix = ".H" if upper else ".h"
    folder = posixpath.basename(posixpath.dirname(source))
    existing = next(
        (posixpath.dirname(h) for h in sorted(headers) if posixpath.basename(posixpath.dirname(h)) == folder), None
    )
    return posixpath.join(
        existing or posixpath.join(root.rstrip("/"), folder), posixpath.splitext(posixpath.basename(source))[0] + suffix
    )


def _private_headers(
    texts: Mapping[str, str], headers: Mapping[str, str], root: str, path: str, target: str, exports: Mapping[str, Decl]
) -> dict[str, set[str]]:
    """The headers only `target` includes that declare nothing but exports of it: header path to the names declared."""
    users: defaultdict[str, set[str]] = defaultdict(set)
    for t, text in texts.items():
        for name in {m.group(1) for m in _INCLUDE.finditer(text)}:
            if root + name in headers:
                users[root + name].add(t)
    found: dict[str, set[str]] = {}
    for h in {root + m.group(1) for m in _INCLUDE.finditer(texts[target])}:
        if h == path or h not in headers or users[h] != {target} or _MACRO_HEADER.search(headers[h]):
            continue
        names: set[str] = set()
        raw = statements(blank(headers[h]))
        pps = [s for s in raw if s.pp]
        guarded = len(pps) >= 3 and pps[0].text.startswith("#ifndef") and pps[1].text.startswith("#define")
        guard = {pps[0].start, pps[1].start, pps[-1].start} if guarded and pps[-1].text == "#endif" else set()
        for s in without_guard(raw):
            d = describe(s)
            if s.pp:
                if s.start not in guard and not re.match(r"#\s*(include|pragma\s+once)\b", s.text):
                    break
            elif d is None or d.name not in exports or d.defined or d.static or s.cond:
                break
            else:
                names.add(d.name)
        else:
            if names:
                found[h] = names
    return found


def _swap_includes(text: str, drop: Sequence[str], keep: str) -> str:
    """`text` with the includes of `drop` replaced by the include of `keep` (once, where the first stood) or removed."""
    seen = _includes(text, keep)

    def one(m: re.Match[str]) -> str:
        nonlocal seen
        if m.group(1) not in drop:
            return m.group(0)
        if seen:
            return "\0"
        seen = True
        return f'#include "{keep}"'

    out = _INCLUDE.sub(one, text)
    return re.sub(r"^\0\n?", "", out, flags=re.M)


def _local_names(stmts: Sequence[Stmt]) -> set[str]:
    names: set[str] = set()
    for s in stmts:
        if s.pp:
            m = _DEFINED.match(s.text.replace("# define", "#define")) or re.match(r"#\s*define\s+(\w+)", s.text)
            if m:
                names.add(m.group(1))
        elif s.text.startswith("typedef ") or re.match(r"(struct|union|enum)\s+\w*\s*\{", s.text):
            names |= provided(s.text)
    return names


_COMMENT = re.compile(r"/\*.*?\*/|//.*")


def _cut(text: str, spans: list[tuple[int, int]]) -> str:
    for start, end in sorted(spans, reverse=True):
        a = text.rfind("\n", 0, start) + 1
        b = text.find("\n", end)
        b = len(text) if b < 0 else b
        if not text[a:start].strip() and not _COMMENT.sub("", text[end:b]).strip():
            start, end = a, min(b + 1, len(text))
        text = text[:start] + text[end:]
    return text


def _add_include(text: str, stmts: Sequence[Stmt], line: str) -> str:
    """The include after the last `#include` that comes before the first declaration, else first in the file."""
    at = 0
    for s in stmts:
        if not s.pp:
            break
        if re.match(r"#\s*include\b", s.text):
            at = text.find("\n", s.end - 1) + 1 if text.find("\n", s.end - 1) >= 0 else len(text)
    if at == len(text) and not text.endswith("\n"):
        text += "\n"
    return text[:at] + line + "\n" + text[at:]


def hoist(texts: Mapping[str, str], tracked: Sequence[str], target: str, header: str | None = None) -> Hoist:
    """The edits that give `target` its own header (module doc); `texts` holds every tracked source and header."""
    headers = {t: texts[t] for t in tracked if posixpath.splitext(t)[1].lower() in HEADER_SUFFIXES and t in texts}
    sources = [t for t in tracked if posixpath.splitext(t)[1].lower() in SOURCE_SUFFIXES and t in texts]
    root = include_root(texts, list(headers))
    path = header or _header_path(target, texts, list(headers), root)
    spelled = path[len(root) :] if path.startswith(root) else posixpath.relpath(path, posixpath.dirname(target))
    result = Hoist(path, path not in headers)

    parsed = {t: statements(blank(texts[t])) for t in sources}
    own = parsed[target]
    decls: dict[str, list[tuple[Stmt, Decl]]] = {}
    for t, stmts in parsed.items():
        decls[t] = [(s, d) for s in stmts if (d := describe(s)) is not None and d.cond == 0]
    result.externs_before = sum(1 for rows in decls.values() for _, d in rows if not d.defined and not d.static)

    # what the target exports
    exports: dict[str, Decl] = {}
    for _, d in decls[target]:
        if d.defined and not d.static:
            exports.setdefault(d.name, d)
    asm_only = {m.group(1) for m in _ASM.finditer(blank(texts[target]))} - set(exports)
    for name in sorted(asm_only):
        seen = {d.shape: d for t in sources for _, d in decls[t] if d.name == name and not d.defined and not d.static}
        if len(seen) == 1:
            exports[name] = next(iter(seen.values()))
        elif seen:
            result.held.append({"name": name, "reason": "asm-conflict", "have": sorted(seen)})

    private = _private_headers(texts, headers, root, path, target, exports)

    # the header: what it holds, what its includes give
    old = texts.get(path, "")
    header_stmts = without_guard(statements(blank(old)))
    header_decls = {d.name: d for s in header_stmts if (d := describe(s)) is not None and s.cond == 0}
    wanted_includes = [m.group(1) for m in _INCLUDE.finditer(texts[target]) if root + m.group(1) not in private]
    gives = _closure(wanted_includes, headers, root)
    gives = {n: g for n, g in gives.items() if root + n != path}
    visible = provided(old) | {t for n in wanted_includes if root + n in headers for t in gives.get(n, set())}
    target_local = _local_names(own)

    add: list[Decl] = []
    needs: set[str] = set()
    for name, d in exports.items():
        have = header_decls.get(name)
        if have is not None:
            if have.shape != d.shape:
                result.held.append({"name": name, "reason": "header-conflict", "have": have.shape, "want": d.shape})
            continue
        clash = sorted(
            {u.shape for t in sources if t != target for _, u in decls[t] if u.name == name and not u.defined}
            - {d.shape}
        )
        if clash:
            result.held.append({"name": name, "reason": "users-differ", "have": clash, "want": d.shape})
            continue
        missing = sorted(t for t in d.types if t not in visible and t not in target_local and t not in KEYWORDS)
        if target_local & d.types or missing:
            result.held.append(
                {"name": name, "reason": "type-unseen", "types": sorted(target_local & d.types) or missing}
            )
            continue
        add.append(d)
        needs |= {n for n, g in gives.items() if g & d.types and n in wanted_includes}
    declared = {d.name for d in add} | {
        n for n in header_decls if n in exports and header_decls[n].shape == exports[n].shape
    }
    result.exported = sorted(d.name for d in add)

    # the header text
    if add:
        have_inc = {m.group(1) for m in _INCLUDE.finditer(old)}
        incs = [n for n in wanted_includes if n in needs and n not in have_inc]
        incs = list(dict.fromkeys(incs))
        result.included = incs
        new = _compose(old, path, incs, [d.text for d in add])
        if new != old:
            result.files[path] = new
    shown = provided(result.files.get(path, old)) | {t for n in needs for t in gives.get(n, set())} | visible

    # the users
    include_line = f'#include "{spelled}"'
    for t in sources:
        rows = [(s, d) for s, d in decls[t] if d.name in declared and not d.defined and not d.static]
        if not rows:
            continue
        local = _local_names(parsed[t]) if t != target else target_local
        if t != target and local & shown:
            result.held.append({"path": t, "reason": "type-clash", "types": sorted(local & shown)})
            continue
        drop: list[tuple[int, int]] = []
        for s, d in rows:
            want = exports[d.name].shape if d.name in exports else d.shape
            if d.shape != want:
                result.held.append({"path": t, "name": d.name, "reason": "differs", "have": d.shape, "want": want})
            elif local & d.types:
                result.held.append(
                    {"path": t, "name": d.name, "reason": "local-type", "types": sorted(local & d.types)}
                )
            else:
                drop.append((s.start, s.end))
        if not drop:
            continue
        new = _cut(texts[t], drop)
        if not (t == target and private) and not _includes(texts[t], spelled):
            new = _add_include(new, statements(blank(new)), include_line)
        result.files[t] = new
        result.removed[t] = len(drop)
    gone = {h for h, names in private.items() if names <= declared}
    for h in sorted(set(private) - gone):
        result.held.append({"path": h, "reason": "unit-header-kept", "names": sorted(private[h] - declared)})
    if gone:
        spelled_gone = {m.group(1) for m in _INCLUDE.finditer(texts[target]) if root + m.group(1) in gone}
        result.files[target] = _swap_includes(result.files.get(target, texts[target]), sorted(spelled_gone), spelled)
        result.deleted = sorted(gone)
    for h, text in headers.items():
        if h == path or h in gone:
            continue
        for s in statements(blank(text)):
            d = describe(s)
            if d is not None and d.name in declared and not d.defined and not d.static:
                result.elsewhere.append({"header": h, "name": d.name})
    after: defaultdict[str, int] = defaultdict(int)
    for t in sources:
        stmts = statements(blank(result.files.get(t, texts[t])))
        for s in stmts:
            d = describe(s)
            if d is not None and not d.defined and not d.static and s.cond == 0:
                after[t] += 1
    result.externs_after = sum(after.values())
    return result


def _includes(text: str, spelled: str) -> bool:
    return any(
        posixpath.normpath(m.group(1)) == posixpath.normpath(spelled) for m in _INCLUDE.finditer(blank(text) and text)
    )


def _compose(old: str, path: str, includes: Sequence[str], lines: Sequence[str]) -> str:
    if not old:
        from unbake import guards

        guard = guards.name(path)
        body = [f"#ifndef {guard}", f"#define {guard}", *(f'#include "{n}"' for n in includes), *lines, "#endif"]
        return "\n".join(body) + "\n"
    stmts = statements(blank(old))
    text = old
    pps = [s for s in stmts if s.pp]
    guarded = (
        len(pps) >= 3
        and pps[0].text.startswith("#ifndef")
        and pps[1].text.startswith("#define")
        and pps[-1].text == "#endif"
    )
    tail = pps[-1].start if guarded else len(text)
    block = "".join(line + "\n" for line in lines)
    text = text[:tail] + block + text[tail:]
    if includes:
        incs = [s for s in stmts if s.pp and re.match(r"#\s*include\b", s.text) and (not guarded or s.start < tail)]
        anchor = incs[-1].end if incs else (pps[1].end if guarded else 0)
        add = "".join(f'#include "{n}"\n' for n in includes)
        nl = text.find("\n", anchor)
        if incs or guarded:
            nl = len(text) if nl < 0 else nl
            text = text[: nl + 1] + add + text[nl + 1 :]
        else:
            text = add + text
    return text
