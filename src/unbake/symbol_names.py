"""Portable source names. One target table owns validation; compilers own symbol decoration."""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from collections import Counter
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path, PurePosixPath

from unbake import config
from unbake.contracts import Finding, Json

_TOKEN = re.compile(r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\'')
_WORD = r"(?:[\w$]|\\[uU][0-9a-fA-F]{4,8})+"
_CONTROL = frozenset(
    ["return", "if", "else", "while", "for", "switch", "case", "goto", "break", "continue", "sizeof", "do"]
)
_ACTION = 'Use a plain C identifier; express an alias with #define and let the compiler decorate extern "C" names.'


@cache
def _table() -> Json:
    return config.load_resource("symbol-names.toml")


@cache
def _issues(
    name: str, kind: str, global_scope: bool, implementation: bool = False, evidenced: bool = False
) -> tuple[str, ...]:
    """Target-qualified reasons a symbol, section, or relative file path cannot be emitted."""
    if kind not in ("symbol", "section", "file"):
        raise ValueError(f"unknown name kind: {kind}")
    out: dict[str, list[str]] = {}
    sdk = evidenced or (
        implementation
        and any(re.fullmatch(pattern, name) for pattern in _table()["implementation_reserved"]["symbol_patterns"])
    )
    for target, row in _table()["target"].items():
        why: list[str] = []
        field = "identifier" if kind == "symbol" else kind
        parts = name.split("/") if kind == "file" else [name]
        for part in parts:
            if not re.fullmatch(row[f"{field}_pattern"], part) or part in (".", ".."):
                why.append("characters")
            limit = row[f"{field}_max"]
            if limit and len(part.encode()) > limit:
                why.append(f"{field} length exceeds {limit} bytes")
            if kind == "file" and re.search(row["file_reserved"], part):
                why.append("reserved file name")
        if kind == "symbol":
            words = _keywords(target)
            if name in words:
                why.append("compiler keyword")
            if not sdk and (
                re.search(row["reserved"], name) or (global_scope and re.search(row["reserved_global"], name))
            ):
                why.append("reserved identifier")
        for reason in dict.fromkeys(why):
            out.setdefault(reason, []).append(target)
    return tuple(f"{name}: {reason} ({', '.join(targets)})" for reason, targets in out.items())


@cache
def _keywords(target: str) -> frozenset[str]:
    return frozenset(w for group in _table()["target"][target]["keyword_sets"] for w in _table()["keywords"][group])


def issues(
    name: str,
    kind: str = "symbol",
    taken: Collection[str] = (),
    global_scope: bool = True,
    implementation: bool = False,
    evidenced: bool = False,
) -> tuple[str, ...]:
    """All target-qualified reasons, including collisions in the caller's namespace. `implementation` admits the SDK's
    own reserved names (`__osDisableInt`), as source admission does for every file. `evidenced`: the caller's project
    evidence declares this name the original SDK name, so it is admitted whatever it looks like (target portability
    other than the reserved-identifier rule still applies)."""
    out = list(_issues(name, kind, global_scope, implementation, evidenced))
    collisions: dict[str, list[str]] = {}
    for target, row in _table()["target"].items():
        folded = kind == "file" and not row["file_case_sensitive"]
        folded |= kind == "symbol" and not row["portable_symbol_case_sensitive"]
        for other in sorted(taken):
            if name == other or (folded and name.casefold() == other.casefold()):
                collisions.setdefault(other, []).append(target)
                break
    out += [f"{name}: collides with {other} ({', '.join(targets)})" for other, targets in collisions.items()]
    return tuple(out)


def _mask(text: str, strings: bool = True) -> str:
    return _TOKEN.sub(lambda m: re.sub(r"[^\n]", " ", m[0]) if strings or m[0].startswith(("/*", "//")) else m[0], text)


def _code(text: str) -> str:
    return re.sub(r"^[ \t]*#(?:\\\n|[^\n])*", lambda m: re.sub(r"[^\n]", " ", m[0]), _mask(text), flags=re.M)


def _heads(head: str, offset: int, global_scope: bool) -> Iterator[tuple[str, int, bool]]:
    """Names declared in a C declaration head, including parameters and comma declarators."""
    offset += len(head) - len(head.lstrip())
    head = head.strip()
    label = re.match(r"(?:(?:case\s+[^:;]+|default|[A-Za-z_]\w*)\s*:\s*)+", head)
    if label:
        offset += label.end()
        head = head[label.end() :]
    if not head or head.split()[0] in _CONTROL or head.startswith(("#", "INCLUDE_")):
        return
    # An assignment or a call is not a declaration. A declaration needs a type before its name.
    pattern = (
        rf"^(?P<type>(?:(?:{_WORD})\s+)+)(?:\*\s*(?:(?:const|volatile|restrict)\s+)*)*"
        rf"(?P<name>{_WORD})(?=\s*(?:[([=,;:]|$))"
    )
    pointer = re.match(rf"^(?:{_WORD}\s+)+\(\s*\*\s*({_WORD})\s*\)", head)
    match = re.match(pattern, head)
    if pointer:
        yield pointer[1], offset + pointer.start(1), global_scope
    elif match:
        name = match["name"]
        if name in {
            "void",
            "char",
            "short",
            "int",
            "long",
            "float",
            "double",
            "signed",
            "unsigned",
            "const",
            "volatile",
            "struct",
            "union",
            "enum",
        }:
            return  # an unnamed parameter, such as unsigned long
        yield name, offset + match.start("name"), global_scope
        tail = head[match.end() :].lstrip()
        if tail.startswith("("):
            end = tail.rfind(")")
            args = tail[1:end] if end >= 0 else tail[1:]
            pos = head.find("(", match.end()) + 1
            for param in re.split(r",(?![^()]*\))", args):
                yield from _heads(param, offset + pos, False)
                pos += len(param) + 1
        elif "," in tail:
            depth = 0
            for separator in re.finditer(r"[()\[\],]", tail):
                char = separator[0]
                if char in "([":
                    depth += 1
                elif char in ")]":
                    depth = max(0, depth - 1)
                elif not depth:
                    extra = re.match(rf"\s*\**\s*({_WORD})\s*(?=[,;=\[]|$)", tail[separator.end() :])
                    if extra:
                        at = head.find(tail, match.end()) + separator.end() + extra.start(1)
                        yield extra[1], offset + at, global_scope
    tag = re.match(rf"(?:typedef\s+)?(?:struct|union|enum)\s+({_WORD})\s*$", head)
    if tag:
        yield tag[1], offset + tag.start(1), global_scope


@dataclass(frozen=True)
class Binding:
    """One lexical declaration, never collapsed by spelling across scopes or conditional copies."""

    name: str
    kind: str
    start: int
    scope: tuple[int, ...]
    owner: str
    type: str
    definition: bool


def _initialized(declarator: str) -> bool:
    """Whether this declarator has an initializer, excluding bounds and following declarators."""
    depth = 0
    for token in re.finditer(r"[()[\],;=]", declarator):
        char = token[0]
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        elif not depth:
            return char == "="
    return False


@lru_cache(maxsize=1)
def bindings(text: str) -> tuple[Binding, ...]:
    """Qualified C declaration facts, including generated/preprocessor names and inactive branches.

    Locations qualify declarations, not linker identities. This lexical reader does not resolve
    conditional activation, macro expansions or C++ templates; consumers must keep those limits.
    """
    code = _code(text)
    start = 0
    stack: list[tuple[str, int, str]] = []
    out: dict[int, Binding] = {}
    scopes: list[tuple[int, int, tuple[int, ...], str]] = []

    def put(
        name: str,
        at: int,
        kind: str,
        owner: str,
        type_: str = "",
        definition: bool = True,
        scope: tuple[int, ...] | None = None,
    ) -> None:
        out[at] = Binding(
            name, kind, at, scope if scope is not None else tuple(s[1] for s in stack), owner, type_, definition
        )

    for token in re.finditer(r"[{};]", code):
        head = code[start : token.start()]
        linkage = bool(re.fullmatch(r"\s*extern\s*", head))
        global_ = not any(s[0] != "linkage" for s in stack)
        record = re.search(rf"\b(struct|union|enum)(?:\s+({_WORD}))?\s*$", head)
        declared = (
            list(_heads(head, start, global_))
            if token[0] in "{;" and not any(s[0] in {"initializer", "enum"} for s in stack)
            else []
        )
        function = next(
            (
                (n, a)
                for n, a, _ in declared
                if re.match(r"\s*\(", code[a + len(n) :]) and not re.search(r"\(\s*\*\s*$", code[start:a])
            ),
            None,
        )
        owner = next((s[2] for s in reversed(stack) if s[0] in {"function", "record", "enum"}), "")
        for name_, at, outer in declared:
            kind = "global"
            type_ = " ".join(code[start:at].strip().split())
            scope = tuple(s[1] for s in stack)
            if record and name_ == record[2]:
                kind = "type"
            elif function and at == function[1]:
                kind = "type" if re.match(r"\s*typedef\b", head) else "function"
            elif function and not outer:
                kind, owner, scope = "parameter", function[0], (*scope, function[1])
                before = code[start:at]
                type_ = " ".join(re.split(r"[(,]", before)[-1].split())
            elif re.match(r"\s*typedef\b", head):
                kind = "type"
            elif stack and stack[-1][0] == "record":
                kind = "member"
            elif any(s[0] == "function" for s in stack):
                kind = "local"
            elif re.match(r"\s*static\b", head):
                kind = "static"
            definition = token[0] == "{" if kind == "function" else True
            if kind in {"global", "local"} and re.search(r"\bextern\b", head):
                definition = _initialized(head[at - start :])
            put(name_, at, kind, owner, type_, definition, scope)
        if token[0] == "{":
            kind = (
                "linkage"
                if linkage
                else "enum"
                if record and record[1] == "enum"
                else "record"
                if record
                else ("initializer" if "=" in head else "function" if function else "body")
            )
            stack.append((kind, token.start(), record[2] or "" if record else function[0] if function else owner))
        elif token[0] == "}" and stack:
            closed = stack.pop()
            scopes.append((closed[1], token.end(), (*tuple(s[1] for s in stack), closed[1]), closed[2]))
            if closed[0] == "enum":
                body = code[closed[1] + 1 : token.start()]
                for m in re.finditer(rf"(?:^|,)\s*({_WORD})\s*(?==|,|$)", body):
                    put(m[1], closed[1] + 1 + m.start(1), "enumerator", closed[2])
            if closed[0] in {"record", "enum"}:
                tail = re.match(rf"\s*({_WORD})\s*(?=[;,])", code[token.end() :])
                if tail:
                    # Distinguish an inline typedef from a variable of an inline record type.
                    prefix = code[: closed[1]].rsplit(";", 1)[-1].rsplit("}", 1)[-1]
                    kind = (
                        "type"
                        if re.search(r"\btypedef\b", prefix)
                        else "member"
                        if stack and stack[-1][0] == "record"
                        else "local"
                        if stack
                        else "global"
                    )
                    definition = not re.search(r"\bextern\b", prefix)
                    put(tail[1], token.end() + tail.start(1), kind, owner, definition=definition)
        start = token.end()
    for loop in re.finditer(r"\bfor\s*\(\s*([^;]*);", code):
        held = min((s for s in scopes if s[0] < loop.start() < s[1]), key=lambda s: s[1] - s[0], default=None)
        for name_, at, _ in _heads(loop[1], loop.start(1), False):
            put(name_, at, "local", held[3] if held else "", scope=(*(held[2] if held else ()), loop.start()))
    masked = _mask(text)
    for macro in re.finditer(rf"^[ \t]*#\s*define\s+({_WORD})(\([^\n]*?\))?", masked, re.M):
        put(macro[1], macro.start(1), "macro", "", scope=())
        if macro[2]:
            for param in re.finditer(_WORD, macro[2]):
                put(param[0], macro.start(2) + param.start(), "parameter", macro[1], scope=(macro.start(1),))
    return tuple(out[at] for at in sorted(out))


def census(texts: Mapping[str, str], revision: str) -> Json:
    """One inventory for declaration facts, source/header paths and their directory views."""
    import bisect

    from unbake import guards

    rows: list[Json] = []
    paths: list[Json] = []
    for path, text in sorted(texts.items()):
        header = config.is_header(path)
        paths.append(
            {
                "path": path,
                "kind": "header" if header else "source",
                "sha256": hashlib.sha256(text.encode(errors="surrogateescape")).hexdigest(),
            }
        )
        if config.is_assembly(path):
            continue
        guard = guards.binding(text) if header else None
        lines = [0, *(m.end() for m in re.finditer("\n", text))]
        for row in bindings(text):
            rows.append(
                {
                    "identity": [path, row.kind, list(row.scope), row.start],
                    "path": path,
                    "name": row.name,
                    "kind": row.kind,
                    "scope": list(row.scope),
                    "owner": row.owner,
                    "line": bisect.bisect_right(lines, row.start),
                    "offset": row.start,
                    "type": row.type,
                    "definition": row.definition,
                    "guard": row.kind == "macro" and row.name == guard,
                    "canonical_guard": guards.name(path) if row.kind == "macro" and row.name == guard else None,
                }
            )
    folders = {
        kind: sorted(
            {
                str(parent)
                for p in paths
                if p["kind"] == kind
                for parent in PurePosixPath(p["path"]).parents
                if str(parent) != "."
            }
        )
        for kind in ("source", "header")
    }
    domains = ("function", "global", "static", "type", "member", "parameter", "local", "macro", "enumerator")
    counts = Counter(r["kind"] for r in rows)
    return {
        "revision": revision,
        "paths": paths,
        "folders": folders,
        "declarations": rows,
        "counts": {kind: counts[kind] for kind in domains},
        "semantic_acceptance": False,
        "limits": [
            "Lexical declarations are not linker identities or semantic naming approval.",
            "Inactive branches are retained; activation and macro expansion require compiler evidence.",
            "Assembly paths are inventoried; assembly definitions and C++ templates are not C declarations.",
        ],
    }


def _finding(path: str, text: str, at: int, reason: str) -> Finding:
    return Finding("source.symbol-name", reason, path=path, line=text.count("\n", 0, at) + 1, action=_ACTION)


def source(path: str, text: str) -> tuple[Finding, ...]:
    """Text-only naming checks, including inactive branches; comments/literals are not declarations."""
    if not (config.is_source(path) or config.is_header(path)) or config.is_assembly(path):
        return ()
    literal, masked = _mask(text, False), _mask(text)
    findings: list[Finding] = []
    # A symbol label follows a declarator and has one string operand. Inline asm statements stay with inline-asm.
    asm = r"\b(?:asm|__asm|__asm__)\s*\("
    for m in re.finditer(asm, literal):
        if not masked[m.start() : m.end()].strip():
            continue
        prefix = literal[: m.start()].rstrip()
        if prefix and (prefix[-1] in ")]" or re.search(r"\w$", prefix)):
            findings.append(_finding(path, text, m.start(), "explicit asm symbol label bypasses target C decoration"))
    hacks = (
        r'\b(?:alias|ifunc|weakref|symver)\s*\(\s*"|^\s*#\s*pragma\s+(?:redefine_extname|weak)\b'
        r'|^\s*#\s*pragma\s+comment\s*\(\s*linker\s*,\s*"/alternatename:'
    )
    hacks += r'|\b_Pragma\s*\(\s*"(?:redefine_extname|weak)\b'
    for m in re.finditer(hacks, literal, re.M):
        if not masked[m.start() : m.end()].strip():
            continue
        findings.append(_finding(path, text, m.start(), "target-specific symbol alias"))
    for m in re.finditer(r'\bextern\s*"C\+\+"', literal):
        if not masked[m.start() : m.end()].strip():
            continue
        findings.append(_finding(path, text, m.start(), 'C ABI symbols require extern "C", not C++ mangling'))
    for m in re.finditer(r'\b(?:section|code_seg|data_seg|bss_seg|const_seg|allocate)\s*\(\s*"([^"\n]*)"', literal):
        if not masked[m.start() : m.end()].strip():
            continue
        for why in issues(m[1], "section"):
            findings.append(_finding(path, text, m.start(), why))
    declarations = [(b.name, b.start, b.kind in {"global", "static", "function", "macro"}) for b in bindings(text)]
    seen: set[tuple[int, str]] = set()
    for name, at, global_scope in declarations:
        for why in _issues(name, "symbol", global_scope, implementation=True):
            key = (text.count("\n", 0, at), why)
            if key not in seen:
                findings.append(_finding(path, text, at, why))
                seen.add(key)
    return tuple(findings)


_PerFile = tuple[tuple[Finding, ...], tuple[tuple[str, int, bool, bool], ...]]
_MEMO: dict[tuple[str, str], _PerFile] = {}  # (path, text) -> its census, for the life of the process
_DISK: dict[tuple[str, str], _PerFile] = {}  # (path, sha1 of text) -> its census, loaded from the census file


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogateescape")).hexdigest()


def _per_file(path: str, text: str) -> _PerFile:
    held = _MEMO.get((path, text))
    if held is None:
        found = _DISK.get((path, _digest(text))) if _DISK else None
        _MEMO[(path, text)] = held = found if found is not None else _census(path, text)
    return held


def _census(path: str, text: str) -> _PerFile:
    """What one file alone says, memoized on its text: its source and file-name findings, and its global declarations as
    (name, offset, static, local-to-a-typedef). `findings` runs over the whole tree several times in one rename (before
    and after, per pack); only the files a change edited are read again."""
    own = list(source(path, text))
    for why in issues(path, "file"):
        own.append(_finding(path, text, 0, why))
    code = _code(text)
    declared = []
    for declaration in bindings(text):
        name, at = declaration.name, declaration.start
        global_scope = declaration.kind in {"function", "global", "static"} and not declaration.owner
        start = max(code.rfind(char, 0, at) for char in ";{}") + 1
        prefix = code[start:at]
        local = re.search(r"\btypedef\b|^\s*(?:struct|union|enum)\s*$", prefix)
        if global_scope and not local:
            declared.append((name, at, bool(re.search(r"\bstatic\b", prefix)), False))
    return tuple(own), tuple(declared)


def _load(cache: Path) -> None:
    try:
        with cache.open("rb") as handle:
            loaded = pickle.load(handle)  # our own file: written by `_save` under the project's .unbake
        if isinstance(loaded, dict) and loaded.get("schema") == 2:
            _DISK.update(loaded["files"])
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        pass  # no census file yet, or one from older code: every file is read again


def _save(cache: Path, files: Mapping[str, str]) -> None:
    """Keep the census of exactly the files just read (a file gone from the tree drops out)."""
    keep = {(p, _digest(t)): _MEMO[(p, t)] for p, t in files.items() if (p, t) in _MEMO}
    if keep.keys() == _DISK.keys():
        return
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
        with tmp.open("wb") as handle:
            pickle.dump({"schema": 2, "files": keep}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache)
        _DISK.clear()
        _DISK.update(keep)
    except OSError:
        pass  # a read-only tree only loses the speed-up


def findings(
    files: Mapping[str, str], cache: Path | None = None, evidenced: Collection[str] = ()
) -> tuple[Finding, ...]:
    """Whole source set naming census: source rules plus path and external-symbol case collisions. `cache` names a
    file that keeps each source file's census between runs (by path and content hash): only the files an edit changed
    are read again (the whole tree was 9 s of a 10-row rename). `evidenced`: names the caller's project evidence
    declares original (see `issues`); the reserved-identifier rule does not report them."""
    if cache is not None and not _DISK:
        _load(cache)
    try:
        out = _findings(files)
        heads = tuple(f"{name}: reserved identifier (" for name in evidenced)
        return tuple(f for f in out if not f.reason.startswith(heads)) if heads else out
    finally:
        if cache is not None:
            _save(cache, files)


def _findings(files: Mapping[str, str]) -> tuple[Finding, ...]:
    """The census proper (see `findings`)."""
    out: list[Finding] = []
    paths: dict[str, str] = {}
    symbols: dict[str, tuple[str, str]] = {}
    for path, text in sorted(files.items()):
        if not (config.is_source(path) or config.is_header(path)) or config.is_assembly(path):
            continue
        own, declared = _per_file(path, text)
        out.extend(own)
        for parent in (*reversed(PurePosixPath(path).parents), PurePosixPath(path)):
            part = str(parent)
            if part == ".":
                continue
            key = part.casefold()
            if key in paths and paths[key] != part:
                for why in issues(part, "file", (paths[key],)):
                    out.append(_finding(path, text, 0, why))
            paths[key] = part
        file_symbols: dict[str, str] = {}
        for name, at, static, _ in declared:
            key = name.casefold()
            if key in file_symbols and file_symbols[key] != name:
                for why in issues(name, taken=(file_symbols[key],)):
                    out.append(_finding(path, text, at, why))
            file_symbols[key] = name
            if static:
                continue
            if key in symbols and symbols[key][0] != name:
                other, origin = symbols[key]
                for why in issues(name, taken=(other,)):
                    out.append(_finding(path, text, at, f"{why}; declared in {origin}"))
            symbols[key] = (name, path)
    return tuple(out)
