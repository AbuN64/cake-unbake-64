"""Catalogue header declarations and fold candidate declarations into their owners."""

import heapq
import json
import pickle
import posixpath
import re
from collections import ChainMap, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, replace
from functools import cache, lru_cache, partial
from itertools import pairwise
from typing import Any, ClassVar, NamedTuple, cast

from pycparser import CParser, c_ast, c_generator
from pycparser.c_parser import Coord, ParseError

from unbake import config as configuration
from unbake import declarations as _declarations
from unbake import effort, pool, store, symbols, types
from unbake import view as _view
from unbake.contracts import Config, Finding, Refusal, Snapshot, SourceView, UnitSpec, digest, sha256


def _blank(text: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", lambda m: re.sub(r"[^\n]", " ", m[0]), text, flags=re.S)


def _kind_of(suffix: str) -> Callable[[str], bool]:
    """The test a suffix names: ".c" any source (.c, .cpp, any case), ".h" any header, else that exact suffix."""
    if suffix == ".c":
        return configuration.is_source
    if suffix == ".h":
        return configuration.is_header
    return lambda path: path.endswith(suffix)


def sources(snapshot: Snapshot, place: str = "include", suffix: str = ".h") -> list[str]:
    root = snapshot.config.project.root
    kind = _kind_of(suffix)
    paths = (
        set()
        if snapshot.tree
        else {rel for p in (root / place).rglob("*") if p.is_file() and kind(rel := p.relative_to(root).as_posix())}
    )
    paths.update(p for p in snapshot.overlays if p.startswith(place + "/") and kind(p))
    return sorted(
        p
        for p in paths
        if (p not in snapshot.overlays or snapshot.overlays[p] is not None)
        and not any(
            p.startswith(folder.rstrip("/") + "/") for folder in getattr(snapshot.config.project, "library_dirs", ())
        )
    )


def _quoted(content: bytes) -> tuple[str, ...]:
    return tuple(name.decode(errors="replace") for delimiter, name in _view.directives(content) if delimiter == b'"')


def included_by(snapshot: Snapshot, changed: Collection[str] = ()) -> dict[str, set[str]]:
    """Per file the files that include it directly. A quote include names a file beside the includer, else under
    include/ or src/; every existing one is taken. Built once per snapshot."""

    def graph() -> dict[str, set[str]]:
        known = {*sources(snapshot), *snapshot.layout.units, *changed}
        found: dict[str, set[str]] = defaultdict(set)
        for path in known:
            content = snapshot.peek(path) or b""
            for name in effort.memo(("quoted", digest(content)), partial(_quoted, content)):
                for base in (posixpath.dirname(path), "include", "src"):
                    if (target := posixpath.normpath(posixpath.join(base, name))) in known:
                        found[target].add(path)
        return found

    return effort.memo(("included-by", snapshot.digest, tuple(sorted(changed))), graph)


def consumers(snapshot: Snapshot, changed: Collection[str]) -> tuple[str, ...]:
    """Every unit that includes a changed header directly or through others."""
    if not changed:
        return ()
    direct = included_by(snapshot, changed)
    reached: set[Any] = set()
    todo = list(changed)
    while todo:
        new = direct.get(todo.pop(), set()) - reached
        reached |= new
        todo.extend(new)
    return tuple(path for path in reached if path in snapshot.layout.units)


def dangling(before: Snapshot, after: Snapshot, changed: Collection[str]) -> tuple[str, ...]:
    """The headers of `after` that name something the changed headers declared before and nothing declares now. Such a
    header no longer parses, so every unit including it fails whether or not its code reaches the name."""
    gone: set[str] = set()
    for path in changed:
        if configuration.is_header(path):
            rows = [_declarations.dependencies((s.peek(path) or b"").decode()) for s in (before, after)]
            if rows[0] is None or rows[1] is None:
                return tuple(changed)  # a header that cannot be read is a header that cannot be trusted
            gone |= {n for names, _, _ in rows[0] for n in names} - {n for names, _, _ in rows[1] for n in names}
    if not gone:
        return ()
    paths = sources(after)
    texts = [(after.peek(p) or b"").decode() for p in paths]
    parsed = pool.map(after.config, "land.declarations", _declarations.dependencies, texts, digest)
    if any(rows is None for rows in parsed):
        return tuple(paths)
    gone -= {n for rows in parsed for names, _, _ in rows for n in names}
    return tuple(p for p, rows in zip(paths, parsed, strict=True) if any(words & gone for _, words, _ in rows))


def readers(before: Snapshot, after: Snapshot, changed: Collection[str]) -> set[str]:
    """The units a change of these files can change: the consumers of a header that names a declaration nothing
    declares now (`dangling`), and the consumers whose text reaches a changed, gained or lost declaration
    (`reaching`). A changed include is no reason to prove every consumer: what it brings in or takes out is."""
    found = consumers(after, changed)
    return {*consumers(after, dangling(before, after, changed)), *reaching(before, after, changed, found)}


def reaching(before: Snapshot, after: Snapshot, changed: Collection[str], units: Sequence[str]) -> tuple[str, ...]:
    """The units among `units` whose own text can reach a declaration the changed headers changed: it names a changed
    name, or a header name (macro, type, function) whose declaration uses one, transitively. A unit that names none
    reads the same reached declarations on both sides, so only these need preprocessing. Every unit when a header
    does not split. A header a changed one now reaches and did not before brings its every declaration in: a unit
    naming one of those reads a declaration it never read (a conflicting local one breaks it); one it reached before
    and no longer does takes its every declaration out."""
    gained = _closure(after, changed) - _closure(before, changed)
    lost = _closure(before, changed) - _closure(after, changed)  # a unit naming what it brought reads it no more
    old = {
        p: _declarations.dependencies((before.peek(p) or b"").decode()) for p in changed if configuration.is_header(p)
    }
    new = {
        p: _declarations.dependencies((after.peek(p) or b"").decode()) for p in changed if configuration.is_header(p)
    }
    if any(rows is None for rows in (*old.values(), *new.values())):
        return tuple(units)
    texts = {
        side: {(n, c) for rows in rows_of.values() for names, _, c in rows or () for n in names}
        for side, rows_of in (("old", old), ("new", new))
    }
    hit = {n for n, _ in texts["old"] ^ texts["new"]}
    for side, paths_of in ((after, gained), (before, lost)):
        for path in paths_of:
            rows = _declarations.dependencies((side.peek(path) or b"").decode())
            if rows is None:
                return tuple(units)
            hit.update(n for names, _, _ in rows for n in names)
    if not hit:
        return ()
    paths = sources(after)
    parsed = pool.map(
        after.config, "land.declarations", _declarations.dependencies, [after.read(p).decode() for p in paths], digest
    )
    if any(rows is None for rows in parsed):
        return tuple(units)
    users: dict[str, set[str]] = {}
    for rows in parsed:
        for names, words, _ in rows:
            for word in words - names:
                users.setdefault(word, set()).update(names)
    todo = list(hit)
    while todo:
        for name in users.get(todo.pop(), ()):
            if name not in hit:
                hit.add(name)
                todo.append(name)
    known = {p.removeprefix("include/") for p in paths}

    def reaches(unit: str) -> bool:
        text = (after.peek(unit) or b"").decode()
        outside = any(n not in known for n in re.findall(r'^\s*#\s*include\s+"([^"]+)"', text, re.M))
        return outside or bool(hit & set(re.findall(r"\b[A-Za-z_]\w*\b", text)))  # a file outside include/ is unread

    return tuple(u for u in units if reaches(u))


_DIRECTIVE = r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*"  # a preprocessor line with its backslash continuations


def _first_branches(blanked: str, clean: str) -> str:
    """The clean text with the brackets of every #elif and #else branch blanked: each branch of a conditional opens
    and closes the same brackets (`#if A if (a) { #else if (b) { #endif`), so only the first one is counted."""
    out: list[str] = []
    at, stack = 0, list[bool]()  # per open conditional: whether the branch now running is a later one
    for m in re.finditer(_DIRECTIVE, blanked, re.M):
        piece = clean[at : m.start()]
        out.append(re.sub(r"[{}()]", " ", piece) if any(stack) else piece)
        word = m[0].strip()
        if re.match(r"#\s*if", word):
            stack.append(False)
        elif re.match(r"#\s*(?:elif|else)\b", word) and stack:
            stack[-1] = True
        elif re.match(r"#\s*endif\b", word) and stack:
            stack.pop()
        out.append(clean[m.start() : m.end()])
        at = m.end()
    out.append(re.sub(r"[{}()]", " ", clean[at:]) if any(stack) else clean[at:])
    return "".join(out)


def statements(text: str, bodies: bool = False) -> Iterator[tuple[int, int, str]]:
    """(start, end, text) of each file-scope statement; with `bodies`, function definitions are statements too."""
    blanked = _blank(text)
    clean = re.sub(_DIRECTIVE, lambda m: re.sub(r"[^\n]", " ", m[0]), blanked, flags=re.M)
    clean = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', lambda m: re.sub(r"[^\n]", " ", m[0]), clean)
    counted = _first_branches(blanked, clean) if "#" in blanked else clean
    start, braces, parens = None, 0, 0
    for i, char in enumerate(counted):
        if start is None and not clean[i].isspace():
            start = i
        parens += (char == "(") - (char == ")")
        braces += (char == "{") - (char == "}")
        if char == "}" and braces == 0 and start is not None:
            prefix = clean[start : clean.find("{", start, i)]
            if prefix.rstrip().endswith(")") and "=" not in prefix:  # a definition, never a record
                if bodies:
                    yield start, i + 1, clean[start : i + 1]
                start = None
        if char == ";" and not braces and not parens and start is not None:
            yield start, i + 1, clean[start : i + 1]
            start = None


def _parts(text: str) -> Iterator[str]:
    start = depth = 0
    for i, char in enumerate(text):
        depth += (char in "([{") - (char in ")]}")
        if char == "," and not depth:
            yield text[start:i]
            start = i + 1
    yield text[start:]


def _declared(text: str) -> list[tuple[str, int, str]]:
    """(name, line, statement) for every declaration one header makes."""
    rows: list[Any] = []
    for start, end, declaration in statements(text):
        names = [f"{kind} {tag}" for kind, tag in re.findall(r"\b(struct|union|enum)\s+(\w+)\s*\{", declaration)]
        if re.search(r"\b(?:extern|typedef)\b", declaration):
            for part in _parts(declaration.rstrip(";")):
                match = _DECLARATOR.search(part)
                if match:
                    names.append(match[1] or match[2])
        elif "{" not in declaration:
            match = re.search(r"\b(\w+)\s*\([^;]*\)\s*;", declaration)
            if match:
                names.append(match[1])
        rows.extend((name, text.count("\n", 0, start) + 1, " ".join(_blank(text[start:end]).split())) for name in names)
    return rows


# a pointer declarator `(*name)`, else a plain name, its arrays and parameters: the parameters of a plain name never
# open with `(*`, so `void (*hook)(int)` names `hook`, not `void`
_DECLARATOR = re.compile(r"\(\s*\*\s*(\w+)\s*\)|\b(\w+)\s*(?:\[[^]]*\]\s*)*(?:\((?!\s*\*)[^{};]*\))?\s*$")

# the names a source gives a value or storage: a declaration line without extern, whatever it sits in
_DEFINITION = re.compile(
    r"^([ \t]*)(?!extern\b|typedef\b|return\b)((?:[\w]+[ \t*]+)+)(\w+)[ \t]*((?:\[[^\]]*\][ \t]*)*)[=;]", re.M
)
_EXTERN = re.compile(r"^[ \t]*extern\b[^;{}]*;", re.M)
# a prototype without extern, at file or block scope: type words, then the name and its parameters (never a call: a
# call statement has no type word before its name, and `return f(x);` is no declaration)
_PROTOTYPED = re.compile(
    r"^[ \t]*(?!(?:return|else|goto|case|do)\b)(?:[A-Za-z_]\w*[ \t*]+)+\w+[ \t]*"
    r"\([^;{}()]*(?:\([^;{}()]*\)[^;{}()]*)*\)[ \t]*;",
    re.M,
)
_SYNONYMS = {
    "f32": "float",
    "f64": "double",
    "s8": "signed char",
    "u8": "unsigned char",
    "s16": "short",
    "u16": "unsigned short",
    "s32": "int",
    "u32": "unsigned int",
    "s64": "long long",
    "u64": "unsigned long long",
    "const": "",
    "volatile": "",
    "extern": "",
}

_FUNCTION = re.compile(r"^[A-Za-z_][\w \t*]*?\b(\w+)\s*\(([^;{}]*)\)\s*\{", re.M)


def _file_job(item: tuple[bytes, bool]) -> dict[str, Any]:
    text, source = item
    code = text.decode(errors="replace")
    clean = _blank(code)
    found = list(_DEFINITION.finditer(clean)) if source else []
    defs = [(f[3], clean.count("\n", 0, f.start()) + 1, f"{f[2]} @{f[4]}") for f in found if not f[1]]  # file scope
    declared = (
        [
            (name, clean.count("\n", 0, f.start()) + 1, decl)
            for f in _EXTERN.finditer(clean)
            for name, _, decl in _declared(f[0])
        ]
        if source
        else _declared(code)
    )
    functions = (
        [
            (f[1], clean.count("\n", 0, f.start()) + 1, " ".join(f[0].split()).rstrip("{ "))
            for f in _FUNCTION.finditer(clean)
        ]
        if source
        else []
    )
    prototypes = (
        [
            (name, clean.count("\n", 0, f.start()) + 1, decl)
            for f in _PROTOTYPED.finditer(clean)
            for name, _, decl in _declared(f[0])
        ]
        if source
        else []
    )
    return {
        "declared": declared,
        "prototypes": prototypes,
        "defined": [f[3] for f in found],
        "defs": defs,
        "functions": functions,
        "words": frozenset(re.findall(r"\w+", code)),
    }


def defined(source: bytes) -> list[str]:
    """The names of the functions a source defines."""
    return [name for name, _, _ in _file_job((source, True))["functions"]]


def signatures(source: bytes) -> list[tuple[str, str]]:
    """The name and head (`static int f(int x)`) of each function a source defines, in order."""
    return [(name, head) for name, _, head in _file_job((source, True))["functions"]]


def variables(source: bytes) -> list[str]:
    """The names a source gives storage at file scope."""
    return [name for name, _, _ in _file_job((source, True))["defs"]]


def declares(snapshot: Snapshot) -> dict[str, list[str]]:
    """Per header the names it declares."""
    paths = sources(snapshot)
    return {p: [n for n, _, _ in s["declared"]] for p, s in zip(paths, _scanned(snapshot, paths), strict=True)}


def duplicates(path: str, text: str) -> list[Finding]:
    """Repeated complete file-scope record definitions, independent of the registered consumer graph."""
    seen: dict[str, Any] = {}
    findings = []
    for name, line, declaration in _declared(text):
        if "{" not in declaration or not re.match(r"(?:typedef\s+)?(?:struct|union|enum)\b", declaration):
            continue
        if name in seen:
            findings.append(
                Finding(
                    "types.conflict",
                    f"{name} has duplicate complete definitions at lines {seen[name]} and {line}",
                    path=path,
                    line=line,
                    unit=name,
                )
            )
        seen[name] = line
    return findings


_SCANS: dict[tuple[str, bool], dict[str, Any]] = {}  # by (sha256 of the text, source or not): a scan is a pure
# function of the text, so every command may share it


pool.carry("headers.scans", _SCANS)  # warm workers get the parent's new scans with each spilled tree


def _scanned(snapshot: Snapshot, paths: list[str], pinned: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """The scan of each path. A text scanned already is not scanned again, so a loop that asks for every header once
    per unit scans each distinct text once. `pinned` gives paths' pins (the sha256 of their text): such a path whose
    text was scanned already is not read again."""
    if len(_SCANS) > 20000:
        _SCANS.clear()
    keys: list[tuple[str, bool]] = []
    fresh: dict[tuple[str, bool], tuple[bytes, bool]] = {}
    for path in paths:
        source, text = configuration.is_source(path), None
        if pinned is None or path not in pinned:
            text = snapshot.read(path)
        key = (pinned[path] if text is None and pinned is not None else sha256(text or b""), source)
        if key not in _SCANS and key not in fresh:
            fresh[key] = (snapshot.read(path) if text is None else text, source)
        keys.append(key)
    done = pool.map(snapshot.config, "headers.catalog", _file_job, list(fresh.values()), digest) if fresh else []
    _SCANS.update(zip(fresh, done, strict=True))
    return [_SCANS[k] for k in keys]


def declared_spellings(snapshot: Snapshot, paths: Sequence[str]) -> dict[str, dict[str, set[str]]]:
    """Per file and name the type spellings its file-scope declarations give the name, normalized as `reconcile`
    compares them; every file scanned in one dispatch."""
    found: dict[str, dict[str, set[str]]] = {p: defaultdict(set) for p in paths}
    for path, scanned in zip(paths, _scanned(snapshot, list(paths)) if paths else (), strict=True):
        for name, _, text in scanned["declared"]:
            if name not in _TYPE_WORDS and not text.startswith("typedef") and "{" not in text:
                found[path][name].add(_unnamed(_normal(_type_of(text, name))))
    return found


_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
_DEFINE = re.compile(r"^\s*#\s*define\s+(\w+)", re.M)


def include_breaks(snapshot: Snapshot, path: str, added: Collection[str]) -> list[str]:
    """The units an include of each of `added` in the header at path would stop compiling, each with the names that
    break it: a unit including the header then reaches every header the added ones reach, and where one of those
    defines a macro the unit names that nothing it reached before defines, or declares a name the unit declares
    itself with another type, the unit fails (an include drags one version's names into every version's consumers)."""
    if not added:
        return []
    texts: dict[str, str | None] = {}

    def text(name: str) -> str | None:
        if name not in texts:
            data = snapshot.peek(name)
            texts[name] = None if data is None else data.decode(errors="replace")
        return texts[name]

    def closure(starts: Collection[str]) -> set[str]:
        seen: set[str] = set()
        pending = list(starts)
        while pending:
            at = pending.pop()
            code = None if at in seen else text(at)
            if code is not None:
                seen.add(at)
                pending += [
                    posixpath.normpath(f"{b}/{n}")
                    for n in _QUOTED_INCLUDE.findall(code)
                    for b in ("include", posixpath.dirname(at))
                ]
        return seen

    defines = cache(lambda name: frozenset(_DEFINE.findall(text(name) or "")))
    reached = closure(
        [posixpath.normpath(f"{b}/{n}") for n in sorted(added) for b in ("include", posixpath.dirname(path))]
    )
    units = sorted(consumers(snapshot, [path]))
    befores = {u: closure([u]) for u in units}
    units = [u for u in units if reached - befores[u]]
    spelled = declared_spellings(snapshot, [*sorted(set().union(*(reached - befores[u] for u in units))), *units])
    broken: list[str] = []
    for unit in units:
        new = reached - befores[unit]
        words = set(re.findall(r"\w+", _blank(text(unit) or "")))
        named = sorted((words & set().union(*map(defines, new))) - set().union(*map(defines, befores[unit])))
        named += [
            name
            for name, spelling in sorted(spelled[unit].items())
            if any(name in spelled[p] and spelled[p][name] != spelling for p in new)
        ]
        broken += [f"{unit} ({', '.join(named[:3])})"] if named else []
    return broken


def definers(snapshot: Snapshot) -> dict[str, set[str]]:
    """Per name the landed sources that define it at file scope, a function or storage, static or not."""
    found: dict[str, set[str]] = defaultdict(set)
    paths = [p for p in sources(snapshot, "src", ".c") if not p.startswith("src/fuzzy/")]
    for path, scanned in zip(paths, _scanned(snapshot, paths), strict=True):
        for name, _, _ in (*scanned["functions"], *scanned["defs"]):
            found[name].add(path)
    return dict(found)


def landed(snapshot: Snapshot) -> dict[str, str]:
    """The head of every function a landed source defines (a fuzzy candidate is not landed). Kept per pin of the
    sources, so the many folds of one wave read it once."""
    key = landed_sources(snapshot)[1]

    def produce() -> dict[str, str]:
        return {
            name: text
            for path, _, scanned in _listed(snapshot)
            if not path.startswith("src/fuzzy/")
            for name, _, text in scanned["functions"]
        }

    return effort.memo(("landed", str(snapshot.config.project.root), key), produce)


def externs(snapshot: Snapshot) -> dict[str, list[tuple[str, int, str]]]:
    """Per name the (path, line, type) of every extern statement a landed source makes of it."""
    paths = sources(snapshot, "src", ".c")

    def produce() -> dict[str, list[tuple[str, int, str]]]:
        found: defaultdict[str, list[tuple[str, int, str]]] = defaultdict(list)
        for path, scanned in zip(paths, _scanned(snapshot, paths), strict=True):
            for name, line, text in scanned["declared"]:
                found[name].append((path, line, _type_of(text, name).replace("@", "").strip()))
        return dict(found)

    key = digest([(p, snapshot.read(p)) for p in paths])
    return store.cached_object(snapshot.config, "externs", key, produce)


def catalog(snapshot: Snapshot) -> dict[str, tuple[str, int, str]]:
    """Per name the first header declaration. A pure function of the text of every header `sources` lists, so it is
    kept per command by the pin of each of those files: an edited, added or removed header makes a new key."""
    key = landed_sources(snapshot, "include", ".h")[1]

    def produce() -> dict[str, tuple[str, int, str]]:
        with effort.stage("headers.catalog"):
            parent = (effort.peek(("parent", getattr(snapshot, "digest", ""))) or (None,))[0]
            if parent is not None and (derived := _catalog_over(parent, snapshot)) is not None:
                return derived
            result: dict[str, Any] = {}
            for path, _, scanned in _listed(snapshot, "include", ".h"):
                for name, line, text in scanned["declared"]:
                    result.setdefault(name, (path, line, text))
            return result

    return effort.memo(("catalog", str(snapshot.config.project.root), key), produce)


def _catalog_over(parent: Snapshot, snapshot: Snapshot) -> dict[str, tuple[str, int, str]] | None:
    """An overlay's catalog as its parent's with only the headers it changes read again, so a fold's overlay costs its
    own writes, not every header. None when a header it changes no longer declares a name it was first to declare:
    the next declarer is then found by reading every header."""
    before, after = _each(parent, "include", ".h"), _each(snapshot, "include", ".h")
    changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    old = [p for p in changed if p in before]
    new = [p for p in changed if p in after]
    olds = dict(zip(old, _scanned(parent, old, before), strict=True))
    news = dict(zip(new, _scanned(snapshot, new, after), strict=True))
    result = dict(catalog(parent))
    for path in changed:
        first: dict[str, tuple[str, int, str]] = {}
        for name, line, text in news[path]["declared"] if path in news else ():
            first.setdefault(name, (path, line, text))
        for name, _, _ in olds[path]["declared"] if path in olds else ():
            if name not in first and result.get(name, ("",))[0] == path:
                return None
        for name, entry in first.items():
            current = result.get(name)
            if current is None or current[0] >= path:
                result[name] = entry
    return result


def _type_of(statement: str, name: str) -> str:
    """The type one declaration gives a name: its specifiers, the stars before it, and what follows it."""
    base: str | None = None
    for part in _parts(statement.rstrip(";")):
        match = _DECLARATOR.search(part)
        if match and base is None:
            base = part[: match.start()].rstrip("* ").strip()
        if match and name in (match[1], match[2]):
            tail = re.search(r"[*\s]*$", part[: match.start()])
            stars = tail[0] if tail else ""
            declarator = match[0].replace(name, "@", 1) + (part[match.end() :] if match[1] else "")  # (*@)(params)
            return " ".join(f"{(base or '').removeprefix('extern')} {stars} {declarator}".split())
    return ""


_TYPE_WORDS = {
    "const",
    "volatile",
    "unsigned",
    "signed",
    "struct",
    "union",
    "enum",
    "int",
    "char",
    "short",
    "long",
    "void",
    "float",
    "double",
}


def _shape(spelled: str) -> str:
    return re.sub(r"\[[^\]]*\]", "[]", _normal(spelled)).replace("static ", "").strip()


def _unnamed(spelled: str) -> str:
    """A function type without its parameter names: `@(s32 a, s32 *b)` and `@(s32, s32*)` are one type, and so are
    the function pointers `(*@)(s32 a)` and `(*@)(s32)`."""
    opener = "@(" if "@(" in spelled else "@)(" if "@)(" in spelled else ""
    if not opener:
        return spelled
    head, _, rest = spelled.partition(opener)
    kept = []
    for param in (p.strip() for p in rest.rpartition(")")[0].split(",")):
        named = re.search(r"(\w+)\s*$", param)
        kept.append(param[: named.start()] if named and named.start() and named[1] not in _TYPE_WORDS else param)
    return re.sub(r"\s+|\bvoid\b(?=\))|\b(?:struct|union|enum)\b", "", head + opener + ",".join(kept) + ")")


def _normal(spelled: str, qualified: bool = False) -> str:
    """The spelling with synonyms folded; const and volatile go too unless they matter, as they do to code."""
    kept = {k: v for k, v in _SYNONYMS.items() if not qualified or k not in ("const", "volatile")}
    return " ".join(re.sub(r"\b\w+\b", lambda w: kept.get(w[0], w[0]), spelled).split())


def disagreements(
    snapshot: Snapshot, defined: Collection[str] = (), paths: Collection[str] | None = None
) -> tuple[dict[str, str], list[Finding]]:
    """The one type each declared name has where every landed declaration agrees (and no source defines it), and a
    finding per name whose declarations disagree (a landed definition is canonical; its symptoms hold {path, line,
    signature} and the count of other landed files using the name)."""
    with effort.stage("headers.disagreements"):
        paths = list(paths) if paths is not None else [*sources(snapshot), *sources(snapshot, "src", ".c")]
        result: tuple[dict[str, str], list[Finding]] = pool.map(
            snapshot.config,
            "headers.disagreements",
            _disagreements,
            [(snapshot, paths, tuple(sorted(defined)))],
            _disagreements_key,
        )[0]
        return result


def _disagreements_key(item: tuple[Any, ...]) -> str:
    snapshot, paths, defined = item
    return digest(([(p, snapshot.read(p)) for p in paths], defined))


def _disagreements(item: tuple[Any, ...]) -> tuple[dict[str, str], list[Finding]]:
    snapshot, paths, defined = item
    return _disagreement_rows((paths, _scanned(snapshot, paths), defined))


def _disagreement_rows(item: tuple[Any, ...]) -> tuple[dict[str, str], list[Finding]]:
    paths, scans, defined = item
    found: dict[str, list[tuple[str, int, str]]] = {}
    owned, definitions, variables = set(defined), {}, defaultdict(list)
    words = [scanned["words"] for scanned in scans]
    for path, scanned in zip(paths, scans, strict=True):
        owned.update(scanned["defined"])
        if not path.startswith("src/fuzzy/"):  # a fuzzy candidate is not landed
            definitions.update({name: (path, line, text) for name, line, text in scanned["functions"]})
            for name, line, text in scanned["defs"]:
                variables[name].append((path, line, text))
        for name, line, text in scanned["declared"]:
            if not text.startswith("typedef") and not re.match(r"(?:struct|union|enum)\b[^;(]*\{", text):
                found.setdefault(name, []).append((path, line, text))
    agreed, findings = {}, []
    for name, rows in sorted(variables.items()):  # a definition spells one type, as every landed declaration does
        spelled = {_shape(t) for _, _, t in rows} | {_shape(_type_of(t, name)) for _, _, t in found.get(name, ())}
        if len(spelled) > 1:
            shown = (
                *(f"{name}: {p}:{n} {' '.join(t.replace('@', name).split())}" for p, n, t in rows),
                *(f"{name}: {p}:{n} {t}" for p, n, t in found.get(name, ())),
            )
            findings.append(
                Finding(
                    "types.conflict",
                    f"{name} is defined or declared {len(spelled)} ways",
                    path=rows[0][0],
                    line=rows[0][1],
                    unit=name,
                    missing=shown,
                    blocking=False,
                    action="declare and define it with one type",
                )
            )
    for name, rows in sorted(found.items()):
        if name in definitions:
            path, line, text = definitions[name]
            canonical = _unnamed(_normal(_type_of(text, name)))
            if canonical and any(_unnamed(_normal(_type_of(t, name))) != canonical for _, _, t in rows):
                shown = tuple(f"{name}: {p}:{n} {t}" for p, n, t in [definitions[name], *rows])
                excluded = {p for p, _, _ in rows} | {path}
                findings.append(
                    Finding(
                        "types.conflict",
                        f"{name} is defined as {text} but declared another way",
                        path=path,
                        line=line,
                        unit=name,
                        missing=shown,
                        blocking=False,
                        action=f"declare it as its definition in {path} does",
                        symptoms={
                            "definition": {"path": path, "line": line, "signature": text},
                            "consumers": sum(
                                name in w and p not in excluded for p, w in zip(paths, words, strict=True)
                            ),
                        },
                    )
                )
            continue
        if name in owned:
            continue
        spellings = [_type_of(text, name) for _, _, text in rows]
        types = {_normal(s) for s in spellings}
        if len(types) == 1 and "" not in spellings:
            agreed[name] = spellings[0]
        else:
            shown = tuple(f"{name}: {path}:{line} {text}" for path, line, text in rows)
            findings.append(
                Finding(
                    "types.conflict",
                    f"{name} is declared {len(types)} ways and defined nowhere",
                    unit=name,
                    missing=shown,
                    blocking=False,
                    action="declare it once, with one type, in a shared header",
                )
            )
    return agreed, findings


def normal(text: str) -> str:
    while match := re.search(r"\b__attribute__\s*\(", text):
        end, depth = match.end(), 1
        while end < len(text) and depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        text = text[: match.start()] + re.sub(r"[^\n]", " ", text[match.start() : end]) + text[end:]
    text = re.sub(r'^(# \d+ "[^"\n]*")(?: \d+)+[ \t]*$', r"\1", text, flags=re.M)  # the flags a marker may end with
    text = re.sub(r"\b(?:__extension__|__restrict)\b", "", text)
    return re.sub(r"\b(?:__inline__|__inline)\b", "inline", text)


_KEYWORDS = frozenset(
    (
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
        "inline",
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
        "__attribute__",
        "__extension__",
        "__restrict",
    )
)


_USE = re.compile(r"^\s*(?:(?:const|volatile|static|extern|register)\s+)*([A-Za-z_]\w*)\b\s*[*\s]\s*[A-Za-z_*]")


_TYPEDEF_NAME = re.compile(r"\(\s*\*+\s*(\w+)\s*\)\s*\(|(\w+)\s*(?:\[[^\]]*\]\s*)*;")
_PARAM_USE = re.compile(r"[(,]\s*(?:(?:const|volatile)\s+)*([A-Za-z_]\w*)\s*(?:\*+\s*\w*|\s+\w+)\s*(?=[,)\[])")
_FIELD_USE = re.compile(r"[{;]\s*(?:(?:const|volatile)\s+)*([A-Za-z_]\w*)\s*(?:\*+\s*|\s+)[A-Za-z_]")
_PROTOTYPE = re.compile(r"^\s*(?:extern\s+)?[A-Za-z_][\w\s*]*?\b\w+\s*\(")


def _culprit(
    text: str, start: int = 0, known: Collection[str] = (), coord: tuple[str, int, int] | None = None, raw: str = ""
) -> tuple[int, str, str, list[str]]:
    """The first use of a type name nothing before it declares, searched from offset `start` (the run the parser
    refused) with `known` the type names the runs before it declared: its 1-based view line, the file:line its
    markers give it, the name, and the chain of headers that included that file (outermost first). With `coord`
    (file, line, column of the parser's error) the name right before that column is taken first; the parser names no
    line for a bad specifier list, so the run is searched for a declaration, field or prototype parameter that names
    an undeclared type. `raw` is the text before normal() took the markers' flags (line for line the same): the
    chain is read from them. Nothing found is (0, "", "", [])."""
    seen = {"__builtin_va_list", "_Bool", *known}
    here, there, depth = "", 0, 0
    chain: list[str] = []
    lines = text.splitlines(keepends=True)
    flagged = raw.splitlines() if raw else []
    flagged = flagged if len(flagged) == len(lines) else []
    offset = 0

    def found(index: int, word: str) -> tuple[int, str, str, list[str]]:
        return index, f"{here}:{there}", word, [c for c in chain[:-1]]

    late = None  # the first searched use: reported only when the parser's column names nothing
    for index, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        marker = re.match(r'# (\d+) "([^"\n]*)"(.*)', line)
        if marker:
            here, there = marker[2], int(marker[1]) - 1
            flags = (flagged[index - 1] if flagged else line).rpartition('"')[2].split()
            if "1" in flags or not chain:
                chain.append(here)
            elif "2" in flags:
                while len(chain) > 1 and chain[-1] != here:
                    chain.pop()
            else:
                chain[-1] = here
            offset += len(raw)
            continue
        there += 1
        offset += len(raw)
        if offset <= start:
            continue
        if coord and (here, there) == coord[:2]:
            before = re.search(r"([A-Za-z_]\w*)\s*$", line[: max(coord[2] - 1, 0)])
            if before and before[1] not in _KEYWORDS and before[1] not in seen:
                return found(index, before[1])
        if late is None:
            words = []
            if (match := _USE.match(line)) and not line.lstrip().startswith("typedef"):
                words.append(match[1])
            if "{" in line:  # fields of a record spelled on one line
                words += [m[1] for m in _FIELD_USE.finditer(line[line.index("{") :])]
            if depth == 0 and _PROTOTYPE.match(line):
                words += [m[1] for m in _PARAM_USE.finditer(line[line.index("(") :])]
            word = next((w for w in words if w not in _KEYWORDS and w not in seen), None)
            if word is not None:
                late = found(index, word)
                if coord is None:
                    return late
        if line.lstrip().startswith(("typedef", "}")) or (depth == 0 and "typedef" in line):
            seen.update(a or b for a, b in _TYPEDEF_NAME.findall(line))
        depth = max(depth + line.count("{") - line.count("}"), 0)
    return late or (0, "", "", [])


def parse_refusal(error: Exception, path: str, text: str = "") -> Refusal:
    match = re.search(r"^(.*?):(\d+)(?::(\d+))?:", str(error))
    number = int(match[2]) if match else 0
    reason = str(error) or "a closing brace matches no opening one"
    if text:
        start, known = getattr(error, "unbake_run", (0, ()))
        coord = (match[1], int(match[2]), int(match[3])) if match and match[3] else None
        view, source, word, chain = _culprit(normal(text), start, known, coord, text)
        if view:
            through = f", included through {' > '.join(chain)}" if chain else ""
            reason += f" (view line {view}, from {source}: type {word} is not declared before its use{through})"
            number = number or int(source.rpartition(":")[2])
    return Refusal(Finding("headers.parse", reason=reason, path=path, line=number))


class _Seeded(CParser):
    """A parser that starts its file scope from `seed` (name -> is a type) instead of an empty one, so one piece of a
    translation unit parses as it would after the pieces before it."""

    seed: Mapping[str, bool] = {}

    @property
    def _scope_stack(self) -> list[dict[str, bool]]:
        return self.__dict__["_stack"]  # type: ignore[no-any-return]

    @_scope_stack.setter
    def _scope_stack(self, stack: list[dict[str, bool]]) -> None:
        self.__dict__["_stack"] = [{**self.seed, **stack[0]}, *stack[1:]]


_LINE_MARKER = re.compile(r'^[ \t]*#[ \t]*(?:line[ \t]+)?\d+[ \t]+"([^"\n]*)"[^\n]*$', re.M)
_INDEX_FORMAT = 3  # raised when a stored index entry changes shape or its key changes meaning
# a view made over an overlay names the files it overlays inside its own build/views/<key> directory: its runs key
# (and its coordinates map back) as the project files they are, so every view of one header text shares one entry
_VIEW_DIR = r"(?:[^\"\n]*?/)?build/views/[0-9a-fA-F]{16}/"
_VIEW_MARK = re.compile(r'(?m)^([ \t]*#[ \t]*(?:line[ \t]+)?\d+[ \t]+")' + _VIEW_DIR)


class _Region(NamedTuple):
    """The declaration index entry of one run of preprocessed text between line markers (a header, or the part of one
    between its own includes): its file-scope declarations parsed, the type-or-not state it leaves names in, and per
    name its first declaration, its typedef and its record, so a fold looks names up instead of walking the run."""

    nodes: list[c_ast.Node]
    changed: dict[str, bool]
    first: dict[str, c_ast.Node]
    typedefs: dict[str, c_ast.Node]
    records: dict[tuple[str, str | None], c_ast.Node]
    file: str  # the file its declarations come from, as their coordinates spell it ("" when it declares nothing)
    words: frozenset[str]  # every name its text spells: what its parse asked of the names before it


class _Index:
    parses = 0  # runs parsed, for the work-count tests
    chars = 0  # characters of text parsed (runs and shape prefixes), for the work-count tests


def _entry(nodes: list[c_ast.Node], changed: dict[str, bool], words: frozenset[str]) -> _Region:
    first: dict[str, c_ast.Node] = {}
    for node in nodes:
        if (name := _name(node)) is not None:
            first.setdefault(name, node)
    typedefs = {n.name: n.type for n in nodes if isinstance(n, c_ast.Typedef)}
    file = next((n.coord.file for n in nodes if n.coord is not None and n.coord.file), "")
    return _Region(nodes, changed, first, typedefs, _records(c_ast.FileAST(nodes)), file, words)


def _words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Za-z_]\w*", text))


def _starts(text: str) -> list[int]:
    """Where the unit's runs begin: at a line marker that names another file than the run before it. A marker of the
    same file (cpp's jump over blank lines, which sits inside a function body as often as between statements) stays
    inside its run, so a run always holds whole statements and parses alone."""
    starts, file = [0], None
    for marker in _LINE_MARKER.finditer(text):
        if marker.group(1) != file and marker.start():
            starts.append(marker.start())
        file = marker.group(1)
    return starts


def _regions(text: str, path: str, config: Config | None = None) -> list[_Region]:
    """The unit as its index entries, one per run between line markers. A run parses alike in every unit that holds
    its text and gives the same names it spells a type on entry (a name declared plain reads as one never declared),
    so each header text is parsed once per process (and, with a config, once per host: the entry is kept in the
    store), never once per unit or per wave. A run that ends inside a statement (an include within a body) is parsed
    joined with the runs after it; any other parse error is the unit's own, raised from the run that holds it."""
    parser, out = _Seeded(), []
    scope: dict[str, bool] = {}
    made: list[tuple[str, bytes]] = []
    begin = 0
    try:
        for end in [*_starts(text)[1:], len(text)]:
            piece = text[begin:end]
            hashed = digest(_VIEW_MARK.sub(r"\1", piece) if "build/views/" in piece else piece)
            names = effort.memo(("words", hashed), partial(_words, piece))
            key = (hashed, path if begin == 0 else "", tuple(sorted(n for n in names if scope.get(n))))
            stored = digest((_INDEX_FORMAT, key))

            def produce(piece: str = piece, names: frozenset[str] = names, stored: str = stored) -> _Region:
                hit, value = store.loads(store.get(config, "declarations", stored)) if config else (False, None)
                if hit and isinstance(value, _Region):
                    return value
                _Index.parses += 1
                _Index.chars += len(piece)
                parser.seed = {n: True for n in names if scope.get(n)}
                ast = parser.parse(piece, filename=path)
                top = parser._scope_stack[0]
                value = _entry(ast.ext, {n: v for n, v in top.items() if parser.seed.get(n, False) != v}, names)
                if config:
                    made.append((stored, pickle.dumps(value, protocol=5)))
                return value

            try:
                region = effort.memo(("declarations", stored), produce)
            except ParseError as error:
                if end < len(text) and str(error).endswith("At end of input"):
                    continue  # the next run holds the rest of its statement
                error.unbake_run = (begin, frozenset(n for n, v in scope.items() if v))  # type: ignore[attr-defined]
                raise
            out.append(region)
            scope.update(region.changed)
            begin = end
    finally:
        if made and config:
            store.put_many(config, "declarations", made)
    return out


_LITERAL = re.compile(r"\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'")


def _valued(text: str) -> str:
    """The text with each GNU statement expression `({ ... })` read as the value 0, its lines kept: the parser knows no
    statement expressions (a gcc va_arg expands to one) and a fold reads declarations, never what a body computes."""
    out, at = [], 0
    while (start := text.find("({", at)) >= 0:
        depth, index = 0, start
        while index < len(text):
            literal = _LITERAL.match(text, index)
            if literal:
                index = literal.end()
                continue
            char = text[index]
            depth += char in "({"
            depth -= char in ")}"
            index += 1
            if depth == 0:
                break
        if depth:
            break  # unbalanced: the parser reports it
        out += [text[at:start], "0", "\n" * text.count("\n", start, index)]
        at = index
    return "".join(out) + text[at:]


def _unit_regions(text: str, path: str, config: Config | None = None) -> list[_Region]:
    """The index entries of a preprocessed translation unit; one that does not parse refuses with its error."""
    try:
        normalized = normal(text)
        return _regions(_valued(normalized) if "({" in normalized else normalized, path, config)
    except (ParseError, AssertionError) as error:  # pycparser asserts on a brace that closes nothing
        raise parse_refusal(error, path, text) from error


def parse(text: str, path: str, config: Config | None = None) -> c_ast.FileAST:
    """Parse a preprocessed translation unit using the catalogue's C dialect and diagnostics."""
    return c_ast.FileAST([n for region in _unit_regions(text, path, config) for n in region.nodes])


class _Read:
    """What the running fold consulted: every name it looked up, found or not, and the files it read whole."""

    names: ClassVar[set[str]] = set()
    paths: ClassVar[set[str]] = set()
    fold: ClassVar[bool] = False  # whether a fold recorded them since the last `consulted`


def consulted() -> tuple[frozenset[str], frozenset[str]] | None:
    """(names, paths) the last fold in this process consulted, taken once: it folds alike while none of them changes.
    None when no fold recorded what it read since the last call (the caller then treats every read as consulted)."""
    if not _Read.fold:
        return None
    _Read.fold = False
    return frozenset(_Read.names), frozenset(_Read.paths)


class _Asked(Mapping[Any, Any]):
    """A mapping whose lookups the running fold records as consulted."""

    def __init__(self, items: Mapping[str, Any]) -> None:
        self.items_ = items

    def __getitem__(self, key: Any) -> Any:
        _Read.names.add(str(key))
        return self.items_[key]

    def __contains__(self, key: object) -> bool:
        _Read.names.add(str(key))
        return key in self.items_

    def __iter__(self) -> Iterator[Any]:
        return iter(self.items_)

    def __len__(self) -> int:
        return len(self.items_)


def changed(old: bytes | None, new: bytes | None, path: str) -> frozenset[str] | None:
    """The names whose declaration, definition, storage or macro differs between two texts of one file; None when
    its includes differ, so what it makes visible changed and every reader of it is affected."""
    if _quoted(old or b"") != _quoted(new or b""):
        return None
    source = configuration.is_source(path)

    def rows(text: bytes | None) -> set[tuple[str, str]]:
        scan = _file_job((text or b"", source))
        found = {(n, t) for kind in ("declared", "defs", "functions") for n, _, t in scan[kind]}
        return found | {(m, "#define") for m in re.findall(r"^[ \t]*#\s*define\s+(\w+)", (text or b"").decode(), re.M)}

    return frozenset(n for n, _ in rows(old) ^ rows(new))


class _Lookup(Mapping[Any, c_ast.Node]):
    """One field of a unit's index entries read as one mapping, the way a dict built over the whole unit would read
    it: `last` takes the latest entry's value (a typedef or record defined again), else the first. It looks up only
    the names asked for: a fold never walks the declarations of headers it does not name."""

    def __init__(self, regions: Sequence[_Region], field: str, last: bool) -> None:
        self.maps: list[dict[Any, c_ast.Node]] = [m for r in regions if (m := getattr(r, field))]
        self.last = last
        if last:
            self.maps.reverse()
        self.found: dict[Any, c_ast.Node | None] = {}

    def get(self, key: Any, default: Any = None) -> Any:
        _Read.names.add(key if isinstance(key, str) else f"{key[0].lower()} {key[1]}")
        if key not in self.found:
            self.found[key] = next((m[key] for m in self.maps if key in m), None)
        value = self.found[key]
        return default if value is None else value

    def __getitem__(self, key: Any) -> c_ast.Node:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value  # type: ignore[no-any-return]

    def __contains__(self, key: object) -> bool:
        return self.get(key) is not None

    def __iter__(self) -> Iterator[Any]:
        return iter(dict.fromkeys(k for m in self.maps for k in m))

    def __len__(self) -> int:
        return len(set().union(*self.maps))

    def order(self, keys: Collection[Any]) -> list[Any]:
        """The keys in the order of their first definition in the unit, as a dict built over it orders them."""
        forward = list(reversed(self.maps)) if self.last else self.maps
        rank: dict[Any, tuple[int, int]] = {}
        for at, m in enumerate(forward):
            for key in keys:
                if key not in rank and key in m:
                    rank[key] = (at, list(m).index(key))
        return sorted(keys, key=lambda k: rank.get(k, (0, 0)))


parts = _parts  # top-level comma-separated parts


def _coord(node: c_ast.Node) -> Coord:
    assert node.coord is not None  # every parsed declaration has a coordinate
    return node.coord


def _path(snapshot: Snapshot, coord: Coord | None) -> str:
    path = coord.file.replace("\\", "/") if coord and coord.file else ""
    path = path.removeprefix(snapshot.config.project.root.as_posix().rstrip("/") + "/")
    return re.sub("^" + _VIEW_DIR, "", path).removeprefix("./")


_TAGGED = (c_ast.Struct, c_ast.Union, c_ast.Enum)


def _name(node: c_ast.Node) -> str | None:
    """The name a node declares; a struct, union or enum tag is spelled with its keyword, as C keeps tags apart."""
    if isinstance(node, c_ast.Decl) and not node.name and isinstance(node.type, _TAGGED):
        return f"{type(node.type).__name__.lower()} {node.type.name}" if node.type.name else None
    if isinstance(node, (c_ast.Decl, c_ast.Typedef)):
        return node.name or getattr(node.type, "name", None)
    return None


def _canonical(node: c_ast.Node, typedefs: Mapping[str, c_ast.Node]) -> str:
    def expand(item: c_ast.Node, seen: frozenset[str] = frozenset()) -> c_ast.Node:
        if isinstance(item, c_ast.TypeDecl) and isinstance(item.type, c_ast.IdentifierType):
            names = item.type.names
            if len(names) == 1 and names[0] in typedefs and names[0] not in seen:
                replacement = expand(deepcopy(typedefs[names[0]]), seen | {names[0]})
                base = replacement
                while hasattr(base, "type") and not isinstance(base, c_ast.TypeDecl):
                    base = base.type
                if hasattr(base, "quals"):
                    base.quals = sorted(set(base.quals + item.quals))
                return replacement
        if isinstance(item, c_ast.TypeDecl):
            item.declname = None
        for field in item.__slots__:
            if field in ("coord", "__weakref__"):
                continue
            value = getattr(item, field)
            if isinstance(value, c_ast.Node):
                setattr(item, field, expand(value, seen))
            elif isinstance(value, list):
                setattr(item, field, [expand(v, seen) if isinstance(v, c_ast.Node) else v for v in value])
        return item

    return " ".join(c_generator.CGenerator().visit(expand(deepcopy(node.type))).split())


def canonical(node: c_ast.Node, typedefs: Mapping[str, c_ast.Node]) -> str:
    """Canonical declaration type with typedefs expanded by the catalogue."""
    return _canonical(node, typedefs)


def _unknown_params(spelled: str) -> bool:
    """Whether a canonical function type spells no parameter list at all (K&R `()`)."""
    return bool(re.search(r"\(\s*\)\s*$", spelled))


def _is_prototype(spelled: str) -> bool:
    return spelled.rstrip().endswith(")") and not _unknown_params(spelled)


def _candidate(node: c_ast.Node) -> bool:
    if isinstance(node, c_ast.Typedef):
        return True
    return (
        isinstance(node, c_ast.Decl)
        and "static" not in node.storage
        and node.init is None
        and (
            "extern" in node.storage
            or isinstance(node.type, c_ast.FuncDecl)
            or (
                node.name is None
                and isinstance(node.type, (c_ast.Struct, c_ast.Union, c_ast.Enum))
                and (getattr(node.type, "decls", None) is not None or getattr(node.type, "values", None) is not None)
            )
        )
    )


def _abi(
    node: c_ast.Node,
    typedefs: Mapping[str, c_ast.Node],
    records: Mapping[tuple[str, str | None], c_ast.Node],
    seen: frozenset[Any] = frozenset(),
) -> tuple[Any, ...] | None:
    """Known N64 storage/call shape; unresolved names and bounds never prove compatibility."""
    if isinstance(node, c_ast.TypeDecl):
        return _abi(node.type, typedefs, records, seen)
    if isinstance(node, c_ast.PtrDecl):
        return ("integer", 4, 4)
    if isinstance(node, c_ast.IdentifierType):
        name = " ".join(node.names)
        if name in typedefs and name not in seen:
            return _abi(typedefs[name], typedefs, records, seen | {name})
        name = _normal(name)
        if name != "signed char":
            name = re.sub(r"\bsigned\b\s*", "", name) or "int"
        name = re.sub(r"\b(short|long(?: long)?) int\b", r"\1", name)
        if name == "unsigned":
            name = "unsigned int"
        widths = {
            "void": 0,
            "char": 1,
            "signed char": 1,
            "unsigned char": 1,
            "short": 2,
            "unsigned short": 2,
            "int": 4,
            "unsigned int": 4,
            "long": 4,
            "unsigned long": 4,
            "long long": 8,
            "unsigned long long": 8,
            "float": 4,
            "double": 8,
        }
        size = widths.get(name)
        return None if size is None else ("float" if name in ("float", "double") else "integer", size, size)
    if isinstance(node, c_ast.Enum):
        return ("integer", 4, 4)
    if isinstance(node, c_ast.ArrayDecl):
        element = _abi(node.type, typedefs, records, seen)
        try:
            count = int(node.dim.value, 0) if node.dim is not None else None
        except (AttributeError, ValueError):
            return None
        return ("array", element, count) if element is not None else None
    if isinstance(node, c_ast.FuncDecl):
        result = _abi(node.type, typedefs, records, seen)
        if node.args is None:
            return ("function", result, None) if result is not None else None
        args = [
            ("integer", 4, 4)
            if isinstance(n.type, (c_ast.ArrayDecl, c_ast.FuncDecl))
            else _abi(n.type, typedefs, records, seen)
            for n in node.args.params
            if hasattr(n, "type")
        ]
        if len(args) != len(node.args.params) or result is None or None in args:
            return None
        return ("function", result, tuple([] if args == [("integer", 0, 0)] else args))
    if isinstance(node, (c_ast.Struct, c_ast.Union)):
        key = (type(node).__name__, node.name)
        body = node.decls or getattr(records.get(key), "decls", None)
        if body is None or key in seen:
            return None
        fields, offset, alignment = [], 0, 1
        for field in body:
            shape = _abi(field.type, typedefs, records, seen | {key})
            if shape is None or field.bitsize is not None:
                return None
            if shape[0] == "array":
                if shape[2] is None or shape[1][0] not in ("integer", "float", "record"):
                    return None
                size, align = shape[1][1] * shape[2], shape[1][2]
            elif shape[0] in ("integer", "float", "record"):
                size, align = shape[1:3]
            else:
                return None
            alignment = max(alignment, align)
            start = 0 if isinstance(node, c_ast.Union) else (offset + align - 1) // align * align
            fields.append((start, shape))
            offset = max(offset, start + size)
        return ("record", (offset + alignment - 1) // alignment * alignment, alignment)
    return None


def _records(ast: c_ast.Node) -> dict[tuple[str, str | None], c_ast.Node]:
    found: dict[tuple[str, str | None], c_ast.Node] = {}

    class Records(c_ast.NodeVisitor):
        def visit_Struct(self, node: c_ast.Struct) -> None:
            if node.decls is not None:
                found[(type(node).__name__, node.name)] = node
            self.generic_visit(node)

        visit_Union = visit_Struct

    Records().visit(ast)
    return found


def _signedness(node: c_ast.Node, typedefs: Mapping[str, c_ast.Node], seen: frozenset[str] = frozenset()) -> str | None:
    """The signedness class of the scalar an object declaration finally names, none for functions and records."""
    if isinstance(node, (c_ast.TypeDecl, c_ast.PtrDecl, c_ast.ArrayDecl)):
        return _signedness(node.type, typedefs, seen)
    if isinstance(node, c_ast.IdentifierType):
        name = " ".join(node.names)
        if name in typedefs and name not in seen:
            return _signedness(typedefs[name], typedefs, seen | {name})
        name = _normal(name)
        if "unsigned" in name:
            return "unsigned"
        return name if name in ("char", "float", "double") else "signed"
    return None


def _opaque(text: str, typedefs: Mapping[str, c_ast.Node]) -> set[str]:
    """The names a spelled type uses as types that are no type word, typedef or synonym: parsed as opaque records."""
    found = set(re.findall(r"(?:^|[(,])\s*(\w+)", text)) - _TYPE_WORDS - set(_SYNONYMS)
    return {n for n in found if n not in typedefs}


class _Shapes:
    """The (signedness, ABI shape) of spelled types under one unit's typedefs and records. The typedef prefix is made
    once, and `prime` parses every declaration a fold will ask about in one parser call; a text not primed (or a batch
    that does not parse) is parsed alone, as before."""

    parses = 0  # parser calls made, for the work-count tests

    def __init__(
        self, typedefs: Mapping[str, c_ast.Node], records: Mapping[tuple[str, str | None], c_ast.Node] | None
    ) -> None:
        self.typedefs, self.records = typedefs, records
        self.texts: dict[str, tuple[str, frozenset[str]]] = {}  # per typedef its declaration and the words it names
        self.known: dict[tuple[str, frozenset[str]], tuple[str | None, Any]] = {}

    def _typedef(self, name: str) -> tuple[str, frozenset[str]]:
        if name not in self.texts:
            text = c_generator.CGenerator().visit(c_ast.Typedef(name, [], ["typedef"], deepcopy(self.typedefs[name])))
            self.texts[name] = (text + ";", frozenset(re.findall(r"[A-Za-z_]\w*", text)))
        return self.texts[name]

    def _body(self, texts: Iterable[str]) -> str:
        """The typedefs the texts name, and the ones those name, in the unit's order: the only ones their parse reads
        (a word that is a typedef anywhere in the texts is in it, so each text parses as under every typedef)."""
        pending = [w for t in texts for w in re.findall(r"[A-Za-z_]\w*", t) if w in self.typedefs]
        named: set[str] = set()
        while pending:
            name = pending.pop()
            if name not in named:
                named.add(name)
                pending += [w for w in self._typedef(name)[1] if w in self.typedefs and w not in named]
        typedefs = self.typedefs
        order = typedefs.order(named) if isinstance(typedefs, _Lookup) else [n for n in typedefs if n in named]
        return "\n".join(self._typedef(n)[0] for n in order)

    def _parse(self, names: Collection[str], declarations: Sequence[str]) -> c_ast.FileAST:
        _Shapes.parses += 1
        prefix = "".join(f"typedef struct __opaque_{n} {n};\n" for n in sorted(names)) + self._body(declarations)
        text = prefix + "\n" + "\n".join(d + ";" for d in declarations)
        _Index.chars += len(text)
        return CParser().parse(text)

    def _measure(self, prefix: Mapping[Any, c_ast.Node], node: c_ast.Node) -> tuple[str | None, Any]:
        records: Mapping[Any, c_ast.Node] = ChainMap(_records(node), dict(prefix), dict(self.records or {}))
        return _signedness(node.type, self.typedefs), _abi(node.type, self.typedefs, records)

    @staticmethod
    def _declaration(text: str, index: int = -1) -> str:
        text, mark = _normal(text), "__candidate" if index < 0 else f"__candidate{index}"
        return text.replace("@", mark) if "@" in text else text + " " + mark

    def prime(self, pairs: Iterable[tuple[str, str]]) -> None:
        """Parses at once every text of these (left, right) pairs not shaped yet, each with its pair's opaque names."""
        wanted: dict[tuple[str, frozenset[str]], None] = {}
        for left, right in pairs:
            names = frozenset(_opaque(left + "," + right, self.typedefs))
            wanted.update(dict.fromkeys((t, names) for t in (left, right) if (t, names) not in self.known))
        if not wanted:
            return
        groups: dict[frozenset[str], list[str]] = defaultdict(list)
        for text, names in wanted:  # one parse per opaque name set: an extra opaque typedef could read a text otherwise
            groups[names].append(text)
        for names, texts in groups.items():
            try:
                ast = self._parse(names, [self._declaration(t, i) for i, t in enumerate(texts)])
            except ParseError:
                continue  # each is parsed alone when asked, so a bad one refuses only itself
            heads = len(ast.ext) - len(texts)
            prefix = _records(c_ast.FileAST(ast.ext[:heads]))
            for text, node in zip(texts, ast.ext[heads:], strict=True):
                self.known[text, names] = self._measure(prefix, node)

    def shape(self, text: str, names: frozenset[str]) -> tuple[str | None, Any]:
        if (text, names) not in self.known:
            try:
                ast = self._parse(names, [self._declaration(text)])
                self.known[text, names] = self._measure(_records(c_ast.FileAST(ast.ext[:-1])), ast.ext[-1])
            except ParseError:
                self.known[text, names] = (None, None)
        return self.known[text, names]


_shaping: list[_Shapes] = []  # the shapes of the fold running in this process


def _compatible(
    left: str,
    right: str,
    typedefs: Mapping[str, c_ast.Node] | None = None,
    records: Mapping[tuple[str, str | None], c_ast.Node] | None = None,
) -> bool:
    if re.findall(r"\b(?:const|volatile)\b", left) != re.findall(r"\b(?:const|volatile)\b", right):
        return False
    if "@)(" in left + right and _unnamed(left) != _unnamed(right):
        return False  # every pointer has one ABI, but a call through it is checked against its parameters
    typedefs = {} if typedefs is None else typedefs
    shapes = next((s for s in _shaping if s.typedefs is typedefs and s.records is records), None)
    shapes = shapes or _Shapes(typedefs, records)
    names = frozenset(_opaque(left + "," + right, typedefs))
    signs: list[str | None] = []

    def shape(text: str) -> tuple[Any, ...] | None:
        sign, found = shapes.shape(text, names)
        signs.append(sign)
        return found  # type: ignore[no-any-return]

    a, b = shape(left), shape(right)
    if signs[0] != signs[1]:
        return False  # an object read as signed or unsigned compiles to other loads and compares
    unspecified = (None, ())
    if (
        a is not None
        and b is not None
        and a[0] == b[0] == "function"
        and a[1] == b[1]
        and (a[2] in unspecified and b[2] in unspecified)
    ):
        return True  # void and an unspecified list emit the same no-argument call
    if a is not None and b is not None and a[0] == b[0] == "array":
        return a[1] == b[1] and (a[2] is None or a[2] == b[2])
    return a is not None and a == b


def _spelled_pair(name: str, statement: str, other: str, adopt: bool = True) -> tuple[str, str]:
    """The types two declarations give a name, as `_clash` compares them."""

    def spelled(text: str) -> str:
        kind = _normal(_type_of(text, name), qualified=True)
        return kind if adopt else kind.partition("@(")[0].strip()

    return spelled(statement), spelled(other)


def _clash(
    path: str,
    line: int,
    name: str,
    statement: str,
    entry: tuple[str, int, str],
    typedefs: Mapping[str, c_ast.Node] | None = None,
    adopt: bool = True,
    records: Mapping[tuple[str, str | None], c_ast.Node] | None = None,
) -> list[Finding]:
    """A finding when a candidate types a name unlike its header: a function by return type, others by folded type."""

    mine, theirs = _spelled_pair(name, statement, entry[2], adopt)
    if not mine or not theirs or mine == theirs or (adopt and _compatible(mine, theirs, typedefs, records)):
        return []
    return [
        Finding(
            "headers.conflict",
            path=path,
            line=line,
            action="declare it the way the project does",
            reason=(
                f"{name}: candidate declares {statement}, the type map has {entry[2]}"
                if entry[0] == "types.toml"
                else f"{name}: candidate declares '{mine}', {entry[0]}:{entry[1]} declares '{theirs}'"
            ),
        )
    ]


def _body_clash(path: str, node: c_ast.Node, name: str, entry: tuple[str, int, str]) -> list[Finding]:
    """A finding when a candidate defines a struct or union that its header defines with other members."""
    body = getattr(getattr(node, "type", None), "decls", None)
    if not isinstance(node, c_ast.Decl) or node.name or not body or "{" not in entry[2]:
        return []

    def named(members: Iterable[str]) -> set[str]:
        return {m for m in members if not m.startswith("pad")}

    mine = named(d.name for d in body if d.name)
    theirs = named(re.findall(r"(\w+)\s*(?:\[[^\]]*\]\s*)*[;,]", entry[2].partition("{")[2]))
    if mine == theirs:
        return []
    return [
        Finding(
            "headers.conflict",
            path=path,
            line=_coord(node).line,
            action=f"declare the members in {entry[0]}",
            reason=f"{name}: candidate defines {sorted(mine)}, {entry[0]}:{entry[1]} defines {sorted(theirs)}",
        )
    ]


_COMMENT = r"/\*(?:(?!\*/).)*\*/"  # one comment: it cannot reach across a later close to swallow code
_OPEN = re.compile(
    r"\A(?:\s|"
    + _COMMENT
    + r"|//[^\n]*\n)*?([ \t]*#[ \t]*ifndef[ \t]+(\w+)[ \t]*\n[ \t]*#[ \t]*define[ \t]+\2\b[^\n]*\n)",
    re.S,
)
_CLOSE = re.compile(r"\n[ \t]*#[ \t]*endif\b[^\n]*(?:\s|" + _COMMENT + r"|//[^\n]*)*\Z", re.S)
_INCLUDE = re.compile(r'^[ \t]*#\s*include\s*"([^"]+)"[^\n]*\n?', re.M)


def _unguarded(text: str) -> str:
    """The text without its include guard: the opening ifndef/define pair (comments may precede it) and the endif that
    closes it. A text whose guard is not the one wrapper of the whole file is left whole, so it stays balanced."""
    opened, closed = _OPEN.match(text), _CLOSE.search(text)
    if opened is None or closed is None or closed.start() < opened.end() - 1:
        return text
    inner = text[opened.end() : closed.start()]
    depth = 0
    for word in re.findall(r"^[ \t]*#[ \t]*(if\w*|endif)\b", _blank(inner), re.M):
        depth += 1 if word.startswith("if") else -1
        if depth < 0:
            return text  # the final endif closes something else
    return text[: opened.start(1)] + inner.lstrip("\n") if depth == 0 else text


def _appended(text: str, body: str) -> str:
    """The header text with body added at its end, inside its include guard when it has one."""
    closed = _CLOSE.search(text)
    at = closed.start() if closed is not None and _unguarded(text) != text else len(text)
    return f"{text[:at].rstrip(chr(10))}\n\n{body.strip(chr(10))}\n{text[at:] or chr(10)}"


def wrapped(path: str, body: str) -> str:
    """The body inside the include guard that the header path names: <PATH BELOW include/>_H, no tool name."""
    guard = re.sub(r"[^A-Z0-9]", "_", f"{re.sub(r'(?i)[.]h$', '', path.removeprefix('include/'))}_H".upper())
    return f"#ifndef {guard}\n#define {guard}\n\n{body.strip(chr(10))}\n\n#endif\n"


def rewrite_includes(text: str, moves: Mapping[str, str], at: str = "") -> str:
    """The text with each quoted include of a moved header spelled for its new path; a target of "" drops the line.
    A name resolves beside the includer `at`, else under include/ or src/ (the rule consumers uses)."""

    seen: set[str] = set()

    def one(match: re.Match[str]) -> str:
        line = match[0]
        for base in (posixpath.dirname(at), "include", "src"):
            target = posixpath.normpath(posixpath.join(base, match[1]))
            if target in moves:
                keep = moves[target] and moves[target] != at  # a header merged into its includer is no include of it
                line = match[0].replace(match[1], posixpath.relpath(moves[target], base), 1) if keep else ""
                break
        if line in seen:  # two includes that now name one header keep the first
            return ""
        seen.add(line)
        return line

    return _INCLUDE.sub(one, text)


def _names(name: str, origin: str) -> set[str]:
    """The paths a quoted include may name from the file at `origin`: beside it, under include/ or src/."""
    return {posixpath.normpath(posixpath.join(base, name)) for base in (posixpath.dirname(origin), "include", "src")}


def _merged(
    path: str,
    texts: Sequence[str],
    origins: Sequence[str] = (),
    reach: Callable[[str, Collection[str]], bool] | None = None,
    read: Callable[[str], str | None] | None = None,
    authority: Callable[[str], str | None] | None = None,
) -> str:
    """The merged header text alone (see _merge)."""
    return _merge(path, texts, origins, reach, read, authority)[0]


@lru_cache(maxsize=1 << 18)  # a chain of merges into one header meets the same statements again
def _canon(statement: str) -> str:
    """A statement's token text: one spelling whatever its spacing and comments."""
    rows = _declarations.dependencies(statement)
    return " ".join(r[2] for r in rows) if rows else " ".join(statement.split())


def _parsed(statement: str) -> str | None:
    """The statement as the C parser prints it back, or None when it does not parse alone."""
    try:
        return " ".join(c_generator.CGenerator().visit(parse(statement, "<merge>")).split())
    except Refusal:  # a statement that names types defined elsewhere does not parse alone
        return None


_TYPE_STATEMENT = re.compile(r"(?:typedef|struct|union|enum)\b")


def _type_names(statement: str) -> set[str]:
    """The names a typedef or a complete record or enum definition gives."""
    spelled = " ".join(statement.split())
    if not _TYPE_STATEMENT.match(spelled) or ("{" not in spelled and not spelled.startswith("typedef")):
        return set()
    named = set(re.findall(r"\b(?:struct|union|enum)\s+(\w+)", statement))
    defined = set(re.findall(r"\b(?:struct|union|enum)\s+(\w+)\s*\{", statement))
    return set(_declarations.pure_names(statement)) - (named - defined)


def declared_names(statement: str) -> set[str]:
    """The objects and functions one declaration without a body declares: a tag it only names is not one of them."""
    tags = set(re.findall(r"\b(?:struct|union|enum)\s+(\w+)", statement))
    return set(_declarations.pure_names(statement)) - tags


def _pulled(path: str, words: Collection[str], outside: str, text: str) -> tuple[list[str], str]:
    """The type statements (typedefs, records, enums) of an outside header that the merged body uses, with the ones
    they use in turn, and the outside header without them. A pulled statement that needs a macro the outside header
    keeps refuses move.cycle: it cannot stand before that header."""
    rows = []
    for start, end, statement in statements(text):
        if _TYPE_STATEMENT.match(" ".join(statement.split())):
            found = _declarations.dependencies(statement) or []
            names = set().union(*(r[0] for r in found)) if found else set()
            uses = set().union(*(r[1] for r in found)) if found else set(re.findall(r"\w+", statement))
            rows.append((start, end, names, uses - names))
    used = set(words)  # the words of the merged body
    taken: set[int] = set()
    grew = True
    while grew:
        grew = False
        for i, (_, _, names, uses) in enumerate(rows):
            if i not in taken and names & used:
                taken.add(i)
                used |= uses
                grew = True
    if not taken:
        return [], text
    macros = set(re.findall(r"^[ \t]*#\s*define\s+(\w+)", text, re.M))
    for i in sorted(taken):
        start, end, _, uses = rows[i]
        if needed := sorted(uses & macros):
            spelled = " ".join(text[start:end].split())
            raise Refusal(
                Finding(
                    "move.cycle",
                    f"{spelled[:120]} from {outside} needs {needed[0]}, which {outside} defines, before {path}",
                    path=path,
                )
            )
    pulled = [text[rows[i][0] : rows[i][1]] for i in sorted(taken)]
    for i in sorted(taken, reverse=True):
        start, end = rows[i][0], rows[i][1]
        text = text[:start] + text[end + (text[end : end + 1] == "\n") :]
    return pulled, text


def _merge(
    path: str,
    texts: Sequence[str],
    origins: Sequence[str] = (),
    reach: Callable[[str, Collection[str]], bool] | None = None,
    read: Callable[[str], str | None] | None = None,
    authority: Callable[[str], str | None] | None = None,
) -> tuple[str, dict[str, str]]:
    """One header at path holding the texts concatenated in include order, and the outside headers it rewrites.

    A text goes before the texts that include its file (`origins` names each text's file, "" for part of one); an
    include of a merged-in file, or of the header itself, is dropped. An include of an outside header stays where it
    stood, after the statements above it. A statement is kept once per token text; two complete definitions of one
    name that the parser prints alike are one, different shapes refuse move.declaration with both. Two declarations of
    one global or prototype that differ keep the one `authority(name)` gives (the type map), and refuse without one.
    With `reach` (header, files -> whether the header includes any of them), an outside header that reaches the
    merged files includes the merged header back: under the guard that inner include is a no-op, so the back-include
    goes after the whole body. With `read`, the type statements of that header which the body uses move into the
    merged header first (they leave it): after the body would be too late for them."""
    origin = [*origins, *[""] * len(texts)][: len(texts)]
    unguarded = [_unguarded(t) for t in texts]
    merged = {path, *filter(None, origin)}

    def inside(name: str, here: str, part: int) -> int | None:
        """What an include names: the header itself is -1, a merged-in file its text index, an outside one None."""
        spelled = _names(name, here)
        found = next((j for j, o in enumerate(origin) if o and j != part and o in spelled), None)
        return -1 if found is None and path in spelled else found

    before: dict[int, set[int]] = {i: set() for i in range(len(texts))}
    for i, text in enumerate(unguarded):
        for match in _INCLUDE.finditer(text):
            j = inside(match[1], origin[i] or path, i)
            if j is not None and j >= 0:
                before[i].add(j)
    order: list[int] = []
    while len(order) < len(texts):
        waiting = [i for i in range(len(texts)) if i not in order]
        order.append(next((i for i in waiting if before[i] <= set(order)), waiting[0]))
    spelled_includes: set[str] = set()
    back: list[tuple[str, str]] = []  # (line, outside header) that includes this one again: after the body
    kept: list[str] = []
    for i in order:
        here = origin[i] or path

        def keep(match: re.Match[str], i: int = i, here: str = here) -> str:
            if inside(match[1], here, i) is not None or match[1] in spelled_includes:
                return ""
            spelled_includes.add(match[1])
            if reach and (found := [o for o in sorted(_names(match[1], here) - merged) if reach(o, merged)]):
                back.append((match[0].rstrip("\n"), found[0]))
                return ""
            return match[0]

        kept.append(_INCLUDE.sub(keep, unguarded[i]))
    rows = [(k, start, end, statement) for k, text in enumerate(kept) for start, end, statement in statements(text)]
    drop, swap = _repeats(path, [r[3] for r in rows], authority)
    bodies = list(kept)
    for n in sorted(drop | swap.keys(), key=lambda n: (rows[n][0], -rows[n][1])):
        k, start, end, _ = rows[n]
        bodies[k] = bodies[k][:start] + swap.get(n, "") + bodies[k][end:]
    body = "\n\n".join(b.strip("\n") for b in bodies if b.strip("\n"))
    rewrites: dict[str, str] = {}
    if read and back:
        have = {_canon(s) for _, _, s in statements(body)}  # tokenized once for every back-include
        words = set(re.findall(r"[A-Za-z_]\w*", _blank(body)))
        above: list[str] = []  # what a back-included header includes that reaches no merged file: above the body
        for _, outside in back:
            if (found := read(outside)) is not None:
                for match in _INCLUDE.finditer(_unguarded(found)):
                    there = [c for c in sorted(_names(match[1], outside) - merged) if read(c) is not None]
                    if there and reach and not any(reach(c, merged) for c in there):
                        line = f'#include "{there[0].removeprefix("include/")}"'
                        if line not in above and line not in body:
                            above.append(line)
                pulled, rest = _pulled(path, words, outside, found)
                fresh = [p for p in pulled if _canon(p) not in have]
                if pulled:
                    rewrites[outside] = rest
                have |= {_canon(p) for p in fresh}
                words |= {w for p in fresh for w in re.findall(r"[A-Za-z_]\w*", _blank(p))}
                body = "\n\n".join([body, *fresh]) if fresh else body
        body = "\n".join([*above, body]) if above else body
    body = _define_before_use(path, body)
    text = wrapped(path, "\n\n".join(b for b in (body.strip("\n"), "\n".join(line for line, _ in back)) if b))
    if clash := duplicates(path, text):
        raise Refusal(replace(clash[0], key="move.declaration"))
    return text, rewrites


def _repeats(
    path: str, spelled: Sequence[str], authority: Callable[[str], str | None] | None
) -> tuple[set[int], dict[int, str]]:
    """Which statements of a merged body go (a repeat) and which are spelled anew (the authority's declaration)."""
    drop: set[int] = set()
    swap: dict[int, str] = {}
    seen: set[str] = set()
    shapes: dict[str, int] = {}  # the first complete definition of each name
    declared: defaultdict[str, list[int]] = defaultdict(list)
    for n, statement in enumerate(spelled):
        canon = _canon(statement)
        if canon in seen:
            drop.add(n)
            continue
        seen.add(canon)
        if names := _type_names(statement):
            if "{" not in statement:
                continue
            again = [m for m in names if m in shapes]
            shape = _parsed(statement) if again else None  # parsed only for a name defined twice
            if again and all(shape is not None and _parsed(spelled[shapes[m]]) == shape for m in again):
                drop.add(n)
                continue
            if again:
                first = " ".join(spelled[shapes[again[0]]].split())
                raise Refusal(
                    Finding(
                        "move.declaration",
                        f"{again[0]} is defined two ways: {first} | {' '.join(statement.split())}",
                        path=path,
                        unit=again[0],
                    )
                )
            shapes.update(dict.fromkeys(names, n))
        elif "{" not in statement:
            found = declared_names(statement)
            if len(found) == 1:
                declared[found.pop()].append(n)
    for name, ns in declared.items():
        kinds = {_spelling(spelled[n], name) for n in ns}
        if len(kinds) < 2:
            continue
        chosen = authority(name) if authority else None
        if chosen is None:
            texts = " | ".join(" ".join(spelled[n].split()) for n in ns[:2])
            raise Refusal(Finding("move.declaration", f"{name} is declared two ways: {texts}", path=path, unit=name))
        right = [n for n in ns if _spelling(spelled[n], name) == _spelling(chosen, name)]
        drop |= set(ns) - set(right[:1])
        if not right:
            swap[ns[0]] = chosen
            drop.discard(ns[0])
    return drop, swap


_TAG_USE = re.compile(r"\b((?:struct|union|enum)\s+\w+)\b(?!\s*\*)")
_FORWARD = re.compile(r"typedef\s+(?:struct|union|enum)\s+\w+\s+\w+\s*;")


def _define_before_use(path: str, text: str) -> str:
    """The top-level units of a merged body (each statement with the directives and comments above it) in a stable
    order where a unit comes after the units that define what it uses. Names used only through pointers need no
    earlier definition. A real cycle refuses with both names."""
    spans, last = [], 0
    for _, end, _ in statements(text):
        spans.append((last, end))
        last = end
    if len(spans) < 2:
        return text
    units: list[str] = []
    barrier: list[bool] = []  # an include line is a unit of its own that no later statement may move above
    for a, b in spans:
        unit = text[a:b].strip("\n")
        cut = max((m.end() for m in _INCLUDE.finditer(unit)), default=0)
        if cut and unit[cut:].strip():
            units += [unit[:cut].strip("\n"), unit[cut:].strip("\n")]
            barrier += [True, False]
        else:
            units.append(unit)
            barrier.append(bool(cut))
    tail = text[last:].strip("\n")
    defines: list[set[str]] = []
    for unit in units:
        macros = re.findall(r"^[ \t]*#\s*define\s+(\w+)", unit, re.M)
        defines.append({n for n, _, _ in _declared(unit)} | set(macros))
    definer: dict[str, int] = {}
    for i, found in enumerate(defines):
        for name in sorted(found):
            definer.setdefault(name, i)
    needs: list[set[int]] = []
    above = -1  # the last include unit before this one: what it declares is what the statements after it use
    for i, unit in enumerate(units):
        # a macro body names what its users need, not what must exist where it is defined: only code counts
        code = re.sub(_DIRECTIVE, "", _blank(unit), flags=re.M).strip()
        used = set(re.findall(r"\b[A-Za-z_]\w*\b", code))
        if not _FORWARD.fullmatch(code):
            used |= {" ".join(m.split()) for m in _TAG_USE.findall(code)}
        needs.append(
            {definer[n] for n in used if n in definer and definer[n] != i} | ({above} if above >= 0 else set())
        )
        if barrier[i]:
            above = i
    # the lowest-numbered unit whose needs are placed goes next (a heap keeps it linear in units and needs)
    left = [len(n) for n in needs]
    users: list[list[int]] = [[] for _ in units]
    for i, need in enumerate(needs):
        for j in need:
            users[j].append(i)
    ready = [i for i, n in enumerate(left) if not n]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        pick = heapq.heappop(ready)
        order.append(pick)
        for user in users[pick]:
            left[user] -= 1
            if not left[user]:
                heapq.heappush(ready, user)
    if len(order) < len(units):
        placed = set(order)
        a = next(i for i in range(len(units)) if i not in placed)
        b = sorted(needs[a] - placed)[0]
        first = sorted(defines[a])[0] if defines[a] else f"unit {a}"
        second = sorted(defines[b])[0] if defines[b] else f"unit {b}"
        raise Refusal(
            Finding("move.cycle", f"{first} and {second} each need the other defined first in {path}", path=path)
        )
    if order == sorted(order):
        return text
    return "\n\n".join([units[i] for i in order] + ([tail] if tail else []))


def _spelled(snapshot: Snapshot, header: str, text: str) -> str:
    """The text with each quoted include spelled from include/ when it names a sibling of `header` by its own folder."""
    return _INCLUDE.sub(lambda m: m[0].replace(f'"{m[1]}"', f'"{_rooted(snapshot, header, m[1])}"', 1), text)


def names(text: str) -> set[str]:
    """Every name a piece of C text declares or defines at file scope, and the macros it defines."""
    raw = text.encode()
    macros = re.findall(r"^[ \t]*#\s*define\s+(\w+)", text, re.M)
    return {n for n, _, _ in _declared(text)} | set(defined(raw)) | set(variables(raw)) | set(macros)


def _empty(text: str) -> bool:
    """Whether a header holds nothing but its guard, includes and blank lines."""
    return not _INCLUDE.sub("", _unguarded(text)).strip()


def _owned(text: str, names: Collection[str]) -> tuple[str, str]:
    """The statements of a header that declare any of the names, and the header without them."""
    taken: list[str] = []
    for start, end, statement in reversed(list(statements(text))):
        if {n for n, _, _ in _declared(statement)} & set(names):
            taken.insert(0, text[start:end])
            text = text[:start] + text[end + (text[end : end + 1] == "\n") :]
    return "\n".join(taken), text


def included(text: str, paths: Collection[str]) -> str:
    """The text with an include of each path it lacks, after its last include before its first statement (a merged
    unit can hold a member's include further down, after code that needs what the new include declares)."""
    wanted = sorted(p for p in paths if not re.search(r'^\s*#\s*include\s*[<"]' + re.escape(p) + r'[>"]', text, re.M))
    if not wanted:
        return text
    first = next((a for a, _, _ in statements(text)), len(text))
    found = [m for m in re.finditer(r"^\s*#\s*include[^\n]*(?:\n|$)", text, re.M)]
    matches = (
        [m for m in found if m.start() < first]
        or found
        or list(re.finditer(r"^[ \t]*#\s*define(?:[^\n]*\\\n)*[^\n]*\n", text, re.M))
    )
    position = matches[-1].end() if matches else 0
    prefix = "\n" if position and text[position - 1] != "\n" else ""
    return text[:position] + prefix + "".join(f'#include "{p}"\n' for p in wanted) + text[position:]


def _needs(declaration: str, known: Mapping[str, tuple[str, int, str]], own: str | None = None) -> set[str]:
    """The headers that declare the types a declaration names: what a header that takes it in must include."""
    return {
        known[n][0].removeprefix("include/")
        for n in re.findall(r"\b(?:struct|union|enum) \w+|\w+", declaration)
        if n in known and n != own
    }


def _closure(snapshot: Snapshot, starts: Collection[str]) -> set[str]:
    """Every existing file the starts include, directly or through others, by a quoted name."""
    seen: set[str] = set()
    pending = list(starts)
    while pending:
        path = pending.pop()
        for name in _quoted(snapshot.peek(path) or b""):
            for base in (posixpath.dirname(path), "include", "src"):
                found = posixpath.normpath(posixpath.join(base, name))
                if found not in seen and snapshot.peek(found) is not None:
                    seen.add(found)
                    pending.append(found)
    return seen


def reaches(snapshot: Snapshot, start: str, target: str) -> bool:
    """Whether the header at start includes, directly or not, the header at target."""
    seen, pending = set(), [start]
    while pending:
        path = pending.pop()
        if path == target:
            return True
        if path not in seen:
            seen.add(path)
            _Read.paths.add(path)
            text = (snapshot.peek(path) or b"").decode(errors="replace")
            for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"', text, re.M):
                pending += [posixpath.normpath(f"{base}/{name}") for base in ("include", posixpath.dirname(path))]
    return False


def pins(snapshot: Snapshot, paths: Sequence[str]) -> str:
    """One key for the text of every path, as `catalog` keys its headers."""
    return digest([(p, _view._pin(snapshot, p)) for p in paths])


def _each(snapshot: Snapshot, place: str, suffix: str) -> dict[str, str]:
    """Per `sources` path of a place its pin, in path order. An overlay (a wave's writes, a fold's writes laid over
    a wave) keeps the pins of the snapshot it was laid over and pins only the files it overlays differently, so only
    the first snapshot of a chain lists from disk."""
    root = str(snapshot.config.project.root)
    libraries = tuple(f.rstrip("/") + "/" for f in getattr(snapshot.config.project, "library_dirs", ()))

    def produce() -> dict[str, str]:
        parent = (effort.peek(("parent", getattr(snapshot, "digest", ""))) or (None,))[0]
        if parent is None:
            return {p: _view._pin(snapshot, p) for p in sources(snapshot, place, suffix)}
        found = dict(_each(parent, place, suffix))
        for path, data in snapshot.overlays.items():
            if not path.startswith(place + "/") or not _kind_of(suffix)(path) or path.startswith(libraries):
                continue
            if path in parent.overlays and parent.overlays[path] == data:
                continue
            if data is None:
                found.pop(path, None)
            else:
                found[path] = _view._pin(snapshot, path)
        return dict(sorted(found.items()))

    return effort.memo(("pin", "each", root, place, suffix, getattr(snapshot, "digest", "")), produce)


def _listed(snapshot: Snapshot, place: str = "src", suffix: str = ".c") -> list[tuple[str, str, dict[str, Any]]]:
    """(path, pin, scan) of every `sources` path of a place: a text whose pin was scanned already is not read."""
    each = _each(snapshot, place, suffix)
    paths = list(each)
    return list(zip(paths, each.values(), _scanned(snapshot, paths, each), strict=True))


def landed_sources(snapshot: Snapshot, place: str = "src", suffix: str = ".c") -> tuple[list[str], str]:
    """The `sources` of a place and one pin of all their texts, listed once per snapshot until a store write (which
    forgets every pin) changes the tree: each fold of a wave asks for them."""
    root = str(snapshot.config.project.root)

    def produce() -> tuple[list[str], str]:
        each = _each(snapshot, place, suffix)
        return list(each), digest(list(each.items()))

    return effort.memo(("pin", "listed", root, place, suffix, getattr(snapshot, "digest", "")), produce)


def _spelled_part(scanned: dict[str, Any]) -> list[tuple[str, str]]:
    """Every type a source declares a name with: its externs and its prototypes, at file or block scope."""
    rows = [*scanned["declared"], *scanned.get("prototypes", ())]
    return [(name, s) for name, _, text in rows if (s := _unnamed(_normal(_type_of(text, name))))]


def spellings(snapshot: Snapshot) -> dict[str, frozenset[tuple[str, str]]]:
    """Per name the (landed source, type) of every declaration a landed source makes of it. Built once per pin of the
    sources, like `catalog`: a fold asks it once per unit, and the move builds it before a wave forks its workers. Each
    text's part is kept by its pin, so a new pin of the sources only merges the parts again."""
    key = landed_sources(snapshot)[1]

    def produce() -> dict[str, frozenset[tuple[str, str]]]:
        with effort.stage("headers.spellings"):
            found: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
            for path, pin, scanned in _listed(snapshot):
                if path.startswith("src/fuzzy/"):
                    continue
                for name, spelled in effort.memo(("spelled", pin), partial(_spelled_part, scanned)):
                    found[name].add((path, spelled))
            return {n: frozenset(v) for n, v in found.items()}

    return effort.memo(("spellings", str(snapshot.config.project.root), key), produce)


def warm(snapshot: Snapshot) -> None:
    """Builds in this process what every fold of the snapshot reads, so pool workers forked after it inherit it."""
    try:
        catalog(snapshot)
        spellings(snapshot)
        types.authority(snapshot)
    except Refusal:
        pass  # each fold meets the refusal itself and reports it for its unit


def _spelled_elsewhere(snapshot: Snapshot, path: str, names: Collection[str]) -> dict[str, set[str]]:
    """Per name the types the other landed sources declare it with."""
    index = spellings(snapshot)
    found = {name: {s for p, s in index.get(name, ()) if p != path} for name in names}
    return {name: spelled for name, spelled in found.items() if spelled}


def _disagree(snapshot: Snapshot, header: str, chosen: Collection[str]) -> bool:
    """Whether the header declares a name with another type than a chosen header does: a source cannot take both."""
    if not chosen:
        return False
    _Read.paths.update([header, *chosen])
    types = [_typed(snapshot, path) for path in [header, *chosen]]
    return any(types[0][name] != spelled for other in types[1:] for name, spelled in other.items() if name in types[0])


def _typed(snapshot: Snapshot, path: str) -> dict[str, str]:
    """Per name a header declares the type it spells, kept by the header's pin: read once per header text."""

    def produce() -> dict[str, str]:
        scanned = _scanned(snapshot, [path])[0]
        return {n: spelled for n, _, text in scanned["declared"] if (spelled := _unnamed(_normal(_type_of(text, n))))}

    return effort.memo(("typed", _view._pin(snapshot, path)), produce)


def _names_of(snapshot: Snapshot, path: str) -> frozenset[str]:
    """The names a header declares, record tags spelled with their keyword, kept by the header's pin."""
    produce = lambda: frozenset(n for n, _, _ in _scanned(snapshot, [path])[0]["declared"])  # noqa: E731
    return effort.memo(("declares", _view._pin(snapshot, path)), produce)


def _declared_in(snapshot: Snapshot, chosen: Collection[str], name: str) -> bool:
    _Read.paths.update(chosen)
    return any(n == name for scanned in _scanned(snapshot, list(chosen)) for n, _, _ in scanned["declared"])


def _walk(node: c_ast.Node) -> Iterable[c_ast.Node]:
    yield node
    for _, child in node.children():
        yield from _walk(child)


def _block_types(nodes: Iterable[c_ast.Node], generator: c_generator.CGenerator) -> dict[str, set[str]]:
    """Per name the types the function definitions among the nodes declare it with at block scope."""
    found: dict[str, set[str]] = {}
    for node in nodes:
        if isinstance(node, c_ast.FuncDef):
            for sub in _walk(node.body):
                if (
                    isinstance(sub, c_ast.Decl)
                    and sub.name
                    and ("extern" in sub.storage or isinstance(sub.type, c_ast.FuncDecl))
                    and (spelled := _unnamed(_normal(_type_of(generator.visit(sub) + ";", sub.name))))
                ):
                    found.setdefault(sub.name, set()).add(spelled)
    return found


def _own_types(nodes: Iterable[c_ast.Node]) -> set[str]:
    """The typedef names and the record tags (`struct X`) the nodes define: a second definition does not compile."""
    found: set[str] = set()
    for n in nodes:
        if isinstance(n, c_ast.Typedef):
            found.add(n.name)
        for sub in _walk(n):
            if isinstance(sub, (c_ast.Struct, c_ast.Union)) and sub.name and sub.decls is not None:
                found.add(f"{'struct' if isinstance(sub, c_ast.Struct) else 'union'} {sub.name}")
    return found


def _viewed(snapshot: Snapshot, text: str) -> set[str]:
    """The files a view's line markers name: what the unit includes today."""
    return {_path(snapshot, Coord(m, 0)) for m in re.findall(r'(?m)^#[ \t]*(?:line[ \t]+)?\d+[ \t]+"([^"]+)"', text)}


def _definitions(nodes: Iterable[c_ast.Node], generator: c_generator.CGenerator) -> dict[str, set[str]]:
    """Per name the types the unit's own function and object definitions spell (prototyped functions only)."""
    found: dict[str, set[str]] = {}
    for node in nodes:
        decl = node.decl if isinstance(node, c_ast.FuncDef) and not node.param_decls else node
        if (
            isinstance(decl, c_ast.Decl)
            and decl.name
            and "extern" not in decl.storage
            and (isinstance(node, c_ast.FuncDef) or not isinstance(decl.type, c_ast.FuncDecl))
            and (
                spelled := _unnamed(
                    _normal(
                        _type_of(
                            generator.visit(
                                c_ast.Decl(
                                    decl.name,
                                    decl.quals,
                                    decl.align,
                                    decl.storage,
                                    decl.funcspec,
                                    decl.type,
                                    None,
                                    None,
                                )
                            )
                            + ";",
                            decl.name,
                        )
                    )
                )
            )
        ):
            found.setdefault(decl.name, set()).add(spelled)
    return found


def _sizeless(spelled: str) -> str:
    """A spelled type without array sizes: `T x[4]` and `extern T x[]` are one type."""
    return re.sub(r"\[[^\]]*\]", "[]", spelled)


class _Mine(NamedTuple):
    inner: Mapping[str, set[str]]  # per name the types it declares or defines it with, at file or block scope
    types: set[str]  # the typedefs and record tags it defines
    words: set[str]  # every word its text spells
    today: set[str]  # the files it includes today


def _contradicts(snapshot: Snapshot, header: str, mine: _Mine) -> bool:
    """Whether the header, or a header it includes that the unit does not include today, declares a name the unit
    declares or defines with another type, defines a type the unit defines, or defines a macro the unit spells."""
    seen, pending = set(), [header]
    while pending:
        path = pending.pop()
        if path in seen or path in mine.today or snapshot.peek(path) is None:
            continue
        seen.add(path)
        _Read.paths.add(path)
        typed = _typed(snapshot, path)
        text = (snapshot.peek(path) or b"").decode(errors="replace")
        if (
            any(
                n in typed and {_sizeless(t) for t in spelled} != {_sizeless(typed[n])}
                for n, spelled in mine.inner.items()
            )
            or mine.types & _names_of(snapshot, path)
            or set(re.findall(r"(?m)^[ \t]*#[ \t]*define[ \t]+(\w+)", text)) & mine.words
        ):
            return True
        for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"', text, re.M):
            pending += [posixpath.normpath(f"{base}/{name}") for base in ("include", posixpath.dirname(path))]
    return False


def _staying(
    snapshot: Snapshot, owner: str, moved: list[Any], known: Mapping[str, Any], held: Sequence[Any] = ()
) -> list[Any]:
    """The moved declarations that must stay in the source: one that names a type from a header which includes the
    owner would be read before that type exists, and so would any moved declaration that names a staying one."""
    generator = c_generator.CGenerator()
    texts = {id(node): generator.visit(node) for _, node in [*moved, *held]}
    staying, names = list(held), {_name(node) for _, node in held}
    while True:
        more = [
            (span, node)
            for span, node in moved
            if (span, node) not in staying
            and (
                set(re.findall(r"\w+", texts[id(node)])) & names
                or any(reaches(snapshot, f"include/{n}", owner) for n in _needs(texts[id(node)], known, _name(node)))
            )
        ]
        if not more:
            return staying
        staying += more
        names |= {_name(node) for _, node in more}


def fold(
    snapshot: Snapshot, unit: UnitSpec, view: SourceView, strict: bool = True
) -> tuple[dict[str, bytes | None], tuple[Finding, ...]]:
    with effort.stage("headers.fold"):
        # the unit's own runs are parsed (once per text); every header run is an index entry, looked up by name
        _Read.names, _Read.paths, _Read.fold = set(), {unit.path}, True
        with effort.stage("headers.index"):
            regions = _unit_regions(view.text, unit.path, snapshot.config if hasattr(snapshot.config, "host") else None)
        ours = [r for r in regions if r.nodes and _path(snapshot, r.nodes[0].coord) == unit.path]
        typedefs = _Lookup(regions, "typedefs", last=True)
        records = _Lookup(regions, "records", last=True)
        visible = _Lookup([r for r in regions if all(r is not o for o in ours)], "first", last=False)
        mine = [node for r in ours for node in r.nodes]
        local = [node for node in mine if _name(node)]
        _Read.names.update(*(r.words for r in ours))  # its own text parses by what each name it spells is
        source = snapshot.read(unit.path).decode()
        spans = list(statements(source))
        offsets = [0] + [m.end() for m in re.finditer("\n", source)]
        grouped: dict[tuple[int, int], list[tuple[c_ast.Node, bool]]] = {}
        includes: set[str] = set()
        moved: list[tuple[tuple[int, int], c_ast.Node]] = []
        findings: list[Finding] = []
        authoritative: dict[tuple[int, int], str] = {}
        with effort.stage("headers.known"):
            known_ = catalog(snapshot)
            spellings(snapshot)
            authority = types.authority(snapshot)
        known = cast(dict[str, tuple[str, int, str]], _Asked(known_))  # a dict to read, its asks recorded
        generator = c_generator.CGenerator()
        named = {name for n in local if _candidate(n) and (name := _name(n)) is not None}
        _Read.names.update(named)
        elsewhere = _spelled_elsewhere(snapshot, unit.path, {name for name in named if name not in known})
        held: list[tuple[tuple[int, int], c_ast.Node]] = []
        chosen: set[str] = set()
        functions = _Asked(authority["function"])
        candidates = [(n, name) for n in local if _candidate(n) and (name := _name(n)) is not None]
        canonical = {id(n): _canonical(n, typedefs) for n, name in candidates if name in visible}
        canonical.update(
            {id(o): _canonical(o, typedefs) for _, name in candidates if (o := visible.get(name)) is not None}
        )
        pairs = [  # what the loop below compares: shaped in one parse
            (canonical[id(n)].replace("(", "@(", 1), canonical[id(visible[name])].replace("(", "@(", 1))
            if name in visible
            else _spelled_pair(name, generator.visit(n) + ";", known[name][2])
            for n, name in candidates
            if name in visible or name in known
        ]
        shapes = _Shapes(typedefs, records)
        shapes.prime(pairs)  # every comparison below is shaped from this one parse
        _shaping[:] = [shapes]
        own = {n.name for n in local if isinstance(n, c_ast.Typedef)}
        for n in local:
            for sub in _walk(n):
                if isinstance(sub, (c_ast.Struct, c_ast.Union)) and sub.name and sub.decls is not None:
                    own.add(sub.name)
        vetoed = {  # headers that contradict a landed source's own declaration are not included by it
            known[name][0]
            for node in local
            if not strict
            and _candidate(node)
            and (name := _name(node)) in known
            and name not in visible
            and (
                _clash(unit.path, 0, name, generator.visit(node) + ";", known[name], typedefs, records=records)
                or _body_clash(unit.path, node, name, known[name])
            )
        } | {  # nor are those that contradict a function it defines: it builds today without them
            known[node.decl.name][0]
            for node in mine
            if not strict
            and isinstance(node, c_ast.FuncDef)
            and node.decl.name in known
            and _clash(unit.path, 0, node.decl.name, generator.visit(node.decl), known[node.decl.name], adopt=False)
        }
        if not strict:  # nor those that would contradict, themselves or through a header they add, what it builds
            # with: a declaration of another type, at file scope or in a function (a merged unit holds each member's
            # own), a type it defines itself, or a macro of a name it spells
            source_words = set(re.findall(r"\w+", snapshot.read(unit.path).decode(errors="replace")))
            declared = _block_types(mine, generator)
            for n, name in candidates:
                if isinstance(n, c_ast.Decl) and (
                    spelled := _unnamed(_normal(_type_of(generator.visit(n) + ";", name)))
                ):
                    declared.setdefault(name, set()).add(spelled)
            for name, kinds in _definitions(mine, generator).items():  # a definition conflicts as a declaration does
                declared.setdefault(name, set()).update(kinds)
            mine_of = _Mine(declared, _own_types(local), source_words, _viewed(snapshot, view.text))
            vetoed |= {h for h in {known[n][0] for n in named if n in known} if _contradicts(snapshot, h, mine_of)}
        for node in local:
            offset = offsets[_coord(node).line - 1] + max((_coord(node).column or 1) - 1, 0)
            span = next(((a, b) for a, b, _ in spans if a <= offset < b), None)
            if span is None:
                continue
            remove = False
            if _candidate(node):
                name = _name(node)
                assert name is not None  # local holds only named declarations
                other = visible.get(name)
                if other is not None:
                    left, right = canonical[id(node)], canonical[id(other)]
                    remove = left == right or (
                        strict
                        and _compatible(left.replace("(", "@(", 1), right.replace("(", "@(", 1), typedefs, records)
                    )
                    if not remove and strict and _unknown_params(left) and _is_prototype(right):
                        remove = True  # a K&R `()` says nothing of its parameters: the prototype is the declaration
                    # landed, only the same type goes: a compatible other spelling (a prototype for a K&R `()`, a
                    # float parameter) can change how its callers pass arguments, so the source keeps its own
                    if not remove and strict:  # a landed source keeps what it builds with beside its headers
                        location = f"{_path(snapshot, other.coord)}:{_coord(other).line}"
                        findings.append(
                            Finding(
                                "headers.conflict",
                                path=unit.path,
                                line=_coord(node).line,
                                reason=f"{name}: local '{left}' vs {location} '{right}'",
                            )
                        )
                elif name in known and (
                    known[name][0] in vetoed
                    or (not strict and len(set(_spelled_pair(name, generator.visit(node) + ";", known[name][2]))) > 1)
                ):
                    held.append((span, node))  # a landed source keeps the types it was proved with
                elif name in known:
                    findings += _clash(
                        unit.path,
                        _coord(node).line,
                        name,
                        generator.visit(node) + ";",
                        known[name],
                        typedefs,
                        records=records,
                    )
                    findings += _body_clash(unit.path, node, name, known[name])
                    if _disagree(snapshot, known[name][0], chosen):
                        # its header contradicts one already taken: that one declares it, or the source keeps its own
                        if _declared_in(snapshot, chosen, name):
                            remove = True
                        else:
                            held.append((span, node))
                    else:
                        chosen.add(known[name][0])
                        includes.add(known[name][0].removeprefix("include/"))
                        remove = True
                else:
                    entry = functions.get(name)
                    text = " ".join(normal(generator.visit(node)).split())
                    differs = bool(
                        entry
                        and entry["evidence"] in ("authored", "landed")
                        and isinstance(node.type, c_ast.FuncDecl)
                        and _unnamed(_normal(_type_of(normal(entry["signature"]), name)))
                        != _unnamed(_normal(_type_of(text, name)))
                    )
                    if differs and not strict:
                        held.append((span, node))  # a landed source keeps the declaration it was proved with
                    elif differs and entry:
                        authority = entry["signature"].rstrip(";") + ";"
                        findings += _clash(
                            unit.path,
                            _coord(node).line,
                            name,
                            text,
                            ("types.toml", 0, authority),
                            typedefs,
                            records=records,
                        )
                        authoritative[span] = authority
                        includes.update(_needs(authority, known, name))
                    elif elsewhere.get(name, set()) - {_unnamed(_normal(_type_of(text, name)))}:
                        held.append((span, node))  # another landed source spells it otherwise: a header would break it
                    elif not strict and set(re.findall(r"\w+", text)) & own:
                        held.append((span, node))  # it names a type only this source defines
                    else:
                        moved.append((span, node))
                        remove = True
            grouped.setdefault(span, []).append((node, remove))
        for node in mine:  # a definition the compiler would see beside a prototype of another return type
            if strict and isinstance(node, c_ast.FuncDef) and node.decl.name in known:  # landed: it builds today
                findings += _clash(
                    unit.path,
                    _coord(node).line,
                    node.decl.name,
                    generator.visit(node.decl),
                    known[node.decl.name],
                    adopt=False,
                )
        owner: str | None = None
        writes: dict[str, bytes | None] = {}
        if moved:
            group = snapshot.layout.groups[unit.group]
            owner = group.header
            for span, node in (staying := _staying(snapshot, owner, moved, known, held)):
                grouped[span] = [(n, False if n is node else remove) for n, remove in grouped[span]]
                includes.update(_needs(generator.visit(node), known, _name(node)))
            moved = [item for item in moved if item not in staying]
        if moved and not strict:
            # landed: a declaration whose types the owner header does not reach today stays in the source, as does
            # one that names it: their headers would reach every unit that includes the owner and could contradict it
            assert owner is not None
            base = owner.removeprefix("include/")
            texts = {id(node): generator.visit(node) for _, node in moved}
            kept_here = [
                (span, node)
                for span, node in moved
                if any(
                    not reaches(snapshot, owner, f"include/{h}")
                    for h in _needs(texts[id(node)], known, _name(node)) - {base}
                )
            ]
            names = {_name(node) for _, node in kept_here}
            while more := [
                (span, node)
                for span, node in moved
                if (span, node) not in kept_here and set(re.findall(r"\w+", texts[id(node)])) & names
            ]:
                kept_here += more
                names |= {_name(node) for _, node in more}
            for span, node in kept_here:
                grouped[span] = [(n, False if n is node else remove) for n, remove in grouped[span]]
            moved = [item for item in moved if item not in kept_here]
        if moved:
            assert owner is not None  # set when anything moved
            includes.add(owner.removeprefix("include/"))
            _Read.paths.add(owner)
            header = (snapshot.peek(owner) or b"").decode()
            if not header:
                header = wrapped(owner, "")
            declarations = []
            for span, node in moved:
                entries = grouped[span]
                declarations.append(source[slice(*span)] if len(entries) == 1 else generator.visit(node) + ";")
            insertion = header.rfind("#endif")
            insertion = insertion if insertion >= 0 else len(header)
            header = header[:insertion].rstrip() + "\n\n" + "\n".join(declarations) + "\n\n" + header[insertion:]
            needs = [_needs(generator.visit(node), known) for _, node in moved]
            needed = set().union(*needs) - {owner.removeprefix("include/")}
            writes[owner] = included(header, needed).encode()
        for (start, end), entries in sorted(grouped.items(), reverse=True):
            if any(remove for _, remove in entries) or (start, end) in authoritative:
                replacement = authoritative.get(
                    (start, end), "\n".join(generator.visit(n) + ";" for n, remove in entries if not remove)
                )
                if not replacement:
                    first = source.rfind("\n", 0, start) + 1
                    last = source.find("\n", end)
                    last = len(source) if last < 0 else last + 1
                    if not source[first:start].strip() and not source[end:last].strip():
                        start, end = first, last
                source = source[:start] + replacement + source[end:]
        source = included(source, includes)
        writes[unit.path] = source.encode()
        return writes, tuple(findings)


def reconcile(
    snapshot: Snapshot, overlay: Snapshot, writes: dict[str, bytes | None]
) -> tuple[dict[str, bytes | None], list[Finding]]:
    """The writes with every name a written header declares once: the declaration in its owner stays, the others in
    written headers go and those include the owner. A name spelled with another type than the owner's, or one this
    cannot drop, is a finding naming both places. overlay is the snapshot with the writes applied."""
    edited = {p for p, d in writes.items() if configuration.is_header(p) and d is not None}
    known = types.authority(snapshot)
    authority = {n: _unnamed(_normal(_type_of(normal(r["signature"]), n))) for n, r in known["function"].items()}
    authority.update(
        {n: _unnamed(_normal(_type_of(normal(r["declaration"]), n))) for n, r in known.get("global", {}).items()}
    )
    paths = sources(overlay)
    rows = defaultdict(list)
    for path, scanned in zip(paths, _scanned(overlay, paths), strict=True):
        for name, line, text in scanned["declared"]:
            if (
                (spelled := _unnamed(_normal(_type_of(text, name))))
                and name not in _TYPE_WORDS
                and not text.startswith("typedef")
                and "{" not in text
            ):
                rows[name].append((path, line, spelled))
    drops: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    findings: list[Finding] = []
    for name, items in sorted(rows.items()):
        if len({p for p, _, _ in items}) < 2 or not edited & {p for p, _, _ in items}:
            continue
        spellings = {s for _, _, s in items}
        want = authority.get(name)
        if want not in spellings:
            want = next((s for p, _, s in items if p not in edited), None) or (
                next(iter(spellings)) if len(spellings) == 1 else None
            )
        if want is None:
            (a, al, asp), (b, bl, bsp) = items[0], next(i for i in items if i[2] != items[0][2])
            findings.append(
                Finding("headers.conflict", path=b, line=bl, reason=f"{name}: {a}:{al} '{asp}' vs {b}:{bl} '{bsp}'")
            )
            continue
        keeper, kline = min(((p, ln) for p, ln, s in items if s == want), key=lambda i: (i[0] in edited, i[0]))
        for path, line, spelled in items:
            clash = None if spelled == want else (f"{name}: {path}:{line} '{spelled}' vs {keeper}:{kline} '{want}'")
            if path == keeper or (spelled == want and path not in edited):
                continue
            if path in edited and not reaches(overlay, keeper, path):
                drops[path][name] = (clash, keeper)
            elif clash:
                findings.append(Finding("headers.conflict", path=path, line=line, reason=clash))
    out = dict(writes)
    for path, names in drops.items():
        text, owners = overlay.read(path).decode(), set()
        wanted = {keeper.removeprefix("include/") for _, keeper in names.values()}
        if breaks := include_breaks(overlay, path, wanted - set(_QUOTED_INCLUDE.findall(text))):
            # the include a drop needs would break a consumer: a copy that agrees stays, one that disagrees refuses
            reason = f"including the owner breaks {'; '.join(breaks[:3])}"
            findings += [
                Finding("headers.conflict", path=path, reason=f"{clash}: {reason}")
                for clash, _ in names.values()
                if clash
            ]
            continue
        for start, end, statement in reversed(list(statements(text))):
            found = _declared(statement)
            if len(found) == 1 and found[0][0] in names:
                text = text[:start] + text[end + (text[end : end + 1] == "\n") :]
                owners.add(names.pop(found[0][0])[1].removeprefix("include/"))
        findings += [Finding("headers.conflict", path=path, reason=clash) for clash, _ in names.values() if clash]
        out[path] = included(text, owners).encode()
    return out, findings


def split_atoms(text: str, members: Collection[str]) -> tuple[list[str], dict[str, str]]:
    """The include lines of a source and its blocks: one per member (its definition with the comment before it) and
    one per other file-scope item, keyed `#<offset>`. A preprocessor region that holds two members refuses."""
    spans: list[tuple[int, int, str]] = []
    depth, begin = 0, 0
    for match in re.finditer(_DIRECTIVE, text, re.M):
        word = match[0].strip()
        if re.match(r"#\s*if", word):
            begin, depth = (match.start() if depth == 0 else begin), depth + 1
        elif re.match(r"#\s*endif", word):
            depth -= 1
            if depth == 0:
                spans.append((begin, match.end(), "region"))
        elif depth == 0:
            spans.append((match.start(), match.end(), "include" if re.match(r"#\s*include\b", word) else ""))
    regions = [(a, b) for a, b, kind in spans if kind == "region"]
    spans += [(a, b, "") for a, b, _ in statements(text, bodies=True) if not any(x <= a < y for x, y in regions)]
    includes: list[str] = []
    blocks: dict[str, str] = {}
    cursor = 0
    for start, end, kind in sorted(spans):
        if end <= cursor:  # a directive region inside a function body is part of that function's block
            continue
        body = text[start:end]
        if kind == "include":
            includes.append(body.strip())
        else:
            raw = body.encode()
            found = {n for n in (*defined(raw), *variables(raw)) if n in members}
            if len(found) > 1 and kind == "region":
                names = ", ".join(sorted(found))
                raise Refusal(Finding("move.directive", "a preprocessor region holds several members", unit=names))
            key = min(found) if found else f"#{start}"
            blocks[key] = (blocks[key] + "\n\n" if key in blocks else "") + text[cursor:end].strip("\n")
        cursor = end
    return includes, blocks


def _incpath(line: str) -> str:
    found = re.search(r'["<]([^">]+)', line)
    return found[1] if found else line


def include_order(lines: Iterable[str], own: str = "") -> list[str]:
    """Include lines with the own header first, the rest sorted by path, each path once."""
    return sorted({_incpath(i): i.strip() for i in lines}.values(), key=lambda i: (_incpath(i) != own, _incpath(i)))


def ordered(text: str, own: str) -> str:
    """The text with its one run of include lines in include_order; a text with scattered includes is left whole."""
    found = list(re.finditer(r"^[ \t]*#\s*include[^\n]*\n", text, re.M))
    if not found or any(text[a.end() : b.start()].strip() for a, b in pairwise(found)):
        return text
    block = "".join(f"{line}\n" for line in include_order((m[0] for m in found), own))
    return text[: found[0].start()] + block + text[found[-1].end() :]


def render(includes: Sequence[str], preamble: Sequence[str], blocks: Sequence[str]) -> bytes:
    parts = ["\n".join(includes), *preamble, *blocks]
    return ("\n\n".join(p for p in parts if p) + "\n").encode()


def _rooted(snapshot: Snapshot, header: str, name: str) -> str:
    """An include of `header` spelled from include/: a sibling it names by its own folder is spelled from the root."""
    sibling = posixpath.normpath(posixpath.join(posixpath.dirname(header), name))
    if sibling != f"include/{name}" and sibling.startswith("include/") and snapshot.peek(sibling) is not None:
        return sibling.removeprefix("include/")
    return name


def relocate(
    snapshot: Snapshot,
    events: Iterable[tuple[str, str, str, frozenset[str]]],
    failed: list[tuple[str, str, tuple[Finding, ...]]] | None = None,
) -> tuple[dict[str, str | None], dict[str, str], list[tuple[str, str, frozenset[str]]]]:
    """Header texts after the events in order, the old header -> new header spelling of every include, and the
    declarations that moved out of headers that stay. With `failed`, an event whose merge refuses is left out and
    recorded there as (old, new, findings), and the other events go on; without it the first refusal raises."""
    texts: dict[str, str | None] = {}
    moved: dict[str, str] = {}
    stay: list[tuple[str, str, frozenset[str]]] = []

    reached: dict[str, set[str]] = {}  # per header every header it includes, kept across the events of this call

    def reach(start: str, targets: Collection[str]) -> bool:
        """Whether the header at start includes, directly or not, any target; an include of a header an earlier event
        merged away is an include of the header it went to."""
        if start not in reached:
            seen: set[str] = set()
            todo = [start]
            while todo:
                here = todo.pop()
                here = here if staying(here) else moved.get(here, here)
                if here not in seen:
                    seen.add(here)
                    for name in _quoted((peek(here) or "").encode()):
                        todo += _names(name, here)
            reached[start] = seen
        return not reached[start].isdisjoint(targets)

    def forget(paths: Collection[str]) -> None:
        """Drop what a rewritten header changes: its own walk and every walk that passed through it."""
        for start in [k for k, seen in reached.items() if k in paths or not seen.isdisjoint(paths)]:
            del reached[start]

    known: list[Any] = []  # the type map, read at the first declaration two merged texts spell differently

    def authority(name: str) -> str | None:
        if not known:
            try:
                known.append(types.authority(snapshot))
            except (Refusal, AttributeError):  # no type map: nothing decides
                known.append({"function": {}, "global": {}})
        chosen = _authority(name, {}, known[0])
        return chosen[0] if chosen else None

    # the parts each target header gathers from its events; a target is merged once, when an event reads it or at the
    # end, so a header that many events land in is merged once with all of them, not once per event
    pending: dict[str, list[tuple[str, str]]] = {}
    held_by: defaultdict[str, set[str]] = defaultdict(set)  # per staying header the names that left it
    # per target the headers its events took text from: (old, old's text before, old's text after, names held)
    taken_from: defaultdict[str, list[tuple[str, str | None, str | None, frozenset[str]]]] = defaultdict(list)

    def peek(path: str) -> str | None:
        """A header's text for include walks: a pending target is its parts together (a shared header only the parts
        it can take: the others go back to their headers when it is merged)."""
        if path in pending:
            parts = pending[path]
            if path in base:
                key = (path, len(parts), len(moved))
                if key not in leaf_memo:
                    leaf_memo[key] = leafless(path, parts)
                parts = [p for i, p in enumerate(parts) if i not in leaf_memo[key]]
            return "\n".join(t for _, t in parts)
        if texts.get(path, "") is None and staying(path):  # its part goes back: it reads as that part
            return next(t for o, t in pending[moved[path]] if o == path)
        return texts[path] if path in texts else _peek(snapshot, path)

    # the shared headers (repo.toml [layout]) are the base every other header builds on: they include only each
    # other, so a header that includes any other one never merges into them (it would put that header, and every
    # header it reaches, under everything that includes the base, in a cycle with its own includers)

    config = getattr(snapshot, "config", None)  # a walk over bare header texts reads the tool's own rows
    rows = configuration.layout_rows(config.project) if config else configuration.load_resource("repo.toml")["layout"]
    base = frozenset(rows["shared_headers"])
    leaf_memo: dict[tuple[str, int, int], set[int]] = {}
    # a names part (origin "") by the identity of its pending tuple: the header it came from and the names it took
    named: dict[int, tuple[str, frozenset[str]]] = {}

    def staying(old: str) -> bool:
        """Whether a header moved into a shared header still gathering is a part that header cannot take: it goes back
        when the shared header is merged, so an include walk reads its own text, not the shared header's."""
        new = moved.get(old)
        if new is None or new not in base or new not in pending:
            return False
        parts = pending[new]
        key = (new, len(parts), len(moved))
        if key not in leaf_memo:
            leaf_memo[key] = leafless(new, parts)
        return any(parts[i][0] == old for i in leaf_memo[key])

    def exists(path: str) -> bool:
        return path in pending or (texts[path] is not None if path in texts else _peek(snapshot, path) is not None)

    def lands(name: str, here: str, back: Collection[str] = ()) -> str | None:
        """The header an include names now: the file it spells, followed through the moves (a header in `back` goes
        back to its own text, so it is itself); None when it names no file."""
        for found in sorted(_names(name, here)):
            seen: set[str] = set()
            while found in moved and found not in back and found not in seen:
                seen.add(found)
                found = moved[found]
            if found in back or exists(found):
                return found
        return None

    def leafless(path: str, parts: Sequence[tuple[str, str]]) -> set[int]:
        """The parts, file or names, a shared header cannot take: one that includes a header which is not shared (or
        that a part it cannot take is), to a fixed point."""
        out: set[int] = set()
        grew = True
        while grew:
            grew = False
            back = {parts[i][0] or named[id(parts[i])][0] for i in out}
            for i, part in enumerate(parts):
                origin, text = part
                if not origin and id(part) in named:
                    origin = named[id(part)][0]
                if i in out or not origin or origin == path:
                    continue
                for match in _INCLUDE.finditer(_unguarded(text)):
                    at = lands(match[1], origin, back)
                    if at is not None and at != path and at not in base:
                        out.add(i)
                        grew = True
                        break
        return out

    def route(start: str, path: str, origins: Collection[str]) -> set[str]:
        """The merged-in files a header reaches without passing through the merged header: what makes it include the
        merged header once their includes are spelled to it."""
        found: set[str] = set()
        seen: set[str] = set()
        todo = [start]
        while todo:
            here = todo.pop()
            hops: set[str] = set()
            while here not in origins and here in moved and here not in hops and not staying(here):
                hops.add(here)
                here = moved[here]
            if here in origins:
                found.add(here)
                continue
            if here == path or here in seen:
                continue
            seen.add(here)
            for name in _quoted((peek(here) or "").encode()):
                todo += _names(name, here)
        return found

    def closing(path: str, parts: Sequence[tuple[str, str]]) -> dict[int, Finding]:
        """The file parts whose merge makes an include cycle that no include order survives. An outside header the
        merged header includes and that reaches a merged-in file goes after the body (see _merge), which holds only
        while no other such header needs it: entered first, it reaches the merged header, whose later back-includes
        then run before its own body."""
        merged = {path, *(o for o, _ in parts if o)}
        back: list[str] = []
        for origin, text in parts:
            for match in _INCLUDE.finditer(_unguarded(text)):
                for outside in sorted(_names(match[1], origin or path) - merged):
                    if outside not in back and reach(outside, merged):
                        back.append(outside)
        origins = {o for o, _ in parts if o and o != path}
        out: dict[int, Finding] = {}
        for outside in back:
            needs = next((x for x in back if x != outside and reach(x, {outside})), None)
            why = f"{needs} needs {outside}"
            if needs is None:
                # one that reaches the merged files only through another header: entered from that header first,
                # the merged header runs before its body, and its back-include then reads it unfinished
                via = [
                    c
                    for name in _quoted((peek(outside) or "").encode())
                    for c in sorted(_names(name, outside))
                    if exists(c) or c in moved
                ]
                if via and not any(current(c) in merged for c in via):
                    # it reads that header unfinished only when it uses a name one on the way declares
                    reach(outside, merged)
                    used = set(re.findall(r"[A-Za-z_]\w*", _blank(peek(outside) or "")))
                    needs = next(
                        (
                            c
                            for c in sorted(reached[outside] - merged - {outside})
                            if reach(c, merged) and used & names(peek(c) or "")
                        ),
                        None,
                    )
                    why = f"{outside} reaches it only through {needs}"
            if needs is None:
                continue
            # the parts it reaches go back; when it reaches only the header itself, the parts that include it do
            closers = route(outside, path, origins) or {
                o
                for o, text in parts
                if o in origins
                and any(
                    reach(c, {outside})
                    for match in _INCLUDE.finditer(_unguarded(text))
                    for c in _names(match[1], o) - merged
                )
            }
            for old in sorted(closers):
                i = next(i for i, (o, _) in enumerate(parts) if o == old)
                out.setdefault(
                    i,
                    Finding(
                        "move.cycle",
                        f"{outside} includes {old} and {why}: merged into {path}, {old} would put "
                        f"{path} in an include cycle that no include order survives",
                        path=old,
                    ),
                )
        return out

    def current(path: str) -> str:
        """Where a header's text is now: an include of one an earlier event moved names the one it went to."""
        seen: set[str] = set()
        while path in moved and path not in seen and not staying(path):
            seen.add(path)
            path = moved[path]
        return path

    def read(path: str) -> str | None:
        """A back-included header's text for the types a merged body pulls from it: the header it moved to, when it
        moved (its own path reads nothing then, and the types it gave are never pulled)."""
        return peek(current(path))

    def give_back(path: str, old: str, finding: Finding) -> None:
        """A file part a merge cannot take goes back to its own header: its includes keep naming it."""
        if failed is None:
            raise Refusal(finding)
        entry = next((e for e in taken_from[path] if e[0] == old and e[2] is None), None)
        if entry is not None:
            taken_from[path].remove(entry)
            texts[old] = entry[1]
        if moved.get(old) == path:
            del moved[old]
        forget({old, path})
        failed.append((old, path, (finding,)))

    def give_names_back(path: str, old: str, held: frozenset[str], text: str, finding: Finding) -> None:
        """A names part a merge cannot take goes back to the header it left: that header gets its declarations
        again (its text before the event when no later event changed it, else the part's body after its text), and
        no include of those names is respelled to the merged header."""
        if failed is None:
            raise Refusal(finding)
        entry = next((e for e in taken_from[path] if e[0] == old and e[2] is not None and e[3] == held), None)
        if entry is None:
            entry = next((e for e in taken_from[path] if e[0] == old and e[2] is None and e[3] == held), None)
        if entry is not None:
            taken_from[path].remove(entry)
            if texts.get(old, "") is entry[2]:
                texts[old] = entry[1]
            else:  # a later event changed the header: the declarations go back after what it has now
                own = f'#include "{old.removeprefix("include/")}"'
                body = "\n".join(line for line in text.splitlines() if line.strip() != own)
                now = texts.get(old, _peek(snapshot, old))
                texts[old] = wrapped(old, body) if now is None else _appended(now, body)
        if moved.get(old) == path:
            del moved[old]
        if (old, path, held) in stay:
            stay.remove((old, path, held))
            held_by[old] -= held
        forget({old, path})
        failed.append((old, path, (finding,)))

    def get(path: str) -> str | None:
        if path in pending:
            flush(path)
        return texts[path] if path in texts else _peek(snapshot, path)

    def flush(path: str) -> None:
        parts = pending.pop(path)
        while True:  # the parts it cannot take go back first, each refused alone; the rest merge
            if path in base and (left := leafless(path, parts)):
                drop = {}
                for i in left:
                    source = parts[i][0] or named[id(parts[i])][0]
                    drop[i] = Finding(
                        "move.cycle",
                        f"{source} includes a header that is not shared: {path} is a shared header, it takes no part "
                        "that includes another",
                        path=source,
                    )
            else:
                drop = closing(path, parts)
            if not drop:
                break
            for i, finding in sorted(drop.items()):
                if not parts[i][0] and id(parts[i]) in named:
                    old, held = named[id(parts[i])]
                    give_names_back(path, old, held, parts[i][1], finding)
                else:
                    give_back(path, parts[i][0], finding)
            parts = [p for i, p in enumerate(parts) if i not in drop]
        if all(o == path for o, _ in parts):  # every part went back: the header keeps its own text, or is never made
            if parts:
                texts[path] = parts[0][1]
            forget({path})
            return
        try:
            text, rewrites = _merge(path, [t for _, t in parts], [o for o, _ in parts], reach, read, authority)
        except Refusal as refusal:
            if failed is None:
                raise
            failed.append((next((o for o, _ in parts if o and o != path), ""), path, refusal.findings))
            texts[path] = parts[0][1] if parts[0][0] == path else _peek(snapshot, path)
            # a target that is not made gives back what its events took: no include is respelled to a header
            # that never exists, and the headers it would have replaced keep their text
            for old, before, after, held in taken_from.pop(path, ()):
                if texts.get(old, "") is not after:
                    continue  # a later event rewrote it: that event's text stands
                texts[old] = before
                if moved.get(old) == path:
                    del moved[old]
                if (old, path, held) in stay:
                    stay.remove((old, path, held))
                    held_by[old] -= held
                forget({old})
            forget({path})
            return
        texts[path] = text
        for outside, rest in ((current(o), r) for o, r in rewrites.items()):
            if outside in pending or texts.get(outside, "") is None:
                continue  # a header merged away or still gathering keeps what it has
            texts[outside] = _owned(rest, held_by[outside])[1] if outside in held_by else rest
        forget({path, *rewrites, *(current(o) for o in rewrites)})

    def gather(new: str, origin: str, text: str, source: tuple[str, frozenset[str]] | None = None) -> None:
        if new not in pending:
            pending[new] = [(new, t) for t in (get(new),) if t]
        part = (origin, text)
        pending[new].append(part)
        if source is not None:
            named[id(part)] = source
        forget({new})

    def step(kind: str, old: str, new: str, held: frozenset[str]) -> None:
        if old == new or (text := get(old)) is None:
            return
        if kind == "file":
            gather(new, old, _spelled(snapshot, old, text))
            texts[old] = None
            moved[old] = new
            taken_from[new].append((old, text, None, held))
            forget({old, new})
            return
        taken, rest = _owned(text, held)
        if not taken:
            return
        if not _empty(rest) and names(rest) & set(re.findall(r"\w+", taken)):
            taken = f'#include "{old.removeprefix("include/")}"\n{taken}'
        # the new header stands alone: it includes what the old one did, where the types its declarations name live
        carried = [_rooted(snapshot, old, m[1]) for m in _INCLUDE.finditer(_unguarded(text))]
        taken = "".join(f'#include "{p}"\n' for p in carried if p != new.removeprefix("include/")) + taken
        gather(new, "", taken, (old, held))
        if _empty(rest):
            texts[old], moved[old] = None, new
        else:
            texts[old] = rest
            held_by[old] |= held
            stay.append((old, new, held))
        taken_from[new].append((old, text, texts[old], held))
        forget({old, new})

    for kind, old, new, held in events:
        try:
            step(kind, old, new, held)
        except Refusal as refusal:
            if failed is None:
                raise
            failed.append((old, new, refusal.findings))
    for path in list(pending):
        if path in pending:
            flush(path)
    for old in moved:  # to the last header of a chain of moves, however long
        seen = {old}
        while moved[old] in moved and moved[old] not in seen:
            seen.add(moved[old])
            moved[old] = moved[moved[old]]
    # a header a later event vacated is no include target: the declarations that left a staying header live where
    # it went (a member move into a new module, then that module's header move)
    stay = [(old, moved.get(new, new), held) for old, new, held in stay if moved.get(new, new) != old]
    return texts, moved, stay


def _peek(snapshot: Snapshot, path: str) -> str | None:
    content = snapshot.peek(path)
    return None if content is None else content.decode(errors="replace")


def _typedefs(text: str) -> list[tuple[int, int, str]]:
    """(start, end, spelling) of every typedef the text makes at file scope without a body."""
    rows = []
    for start, end, statement in statements(text):
        spelled = " ".join(_blank(statement).split())
        if re.match(r"typedef\b", spelled) and "{" not in spelled:
            rows.append((start, end, spelled))
    return rows


@lru_cache(maxsize=8192)
def _typedef_spellings(text: str) -> frozenset[str]:
    """The typedef spellings one included file makes, parsed once per text: a respell walks the same headers for
    every file it dedupes (859 files read 111,879 headers, 212 s, in the 2026-10-15 fakes move)."""
    return frozenset(spelled for _, _, spelled in _typedefs(text))


def _inherited(
    read: Callable[[str], str | None], path: str, text: str, edges: dict[str, list[tuple[str, str]]] | None = None
) -> set[str]:
    """The typedef spellings of every file `text` includes by a quoted name, directly or through others. `edges`
    keeps each included file's resolved includes across the calls of one respell: a header met in hundreds of
    closures is resolved and parsed once."""
    found: set[str] = set()
    visited = {path}
    pending = [(path, text)]
    edges = {} if edges is None else edges
    while pending:
        here, body = pending.pop()
        if here == path or here not in edges:
            resolved = []
            for name in _quoted(body.encode()):
                for base in (posixpath.dirname(here), "include", "src"):
                    target = posixpath.normpath(posixpath.join(base, name))
                    if (content := read(target)) is not None:
                        resolved.append((target, content))
            if here != path:
                edges[here] = resolved
        else:
            resolved = edges[here]
        for target, content in resolved:
            if target not in visited:
                visited.add(target)
                found.update(_typedef_spellings(content))
                pending.append((target, content))
    return found


def _dedupe(text: str, inherited: Collection[str] = ()) -> str:
    """The text with each file-scope typedef kept once: a repeat of an earlier one, or of one an included file already
    makes, is dropped (what a many-to-one rename leaves behind: old gcc 2 rejects a repeated typedef). A repeat after a
    preprocessor conditional line stays: it may sit in the other branch."""
    seen: dict[str, int] = dict.fromkeys(inherited, -1)
    found = list(re.finditer(r"(?m)^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b", text))
    guard = re.match(
        r"\s*(?:/\*.*?\*/\s*|//[^\n]*\n\s*)*#[ \t]*ifndef[ \t]+(\w+)\s*\n[ \t]*#[ \t]*define[ \t]+(\w+)", text, re.S
    )
    if guard and guard[1] == guard[2] and found and found[-1][1] == "endif":  # the include guard is no branch
        found = found[1:-1]
    conditionals = [m.start() for m in found]
    drop = []
    for start, end, spelled in _typedefs(text):
        earlier = seen.get(spelled)
        if earlier is not None and not any(earlier < c < start for c in conditionals):
            drop.append((start, end))
        else:
            seen[spelled] = start
    for start, end in reversed(drop):
        stop = end + 1 if text[end : end + 1] == "\n" else end
        text = text[:start] + text[stop:]
    return text


def respell(
    snapshot: Snapshot,
    files: Mapping[str, str | None],
    moved: Mapping[str, str],
    stay: Sequence[tuple[str, str, frozenset[str]]],
    renames: Mapping[str, str],
    folded: Collection[str] = (),
) -> dict[str, bytes | None]:
    """Every tracked source and header with its includes and symbol spellings brought up to date. A `folded` record
    tag loses its definition (its body, and its types.toml row) before the names are spelled: its name then names the
    record it folds into."""
    scan = bool(moved or stay or renames)  # nothing to spell again: only the files the moves wrote
    paths = {
        *files,
        *(sources(snapshot, "src", ".c") if scan else ()),
        *(sources(snapshot) if scan else ()),
        *(["types.toml"] if scan and snapshot.peek("types.toml") else []),
    }
    writes: dict[str, bytes | None] = {}
    texts: dict[str, str] = {}
    deduped: list[str] = []
    targets = set(renames.values())
    for path in sorted(paths):
        if path in files and files[path] is None:
            writes[path] = None
            continue
        known = files.get(path)
        current = known if known is not None else (snapshot.peek(path) or b"").decode(errors="replace")
        text = rewrite_includes(current, moved, path) if configuration.is_text(path) else current
        words = set(re.findall(r"\w+", text))
        for old, new, names in stay:
            if path != new and names & words and rewrite_includes(text, {old: ""}, path) != text:
                text = included(text, [new.removeprefix("include/")])
        for tag in sorted(set(folded) & words):
            text = drop_record(text, tag) if configuration.is_text(path) else _drop_row(text, tag)
        spelled = sorted(symbols.tokens(text) & renames.keys())
        for old in spelled:
            text = symbols.rewrite(text, old, renames[old])
        texts[path] = text
        # several names may now be one: each typedef stays once per file, also where an included file now repeats it
        if (spelled or words & targets) and configuration.is_text(path):
            deduped.append(path)
        if path in files or text != current:
            writes[path] = text.encode()

    read_ = cache(lambda p: texts[p] if p in texts else files[p] if p in files else _peek(snapshot, p))
    edges: dict[str, list[tuple[str, str]]] = {}  # an included file's includes, resolved once per respell

    def read(p: str) -> str | None:  # an included file as this respell spells it, read once
        return read_(p)

    for path in deduped:
        text = _dedupe(texts[path], _inherited(read, path, texts[path], edges))
        if path in files or text != texts[path] or path in writes:
            writes[path] = text.encode()
    return writes


def type_names(snapshot: Snapshot) -> tuple[dict[str, str], dict[str, str]]:
    """(per struct, union or enum tag its defining header statement, per typedef name its statement) of every type
    the headers declare."""
    records: dict[str, str] = {}
    typedefs: dict[str, str] = {}
    for key, (_, _, statement) in catalog(snapshot).items():
        kind, _, tag = key.partition(" ")
        if tag and kind in ("struct", "union", "enum"):
            records.setdefault(tag, statement)
        elif re.match(r"typedef\b", statement):
            typedefs.setdefault(key, statement)
    return records, typedefs


def source_types(snapshot: Snapshot, name: str) -> tuple[dict[str, str], dict[str, str]]:
    """type_names of the landed sources that spell `name`, as their own file-scope declarations make them."""
    records: dict[str, str] = {}
    typedefs: dict[str, str] = {}
    for path, _, scanned in _listed(snapshot, "src", ".c"):
        if name not in scanned["words"]:
            continue
        for key, _, statement in _declared(snapshot.read(path).decode(errors="replace")):
            kind, _, tag = key.partition(" ")
            if tag and kind in ("struct", "union", "enum"):
                records.setdefault(tag, statement)
            elif re.match(r"typedef\b", statement):
                typedefs.setdefault(key, statement)
    return records, typedefs


def _spellers(snapshot: Snapshot, place: str, suffix: str) -> dict[str, list[str]]:
    """Per word the `sources` of a place that spell it, in path order: one index per listing of the place (its pins),
    so a move's thousands of name lookups never walk every scan again (3,257 walks of 5,000 scans, 17 s a round)."""
    key = landed_sources(snapshot, place, suffix)[1]

    def produce() -> dict[str, list[str]]:
        index: defaultdict[str, list[str]] = defaultdict(list)
        for path, _, scanned in _listed(snapshot, place, suffix):
            for word in scanned["words"]:
                index[word].append(path)
        return dict(index)

    return effort.memo(("spellers", str(snapshot.config.project.root), place, suffix, key), produce)


def spelled_in(snapshot: Snapshot, name: str) -> tuple[list[str], list[str]]:
    """(the landed sources, the headers) whose text spells `name`, from the scans the pins keep."""
    return (
        list(_spellers(snapshot, "src", ".c").get(name, ())),
        list(_spellers(snapshot, "include", ".h").get(name, ())),
    )


def _literal(match: re.Match[str]) -> str:
    return str(int(match[0], 16 if match[0][:2].lower() == "0x" else 10))


def members(statement: str) -> list[tuple[str, str]] | None:
    """(name, normal type) of each member of the first record body the statement defines, in order; a nested body
    is one member with no name. None when the statement defines no body."""
    start = statement.find("{")
    if start < 0:
        return None
    depth, end = 0, len(statement)
    for i in range(start, len(statement)):
        depth += (statement[i] == "{") - (statement[i] == "}")
        if depth == 0:
            end = i
            break
    body, rows, depth, piece = statement[start + 1 : end], [], 0, ""
    for char in body:
        depth += (char == "{") - (char == "}")
        if char == ";" and not depth:
            rows.append(piece.strip())
            piece = ""
        else:
            piece += char
    out: list[tuple[str, str]] = []
    for row in filter(None, rows):
        if "{" in row:
            out.append(("", _normal(row)))
            continue
        for part in _parts(row):
            found = _DECLARATOR.search(part)
            name = (found[1] or found[2]) if found else ""
            spelled = _type_of(row, name) if name else part
            out.append((name, re.sub(r"\b0[xX][0-9a-fA-F]+\b|\b\d+\b", _literal, _normal(spelled)).replace(" ", "")))
    return out


def drop_record(text: str, tag: str) -> str:
    """The text without the body of record `tag`: a definition alone goes; a typedef that defines it keeps naming it."""
    for start, end, statement in reversed(list(statements(text))):
        found = re.search(rf"\b(?:struct|union)\s+{re.escape(tag)}\s*\{{", statement)
        if not found:
            continue
        if re.match(r"\s*typedef\b", statement):
            open_at = start + found.end() - 1
            depth, close = 0, open_at
            for i in range(open_at, end):
                depth += (text[i] == "{") - (text[i] == "}")
                if depth == 0:
                    close = i
                    break
            text = text[:open_at].rstrip() + " " + text[close + 1 :].lstrip()
        else:
            stop = end + 1 if text[end : end + 1] == "\n" else end
            text = text[:start] + text[stop:]
    return text


def _drop_row(text: str, name: str) -> str:
    """A types.toml text without the row a name keys."""
    return re.sub(rf'(?m)^[ \t]*"?{re.escape(name)}"?[ \t]*=.*\n?', "", text)


def _spelling(text: str, name: str) -> str:
    return _unnamed(_normal(_type_of(text, name)))


def _authority(
    name: str, defs: Mapping[str, tuple[str, str]], known: Mapping[str, Any]
) -> tuple[str, str | None] | None:
    """(the one prototype statement of a name, the path of its definition): the definition's own signature when a
    source defines it, else the type DB's entry. None when nothing says what the name is."""
    if name in defs:
        path, text = defs[name]
        text = " ".join(text.replace("@", name, 1).split())
        return ("extern " + text + ";", path) if not re.match(r"(?:static|inline)\b", text) else None
    entry = known["function"].get(name)
    text = (entry["signature"] if entry else (known.get("global", {}).get(name) or {}).get("declaration")) or ""
    text = re.sub(r"^\s*extern\s+|\s*;\s*$", "", text)
    return ("extern " + text + ";", None) if text and not text.startswith("static") else None


def _type_text(statement: str, name: str, known: Mapping[str, Any]) -> str:
    """The prototype's text with its own name and its parameter names left out: only the types it spells."""
    head, _, rest = statement.removeprefix("extern").partition("(")
    kept = [head.replace(name, " ", 1)]
    for param in _parts(rest.rpartition(")")[0]):
        named = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*$", param)
        if (
            named
            and param[: named.start()].strip()
            and named[1] not in _TYPE_WORDS
            and f"struct {named[1]}" not in known
        ):
            param = param[: named.start()]
        kept.append(param)
    return re.sub(r"\b(?:const|volatile|unsigned|signed|int|char|short|long|void|float|double)\b", " ", " ".join(kept))


def _nameable(statement: str, name: str, known: Mapping[str, Any]) -> bool:
    """Whether every type a prototype spells can be named by a header: a keyword, or a type a header declares. A type
    only a source declares (its own record) cannot be."""
    tags = {w.rpartition(" ")[2] for w in known if " " in w}
    return all(
        w in _TYPE_WORDS or w in known or w in tags
        for w in re.findall(r"[A-Za-z_]\w*", _type_text(statement, name, known))
    )


def prototypes(
    snapshot: Snapshot, skip: Collection[str] = ()
) -> tuple[dict[str, tuple[bool, frozenset[str]]], dict[str, bytes], list[Finding]]:
    """One prototype per name. Where a source defines a name its signature is the prototype, else the type DB's entry.
    It lives in the header that already declares the name, else in the header of the defining unit. Every extern of
    the name in a source goes and the source includes that header; a header spelling that differs is rewritten.
    Returns ({name: (typed, paths written)}, writes, findings): `typed` marks a change of the spelling some file
    reads, which changes code and so needs a proof, not just a compile; names in `skip` are left alone. A name whose
    prototype needs a type no header can name, or a header that would have to include its own includer, stays as it
    is. A name nothing types (no definition, no type DB entry) takes the one spelling its declarations agree on, in
    the header of the unit that holds it; declarations that disagree are a finding (headers.prototype_conflict)."""
    with effort.stage("headers.prototypes"):
        c_paths, h_paths = sources(snapshot, "src", ".c"), sources(snapshot)
        scans = dict(zip(c_paths, _scanned(snapshot, c_paths), strict=True))
        hscans = dict(zip(h_paths, _scanned(snapshot, h_paths), strict=True))
        defs: dict[str, tuple[str, str]] = {}
        for path, scanned in scans.items():
            if not path.startswith("src/fuzzy/"):  # a fuzzy candidate is not landed
                defs.update({n: (path, t) for n, _, t in scanned["functions"]})
                defs.update({n: (path, t) for n, _, t in scanned["defs"] if n not in defs})
        known = types.load(snapshot)
        rows: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        for path, scanned in scans.items():
            for name, line, text in scanned["declared"]:
                rows[name].append((path, line, text))
        homes: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        for path, scanned in hscans.items():
            for name, line, text in scanned["declared"]:
                if not text.startswith("typedef") and "{" not in text and _spelling(text, name):
                    homes[name].append((path, line, text))
        catalogued = catalog(snapshot)
        texts: dict[str, str] = {}
        spans: dict[str, dict[int, tuple[int, int, str]]] = {}

        def statement_at(path: str, line: int, name: str) -> tuple[int, int, str] | None:
            if path not in spans:
                texts[path] = snapshot.read(path).decode()
                found = {texts[path].count("\n", 0, a) + 1: (a, b, c) for a, b, c in statements(texts[path])}
                for m in _EXTERN.finditer(_blank(texts[path])):  # an extern inside a function body is one too
                    a = m.start() + len(m[0]) - len(m[0].lstrip())
                    found.setdefault(texts[path].count("\n", 0, a) + 1, (a, m.end(), " ".join(m[0].split())))
                spans[path] = found
            hit = spans[path].get(line)
            return hit if hit and name in {n for n, _, _ in _declared(hit[2])} else None

        items: dict[str, tuple[bool, frozenset[str]]] = {}
        conflicts: list[Finding] = []
        replaced: dict[str, dict[tuple[int, int], str]] = defaultdict(dict)  # a header statement made the prototype
        dropped: dict[str, dict[tuple[int, int, str], set[str]]] = defaultdict(lambda: defaultdict(set))
        needs: dict[str, set[str]] = defaultdict(set)
        appended: dict[str, list[str]] = defaultdict(list)
        for name in sorted(rows):
            if name in skip or name in _TYPE_WORDS:
                continue
            found = _authority(name, defs, known) or _agreed(name, homes.get(name, []), rows[name], conflicts)
            if found is None:
                continue
            statement, owner_path = found
            want = _spelling(statement, name)
            if not want:
                continue
            homed = homes.get(name, [])
            owner = next((p for p, _, t in homed if _spelling(t, name) == want), homed[0][0] if homed else None)
            fix: dict[str, tuple[int, int]] = {}
            extra: dict[str, set[str]] = defaultdict(set)
            add: list[tuple[str, str]] = []
            if owner is None:  # nothing declares it yet: the defining unit's header does
                unit = snapshot.layout.units.get(owner_path or "") or next(  # else the unit that holds it
                    (u for u in snapshot.layout.units.values() if name in getattr(u, "members", ())), None
                )
                group = snapshot.layout.groups.get(unit.group) if unit else None
                if group is None:
                    continue
                owner = group.header
                add.append((owner, statement))
            for path, line, text in homed:
                if _spelling(text, name) != want:
                    hit = statement_at(path, line, name)
                    if hit is None or len(_declared(hit[2])) != 1:
                        break
                    fix[path] = hit[:2]
            else:
                if add or fix:  # a header that takes the prototype in must be able to name its types
                    if not _nameable(statement, name, catalogued):
                        continue
                    text = _type_text(statement, name, catalogued)
                    required = _needs(text, catalogued) - {owner.removeprefix("include/")}
                    if any(reaches(snapshot, f"include/{n}", owner) for n in required):
                        continue
                    extra[owner].update(required)
                cut: list[tuple[str, tuple[int, int, str]]] = []
                for path, line, _ in rows[name]:
                    if hit := statement_at(path, line, name):
                        cut.append((path, hit))
                if cut or add or fix:
                    typed = bool(fix) or any(_spelling(t, name) != want for _, _, t in rows[name])
                    for path, hit in cut:
                        dropped[path][hit].add(name)
                        extra[path].add(owner.removeprefix("include/"))
                    for path, span in fix.items():
                        replaced[path][span] = statement
                    for path, text in add:
                        appended[path].append(text)
                    for path, names_ in extra.items():
                        needs[path].update(names_)
                    items[name] = (typed, frozenset({*(p for p, _ in cut), *fix, *extra, *(p for p, _ in add)}))
        writes: dict[str, bytes] = {}
        for path in sorted({*replaced, *dropped, *needs, *appended}):
            was = snapshot.peek(path)
            text = was.decode() if was is not None else wrapped(path, "")
            if path not in texts and was is not None:
                texts[path] = text
            edits = [(a, b, s) for (a, b), s in replaced.get(path, {}).items()]
            for (a, b, clean), gone in dropped.get(path, {}).items():
                left = [n for n, _, _ in _declared(clean) if n not in gone]
                edits.append((a, b, "\n".join("extern " + _type_of(clean, n).replace("@", n, 1) + ";" for n in left)))
            for start, end, replacement in sorted(edits, reverse=True):
                a, b = text.rfind("\n", 0, start) + 1, text.find("\n", end)
                b = len(text) if b < 0 else b + 1
                if replacement or text[a:start].strip() or text[end:b].strip():
                    a, b = start, end
                text = text[:a] + replacement + text[b:]
            if appended.get(path):
                cut_at = text.rfind("#endif")
                cut_at = cut_at if cut_at >= 0 else len(text)
                text = text[:cut_at].rstrip() + "\n\n" + "\n".join(appended[path]) + "\n\n" + text[cut_at:]
            text = included(text, needs.get(path, ()))
            if was is None or text != was.decode():
                writes[path] = text.encode()
        return items, writes, conflicts


def _agreed(
    name: str, homed: Sequence[tuple[str, int, str]], local: Sequence[tuple[str, int, str]], conflicts: list[Finding]
) -> tuple[str, None] | None:
    """The one prototype the declarations of a name nothing else types agree on: a header's or the local externs'
    (a K&R `()` says nothing of its parameters). Function declarations that disagree are a finding naming every
    spelling and where it is, never a pick."""
    seen: dict[str, tuple[str, set[str]]] = {}
    for path, _, text in (*homed, *local):
        seen.setdefault(_spelling(text, name), (text, set()))[1].add(path)
    typed = {k: v for k, v in seen.items() if not _unknown_params(k)} or seen
    if len(typed) > 1:
        if all("(" in k for k in typed):
            said = "; ".join(f"'{t}' in {', '.join(sorted(ps))}" for t, ps in sorted(typed.values()))
            conflicts.append(
                Finding(
                    "headers.prototype_conflict",
                    f"{name} has no prototype and its declarations disagree: {said}",
                    unit=name,
                    blocking=False,
                    action=f"give {name} its signature in types.toml [function], then unbake check --repair",
                )
            )
        return None
    text = next(iter(typed.values()))[0] if typed else ""
    if not text or len(list(_parts(text.rstrip(";")))) != 1:
        return None
    return (text if text.startswith("extern") else "extern " + text), None


_ERROR_LINE = re.compile(r":\d+:(?:\d+:)? (?!warning)")


def _probe_job(item: tuple[Snapshot, UnitSpec, str, str, Any]) -> dict[str, Any]:
    """One header alone in one version: the probe unit preprocessed like a unit of the project and compiled to
    nothing. Rows carry the error count and the first three error lines."""
    snapshot, unit, version, header, recipe = item
    from unbake import adapters, process

    row: dict[str, Any] = {"header": header, "version": version, "errors": 0, "first": []}
    try:
        seen = _view.get(snapshot, unit, version, recipe, lines=False)
    except Refusal as refusal:
        return {**row, "errors": len(refusal.findings), "first": [f.reason for f in refusal.findings[:3]]}
    step = adapters.chain_steps(snapshot.config, recipe.toolchain, recipe, (), "probe")[0]
    argv = [w.replace("{in}", "probe.i").replace("{out}", "/dev/null") for w in step.argv]
    tools = effort.memo(
        ("tool-identity", recipe.toolchain), lambda: adapters.tool_identity(snapshot.config, recipe.toolchain)
    )

    def compiled() -> bytes:
        with store.work(snapshot.config) as home:
            (home / "probe.i").write_text(seen.text, encoding="utf-8")
            result = process.run("headers.probe", argv, home, tmp=process.scratch(snapshot.config.project.root))
        lines = result.stderr.decode(errors="replace").splitlines()
        errors = [line for line in lines if _ERROR_LINE.search(line)]
        if not errors and (result.exit != 0 or result.signal is not None):
            errors = lines[-1:] or [f"the compiler exited {result.exit}"]
        return json.dumps(errors).encode()

    # keyed by what the compiler reads (the preprocessed text, its command and tools): a header probed in an earlier
    # round, level or command with the same text is never compiled again, whichever tree state asks
    key = digest((seen.text, argv, str(tools), "probe"))
    errors = json.loads(store.cached(snapshot.config, "header-probe", key, compiled))
    return {**row, "errors": len(errors), "first": errors[:3]}


def probe(snapshot: Snapshot, versions: Sequence[str]) -> list[dict[str, Any]]:
    """Every header under include/ compiled alone in each version, with the flags of the project's first C unit
    outside the libraries, over the pool: the header oracle that needs no make."""
    from unbake import layout, recipes

    library = tuple(f.rstrip("/") + "/" for f in snapshot.config.project.library_dirs)
    compiled = {"preprocess", "compile"}
    units = [
        u
        for _, u in sorted(snapshot.layout.units.items())
        if compiled <= set(configuration.kind(u.kind)["phases"]) and not u.path.startswith(library)
    ]
    if not units:
        raise Refusal(Finding("headers.probe", "the project has no C unit to take the compile flags from"))
    paths = sources(snapshot)
    probes = {h: f"src/.probe/{digest(h)[:16]}.c" for h in paths}
    overlay = layout.overlay(
        snapshot, {p: f'#include "{h.removeprefix("include/")}"\n'.encode() for h, p in probes.items()}
    )
    recipe = recipes.resolve(snapshot.config, units[0], units[0].options)
    items = [(overlay, replace(units[0], path=probes[h]), v, h, recipe) for h in paths for v in versions]
    with effort.stage("headers.probe"):
        return pool.map(snapshot.config, "headers.probe", _probe_job, items)


def probe_command(config: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """unbake headers --probe [--versions all|V,..]: per header and version the errors of the header compiled alone."""
    from unbake import layout

    if not params.get("probe"):
        raise Refusal(Finding("headers.probe", "nothing to do: pass --probe", action="unbake headers --probe"))
    snapshot = layout.capture(config)
    wanted = params.get("versions") or ""
    every = list(config.project.versions)
    versions = every if wanted == "all" else [v for v in wanted.split(",") if v] or every[:1]
    if unknown := sorted(set(versions) - set(every)):
        raise Refusal(Finding("headers.probe", f"{unknown[0]} is not a version of the project"))
    rows = probe(snapshot, versions)
    failing = [r for r in rows if r["errors"]]
    refused = [
        {
            "index": i,
            "findings": [
                asdict(
                    Finding(
                        "headers.probe",
                        f"{r['header']} has {r['errors']} errors alone in {r['version']}: {(r['first'] or [''])[0]}",
                        path=r["header"],
                        versions=(r["version"],),
                    )
                )
            ],
        }
        for i, r in enumerate(failing)
    ]
    return {
        "headers": rows,
        "failing": len(failing),
        "refused": refused,
        "message": f"headers: {len(rows) - len(failing)} of {len(rows)} header probes compile",
    }
