"""Module ownership and link order, read from the source tree.

A file owns the names it defines in C (functions and non-extern data at brace depth 0) and the names its stand-ins
(`.include "nonmatchings/<seg>/<name>.s"`, `.include "data/<seg>/<name>.s"`) assemble. The preprocessor decides which
of them a version sees; the preprocessed text is scanned, never compiled. The result of one file is cached by content
alone (a comment is no content: `_meaning`), so an edit reparses the edited file and a rename reparses nothing."""

from __future__ import annotations

import functools
import json
import re
import warnings
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unbake import adapters, effort, libasm, pool, process, recipes, store, symbols
from unbake import config as configuration
from unbake.contracts import Config, Finding, Refusal, sha256
from unbake.resources import cc_cache

KIND = "tree.file"
TREE = "tree.captured"  # a whole captured tree, keyed by its inputs
LITERALS = "literals_"  # a fact `literals_<function>` places that function's file's compiler-emitted rodata
SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".s")
_INCLUDE = re.compile(rb'^[ \t]*#[ \t]*include[ \t]*"([^"\n]+)"', re.M)
_TOKEN = re.compile(
    r"""/\*.*?\*/|//[^\n]*|(?P<str>"(?:\\.|[^"\\\n])*")|'(?:\\.|[^'\\\n])*'|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|\d[\w.]*|(?P<p>[^\s])""",
    re.S,
)
_UNEXPANDED = re.compile(r"^[ \t]*(INCLUDE_(?:ASM|DATA|RODATA))[ \t]*\(", re.M)  # a stand-in call cpp left as it was
_STAND_IN = re.compile(r'\.include\s+"(?:nonmatchings|data)/[^/"]+/([^/"]+)\.s"')


# ---------------------------------------------------------------- the records
@dataclass(frozen=True)
class Module:
    path: str  # "src/IO/80200500.CPP"
    segment: str
    owns: Mapping[str, tuple[str, ...]]  # version -> names in file order (C definitions + stand-ins)
    fuzzy: tuple[str, ...]  # names with a NON_MATCHING body
    stand_ins: Mapping[str, tuple[str, ...]]  # version -> names still behind INCLUDE_*
    sections: Mapping[str, Mapping[str, str]] = field(default_factory=dict, compare=False)  # version -> name -> section
    placed: Mapping[str, Mapping[str, str]] = field(default_factory=dict, compare=False)  # version -> name -> segment
    stale: Mapping[str, tuple[str, ...]] = field(default_factory=dict, compare=False)  # version -> superseded stand-ins
    library: bool = False
    unnamed: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict, compare=False
    )  # version -> NON_MATCHING functions that no version has a symbol for


@dataclass(frozen=True)
class Tree:
    digest: str  # content digest: blob shas + config + symbol_addrs shas
    modules: tuple[Module, ...]
    symbols: Mapping[str, Mapping[str, symbols.Symbol]] = field(default_factory=dict, compare=False, repr=False)
    _owners: Mapping[tuple[str, str], str] = field(default_factory=dict, init=False, compare=False, repr=False)
    _addresses: dict[str, dict[int, str]] = field(default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        owners = {(v, n): m.path for m in self.modules for v, names in m.owns.items() for n in names}
        object.__setattr__(self, "_owners", owners)

    def owned(self, version: str, name: str) -> str | None:
        """The file that defines or stands in for exactly this name, nothing else."""
        return self._owners.get((version, name))

    def owner(self, version: str, name: str) -> str | None:
        """Who owns `name`: the file that defines it, else the file that owns a second name of the same bytes (the C
        spells an alias, a fact names the canonical symbol: both are one function, one owner)."""
        found = self._owners.get((version, name))
        if found is not None:
            return found
        fact = self.symbols.get(version, {}).get(name)
        return None if fact is None else self._at(version).get(fact[0])

    def _at(self, version: str) -> dict[int, str]:
        """address -> owning file of every owned name of the version that has a symbol fact."""
        if version not in self._addresses:
            table = self.symbols.get(version, {})
            found: dict[int, str] = {}
            for (v, n), path in sorted(self._owners.items()):
                if v == version and n in table:
                    found.setdefault(table[n][0], path)
            self._addresses[version] = found
        return self._addresses[version]

    def order(self, version: str, segment: str) -> tuple[str, ...]:
        """The module paths of a segment by the first address they own in the version."""
        table = self.symbols.get(version, {})
        rows = [
            (min(_rom(table, n) for n in m.owns[version]), m.path)
            for m in self.modules
            if m.segment == segment and m.owns.get(version)
        ]
        return tuple(path for _, path in sorted(rows))

    def unowned(self, version: str) -> tuple[str, ...]:
        """The symbols of the version no file owns, by address. A label inside data needs no owner: a data name off a
        word boundary, and a data name whose nearest name before it in the same mapping (ROM and RAM both contiguous)
        is data (owned, unowned or such a label): its bytes are that piece's. Nor does any name at the address of an
        owned name (a second name of the same bytes), nor a name the facts
        type `label`: it lies inside the bytes before it."""
        table = self.symbols.get(version, {})
        rows = sorted(
            (s[1], (version, n) not in self._owners, n) for n, s in table.items() if s[1] is not None
        )  # owned first at one address
        out: list[str] = []
        inside, mapping, held = False, None, -1
        for rom, free, name in rows:
            vram, kind = table[name][0], table[name][3]
            if rom - vram != mapping:
                inside, mapping = False, rom - vram
            if not free:
                held = rom
            if free and (
                rom == held or kind == "label" or (kind != "func" and (inside or symbols.inside_data(kind, vram)))
            ):
                continue
            if free:
                out.append(name)  # its stand-in is a piece of its own: the data names after it are its labels
            inside = kind != "func"
        return tuple(out)


def _rom(table: Mapping[str, symbols.Symbol], name: str) -> int:
    rom = table[name][1] if name in table else None
    return rom if rom is not None else 1 << 62


# ---------------------------------------------------------------- facts from the configuration
def _project(config: Config) -> Any:
    return config.project


def macro(version: str) -> str:
    return "VERSION_" + version.upper().replace("-", "_")


def _argv(
    config: Config, path: str, version: str, source: str, extra: Sequence[str] = (), markers: bool = False
) -> list[str]:
    """The build's preprocess command of `path` for `version`, as the Makefile spells it: the toolchain row's template
    with the row's, the project's and the [file] row's preprocessor flags, the version macro, `include` and the file's
    folder; `extra` flags follow the flags. With `markers`, a row's -P is dropped so line markers stay."""
    recipe = recipes.of(config, path)
    directory = path.rpartition("/")[0] or "."
    return adapters.toolchain(config, recipe.toolchain).argv(
        "preprocess",
        cppflags=[*(f for f in recipe.cppflags if not (markers and f == "-P")), *extra],
        defines=[f"-D{macro(version)}"],
        includes=["-Iinclude", f"-I{directory}/"],
        source=[source],
    )


# ---------------------------------------------------------------- one file
LOCAL_TEXT = ".text.local"


_DIRECTIVE = re.compile(r"(?m)^[ \t]*#.*$")
_State = tuple[int, int, tuple[str, ...]]  # (brace depth, parenthesis depth, the open top-level statement's tokens)
_Fold = tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...], _State]


def _scan(text: str, kinds: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
    """(names defined at depth 0, names whose stand-in the text assembles), each in text order. With `kinds`, also the
    linker section of each name: `.text` for a function, a stand-in's own `.section`, `.rodata` for a const, `.data`
    for an initialised variable, `.bss` for the rest."""
    defined, stand_ins, found, _ = _fold(_DIRECTIVE.sub("", text), (0, 0, ()))
    if kinds is not None:
        kinds.update(found)
    return list(defined), list(stand_ins)


_CHUNKS: dict[tuple[str, _State], _Fold] = {}
_CHUNKS_HELD = 1 << 18  # chunk entries one process keeps before it starts over


def _scan_expanded(text: str, kinds: dict[str, str]) -> tuple[list[str], list[str]]:
    """`_scan` of a preprocessor's output, read in runs of lines: a run of the same text met in the same state (the
    same header expanded the same way, ahead of thousands of files) is read once per process. A run ends after a line
    whose hash says so, so a run's bounds follow its own text, wherever the file puts it. No token of preprocessed
    text spans a line (no comment is left), so the runs give `_scan` exactly; text that still holds a comment is read
    whole."""
    if "/*" in text:
        return _scan(text, kinds)
    defined: list[str] = []
    stand_ins: list[str] = []
    state: _State = (0, 0, ())
    lines = text.split("\n")
    begin = 0
    for at, line in enumerate(lines):
        if hash(line) & 31 and at + 1 < len(lines):
            continue
        run = "\n".join(lines[begin : at + 1])
        begin = at + 1
        key = (run, state)
        folded = _CHUNKS.get(key)
        if folded is None:
            if len(_CHUNKS) >= _CHUNKS_HELD:
                _CHUNKS.clear()
            folded = _CHUNKS[key] = _fold(_DIRECTIVE.sub("", run), state)
        defined += folded[0]
        stand_ins += folded[1]
        kinds.update(folded[2])
        state = folded[3]
    return defined, stand_ins


def _fold(text: str, state: _State) -> _Fold:
    """The scan of `text` from `state`: the names it defines, the stand-ins it assembles and their sections, in text
    order, and the state it leaves."""
    defined: list[str] = []
    stand_ins: list[str] = []
    kinds: dict[str, str] = {}
    depth, parens, held = state
    stmt: list[str] = list(held)  # tokens of the current top-level statement; a braced body is one "{}" token
    for match in _TOKEN.finditer(text):
        token = match[0]
        if match["str"] is None and match["id"] is None and match["p"] is None:
            continue  # comment, character constant or number
        if match["str"] is not None:
            if depth == 0:
                stmt.append(token)
            continue
        if match["id"] is not None:
            if depth == 0:
                stmt.append(token)
            continue
        if token == "{":
            if depth == 0 and parens == 0 and _is_function(stmt):
                name = _function_name(stmt)
                if name and not _inlined(stmt):
                    defined.append(name)
                    # code, though a version knows the name only from an untyped alias; a static one is the file's
                    # own helper, which no symbol of any version is ever meant to name
                    kinds[name] = LOCAL_TEXT if "static" in stmt[: stmt.index("(")] else ".text"
                stmt = [*stmt, "{}"]
                stmt.append("<body>")
            elif depth == 0:
                stmt.append("{}")
            depth += 1
        elif token == "}":
            depth = max(0, depth - 1)
            if depth == 0 and stmt and stmt[-1] == "<body>":
                stmt = []
        elif depth == 0:
            if token == "(":
                parens += 1
            elif token == ")":
                parens = max(0, parens - 1)
            if token == ";" and parens == 0:
                _statement(stmt, defined, stand_ins, kinds)
                stmt = []
            else:
                stmt.append(token)
    return tuple(defined), tuple(stand_ins), tuple(kinds.items()), (depth, parens, tuple(stmt))


def _is_function(stmt: Sequence[str]) -> bool:
    if not stmt or stmt[-1] != ")" or "=" in stmt or stmt[0] == "typedef":
        return False
    return "(" in stmt and not (stmt[0].startswith('"'))


_INLINE = {"inline", "__inline", "__inline__"}


def _inlined(stmt: Sequence[str]) -> bool:
    """A `static inline` body is a copy the compiler folds into its callers: it gives the file no object bytes of that
    name, so another file's definition stays the one owner."""
    head = stmt[: stmt.index("(")]
    return "static" in head and any(t in _INLINE for t in head)


def _function_name(stmt: Sequence[str]) -> str:
    """The identifier before the first parenthesis: `int f(int a)` and `T (*g(void))(int)` give f and g."""
    at = stmt.index("(")
    if at + 1 < len(stmt) and stmt[at + 1] == "*":  # a function returning a function pointer
        inner = stmt[at + 2 :]
        return next((t for t in inner if _ident(t)), "")
    return stmt[at - 1] if at and _ident(stmt[at - 1]) else ""


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _ident(token: str) -> bool:
    return _IDENT.fullmatch(token) is not None


_ASM = {"__asm__", "asm", "__asm"}


_SECTION = re.compile(r"\.section\s+(\.[\w.]+)")  # a stand-in may give each name a section of its own


def _statement(stmt: Sequence[str], defined: list[str], stand_ins: list[str], kinds: dict[str, str]) -> None:
    if not _ASM.isdisjoint(stmt):
        joined = "".join(_unquote(t) for t in stmt if t.startswith('"'))
        for found in _STAND_IN.finditer(joined):
            sections = _SECTION.findall(joined[: found.start()])
            kinds[found[1]] = sections[-1] if sections else ".text"
            stand_ins.append(found[1])
        return
    if not stmt or "typedef" in stmt or "extern" in stmt:
        return
    pieces: list[list[str]] = [[]]
    depth = 0
    for token in stmt:
        depth += token in {"(", "["} or -(token in {")", "]"})
        if token == "," and depth == 0:
            pieces.append([])
        else:
            pieces[-1].append(token)
    for index, piece in enumerate(pieces):
        name = _data_name(piece, first=index == 0)
        if name:
            defined.append(name)
            kinds[name] = ".rodata" if "const" in stmt else ".data" if "=" in piece else ".bss"


def _unquote(token: str) -> str:
    return token[1:-1].replace('\\"', '"').replace("\\\\", "\\")


_AGGREGATE = {"struct", "union", "enum"}


def _data_name(piece: Sequence[str], first: bool) -> str:
    """The declared name of one declarator, "" for a prototype, a bare type or a tag."""
    tokens = list(piece[: piece.index("=")] if "=" in piece else piece)
    if "(" in tokens:
        at = tokens.index("(")
        # a function pointer variable names itself after the star; any other parenthesis is a prototype
        return next((t for t in tokens[at + 2 :] if _ident(t)), "") if tokens[at + 1 : at + 2] == ["*"] else ""
    flat: list[str] = []
    depth = 0
    for token in tokens:
        depth += token == "[" or -(token == "]")
        if depth == 0 and token not in {"[", "]", "{}"}:
            flat.append(token)
    if first and flat and flat[0] in _AGGREGATE:
        flat = flat[2:] if len(flat) > 1 and _ident(flat[1]) and flat[1] not in _AGGREGATE else flat[1:]
        needed = 1
    else:
        needed = 2 if first else 1
    idents = [t for t in flat if _ident(t)]
    return idents[-1] if len(idents) >= needed else ""


_LEXEME = re.compile(rb'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|/\*.*?\*/|//[^\n]*', re.S)


def _meaning(data: bytes) -> bytes:
    """`data` as far as the preprocessor's output can depend on it: each comment one space (cpp's own rule), so an edit
    of a comment alone (a renamed function in a field note) changes nothing a parse or a build reads. A comment over
    several lines keeps its line breaks (`__LINE__` and every later directive stay on their lines) and stays as written
    when a directive holds it or its last line goes on with more text; so does a line comment a backslash continues
    and anything inside a string or character literal."""

    def plain(m: re.Match[bytes]) -> bytes:
        lexeme = m[0]
        if lexeme[:1] in b"\"'" or lexeme.endswith(b"\\"):
            return lexeme
        breaks = lexeme.count(b"\n")
        if breaks:
            start = data.rfind(b"\n", 0, m.start()) + 1
            spliced = data[max(0, start - 3) : start].rstrip(b"\r\n").endswith(b"\\")  # the line goes on from above
            end = data.find(b"\n", m.end())
            after = data[m.end() : len(data) if end < 0 else end].strip()  # what the comment's last line goes on with
            if spliced or after or data[start : m.start()].lstrip(b" \t").startswith(b"#"):
                return lexeme
        return b" " + b"\n" * breaks

    return _LEXEME.sub(plain, data)


_HEADER_DIRECTIVE = re.compile(rb"[ \t]*#")


def _skeleton(data: bytes) -> bytes:
    """`data` (a header, comments gone) without the declarations that cannot change which names a source defines:
    every top-level type definition (typedef, struct, union, enum) and static inline function. Directives stay as
    written, and so does any run of lines that is not whole declarations (a struct body split by a conditional). A
    header edit that only adds, drops or changes such declarations keeps this text, so a parse keyed on it is reused
    where the preprocessor would otherwise run again for every source that includes the header."""
    out: list[bytes] = []
    run: list[bytes] = []
    joined = False

    def flush() -> None:
        if run:
            text = b"\n".join(run)
            run.clear()
            if cc_cache.complete(text):
                kept = [c for c in cc_cache.chunks(text) if cc_cache.defined(c) is None]
                out.append(b" ".join(b" ".join(c) for c in kept))
            else:
                out.append(text)

    for line in data.split(b"\n"):
        if joined or _HEADER_DIRECTIVE.match(line):
            flush()
            out.append(line)
            joined = line.rstrip(b"\r").endswith(b"\\")
        else:
            run.append(line)
    flush()
    return b"\n".join(out)


class _Headers:
    """What the include walk learns about a header, kept per reader: its sha, where each name resolves from a folder
    and its whole closure. A tree of thousands of files reaches the same few hundred headers, so each is hashed and
    walked once, not once per file. With `skeleton` a header counts by its `_skeleton` (the parse's second key)."""

    def __init__(self, read: Callable[[str], bytes | None], skeleton: bool = False) -> None:
        self._read = read
        self._skeleton = skeleton
        self._blobs: dict[str, bytes | None] = {}
        self._sha: dict[str, str] = {}
        self._includes: dict[bytes, list[str]] = {}
        self._resolved: dict[tuple[str, str, str, str, tuple[str, ...]], str] = {}
        self._reach: dict[tuple[str, str, str, tuple[str, ...]], frozenset[tuple[str, str]]] = {}

    def read(self, path: str) -> bytes | None:
        if path not in self._blobs:
            self._blobs[path] = self._read(path)
        return self._blobs[path]

    def sha(self, path: str, blob: bytes | None) -> str:
        if blob is None:
            return ""
        if path not in self._sha:
            meaning = _meaning(blob)
            self._sha[path] = sha256(_skeleton(meaning) if self._skeleton else meaning)
        return self._sha[path]

    def resolve(self, here: str, name: str, directory: str, version: str, extra: tuple[str, ...]) -> str:
        key = (here, name, directory, version, extra)
        if key not in self._resolved:
            self._resolved[key] = next(
                (
                    p
                    for p in (
                        f"{here}/{name}",
                        *(f"{d}/{name}" for d in extra),
                        f"include/{name}",
                        f"{directory}/{name}",
                        f"asm/{version}/include/{name}",
                    )
                    if self.read(p) is not None
                ),
                "",
            )
        return self._resolved[key]

    def direct(
        self, here: str, data: bytes, directory: str, version: str, extra: tuple[str, ...]
    ) -> list[tuple[tuple[str, str], str]]:
        """((spelled name, blob sha or ""), resolved path or "") of each include of `data` found from `here`."""
        if data not in self._includes:  # the same text is scanned once, whatever version or folder reads it
            self._includes[data] = [raw.decode(errors="replace") for raw in _INCLUDE.findall(data)]
        out = []
        for name in self._includes[data]:
            resolved = self.resolve(here, name, directory, version, extra)
            blob = self.read(resolved) if resolved else None
            out.append(((name, self.sha(resolved, blob)), resolved if blob is not None else ""))
        return out

    def reach(self, path: str, directory: str, version: str, extra: tuple[str, ...]) -> frozenset[tuple[str, str]]:
        """Every include entry reachable from the header `path`, its own and its descendants'."""
        key = (path, directory, version, extra)
        if key not in self._reach:
            found: set[tuple[str, str]] = set()
            seen = {path}
            queue = [path]
            while queue:
                here = queue.pop()
                for entry, resolved in self.direct(
                    here.rpartition("/")[0], self.read(here) or b"", directory, version, extra
                ):
                    found.add(entry)
                    if resolved and resolved not in seen:
                        seen.add(resolved)
                        queue.append(resolved)
            self._reach[key] = frozenset(found)
        return self._reach[key]


def _closure(
    read: Callable[[str], bytes | None] | _Headers, directory: str, version: str, text: bytes, flags: Sequence[str] = ()
) -> list[tuple[str, str]]:
    """The headers `text` includes, through nested includes: (spelled name, blob sha or "") with no path in them. A
    header resolves the way cpp finds it: the including file's folder, then each `-I` of `flags`, then the defaults."""
    headers = read if isinstance(read, _Headers) else _Headers(read)
    extra = tuple(f[2:].rstrip("/") for f in flags if f.startswith("-I") and len(f) > 2)
    found: set[tuple[str, str]] = set()
    for entry, resolved in headers.direct(directory, text, directory, version, extra):
        found.add(entry)
        if resolved:
            found |= headers.reach(resolved, directory, version, extra)
    return sorted(found)


@functools.cache
def _code() -> str:
    """The bytes of this module, the one place a file's parse is made (preprocessing, scanning)."""
    return sha256(Path(__file__).read_bytes())


def _key(config: Config, path: str, text: bytes, version: str, read: Callable[[str], bytes | None] | _Headers) -> str:
    """A file's parse key names every input of the parse: the text, the version, the preprocessor and its flags, the
    headers it reaches, and this module's code (never the rest of the tool, so another tool edit keeps it). The text and
    every header count by `_meaning`: an edit of their comments alone keeps the key."""
    directory = path.rpartition("/")[0]
    argv = _argv(config, path, version, "-")
    basis = ["sections", _code(), sha256(_meaning(text)), argv, _closure(read, directory, version, text, argv)]
    return sha256(json.dumps(basis, sort_keys=True).encode())


def preprocess(config: Config, path: str, version: str, text: bytes | None = None, markers: bool = False) -> str:
    """The text the build's compiler reads for the source `path` in `version`, byte for byte the Makefile's preprocess
    step: of the file on disk, or of `text` as that file; `markers` keeps line markers a row's -P would drop."""
    return _preprocess(config, path, text, version, (), markers)


_CONDITIONAL = re.compile(r"[ \t]*#[ \t]*(if|ifdef|ifndef|else|elif|endif)\b")
_OPENER = re.compile(r"([ \t]*#[ \t]*)(ifdef|if)\b(.*?)(\r?\n?)$", re.S)
_EXPR_TOKEN = re.compile(r"\s*(defined\b|[A-Za-z_]\w*|0[xX][0-9a-fA-F]+|\d+|&&|\|\||[!=<>]=|[<>]|!|[()+\-*/%])")
_DEFINED = re.compile(r"\s*(?:\(\s*([A-Za-z_]\w*)\s*\)|([A-Za-z_]\w*))")


def _condition(text: str, matching: bool) -> bool | None:
    """The truth of an `#if` expression with NON_MATCHING defined or not, every other name defined and every other
    identifier 0 (the build of the one version being measured), None when it is not an expression of those."""
    text = re.sub(r"/\*.*?\*/|//.*", " ", text).rstrip()
    out: list[str] = []
    at = 0
    while at < len(text):
        m = _EXPR_TOKEN.match(text, at)
        if m is None:
            return None
        at, word = m.end(), m[1]
        if word == "defined":
            named = _DEFINED.match(text, at)
            if named is None:
                return None
            at = named.end()
            out.append(str(matching if (named[1] or named[2]) == "NON_MATCHING" else True))
        elif word[0].isalpha() or word[0] == "_":
            out.append("0")
        else:
            out.append({"&&": " and ", "||": " or ", "!": " not "}.get(word, word))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # `FOO(1)` reads as a call of 0: not an expression, so None below
            return bool(eval("".join(out), {"__builtins__": {}}, {}))  # only the tokens built above
    except Exception:
        return None


_PLAIN = re.compile(r"\s*(?:\(\s*)?NON_MATCHING\s*\)?\s*|\s*defined\s*\(?\s*NON_MATCHING\s*\)?\s*")


def _plain(line: str) -> tuple[str, str] | None:
    """`guard` for the one-condition spellings only (`#ifdef NON_MATCHING`, `#if defined(NON_MATCHING)`): the guards
    whose sibling bodies a measured module switches off. A compound guard (`defined(A) && defined(NON_MATCHING)`)
    stays on, as the file's own build has it: its stand-in side can need a version's symbols and flags that a
    measure of another name does not have."""
    m = _OPENER.match(line)
    return guard(line) if m and _PLAIN.fullmatch(m[3]) else None


def guard(line: str) -> tuple[str, str] | None:
    """(the directive's leading `#`, its line ending) when `line` opens a conditional whose first side is the
    NON_MATCHING side, whatever its spelling (`#ifdef NON_MATCHING`, `#if defined(A) && defined(NON_MATCHING)`): true
    with NON_MATCHING defined and false without it."""
    m = _OPENER.match(line)
    if m is None or "NON_MATCHING" not in m[3]:
        return None
    expression = f"defined({m[3].strip()})" if m[2] == "ifdef" else m[3]
    if _condition(expression, True) is True and _condition(expression, False) is False:
        return m[1], m[4]
    return None


def non_matching_blocks(lines: Sequence[str]) -> list[tuple[int, int, int]]:
    """The NON_MATCHING conditionals of the lines: (opening line, `#else` line or -1, `#endif` line), 0-based."""
    stack: list[list[int]] = []  # [line of the opening directive, line of its #else or -1]
    blocks: list[tuple[int, int, int]] = []
    for number, line in enumerate(lines):
        kind = _CONDITIONAL.match(line)
        if kind is None:
            continue
        if kind[1] in ("if", "ifdef", "ifndef"):
            stack.append([number, -1])
        elif kind[1] in ("else", "elif") and stack and stack[-1][1] < 0:
            stack[-1][1] = number
        elif kind[1] == "endif" and stack:
            opened, other = stack.pop()
            if guard(lines[opened]):
                blocks.append((opened, other, number))
    return blocks


def defined(text: str) -> list[str]:
    """The names `text` defines as C functions."""
    return _scan(text)[0]


def fuzzy_only(text: str, names: Collection[str]) -> str:
    """`text` with the NON_MATCHING body of every function but `names` switched off: a measured module compiles the
    bodies of the names it measures that no plain definition gives, and a sibling's body (its own includes and
    declarations, often the older context of the file) is no part of what a name's bytes depend on. A switched-off
    block keeps its line count and its `#else` side, so every diagnostic names the line it names in `text`."""
    lines = text.splitlines(keepends=True)
    blocks = non_matching_blocks(lines)
    fenced = {n for opened, _, closed in blocks for n in range(opened, closed + 1)}
    # a name the file defines outside every block (the candidate standing where its stand-in was) is measured from that
    # definition: an older draft of it in a block, often one shared with other functions, is no part of the measure
    wanted = set(names) - set(defined("".join(ln for n, ln in enumerate(lines) if n not in fenced)))
    for opened, other, closed in blocks:
        body = "".join(lines[opened + 1 : other if other >= 0 else closed])
        if not set(defined(body)) & wanted and (m := _plain(lines[opened])):
            lines[opened] = f"{m[0]}if 0{m[1]}"
    return "".join(lines)


def origins(config: Config, path: str, version: str, text: bytes | None = None) -> list[tuple[str, int]]:
    """Per line of `preprocess(config, path, version, text)`, the file and line it was written on: the build's own
    output has no line markers, so a compiler diagnostic that names a line of it is read through this. A line the
    preprocessor made from nothing (a blank one) takes the line after the last written one."""
    marked = preprocess(config, path, version, text, True).splitlines()
    written: list[tuple[str, int, bool]] = []  # (file, line, blank) of every line that is not a marker
    current, number = path, 1
    for line in marked:
        if m := re.match(r'# (\d+) "([^"]*)"', line):
            current, number = (path if m[2] == "<stdin>" else m[2]), int(m[1])
        else:
            written.append((current, number, not line.strip()))
            number += 1
    found: list[tuple[str, int]] = []
    at = 0
    for line in preprocess(config, path, version, text).splitlines():
        if line.strip():
            while at < len(written) and written[at][2]:
                at += 1
            found.append(written[at][:2] if at < len(written) else (path, 0))
            at += 1
        else:
            last = found[-1] if found else (path, 0)
            found.append((last[0], last[1] + 1))
    return found


def _materialize(home: Path, overlays: Mapping[str, bytes | None] | None) -> tuple[str, ...]:
    """Write every overlay file under `home` and the include flags that put it ahead of the tree's `include` and
    `src`, so a header that only the overlay holds is found like one the tree has."""
    written = {p: b for p, b in (overlays or {}).items() if b is not None}
    for name, content in written.items():
        target = home / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)  # never write through a link into the real tree
        target.write_bytes(content)
    return (f"-I{home / 'include'}", f"-I{home / 'src'}") if written else ()


def _preprocess(
    config: Config, path: str, text: bytes | None, version: str, extra: Sequence[str], markers: bool = False
) -> str:
    """What the build's preprocessor makes of `path` for `version`, with `extra` flags: of `text` as that file (read
    on stdin), or of the file on disk when `text` is None (its own name in any line markers, as the build has it)."""
    return _expand(config, path, text, version, extra, markers, listing=False)[0]


def _expand(
    config: Config,
    path: str,
    text: bytes | None,
    version: str,
    extra: Sequence[str],
    markers: bool = False,
    listing: bool = True,
) -> tuple[str, list[str]]:
    """`_preprocess`, and with `listing` the paths of the headers the preprocessor opened (its `-H` list)."""
    root = config.project.root
    argv = _argv(config, path, version, path if text is None else "-", [*extra, *(["-H"] if listing else [])], markers)
    result = process.run("tree.cpp", argv, root, tmp=process.scratch(root), stdin=text or b"")
    if result.exit != 0:
        lines = [ln.strip() for ln in result.stderr.decode(errors="replace").splitlines() if ln.strip()]
        shown = [ln for ln in lines if ln != "compilation terminated."]  # the footer says nothing of the cause
        site = next((m for ln in shown if (m := re.match(r"(?:<stdin>|[\w./-]+):(\d+)(?::\d+)?: ", ln))), None)
        raise Refusal(
            Finding(
                "native.exit",
                reason="; ".join(shown or lines) or "cpp failed",
                path=path,
                line=int(site[1]) if site else 0,
                versions=(version,),
            )
        )
    opened = [m[1] for ln in result.stderr.decode(errors="replace").splitlines() if (m := _OPENED.match(ln))]
    return result.stdout.decode(errors="replace"), opened


_OPENED = re.compile(r"\.+ (\S.*)$")  # a line of the preprocessor's -H list: dots for the depth, then the header
_FUZZY_WORD = b"NON_"  # what a file must hold for -DNON_MATCHING to be able to change what it expands to
_MENTIONS: dict[tuple[str, int, int], bool] = {}


def _mentions(root: Path, header: str) -> bool:
    """Whether the header holds the word, read once per process while the file is unchanged. A header that cannot be
    read counts as holding it."""
    full = root / header
    try:
        stat = full.stat()
        key = (str(full), stat.st_mtime_ns, stat.st_size)
        if key not in _MENTIONS:
            _MENTIONS[key] = _FUZZY_WORD in full.read_bytes()
        return _MENTIONS[key]
    except OSError:
        return True


def _read_from(config: Config) -> Callable[[str], bytes | None]:
    root = _project(config).root

    def read(path: str) -> bytes | None:
        try:
            data: bytes = (root / path).read_bytes()
            return data
        except OSError:
            return None

    return read


def parse(
    config: Config,
    path: str,
    text: bytes,
    version: str,
    read: Callable[[str], bytes | None] | None = None,
    overlays: Mapping[str, bytes | None] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """(C-defined names, stand-in names, NON_MATCHING names) of one file in one version. Never compiles. Every file of
    `overlays`, new or existing, is in the preprocessor's search ahead of the tree."""
    first, second, third, _ = _parsed(config, path, text, version, read, None, overlays)
    return first, second, third


def _parsed(
    config: Config,
    path: str,
    text: bytes,
    version: str,
    read: Callable[[str], bytes | None] | None = None,
    key: str | None = None,
    overlays: Mapping[str, bytes | None] | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """`parse` and the linker section of each defined or stand-in name, kept together in one cache entry. A caller that
    already made the key (`_key` over the reader the entry is looked up with) passes it, so both name one entry."""
    if configuration.is_assembly(path):
        kinds = libasm.scan(text.decode())
        return list(kinds), [], [], kinds
    key = key or _key(config, path, text, version, read or _read_from(config))

    def produce() -> bytes:
        kinds: dict[str, str] = {}
        with store.work(config) as home:
            ahead = _materialize(home, overlays)
            source = text
            expanded, opened = _expand(config, path, source, version, ("-P", *ahead))
            if _UNEXPANDED.search(expanded):  # no header yet: read the file as sync will write it
                source = b'#include "include_asm.h"\n' + text
                expanded, opened = _expand(config, path, source, version, ("-P", *ahead))
            plain = _scan_expanded(expanded, kinds)
            fuzzy_kinds: dict[str, str] = {}
            root = config.project.root
            if _FUZZY_WORD in source or any(_mentions(root, h) for h in opened):
                other = _preprocess(config, path, source, version, ("-P", "-DNON_MATCHING", *ahead))
                fuzzy = plain if other == expanded else _scan_expanded(other, fuzzy_kinds)
            else:  # nothing the preprocessor read can tell the macro: the second expansion is the first
                fuzzy = plain
        for name in fuzzy[0]:  # a NON_MATCHING body's kind too, so a body of nothing can be told from a data row
            if name in fuzzy_kinds:
                kinds.setdefault(name, fuzzy_kinds[name])
        mine = set(plain[0])
        row = [plain[0], plain[1], [n for n in dict.fromkeys(fuzzy[0]) if n not in mine], kinds]
        return json.dumps(row).encode()

    first, second, third, kinds = json.loads(store.cached(config, KIND, key, produce))
    return first, second, third, kinds


# ---------------------------------------------------------------- the tree
def _is_source(path: str) -> bool:
    return path.lower().endswith(SOURCE_SUFFIXES)


def _listing(config: Config, folder: str) -> list[str]:
    out = process.git(config, "ls-files", "-co", "--exclude-standard", "-z", folder).stdout
    return sorted({p for p in out.decode().split("\0") if p})


def _files(config: Config, overlay: Mapping[str, bytes | None]) -> dict[str, bytes]:
    """The bytes of every file the parse reads (src, include, symbol files), overlay applied."""
    root = _project(config).root
    paths = {p for folder in ("src", "include") for p in _listing(config, folder)}
    paths |= {f(v) for v in _project(config).versions for f in (symbols.path, symbols.aliases_path)}
    paths |= {p for p in overlay if p.startswith(("src/", "include/"))}
    blobs: dict[str, bytes] = {}
    for path in sorted(paths):
        if path in overlay:
            data = overlay[path]
        else:
            try:
                data = (root / path).read_bytes()
            except OSError:
                data = None
        if data is not None:
            blobs[path] = data
    return blobs


def _reader(
    config: Config, blobs: Mapping[str, bytes], overlay: Mapping[str, bytes | None]
) -> Callable[[str], bytes | None]:
    """What a file's parse reads: the captured blobs, else (a file outside src/include: a relative or generated header)
    the disk, an overlay's deletion staying absent. Each path is read once."""
    disk = _read_from(config)
    seen: dict[str, bytes | None] = {}

    def read(path: str) -> bytes | None:
        if path in blobs:
            return blobs[path]
        if path not in seen:
            seen[path] = None if path in overlay else disk(path)
        return seen[path]

    return read


def _parse_job(item: tuple[Any, ...]) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    config, path, text, version, key, second = item
    row = _parsed(config, path, text, version, key=key)
    if second and (stored := store.get(config, KIND, key)) is not None:
        store.put(config, KIND, second, stored)  # the same row under the skeleton key, for the next header edit
    return row


def _digest(config: Config, blobs: Mapping[str, bytes]) -> str:
    rows = [(p, sha256(b)) for p, b in blobs.items()]
    return sha256(json.dumps([rows, _project(config).digest], sort_keys=True).encode())


def capture(config: Config, overlay: Mapping[str, bytes | None] = {}) -> Tree:
    with effort.stage("tree.capture"):
        blobs = _files(config, overlay)
        digest = _digest(config, blobs)
        return effort.memo(("tree", digest), lambda: _captured(config, blobs, digest, overlay))


def _captured(config: Config, blobs: Mapping[str, bytes], digest: str, overlay: Mapping[str, bytes | None]) -> Tree:
    """The tree of these files: a pure function of them, the parse keys (headers read off disk included) and the
    tool's code, so a later command finds it in the durable cache and builds nothing."""
    jobs = _jobs(config, blobs)
    keys = _keys(config, blobs, jobs, overlay)
    key = sha256(json.dumps([store.code(), digest, keys]).encode())
    return store.cached_object(config, TREE, key, lambda: _build(config, blobs, jobs, keys, overlay))


def _jobs(config: Config, blobs: Mapping[str, bytes]) -> list[tuple[str, str]]:
    versions: tuple[str, ...] = tuple(_project(config).versions)
    return [(p, v) for p in blobs if p.startswith("src/") and _is_source(p) for v in versions]


def _build(
    config: Config,
    blobs: Mapping[str, bytes],
    jobs: Sequence[tuple[str, str]],
    keys: Sequence[str],
    overlay: Mapping[str, bytes | None],
) -> Tree:
    project = _project(config)
    versions: tuple[str, ...] = tuple(project.versions)
    tables = {
        v: symbols.merged(
            blobs.get(symbols.path(v), b"").decode(errors="replace"),
            blobs.get(symbols.aliases_path(v), b"").decode(errors="replace"),
            symbols.spans(config, v),
            (symbols.path(v), symbols.aliases_path(v)),
        )
        for v in versions
    }
    parsed = _parse_all(config, blobs, jobs, keys, overlay)
    owned: dict[str, dict[str, list[str]]] = defaultdict(dict)
    stand: dict[str, dict[str, list[str]]] = defaultdict(dict)
    fuzzy: dict[str, set[str]] = defaultdict(set)
    unnamed: dict[str, dict[str, list[str]]] = defaultdict(dict)
    sections: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    defined = {(version, n) for (path, version), (defs, *_) in zip(jobs, parsed, strict=True) for n in defs}
    stale: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for (path, version), (defs, ins, nm, kinds) in zip(jobs, parsed, strict=True):
        table = tables[version]
        mine = set(defs)
        old = [n for n in ins if n in table and n not in mine and (version, n) in defined]  # a C body supersedes it
        if old or set(ins) & mine:
            stale[path][version] = [n for n in dict.fromkeys(ins) if n in table and (n in mine or n in old)]
        names = [n for n in dict.fromkeys([*defs, *ins]) if n in table and n not in old]
        if lost := [
            n for n in dict.fromkeys(nm) if kinds.get(n) == ".text" and not any(n in t for t in tables.values())
        ]:
            unnamed[path][version] = lost  # a body for a function no version has is built into nothing
        if names:
            owned[path][version] = sorted(names, key=lambda n, t=table: (_rom(t, n), n))  # type: ignore[misc]
            stand[path][version] = [n for n in ins if n in table and n not in mine and n not in old]
            sections[path][version] = {n: _section(table[n], kinds.get(n)) for n in names}
            fuzzy[path] |= {n for n in nm if n in table}
    _yield_to_source(tables, owned, sections)
    findings = _duplicates(owned, versions)
    findings += _literals(tables, owned, sections)
    _absorbed(tables, owned, sections, stale)
    modules = tuple(
        _module(
            config,
            path,
            owned[path],
            stand[path],
            fuzzy[path],
            sections[path],
            tables,
            findings,
            stale[path],
            unnamed[path],
        )
        for path in sorted({*owned, *stale})
    )
    if findings:
        raise Refusal(*findings)
    return Tree(_digest(config, blobs), modules, tables)


def _literals(
    tables: Mapping[str, Mapping[str, symbols.Symbol]],
    owned: dict[str, dict[str, list[str]]],
    sections: dict[str, dict[str, dict[str, str]]],
) -> list[Finding]:
    """A fact `literals_<function>` places the read-only data the compiler emits for the file that owns <function>
    (its literal pools, jump tables): the file owns it as `.rodata`. A row whose function no file owns is a finding."""
    findings: list[Finding] = []
    for version, table in tables.items():
        owner = {n: path for path in owned for n in owned[path].get(version, ())}
        for name in table:
            if not name.startswith(LITERALS):
                continue
            path = owner.get(name[len(LITERALS) :])
            if path is None:
                findings.append(
                    Finding(
                        "tree.literals",
                        f"{name} in {version} places literals for {name[len(LITERALS) :]}, which no file owns",
                        path=symbols.path(version),
                        unit=name,
                        versions=(version,),
                    )
                )
                continue
            names = [*owned[path][version], name]
            owned[path][version] = sorted(names, key=lambda n, t=table: (_rom(t, n), n))  # type: ignore[misc]
            sections[path][version][name] = ".rodata"
    return findings


def _absorbed(
    tables: Mapping[str, Mapping[str, symbols.Symbol]],
    owned: dict[str, dict[str, list[str]]],
    sections: dict[str, dict[str, dict[str, str]]],
    stale: dict[str, dict[str, list[str]]],
) -> None:
    """A `literals_<function>` row hands the read-only bytes from its address over its size (one byte when it has no
    size) to the file that owns the function: a stand-in of another file for a name inside them (a data chunk, an
    array of words, a float the old pool named on its own) no longer stands for anything and is superseded, as a
    stand-in a C body replaces is. Only a source file can hold a stand-in."""
    for version, table in tables.items():
        spans = sorted(
            (rom, rom + (table[n][2] or 1))
            for by_version in owned.values()
            for n in by_version.get(version, ())
            if n.startswith(LITERALS) and (rom := table[n][1]) is not None
        )
        if not spans:
            continue

        def inside(rom: int | None, spans: Sequence[tuple[int, int]] = spans) -> bool:
            return rom is not None and any(start <= rom < end for start, end in spans)

        for path in [p for p in owned if not configuration.is_assembly(p)]:
            names = owned[path].get(version, [])
            gone = [
                n
                for n in names
                if not n.startswith(LITERALS)
                and inside(table[n][1])
                and sections[path][version].get(n, "").startswith(".rodata")
            ]
            if not gone:
                continue
            kept = [n for n in names if n not in gone]
            stale[path][version] = [*dict.fromkeys([*stale[path].get(version, ()), *gone])]
            if kept:
                owned[path][version] = kept
                sections[path][version] = {n: sections[path][version][n] for n in kept}
            else:
                owned[path].pop(version, None)
                sections[path].pop(version, None)


def _keys(
    config: Config,
    blobs: Mapping[str, bytes],
    jobs: Sequence[tuple[str, str]],
    overlay: Mapping[str, bytes | None],
    skeleton: bool = False,
) -> list[str]:
    """The parse key of each (file, version): the one key function, whichever process parses or looks it up. With
    `skeleton` the second key, where each header counts without its type definitions and static inline functions."""
    read = _Headers(_reader(config, blobs, overlay), skeleton)
    keys = [_key(config, p, blobs[p], v, read) for p, v in jobs]
    return [sha256(b"skeleton " + k.encode()) for k in keys] if skeleton else keys


def _parse_all(
    config: Config,
    blobs: Mapping[str, bytes],
    jobs: Sequence[tuple[str, str]],
    keys: Sequence[str],
    overlay: Mapping[str, bytes | None] = {},
) -> list[tuple[list[str], list[str], list[str], dict[str, str]]]:
    """Every (file, version) result: hits of the cache are read here, only the misses are parsed (in the pool). A miss
    is looked up again under its skeleton key (the headers without their type definitions and static inline functions):
    an edit of those leaves every including source's names alone, and the preprocessor does not run for any of them."""
    hits = {k: store.get(config, KIND, k) for k in set(keys)}
    todo = [i for i, k in enumerate(keys) if hits[k] is None]
    seconds = dict(zip(todo, _keys(config, blobs, [jobs[i] for i in todo], overlay, skeleton=True), strict=True))
    found = {s: store.get(config, KIND, s) for s in set(seconds.values())}
    for i in todo:
        if found[seconds[i]] is not None and hits[keys[i]] is None:
            hits[keys[i]] = found[seconds[i]]
            store.put(config, KIND, keys[i], found[seconds[i]])  # a later run finds it under the first key
    todo = [i for i in todo if hits[keys[i]] is None]
    items = [(config, jobs[i][0], blobs[jobs[i][0]], jobs[i][1], keys[i], seconds[i]) for i in todo]
    done = dict(zip(todo, pool.map(config, "tree.parse", _parse_job, items), strict=True)) if items else {}
    out = []
    for i, k in enumerate(keys):
        if i in done:
            out.append(done[i])
        else:
            first, second, third, kinds = json.loads(hits[k] or b"[[],[],[],{}]")
            out.append((first, second, third, kinds))
    return out


def _yield_to_source(
    tables: Mapping[str, Mapping[str, symbols.Symbol]],
    owned: dict[str, dict[str, list[str]]],
    sections: dict[str, dict[str, dict[str, str]]],
) -> None:
    """An assembly unit does not own, in a version, a name whose ROM address another source already owns there: the
    name is shared across versions, but this version's bytes at that address belong to the other source (a data
    chunk, a differently named routine)."""
    for version, table in tables.items():
        taken = {
            table[n][1]
            for path, by_version in owned.items()
            if not configuration.is_assembly(path)
            for n in by_version.get(version, ())
            if table[n][1] is not None
        }
        for path in [p for p in owned if configuration.is_assembly(p)]:
            names = owned[path].get(version, [])
            kept = [n for n in names if table[n][1] not in taken]
            if len(kept) == len(names):
                continue
            if kept:
                owned[path][version] = kept
                sections[path][version] = {n: sections[path][version][n] for n in kept}
            else:
                owned[path].pop(version, None)
                sections[path].pop(version, None)


def _duplicates(owned: Mapping[str, Mapping[str, Iterable[str]]], versions: Sequence[str]) -> list[Finding]:
    findings: list[Finding] = []
    for version in versions:
        seen: dict[str, str] = {}
        for path in sorted(owned):
            for name in owned[path].get(version, ()):
                if name in seen:
                    findings.append(
                        Finding(
                            "tree.duplicate",
                            f"{name} is owned by {seen[name]} and {path} in {version}",
                            path=path,
                            unit=name,
                            versions=(version,),
                        )
                    )
                seen.setdefault(name, path)
    return findings


def _section(fact: symbols.Symbol, kind: str | None) -> str:
    """The linker section a name's bytes are in: code for a function, else what the source says (a stand-in's own
    `.section`, const, initialised or not)."""
    return ".text" if fact[3] == "func" or kind == LOCAL_TEXT else kind or ".data"


def _module(
    config: Config,
    path: str,
    owns: Mapping[str, list[str]],
    stand: Mapping[str, list[str]],
    fuzzy: set[str],
    sections: Mapping[str, Mapping[str, str]],
    tables: Mapping[str, Mapping[str, symbols.Symbol]],
    findings: list[Finding],
    stale: Mapping[str, list[str]],
    unnamed: Mapping[str, list[str]],
) -> Module:
    """The module record. Its segment is the one its first function lies in: a file may also own data that lies in
    segments of their own (the rodata of its code), and each name's own segment is found from its address."""
    candidates: list[tuple[bool, int, str]] = []  # (is data, rom, segment): the first function decides
    placed: dict[str, dict[str, str]] = {}
    for version, names in owns.items():
        for name in names:
            fact = tables[version][name]
            at = symbols.segment_at(config, version, fact[1]) if fact[1] is not None else None
            if fact[1] is not None and at is None:
                findings.append(
                    Finding(
                        "tree.segments",
                        f"{name} in {version} lies in no segment of the project config",
                        path=path,
                        unit=name,
                        versions=(version,),
                    )
                )
            elif at is not None and fact[1] is not None:
                candidates.append((fact[3] != "func", fact[1], at))
                placed.setdefault(version, {})[name] = at
    return Module(
        path,
        min(candidates)[2] if candidates else "",
        {v: tuple(n) for v, n in owns.items()},
        tuple(sorted(fuzzy)),
        {v: tuple(n) for v, n in stand.items() if n},
        {v: dict(n) for v, n in sections.items()},
        placed,
        {v: tuple(n) for v, n in stale.items() if n},
        library=configuration.is_assembly(path) and configuration.library(config.project, path),
        unnamed={v: tuple(n) for v, n in unnamed.items()},
    )
