"""Module ownership and link order, read from the source tree.

A file owns the names it defines in C (functions and non-extern data at brace depth 0) and the names its stand-ins
(`.include "nonmatchings/<seg>/<name>.s"`, `.include "data/<seg>/<name>.s"`) assemble. The preprocessor decides which
of them a version sees; the preprocessed text is scanned, never compiled. The result of one file is cached by content
alone, so an edit reparses the edited file and a rename reparses nothing."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unbake import config as configuration
from unbake import effort, pool, process, store, symbols
from unbake.contracts import Config, Finding, Refusal, sha256

KIND = "tree.file"
SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx")
_INCLUDE = re.compile(rb'^[ \t]*#[ \t]*include[ \t]*"([^"\n]+)"', re.M)
_TOKEN = re.compile(
    r"""/\*.*?\*/|//[^\n]*|(?P<str>"(?:\\.|[^"\\\n])*")|'(?:\\.|[^'\\\n])*'|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|\d[\w.]*|(?P<p>[^\s])""",
    re.S,
)
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


@dataclass(frozen=True)
class Tree:
    digest: str  # content digest: blob shas + config + symbol_addrs shas
    modules: tuple[Module, ...]
    symbols: Mapping[str, Mapping[str, symbols.Symbol]] = field(default_factory=dict, compare=False, repr=False)
    _owners: Mapping[tuple[str, str], str] = field(default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        owners = {(v, n): m.path for m in self.modules for v, names in m.owns.items() for n in names}
        object.__setattr__(self, "_owners", owners)

    def owner(self, version: str, name: str) -> str | None:
        return self._owners.get((version, name))

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
        """The symbols of the version no file owns, by address."""
        rows = [(s[1], n) for n, s in self.symbols.get(version, {}).items() if s[1] is not None]
        return tuple(n for _, n in sorted(rows) if (version, n) not in self._owners)


def _rom(table: Mapping[str, symbols.Symbol], name: str) -> int:
    rom = table[name][1] if name in table else None
    return rom if rom is not None else 1 << 62


# ---------------------------------------------------------------- facts from the configuration
def _project(config: Config) -> Any:
    return config.project


def macro(version: str) -> str:
    return "VERSION_" + version.upper().replace("-", "_")


def _cppflags(config: Config) -> list[str]:
    project = _project(config)
    row = configuration.load_resource("toolchains.toml")["toolchain"].get(project.toolchain, {})
    return [*row.get("cppflags", ()), *project.cppflags]


# ---------------------------------------------------------------- one file
def _scan(text: str, kinds: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
    """(names defined at depth 0, names whose stand-in the text assembles), each in text order. With `kinds`, also the
    linker section of each name: a stand-in's own `.section`, `.rodata` for a const, `.data` for an initialised
    variable, `.bss` for the rest."""
    text = re.sub(r"(?m)^[ \t]*#.*$", "", text)
    defined: list[str] = []
    stand_ins: list[str] = []
    stmt: list[str] = []  # tokens of the current top-level statement; a braced body is one "{}" token
    depth = parens = 0
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
                if name:
                    defined.append(name)
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
                _statement(stmt, defined, stand_ins, {} if kinds is None else kinds)
                stmt = []
            else:
                stmt.append(token)
    return defined, stand_ins


def _is_function(stmt: Sequence[str]) -> bool:
    if not stmt or stmt[-1] != ")" or "=" in stmt or stmt[0] in {"struct", "union", "enum", "typedef"}:
        return False
    return "(" in stmt and not (stmt[0].startswith('"'))


def _function_name(stmt: Sequence[str]) -> str:
    """The identifier before the first parenthesis: `int f(int a)` and `T (*g(void))(int)` give f and g."""
    at = stmt.index("(")
    if at + 1 < len(stmt) and stmt[at + 1] == "*":  # a function returning a function pointer
        inner = stmt[at + 2 :]
        return next((t for t in inner if _ident(t)), "")
    return stmt[at - 1] if at and _ident(stmt[at - 1]) else ""


def _ident(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token))


_ASM = {"__asm__", "asm", "__asm"}


_SECTION = re.compile(r"\.section\s+(\.\w+)")


def _statement(stmt: Sequence[str], defined: list[str], stand_ins: list[str], kinds: dict[str, str]) -> None:
    if any(t in _ASM for t in stmt):
        joined = "".join(_unquote(t) for t in stmt if t.startswith('"'))
        for found in _STAND_IN.finditer(joined):
            sections = _SECTION.findall(joined[: found.start()])
            kinds[found[1]] = sections[-1] if sections else ".text"
            stand_ins.append(found[1])
        return
    if not stmt or stmt[0] in {"typedef", "extern"} or "typedef" in stmt or "extern" in stmt:
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


def _closure(read: Callable[[str], bytes | None], directory: str, version: str, text: bytes) -> list[tuple[str, str]]:
    """The headers `text` includes, through nested includes: (spelled name, blob sha or "") with no path in them."""
    found: dict[tuple[str, str], None] = {}
    seen: set[str] = set()
    queue = [(directory, text)]
    while queue:
        here, data = queue.pop()
        for raw in _INCLUDE.findall(data):
            name = raw.decode(errors="replace")
            resolved = next(
                (
                    p
                    for p in (
                        f"{here}/{name}",
                        f"include/{name}",
                        f"{directory}/{name}",
                        f"asm/{version}/include/{name}",
                    )
                    if read(p) is not None
                ),
                "",
            )
            blob = read(resolved) if resolved else None
            found[name, sha256(blob) if blob is not None else ""] = None
            if blob is not None and resolved not in seen:
                seen.add(resolved)
                queue.append((resolved.rpartition("/")[0], blob))
    return sorted(found)


def _key(config: Config, path: str, text: bytes, version: str, read: Callable[[str], bytes | None]) -> str:
    directory = path.rpartition("/")[0]
    basis = ["sections", sha256(text), macro(version), _cppflags(config), _closure(read, directory, version, text)]
    return sha256(json.dumps(basis, sort_keys=True).encode())


def _preprocess(config: Config, path: str, text: bytes, version: str, extra: Sequence[str]) -> str:
    root = _project(config).root
    directory = path.rpartition("/")[0]
    argv: list[str | Path] = [
        process.tool(config, "cpp"), "-P", "-undef", "-nostdinc", *_cppflags(config), f"-D{macro(version)}", *extra,
        "-Iinclude", f"-I{directory}", f"-Iasm/{version}/include", "-",
    ]  # fmt: skip
    result = process.run("tree.cpp", argv, root, tmp=process.scratch(root), stdin=text)
    if result.exit != 0:
        lines = result.stderr.decode(errors="replace").strip().splitlines()
        raise Refusal(
            Finding("native.exit", reason=lines[-1] if lines else "cpp failed", path=path, versions=(version,))
        )
    return result.stdout.decode(errors="replace")


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
    config: Config, path: str, text: bytes, version: str, read: Callable[[str], bytes | None] | None = None
) -> tuple[list[str], list[str], list[str]]:
    """(C-defined names, stand-in names, NON_MATCHING names) of one file in one version. Never compiles."""
    first, second, third, _ = _parsed(config, path, text, version, read)
    return first, second, third


def _parsed(
    config: Config, path: str, text: bytes, version: str, read: Callable[[str], bytes | None] | None = None
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """`parse` and the linker section of each defined or stand-in name, kept together in one cache entry."""
    key = _key(config, path, text, version, read or _read_from(config))

    def produce() -> bytes:
        kinds: dict[str, str] = {}
        plain = _scan(_preprocess(config, path, text, version, ()), kinds)
        fuzzy = _scan(_preprocess(config, path, text, version, ("-DNON_MATCHING",)))
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
    paths |= {symbols.path(v) for v in _project(config).versions}
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


def _parse_job(item: tuple[Any, ...]) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    config, path, text, version, overlay = item
    disk = _read_from(config)
    return _parsed(config, path, text, version, lambda p: overlay[p] if p in overlay else disk(p))


def _digest(config: Config, blobs: Mapping[str, bytes]) -> str:
    rows = [(p, sha256(b)) for p, b in blobs.items()]
    return sha256(json.dumps([rows, _project(config).digest], sort_keys=True).encode())


def capture(config: Config, overlay: Mapping[str, bytes | None] = {}) -> Tree:
    with effort.stage("tree.capture"):
        blobs = _files(config, overlay)
        return effort.memo(("tree", _digest(config, blobs)), lambda: _build(config, blobs, overlay))


def _build(config: Config, blobs: Mapping[str, bytes], overlay: Mapping[str, bytes | None]) -> Tree:
    project = _project(config)
    versions: tuple[str, ...] = tuple(project.versions)
    tables = {v: symbols.parse(blobs.get(symbols.path(v), b"").decode(errors="replace")) for v in versions}
    sources = [p for p in blobs if p.startswith("src/") and _is_source(p)]
    jobs = [(p, v) for p in sources for v in versions]
    parsed = _parse_all(config, blobs, jobs, overlay)
    owned: dict[str, dict[str, list[str]]] = defaultdict(dict)
    stand: dict[str, dict[str, list[str]]] = defaultdict(dict)
    fuzzy: dict[str, set[str]] = defaultdict(set)
    sections: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for (path, version), (defs, ins, nm, kinds) in zip(jobs, parsed, strict=True):
        table = tables[version]
        names = [n for n in dict.fromkeys([*defs, *ins]) if n in table]
        if names:
            owned[path][version] = sorted(names, key=lambda n, t=table: (_rom(t, n), n))  # type: ignore[misc]
            stand[path][version] = [n for n in ins if n in table]
            sections[path][version] = {n: _section(table[n], kinds.get(n)) for n in names}
            fuzzy[path] |= {n for n in nm if n in table}
    findings = _duplicates(owned, versions)
    modules = tuple(
        _module(config, path, owned[path], stand[path], fuzzy[path], sections[path], tables, findings)
        for path in sorted(owned)
    )
    if findings:
        raise Refusal(*findings)
    return Tree(_digest(config, blobs), modules, tables)


def _parse_all(
    config: Config, blobs: Mapping[str, bytes], jobs: Sequence[tuple[str, str]], overlay: Mapping[str, bytes | None]
) -> list[tuple[list[str], list[str], list[str], dict[str, str]]]:
    """Every (file, version) result: hits of the cache are read here, only the misses are parsed (in the pool)."""
    read = blobs.get
    keys = [_key(config, p, blobs[p], v, read) for p, v in jobs]
    hits = {k: store.get(config, KIND, k) for k in set(keys)}
    todo = [i for i, k in enumerate(keys) if hits[k] is None]
    items = [(config, jobs[i][0], blobs[jobs[i][0]], jobs[i][1], dict(overlay)) for i in todo]
    done = dict(zip(todo, pool.map(config, "tree.parse", _parse_job, items), strict=True)) if items else {}
    out = []
    for i, k in enumerate(keys):
        if i in done:
            out.append(done[i])
        else:
            first, second, third, kinds = json.loads(hits[k] or b"[[],[],[],{}]")
            out.append((first, second, third, kinds))
    return out


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
    return ".text" if fact[3] == "func" else kind or ".data"


def _module(
    config: Config,
    path: str,
    owns: Mapping[str, list[str]],
    stand: Mapping[str, list[str]],
    fuzzy: set[str],
    sections: Mapping[str, Mapping[str, str]],
    tables: Mapping[str, Mapping[str, symbols.Symbol]],
    findings: list[Finding],
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
    )
