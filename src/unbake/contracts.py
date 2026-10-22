"""Frozen records and protocols for unbake (contracts revision 4). Every module imports from here."""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import pickle
import re
import sys
import types
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Protocol


def pickled[T](
    cache: Callable[[str, str, Callable[[], bytes]], bytes], kind: str, key: str, produce: Callable[[], T]
) -> T:
    """produce()'s value kept pickled in a content cache under kind and key."""
    raw = cache(kind, key, lambda: pickle.dumps(produce(), protocol=5))
    try:
        value: T = pickle.loads(raw)
    except Exception:  # an entry another build wrote in a shape this one cannot read is a miss, never a crash
        value = produce()
    return value


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


Json = Mapping[str, Any]  # a document already validated by config.load_resource against its schema


def digest(value: Any) -> str:
    """sha256 hex of canonical JSON (sorted keys, no spaces); dataclasses, paths, bytes and sets are normalised."""

    def norm(item: Any) -> Any:  # JSON writes the rest natively
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        if isinstance(item, Mapping):
            return {str(k): v for k, v in item.items()}
        if isinstance(item, (frozenset, set)):
            return sorted(item, key=lambda v: json.dumps(v, sort_keys=True, default=norm))
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, bytes):
            return sha256(item)
        raise TypeError(f"cannot digest {type(item).__name__}")

    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=norm)
    return sha256(text.encode())


# ---------------------------------------------------------------- code identity
_PLAIN = (str, bytes, int, float, bool, type(None))


def _plain(value: Any) -> str | None:
    """A canonical text for an immutable constant, else None: mutable containers (caches, memos) differ per process."""
    if isinstance(value, _PLAIN):
        return repr(value)
    if isinstance(value, re.Pattern):
        return repr((value.pattern, value.flags))
    if isinstance(value, tuple):
        parts = [_plain(v) for v in value]
        return None if None in parts else f"tuple({','.join(p or '' for p in parts)})"
    if isinstance(value, frozenset):
        parts = [_plain(v) for v in value]
        return None if None in parts else f"frozenset({','.join(sorted(p or '' for p in parts))})"
    return None


@functools.cache
def _assigned(module: str) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Per top-level name of a module: its assigned expression as a syntax tree, and the names that expression reads."""
    path = getattr(sys.modules.get(module), "__file__", None)
    found: dict[str, tuple[str, tuple[str, ...]]] = {}
    if path:
        for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                reads = tuple(sorted({n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}))
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    if isinstance(target, ast.Name):
                        found[target.id] = (ast.dump(node.value), reads)
    return found


def _home(value: Any, name: str, home: str, package: str) -> str:
    """The module that assigns `name`: `home` itself, or for a `from x import NAME` the package module holding it."""
    if name in _assigned(home):
        return home
    for module_name, module in list(sys.modules.items()):
        ours = module_name == package or module_name.startswith(package + ".")
        if ours and getattr(module, name, None) is value and name in _assigned(module_name):
            return module_name
    return home


def _methods() -> dict[str, list[Any]]:
    """Every method of the contracts' classes by name: a call on a record (snapshot.read) names no global to follow."""
    found: dict[str, list[Any]] = {}
    for value in list(vars(sys.modules[__name__]).values()):
        if inspect.isclass(value) and value.__module__ == __name__:
            for name, member in vars(value).items():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                elif isinstance(member, property):
                    member = member.fget
                if isinstance(member, types.FunctionType):
                    found.setdefault(name, []).append(member)
    return found


class _Walk:
    def __init__(self, package: str) -> None:
        self.package = package
        self.hasher = hashlib.sha256()
        self.seen: set[Any] = set()
        self.queue: deque[tuple[Callable[[Any], None], Any]] = deque()
        self.methods = _methods()

    def add(self, *parts: str) -> None:
        for part in parts:
            self.hasher.update(part.encode())
            self.hasher.update(b"\0")

    def ours(self, value: Any) -> bool:
        module = getattr(value, "__module__", None) or ""
        return module == self.package or module.startswith(self.package + ".")

    def plumbing(self, value: Any) -> bool:
        """Code of a module that declares `__plumbing__ = True`: it schedules, times or memoises work and never shapes
        a result, so it is named and not followed (a pool or timing edit leaves every stored result warm)."""
        module = sys.modules.get(getattr(value, "__module__", None) or "")
        if getattr(module, "__plumbing__", False):
            self.add("plumbing", module.__name__, getattr(value, "__qualname__", ""))  # type: ignore[union-attr]
            return True
        return False

    def value(self, value: Any, name: str, home: str = "", declared: bool = False) -> None:
        """A global or attribute the code names: followed when it is code of the package, else its constants."""
        while isinstance(getattr(value, "__wrapped__", None), types.FunctionType):
            value = value.__wrapped__
        if isinstance(value, types.ModuleType):
            self.add("module", value.__name__)
        elif isinstance(value, types.FunctionType) and self.ours(value):
            if not self.plumbing(value):
                self.queue.append((self.function, value))
        elif inspect.isclass(value) and self.ours(value):
            if not self.plumbing(value):
                self.queue.append((self.klass, value))
        elif declared and name in _assigned(_home(value, name, home, self.package)):
            self.data(value, name, home, True)  # a global a module assigns: its expression, not what a run left in it
        elif (text := _plain(value)) is not None:
            self.add("const", name, text)
        else:
            self.data(value, name, home)

    def data(self, value: Any, name: str, home: str, declared: bool = False) -> None:
        """The expression that built the global at import, never what it holds now (memos fill up per process, and
        a global a run reassigns would give every process another identity)."""
        home = _home(value, name, home, self.package)
        text, reads = _assigned(home).get(name, ("", ()))
        self.add("data", name, "global" if declared else type(value).__name__, text)
        if ("data", home, name) not in self.seen:  # the module names that expression reads (HARD reads _ABI)
            self.seen.add(("data", home, name))
            scope = vars(sys.modules[home]) if home in sys.modules else {}
            for read in reads:
                if read in scope and read != name:
                    self.value(scope[read], read, home, True)

    def klass(self, cls: type) -> None:
        if id(cls) in self.seen:
            return
        self.seen.add(id(cls))
        self.add("class", cls.__qualname__, repr(sorted(getattr(cls, "__annotations__", {}).items())))
        for base in cls.__bases__:
            if self.ours(base):
                self.queue.append((self.klass, base))
        for name, member in vars(cls).items():
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            elif isinstance(member, property):
                member = member.fget
            if isinstance(member, types.FunctionType):
                self.queue.append((self.function, member))
            elif not name.startswith("__"):
                self.value(member, name, cls.__module__)

    def function(self, function: Any) -> None:
        code = getattr(function, "__code__", None)
        if code is None:
            self.add(
                "opaque",
                getattr(function, "__module__", None) or "builtins",
                getattr(function, "__qualname__", repr(type(function))),
            )
            return
        if id(function) in self.seen:
            return
        self.seen.add(id(function))
        self.add(
            "function",
            function.__qualname__,
            _plain(function.__defaults__) or "",
            _plain(function.__kwdefaults__) or "",
        )
        for cell in function.__closure__ or ():
            with suppress(ValueError):  # an empty cell
                self.value(cell.cell_contents, "closure")
        self.code(code, function.__globals__)

    def code(self, code: types.CodeType, scope: dict[str, Any]) -> None:
        self.add("code", code.co_code.hex(), *code.co_varnames, *code.co_names)
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                self.code(const, scope)
            else:
                self.add("const", _plain(const) or type(const).__name__)
        for name in code.co_names:
            if name in scope:
                value = scope[name]
                if isinstance(value, types.ModuleType) and (value.__name__ + ".").startswith(self.package + "."):
                    for attribute in code.co_names:
                        if attribute != name and hasattr(value, attribute):
                            self.value(getattr(value, attribute), attribute, value.__name__, True)
                else:
                    self.value(value, name, scope.get("__name__", ""), True)
            for method in self.methods.get(name, ()):
                self.queue.append((self.function, method))


@functools.cache
def code_identity(*functions: Any) -> str:
    """A digest of the code these functions reach: bytecode, constants, names and every package function, class and
    module attribute they name, so an edit anywhere else (line shifts included) leaves it equal. Computed once per
    process. Strings picked by name at run time and adapters picked by config are not seen: pass them in."""
    first = functions[0] if functions else None
    package = (
        getattr(first, "__name__" if isinstance(first, types.ModuleType) else "__module__", None) or ""
    ).partition(".")[0]
    walk = _Walk(package)
    for function in functions:
        if isinstance(function, types.ModuleType):  # every function and class it defines, for code picked by name
            for name, member in sorted(vars(function).items()):
                if getattr(member, "__module__", None) == function.__name__:
                    walk.value(member, name)
        else:
            walk.value(function, "root")
    while walk.queue:
        visit, item = walk.queue.popleft()
        visit(item)
    return walk.hasher.hexdigest()


# ---------------------------------------------------------------- findings and refusals
@dataclass(frozen=True)
class Origin:
    key: str  # dotted config key, e.g. "resources.memory_worker_bytes"
    file: str  # absolute path of the file that set it
    sha256: str  # that file's content digest when it was read
    note: str = ""  # how a value was resolved, e.g. "auto: os.sched_getaffinity -> 12"


@dataclass(frozen=True)
class Finding:
    key: str  # a key listed in resources/data/keys.toml
    reason: str  # one plain sentence
    path: str = ""  # project-relative file
    line: int = 0  # 1-based; 0 when not line-bound
    unit: str = ""  # unit path or member name
    versions: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()  # named missing pieces; nonempty for every *.not_exact key
    origin: Origin | None = None  # config value that caused it
    blocking: bool = True  # False = project debt, reported but never refuses the operation
    action: str = ""  # exact next command or edit
    symptoms: Json = field(default_factory=dict)  # measured compare facts; absent facts are unknown, not zero


class Refusal(Exception):
    def __init__(self, *findings: Finding) -> None:
        if not findings:
            raise TypeError("Refusal needs at least one Finding")
        self.findings = tuple(findings)
        super().__init__("; ".join(f"{f.key}: {f.reason}" for f in findings))

    def __reduce__(self) -> tuple[type[Refusal], tuple[Finding, ...]]:
        return (Refusal, self.findings)  # pool workers return Refusals across processes


# ---------------------------------------------------------------- configuration
@dataclass(frozen=True)
class Host:
    cores: int  # host.toml integer, or "auto" resolved by config to len(os.sched_getaffinity(0)) (Origin.note says so)
    workers: int  # same rule; pool width and make -j both use this one value
    memory_parent_bytes: int
    memory_worker_bytes: int
    cache_dir: Path  # cache.dir: the one content cache every project and worktree on the host shares
    cache_max_bytes: int
    toolchain_root: Path
    # tools, exactly: git make cpp mips_as mips_ld mips_objcopy mips_objdump n64link splat armips
    tools: Mapping[str, Path]
    sdk_catalog: Path | None  # n64sym-style signature JSON, None = no SDK identification
    serial_seconds: float  # budgets.*: a leaf stage this long must use at least serial_cores
    serial_cores: float
    pool_fill: float  # a pool of pool_fanout jobs per worker must keep this share of its workers busy
    pool_fanout: float
    author: tuple[str, str]  # (publish.author_name, publish.author_email) for the repository's first commit
    origins: Mapping[str, Origin]
    digest: str
    move_wait_s: float = 0.0  # resources.move_wait_s: how long a move or setup waits for the structure lock
    slots: int = 0  # host-wide worker budget all heavy work divides: floor(cores * cap), at most explicit workers
    run_slice: str | None = None  # resources.run_slice: the systemd user slice `unbake-run` starts itself in, if any
    cache_keep_commits: int = 20  # cache.keep_commits: how many commits of a clone's HEAD keep their cache entries live


@dataclass(frozen=True)
class Segment:
    name: str
    rom: int  # start; the end is the next row's rom (or the end of the ROM)
    vram: int | None  # None = not given (a bin segment, or splat detects it)
    kind: str  # "code" | "bin" (default "code")


@dataclass(frozen=True)
class VersionFacts:
    sha1: str  # sha1 of roms/baserom.<v>.z64
    size: int  # byte size of roms/baserom.<v>.z64: where the last segment ends
    segments: tuple[Segment, ...]
    meta: Json  # cartridge_id, region and description (README rows); the first two default from the ROM header


@dataclass(frozen=True)
class FileFacts:
    """A [file] row: per-file toolchain, flags added to the defaults and flags removed from them."""

    toolchain: str | None = None
    cflags: tuple[str, ...] = ()
    omit: tuple[str, ...] = ()


@dataclass(frozen=True)
class Project:
    root: Path
    name: str
    title: str
    toolchain: str  # default toolchain id from resources/data/toolchains.toml
    names_from: str
    libraries: tuple[str, ...]
    cppflags: tuple[str, ...]
    cflags: tuple[str, ...]
    asflags: tuple[str, ...]
    readme_map: str | None  # the version whose memory map the README progress block shows
    readme_map_link: str | None  # the image links to this page at the anchor of the version's lower-cased cartridge id
    versions: tuple[str, ...]  # the sorted [version] keys
    version: Mapping[str, VersionFacts]
    files: Mapping[str, FileFacts]  # [file] rows by path or "dir/" prefix
    digest: str
    origins: Mapping[str, Origin] = field(default_factory=dict)
    library_origin: Mapping[str, str] = field(default_factory=dict)  # [library] row path -> where it came from


def macro(version: str) -> str:
    """The preprocessor macro naming a version."""
    return "VERSION_" + version.upper().replace("-", "_")


def file_facts(project: Project, path: str) -> FileFacts:
    """The [file] row for a path: the exact key, else the longest "dir/" prefix, else no facts."""
    if path in project.files:
        return project.files[path]
    prefixes = [k for k in project.files if k.endswith("/") and path.startswith(k)]
    return project.files[max(prefixes, key=len)] if prefixes else FileFacts()


@dataclass(frozen=True)
class Config:
    project: Project
    host: Host
    digest: str


# ---------------------------------------------------------------- versions and layout
@dataclass(frozen=True)
class Placement:
    version: str
    section: str  # ".text" | ".rodata" | ".data" | ".bss"
    rom_start: int  # bss: rom_start == rom_end
    rom_end: int
    vram: int
    bss_size: int = 0  # memory extent; no ROM bytes for .bss

    @property
    def size(self) -> int:
        return self.bss_size if self.section == ".bss" else self.rom_end - self.rom_start


@dataclass(frozen=True)
class Member:
    name: str
    kind: str  # "function" | "data"
    placements: tuple[Placement, ...]  # sorted by (version, section)

    def holders(self) -> tuple[str, ...]:
        return tuple(sorted({p.version for p in self.placements}))

    def reference(self, preferred: str) -> str:
        return preferred if preferred in self.holders() else self.holders()[0]


@dataclass(frozen=True)
class Version:
    id: str
    rom: Path
    rom_sha256: str
    split: str  # project-relative splat yaml, e.g. "versions/de/Game.yaml"
    symbols_file: str  # project-relative symbol_addrs.txt
    symbols: Mapping[str, int]  # name -> vram, from versions/<id>/symbol_addrs.txt
    segments: tuple[tuple[str, int, int, int], ...]  # (name, rom_start, rom_end, vram)
    code: frozenset[str] = frozenset()  # symbol names the generated files show to be code: calls, function labels


@dataclass(frozen=True)
class UnitSpec:
    path: str  # project-relative source file, e.g. "src/code_80200610.c"
    kind: str  # id from resources/data/units.toml: "c" | "asm" | "data" | "resource"
    members: tuple[str, ...]  # contiguous run in ROM, address order
    toolchain: str
    options: Json  # validated per-unit option overrides


@dataclass(frozen=True)
class Move:
    """One structure move: `op` applies `subject` to `target`; `index` is its position in the input."""

    index: int
    op: str
    subject: str
    target: str


@dataclass(frozen=True)
class Snapshot:
    """Immutable view of one project state; overlays are proposed writes not yet installed."""

    config: Config
    commit: str  # git HEAD sha at capture
    files: Any  # the C7 unbake.tree.Tree (Any: tree imports contracts)
    versions: Mapping[str, Version]
    overlays: Mapping[str, bytes | None]  # project-relative path -> proposed bytes (None = delete)
    digest: str
    reads: dict[str, str | None] = field(default_factory=dict, compare=False, repr=False)  # path -> sha256 it read
    tree: bool = False  # source/configuration reads are confined to Git blobs

    def read(self, path: str) -> bytes:
        if path in self.overlays:
            data = self.overlays[path]
            if data is None:
                raise FileNotFoundError(path)
            return data
        if self.tree and (path.startswith(("src/", "include/")) or path.endswith((".toml", ".yaml", ".ld"))):
            raise FileNotFoundError(path)
        try:
            data = (self.config.project.root / path).read_bytes()
        except FileNotFoundError:
            self.reads.setdefault(path, None)
            raise
        self.reads.setdefault(path, sha256(data))
        return data

    def peek(self, path: str) -> bytes | None:
        """read, or None when the file does not exist."""
        try:
            return self.read(path)
        except FileNotFoundError:
            return None


# ---------------------------------------------------------------- native work
@dataclass(frozen=True)
class Recipe:
    toolchain: str
    cppflags: tuple[str, ...]
    cflags: tuple[str, ...]
    asflags: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class SourceView:
    unit: str
    version: str
    key: str  # digest of (source bytes, dependency pins, recipe digest, version macros)
    text: str  # preprocessed text with line markers
    dependencies: tuple[tuple[str, str], ...]  # (project-relative path, sha256), source first
    lines: tuple[tuple[int, str, int], ...]  # (output line, file, source line) from line markers


@dataclass(frozen=True)
class NativeResult:
    argv: tuple[str, ...]
    cwd: str
    exit: int | None
    signal: int | None
    stdout: bytes
    stderr: bytes
    wall_seconds: float
    cpu_seconds: float  # wait4 user+sys of this child and its reaped descendants
    max_rss_bytes: int
    outputs: Mapping[str, Path]  # role -> file the tool wrote ("object", "asm", "linked", ...)


@dataclass(frozen=True)
class Proof:
    unit: str
    member: str
    version: str
    recipe: str  # Recipe.digest
    source_sha256: str
    object_sha256: str
    built_sha256: str  # bytes placed at the member's placements
    target_sha256: str  # ROM bytes at the same placements
    exact: bool
    missing: tuple[str, ...]  # nonempty iff not exact
    score: float  # matched fraction of the member's owned bytes in this version, 0..1; 1.0 iff exact
    symptoms: Json  # symptoms.measure vocabulary; {} when exact


def proof_from(row: Json) -> Proof:  # a Proof as stored in JSON
    return Proof(**{**row, "missing": tuple(row["missing"])})


@dataclass(frozen=True)
class Plan:
    operation: str  # "publish" | "fuzzy" | "layout" | "setup" | "move"
    base: str  # Snapshot.digest it was made from
    writes: Mapping[str, bytes | None]
    affected: tuple[str, ...]  # unit paths whose objects must be reproved
    blocking: tuple[Finding, ...]
    debt: tuple[Finding, ...]
    message: str  # the one-line git commit subject the caller is told to use
    digest: str


@dataclass(frozen=True)
class Receipt:
    operation: str
    files: tuple[str, ...]  # the paths written, uncommitted
    message: str  # the suggested git commit subject
    plan: str  # Plan.digest
    proofs: tuple[Proof, ...]
    debt: int
    invocation: str


# ---------------------------------------------------------------- flow: inbox
@dataclass(frozen=True)
class Submission:
    """One inbox entry (.unbake/inbox/<id>.json + <id>.c). land.submit writes it; only land.drain consumes it."""

    id: str  # digest((member, source_sha256, overrides, operation))
    operation: str  # "publish" | "fuzzy"
    member: str
    function: str | None  # --function given at submit time
    source: str  # project-relative inbox copy, ".unbake/inbox/<id>.c"
    source_sha256: str
    overrides: Json  # {"toolchain"?: str, "add": [str], "omit": [str]}
    base: str  # git HEAD the proofs were measured at
    proofs: tuple[Proof, ...]
    origin: str  # command that submitted: "submit" | "compare" | "check"
    note: str  # free-text note given at submit time ("" when none)
    invocation: str
    extras: Json  # change set: project path -> inbox copy, landed with the source; {} for one file
    extra_bases: Mapping[str, str | None]  # measured extra inputs: inbox copy, or absent at admission
    withheld: tuple[str, ...]  # versions the source is not exact in, landed withheld; () when exact in all
    session: str = ""  # RECIPE_SESSION of the submitting command ("" when unnamed): who owns the entry
    entered: float = 0.0  # epoch seconds at admission; an old entry of a dead or silent session goes to inbox/attic


# ---------------------------------------------------------------- observability
@dataclass(frozen=True)
class StageRecord:
    invocation: str
    span: str
    parent: str | None
    path: tuple[str, ...]
    kind: str  # "command" | "stage" | "pool" | "job" | "native" | "own"
    status: str  # "ok" | "refused" | "failed" | "interrupted"
    execution: str  # "serial" | "parallel"
    start_ns: int
    wall_seconds: float
    parent_cpu_seconds: float
    worker_cpu_seconds: float
    native_cpu_seconds: float
    cores: float
    items: int
    jobs: int
    workers_used: int
    workers_admitted: int
    parent_rss_peak_bytes: int
    worker_rss_peak_bytes: int
    memory_limits: Mapping[str, Origin] = field(default_factory=dict)
    memory_limit_values: Mapping[str, int] = field(default_factory=dict)
    cache: Mapping[str, tuple[int, int]] = field(default_factory=dict)  # cache kind -> (hits, misses)
    findings: tuple[Finding, ...] = ()
    wait_seconds: float = 0.0  # time this stage itself blocked on a lock
    dispatch_seconds: float = 0.0  # pool: wall the parent spent outside waiting on workers
    busy_seconds: float = 0.0  # pool: wall the workers spent inside jobs, summed
    spans: int = 1  # how many worker spans of one pool this record folds, by path
    slot_seconds: float = 0.0  # pool: host slots held times the seconds held (the share peers left it), summed
    slot_wait_seconds: float = 0.0  # command: seconds spent waiting for host worker slots held by other commands
    lock_wait_seconds: float = 0.0  # command: seconds spent waiting for project locks held by other commands


# ---------------------------------------------------------------- adapter protocols
class Assembler(Protocol):
    def assemble(self, source: Path, recipe: Recipe, include: Sequence[Path], out: Path) -> NativeResult: ...


class Linker(Protocol):
    def link(self, obj: Path, sections: Sequence[Placement], symbols: Path, work: Path, trim: bool) -> NativeResult: ...


class Span(Protocol):
    def add(self, *, items: int = 0, findings: Sequence[Finding] = ()) -> None: ...
