"""Configuration and resource loading: validated host, project and tool data."""

from __future__ import annotations

import json
import math
import os
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from functools import cache, partial
from importlib import resources
from pathlib import Path
from typing import Any

import psutil
from jsonschema import Draft202012Validator

from unbake import process
from unbake.contracts import (
    Config,
    FileFacts,
    Finding,
    Host,
    Json,
    Origin,
    Project,
    Refusal,
    Segment,
    VersionFacts,
    digest,
    pickled,
    sha1,
    sha256,
)

SCHEMA_OF = {
    "symbol-names.toml": "symbol-names",
    "keys.toml": "keys",
    "rules.toml": "rules",
    "units.toml": "units",
    "commands.toml": "commands",
    "toolchains.toml": "toolchains",
    "flow.toml": "flow",
    "rom.toml": "rom",
    "repo.toml": "repo",
}


@cache
def _schema(name: str) -> Json:
    node = resources.files("unbake.resources") / "schemas" / f"{name}.schema.json"
    if not node.is_file():
        raise Refusal(Finding("config.resource", f"schema {name} is not packaged", path=f"schemas/{name}.schema.json"))
    schema: Json = json.loads(node.read_text(encoding="utf-8"))
    return schema


Cached = Callable[[str, str, Callable[[], bytes]], bytes]  # store.content(config).cached


def toml(schema: str, data: bytes, file: str, key: str = "config.schema", cache: Cached | None = None) -> Json:
    """The schema-valid document of TOML bytes (non-TOML is a finding under `key`); with a project, read back from its
    content cache once valid."""

    def read() -> Json:
        try:
            document = tomllib.loads(data.decode())
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
            raise Refusal(Finding(key, str(error), path=file)) from error
        validate(schema, document, file)
        return document

    if cache is None:
        return read()
    return pickled(cache, "toml", digest((schema, data, _schema(schema))), read)


def validate(schema: str, value: Json, file: str) -> None:
    """Raise one Refusal listing every schema error of value, sorted by path (another schema version is one finding)."""
    want = _schema(schema).get("properties", {}).get("schema", {}).get("const")
    have = value.get("schema") if isinstance(value, dict) else None
    if want is not None and have is not None and have != want:
        raise Refusal(
            Finding(
                "config.schema",
                f"{file} is schema {have}, but this unbake reads schema {want}",
                path=f"{file}:schema",
                action=f"re-create {file} as schema {want}",
            )
        )
    errors = sorted(
        Draft202012Validator(_schema(schema)).iter_errors(value),
        key=lambda e: ([str(p) for p in e.absolute_path], e.message),
    )
    if errors:
        raise Refusal(
            *(
                Finding("config.schema", e.message, path=f"{file}:{'.'.join(map(str, e.absolute_path)) or '<root>'}")
                for e in errors
            )
        )


@cache
def load_resource(name: str) -> Json:
    """Load one packaged data file, validated against its schema."""
    if name not in SCHEMA_OF:
        raise Refusal(Finding("config.resource", f"{name} is not a packaged data file", path=name))
    node = resources.files("unbake.resources") / "data" / name
    if not node.is_file():
        raise Refusal(Finding("config.resource", f"{name} is missing from the installed package", path=name))
    document = tomllib.loads(node.read_text(encoding="utf-8"))
    validate(SCHEMA_OF[name], document, name)
    return document


def _table(name: str, *keys: str) -> Json:
    """The table at the key path inside one packaged data file."""
    node: Json = load_resource(name)
    for key in keys:
        node = node[key]
    return node


def kinds() -> Json:
    """The unit kinds of units.toml by id."""
    return _table("units.toml", "kind")


def layout(project: Project) -> Json:
    """The project's layout: repo.toml [layout] with each key its config.toml [layout] spells replaced."""
    return {**load_resource("repo.toml")["layout"], **getattr(project, "layout", {})}


def cheat(project: Project) -> Json:
    """The no-cheat rules: repo.toml [cheat] with each key its config.toml [cheat] spells replaced."""
    return {**load_resource("repo.toml")["cheat"], **getattr(project, "cheat", {})}


def layout_at(root: Path) -> Json:
    """`layout` for a caller holding only the project root (the sibling toolkits): its config.toml [layout] over
    repo.toml's, validated like the project's own (the packaged layout where the root has no config.toml)."""
    path = root / "config.toml"
    try:
        declared = tomllib.loads(path.read_text(encoding="utf-8")).get("layout", {}) if path.is_file() else {}
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise Refusal(Finding("config.schema", f"not valid TOML: {error}", path=str(path))) from error
    errors = sorted(
        e.message for e in Draft202012Validator(_schema("project")["properties"]["layout"]).iter_errors(declared)
    )
    if errors:
        raise Refusal(*(Finding("config.schema", m, path=f"{path}:layout") for m in errors))
    return {**load_resource("repo.toml")["layout"], **declared}


def library(project: Project, path: str) -> bool:
    """Whether the project-relative path is inside one of the configured library directories."""
    return any(Path(path).is_relative_to(folder) for folder in project.libraries)


@cache
def _suffixes(role: str) -> tuple[str, ...]:
    return tuple(load_resource("units.toml")["files"][role])


def is_assembly(path: str) -> bool:
    """Whether a source path uses the assembly suffix of the unit contract."""
    return path.lower().endswith(kind("asm")["suffix"])


_ROOTS: dict[str, tuple[str, ...]] = {"source_roots": ("src",), "header_roots": ("include",)}  # the loaded project's


def source_roots() -> tuple[str, ...]:
    """The folders holding compiled sources (layout `source_roots`, default src), as of the last project loaded."""
    return _ROOTS["source_roots"]


def header_roots() -> tuple[str, ...]:
    """The folders includes are written against (layout `header_roots`, default include), as of the last project loaded."""
    return _ROOTS["header_roots"]


def under_source(path: str) -> bool:
    """Whether a project-relative path lies under a source root."""
    return path.startswith(tuple(r + "/" for r in source_roots()))


def under_tree(path: str) -> bool:
    """Whether a project-relative path lies under a source or a header root."""
    return path.startswith(tuple(r + "/" for r in (*source_roots(), *header_roots())))


def is_source(path: str) -> bool:
    """Whether a path is C source or source-tree assembly, with case-insensitive suffixes."""
    return path.lower().endswith(_suffixes("source")) or (under_source(path) and is_assembly(path))


def is_header(path: str) -> bool:
    """Whether a path is a header (units.toml [files] header, any case)."""
    return path.lower().endswith(_suffixes("header"))


def is_text(path: str) -> bool:
    """A source or a header: text the C tools read."""
    return is_source(path) or is_header(path)


def source_stem(path: str) -> str:
    """A source path without its source suffix: src/AI/ENEMIES/TURRETAI.CPP -> src/AI/ENEMIES/TURRETAI."""
    return str(Path(path).with_suffix("")) if is_source(path) else path


def kind(name: str) -> Json:
    return _table("units.toml", "kind", name)


def sections() -> Json:
    """The linker sections of units.toml by id."""
    return _table("units.toml", "section")


def toolchains() -> Json:
    """The toolchains of toolchains.toml by id."""
    return _table("toolchains.toml", "toolchain")


def toolchain(name: str) -> Json:
    return _table("toolchains.toml", "toolchain", name)


def fuzzy() -> Json:
    """The fuzzy-landing settings of flow.toml."""
    return _table("flow.toml", "fuzzy")


@cache
def template(name: str) -> str:
    """Text of one packaged template; name may contain a subdirectory such as sdk/."""
    node = resources.files("unbake.resources") / "templates" / name
    if not node.is_file():
        raise Refusal(Finding("config.resource", f"template {name} is not packaged", path=f"templates/{name}"))
    return node.read_text(encoding="utf-8")


def sentence(key: str) -> str:
    return str(_table("keys.toml", "key", key)["sentence"])


def _flatten(node: Any, prefix: str, out: dict[str, bool]) -> None:
    """Dotted key -> is_leaf for tables, arrays-of-tables rows and values."""
    if isinstance(node, dict):
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else k
            tables = isinstance(v, dict) or (isinstance(v, list) and any(isinstance(i, dict) for i in v))
            out[key] = not tables
            _flatten(v, key, out)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict):
                _flatten(item, prefix, out)


def _read(
    path: Path, file: str, reader: Callable[[Path], bytes] = Path.read_bytes
) -> tuple[dict[str, Any], str, dict[str, Origin]]:
    try:
        raw = reader(path)
    except FileNotFoundError as error:
        raise Refusal(
            Finding("config.missing", f"{path} does not exist", path=str(path), action=f"create {path}")
        ) from error
    sha = sha256(raw)
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise Refusal(Finding("config.schema", f"not valid TOML: {error}", path=f"{file}:<root>")) from error
    validate(file, document, file)
    keys: dict[str, bool] = {}
    _flatten(document, "", keys)
    origins = {k: Origin(k, str(path), sha) for k, leaf in keys.items() if leaf}
    return document, sha, origins


def _bad(file: str, key: str, reason: str, origins: dict[str, Origin]) -> Finding:
    return Finding("config.file", reason, path=f"{file}:{key}", origin=origins.get(key))


def _cgroup_memory_max() -> str:
    file = Path("/sys/fs/cgroup/memory.max")
    return file.read_text(encoding="utf-8").strip() if file.is_file() else "max"


def _usable_memory() -> int:
    limit = _cgroup_memory_max()
    return int(limit) if limit.isdigit() else psutil.virtual_memory().total


DEFAULT_HOST = Path("~/.config/unbake/unbake.toml")
NO_HOST = (
    "no host config: pass --host, set UNBAKE_HOST, or create ~/.config/unbake/unbake.toml "
    "(a seat's `decomp --profile P unbake|recipe ...` passes the profile's tools.host itself)"
)


def default_host() -> Path | None:
    """The host config a call names nowhere: the user's ~/.config/unbake/unbake.toml when it exists."""
    path = DEFAULT_HOST.expanduser()
    return path if path.is_file() else None


def load_host(path: Path) -> Host:
    document, _, origins = _read(path, "host")
    problems: list[Finding] = []
    tools: dict[str, Path] = {}
    for name, text in document["tools"].items():
        tool = Path(text)
        tools[name] = tool
        key = f"tools.{name}"
        if not tool.is_absolute():
            problems.append(_bad("host", key, f"{text} is not an absolute path", origins))
        elif not tool.is_file() or not os.access(tool, os.X_OK):
            problems.append(_bad("host", key, f"{text} is not an executable file", origins))
    if not Path(document["cache"]["dir"]).is_absolute():
        problems.append(_bad("host", "cache.dir", f"{document['cache']['dir']} is not an absolute path", origins))
    if not Path(document["toolchains"]["root"]).is_absolute():
        text = document["toolchains"]["root"]
        problems.append(_bad("host", "toolchains.root", f"{text} is not an absolute path", origins))
    catalog: Path | None = None
    if "sdk" in document:
        catalog = Path(document["sdk"]["catalog"])
        if not catalog.is_file():
            problems.append(_bad("host", "sdk.catalog", f"{catalog} does not exist", origins))
        else:
            try:
                validate("sdk", json.loads(catalog.read_text(encoding="utf-8")), f"sdk:{catalog}")
            except (Refusal, ValueError) as error:
                reason = "; ".join(f.reason for f in error.findings) if isinstance(error, Refusal) else str(error)
                problems.append(_bad("host", "sdk.catalog", f"{catalog} is not a valid catalog: {reason}", origins))
    res, budgets, publish = document["resources"], document["budgets"], document["publish"]
    counts: dict[str, int] = {}
    for name in ("cores", "workers"):
        key, value = f"resources.{name}", res[name]
        if value == "auto":
            counts[name] = len(os.sched_getaffinity(0))
            note = f"auto: os.sched_getaffinity -> {counts[name]}"
            if name == "workers":
                # The shared pool divides capacity between commands. Subtracting system load here counts
                # those commands twice and freezes a busy host at one worker for the entire invocation.
                affinity = counts[name]
                counts[name] = min(affinity, counts["cores"])
                note = f"auto: min(affinity {affinity}, cores {counts['cores']}) -> {counts[name]}"
                if "cap" in res:  # a cap applies only when the host file sets one
                    capacity, cap = counts[name], res["cap"]
                    counts[name] = max(1, min(capacity, math.floor(counts["cores"] * cap)))
                    note = f"auto: min({capacity} by capacity, cap {cap}) -> {counts[name]}"
                by_memory = max(1, (_usable_memory() - res["memory_parent_bytes"]) // res["memory_worker_bytes"])
                if by_memory < counts[name]:
                    note = f"auto: min({counts[name]} by capacity, {by_memory} by memory) -> {by_memory}"
                    counts[name] = by_memory
            old = origins[key]
            origins[key] = Origin(key, old.file, old.sha256, note=note)
        else:
            counts[name] = value
    if counts["workers"] > counts["cores"]:
        problems.append(_bad("host", "resources.workers", "workers exceeds cores", origins))
    if problems:
        raise Refusal(*problems)
    return Host(
        cores=counts["cores"],
        workers=counts["workers"],
        memory_parent_bytes=res["memory_parent_bytes"],
        memory_worker_bytes=res["memory_worker_bytes"],
        cache_dir=Path(document["cache"]["dir"]),
        cache_max_bytes=document["cache"]["max_bytes"],
        cache_keep_commits=document["cache"].get("keep_commits", 20),
        cache_gc_bytes=document["cache"].get("gc_bytes", Host.cache_gc_bytes),
        toolchain_root=Path(document["toolchains"]["root"]),
        tools=tools,
        sdk_catalog=catalog,
        **{k: float(budgets[k]) for k in ("serial_seconds", "serial_cores", "pool_fill", "pool_fanout")},
        author=(publish["author_name"], publish["author_email"]),
        origins=origins,
        digest=digest(document),
        move_wait_s=float(res.get("move_wait_s", 600)),
        run_slice=res.get("run_slice"),
        # the pool never forks more than the workers, and a host file's cap bounds it further
        slots=max(
            1, min(counts["workers"], math.floor(counts["cores"] * res["cap"])) if "cap" in res else counts["workers"]
        ),
    )


def _header(raw: bytes) -> tuple[str, str]:
    """(cartridge id, region) of a ROM header: bytes 0x3B-0x3E are the game code and 0x3F the version."""
    code, country, revision = raw[0x3B:0x3F].decode("ascii", "replace"), chr(raw[0x3E]), raw[0x3F]
    return f"NUS-{code}-{revision}", _table("rom.toml", "country").get(country, f"country {country}")


_VERIFIED: dict[tuple[str, int, int], str] = {}


def rom(project: Project, vid: str) -> Path:
    """The baserom of a version, once its bytes are known to be the configured ones: a command that uses ROM bytes
    asks here and a missing or different ROM refuses by name; a command that never does runs without ROMs."""
    path = project.root / "roms" / f"baserom.{vid}.z64"
    key = f"version.{vid}.sha1"
    try:
        stat = path.stat()
        size = project.version[vid].size
        if stat.st_size != size:
            raise Refusal(
                Finding(
                    "config.file", f"configured {size} bytes, rom is {stat.st_size}", path=f"project:version.{vid}.size"
                )
            )
        got = _VERIFIED.get((str(path), stat.st_mtime_ns, stat.st_size)) or sha1(path.read_bytes())
    except FileNotFoundError:
        raise Refusal(Finding("config.file", f"roms/baserom.{vid}.z64 does not exist", path=f"project:{key}")) from None
    _VERIFIED[(str(path), stat.st_mtime_ns, stat.st_size)] = got
    if got != project.version[vid].sha1:
        raise Refusal(
            Finding("config.file", f"configured {project.version[vid].sha1}, rom is {got}", path=f"project:{key}")
        )
    return path


def rom_meta(project: Project, vid: str) -> Json:
    """cartridge_id and region from the ROM header unless the version row spells them, then the row's description."""
    meta = dict(project.version[vid].meta)
    if "cartridge_id" not in meta or "region" not in meta:
        header = rom(project, vid).read_bytes()[:0x40]
        meta = dict(zip(("cartridge_id", "region"), _header(header), strict=True)) | meta
    return meta


def _tracked(root: Path) -> set[str]:
    return process.tracked(root)


def _folder(path: str, slash: bool = True, roots: Sequence[str] = ("src",)) -> bool:
    """Whether a row path is a source directory under a source root (src/ by default), ending in / when `slash`."""
    parts = Path(path).parts
    ok = not (Path(path).is_absolute() or ".." in parts) and any(
        parts[: len(top)] == top and len(parts) > len(top) for top in (Path(r).parts for r in roots)
    )
    return ok and (path.endswith("/") or not slash)


def _above(files: Mapping[str, FileFacts], path: str) -> FileFacts | None:
    """The row a path would get without its own row: the longest directory row above it."""
    folders = [k for k in files if k.endswith("/") and path.startswith(k) and k != path]
    return files[max(folders, key=len)] if folders else None


def dead_paths(files: Mapping[str, FileFacts], default: str, keep: Iterable[str] = ()) -> list[str]:
    """Rows that give their path what the row above it (else the project default) gives already: an exception that
    excepts nothing (rows in `keep` stay: a library row marks a boundary). A row's toolchain falls back to the
    project's, and only the deepest row applies."""

    def facts(row: FileFacts | None) -> tuple[str, frozenset[str], frozenset[str], bool]:
        return (
            (row.toolchain or default, frozenset(row.cflags), frozenset(row.omit), row.hilo_nops)
            if row
            else (default, frozenset(), frozenset(), False)
        )

    return sorted(p for p, row in files.items() if p not in set(keep) and facts(row) == facts(_above(files, p)))


def dead_rows(project: Project) -> list[str]:
    """The [file], [dir] and [library] rows of a project that equal the row above them (see `_dead`). Loading refuses
    only a row that equals a directory row above it; one that equals the project default is project debt."""
    return dead_paths(project.files, project.toolchain, project.libraries)


def load_project(
    root: Path, reader: Callable[[str], bytes] | None = None, tracked: Callable[[], Iterable[str]] | None = None
) -> Project:
    read = reader or (lambda path: (root / path).read_bytes())
    document, _, origins = _read(root / "config.toml", "project", lambda path: read(path.name))
    table = document["version"]
    versions = tuple(sorted(table))
    names_from = document.get("names_from", versions[0])
    problems: list[Finding] = []
    bad = partial(_bad, "project", origins=origins)
    layout_rows = {**load_resource("repo.toml")["layout"], **document.get("layout", {})}
    roots = tuple(layout_rows["source_roots"])
    _ROOTS["source_roots"], _ROOTS["header_roots"] = roots, tuple(layout_rows["header_roots"])
    if names_from not in versions:
        problems.append(bad("names_from", f"{names_from} is not a [version] table"))
    readme_map = document.get("readme_map")
    if readme_map is not None and readme_map not in versions:
        problems.append(bad("readme_map", f"{readme_map} is not a [version] table"))
    if document["toolchain"] not in load_resource("toolchains.toml")["toolchain"]:
        problems.append(bad("toolchain", f"{document['toolchain']} is not in toolchains.toml"))
    spelled_map = document.get("module_map")
    if spelled_map is not None and (Path(spelled_map).is_absolute() or ".." in Path(spelled_map).parts):
        problems.append(bad("module_map", f"{spelled_map} must be a path inside the repository"))
    for folder in document.get("libraries", ()):
        if not _folder(folder, False, roots):
            problems.append(bad("libraries", f"{folder} must be a source subdirectory"))
    version: dict[str, VersionFacts] = {}
    for vid, row in table.items():
        segments = tuple(
            Segment(s["name"], s["rom"], s.get("vram"), s.get("kind", "code"), s.get("address_space", "main"))
            for s in row.get("segments", ())
        ) or (Segment("main", 0x1000, None, "code"),)
        for segment in segments:
            if segment.address_space != "main" and (segment.kind != "code" or segment.vram is None):
                problems.append(
                    bad(f"version.{vid}.segments", "a separate address space needs code and an explicit vram")
                )
        meta = {k: row[k] for k in ("cartridge_id", "region", "description") if k in row}
        version[vid] = VersionFacts(row["sha1"], row["size"], segments, meta)
    files: dict[str, FileFacts] = {}
    spelled: dict[str, str] = {}  # row path -> the table it was written in
    for kind in ("file", "dir", "library"):
        for path, row in document.get(kind, {}).items():
            facts = FileFacts(
                row.get("toolchain"),
                tuple(row.get("cflags", ())),
                tuple(row.get("omit", ())),
                bool(row.get("hilo_nops")),
            )
            if path in files:
                problems.append(bad(f"{kind}.{path}", f"{path} is also a [{spelled[path]}] row"))
            elif kind != "file" and not _folder(path, True, roots):
                problems.append(bad(f"{kind}.{path}", f"[{kind}] key {path} must be a source directory ending in /"))
            else:
                files[path], spelled[path] = facts, kind
    libraries = [*document.get("libraries", ()), *(p for p, k in spelled.items() if k == "library")]
    for folder in {f for f in libraries if libraries.count(f) > 1}:
        problems.append(bad("libraries", f"{folder} is a library twice"))
    for path, row in files.items():
        if row.toolchain is not None and row.toolchain not in load_resource("toolchains.toml")["toolchain"]:
            problems.append(bad(f"{spelled[path]}.{path}.toolchain", f"{row.toolchain} is not in toolchains.toml"))
    if files:
        have = set((tracked or (lambda: _tracked(root)))())
        for path in files:
            if not any(h.startswith(path) for h in have) if path.endswith("/") else path not in have:
                problems.append(
                    Finding(
                        "config.file_unknown",
                        f"[{spelled[path]}] key {path} matches no tracked file",
                        path=f"project:{spelled[path]}.{path}",
                        origin=origins.get(f"{spelled[path]}.{path}"),
                    )
                )
    boundaries = [p for p, k in spelled.items() if k == "library"]
    for path in (p for p in dead_paths(files, document["toolchain"], boundaries) if _above(files, p) is not None):
        problems.append(
            Finding(
                "config.dead_row",
                f"[{spelled[path]}] row {path} gives its files what the row above it gives already",
                path=f"project:{spelled[path]}.{path}",
                origin=origins.get(f"{spelled[path]}.{path}.toolchain") or origins.get(f"{spelled[path]}.{path}"),
                action=f'delete [{spelled[path]}."{path}"]',
            )
        )
    if problems:
        raise Refusal(*problems)
    return Project(
        root=root,
        name=document["name"],
        title=document.get("title", document["name"]),
        toolchain=document["toolchain"],
        names_from=names_from,
        libraries=tuple(libraries),
        cppflags=tuple(document.get("cppflags", ())),
        cflags=tuple(document.get("cflags", ())),
        asflags=tuple(document.get("asflags", ())),
        readme_map=readme_map,
        readme_map_link=document.get("readme_map_link"),
        readme_map_repo=document.get("readme_map_repo"),
        versions=versions,
        version=version,
        files=files,
        digest=digest(document),
        origins=origins,
        library_origin={p: o for p, o in ((p, r.get("origin")) for p, r in document.get("library", {}).items()) if o},
        non_matching=(document["non_matching"].get("define", "NON_MATCHING") if "non_matching" in document else None),
        module_map=spelled_map,
        layout=document.get("layout", {}),
        cheat=document.get("cheat", {}),
    )


def load(root: Path, host: Path) -> Config:
    project_config, host_config = load_project(root), load_host(host)
    with suppress(OSError):  # a read-only tree: commands that only read still run
        process.state(root)  # every tool writes under it; it ignores itself before the first write
    return Config(project_config, host_config, digest((project_config.digest, host_config.digest)))
