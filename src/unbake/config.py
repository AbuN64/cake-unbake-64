"""Configuration and resource loading: validated host, project and tool data."""

from __future__ import annotations

import json
import math
import os
import tomllib
from collections.abc import Callable
from functools import cache, partial
from importlib import resources
from pathlib import Path
from typing import Any

import psutil
from jsonschema import Draft202012Validator

from unbake.contracts import (
    Config,
    Finding,
    Host,
    Json,
    Origin,
    Project,
    Refusal,
    Resident,
    VersionFiles,
    digest,
    pickled,
    sha1,
    sha256,
)

SCHEMA_OF = {
    "keys.toml": "keys",
    "rules.toml": "rules",
    "units.toml": "units",
    "commands.toml": "commands",
    "toolchains.toml": "toolchains",
    "flow.toml": "flow",
    "rom.toml": "rom",
    "repo.toml": "repo",
    "boards.toml": "boards",
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


def library(project: Project, path: str) -> bool:
    """Whether the project-relative path is inside one of the configured library directories."""
    return any(Path(path).is_relative_to(folder) for folder in project.library_dirs)


@cache
def _suffixes(role: str) -> tuple[str, ...]:
    return tuple(load_resource("units.toml")["files"][role])


def layout_rows(project: Project) -> Json:
    """repo.toml [layout] with the project's own [layout] rows over it: where its module sources and headers live and
    which headers are the shared base (a project may keep its evidence's style, e.g. src/{subsystem}, upper case)."""
    return {**load_resource("repo.toml")["layout"], **project.layout}


def is_source(path: str) -> bool:
    """Whether a path is a C source (units.toml [files] source, any case: src/AI/ENEMIES/TURRETAI.CPP is one)."""
    return path.lower().endswith(_suffixes("source"))


def is_header(path: str) -> bool:
    """Whether a path is a header (units.toml [files] header, any case)."""
    return path.lower().endswith(_suffixes("header"))


def is_text(path: str) -> bool:
    """A source or a header: text the C tools read."""
    return is_source(path) or is_header(path)


def source_stem(path: str) -> str:
    """A source path without its source suffix: src/AI/ENEMIES/TURRETAI.CPP -> src/AI/ENEMIES/TURRETAI."""
    return path[: -len(next(s for s in _suffixes("source") if path.lower().endswith(s)))] if is_source(path) else path


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
                cap, load = res.get("cap", 0.75), int(os.getloadavg()[0])
                counts[name] = max(1, math.floor(counts[name] * cap) - load)  # leave the host its share and busy cores
                note = f"auto: max(1, floor({counts['cores']} cores * {cap}) - load {load}) -> {counts[name]}"
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
        toolchain_root=Path(document["toolchains"]["root"]),
        tools=tools,
        sdk_catalog=catalog,
        **{k: float(budgets[k]) for k in ("serial_seconds", "serial_cores", "pool_fill", "pool_fanout")},
        author=(publish["author_name"], publish["author_email"]),
        origins=origins,
        digest=digest(document),
        move_wait_s=float(res.get("move_wait_s", 600)),
        # the pool never forks more than a host file's own workers; auto workers (sized by this moment's load) leave
        # the slots at the cap, which every command shares
        slots=max(
            1,
            min(
                counts["workers"] if res["workers"] != "auto" else counts["cores"],
                math.floor(counts["cores"] * res.get("cap", 0.75)),
            ),
        ),
    )


def version_files(table: Json) -> dict[str, VersionFiles]:
    """Validated version config rows to the sole per-version path records."""
    return {
        v: VersionFiles(
            row["baserom"],
            row["baserom_sha1"],
            row["split"],
            row["symbols"],
            {k: row[k] for k in ("cartridge_id", "region", "description") if k in row},
        )
        for v, row in table.items()
    }


def load_project(root: Path, reader: Callable[[str], bytes] | None = None) -> Project:
    read = reader or (lambda path: (root / path).read_bytes())
    document, _, origins = _read(root / "config.toml", "project", lambda path: read(path.name))
    info, table = document["project"], document["version"]
    versions = tuple(info["versions"])
    problems: list[Finding] = []
    bad = partial(_bad, "project", origins=origins)
    if info["names_from"] not in versions:
        problems.append(bad("project.names_from", f"{info['names_from']} is not in project.versions"))
    if info.get("readme_map", info["names_from"]) not in versions:
        problems.append(bad("project.readme_map", f"{info['readme_map']} is not in project.versions"))
    if set(table) != set(versions):
        problems.append(bad("version", f"[version] keys {sorted(table)} differ from versions"))
    extra = sorted(set(document["resident"]) - set(versions))
    if extra:
        problems.append(bad("resident", f"[resident] names unknown versions {extra}"))
    if info["toolchain"] not in load_resource("toolchains.toml")["toolchain"]:
        problems.append(bad("project.toolchain", f"{info['toolchain']} is not in toolchains.toml"))
    for folder in info.get("library_dirs", ()):
        path = Path(folder)
        if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != "src":
            problems.append(bad("project.library_dirs", f"{folder} must be a source subdirectory"))
    for vid, row in table.items():
        for field in ("baserom", "split", "symbols"):
            try:
                raw = read(row[field])
            except FileNotFoundError:
                problems.append(bad(f"version.{vid}.{field}", f"{row[field]} does not exist"))
                continue
            if field == "baserom" and (got := sha1(raw)) != row["baserom_sha1"]:
                problems.append(bad(f"version.{vid}.baserom_sha1", f"configured {row['baserom_sha1']}, rom is {got}"))
    if problems:
        raise Refusal(*problems)
    return Project(
        root=root,
        id=info["id"],
        name=info["name"],
        title=info["title"],
        versions=versions,
        names_from=info["names_from"],
        toolchain=info["toolchain"],
        build=document["build"],
        version_files=version_files(table),
        version_macros={v: tuple(f"-D{m}" for m in row["macros"]) for v, row in table.items()},
        resident={v: tuple(Resident(**r) for r in document["resident"].get(v, ())) for v in versions},
        origins=origins,
        digest=digest(document),
        library_dirs=tuple(info.get("library_dirs", ())),
        version_differences=tuple(info.get("version_differences", ())),
        readme_map=info.get("readme_map"),
        readme_map_link=info.get("readme_map_link"),
        evidence_objects=tuple(document.get("evidence", {}).get("object", ())),
        lattice=tuple(document.get("flags", {}).get("lattice", ())),
        layout=dict(document.get("layout", {})),
    )


def load(root: Path, host: Path) -> Config:
    project_config, host_config = load_project(root), load_host(host)
    return Config(project_config, host_config, digest((project_config.digest, host_config.digest)))
