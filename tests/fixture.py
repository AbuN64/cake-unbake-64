"""Test support: schema-3 projects, host files and small native-shaped values. Tests `import fixture`."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from string import Template
from types import MappingProxyType
from typing import Any

from unbake.contracts import (
    Config,
    Host,
    Json,
    NativeResult,
    Origin,
    Project,
    Proof,
    Submission,
    VersionFiles,
    digest,
    sha1,
    sha256,
)

VRAM_BASE, ROM_BASE, PAGE = 0x80000400, 0x1000, 0x1000
TOOLS = (*("git", "make", "cpp", "mips_as", "mips_ld", "mips_objcopy", "mips_objdump"), *("n64link", "splat", "armips"))
Placed = Mapping[str, Mapping[str, tuple[int, bytes]]]  # member name -> version id -> (rom offset, bytes)


def _value(item: Any) -> str:
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, (int, float)):
        return repr(item)
    if isinstance(item, str):
        return json.dumps(item)
    if isinstance(item, Mapping):
        return "{" + ", ".join(f"{k} = {_value(v)}" for k, v in item.items()) + "}"
    return "[" + ", ".join(_value(v) for v in item) + "]"


def _table(name: str, body: Mapping[str, Any]) -> str:
    return f"[{name}]\n" + "".join(f"{k} = {_value(v)}\n" for k, v in body.items()) + "\n"


def _rows(name: str, rows: Sequence[Json]) -> str:
    return "".join(f"[[{name}]]\n" + "".join(f"{k} = {_value(v)}\n" for k, v in row.items()) + "\n" for row in rows)


def _place(items: Placed, kind: str, version: str) -> list[tuple[int, str, str, bytes]]:
    out = [(rom, kind, name, blob) for name, by in items.items() for v, (rom, blob) in by.items() if v == version]
    if any(rom < ROM_BASE for rom, *_ in out):
        raise ValueError(f"member offsets in {version} must be at least 0x{ROM_BASE:X}")
    return out


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)


def _git(root: Path, *args: str) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid"}
    env |= {"GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    argv = ["git", "-C", str(root), "-c", "commit.gpgsign=false", *args]
    subprocess.run(argv, check=True, capture_output=True, env=env)


def project(
    tmp: Path,
    *,
    versions: Sequence[str] = ("a", "b"),
    names_from: str = "a",
    functions: Placed,
    data: Placed = MappingProxyType({}),
    units: Sequence[Json] = (),
    groups: Sequence[Json] | None = None,
    headers: Mapping[str, str] = MappingProxyType({}),
    sources: Mapping[str, str] = MappingProxyType({}),
    symbols: Mapping[str, Mapping[str, int]] = MappingProxyType({}),
    fuzzy: Sequence[Json] = (),
    cartridge_id: Mapping[str, str] = MappingProxyType({}),
    region: Mapping[str, str] = MappingProxyType({}),
    description: Mapping[str, str] = MappingProxyType({}),
) -> Path:
    """Write a committed project under tmp/project; headers go under include/, sources under src/.

    fuzzy rows are written as [[fuzzy]]; cartridge_id, region and description map version id -> value.
    """
    root = tmp / "project"
    root.mkdir(parents=True)
    from unbake import config as configuration

    table: dict[str, Any] = {}
    named: dict[str, dict] = {}
    for vid in versions:
        placed = sorted(_place(functions, "asm", vid) + _place(data, "data", vid))
        for (a_rom, _, a_name, a_blob), (b_rom, _, b_name, _) in pairwise(placed):
            if a_rom + len(a_blob) > b_rom:
                raise ValueError(f"{a_name} overlaps {b_name} in {vid}")
        end = max([ROM_BASE] + [rom + len(blob) for rom, _, _, blob in placed])
        rom = bytearray(-(-end // PAGE) * PAGE)
        for at, _, _, blob in placed:
            rom[at : at + len(blob)] = blob
        _write(root / f"roms/{vid}.z64", bytes(rom))
        subs: list[Any] = [[at, kind, name] for at, kind, name, _ in placed] + [[len(rom)]]
        segment = {"name": "main", "type": "code", "start": ROM_BASE, "vram": VRAM_BASE, "subsegments": subs}
        document = {"name": vid, "sha1": sha1(rom), "segments": [segment]}  # JSON is valid YAML
        _write(root / f"versions/{vid}/Game.yaml", json.dumps(document, indent=1) + "\n")
        rows = symbols.get(vid) or {name: VRAM_BASE + at - ROM_BASE for at, _, name, _ in placed}
        for n, a in rows.items():
            named.setdefault(n, {"kind": "function" if n in functions else "data"})[vid] = a
        table[vid] = {
            "baserom": f"roms/{vid}.z64",
            "baserom_sha1": sha1(rom),
            "split": f"versions/{vid}/Game.yaml",
            "symbols": f"versions/{vid}/symbol_addrs.txt",
            "macros": [f"VERSION_{vid.upper()}"],
        }
        for key, values in (("cartridge_id", cartridge_id), ("region", region), ("description", description)):
            if vid in values:
                table[vid][key] = values[vid]
    from unbake import symbols as symbol_table

    _write(root / "symbols.toml", symbol_table.dump(named).decode())
    for vid in versions:
        _write(root / f"versions/{vid}/symbol_addrs.txt", symbol_table.render(named, vid).decode())
    info = {"id": "fixture", "name": "fixture", "title": "Fixture", "versions": list(versions)}
    info["names_from"] = names_from
    info |= {"toolchain": "gcc-test"}
    build = {key: [] for key in ("cppflags", "cflags", "asflags", "gnu_asflags")}
    config = "schema = 3\n\n" + _table("project", info) + _table("build", build) + "[resident]\n\n"
    config += "".join(_table(f"version.{vid}", row) for vid, row in table.items())
    _write(root / "config.toml", config)
    first = sorted(_place(functions, "asm", names_from))
    default_group = {"name": "code_80000400", "segment": "main", "members": [name for _, _, name, _ in first]}
    default_group |= {"evidence": "authored", "signals": [], "subsystem": "", "library": ""}
    empty = ("" if units else "unit = []\n") + ("group = []\n" if groups == [] else "")  # the schema requires both keys
    named = [
        {"source": f"src/{g['name']}.c", "header": f"include/{g['segment']}/{g['name']}.h", **g}
        for g in ([default_group] if groups is None else groups)
    ]
    rows = _rows("group", named) + _rows("unit", units)
    layout = f"schema = 5\n{empty}\n" + rows
    _write(root / "layout.toml", layout + _rows("fuzzy", fuzzy))
    _write(root / "types.toml", "schema = 1\n\n[function]\n\n[global]\n\n[struct]\n")
    marks = configuration.load_resource("repo.toml")["readme"]
    readme = Template(configuration.template("README.md.in"))
    _write(root / "README.md", readme.substitute(title=info["title"], begin=marks["begin"], end=marks["end"]))
    for relative, text in headers.items():
        _write(root / "include" / relative, text)
    for relative, text in sources.items():
        _write(root / "src" / relative, text)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture project")
    return root


def host(tmp: Path, **overrides: Any) -> Path:
    """Write tmp/host.toml, every tool a copy of true in tmp/bin.

    overrides: dotted keys such as resources.workers; cores and workers may be "auto".
    """
    tmp = tmp.resolve()
    (tmp / "bin").mkdir(parents=True, exist_ok=True)
    true = shutil.which("true")
    if true is None:
        raise RuntimeError("the true executable is required for the fixture host")
    tools = {}
    for name in TOOLS:
        shutil.copy2(true, tmp / "bin" / name)
        tools[name] = str(tmp / "bin" / name)
    document: dict[str, dict[str, Any]] = {
        "resources": {
            "cores": 2,
            "workers": 2,
            "memory_parent_bytes": 1 << 30,
            "memory_worker_bytes": 1 << 30,
            "move_wait_s": 5,
        },
        "cache": {"dir": str(tmp / "cache"), "max_bytes": 1 << 30},
        "toolchains": {"root": str(tmp / "toolchains")},
        "tools": tools,
        "budgets": {"serial_seconds": 2, "serial_cores": 2, "pool_fill": 0.8, "pool_fanout": 2},
        "publish": {"author_name": "test", "author_email": "test@example.invalid"},
    }
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        document.setdefault(section, {})[key] = value
    path = tmp / "host.toml"
    _write(path, "schema = 2\n\n" + "".join(_table(name, body) for name, body in document.items()))
    return path


def config(tmp: Path, **kw: Any) -> Config:
    """project() + host() + unbake.config.load. kw go to project(); kw["host"] is a dict of host overrides."""
    from unbake import config as configuration

    overrides = kw.pop("host", {})
    if "functions" not in kw:
        code = bytes.fromhex("03e0000800000000")
        kw["functions"] = {"func_80000400": {v: (ROM_BASE, code) for v in kw.get("versions", ("a", "b"))}}
    return configuration.load(project(tmp, **kw), host(tmp, **overrides))


def bare_config(
    root: Path,
    *,
    versions: Sequence[str] = ("a", "b"),
    names_from: str | None = None,
    toolchain: str = "gcc-test",
    name: str = "fixture",
    title: str = "Fixture",
    cores: int = 2,
    workers: int = 2,
    memory: int = 1024,
    toolchain_root: Path | None = None,
    tools: Mapping[str, Path] | None = None,
    author: tuple[str, str] = ("test", "test@example.invalid"),
    origins: Mapping[str, Origin] | None = None,
    macros: Mapping[str, tuple[str, ...]] | None = None,
    version_files: Mapping[str, VersionFiles] | None = None,
    resident: Mapping[str, tuple[Any, ...]] | None = None,
) -> Config:
    """Contract records for a project and host without projecting files, ROMs or Git (unlike config)."""
    host = Host(
        cores, workers, memory, memory, root / "cache", memory, toolchain_root or root / "tools", tools or {}, None,
        2, 2, 0.8, 2, author, origins or {}, "host",
    )  # fmt: skip
    project = Project(
        root, name, name, title, tuple(versions), versions[0] if names_from is None and versions else names_from or "",
        toolchain, {}, version_files or {}, macros or {}, resident or {}, {}, "project",
    )  # fmt: skip
    return Config(project, host, "config")


def elf_object(names: Sequence[str]) -> bytes:
    """An ELF32 big-endian MIPS object that holds only a symbol table naming NAMES."""
    strtab = b"\0" + b"\0".join(n.encode() for n in names) + b"\0"
    offsets = [1 + sum(len(n) + 1 for n in names[:i]) for i in range(len(names))]
    symtab = b"\0" * 16 + b"".join(struct.pack(">IIIBBH", o, 0, 0, 0x10, 0, 0) for o in offsets)
    shstr = b"\0.symtab\0.strtab\0.shstrtab\0"
    end = 52 + len(strtab) + len(symtab) + len(shstr)
    symbols_at = 52 + len(strtab)
    rows = [
        (0,) * 10,
        (1, 2, 0, 0, symbols_at, len(symtab), 2, 0, 4, 16),
        (9, 3, 0, 0, 52, len(strtab), 0, 0, 1, 0),
        (17, 3, 0, 0, symbols_at + len(symtab), len(shstr), 0, 0, 1, 0),
    ]
    fields = struct.pack(">HHIIIIIHHHHHH", 1, 8, 1, 0, 0, end, 0, 52, 0, 0, 40, 4, 3)
    header = b"\x7fELF\x01\x02\x01" + b"\0" * 9 + fields
    return header + strtab + symtab + shstr + b"".join(struct.pack(">10I", *r) for r in rows)


def native_result(
    argv: Sequence[str] = ("tool",),
    exit: int | None = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    cpu: float = 0.0,
    outputs: Mapping[str, Path] = MappingProxyType({}),
) -> NativeResult:
    return NativeResult(tuple(argv), "/", exit, None, stdout, stderr, 0.0, cpu, 0, dict(outputs))


def proof(
    unit: str,
    member: str,
    version: str,
    exact: bool,
    missing: Sequence[str] = (),
    score: float | None = None,
    symptoms: Json | None = None,
) -> Proof:
    """Test-helper defaults only: score 1.0 when exact else 0.5; symptoms {} when exact else bytes_differ."""
    score = (1.0 if exact else 0.5) if score is None else score
    if symptoms is None:
        symptoms = {} if exact else {"bytes_differ": True, "score": score}
    if exact == bool(missing):
        raise ValueError("a Proof is exact with no missing pieces, or not exact with at least one")
    sha = sha256(f"{unit}:{member}:{version}".encode())
    built = sha if exact else sha256(sha.encode())
    return Proof(unit, member, version, "recipe", sha, sha, built, sha, exact, tuple(missing), score, symptoms)


def submission(**fields: Any) -> Submission:
    """A complete Submission for tests; fields override the defaults."""
    member, source_sha = fields.get("member", "func_80000400"), "0" * 64
    values: dict[str, Any] = {"operation": "publish", "member": member, "function": None, "source_sha256": source_sha}
    values |= {"overrides": {"add": [], "omit": []}, "base": "0" * 40, "origin": "submit", "note": ""}
    values |= {"invocation": "test-invocation", "extras": {}, "extra_bases": {}, "withheld": ()}
    values |= fields
    values.setdefault("proofs", (proof(f"src/{member}.c", member, "a", True),))
    id_basis = (values["member"], values["source_sha256"], values["overrides"], values["operation"])
    values.setdefault("id", digest(id_basis))
    values.setdefault("source", f".unbake/inbox/{values['id']}.c")
    return Submission(**values)


def rom_bytes(game_code: bytes = b"NXX", region: bytes = b"E", order: str = "z64", body: bytes = b"") -> bytes:
    """A 0x101000-byte ROM image: z64 magic, game code at 0x3B, region at 0x3E, body at 0x1000.

    order is z64, v64 (byte pairs swapped) or n64 (little-endian words), the orders rom.toml names.
    """
    if len(game_code) != 3 or len(region) != 1:
        raise ValueError("game_code is 3 bytes and region is 1 byte")
    image = bytearray(0x101000)
    image[0:4] = bytes.fromhex("80371240")
    image[0x20:0x34] = b"FIXTURE".ljust(20, b" ")
    image[0x3B:0x3E], image[0x3E:0x3F] = game_code, region
    image[0x1000 : 0x1000 + len(body)] = body
    if order == "z64":
        return bytes(image)
    if order == "v64":
        image[0::2], image[1::2] = image[1::2], image[0::2]
    elif order == "n64":
        original = bytes(image)
        for lane in range(4):
            image[lane::4] = original[3 - lane :: 4]
    else:
        raise ValueError(f"order {order!r} is not z64, v64 or n64")
    return bytes(image)


def preprocessed(files: Mapping[str, str]) -> str:
    """Concatenate files as cpp output: a '# 1 "file"' marker, then the text."""
    return "".join(f'# 1 "{name}"\n{text.rstrip(chr(10))}\n' for name, text in files.items())


def elf_with_symbols(defined: Sequence[str], undefined: Sequence[str]) -> bytes:
    """Minimal big-endian MIPS ELF32 .o: .text (jr ra; nop), global FUNC symbols, undefined NOTYPE symbols."""
    text = struct.pack(">II", 0x03E00008, 0)
    names, strtab = {}, b"\0"
    for name in (*defined, *undefined):
        names[name] = len(strtab)
        strtab += name.encode() + b"\0"
    symtab = struct.pack(">IIIBBH", 0, 0, 0, 0, 0, 0)
    symtab += b"".join(struct.pack(">IIIBBH", names[n], 0, 8, 0x12, 0, 1) for n in defined)
    symtab += b"".join(struct.pack(">IIIBBH", names[n], 0, 0, 0x10, 0, 0) for n in undefined)
    shstr, section_names = b"\0", {}
    for name in (".text", ".symtab", ".strtab", ".shstrtab"):
        section_names[name] = len(shstr)
        shstr += name.encode() + b"\0"
    body, offsets = b"", {}
    for name, blob in ((".text", text), (".symtab", symtab), (".strtab", strtab), (".shstrtab", shstr)):
        body += b"\0" * (-(52 + len(body)) % 4)
        offsets[name] = 52 + len(body)
        body += blob
    body += b"\0" * (-(52 + len(body)) % 4)
    shoff = 52 + len(body)

    def header(name: str, kind: int, flags: int, blob: bytes, link: int = 0, info: int = 0, entsize: int = 0) -> bytes:
        fields = (section_names[name], kind, flags, 0, offsets[name], len(blob), link, info, 4, entsize)
        return struct.pack(">10I", *fields)

    sections = struct.pack(">10I", *([0] * 10))
    sections += header(".text", 1, 6, text) + header(".symtab", 2, 0, symtab, 3, 1, 16)
    sections += header(".strtab", 3, 0, strtab) + header(".shstrtab", 3, 0, shstr)
    ident = b"\x7fELF" + bytes([1, 2, 1, 0]) + bytes(8)
    return ident + struct.pack(">HHIIIIIHHHHHH", 1, 8, 1, 0, 0, shoff, 0, 52, 0, 0, 40, 5, 4) + body + sections
