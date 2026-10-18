"""Version facts: ROMs, segments and the hand-kept symbol_addrs.txt. Never computes an address."""

from __future__ import annotations

import mmap
import re
import struct
from bisect import bisect_right
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unbake import config as configuration
from unbake import effort, store, symbols
from unbake.contracts import Config, Finding, Refusal, Version, digest, sha256

_WORD = 0xFFFFFFFF
_DATA = "data"  # the symbol_addrs type of a data symbol
_ROW = re.compile(r"^\s*([\w.$@]+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)\s*;\s*(?://\s*(.*))?$")
_ATTR = re.compile(r"(\w+):(\S+)")
_ROMS: dict[str, tuple[Any, mmap.mmap]] = {}


@dataclass(frozen=True)
class Row:
    """One line of a splat symbol_addrs.txt: `name = 0xVRAM; // type:func rom:0x… size:0x…`."""

    name: str
    vram: int
    kind: str  # "func" | "data" | "" (splat decides)
    rom: int | None
    size: int | None
    line: int


def _number(text: str) -> int:
    return int(text, 16) if text.lower().startswith("0x") else int(text)


def parse_symbol_addrs(data: bytes, file: str = "symbol_addrs.txt") -> tuple[Row, ...]:
    """The rows of a splat symbol file in file order. A name with two addresses, or with no C identifier, is refused."""
    rows: list[Row] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
        match = _ROW.match(line)
        if not match:
            continue
        name, vram = match[1], _number(match[2])
        symbols.identifier(name, f"{file}:{number}")
        attrs = dict(_ATTR.findall(match[3] or ""))
        if seen.setdefault(name, vram) != vram:
            raise Refusal(
                Finding(
                    "symbols.conflict",
                    reason=f"{name} has two addresses in {file}",
                    path=f"{file}:{number}",
                    line=number,
                    unit=name,
                )
            )
        rows.append(
            Row(
                name,
                vram,
                attrs.get("type", ""),
                _number(attrs["rom"]) if "rom" in attrs else None,
                _number(attrs["size"]) if "size" in attrs else None,
                number,
            )
        )
    return tuple(rows)


def symbol_file(version: str) -> str:
    return f"versions/{version}/symbol_addrs.txt"


def symbol_addrs(config: Config, version: str, reader: Callable[[str], bytes] | None = None) -> tuple[Row, ...]:
    """The fact file of one version, parsed once per content."""
    path = symbol_file(version)
    data = reader(path) if reader else (config.project.root / path).read_bytes()
    return effort.memo(("symbol_addrs", path, sha256(data)), lambda: parse_symbol_addrs(data, path))


def segments(config: Config, version: str) -> tuple[tuple[str, int, int, int], ...]:
    """(name, rom start, rom end, vram or 0) per segment; each ends where the next begins, the last at the ROM end."""
    rows = config.project.version[version].segments
    rom = config.project.root / "roms" / f"baserom.{version}.z64"
    last = rom.stat().st_size if rom.is_file() else rows[-1].rom
    return tuple(
        (s.name, s.rom, rows[i + 1].rom if i + 1 < len(rows) else max(s.rom, last), s.vram or 0)
        for i, s in enumerate(rows)
    )


def pieces(config: Config, version: str) -> tuple[tuple[str, int, int, int], ...]:
    """`segments`, each with the name splat and the link give its piece. A segment name is the tag the sources'
    stand-ins name (the asm folder), and several rows of a version may share one: one data segment that the version's
    link cuts around blobs. splat and the link need a name per piece: the first row keeps the tag, a later row of the
    same tag is `<tag>_<ROM>`."""
    out: list[tuple[str, int, int, int]] = []
    seen: set[str] = set()
    rows = segments(config, version)
    tags = {row[0] for row in rows}
    for name, start, end, vram in rows:
        piece = name if name not in seen else f"{name}_{start:X}"
        if piece != name and piece in tags:
            raise Refusal(
                Finding(
                    "config.segments",
                    f"{version}: segment {name} at rom 0x{start:X} would be piece {piece}, the name of another segment",
                    path="config.toml",
                    versions=(version,),
                )
            )
        seen.add(name)
        out.append((piece, start, end, vram))
    return tuple(out)


def segment_of(config: Config, version: str, row: Row) -> str:
    """The segment a symbol lives in: by its rom offset, else by its vram inside a code segment."""
    cuts = segments(config, version)
    if row.rom is not None:
        at = bisect_right([c[1] for c in cuts], row.rom) - 1
        if at >= 0 and row.rom < cuts[at][2]:
            return cuts[at][0]
    for name, start, end, vram in cuts:
        if vram and vram <= row.vram < vram + (end - start):
            return name
    raise Refusal(
        Finding(
            "version.split",
            reason=f"{row.name} in {version} lies in no segment of config.toml",
            path=symbol_file(version),
            line=row.line,
            unit=row.name,
            versions=(version,),
        )
    )


def asm_path(config: Config, version: str, name: str) -> Path:
    """The asm file splat writes for one function: asm/<v>/nonmatchings/<segment>/<name>.s."""
    with effort.stage("versions.asm_path"):
        rows = {r.name: r for r in symbol_addrs(config, version)}
        row = rows.get(name)
        if row is None:
            raise Refusal(
                Finding(
                    "version.split",
                    reason=f"no assembly for {name} in {version}",
                    unit=name,
                    versions=(version,),
                    action=f"add {name} to {symbol_file(version)}",
                )
            )
        return config.project.root / "asm" / version / "nonmatchings" / segment_of(config, version, row) / f"{name}.s"


def symbols_ld(config: Config, version: str) -> bytes:
    """The scoring symbol script of one version (PROVIDE per symbol), kept in the cache only, keyed by the fact file."""
    with effort.stage("versions.symbols_ld"):
        data = (config.project.root / symbol_file(version)).read_bytes()
        mask = configuration.load_resource("rom.toml").get("text_mask", _WORD)
        key = digest((version, sha256(data), mask))

        def produce() -> bytes:
            cuts = segments(config, version)
            spans = [(v, v + e - s) for _, s, e, v in cuts if v]
            out: dict[str, int] = {}
            for row in parse_symbol_addrs(data):
                inside = any(lo <= row.vram < hi for lo, hi in spans)
                out[row.name] = row.vram & mask if row.kind == "func" or (inside and row.kind != _DATA) else row.vram
            return "".join(f"PROVIDE({n} = 0x{a:08X});\n" for n, a in sorted(out.items())).encode()

        return store.cached(config, "generated", key, produce)


def read(
    config: Config,
    reader: Callable[[str], bytes],
    only: Collection[str] | None = None,
    reuse: Mapping[str, Version] | None = None,
) -> dict[str, Version]:
    with effort.stage("versions.read"):
        root, wanted = config.project.root, [v for v in config.project.versions if only is None or v in only]
        out: dict[str, Version] = {}
        for vid in wanted:
            rows = symbol_addrs(config, vid, reader)
            rom = root / "roms" / f"baserom.{vid}.z64"
            table = {r.name: r.vram for r in rows}
            code = frozenset(r.name for r in rows if r.kind == "func")
            out[vid] = Version(
                vid,
                rom,
                _rom_sha(rom),
                f"versions/{vid}/splat.yaml",
                symbol_file(vid),
                table,
                segments(config, vid),
                code,
            )
        return out


def facts_digest(version: Version) -> str:
    """A version's symbol table and code names as one hash (tens of thousands of names), computed once per command."""
    return effort.memo(
        ("facts", id(version.symbols), id(version.code)),
        lambda: (
            version.symbols,
            version.code,
            sha256(repr((sorted(version.symbols.items()), sorted(version.code))).encode()),
        ),
    )[2]


_IDENT = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")


def idents(data: bytes | None) -> frozenset[str]:
    """Every identifier a file spells, once per content."""
    if data is None:
        return frozenset()
    return effort.memo(("idents", sha256(data)), lambda: frozenset(m.decode() for m in _IDENT.findall(data)))


def _rom_sha(rom: Path) -> str:
    stat = rom.stat()
    return effort.memo(("rom", str(rom), stat.st_mtime_ns, stat.st_size), lambda: sha256(rom.read_bytes()))


def delay_slot(word: int) -> bool:  # a MIPS jump or branch: the next word is its delay slot
    op, rt = word >> 26, word >> 16 & 31
    return (
        (op == 0 and word & 62 == 8)
        or (op == 1 and rt & 12 == 0)
        or 2 <= op <= 7
        or 20 <= op <= 23
        or (16 <= op <= 18 and word >> 21 & 31 == 8)
    )


def rom_view(version: Version) -> mmap.mmap:
    key = str(version.rom)
    if key not in _ROMS:
        handle = version.rom.open("rb")
        _ROMS[key] = (handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
    return _ROMS[key][1]


def rom_bytes(version: Version, start: int, end: int) -> bytes:
    view = rom_view(version)
    if start < 0 or end < start or end > len(view):
        raise Refusal(
            Finding(
                "version.rom",
                reason=f"range 0x{start:X}-0x{end:X} is outside {version.id}'s ROM of {len(view)} bytes",
                versions=(version.id,),
                path=str(version.rom),
            )
        )
    return bytes(view[start:end])


def undefined(obj: Path) -> tuple[str, ...]:
    """The global and weak names an object leaves undefined, read straight from its big-endian ELF32 symbol table."""
    with effort.stage("versions.undefined"):
        data = obj.read_bytes()
        if data[:6] != b"\x7fELF\x01\x02":
            raise Refusal(
                Finding("native.exit", reason="the object is not a 32-bit big-endian ELF file", path=str(obj))
            )
        (shoff,) = struct.unpack_from(">I", data, 32)
        shentsize, shnum = struct.unpack_from(">HH", data, 46)
        names: set[str] = set()
        for i in range(shnum):
            _, kind, _, _, offset, size, link, _, _, entsize = struct.unpack_from(">10I", data, shoff + i * shentsize)
            if kind != 2:  # SHT_SYMTAB
                continue
            strings = struct.unpack_from(">10I", data, shoff + link * shentsize)[4]
            for at in range(offset + entsize, offset + size, entsize):  # entry 0 is the null symbol
                name, _, _, info, _, index = struct.unpack_from(">IIIBBH", data, at)
                if index == 0 and name and info >> 4 in (1, 2):  # undefined, global or weak
                    names.add(data[strings + name : data.index(b"\0", strings + name)].decode())
        return tuple(sorted(names))


def resolve(version: Version, names: Sequence[str], unit: str) -> tuple[Finding, ...]:
    return tuple(
        Finding(
            "symbols.unknown",
            reason=f"{name} has no address in {version.id}",
            versions=(version.id,),
            missing=(name,),
            unit=unit,
            path=symbol_file(version.id),
            action="add the row to the version's symbol_addrs.txt",
        )
        for name in names
        if name not in version.symbols
    )


def defined(obj: Path) -> tuple[dict[str, tuple[str, int, int]], dict[str, int]]:
    """(name -> (section name, value, size) of every symbol an ELF32 big-endian object defines, section name -> its
    total size), read from the section and symbol tables."""
    with effort.stage("versions.defined"):
        data = obj.read_bytes()
        if data[:6] != b"\x7fELF\x01\x02":
            raise Refusal(
                Finding("native.exit", reason="the object is not a 32-bit big-endian ELF file", path=str(obj))
            )
        (shoff,) = struct.unpack_from(">I", data, 32)
        shentsize, shnum, shstrndx = struct.unpack_from(">HHH", data, 46)
        headers = [struct.unpack_from(">10I", data, shoff + i * shentsize) for i in range(shnum)]
        names = headers[shstrndx][4]

        def label(table: int, at: int) -> str:
            return data[table + at : data.index(b"\0", table + at)].decode()

        titles = [label(names, h[0]) for h in headers]
        sizes: dict[str, int] = {}
        for title, header in zip(titles, headers, strict=True):
            if header[1] == 1 and header[2] & 2:  # PROGBITS or NOBITS that is allocated
                sizes[title] = sizes.get(title, 0) + header[5]
        found: dict[str, tuple[str, int, int]] = {}
        for header in headers:
            if header[1] != 2:  # SHT_SYMTAB
                continue
            strings = headers[header[6]][4]
            for at in range(header[4] + header[9], header[4] + header[5], header[9]):
                name, value, size, info, _, index = struct.unpack_from(">IIIBBH", data, at)
                if name and 0 < index < 0xFF00 and info & 15 in (0, 1, 2):
                    found.setdefault(label(strings, name), (titles[index], value, size))
        return found, sizes
