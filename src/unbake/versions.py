"""Version facts: ROMs, splat rows, symbol tables, undefined symbols. Never computes an address."""

from __future__ import annotations

import csv
import io
import mmap
import os
import pickle
import re
import struct
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from unbake import config as configuration
from unbake import effort, pool, store, symbols
from unbake.contracts import Config, Finding, Placement, Refusal, Version, code_identity, digest, pickled, sha256

Cached = Callable[[str, str, Callable[[], bytes]], bytes]  # store.content(config).cached
_SYMBOL = re.compile(r"^\s*([\w.$]+)\s*=\s*(0x[0-9A-Fa-f]+)\s*;")
_ROMS: dict[str, tuple[Any, mmap.mmap]] = {}


def _parsed(path: str, data: bytes, parse: Callable[[bytes], Any], cache: Cached | None = None) -> Any:
    """What a parse of the bytes gives: once per command, and with the content cache once per distinct content."""
    sha = sha256(data)

    def produce() -> Any:
        if cache is None:
            return parse(data)
        return pickled(cache, "parsed", digest((path, sha, code_identity(parse))), lambda: parse(data))

    return effort.memo((path, sha), produce)


def _entry(raw: Any) -> dict[str, Any]:
    """Normalise a splat segment or subsegment (list or mapping form) to a mapping."""
    if isinstance(raw, Mapping):
        return dict(raw)
    return {
        k: v
        for k, v in zip(("start", "type", "name", "section"), raw, strict=False)
        if k in ("start", "type") or isinstance(v, str)
    }


def _document(data: bytes, file: str) -> list[dict[str, Any]]:
    try:
        document = yaml.load(data, Loader=yaml.CSafeLoader)
        return [_entry(s) for s in document["segments"]]
    except (yaml.YAMLError, KeyError, TypeError, IndexError) as error:
        raise Refusal(Finding("version.split", reason=f"{file} is not a splat yaml: {error}", path=file)) from error


def _segment_rows(data: bytes, cache: Cached | None = None) -> tuple[tuple[str, int, int, int], ...]:
    """(name, start, next start, vram or 0) per segment; the last segment ends at its own start (caller widens)."""
    segments = _parsed("document", data, partial(_document, file="split"), cache)
    out = []
    for index, seg in enumerate(segments):
        end = segments[index + 1]["start"] if index + 1 < len(segments) else seg["start"]
        out.append((str(seg.get("name", "")), int(seg["start"]), int(end), int(seg.get("vram", 0) or 0)))
    return tuple(out)


def _symbol_table(data: bytes, file: str) -> dict[str, int]:
    table: dict[str, int] = {}
    for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
        match = _SYMBOL.match(line)
        if not match:
            continue
        name, address = match.group(1), int(match.group(2), 16)
        if table.setdefault(name, address) != address:
            raise Refusal(
                Finding(
                    "symbols.conflict",
                    reason=f"{name} has two addresses in {file}",
                    path=f"{file}:{number}",
                    line=number,
                    unit=name,
                )
            )
    return table


def _asm_symbols(data: bytes) -> tuple[dict[str, int], dict[str, int], frozenset[str]]:
    text = data.decode(errors="replace")
    label = r"(?m)^\s*(?:([gd])label\s+([\w.$]+)|([\w.$]+):)\s*\n\s*/\*\s*\w+\s+([0-9A-Fa-f]{8})\s"
    jump = r"(?m)/\*\s*\w+\s+([0-9A-Fa-f]{8})\s+([0-9A-Fa-f]{8})\s*\*/\s+j(?:al)?\s+([\w.$]+)\s*$"
    target = {n: (int(v, 16) & 0xF0000000) | ((int(w, 16) & 0x3FFFFFF) << 2) for v, w, n in re.findall(jump, text)}
    highs = {n: int(w, 16) & 0xFFFF for w, n in re.findall(r"(\w{8})\s*\*/\s+lui\s+\S+\s+%hi\(([\w.$]+)\)", text)}
    for w, n in re.findall(r"(\w{8})\s*\*/\s+\w+\s+[^%\n]*%lo\(([\w.$]+)\)", text):
        if n in highs:
            target.setdefault(n, ((highs[n] << 16) + ((int(w, 16) & 0xFFFF) ^ 0x8000) - 0x8000) & 0xFFFFFFFF)
    found = re.findall(label, text)
    branches = re.findall(r"(?m)\*/[ \t]+[bj][a-z]*[ \t]+(?:[^\n,]*,[ \t]*)*([A-Za-z_.][\w.$]*)[ \t]*(?:/\*.*)?$", text)
    code = {n or other for kind, n, other, _ in found if kind == "g" or other}
    code.update(branches, (n for _, _, n in re.findall(jump, text)))
    return {a or b: int(address, 16) for _, a, b, address in found}, target, frozenset(code)


def _asm_tree(config: Config, vid: str) -> tuple[list[str], str]:
    """The sorted asm files of a version and a stamp of the tree. Each directory is listed once and remembered with its
    mtime: a file added, removed or replaced (every write is a rename) moves its directory's mtime, so a later call
    stats the directories and lists only the ones that moved, never every file."""
    base, top = str(config.project.root), f"asm/{vid}"
    slot = sha256(f"{base}/{top}".encode())
    blob = store.get(config, "asmtree", slot)
    known: dict[str, tuple[int, list[str], list[str]]] = pickle.loads(blob) if blob else {}
    fresh: dict[str, tuple[int, list[str], list[str]]] = {}
    pending = [top]
    while pending:
        rel = pending.pop()
        try:
            mtime = os.stat(f"{base}/{rel}").st_mtime_ns
        except OSError:
            continue
        row = known.get(rel)
        if row is None or row[0] != mtime:
            names, dirs = [], []
            with os.scandir(f"{base}/{rel}") as listing:
                for entry in listing:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(entry.name)
                    elif entry.name.endswith(".s"):
                        names.append(entry.name)
            row = (mtime, names, dirs)
        fresh[rel] = row
        pending.extend(f"{rel}/{d}" for d in row[2])
    if fresh != known:
        store.put(config, "asmtree", slot, pickle.dumps(fresh, protocol=5))
    files = sorted(f"{rel}/{name}" for rel, (_, names, _) in fresh.items() for name in names)
    return files, digest(sorted((rel, row[0]) for rel, row in fresh.items()))


def fact_files(config: Config, vid: str) -> tuple[list[str], list[str]]:
    return fact_inputs(config, vid)[:2]


def fact_inputs(config: Config, vid: str) -> tuple[list[str], list[str], str]:
    """The auto symbol files, the asm files and the stamp of the asm tree."""
    root = config.project.root
    text = (root / config.project.version_files[vid].split).read_text()
    keys = ("undefined_funcs_auto_path", "undefined_syms_auto_path")
    paths = (re.search(rf"^\s+{key}:\s*(\S+)", text, re.M) for key in keys)
    autos = [p for m in paths if m and (root / (p := m[1].strip("'\""))).is_file()]
    return (autos, *_asm_tree(config, vid))


def _asm_job(item: tuple[str, str]) -> tuple[dict[str, int], dict[str, int], frozenset[str]]:
    root, rel = item
    return _asm_symbols((Path(root) / rel).read_bytes())


def _asm_key(item: tuple[str, str]) -> str:
    """A splat asm file is rewritten only when its bytes change, so name, size and mtime stand for its content."""
    stat = os.stat(f"{item[0]}/{item[1]}")
    return digest((item[1], stat.st_size, stat.st_mtime_ns))


def _generated(
    config: Config, vid: str, files: tuple[list[str], list[str]], parsed: Sequence[Any], declared: Mapping[str, int]
) -> tuple[dict[str, int], frozenset[str]]:
    root = config.project.root
    facts: dict[str, int] = {}
    code: set[Any] = set()
    autos, sources = files
    for path in autos:
        found = _parsed(path, (root / path).read_bytes(), partial(_symbol_table, file=path))
        facts.update(found)
        code.update(found if "funcs" in Path(path).name else ())
    contexts = sorted((root / ".unbake" / "symbols" / vid).glob("spim_context*.csv"))
    rows = (r for p in contexts for r in csv.DictReader(io.StringIO(p.read_text())))
    facts.update({r["getName"]: int(r["address"], 16) for r in rows if r["category"] == "symbol"})
    for rel, (labels, jumps, called) in zip(sources, parsed, strict=True):
        code |= called
        facts.update({n: a for n, a in jumps.items() if n not in declared and n not in facts})
        for name, address in ({} if contexts else labels).items():
            if facts.get(name, address) != address:
                raise Refusal(Finding("symbols.conflict", f"{name}: generated addresses disagree", path=rel))
            if name not in declared:
                facts[name] = address
    return facts, frozenset(code)


def _generated_key(config: Config, vid: str, files: tuple[list[str], list[str]], declared: Mapping[str, int]) -> str:
    """What _generated reads: the auto symbol files, the spim contexts, each asm file's stat, the declared names."""
    root = config.project.root
    contexts = sorted((root / ".unbake" / "symbols" / vid).glob("spim_context*.csv"))
    return digest(
        (
            vid,
            [(p, (root / p).read_bytes()) for p in files[0]],
            [p.read_bytes() for p in contexts],
            [_asm_key((str(root), rel)) for rel in files[1]],
            declared,
            code_identity(_generated),
        )
    )


def unowned(name: str) -> bool:  # a data row no function has claimed
    return re.match(r"\w+/unresolved/", name) is not None


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


def facts_of(version: Version, names: frozenset[str]) -> str:
    """A version's facts as a unit reads them: the address and code mark of each name its text can spell. A landing
    elsewhere changes other names only (addresses are the ROM's), so it moves no other unit's stamp."""
    table, code = version.symbols, version.code
    return effort.memo(
        ("facts-of", id(table), id(code), names),
        lambda: (
            table,
            code,
            sha256(repr(sorted((n, table.get(n), n in code) for n in names if n in table or n in code)).encode()),
        ),
    )[2]


def _rom_sha(rom: Path) -> str:
    stat = rom.stat()
    return effort.memo(("rom", str(rom), stat.st_mtime_ns, stat.st_size), lambda: sha256(rom.read_bytes()))


def read(
    config: Config,
    reader: Callable[[str], bytes],
    only: Collection[str] | None = None,
    reuse: Mapping[str, Version] | None = None,
) -> dict[str, Version]:
    with effort.stage("versions.read"):
        root, table = config.project.root, config.project.version_files
        cached = store.content(config).cached
        wanted = [v for v in table if only is None or v in only]
        if not wanted:  # a map-only overlay reads no version
            return {}
        key = digest(
            (
                str(root),
                wanted,
                [(reader(table[v].split), reader(table[v].symbols)) for v in wanted],
                reader(symbols.path()),
            )
        )  # files the command rewrites itself drop the memos (effort.forget)

        def compute() -> dict[str, Version]:
            named = symbols.load(reader, table, cached)
            seeds: dict[str, tuple[Mapping[str, int], frozenset[str], Mapping[str, int]]] = {
                v: (x.symbols, x.code, {}) for v, x in (reuse or {}).items()
            }
            built = effort.memo(("generated", str(root), id(named)), lambda: (named, seeds))[1]
            todo = [v for v in wanted if v not in built]
            files = {v: fact_files(config, v) for v in todo}
            declared = {v: symbols.declared(named, v) for v in todo}
            keys = {v: _generated_key(config, v, files[v], declared[v]) for v in todo}
            # a version whose inputs held reads one entry
            blobs: dict[str, bytes | None] = {v: store.get(config, "generated", keys[v]) for v in todo}
            parsed = iter(
                pool.map(
                    config,
                    "versions.asm",
                    _asm_job,
                    [(str(root), rel) for v in todo if blobs[v] is None for rel in files[v][1]],
                    _asm_key,
                )
            )
            for vid in todo:
                effort.count("generated", blobs[vid] is not None)
                if blobs[vid] is None:
                    found = [next(parsed) for _ in files[vid][1]]
                    blob = pickle.dumps((*_generated(config, vid, files[vid], found, declared[vid]), declared[vid]), 5)
                    store.put(config, "generated", keys[vid], blob)
                    blobs[vid] = blob
                built[vid] = pickle.loads(blobs[vid] or b"")
            out: dict[str, Version] = {}
            for vid in wanted:
                facts, code, extra = built[vid]
                row, rom = table[vid], root / table[vid].baserom
                segments = _parsed(row.split, reader(row.split), partial(_segment_rows, cache=cached))
                if segments:  # the last segment runs to the end of the ROM
                    name, start, _, vram = segments[-1]
                    segments = (*segments[:-1], (name, start, max(start, rom.stat().st_size), vram))
                known = {**facts, **extra} if extra else facts  # a reused table stays the one object
                out[vid] = Version(vid, rom, _rom_sha(rom), row.split, row.symbols, known, segments, code)
            return out

        return dict(effort.memo(("versions.read", key), compute))


def section_of(version: Version, kind: str, where: str) -> str:
    section = next(("." + key for key, row in configuration.sections().items() if kind in row["types"]), None)
    if section is None:
        raise Refusal(Finding("version.split", reason=f"{where} has unknown type {kind!r}", path=version.split))
    return section


def rows(
    version: Version, reader: Callable[[str], bytes], cache: Cached | None = None
) -> list[tuple[str, str, Placement]]:
    text = reader(version.split)  # the same text of the same version is parsed once per command
    found: list[tuple[str, str, Placement]] = effort.memo(
        ("rows", version.id, sha256(text), version.segments), lambda: _kept_rows(version, text, cache)
    )
    return list(found)


def _kept_rows(version: Version, text: bytes, cache: Cached | None) -> list[tuple[str, str, Placement]]:
    """The rows of one version's split text, kept by that text, its segments and this code."""
    if cache is None:
        return _rows(version, text, cache)
    key = digest(("rows", version.id, sha256(text), version.segments, code_identity(_rows)))
    return pickled(cache, "rows", key, lambda: _rows(version, text, cache))


def _rows(version: Version, text: bytes, cache: Cached | None) -> list[tuple[str, str, Placement]]:
    with effort.stage("versions.rows") as span:
        segments = _parsed("document", text, partial(_document, file=version.split), cache)
        ends = {name: end for name, _, end, _ in version.segments}
        out: list[tuple[str, str, Placement]] = []
        for seg in segments:
            if "type" not in seg:
                continue
            seg_name, seg_start = str(seg.get("name", "")), int(seg["start"])
            seg_end, seg_vram = ends.get(seg_name, seg_start), int(seg.get("vram", 0) or 0)
            subs = [_entry(s) for s in seg.get("subsegments", ())]
            end_markers = [int(s["start"]) for s in subs if "type" not in s and s.get("start") is not None]
            subs = [s for s in subs if "type" in s]
            if not subs:
                subs = [dict(seg)]
            for index, sub in enumerate(subs):
                start = int(sub["start"]) if sub.get("start") is not None else seg_end
                next_start = subs[index + 1].get("start") if index + 1 < len(subs) else None
                end = (
                    int(next_start)
                    if next_start is not None
                    else min((e for e in end_markers if e >= start), default=seg_end)
                )
                where = f"{seg_name} subsegment {index} at 0x{start:X}"
                if "name" not in sub:
                    reason = f"{where} has no name"
                    raise Refusal(Finding("version.split", reason=reason, path=version.split, unit=seg_name))
                kind = str(sub["type"])
                section = section_of(version, str(sub.get("section", kind)), where)
                vram = int(sub["vram"]) if "vram" in sub else seg_vram + (start - seg_start)
                size = 0
                if section == ".bss":
                    next_sub = subs[index + 1] if index + 1 < len(subs) else {}
                    next_vram = next_sub.get("vram", seg.get("vram_end"))
                    size = int(sub.get("size", int(next_vram) - vram if next_vram is not None else end - start))
                    if size < 0:
                        raise Refusal(Finding("version.split", "negative bss extent", path=version.split))
                    end = start
                out.append((str(sub["name"]), kind, Placement(version.id, section, start, end, vram, size)))
        out.sort(key=lambda r: (r[2].rom_start, r[2].vram, r[0]))
        twice = next((n for n, c in Counter(r[0] for r in out).items() if c > 1), None)
        if twice is not None:  # one name is one row: a second row of that name would be silently left unowned
            at = [f"0x{r[2].rom_start:X}" for r in out if r[0] == twice]
            raise Refusal(
                Finding(
                    "version.split",
                    reason=f"{twice} names two rows of {version.id} ({', '.join(at)})",
                    path=version.split,
                    unit=twice,
                )
            )
        span.add(items=len(out))
        return out


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


def asm_path(config: Config, version: str, member: str) -> Path:
    with effort.stage("versions.asm_path"):
        root = config.project.root
        found = effort.memo(
            ("asm", str(root), version), lambda: {Path(p).stem: root / p for p in _asm_tree(config, version)[0]}
        ).get(member)
        if found is None:
            raise Refusal(
                Finding(
                    "version.split",
                    reason=f"no assembly for {member} in {version}",
                    unit=member,
                    versions=(version,),
                    action="run unbake setup",
                )
            )
        return found


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
            path=symbols.path(),
            action="add the row to symbols.toml or run unbake setup to join identities",
        )
        for name in names
        if name not in version.symbols
    )
