"""Immutable layouts, source placement options, and conservative boundary repair."""

from __future__ import annotations

import bisect
import io
import json
import os
import pickle
import re
import struct
import tarfile
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import chain, count, pairwise
from typing import Any

import rabbitizer

from unbake import config as configuration
from unbake import effort, headers, journal, pool, process, store, symbols, view
from unbake import versions as version_data
from unbake.contracts import (
    Claim,
    Config,
    Finding,
    Group,
    Json,
    LayoutMap,
    Member,
    Placement,
    Plan,
    Refusal,
    Snapshot,
    UnitSpec,
    Version,
    code_identity,
    digest,
    sha256,
    sha256_file,
)

_LINE = re.compile(
    r"^(\s*-\s*\[\s*)(0[xX][0-9a-fA-F]+|\d+)(\s*,\s*)(\.?\w+)(\s*,\s*)(['\"]?)([^,\]\s'\"]+)(['\"]?)(.*)$"
)
_AUTO = re.compile(r"^(func|D)_([0-9A-F]{8})(_\w+)?$")


def _kinds() -> tuple[str, str, str]:
    rows = configuration.kinds()
    compiled = next(k for k, r in rows.items() if r["credit"] == "code" and "compile" in r["phases"])
    assembly = next(k for k, r in rows.items() if r["credit"] == "code" and "compile" not in r["phases"])
    datum = next(k for k, r in rows.items() if r["credit"] != "code" and "compile" in r["phases"])
    return compiled, assembly, datum


def is_datum(unit: UnitSpec) -> bool:
    """Whether the unit compiles rows of data (the kind that is not credited as code and has a compile phase)."""
    row = configuration.kind(unit.kind)
    return row["credit"] != "code" and "compile" in row["phases"]


def _content(config: Config, layout: LayoutMap, vers: Mapping[str, Version], read: Callable[[str], bytes]) -> str:
    """What a snapshot is: map, version files, symbol table and extraction facts; equal content is an equal snapshot."""
    return digest(
        (
            layout.digest,
            config.digest,
            code_identity(capture, tree),
            read(symbols.path()),
            [
                (v.id, v.rom_sha256, read(v.split), read(v.symbols_file), version_data.facts_digest(v))
                for v in sorted(vers.values(), key=lambda v: v.id)
            ],
        )
    )


def _pins(config: Config, paths: list[str]) -> list[tuple[str, str]]:
    """(path, sha256) of each file the capture reads; a file with unchanged stat keeps its recorded hash."""
    base, slot = str(config.project.root), sha256(str(config.project.root).encode())
    blob = store.get(config, "pins", slot)
    known: dict[str, tuple[int, int, str]] = pickle.loads(blob) if blob else {}
    fresh, pins = {}, []
    for p in paths:
        st = os.stat(f"{base}/{p}")
        row = known.get(p)
        if row is None or row[:2] != (st.st_mtime_ns, st.st_size):
            row = (st.st_mtime_ns, st.st_size, sha256_file(f"{base}/{p}"))
        fresh[p] = row
        pins.append((p, row[2]))
    if fresh != known:
        store.put(config, "pins", slot, pickle.dumps(fresh, protocol=5))
    return pins


def capture(config: Config) -> Snapshot:
    with effort.stage("layout.capture"), journal.reading(config):

        def reader(p: str) -> bytes:
            return (config.project.root / p).read_bytes()

        for _ in range(3):
            commit = process.git(config, "rev-parse", "HEAD").stdout.decode().strip()
            paths, trees = {"layout.toml", symbols.path()}, []
            for vid, vf in config.project.version_files.items():
                paths.update((vf.split, vf.symbols, vf.baserom))
                autos, _, stamp = version_data.fact_inputs(config, vid)  # the asm files are keyed by their tree's stamp
                paths.update(autos)
                trees.append(stamp)
            paths.update(
                p.relative_to(config.project.root).as_posix()
                for p in config.project.root.glob(".unbake/symbols/*/*.csv")
            )
            pins = _pins(config, sorted(paths))
            # the root keeps copies of one project apart; not the commit: caches stay warm across setup's own
            key = digest((str(config.project.root), config.digest, pins, trees, code_identity(capture)))

            def produce() -> tuple[dict[str, Version], LayoutMap]:
                vers = version_data.read(config, reader)
                return vers, load_map(config, vers, reader("layout.toml"), reader)

            vers, layout = effort.memo(("capture", key), partial(store.cached_object, config, "capture", key, produce))
            if commit == process.git(config, "rev-parse", "HEAD").stdout.decode().strip():
                return Snapshot(config, commit, layout, vers, {}, _content(config, layout, vers, reader), dict(pins))
        raise Refusal(
            Finding("layout.busy", "HEAD moved while reading the project three times", action="retry the command")
        )


def tree(config: Config, revision: str, commit: str) -> Snapshot:
    """Capture Git blobs, including the caller's private index, without borrowing live source files."""
    archive = process.git(config, "archive", "--format=tar", revision).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        blobs = {f.name: handle.read() for f in files if f.isfile() and (handle := files.extractfile(f)) is not None}

    def read(path: str) -> bytes:
        if path in blobs:
            return blobs[path]
        if path.startswith(("src/", "include/")) or path.endswith((".toml", ".yaml", ".ld")):
            raise FileNotFoundError(path)
        return (config.project.root / path).read_bytes()  # extracted assembly and ROM inputs

    project = configuration.load_project(config.project.root, read)
    config = Config(project, config.host, digest((project.digest, config.host.digest)))
    vers = version_data.read(config, read)
    mapped = load_map(config, vers, read("layout.toml"), read)
    return Snapshot(
        config, commit, mapped, vers, blobs, digest((_content(config, mapped, vers, read), blobs)), tree=True
    )


def overlay(snapshot: Snapshot, writes: Mapping[str, bytes | None]) -> Snapshot:
    """The snapshot with the writes laid over it, built once per command however many stages ask for it."""
    return effort.memo(("overlay", snapshot.digest, digest(writes)), lambda: _overlay(snapshot, writes))


def pure(snapshot: Snapshot, over: Snapshot) -> bool:
    """Whether `over` is `snapshot` with writes laid and nothing derived again: its versions and map are the base's, so
    its writes and digest alone make it anywhere, without reading a file."""
    return over.versions is snapshot.versions and over.layout is snapshot.layout


def lay(snapshot: Snapshot, writes: Mapping[str, bytes | None], digest_: str) -> Snapshot:
    """A pure overlay made again from its base, its writes and the digest the parent gave it (never read again from
    files a worker may see rewritten)."""
    new = replace(snapshot, overlays={**snapshot.overlays, **writes}, digest=digest_)
    effort.memo(("overlay", snapshot.digest, digest(writes)), lambda: new)
    effort.memo(("parent", digest_), lambda: (snapshot, dict(writes)))
    return new


def _overlay(snapshot: Snapshot, writes: Mapping[str, bytes | None]) -> Snapshot:
    with effort.stage("layout.overlay"):
        overlays = {**snapshot.overlays, **writes}
        symbol = symbols.path()
        watched = {"layout.toml", symbol, *(p for v in snapshot.versions.values() for p in (v.split, v.symbols_file))}
        others = sorted((p, sha256(b) if b is not None else None) for p, b in overlays.items() if p not in watched)
        new = replace(snapshot, overlays=overlays, digest=digest((snapshot.digest, others)))
        moved = bool(snapshot.versions) and (symbol in overlays or snapshot.reads.get(symbol) != view._pin(new, symbol))
        if moved or watched.intersection(writes):
            changed = {v.id for v in snapshot.versions.values() if moved or {v.split, v.symbols_file} & writes.keys()}
            vers = {
                **snapshot.versions,
                **version_data.read(snapshot.config, new.read, changed, None if moved else snapshot.versions),
            }
            layout = load_map(snapshot.config, vers, new.read("layout.toml"), new.read)
            new = replace(
                new,
                versions=vers,
                layout=layout,
                digest=digest((_content(snapshot.config, layout, vers, new.read), others)),
            )
        effort.memo(("parent", new.digest), lambda: (snapshot, dict(writes)))  # indexes and pool items derive from it
        return new


def split_states(snapshot: Snapshot, version: Version) -> list[tuple[str, str, Placement]]:
    return version_data.rows(version, snapshot.read, store.content(snapshot.config).cached)


def _written(writes: Mapping[str, bytes | None], snapshot: Snapshot, path: str) -> bytes:
    """The text a path has in the writes so far, else in the snapshot."""
    current = writes.get(path)
    return snapshot.read(path) if current is None else current


def load_map(config: Config, versions: Mapping[str, Version], text: bytes, reader: Callable[[str], bytes]) -> LayoutMap:
    """The map of `text` over these versions, built once per command however many overlays ask for it."""
    held = effort.memo(
        ("load-map", config.digest, digest(text), tuple(id(v) for v in versions.values())),
        lambda: (versions, _load_map(config, versions, text, reader)),
    )  # holds versions: ids stay unique
    return held[1]


def _load_map(
    config: Config, versions: Mapping[str, Version], text: bytes, reader: Callable[[str], bytes]
) -> LayoutMap:
    with effort.stage("layout.load_map"):
        sha = sha256(text)
        if sha in _dumped:  # text dump_map wrote: its tables stand, nothing to parse again
            known = _dumped[sha]
            groups, units, fuzzy, authored = known.groups, known.units, known.fuzzy, known.authored
            boundaries = known.boundaries
        else:
            doc = configuration.toml("layout", text, "layout.toml", "layout.map", store.content(config).cached)
            authored = tuple(doc.get("authored", ()))
            boundaries = tuple(doc.get("boundaries", ()))
            groups = {
                r["name"]: Group(
                    **{**r, "members": tuple(r["members"]), "signals": tuple(r["signals"]), "sdk": bool(r["library"])}
                )
                for r in doc["group"]
            }
            units = {r["path"]: UnitSpec(**{**r, "members": tuple(r["members"])}) for r in doc["unit"]}
            fuzzy = {
                r["member"]: {
                    "path": r["path"],
                    "scores": dict(r["scores"]),
                    **({"key": dict(r["key"])} if "key" in r else {}),
                }
                for r in doc.get("fuzzy", ())
            }
        collected: dict[str, Any] = {}
        for key in sorted(versions):
            for name, state, placement in version_data.rows(versions[key], reader, store.content(config).cached):
                collected.setdefault(name, []).append((state, placement))
        datum = _kinds()[2]
        members = {}
        group_of = {m: g.name for g in groups.values() for m in g.members}
        for name, rows in collected.items():
            state = next((s for s, p in rows if p.version == config.project.names_from), rows[0][0])
            group = group_of.get(name, "")
            placements = tuple(sorted((p for _, p in rows), key=lambda p: (p.version, p.section)))
            members[name] = Member(
                name, "function" if any(p.section == ".text" for _, p in rows) else datum, state, group, placements
            )
        for unit in units.values():
            missing = tuple(n for n in unit.members if n not in members)
            if missing:
                raise Refusal(
                    Finding(
                        "layout.map", reason="unit members are absent from split files", unit=unit.path, missing=missing
                    )
                )
        missing = tuple(sorted(set(fuzzy) - members.keys()))
        if missing:
            raise Refusal(Finding("layout.map", reason="fuzzy members are absent from split files", missing=missing))
        known_digest = digest((authored, groups, units, fuzzy, *((boundaries,) if boundaries else ())))
        return LayoutMap(groups, members, units, known_digest, authored, fuzzy, boundaries)


_dumped: dict[str, LayoutMap] = {}


def _toml(value: Any) -> str:
    return f"[{', '.join(map(_toml, value))}]" if isinstance(value, (list, tuple)) else json.dumps(value)


def dump_map(layout: LayoutMap) -> bytes:
    """Written as text, not through a TOML document model: the map is megabytes and a model walks it for seconds."""
    tables: tuple[tuple[str, Sequence[Any]], ...] = (
        ("group", sorted(layout.groups.values(), key=lambda g: g.name)),
        ("unit", sorted(layout.units.values(), key=lambda u: u.path)),
        ("authored", layout.authored),
    )
    out = ["schema = 5", *([f"boundaries = {_toml(sorted(layout.boundaries))}"] if layout.boundaries else [])]
    out += [f"{key} = []" for key, rows in tables[:2] if not rows]
    for key, rows in tables:
        for row in rows:
            values = {k: row[k] for k in ("version", "rom", "kind", "name")} if key == "authored" else asdict(row)
            if key == "group":
                del values["sdk"]  # derived from `library` on load
            options = values.pop("options", None)
            out += [f"\n[[{key}]]", *(f"{n} = {_toml(v)}" for n, v in values.items() if n != "withheld" or v)]
            out += [f"\n[{key}.options]", *(f"{k} = {_toml(options[k])}" for k in ("add", "omit"))] if options else []
    for member, row in sorted(layout.fuzzy.items()):
        scores = ", ".join(f"{v} = {_toml(score)}" for v, score in sorted(row["scores"].items()))
        out += ["\n[[fuzzy]]", f"member = {_toml(member)}", f"path = {_toml(row['path'])}", f"scores = {{{scores}}}"]
        out += (
            [f"key = {{{', '.join(f'{v} = {_toml(k)}' for v, k in sorted(row['key'].items()))}}}"]
            if "key" in row
            else []
        )
    text = ("\n".join(out) + "\n").encode()
    _dumped[sha256(text)] = layout
    if len(_dumped) > 8:
        del _dumped[next(iter(_dumped))]
    return text


def unit_of(snapshot: Snapshot, member: str) -> UnitSpec | None:
    units = snapshot.layout.units  # the unit holding each member first, indexed once per mapping
    return effort.memo(
        ("owners", id(units)),
        lambda: (units, {m: u for u in reversed(list(units.values())) for m in reversed(u.members)}),
    )[1].get(member)


def asm_unit(snapshot: Snapshot, member: str, version: str) -> UnitSpec:
    with effort.stage("layout.asm_unit"):
        item = snapshot.layout.members.get(member)
        if item is None or not any(p.version == version and p.section == ".text" for p in item.placements):
            raise Refusal(
                Finding(
                    "layout.member",
                    reason="member has no text placement in this version",
                    unit=member,
                    versions=(version,),
                )
            )
        config = snapshot.config
        path = version_data.asm_path(config, version, member).relative_to(config.project.root).as_posix()
        return UnitSpec(path, _kinds()[1], item.group, (member,), config.project.toolchain, {"add": [], "omit": []})


def edit_split(
    text: bytes,
    name: str,
    *,
    state: str | None = None,
    start: int | None = None,
    rename: str | None = None,
    additions: tuple[tuple[int, str], ...] = (),
    drop: bool = False,
    types: frozenset[str] | None = None,
) -> bytes:
    body = text.decode()
    for hit in re.finditer(re.escape(name), body):
        lo, hi = body.rfind("\n", 0, hit.start()) + 1, body.find("\n", hit.end())
        line = body[lo : len(body) if hi < 0 else hi + 1]
        match = _LINE.match(line.rstrip("\r\n"))
        if not match or match[7] != name:
            continue
        ending = line[len(line.rstrip("\r\n")) :]
        parts = list(match.groups())
        editable = types if types is not None else configuration.sections()["text"]["types"]
        if state is not None and parts[3] not in editable:
            continue
        if state is not None:
            parts[3] = state
        if start is not None:
            parts[1] = f"0x{start:X}" if parts[1].lower().startswith("0x") else str(start)
        if rename is not None:
            parts[6] = rename
        replacement = [] if drop else ["".join(parts) + ending]
        for address, new_name in additions:
            added = parts.copy()
            added[1], added[6] = f"0x{address:X}", new_name
            added[8] = re.sub(r"(\].*?)\s*#.*", r"\1", added[8])
            if replacement and not replacement[-1].endswith("\n"):
                replacement[-1] += "\n"
            replacement.append("".join(added) + ending)
        return (body[:lo] + "".join(replacement) + body[lo + len(line) :]).encode()
    raise Refusal(Finding("layout.member", reason="member has no editable subsegment line", unit=name))


def _claim_row(version: Version, rows: list[list[Any]], names: set[str], c: Claim) -> bool:
    """Make the claim's bytes one row named by the claim: covered rows are replaced, the row it starts in ends there,
    what it leaves of the row it ends inside is a row named by its address (so is the head of a row of its own name).
    A row is [start, end, vram, name, type, name of the line it is written at, whether it takes that line or follows
    it]."""
    start, end, name = c.start, c.end, c.rows[0]
    at = bisect.bisect_right(rows, start, key=lambda r: r[0]) - 1
    after = bisect.bisect_left(rows, end, key=lambda r: r[0])
    if at < 0 or rows[at][1] <= start:
        raise Refusal(
            Finding(
                "layout.map",
                f"claim 0x{start:X}-0x{end:X} of {c.unit} starts outside the data rows",
                unit=c.unit,
                versions=(c.version,),
                path=version.split,
            )
        )
    first, last = rows[at], rows[after - 1]
    kind = (
        first[4]
        if version_data.section_of(version, first[4], name) == c.section
        else ("asm" if c.section == ".text" else c.section[1:])
    )
    if after - at == 1 and (first[0], first[1], first[3], first[4]) == (start, end, name, kind):
        return False
    if name in names and name not in {r[3] for r in rows[at:after]}:
        raise Refusal(
            Finding(
                "layout.map",
                f"{name} names another row of {version.id}",
                unit=c.unit,
                versions=(c.version,),
                path=version.split,
            )
        )
    new = [start, end, first[2] + start - first[0], name, kind, first[5], first[6] and first[0] == start]
    out = [new]
    if last[1] > end:
        vram = last[2] + end - last[0]
        section = version_data.section_of(version, last[4], name)[1:]
        out.append([end, last[1], vram, f"{section}/unresolved/{vram:08X}", last[4], first[5], False])
    if first[0] < start:
        first[1], at = start, at + 1
        if first[3] == name:  # the head a claim of the row's own name leaves is named by its address, as a tail is
            first[3] = f"{version_data.section_of(version, first[4], name)[1:]}/unresolved/{first[2]:08X}"
            names.add(first[3])
    names.difference_update(r[3] for r in rows[at:after])
    names.update(r[3] for r in out)
    rows[at:after] = out
    return True


def _claim_text(text: str, rows: list[list[Any]], origin: dict[str, tuple[Any, ...]]) -> str:
    """The split file with its data rows as rows has them; the lines of the rows are written the way their own are."""
    lines = text.splitlines(keepends=True)
    group: dict[str, Any] = {}
    for row in rows:
        group.setdefault(row[5], []).append(row)
    out, found = [], set()
    for line in lines:
        match = _LINE.match(line.rstrip("\r\n"))
        if not match or match[7] not in origin:
            out.append(line)
            continue
        found.add(match[7])
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        for row in group.get(match[7], ()):
            if row[6] and (row[0], row[4], row[3]) == (*origin[match[7]],):
                out.append(line)
                continue
            parts = list(match.groups())
            parts[1] = f"0x{row[0]:X}" if parts[1].lower().startswith("0x") else str(row[0])
            parts[3], parts[6] = row[4], row[3]
            parts[8] = re.sub(r"(\].*?)\s*#.*", r"\1", parts[8])
            out.append("".join(parts) + (line[len(line.rstrip("\r\n")) :] or "\n"))
    for name in origin.keys() - found:  # a row written as a mapping has no line to change
        same = [r for r in group.get(name, ()) if r[6] and (r[0], r[4], r[3]) == origin[name]]
        if len(group.get(name, ())) != 1 or not same:
            raise Refusal(Finding("layout.map", "row has no editable subsegment line", unit=name))
    return "".join(out)


def claim_rows(snapshot: Snapshot, claims: Sequence[Claim]) -> dict[str, bytes]:
    """The split files with every claim made one row named by it, whatever rows its bytes were before: the rows of
    the data a unit owns are generated from the claims alone. Nothing changes where the rows are already the claims."""
    with effort.stage("layout.claim_rows"):
        writes: dict[str, bytes] = {}
        for vid in sorted({c.version for c in claims}):
            version, text = snapshot.versions[vid], snapshot.read(snapshot.versions[vid].split).decode()
            # the data rows of a split file are read once per text, however many units ask for their claims
            base = effort.memo(
                ("claim-rows", vid, digest(text), version.segments), partial(_data_rows, snapshot, version)
            )
            mine = sorted((c for c in claims if c.version == vid), key=lambda c: c.start)
            if all(_is_row(version, base, c) for c in mine):
                continue  # every claim is a row already: nothing to write, and no row list is copied
            rows: list[list[Any]] = [list(r) for r in base]
            origin: dict[str, tuple[Any, ...]] = {r[3]: (r[0], r[4], r[3]) for r in rows}
            names: set[str] = {r[3] for r in rows}
            changed = [_claim_row(version, rows, names, c) for c in mine]
            if any(changed):
                writes[version.split] = _claim_text(text, rows, origin).encode()
        return writes


def _data_rows(snapshot: Snapshot, version: Version) -> list[tuple[Any, ...]]:
    return [
        (p.rom_start, p.rom_end, p.vram, n, k, n, True)
        for n, k, p in split_states(snapshot, version)
        if p.section in (".rodata", ".data")
    ]


def _is_row(version: Version, rows: Sequence[Sequence[Any]], c: Claim) -> bool:
    """Whether the claim is already exactly one row named by it (what _claim_row leaves alone)."""
    at = bisect.bisect_right(rows, c.start, key=lambda r: r[0]) - 1
    if at < 0 or rows[at][1] <= c.start or bisect.bisect_left(rows, c.end, key=lambda r: r[0]) - at != 1:
        return False
    first, name = rows[at], c.rows[0]
    kind = (
        first[4]
        if version_data.section_of(version, first[4], name) == c.section
        else ("asm" if c.section == ".text" else c.section[1:])
    )
    return (first[0], first[1], first[3], first[4]) == (c.start, c.end, name, kind)


def _grouped(item: Member) -> None:
    if not item.group:
        raise Refusal(
            Finding(
                "layout.member", reason=f"{item.name} belongs to no group", unit=item.name, action="run unbake setup"
            )
        )


def unit_path(layout: LayoutMap, group: str, taken: Collection[str]) -> str:
    """The path of the group's next unit: its source, then `<source stem>_2.c`, `_3.c` (CONTRACTS 2.2 step 4)."""
    known = layout.groups.get(group)
    base = known.source if known else f"src/{group}.c"
    stem = configuration.source_stem(base)
    suffix = base[len(stem) :]  # a unit keeps its module's suffix and case: TURRETAI.CPP, TURRETAI_2.CPP
    return next(p for p in chain([base], (f"{stem}_{k}{suffix}" for k in count(2))) if p not in taken)


def _defines(snapshot: Snapshot, unit: UnitSpec) -> tuple[str, ...]:
    """The macro switches a source sets itself: sources that set them differently are built under other options."""
    text = (snapshot.peek(unit.path) or b"").decode(errors="replace")
    return tuple(sorted(set(re.findall(r"(?m)^[ \t]*#[ \t]*(?:define|undef)\b[^\n]*", text))))


_Spellings = dict[
    str, dict[str, set[str]]
]  # per name, per spelling, where it comes from (decl, def, many, type, header)


_SPELLINGS = "layout.spellings.4"  # the cache kind of _spellings tables: a new parse gets a new number
_BLOCK_DECLARATION = re.compile(  # an extern, or a prototype: two words before its parameters, never a statement word
    r"(?m)^[ \t]*(?:extern\b[^;{}]*;|(?!(?:return|else|case|goto|sizeof|do)\b)"
    r"(?:(?:const|volatile|unsigned|signed|struct|union|enum)\s+)*\w+[\s*]+\w+\s*\([^;{}=]*\)\s*;)"
)


_MACRO = re.compile(r"(?m)^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)")


def _qualified(text: str, name: str) -> str:
    """The type a declaration gives a name, qualifiers kept: one file cannot hold `const float x` and `float x`."""
    return headers._unnamed(headers._normal(headers._type_of(text, name), qualified=True))


def _spellings(text: str, source: bool) -> _Spellings:
    """The file-scope declarations, definitions and types of one text: per name its spellings and their kinds. In a
    source, a declaration or type no definition uses is left out: a composed unit drops it (compose takes only the
    blocks its members use), so the units of a relayout read the same as the ones they came from."""
    rows: list[tuple[str, str, str, str]] = []  # (name, spelling, kind, statement)
    # a macro renames every later use of its name: a unit that declares the name cannot share a file with its header
    rows += [(m, "#define", "macro", "") for m in _MACRO.findall(text)]
    for _, _, statement in headers.statements(text, bodies=source):
        if names := headers._type_names(statement):
            rows += [(n, headers._canon(statement), "type", statement) for n in names]
        elif "{" in statement and statement.partition("{")[0].rstrip().endswith(")"):  # a function definition
            head, _, body = statement.partition("{")
            rows += [(n, _qualified(head, n), "def", statement) for n in headers.declared_names(head + ";")]
            for inner in _BLOCK_DECLARATION.findall(body):  # a merged unit's block-scope declarations
                rows += [(n, _qualified(inner, n), "block", inner) for n in headers.declared_names(inner)]
        elif "=" not in statement and (declared := headers.declared_names(statement)):
            kind = "decl" if len(declared) == 1 else "many"
            rows += [(n, _qualified(statement, n), kind, statement) for n in declared]
        else:  # an object definition, with or without an initializer: spelled as its declaration would be
            head = "extern " + re.sub(r"^\s*(?:static|extern)\s+", "", statement.partition("=")[0].rstrip(" ;")) + ";"
            rows += [(n, _qualified(head, n), "def", statement) for n in headers.declared_names(head)]
    if source:
        used = set(re.findall(r"\w+", " ".join(r[3] for r in rows if r[2] in ("def", "block"))))
        rows = [r for r in rows if r[2] in ("def", "block", "type", "macro") or r[0] in used]
        used |= set(re.findall(r"\w+", " ".join(r[3] for r in rows if r[2] != "type")))
        rows = [r for r in rows if r[2] != "type" or r[0] in used]
    found: _Spellings = {}
    for n, spelled, kind, _ in rows:
        found.setdefault(n, {}).setdefault(spelled, set()).add(kind)
    return found


@dataclass(frozen=True)
class _Side:
    own: _Spellings
    reach: set[str]
    names: frozenset[str]
    paths: frozenset[str]
    header: str


class _Declarations:
    """What each unit's text declares at file scope, its own and through the headers it includes (quoted, beside it,
    under include/ or src/). Read once per text, only when two units would join."""

    def __init__(self, snapshot: Snapshot, old: Mapping[str, UnitSpec]) -> None:
        self.snapshot, self.old = snapshot, old
        self.units = {u.path: u for u in old.values()}
        self.pieces: dict[tuple[str, tuple[str, ...]], _Spellings] = {}
        self.reach_of: dict[str, frozenset[str]] = {}
        self.names_of: dict[frozenset[str], frozenset[str]] = {}
        self.texts: dict[str, bytes | None] = {}
        self.tables: dict[tuple[str, bool], _Spellings] = {}
        self.quotes: dict[str, tuple[str, ...]] = {}
        self.singles: dict[str, _Side] = {}
        self.word_sets: dict[str, frozenset[str]] = {}
        self.call_sets: dict[str, frozenset[str]] = {}

    def peek(self, path: str) -> bytes | None:
        if path not in self.texts:
            self.texts[path] = self.snapshot.peek(path)
        return self.texts[path]

    def table(self, path: str, source: bool) -> _Spellings:
        if (path, source) not in self.tables:
            content = self.peek(path) or b""
            key = f"{digest(content)}:{int(source)}"
            # content-keyed in the host cache: each text is parsed once, by the first command that reads it
            self.tables[path, source] = effort.memo(
                ("spellings", key),
                lambda: store.cached_object(
                    self.snapshot.config, _SPELLINGS, key, partial(_spellings, content.decode("latin-1"), source)
                ),
            )
        return self.tables[path, source]

    def quoted(self, path: str) -> tuple[str, ...]:
        if path not in self.quotes:
            self.quotes[path] = headers._quoted(self.peek(path) or b"")
        return self.quotes[path]

    def reach(self, path: str) -> frozenset[str]:
        if path not in self.reach_of:
            seen: set[str] = set()
            pending = [(path, n) for n in self.quoted(path)]
            while pending:
                at, name = pending.pop()
                for base in (os.path.dirname(at), "include", "src"):
                    found = os.path.normpath(os.path.join(base, name))
                    if found not in seen and self.peek(found) is not None:
                        seen.add(found)
                        pending += [(found, n) for n in self.quoted(found)]
                        break
            self.reach_of[path] = frozenset(seen)
        return self.reach_of[path]

    def piece(self, path: str, members: tuple[str, ...]) -> _Spellings:
        """What the part of an old unit that holds these members declares: its shared blocks and those members' blocks,
        as compose would take them; the whole unit when it holds no other member or does not split."""
        unit = self.units[path]
        if set(members) >= set(unit.members):
            return self.table(path, True)
        if (path, members) not in self.pieces:
            try:
                _, blocks = headers.split_atoms((self.peek(path) or b"").decode("latin-1"), unit.members)
            except Refusal:
                self.pieces[path, members] = self.table(path, True)
            else:
                text = "\n".join(v for k, v in blocks.items() if k.startswith("#") or k in members)
                self.pieces[path, members] = _spellings(text, True)
        return self.pieces[path, members]

    def side(self, members: Collection[str], header: str = "") -> _Side:
        """What the old units holding some members declare together, each only as far as it holds them: their own
        spellings, the headers they reach and the names those declare."""
        own: _Spellings = {}
        reach: set[str] = set()
        held: dict[str, list[str]] = {}
        for m in members:
            held.setdefault(self.old[m].path, []).append(m)
        for p, names in held.items():
            for n, spelled in self.piece(p, tuple(sorted(names))).items():
                for s, kinds in spelled.items():
                    own.setdefault(n, {}).setdefault(s, set()).update(kinds)
            reach |= self.reach(p)
        if header and self.peek(header) is not None:  # a composed unit includes its module header
            reach |= {header, *self.reach(header)}
        return _Side(own, reach, self.header_names(frozenset(reach)), frozenset(held), header)

    def header_names(self, reach: frozenset[str]) -> frozenset[str]:
        if reach not in self.names_of:
            self.names_of[reach] = frozenset().union(*(self.table(h, False).keys() for h in reach))
        return self.names_of[reach]

    def spelled(self, side: _Side, name: str) -> dict[str, set[str]]:
        spelled = {s: set(k) for s, k in side.own.get(name, {}).items()}
        if name in side.names:
            for h in side.reach:
                for s, kinds in self.table(h, False).get(name, {}).items():
                    spelled.setdefault(s, set()).update({"header"} | kinds)
        return spelled

    def clashes(self, run: _Side, unit: _Side) -> list[str]:
        """The names the units of a run and the units of an atom declare so that one file cannot hold both: together
        they spell a name more than one way, a header, a definition, a type, a block-scope extern or a declaration of
        several names holds one of the spellings (extern declarations alone move to block scope when the units merge),
        and no one of the units already holds exactly those spellings (it builds today)."""
        both = (unit.own.keys() & (run.own.keys() | run.names)) | (run.own.keys() & unit.names)
        clashed = []
        for n in sorted(both):
            a, b = self.spelled(run, n), self.spelled(unit, n)
            merged = {s: a.get(s, set()) | b.get(s, set()) for s in a.keys() | b.keys()}
            if len(merged) < 2:
                continue
            singles = [self.spelled(self.single(p), n) for p in run.paths | unit.paths]
            # one unit spelling a name two ways does so in preprocessor branches (#if per version): a merged file
            # moves the declarations out of those branches, so that unit holds nothing another can share
            branched = any(len(self.single(p).own.get(n, {})) > 1 for p in run.paths | unit.paths)
            placed = any(k - {"decl"} for k in merged.values())  # not extern declarations alone
            if branched or (placed and not any(self._holds(one, merged) for one in singles)):
                clashed.append(n)
        # a type one side defines in its own text that a header only the other side reaches defines too: the same
        # body twice is a redefinition all the same
        for mine, other in ((run, unit), (unit, run)):
            clashed += sorted(
                n
                for n in mine.own.keys() & other.names
                if n not in clashed and n not in mine.names and any("type" in k for k in mine.own[n].values())
            )
        # a function one side calls without declaring it (an implicit declaration, any arguments) that the other
        # side declares: the declaration would check the call
        for mine, other in ((run, unit), (unit, run)):
            undeclared = self.calls(mine) - mine.own.keys() - mine.names
            clashed += sorted(n for n in undeclared & (other.own.keys() | other.names) if n not in clashed)
        # a name both define (a static or inline helper each old unit kept its own copy of): one file holds one
        clashed += sorted(n for n in both if n not in clashed and self.defines(run, n) and self.defines(unit, n))
        # a const object defined with its value beside code that reads it: the compiler folds the read into the
        # value, so the code changes
        clashed += sorted(self.constants(run) & self.words(unit) | self.constants(unit) & self.words(run))
        return clashed

    @staticmethod
    def defines(side: _Side, name: str) -> bool:
        """Whether the side defines the name itself: a function body or an object."""
        return any("def" in kinds for kinds in side.own.get(name, {}).values())

    @staticmethod
    def constants(side: _Side) -> set[str]:
        """The const objects the side defines (with a value or not, a definition the compiler sees)."""
        return {
            n
            for n, spelled in side.own.items()
            if any("def" in kinds and "const" in s and "(" not in s for s, kinds in spelled.items())
        }

    def calls(self, side: _Side) -> set[str]:
        """Every name the old units of the side call (a word before an opening parenthesis)."""
        return set().union(*(self.called(p) for p in side.paths))

    def called(self, path: str) -> frozenset[str]:
        if path not in self.call_sets:
            text = headers._blank((self.peek(path) or b"").decode("latin-1"))
            self.call_sets[path] = frozenset(re.findall(r"\b([A-Za-z_]\w*)\s*\(", text))
        return self.call_sets[path]

    def words(self, side: _Side) -> set[str]:
        """Every word the old units of the side spell (their whole texts)."""
        return set().union(*(self.spelled_words(p) for p in side.paths))

    def spelled_words(self, path: str) -> frozenset[str]:
        if path not in self.word_sets:
            self.word_sets[path] = frozenset(re.findall(r"\w+", (self.peek(path) or b"").decode("latin-1")))
        return self.word_sets[path]

    @staticmethod
    def _holds(one: Mapping[str, set[str]], merged: Mapping[str, set[str]]) -> bool:
        """Whether one unit already spells a name every way the merged units do, from the same kinds of place (an
        extern declaration aside): what it holds builds today."""
        return one.keys() == merged.keys() and all(merged[s] - {"decl"} <= one[s] for s in merged)

    def single(self, path: str) -> _Side:
        """What one old unit declares as it builds today: its own text and the headers it includes, never the module
        header a composed unit would add."""
        if path not in self.singles:
            self.singles[path] = self.side(self.units[path].members)
        return self.singles[path]

    def join(self, run: _Side, unit: _Side) -> _Side:
        own = {n: {s: set(k) for s, k in spelled.items()} for n, spelled in run.own.items()}
        for n, spelled in unit.own.items():
            for s, kinds in spelled.items():
                own.setdefault(n, {}).setdefault(s, set()).update(kinds)
        reach = run.reach | unit.reach
        return _Side(own, reach, self.header_names(frozenset(reach)), run.paths | unit.paths, run.header)


def _options_of(group: Group, unit: UnitSpec) -> Json:
    """The flag set a unit of the module compiles with: the module's own once it has one (one set per object; empty
    lists are the defaults), else what the unit carries from before modules held options."""
    return group.options or unit.options


def physical(
    snapshot: Snapshot,
    layout: LayoutMap,
    split: list[Finding] | None = None,
    keep: Collection[str] = (),
    reflow: Collection[str] | None = None,
) -> dict[str, UnitSpec]:
    """The units of a map, a function of its groups, the split rows and the unit toolchains (CONTRACTS 2.2): landed
    atoms in names_from order join the open unit of their module while holders, toolchain, options and withheld
    agree, every holder's rows of each section stay one gapless run and their units' declarations can share one file.
    A member the map records as a boundary (a proof found bytes change when it shared a file) starts a new unit, and
    a unit in `keep` (one whose proof failed this command) stays exactly as it is.
    `reflow` names the modules laid again (None: every module, a relayout); any other module whose units already hold
    every member it lists keeps its units as they are, whatever their file names, extra rows or inside boundaries (a
    unit landed under its member's name stays so): a move re-cuts only the modules it changes.
    A unit laid again keeps the file name of the old unit its first member came from when that name is still its
    module's, so one new unit renames no other.
    Each name that keeps units apart is a non-blocking layout.decl_split in `split`."""
    with effort.stage("layout.physical"):
        apart: dict[str, set[str]] = {}
        compiled, _, datum = _kinds()
        library = tuple(f.rstrip("/") + "/" for f in snapshot.config.project.library_dirs)
        result = {
            p: u
            for p, u in layout.units.items()
            if u.kind not in (compiled, datum) or p.startswith(library) or p in keep  # a kept unit stays as it is
        }
        listed = {n for g in layout.groups.values() for n in g.members}
        placed = listed | {
            n for u in layout.units.values() for n in u.members if n.count("/") == 2 and n.split("/")[1] in listed
        }
        for path, unit in layout.units.items():  # a unit no group places (a batch-data land) keeps its claim as it is
            if unit.members and not placed.intersection(unit.members):
                result.setdefault(path, unit)
        old = {n: u for u in layout.units.values() if u.path not in result for n in u.members}
        boundaries = set(layout.boundaries)
        declarations = _Declarations(snapshot, old)
        place: dict[tuple[str, str], dict[str, int]] = {}
        breaks: dict[tuple[str, str], set[int]] = {}  # column indexes whose bytes do not follow the row before
        off_word: set[str] = set()  # members whose rows start off a word boundary in some version
        for vid, version in snapshot.versions.items():
            last: dict[str, Any] = {}
            for name, _, p in sorted(split_states(snapshot, version), key=lambda r: (r[2].rom_start, r[2].vram)):
                column = place.setdefault((vid, p.section), {})
                if p.section != ".text" and p.vram % 4:
                    off_word.add(name)
                before, last[p.section] = last.get(p.section), p
                index = column.setdefault(name, len(column))
                if before and (
                    before.vram + before.size != p.vram or (p.section != ".bss" and before.rom_end != p.rom_start)
                ):
                    breaks.setdefault((vid, p.section), set()).add(index)
        members, names_from = snapshot.layout.members, snapshot.config.project.names_from

        def rank(name: str) -> tuple[int, str, str]:
            item = members[name]
            holder = item.reference(names_from)
            return next((p.rom_start for p in item.placements if p.version == holder), 0), holder, name

        owned = {
            n for n in old if n in members and n.count("/") == 2 and n not in listed
        }  # unit members no group lists

        def rows_of(atom: Sequence[str]) -> dict[tuple[str, str], list[int]]:
            rows: dict[tuple[str, str], list[int]] = {}
            for n in atom:
                for p in members[n].placements:
                    rows.setdefault((p.version, p.section), []).append(place[p.version, p.section][n])
            return rows

        def gapless(spans: Mapping[tuple[str, str], tuple[int, int, int]]) -> bool:
            """Every holder's rows of each section are one run: consecutive columns whose bytes follow each other."""
            return all(
                s[1] - s[0] + 1 == s[2] and not any(i in breaks.get(k, ()) for i in range(s[0] + 1, s[1] + 1))
                for k, s in spans.items()
            )

        def unaligned(atom: Sequence[str]) -> bool:
            """Whether a data member of the atom starts off a word boundary in some version: in a file with other
            units' data the compiler aligns what follows it (a word after three bytes moves by one): it stays apart."""
            return any(n in off_word and members[n].kind != "function" for n in atom)

        def spans_of(rows: Mapping[tuple[str, str], list[int]]) -> dict[tuple[str, str], tuple[int, int, int]]:
            return {k: (min(v), max(v), len(v)) for k, v in rows.items()}

        listing: dict[str, list[str]] = {}  # per member the groups that list it, in map order
        for group in layout.groups.values():
            for n in group.members:
                listing.setdefault(n, []).append(group.name)
        # a member several groups list joins one unit, never one per group (its bytes would link twice): the group
        # its unit names today, else the first that lists it
        home = {n: old[n].group if n in old and old[n].group in gs else gs[0] for n, gs in listing.items()}
        by_group: defaultdict[str, list[UnitSpec]] = defaultdict(list)
        for unit in layout.units.values():
            if unit.path not in result:
                by_group[unit.group].append(unit)

        def scheme(group: str, path: str) -> bool:
            """Whether a path is one of the file names unit_path gives the module's units."""
            base = unit_path(layout, group, ())
            stem = configuration.source_stem(base)
            return path == base or bool(
                re.fullmatch(rf"{re.escape(stem)}_(?:[2-9]|[1-9]\d+){re.escape(base[len(stem) :])}", path)
            )

        def settled(group: Group, landed: set[str]) -> bool:
            """Whether a module the command did not change keeps its units: they hold every member it lists once, with
            its flag set. What else they hold (a padding or another version's row) and a boundary inside a unit stay
            as they are: only a relayout, or a move or proof that names the module, lays it again."""
            units = by_group.get(group.name, [])
            held = [n for u in units for n in u.members if n.count("/") != 2]
            elsewhere = {n for n in landed if n.count("/") != 2} - set(held)  # held by another module's unit
            return (
                bool(units)
                and all(u.options == _options_of(group, u) for u in units)  # a module flag set changed: lay again
                and len(held) == len(set(held))
                # a member it lists that a re-laid module's unit holds comes back: that module lets it go
                and not any(old[n].group in (reflow or ()) for n in elsewhere)
            )

        for group in layout.groups.values():
            landed = {n for n in group.members if n in old and n in members and home[n] == group.name}
            if reflow is not None and group.name not in reflow and settled(group, landed):
                result.update((u.path, u) for u in by_group[group.name])
                continue
            # an owned member follows its owner; one whose name points at no member (an old unit's name: the row was
            # cut for a unit, not a function) follows the first member of its old unit, or the map drops its bytes
            mate = {
                n: next((m for m in old[n].members if m in landed and m.count("/") != 2), "")
                for n in owned
                if n.split("/")[1] not in landed
            }
            landed |= {n for n in owned if n.split("/")[1] in landed or mate.get(n)}
            atoms: dict[str, list[str]] = {}
            for n in sorted(landed, key=rank):
                owner = (n.split("/")[1] if n.split("/")[1] in landed else mate.get(n, "")) if n.count("/") == 2 else n
                atoms.setdefault(owner if owner in landed else n, []).append(n)
            for leader, atom in list(atoms.items()):  # the keyed leader already starts its atom, even after data
                for n in [n for n in atom if n != leader and n in boundaries]:
                    atom.remove(n)
                    atoms[n] = [n]
            for leader, atom in list(atoms.items()):
                # an owned member whose rows do not adjoin its owner's (another unit's bytes lie between) stays an atom
                # of its own: one unit owns one gapless run per section, or it cannot link
                if len(atom) > 1 and not gapless(spans_of(rows_of(atom))):
                    kept = [leader] if leader in atom else []
                    for n in atom:
                        if n not in kept and gapless(spans_of(rows_of([*kept, n]))):
                            kept.append(n)
                        elif n not in kept:
                            atoms[n] = [n]
                    atoms[leader] = sorted(kept, key=rank)
            for leader, atom in list(atoms.items()):
                # an owned member from another old unit that defines a const the owner's code reads (or reads one the
                # owner defines) stays an atom of its own: in one file the compiler folds the read into the value
                if len(atom) > 1 and leader in atom and len({old[n].path for n in atom}) > 1:
                    mine = declarations.side([n for n in atom if old[n].path == old[leader].path])
                    for n in [n for n in atom if old[n].path != old[leader].path]:
                        other = declarations.side([n])
                        if declarations.constants(other) & declarations.words(mine) or declarations.constants(
                            mine
                        ) & declarations.words(other):
                            atom.remove(n)
                            atoms[n] = [n]
                    if not gapless(spans_of(rows_of(atom))):  # what it left must still be one run per section
                        kept = [leader]
                        for n in atom:
                            if n != leader and gapless(spans_of(rows_of([*kept, n]))):
                                kept.append(n)
                            elif n != leader:
                                atoms[n] = [n]
                        atoms[leader] = sorted(kept, key=rank)
            runs: list[tuple[list[str], Any, dict[tuple[str, str], tuple[int, int, int]]]] = []
            homes: list[set[str]] = []  # per run the old units of its atoms
            sides: list[_Side | None] = []  # and what they declare, read when a second old unit would join
            for leader in sorted(atoms, key=rank):
                atom, unit = atoms[leader], old[leader]
                rows = rows_of(atom)
                props = (
                    tuple(sorted({v for n in atom for v in members[n].holders()})),
                    unit.toolchain,
                    json.dumps(_options_of(group, unit), sort_keys=True),
                    tuple(sorted(unit.withheld)),
                    _defines(snapshot, unit),
                )
                tail = runs[-1] if runs else None
                rides = (  # owned adjacent data rides in the module unit whatever the data unit's own options were
                    tail is not None
                    and all(members[n].kind != "function" for n in atom)
                    and any(members[n].kind == "function" for n in tail[0])
                    and tail[1][0] == props[0]
                )
                if tail and (tail[1] == props or rides) and not boundaries.intersection(atom):
                    spans = tail[2]
                    joined = {
                        k: (min(spans[k][0], *v), max(spans[k][1], *v), spans[k][2] + len(v))
                        if k in spans
                        else (min(v), max(v), len(v))
                        for k, v in rows.items()
                    }
                    if gapless(joined):
                        paths = homes[-1] | {old[n].path for n in atom}
                        if len(paths) > 1 and (unaligned(atom) or unaligned(runs[-1][0])):
                            runs.append((list(atom), props, spans_of(rows)))
                            homes.append({old[n].path for n in atom})
                            sides.append(None)
                            continue
                        if len(paths) > 1:  # atoms of one old unit build together today: nothing to read
                            run_side = sides[-1] or declarations.side(runs[-1][0], group.header)
                            unit_side = declarations.side(atom, group.header)
                            if clashed := declarations.clashes(run_side, unit_side):
                                for n in clashed:
                                    apart.setdefault(n, set()).update(paths)
                                runs.append((list(atom), props, spans_of(rows)))
                                homes.append({old[n].path for n in atom})
                                sides.append(unit_side)
                                continue
                            sides[-1] = declarations.join(run_side, unit_side)
                        else:
                            sides[-1] = None
                        runs[-1][0].extend(atom)
                        spans.update(joined)
                        homes[-1] = paths
                        continue
                runs.append((list(atom), props, spans_of(rows)))
                homes.append({old[n].path for n in atom})
                sides.append(None)
            claimed: dict[int, str] = {}  # a run keeps its first member's old file name while it is the module's
            for index, (run, _, _) in enumerate(runs):
                was_at = old[run[0]].path
                if was_at not in result and was_at not in claimed.values() and scheme(group.name, was_at):
                    claimed[index] = was_at
            for index, (run, _, _) in enumerate(runs):
                path = claimed.get(index) or unit_path(layout, group.name, {*result, *claimed.values()})
                kind = compiled if any(members[n].kind == "function" for n in run) else datum
                unit = old[run[0]]
                options = _options_of(group, unit)
                result[path] = UnitSpec(path, kind, group.name, tuple(run), unit.toolchain, options, unit.withheld)
        if reflow is not None:  # units of a module the map no longer names keep their rows while no unit took them
            taken = {n for u in result.values() for n in u.members}
            for name, units in by_group.items():
                if name not in layout.groups and name not in reflow:
                    result.update((u.path, u) for u in units if not taken.intersection(u.members))
        owned_data = {n for n in old.keys() & members.keys() if members[n].kind != "function"}
        lost = owned_data - {n for u in result.values() for n in u.members}
        if lost:
            raise Refusal(
                Finding(
                    "layout.map",
                    "physical layout dropped existing data owners",
                    missing=tuple(sorted(lost)),
                    action="report this layout change; existing data members must keep an owner",
                )
            )
        if split is not None:
            split += [
                Finding(
                    "layout.decl_split",
                    f"{n} is declared differently by units that would share one file: they stay apart",
                    unit=", ".join(sorted(paths)),
                    blocking=False,
                )
                for n, paths in sorted(apart.items())
            ]
        return result


def _ordered(snapshot: Snapshot, group: str, source: bytes) -> bytes:
    """A new unit's text with its include block in the order move composes (own module header first)."""
    from unbake import headers

    known = snapshot.layout.groups.get(group)
    return headers.ordered(
        source.decode(errors="replace"), known.header.removeprefix("include/") if known else ""
    ).encode()


def unit_options(snapshot: Snapshot, member: str, source: bytes) -> list[tuple[UnitSpec, dict[str, bytes | None]]]:
    return effort.memo(
        ("unit_options", snapshot.digest, member, digest(source)), lambda: _options(snapshot, member, source)
    )


def source_unit(snapshot: Snapshot, member: str, source: bytes) -> tuple[UnitSpec, dict[str, bytes | None]] | None:
    """Bind an explicit shared unit, keeping its landed prefix with contiguous text; the whole unit is proved."""
    from unbake import headers

    names = tuple(
        n
        for n in headers.defined(source)
        if n in snapshot.layout.members and snapshot.layout.members[n].kind == "function"
    )
    if len(names) < 2 or member not in names:
        return None
    group = snapshot.layout.members[member].group
    if len(set(names)) != len(names) or any(snapshot.layout.members[n].group != group for n in names):
        raise Refusal(
            Finding("layout.member", "shared source must define distinct functions of one group", unit=member)
        )
    owners = {u.path: u for n in names if (u := unit_of(snapshot, n)) is not None}
    if len(owners) > 1:
        raise Refusal(Finding("layout.member", "shared source has more than one landed prefix", unit=member))
    prefix = next(iter(owners.values()), None)
    if prefix and (
        any(n not in names for n in prefix.members if snapshot.layout.members[n].kind == "function")
        or unit_of(snapshot, names[0]) != prefix
    ):
        raise Refusal(Finding("layout.member", "shared source must retain its complete landed prefix", unit=member))
    holders = snapshot.layout.members[member].holders()
    if any(snapshot.layout.members[n].holders() != holders for n in names):
        raise Refusal(Finding("layout.member", "shared source functions must hold the same versions", unit=member))
    for vid in holders:
        places = [
            p for n in names for p in snapshot.layout.members[n].placements if p.version == vid and p.section == ".text"
        ]
        if len(places) != len(names) or any(
            a.rom_end != b.rom_start or a.vram + a.size != b.vram for a, b in pairwise(places)
        ):
            raise Refusal(
                Finding(
                    "layout.member",
                    "shared source text must be contiguous in every holder",
                    unit=member,
                    versions=(vid,),
                )
            )
    writes: dict[str, bytes | None] = {}
    compiled, assembly, _ = _kinds()
    for n in names:
        if unit_of(snapshot, n) is None:
            for vid in snapshot.layout.members[n].holders():
                path = snapshot.versions[vid].split
                writes[path] = edit_split(
                    _written(writes, snapshot, path), n, state=compiled, types=frozenset({assembly})
                )
    data = tuple(n for n in prefix.members if snapshot.layout.members[n].kind != "function") if prefix else ()
    unit = (
        replace(prefix, members=(*names, *data))
        if prefix
        else UnitSpec(
            unit_path(snapshot.layout, group, snapshot.layout.units),
            compiled,
            group,
            names,
            snapshot.config.project.toolchain,
            {"add": [], "omit": []},
        )
    )
    units = {**snapshot.layout.units, unit.path: unit}
    return unit, {
        **writes,
        unit.path: _ordered(snapshot, group, source),
        "layout.toml": dump_map(replace(snapshot.layout, units=units)),
    }


def _options(snapshot: Snapshot, member: str, source: bytes) -> list[tuple[UnitSpec, dict[str, bytes | None]]]:
    with effort.stage("layout.unit_options"):
        if member not in snapshot.layout.members:
            raise Refusal(Finding("layout.member", reason="member is absent from layout", unit=member))
        item = snapshot.layout.members[member]
        compiled, assembly, datum = _kinds()
        if item.kind != "function":
            return [_data_option(snapshot, member, source, datum)]
        _grouped(item)
        split_writes = {}
        holder_rows = {}
        for holder in item.holders():
            version = snapshot.versions[holder]
            rows = split_states(snapshot, version)
            state = next((s for n, s, p in rows if n == member and p.section == ".text"), item.state)
            if state != assembly:
                raise Refusal(Finding("layout.member", reason=f"{member} is {state}, not {assembly}", unit=member))
            holder_rows[holder] = [n for n, _, p in sorted(rows, key=lambda r: r[2].rom_start)]
            split_writes[version.split] = edit_split(snapshot.read(version.split), member, state=compiled)
        group_units = sorted((u for u in snapshot.layout.units.values() if u.group == item.group), key=lambda u: u.path)
        options = []
        for unit in group_units:
            if unit.kind != compiled or not unit.members:
                continue
            if all(
                names.index(member) > 0 and names[names.index(member) - 1] == unit.members[-1]
                for names in holder_rows.values()
            ):
                options.append(
                    (replace(unit, members=(*unit.members, member)), snapshot.read(unit.path) + b"\n" + source)
                )
        standalone = UnitSpec(
            unit_path(snapshot.layout, item.group, snapshot.layout.units),
            compiled,
            item.group,
            (member,),
            group_units[0].toolchain if group_units else snapshot.config.project.toolchain,
            {"add": [], "omit": []},
        )
        options.append((standalone, _ordered(snapshot, item.group, source)))
        result: list[tuple[UnitSpec, dict[str, bytes | None]]] = []
        for unit, content in options:
            units = dict(snapshot.layout.units)
            units[unit.path] = unit
            result.append(
                (
                    unit,
                    {
                        **split_writes,
                        unit.path: content,
                        "layout.toml": dump_map(replace(snapshot.layout, units=units)),
                    },
                )
            )
        return result


def _data_option(snapshot: Snapshot, member: str, source: bytes, kind: str) -> tuple[UnitSpec, dict[str, bytes | None]]:
    """A data member published from C: each holder's split row turns from extracted type to compiled section."""
    item, writes = snapshot.layout.members[member], dict[str, bytes | None]()
    sections = configuration.sections()
    raw = frozenset(t for row in sections.values() for t in row["types"] if not t.startswith("."))
    for holder in item.holders():
        version = snapshot.versions[holder]
        state = next((s for n, s, _ in split_states(snapshot, version) if n == member), item.state)
        if state not in raw:
            raise Refusal(
                Finding(
                    "layout.member",
                    reason=f"{member} is {state} in {holder}, not extracted data",
                    unit=member,
                    versions=(holder,),
                )
            )
        section = next(name for name, row in sections.items() if state in row["types"])
        writes[version.split] = edit_split(
            _written(writes, snapshot, version.split), member, state=f".{section}", types=raw
        )
    group, new = (item.group, None) if item.group else data_group(snapshot, member)
    groups = {**snapshot.layout.groups, **({group: new} if new else {})}
    path = unit_path(replace(snapshot.layout, groups=groups), group, snapshot.layout.units)
    unit = UnitSpec(path, kind, group, (member,), snapshot.config.project.toolchain, {"add": [], "omit": []})
    units = {**snapshot.layout.units, unit.path: unit}
    return unit, {
        **writes,
        unit.path: _ordered(replace(snapshot, layout=replace(snapshot.layout, groups=groups)), group, source),
        "layout.toml": dump_map(replace(snapshot.layout, units=units, groups=groups)),
    }


def data_group(snapshot: Snapshot, member: str) -> tuple[str, Group | None]:
    """The module a data item joins: its owner function's, else the module of the row before it in its segment (the
    row after it for a segment head), else a new inferred module (CONTRACTS 2.4)."""
    layout, item = snapshot.layout, snapshot.layout.members[member]
    owner = member.split("/")[1] if member.count("/") == 2 else ""
    if owner in layout.members and layout.members[owner].group:
        return layout.members[owner].group, None
    version = snapshot.versions[item.reference(snapshot.config.project.names_from)]
    rows = sorted(split_states(snapshot, version), key=lambda r: r[2].rom_start)
    at = next(i for i, r in enumerate(rows) if r[0] == member)
    segment = next(s for s in version.segments if s[1] <= rows[at][2].rom_start < s[2])
    inside = [r for r in (*reversed(rows[:at]), *rows[at + 1 :]) if segment[1] <= r[2].rom_start < segment[2]]
    near = next(
        (layout.members[r[0]].group for r in inside if r[0] in layout.members and layout.members[r[0]].group), ""
    )
    if near:
        return near, None
    name = f"data_{rows[at][2].vram:08X}"
    return name, None if name in layout.groups else Group(
        name, segment[0], (member,), "inferred", ("adjacent",), False, f"src/{name}.c", f"include/{segment[0]}/{name}.h"
    )


def _decoded(
    snapshot: Snapshot, version: Version, rows: list[Any]
) -> tuple[dict[str, tuple[int, ...]], frozenset[int]]:
    def decode() -> tuple[dict[str, tuple[int, ...]], frozenset[int]]:
        addresses = {p.vram for _, _, p in rows if p.section == ".text"}
        words = {}
        references: set[Any] = set()
        for name, _, placement in rows:
            if placement.section not in (".text", ".data", ".rodata"):
                continue
            blob = version_data.rom_bytes(version, placement.rom_start, placement.rom_end)
            values = struct.unpack(f">{len(blob) // 4}I", blob[: len(blob) // 4 * 4])
            if placement.section == ".text":
                words[name] = values
                references.update(
                    ((placement.vram + 4 * i + 4) & 0xF0000000) | ((word & 0x3FFFFFF) << 2)
                    for i, word in enumerate(values)
                    if word >> 26 == 3
                )
            else:
                references.update(addresses.intersection(values))
        return words, frozenset(references)

    split = sha256(snapshot.read(version.split))
    return effort.memo(("decoded", version.rom_sha256, split), decode)


def _rename(name: str, old: int, new: int) -> str:
    match = _AUTO.fullmatch(name)
    return f"{match[1]}_{new:08X}{match[3] or ''}" if match and int(match[2], 16) == old else name


def _retained(snapshot: Snapshot) -> dict[str, bytes]:
    from unbake.infer import _code, masked_code

    targets_by_version = {
        v: {version.symbols[n] for n in version.code if n in version.symbols}
        for v, version in snapshot.versions.items()
    }
    retained = [
        (v, p)
        for v, version in snapshot.versions.items()
        for n, state, p in split_states(snapshot, version)
        if p.section == ".data"
        and (n.startswith("retained_") or version_data.unowned(n))
        and any(p.vram <= a < p.vram + p.size for a in targets_by_version[v])
    ]
    if not retained:
        return {}
    signatures = defaultdict(list)
    for name, m in snapshot.layout.members.items():
        held = [p for p in m.placements if p.section == ".text" and p.size >= 32]
        if m.kind == "function" and held:
            p = held[0]
            code = masked_code(_code(snapshot, p))
            signatures[code[:32]].append((name, p.size, code, m.holders()))
    claims, table = [], symbols.edit(snapshot)
    for v, p in retained:
        for address in sorted(a for a in targets_by_version[v] if p.vram <= a < p.vram + p.size):
            start = p.rom_start + address - p.vram
            sample = replace(p, rom_start=start, rom_end=min(start + 32, p.rom_end), vram=address)
            matches = [
                (n, size)
                for n, size, code, holders in signatures.get(masked_code(_code(snapshot, sample)), ())
                if v not in holders
                and start + size <= p.rom_end
                and code == masked_code(_code(snapshot, replace(sample, rom_end=start + size)))
            ]
            if len(matches) == 1:
                n, size = matches[0]
                claims.append(Claim("", v, ".text", start, start + size, (n,)))
                row = table.setdefault(n, {"kind": "function"})
                for place in snapshot.layout.members[n].placements:
                    if place.section == ".text":
                        row.setdefault(place.version, place.vram)
                row[v] = address
    if not claims:
        return {}
    writes = {
        **claim_rows(snapshot, claims),
        **symbols.files(table, {v: x.symbols for v, x in snapshot.config.project.version_files.items()}),
    }
    remaining = {
        n
        for v in snapshot.versions.values()
        for n, _, _ in version_data.rows(
            v, lambda p: writes.get(p, snapshot.read(p)), store.content(snapshot.config).cached
        )
    }
    units = {p: u for p, u in snapshot.layout.units.items() if set(u.members) <= remaining}
    if units != snapshot.layout.units:
        writes["layout.toml"] = dump_map(replace(snapshot.layout, units=units))
    return writes


_BRANCHES = frozenset({4, 5, 6, 7, 20, 21, 22, 23})


def _frame(word: int) -> int:
    return 0x10000 - (word & 0xFFFF) if word >> 16 == 0x27BD and word & 0x8000 else 0


def _target(word: int, at: int) -> int | None:
    op = word >> 26
    if op in (2, 3):
        return ((at + 4) & 0xF0000000) | ((word & 0x3FFFFFF) << 2)
    if (
        op in _BRANCHES
        or (op == 1 and (word >> 16 & 31) in (0, 1, 2, 3, 16, 17, 18, 19))
        or (op == 17 and word >> 21 & 31 == 8)
    ):
        return at + 4 + ((word & 0xFFFF ^ 0x8000) - 0x8000) * 4
    return None


def _end(code: Sequence[int], vram: int) -> int | None:
    """The last reachable word including delay slots, only when every path stays inside or terminates."""
    todo, seen, covered = [0], set(), set()
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        if not 0 <= i < len(code):
            return None
        seen.add(i)
        covered.add(i)
        w, at = code[i], vram + 4 * i
        if not version_data.delay_slot(w):
            todo.append(i + 1)
            continue
        if i + 1 >= len(code) or version_data.delay_slot(code[i + 1]):
            return None
        covered.add(i + 1)
        if w == 0x03E00008:
            continue
        op, target = w >> 26, _target(w, at)
        if op == 3 or w & 0xFC00003F == 9:
            todo.append(i + 2)
        elif target is None:
            return None
        elif op == 2 and not vram <= target < vram + len(code) * 4:
            continue
        else:
            todo.append((target - vram) // 4)
            if op != 2 and not (op == 4 and w >> 21 & 31 == w >> 16 & 31):
                todo.append(i + 2)
    return max(covered) + 1 if covered else None


def _unlist(
    snapshot: Snapshot,
    gone: Collection[str],
    writes: dict[str, bytes | None],
    withheld: Mapping[str, Collection[str]] | None = None,
) -> None:
    """The map without members whose rows the writes drop from every split that held them: no group lists them and
    no unit holds them (a unit left holding only padding goes with its file), or the map names rows no split has.
    `withheld`: per unit path the versions it newly holds rows in that its source was never built for (it stays
    withheld there, the rows build from their assembly)."""
    withheld = withheld or {}
    if not gone and not withheld:
        return
    groups = {
        k: replace(g, members=tuple(n for n in g.members if n not in gone)) for k, g in snapshot.layout.groups.items()
    }
    units = {}
    for path, u in snapshot.layout.units.items():
        members = tuple(n for n in u.members if n not in gone)
        if members:
            units[path] = replace(u, members=members, withheld=tuple(sorted({*u.withheld, *withheld.get(path, ())})))
        elif all("_padding_" in n for n in u.members):
            writes[path] = None
    writes["layout.toml"] = dump_map(
        replace(
            snapshot.layout,
            groups=groups,
            units=units,
            fuzzy={n: r for n, r in snapshot.layout.fuzzy.items() if n not in gone},
        )
    )


def _versioned(snapshot: Snapshot, counts: dict[str, Any]) -> dict[str, bytes | None]:
    """The rows of a version whose unowned data row lies between two members exactly where another version holds the
    same instructions (addresses masked) as member rows between the same two members: those rows, rebased, replace
    it (code typed as data in one version only because nothing named it there). A code row comes as assembly: its
    bytes are this version's own, whatever source another version builds it from. The version with the same addresses
    is asked first."""
    from unbake.infer import masked_code  # infer reads layout, so the import waits for the first use

    compiled, assembly, _ = _kinds()

    def masked(v: Version, start: int, end: int) -> bytes:
        blob = version_data.rom_bytes(v, start, end)
        words = struct.unpack(f">{len(blob) // 4}I", blob[: len(blob) // 4 * 4])
        return masked_code([(w, rabbitizer.Instruction(w, vram=0)) for w in words])

    rows = {h: sorted(split_states(snapshot, v), key=lambda r: r[2].rom_start) for h, v in snapshot.versions.items()}
    at = {h: {name: i for i, (name, _, _) in enumerate(found)} for h, found in rows.items()}
    held = {name: set(member.holders()) for name, member in snapshot.layout.members.items()}
    order = sorted(snapshot.versions, key=lambda h: (h != snapshot.config.project.names_from, h))
    writes: dict[str, bytes | None] = {}
    dropped: defaultdict[str, set[str]] = defaultdict(set)  # row name -> the versions whose split no longer has it
    gained: defaultdict[str, set[str]] = defaultdict(set)  # compiled unit -> the versions it now holds code rows in
    table = symbols.edit(snapshot)
    for h, version in sorted(snapshot.versions.items()):
        for i, (name, state, p) in enumerate(rows[h]):
            if not version_data.unowned(name) or not 0 < i < len(rows[h]) - 1:
                continue
            (a, _, pa), (b, _, pb) = rows[h][i - 1], rows[h][i + 1]
            if (
                version_data.unowned(a)
                or version_data.unowned(b)
                or (pa.rom_end, pb.rom_start) != (p.rom_start, p.rom_end)
            ):
                continue
            others = [w for w in order if w != h and a in at[w] and b in at[w] and at[w][b] - at[w][a] >= 2]
            for w in sorted(others, key=lambda w: rows[w][at[w][a]][2].vram != pa.vram):  # same addresses first
                gap = rows[w][at[w][a] + 1 : at[w][b]]
                start, end = rows[w][at[w][a]][2].rom_end, rows[w][at[w][b]][2].rom_start
                if (
                    end - start != p.size
                    or not any(q.section == ".text" for _, _, q in gap)  # code typed as data, never data for data
                    or any(version_data.unowned(n) or h in held.get(n, {h}) for n, _, _ in gap)
                    or gap[0][2].rom_start != start
                    or any(  # each code row the same instructions here; a fill or data row is this version's own
                        masked(snapshot.versions[w], q.rom_start, q.rom_end)
                        != masked(version, p.rom_start + q.rom_start - start, p.rom_start + q.rom_end - start)
                        for _, _, q in gap
                        if q.section == ".text"
                    )
                ):
                    continue
                pieces = tuple((p.rom_start + q.rom_start - start, n) for n, _, q in gap)
                text = edit_split(_written(writes, snapshot, version.split), name, drop=True, additions=pieces)
                for n, kind, q in gap:
                    code = q.section == ".text"
                    text = edit_split(text, n, state=assembly if code else kind, types=frozenset({state}))
                writes[version.split] = text
                for offset, n in pieces:
                    if n in table:  # a name the table gives other versions gets this version's address too
                        table[n][h] = p.vram + offset - p.rom_start
                counts["versioned"] += len(gap)
                dropped[name].add(h)
                for n, _, q in gap:  # C another version builds this code from never had this version's branch
                    if q.section == ".text" and (unit := unit_of(snapshot, n)) and unit.kind == compiled:
                        gained[unit.path].add(h)
                break
    if writes:
        writes.update(
            {
                path: data
                for path, data in symbols.files(
                    table, {v.id: v.symbols_file for v in snapshot.versions.values()}
                ).items()
                if data != snapshot.read(path)
            }
        )
    # a dropped row's member goes from the map too once no split holds it
    gone = {n for n, hs in dropped.items() if hs >= held.get(n, set())}
    _unlist(snapshot, gone, writes, _unproved(snapshot, writes, gone, gained))
    return writes


def _unproved_key(item: tuple[Snapshot, UnitSpec, str]) -> str:
    from unbake import native

    return native.stamp(*item, pinned=True)


def _unproved_job(item: tuple[Snapshot, UnitSpec, str]) -> bool:
    from unbake import native

    proofs = native.prove_job(item)
    return bool(proofs) and all(p.exact for p in proofs)


def _unproved(
    snapshot: Snapshot, writes: Mapping[str, bytes | None], gone: Collection[str], gained: Mapping[str, set[str]]
) -> dict[str, set[str]]:
    """Of the versions a compiled unit gains code rows in, those its C does not build exactly there (proved on the
    rows the writes give): it stays withheld only in those, and builds wherever its C matches."""
    if not gained:
        return {}
    trial = dict(writes)
    _unlist(snapshot, gone, trial)
    over = overlay(snapshot, trial)
    jobs = [
        (over, over.layout.units[p], v)
        for p, vs in sorted(gained.items())
        if p in over.layout.units
        for v in sorted(vs)
    ]
    exact = pool.map(snapshot.config, "layout.versioned", _unproved_job, jobs, _unproved_key)
    failing: dict[str, set[str]] = defaultdict(set)
    for (_, unit, version), ok in zip(jobs, exact, strict=True):
        if not ok:
            failing[unit.path].add(version)
    return failing


def _framed(code: Sequence[int], vram: int, frame: int) -> bool:
    """Whether a body works in a frame of `frame` bytes it does not set up: it gives the frame back (`addiu sp,
    +frame`), or it never returns (no `jr $ra` its entry reaches) and stores into the frame within its first words."""
    if 0x27BD0000 | frame in code:
        return True
    end = _end(code, vram)
    reached = code[:end] if end is not None else code
    return 0x03E00008 not in reached and any(
        w >> 26 in (0x2B, 0x39, 0x3D, 0x3F) and w >> 21 & 31 == 29 and (w & 0xFFFF) < frame for w in code[:8]
    )


_ENTRY = frozenset(  # registers that hold a value when a function is entered: arguments, saved, gp/sp/fp/ra
    [("r", n) for n in (0, 4, 5, 6, 7, *range(16, 24), 28, 29, 30, 31)]
    + [("f", n) for n in (12, 13, 14, 15, *range(20, 32))]
)


def _effects(word: int, vram: int) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    """The registers an instruction reads and the ones it writes, as ("r" | "f", number)."""
    ins = rabbitizer.Instruction(word, vram=vram)
    reads: set[tuple[str, int]] = set()
    writes: set[tuple[str, int]] = set()
    for field, kind in (("rs", "r"), ("rt", "r"), ("rd", "r"), ("fs", "f"), ("ft", "f"), ("fd", "f")):
        key = field[0].upper() + field[1:]
        if getattr(ins, f"reads{key}")():
            reads.add((kind, int(getattr(ins, field).value)))
        if getattr(ins, f"modifies{key}")():
            writes.add((kind, int(getattr(ins, field).value)))
    if ins.isJumpWithAddress() and ins.getOpcodeName() == "jal":
        writes.add(("r", 31))
    return reads, writes


def _entry_reads(before: Sequence[int], code: Sequence[int], vram: int) -> bool:
    """Whether the body, from its first word to its first branch and slot, reads a register that holds nothing at a
    function's entry before it writes it, and the words before it write that register: they are its real start."""
    written = set().union(*(_effects(w, vram + 4 * i)[1] for i, w in enumerate(before)))
    start = vram + 4 * len(before)
    own: set[tuple[str, int]] = set()
    unset: set[tuple[str, int]] = set()
    for i, word in enumerate(code[:64]):
        reads, writes = _effects(word, start + 4 * i)
        unset |= reads - own - _ENTRY
        own |= writes
        if i and version_data.delay_slot(code[i - 1]):
            break
    return bool(unset & written)


def _unreached(code: Sequence[int], vram: int, references: Collection[int] = ()) -> tuple[int, ...]:
    """The word offsets where another routine starts inside a member: past the last word its entry's every path
    reaches (`_end`), the first word that sets up a frame, when nothing references a word between. A piece that is
    one jump and its slot is a fill the prelude rule peels, never a routine. Each piece is walked again from its own
    entry."""
    cuts: list[int] = []
    start = 0
    while (end := _end(code[start:], vram + 4 * start)) is not None and 2 < end < len(code) - start:
        head = next((i for i in range(start + end, len(code)) if _frame(code[i])), None)
        if head is None or any(vram + 4 * i in references for i in range(start + end, head)):
            break
        cuts.append(4 * head)
        start = head
    return tuple(cuts)


def _ends(code: Sequence[int], vram: int) -> bool:
    """The member before ends at its last word: every path from its entry ends there, or (a jump table or a second
    function the walk cannot follow) its last two words are `jr $ra` and a slot that is no jump itself."""
    return _end(code, vram) == len(code) or (code[-2] == 0x03E00008 and not version_data.delay_slot(code[-1]))


def _heads(snapshot: Snapshot, counts: dict[str, Any]) -> tuple[dict[str, bytes | None], set[str]]:
    """Return a detached prologue to its body without changing the body's established name."""
    _, assembly, _ = _kinds()
    protected = {r["name"] for r in snapshot.layout.authored}
    landed = {m for u in snapshot.layout.units.values() for m in u.members}
    held: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for h, version in snapshot.versions.items():
        rows = sorted(split_states(snapshot, version), key=lambda r: r[2].rom_start)
        words, refs = _decoded(snapshot, version, rows)
        for i, (name, _state, p) in enumerate(rows):
            if p.section != ".text" or not words.get(name):
                continue
            code, shift, prefix = words[name], 0, None
            frame = _frame(code[0])
            if not frame and i and not any(_frame(w) for w in code[:4]):
                old, old_state, prev = rows[i - 1]
                if (
                    prev.rom_end != p.rom_start
                    or not 0 < prev.size <= (12 if prev.section == ".data" else 16)
                    or old in protected
                    or p.rom_start in {start for _, start, _, _ in version.segments}
                    or not (
                        (prev.section == ".data" and "_padding_" in old)
                        or (old_state == assembly and old not in landed)
                    )
                ):
                    continue
                blob = version_data.rom_bytes(version, prev.rom_start, prev.rom_end)
                before = struct.unpack(f">{len(blob) // 4}I", blob)
                k = next((j for j, w in enumerate(before) if _frame(w)), len(before))
                if k == len(before) and prev.section == ".data" and _entry_reads(before, code, prev.vram):
                    k = 0  # a leaf's first words: the body reads what only they set
                if (
                    k == len(before)
                    or (prev.section == ".data" and k)
                    or any(version_data.delay_slot(w) for w in before[k:-1])
                    or (version_data.delay_slot(before[-1]) and before[-1] >> 26 != 3)  # a jal's slot is the body's
                ):
                    continue
                if k and (
                    i < 2
                    or rows[i - 2][2].rom_end != prev.rom_start
                    or _end(words.get(rows[i - 2][0], ()), rows[i - 2][2].vram) != rows[i - 2][2].size // 4
                ):
                    continue
                frame = _frame(before[k])
                if (frame and not _framed(code, p.vram, frame)) or any(
                    prev.vram + j * 4 in refs for j in range(k + 1, len(before))
                ):
                    continue
                shift, prefix = (k - len(before)) * 4, (old, prev, k)
            if frame or shift or not any(_frame(w) for w in code):  # a leaf is held too: its versions must agree
                held[name][h] = (shift, frame, prefix, p)
    writes: dict[str, Any] = {}
    table = symbols.edit(snapshot)
    consumed = defaultdict(set)
    blocked: set[Any] = set()
    for name, versions in held.items():
        moved = {h: row for h, row in versions.items() if row[0]}
        if not moved:
            continue
        counts["prelude"]["proposed"] += 1
        reason = (
            "versions disagree"
            if set(versions) != set(snapshot.layout.members[name].holders())
            or len({r[1] for r in versions.values()}) != 1
            else "protected member"
            if name in protected
            or any(
                (name in landed or name in snapshot.layout.fuzzy)
                and snapshot.layout.fuzzy.get(name, {}).get("scores", {}).get(h, 1.0) >= 1.0
                for h in moved
            )
            else ""
        )
        if reason:
            blocked.update((name, *(row[2][0] for row in moved.values())))
            counts["prelude"]["withheld"] += 1
            counts["withheld_reasons"][reason] = counts["withheld_reasons"].get(reason, 0) + 1
            continue
        counts["prelude"]["applied"] += 1
        counts["heads"] += 1
        for h, (shift, _, prefix, p) in moved.items():
            old, prev, k = prefix
            split = snapshot.versions[h].split
            text = edit_split(_written(writes, snapshot, split), name, start=p.rom_start + shift)
            if not k:
                text = edit_split(text, old, drop=True)
                consumed[old].add(h)
            writes[split] = text
            table.setdefault(name, {"kind": "function"})[h] = p.vram + shift
    gone = {n for n, holders in consumed.items() if holders == set(snapshot.layout.members[n].holders())}
    if not writes:
        return {}, blocked
    for n, holders in consumed.items():
        for key in (n, f"D_auto_{n}"):
            for h in holders:
                table.get(key, {}).pop(h, None)
            if key in table and table[key].keys() <= {"kind"}:
                table.pop(key)
    _unlist(snapshot, gone, writes)
    writes.update(
        {
            p: b
            for p, b in symbols.files(table, {v.id: v.symbols_file for v in snapshot.versions.values()}).items()
            if b != snapshot.read(p)
        }
    )
    return writes, blocked


def _entered(
    version: Version,
    text_rows: list[Any],
    words: Mapping[str, tuple[int, ...]],
    rows: list[Any],
    spans: Sequence[tuple[int, int]],
) -> set[tuple[int, int]]:
    """The vram spans that a jump, branch or data pointer from outside the span enters."""
    found: set[tuple[int, int]] = set()
    if not spans:
        return found

    def inside(address: int) -> tuple[int, int] | None:
        return next((s for s in spans if s[0] <= address < s[1]), None)

    for name, _, p in text_rows:
        for i, w in enumerate(words.get(name, ())):
            at = p.vram + 4 * i
            target = _target(w, at) if w >> 26 != 3 else None
            if target is None:
                continue
            if (hit := inside(target)) and not hit[0] <= at < hit[1]:
                found.add(hit)
    points = {a: s for s in spans for a in range(s[0], s[1], 4)}
    for _, _, p in rows:
        if p.section in (".data", ".rodata") and not inside(p.vram):
            blob = version_data.rom_bytes(version, p.rom_start, p.rom_end)
            for value in struct.unpack(f">{len(blob) // 4}I", blob[: len(blob) // 4 * 4]):
                if value in points:
                    found.add(points[value])
    return found


def _group_of(snapshot: Snapshot, region: Mapping[str, Any]) -> str:
    """The module a fill joins: its first member's, else the member before it."""
    members = snapshot.layout.members
    lead = region["rows"][0][0] if region["rows"] and not region["owner"] else region["owner"]
    return members[lead].group or members[region["near"]].group


def _fills(
    snapshot: Snapshot,
    version: Version,
    rows: list[Any],
    words: Mapping[str, tuple[int, ...]],
    references: frozenset[int],
    blocked: Callable[[str], bool],
    protected: set[str],
    landed: set[str],
    inexact: Callable[[str], bool],
) -> list[dict[str, Any]]:
    """Unreferenced alignment words after all reachable paths terminate, excluding stack prologues."""
    _, assembly, _datum = _kinds()
    ordered = sorted(rows, key=lambda r: r[2].rom_start)
    starts = [r[2].rom_start for r in ordered]
    segment_starts = {start for _, start, _, _ in version.segments}
    found: list[dict[str, Any]] = []

    def epilogue(name: str) -> bool:  # a return that gives back a frame no member here took: another member's end
        code = words.get(name, ())
        return len(code) == 2 and code[0] == 0x03E00008 and code[1] >> 16 == 0x27BD and 0 < code[1] & 0xFFFF < 0x8000

    def fill_like(row: tuple[str, str, Placement]) -> bool:
        name, state, p = row
        if p.section == ".text":
            return (
                state == assembly
                and name not in protected
                and name not in landed
                and not blocked(name)
                and (name not in snapshot.layout.fuzzy or (epilogue(name) and inexact(name)))
            )
        return p.section == ".data" and not version_data.unowned(name) and (name not in landed or "_padding_" in name)

    for n, (_, _, p) in enumerate(ordered):
        if n == 0 or p.section != ".text" or p.vram % 16 or p.rom_start in segment_starts:
            continue
        for m in (1, 2, 3):
            lo = p.rom_start - 4 * m
            at = bisect.bisect_right(starts, lo) - 1
            if at < 0 or any(
                ordered[i][2].rom_end != ordered[i + 1][2].rom_start or ordered[i + 1][2].rom_start in segment_starts
                for i in range(at, n)
            ):
                break
            first = ordered[at]
            tail = first[2].rom_start < lo
            owner = first if tail else ordered[at - 1] if at else None
            if (
                owner is None
                or owner[2].section != ".text"
                or owner[0] in protected
                or blocked(owner[0])
                or owner[2].rom_start > lo - 8
                or (tail and (owner[0] in landed or owner[0] in snapshot.layout.fuzzy) and not inexact(owner[0]))
                or _end(words.get(owner[0], ())[: (lo - owner[2].rom_start) // 4], owner[2].vram)
                != (lo - owner[2].rom_start) // 4
                or (not tail and owner[2].rom_end != lo)
            ):
                continue
            inner = ordered[at + tail : n]
            data = version_data.rom_bytes(version, lo, p.rom_start)
            if (
                (not inner and not tail)
                or not data.strip(b"\0")
                or not all(fill_like(r) for r in inner)
                or any(_frame(w[0]) for w in struct.iter_unpack(">I", data))
            ):
                continue
            if not tail and len(inner) == 1 and inner[0][2].section == ".data":
                break  # a fill already made data
            vram = p.vram - 4 * m
            if any(a in references for a in range(vram, p.vram, 4)):
                continue
            found.append(
                {
                    "lo": lo,
                    "vram": vram,
                    "span": (vram, p.vram),
                    "owner": owner[0] if tail else None,
                    "near": owner[0],
                    "owner_state": owner[1],
                    "rows": inner,
                    "data": data,
                }
            )
            break
    return found


_ROUNDS = 4


def _added(left: dict[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """The counts of two boundary rounds as one: every counter adds, the reasons add by name."""
    out = {k: dict(left[k]) if isinstance(left[k], dict) else left[k] + right[k] for k in left}
    for k, v in right.items():
        if isinstance(v, dict):
            for name, n in v.items():
                out[k][name] = out[k].get(name, 0) + n
    return out


def boundary_plan(snapshot: Snapshot) -> tuple[Plan, dict[str, Any]]:
    """The plan is a function of the snapshot and this code, so one computation serves every later command."""
    with effort.stage("layout.boundary_plan"):
        key = digest(
            (
                snapshot.layout.groups,
                snapshot.layout.units,
                snapshot.layout.authored,
                {
                    m: tuple(sorted(v for v, score in row["scores"].items() if score < 1.0))
                    for m, row in snapshot.layout.fuzzy.items()
                },  # which members still differ, not by how much
                snapshot.config.digest,
                snapshot.read(symbols.path()),
                [
                    (v.id, v.rom_sha256, snapshot.read(v.split), version_data.facts_digest(v))
                    for v in snapshot.versions.values()
                ],
                code_identity(boundary_plan),
            )
        )

        def produce() -> tuple[Plan, dict[str, Any]]:
            writes: dict[str, bytes | None] = {**_retained(snapshot)}
            trial = overlay(snapshot, writes) if writes else snapshot
            first: Plan | None = None
            total: dict[str, Any] = {}
            for _ in range(_ROUNDS):  # a fill taken out can leave the next boundary to find: rounds until none is left
                plan, counts = _boundary(trial)
                first, total = (
                    (plan, dict(counts)) if first is None else (first, _added(total, counts) if plan.writes else total)
                )
                writes = {**writes, **plan.writes}
                if not plan.writes or not (counts["fill"] or counts["heads"] or counts["versioned"]):
                    break
                trial = overlay(snapshot, writes)
            assert first is not None  # the rounds run at least once
            if not writes:
                return first, total
            return (
                replace(first, base=snapshot.digest, writes=writes, digest=digest((snapshot.digest, writes))),
                total,
            )

        return store.cached_object(snapshot.config, "boundary", key, produce)


def _decided(candidates: Mapping[str, Mapping[str, Any]], holder: str, name: str) -> bool:
    return bool(candidates.get(name, {}).get(holder))


def _boundary(snapshot: Snapshot) -> tuple[Plan, dict[str, Any]]:
    from unbake.infer import masked_code

    compiled, assembly, datum = _kinds()
    counts: dict[str, Any] = {
        rule: {"proposed": 0, "applied": 0, "withheld": 0} for rule in ("prelude", "split", "merge")
    }
    counts["withheld_reasons"], counts["join"], counts["align"], counts["retype"] = {}, 0, 0, 0
    counts["fill"], counts["heads"], counts["versioned"] = 0, 0, 0
    if versioned := _versioned(snapshot, counts):
        return Plan(
            "layout",
            snapshot.digest,
            versioned,
            (),
            (),
            (),
            "layout: rows another version holds over the same bytes",
            digest((snapshot.digest, versioned)),
        ), counts
    heads, blocked = _heads(snapshot, counts)
    if heads:
        return Plan(
            "layout",
            snapshot.digest,
            heads,
            (),
            (),
            (),
            "layout: restore function prologues",
            digest((snapshot.digest, heads)),
        ), counts
    protected = {row["name"] for row in snapshot.layout.authored} | blocked
    candidates: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {}
    placements: dict[tuple[str, str], Placement] = {}
    slots: set[str] = set()
    fills: dict[str, Any] = {}
    scans: dict[str, Any] = {}
    states: dict[tuple[str, str], str] = {}
    landed = {m for u in snapshot.layout.units.values() for m in u.members}

    def inexact(name: str, holder: str) -> bool:  # a landed member that still differs from the ROM in this version
        return bool(snapshot.layout.fuzzy.get(name, {}).get("scores", {}).get(holder, 1.0) < 1.0)

    for holder, version in sorted(snapshot.versions.items()):
        rows = split_states(snapshot, version)
        words, references = _decoded(snapshot, version, rows)
        scans[holder] = (rows, words, references)
        text_rows = sorted((r for r in rows if r[2].section == ".text"), key=lambda r: r[2].rom_start)
        segment_starts = {start for _, start, _, _ in version.segments}
        linked = [
            b[2].rom_start == a[2].rom_end and b[2].rom_start not in segment_starts  # b follows a in the ROM
            for a, b in pairwise(text_rows)
        ]
        for index, (name, state, p) in enumerate(text_rows):
            placements[holder, name], states[holder, name] = p, state
            decisions: dict[str, tuple[int, ...]] = {}
            previous = text_rows[index - 1] if index and linked[index - 1] else None
            following = text_rows[index + 1] if index < len(linked) and linked[index] else None
            code, prior = words[name], words[previous[0]] if previous else ()
            editable = (name not in landed and name not in snapshot.layout.fuzzy) or inexact(name, holder)
            k = next((i for i, w in enumerate(code[:4]) if _frame(w)), len(code))
            if (
                (not k or (previous is not None and len(prior) >= 2 and _ends(prior, previous[2].vram)))
                and k < len(code)
                and 0x03E00008 not in code[:k]
                and (
                    not k
                    or (
                        name not in protected
                        and editable
                        and (state == assembly or (state == compiled and name in landed))
                    )
                )
                and not any(p.vram + 4 * i in references for i in range(k))
            ):
                body = [(w, rabbitizer.Instruction(w, vram=p.vram + 4 * i)) for i, w in enumerate(code) if i >= k]
                if not any(
                    (ins.isBranch() and 0 <= i + k + ins.getBranchOffset() // 4 < k)
                    or (
                        w >> 26 in (2, 3)
                        and p.vram <= ((p.vram & 0xF0000000) | ((w & 0x3FFFFFF) << 2)) < p.vram + 4 * k
                    )
                    or (w >> 26 == 0 and w & 63 == 8 and w >> 21 & 31 != 31)
                    for i, (w, ins) in enumerate(body)
                ):
                    fills.setdefault(name, {})[holder] = (k, code[:k], masked_code(body))
            owner = (
                prior
                if previous
                and previous[1] == assembly
                and previous[0] not in protected
                and (
                    (previous[0] not in landed and previous[0] not in snapshot.layout.fuzzy)
                    or inexact(previous[0], holder)
                )
                else ()
            )
            if (
                state == assembly
                and name not in protected
                and editable
                and owner
                and version_data.delay_slot(owner[-1])
                and not _frame(code[0])
            ):
                # the jump that ends the member before leaves its delay slot here: the slot goes back to its owner
                decisions["merge" if len(code) == 1 else "prelude"] = (0,) if len(code) == 1 else (4,)
                slots.add(name)
            elif (
                state == assembly
                and name not in protected
                and editable
                and all(r is None or r[1] == assembly for r in (previous, following))
            ):
                k = next((i for i, word in enumerate(code) if word != 0), len(code))
                if previous and k and k < len(code) and p.vram + 4 * k in references and p.vram not in references:
                    decisions["prelude"] = (4 * k,)
                returns = [i for i in range(2, len(code)) if p.vram + 4 * i in references and code[i - 2] == 0x03E00008]
                splits = tuple(
                    4 * i for i, end in zip(returns, [*returns[1:], len(code)], strict=False) if any(code[i:end])
                )  # a piece of only nops is padding: extraction writes no asm for it
                if splits:
                    decisions["split"] = splits
                if (
                    previous
                    and p.vram not in references
                    and not any(_frame(w) for w in code[:4])
                    and not (len(prior) <= 3 and _frame(prior[0]))
                    and _end(prior, previous[2].vram) is None
                    and (len(prior) < 2 or (prior[-2] & 0xFC00003F != 8 and prior[-2] >> 26 != 2))
                    and "prelude" not in decisions
                ):
                    decisions["merge"] = (0,)
            if state == assembly and name not in protected and editable and "merge" not in decisions:
                # a routine no entry path reaches is its own, whatever its neighbours are
                shift = decisions.get("prelude", (0,))[0]
                if pieces := _unreached(code[shift // 4 :], p.vram + shift, references):
                    decisions["split"] = tuple(sorted({*decisions.get("split", ()), *(shift + c for c in pieces)}))
            candidates.setdefault(name, {})[holder] = decisions
    padding: dict[str, dict[str, str]] = {}
    for name, held in fills.items():
        bare = {r[2] for r in held.values() if not r[0]}  # the bodies of the versions with no leading words
        # one body in every version, or each peeled version's body is one an unpeeled version already holds (versions
        # whose code differs elsewhere: the peeled word is still no part of the function)
        same = len({r[2] for r in held.values()}) == 1 or all(r[2] in bare for r in held.values() if r[0])
        if set(held) == set(candidates[name]) and same:
            pads = {h: f"{name}_padding_{placements[h, name].rom_start:X}" for h, r in held.items() if r[0]}
            if (
                pads
                and any(any(r[1]) for r in held.values())
                and not any("split" in d or "merge" in d for d in candidates[name].values())
                and all(len({held[h][1] for h in pads if pads[h] == n}) == 1 for n in pads.values())
            ):
                padding[name] = pads
                candidates[name] = {h: {"prelude": (r[0] * 4,)} if r[0] else {} for h, r in held.items()}
    writes: dict[str, Any] = {}
    replacements: dict[str, tuple[str, ...]] = {}
    added_units: dict[str, UnitSpec] = {}
    table, applied = symbols.edit(snapshot), 0
    for name, by_version in candidates.items():
        proposed = {rule for decisions in by_version.values() for rule in decisions}
        if not proposed:
            continue
        reference = min(by_version, key=lambda h: (h != snapshot.config.project.names_from, h))
        decisions = by_version[reference]
        agrees = set(by_version) == set(snapshot.layout.members[name].holders()) and (
            name in slots or name in padding or all(d == decisions for d in by_version.values())
        )
        base = placements[reference, name].vram
        shift, splits = decisions.get("prelude", (0,))[0], decisions.get("split", ())
        drop = all("merge" in d for d in by_version.values())
        renamed = name if name in landed or name in snapshot.layout.fuzzy else _rename(name, base, base + shift)
        auto = _AUTO.fullmatch(name)  # a piece is named by its own address, whatever address the kept name spells
        split_names = tuple(
            f"{auto[1]}_{base + offset:08X}{auto[3] or ''}" if auto else f"func_{base + offset:08X}"
            for offset in splits
        )
        names = split_names if drop else (renamed, *split_names)
        # a piece that would take a name a member already has would make the split rows name it twice
        taken = len(set(names)) < len(names) or bool(set(split_names) & (snapshot.layout.members.keys() - {name}))
        for rule in proposed:
            total = max(len(offsets) for ds in by_version.values() for r, offsets in ds.items() if r == rule)
            counts[rule]["proposed"] += total
            counts[rule]["applied" if agrees and not taken else "withheld"] += total
        if not agrees or taken:
            reason = "versions disagree" if not agrees else "name taken"
            counts["withheld_reasons"][reason] = counts["withheld_reasons"].get(reason, 0) + 1
            continue
        if names != (name,):
            replacements[name] = names
        applied += sum(map(len, decisions.values()))
        if not drop and renamed != name and name in table:
            table[renamed] = {**table.get(renamed, {}), **table.pop(name)}
        for holder in sorted(by_version):
            version, p = snapshot.versions[holder], placements[holder, name]
            cuts = tuple(zip(splits, split_names, strict=True))
            mine = by_version[holder]
            here, gone = mine.get("prelude", (0,))[0], "merge" in mine
            text = _written(writes, snapshot, version.split)
            if pad := padding.get(name, {}).get(holder):
                text = edit_split(text, name, state=datum, rename=pad, additions=((p.rom_start + here, renamed),))
                text = edit_split(text, renamed, state=states[holder, name], types=frozenset({datum}))
                symbol, values, path = f"D_auto_{pad}", fills[name][holder][1], f"src/{pad}.c"
                pad_bytes = b"".join(word.to_bytes(4, "big") for word in values)
                writes[path] = (
                    f"unsigned char {symbol}[{len(pad_bytes)}] = {{ "
                    + ", ".join(f"0x{byte:02X}" for byte in pad_bytes)
                    + " };\n"
                ).encode()
                table.setdefault(symbol, {"kind": datum})[holder] = p.vram
                added_units[path] = UnitSpec(
                    path,
                    datum,
                    snapshot.layout.members[name].group,
                    (pad,),
                    snapshot.config.project.toolchain,
                    {"add": [], "omit": []},
                )
            else:
                text = edit_split(
                    text,
                    name,
                    start=p.rom_start + here if here else None,
                    rename=renamed,
                    drop=gone,
                    additions=tuple((p.rom_start + o, new) for o, new in cuts),
                )
            writes[version.split] = text
            if gone:
                table.get(name, {}).pop(holder, None)
            else:
                table.setdefault(renamed, {"kind": "function"})[holder] = p.vram + here
            for offset, new in cuts:
                table.setdefault(new, {"kind": "function"})[holder] = p.vram + offset
        if drop and table.get(name, {}).keys() <= {"kind"}:
            table.pop(name, None)
    regions: dict[str, list[dict[str, Any]]] = {}
    for holder, (rows, words, references) in scans.items():
        version = snapshot.versions[holder]
        found = _fills(
            snapshot,
            version,
            rows,
            words,
            references,
            partial(_decided, candidates, holder),
            protected,
            landed,
            partial(inexact, holder=holder),
        )
        text_rows = [r for r in rows if r[2].section == ".text"]
        entered = _entered(version, text_rows, words, rows, [r["span"] for r in found])

        def shared(r: Mapping[str, Any], holder: str = holder) -> bool:
            return (
                bool(r["rows"])
                and not r["owner"]
                and "_padding_" in r["rows"][0][0]
                and (snapshot.layout.members[r["rows"][0][0]].holders() != (holder,))
            )

        regions[holder] = [r for r in found if r["span"] not in entered and not shared(r) and _group_of(snapshot, r)]
    while True:  # a member that is fill in one version is fill in all its versions
        seen = defaultdict(set)
        for holder, items in regions.items():
            for r in items:
                for n, _, p in r["rows"]:
                    if p.section == ".text":
                        seen[n].add(holder)
        bad = {n for n, h in seen.items() if h != set(snapshot.layout.members[n].holders())}
        kept = {h: [r for r in items if not bad & {n for n, _, _ in r["rows"]}] for h, items in regions.items()}
        if kept == regions:
            break
        regions = kept
    pad_data: defaultdict[str, set[bytes]] = defaultdict(set)
    for items in regions.values():
        for r in items:
            first = r["rows"][0][0] if r["rows"] and not r["owner"] else r["owner"]
            r["pad"] = first if "_padding_" in first and not r["owner"] else f"{first}_padding_{r['lo']:X}"
            pad_data[r["pad"]].add(r["data"])
    consumed, fill_pads, fill_after = defaultdict(set), defaultdict(list), defaultdict(list)
    for holder, items in sorted(regions.items()):
        version = snapshot.versions[holder]
        for r in items:
            pad = r["pad"] if len(pad_data[r["pad"]]) == 1 else f"{r['pad']}_{holder.replace('-', '_')}"
            text = _written(writes, snapshot, version.split)
            lead = r["rows"][0] if r["rows"] and not r["owner"] else None
            if r["owner"]:
                text = edit_split(text, r["owner"], additions=((r["lo"], pad),))
                text = edit_split(text, pad, state=datum, types=frozenset({r["owner_state"]}))
                fill_after[r["owner"]].append(pad)
            elif lead is not None and lead[0] != pad:
                text = edit_split(text, lead[0], state=datum, rename=pad)
                fill_pads[lead[0]].append(pad)
            for name, _, _ in r["rows"][not r["owner"] :]:
                text = edit_split(text, name, drop=True)
            writes[version.split] = text
            for name, _, _ in r["rows"]:
                if name != pad:
                    consumed[name].add(holder)
                    table.get(name, {}).pop(holder, None)
                    if name in table and table[name].keys() <= {"kind"}:
                        table.pop(name)
            symbol, blob = f"D_auto_{pad}", r["data"]
            holder_unit = unit_of(snapshot, pad)
            path = holder_unit.path if holder_unit else f"src/{pad}.c"
            writes[path] = (
                f"unsigned char {symbol}[{len(blob)}] = {{ " + ", ".join(f"0x{byte:02X}" for byte in blob) + " };\n"
            ).encode()
            table.setdefault(symbol, {"kind": datum})[holder] = r["vram"]
            if path not in snapshot.layout.units:
                added_units[path] = UnitSpec(
                    path,
                    datum,
                    _group_of(snapshot, r),
                    (pad,),
                    snapshot.config.project.toolchain,
                    {"add": [], "omit": []},
                )
            counts["prelude"]["proposed"] += 1
            counts["prelude"]["applied"] += 1
            counts["fill"] += 1
            applied += 1
    fill_gone = {n for n, h in consumed.items() if h >= set(snapshot.layout.members[n].holders())}
    rows_after: dict[str, list[tuple[str, str, Placement]]] = {}  # the rows as they will be, misaligned ones cut
    raw = {k: next(t for t in row["types"] if not t.startswith(".")) for k, row in configuration.sections().items()}
    for holder, version in sorted(snapshot.versions.items()):
        rows_after[holder] = []
        tables = [a for n, a in version.symbols.items() if n.startswith("jtbl_") and n in table]
        for name, was, p in split_states(snapshot, version):
            gap = -p.vram % 4
            cut = p.section in (".data", ".rodata") and version_data.unowned(name) and 0 < gap < p.size
            retype = (
                p.section == ".data"
                and not was.startswith(".")
                and version_data.unowned(name)
                and any(p.vram <= a < p.vram + p.size for a in tables)
            )
            state = raw["rodata"] if retype else was  # only read-only data has its jump table words labelled
            rows_after[holder].append((name, state, p))
            if cut or retype:
                fresh = f"{version_data.section_of(version, state, name)[1:]}/unresolved/{p.vram + gap:08X}"
                writes[version.split] = edit_split(
                    _written(writes, snapshot, version.split),
                    name,
                    state=state if retype else None,
                    types=frozenset({was}),
                    additions=((p.rom_start + gap, fresh),) if cut else (),
                )
            if cut:
                rows_after[holder][-1:] = [
                    (name, state, replace(p, rom_end=p.rom_start + gap)),
                    (fresh, state, replace(p, rom_start=p.rom_start + gap, vram=p.vram + gap)),
                ]
            counts["align"] += cut
            counts["retype"] += retype
    reference, spelled = snapshot.config.project.names_from, defaultdict[str, list[tuple[str, str]]](list)
    for holder in sorted(snapshot.versions, key=lambda v: (v != reference, v)):
        at = {a: n for n, a in snapshot.versions[holder].symbols.items()}
        for name, _, p in rows_after[holder]:
            declared = at.get(p.vram)
            if p.section in (".data", ".rodata") and declared is not None and declared in table:  # declared names only
                spelled[declared].append((holder, name))
    moves: dict[tuple[str, str], str] = {}
    for shown in spelled.values():  # a landed name is authoritative: it is kept, never renamed, never given a version
        keep = next((n for _, n in shown if n in landed), shown[0][1])
        held_by = snapshot.layout.members[keep].holders() if keep in landed else None
        moves.update(
            {(h, n): keep for h, n in shown if n != keep and n not in landed and (held_by is None or h in held_by)}
        )
    held = {h: {n for n, _, _ in items} for h, items in rows_after.items()}
    while step := [(h, n, k) for (h, n), k in moves.items() if k not in held[h]]:  # a row takes only a free name
        for holder, name, keep in step:
            held[holder] -= {name}
            held[holder] |= {keep}
            del moves[holder, name]
            split = snapshot.versions[holder].split
            with suppress(Refusal):  # a row written as a mapping has no line to rename
                writes[split] = edit_split(_written(writes, snapshot, split), name, rename=keep)
                replacements[name] = (keep,)
                counts["join"] += 1
    # the symbol files follow the table
    wanted = symbols.files(table, {v.id: v.symbols_file for v in snapshot.versions.values()})
    writes.update({path: text for path, text in wanted.items() if text != snapshot.read(path)})
    if replacements or added_units or fill_gone:

        def expanded(names: Iterable[str]) -> tuple[str, ...]:
            return tuple(new for name in names for new in replacements.get(name, (name,)))

        def listed(n: str) -> tuple[str, ...]:
            return (
                *dict.fromkeys(padding.get(n, {}).values()),
                *(() if n in fill_gone else expanded((n,))),
                *dict.fromkeys(fill_pads.get(n, ())),
                *dict.fromkeys(fill_after.get(n, ())),
            )

        groups = {
            k: replace(g, members=tuple(pad for n in g.members for pad in listed(n)))
            for k, g in snapshot.layout.groups.items()
        }
        units = {
            k: replace(u, members=expanded(u.members)) for k, u in snapshot.layout.units.items() if expanded(u.members)
        }
        fuzzy: dict[str, Json] = {}
        for old, row in snapshot.layout.fuzzy.items():
            if old in fill_gone:
                continue
            names = expanded((old,))
            if names:
                fuzzy.setdefault(names[0], {**row, "scores": {}})["scores"].update(row["scores"])
        writes["layout.toml"] = dump_map(
            replace(snapshot.layout, groups=groups, units={**units, **added_units}, fuzzy=fuzzy)
        )
    message = f"layout: {applied + counts['join'] + counts['align'] + counts['retype']} boundary edits"
    return Plan(
        "layout", snapshot.digest, writes, (), (), (), message, digest(("layout", snapshot.digest, writes, message))
    ), counts


def renamed(name: str, renames: Mapping[str, str]) -> str:
    parts = name.split("/")
    if name in renames:
        return renames[name]
    if len(parts) == 3 and parts[1] in renames:
        return f"{parts[0]}/{renames[parts[1]]}/{parts[2]}"
    return name


def _inline(head: str) -> bool:
    return bool(re.search(r"\b(?:inline|__inline|__inline__)\b", head))


def _emits(block: str) -> bool:
    """Whether a block gives bytes a place: an object definition, or a function definition that is not inline."""
    raw = block.encode()
    return bool(headers.variables(raw)) or any(not _inline(head) for _, head in headers.signatures(raw))


def _external(block: str, heads: Sequence[tuple[str, str]]) -> tuple[str, list[str]]:
    """The block with `static` dropped from each function head, and the prototype of each: a function one unit owns
    and other units call is external, its prototype goes to the owner's header (fold) and the callers declare it."""
    protos = []
    for name, head in heads:
        block = re.sub(rf"\bstatic\s+(?=[^;{{}}]*\b{re.escape(name)}\s*\()", "", block, count=1)
        protos.append(re.sub(r"\bstatic\s+", "", head).strip() + ";")
    return block, protos


def _data_member(final: LayoutMap, snapshot: Snapshot, name: str) -> bool:
    """Whether the member is a data row (not a function)."""
    member = final.members.get(name) or snapshot.layout.members.get(name)
    return member is not None and member.kind != "function"


def _holder(order: Sequence[str], key: str, owner_of: Mapping[str, str]) -> str | None:
    """The unit that owns the ROM bytes of a non-member block: the compiler emits it in text order, inside the extent
    of the member before it, else before the member after it."""
    at = order.index(key)
    for k in [*reversed(order[:at]), *order[at + 1 :]]:
        if k in owner_of:
            return owner_of[k]
    return None


def _scope_rows(blocks: Mapping[str, str]) -> dict[str, list[tuple[str, str, str]]]:
    """Per shared block of one old unit the declarations it makes: (name, "type" or "decl", spelling). A type is a
    typedef, a record or an enum; a decl is an extern object or a prototype that declares one name alone."""
    rows: dict[str, list[tuple[str, str, str]]] = {}
    for key, block in blocks.items():
        if not key.startswith("#"):
            continue
        found: list[tuple[str, str, str]] = []
        for _, _, statement in headers.statements(block):
            if names := headers._type_names(statement):
                found += [(n, "type", headers._canon(statement)) for n in sorted(names)]
            elif "{" not in statement or re.match(r"\s*extern\b", statement):  # an extern of an unnamed record too
                declared = headers.declared_names(statement)
                if len(declared) == 1:
                    name = declared.pop()
                    found.append((name, "decl", headers._spelling(statement, name)))
                elif declared:
                    found += [(n, "many", "") for n in sorted(declared)]
        rows[key] = found
    return rows


def _scoped(block: str, line: str) -> str:
    """The member block with `line` as the first line of its function body."""
    for start, _, statement in headers.statements(block, bodies=True):
        if statement.rstrip().endswith("}") and "{" in statement:
            at = start + statement.index("{") + 1
            return block[:at] + "\n    " + line + block[at:]
    return block


def _without(block: str, name: str) -> tuple[str, str]:
    """The block without the one statement that declares `name`, and that statement."""
    for start, end, statement in headers.statements(block):
        if headers.declared_names(statement) == {name}:
            return block[:start] + block[end:], " ".join(block[start:end].split())
    return block, ""


def _localized(
    unit: UnitSpec,
    sources: Sequence[str],
    take: Mapping[tuple[str, str], set[str]],
    scope: Mapping[str, Mapping[str, list[tuple[str, str, str]]]],
    split: Mapping[str, tuple[list[str], dict[str, str]]],
) -> tuple[dict[str, dict[str, str]], str]:
    """The extern objects and prototypes the old units of one new unit declare with different types: per name, per
    old unit, its declaration, which moves to block scope in that unit's functions. The second value says why the
    units cannot be one: a type defined two ways, a name one of them defines, or a declaration of several names."""
    if len(sources) < 2:
        return {}, ""
    types: defaultdict[str, dict[str, str]] = defaultdict(dict)
    decls: defaultdict[str, dict[str, str]] = defaultdict(dict)
    many: set[str] = set()
    for path in sources:
        for key in take.get((path, unit.path), ()):
            for name, kind, spelled in scope.get(path, {}).get(key, ()):
                if kind == "type":
                    types[name].setdefault(path, spelled)
                elif kind == "decl":
                    decls[name].setdefault(path, spelled)
                else:
                    many.add(name)
    for name, by in sorted(types.items()):
        if len(set(by.values())) > 1:
            return {}, f"{name} is defined differently by {', '.join(sorted(by))}"
    local: dict[str, dict[str, str]] = {}
    defined = {n for p in sources for n in split[p][1] if not n.startswith("#")}
    for name, by in sorted(decls.items()):
        if len(set(by.values())) < 2:
            continue
        if name in defined or name in many:
            return {}, f"{name} is declared differently by {', '.join(sorted(by))} and cannot move to block scope"
        lines = {}
        for path in by:
            for key in take.get((path, unit.path), ()):
                if any(r[0] == name for r in scope[path].get(key, ())):
                    lines[path] = _without(split[path][1][key], name)[1]
        local[name] = lines
    return local, ""


def _header_names(snapshot: Snapshot, includes: Sequence[str], memo: dict[str, set[str]]) -> set[str]:
    """Every name the headers the include lines reach declare (quoted includes, under include/ or src/)."""
    found: set[str] = set()
    pending = [m[1] for line in includes if (m := re.search(r'"([^"]+)"', line))]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        for path in (f"include/{name}", f"src/{name}"):
            if path in seen or (content := snapshot.peek(path)) is None:
                continue
            seen.add(path)
            if path not in memo:
                memo[path] = headers.names(content.decode(errors="replace"))
            found |= memo[path]
            pending += headers._quoted(content)
    return found


def compose(
    snapshot: Snapshot, final: LayoutMap, was: Mapping[str, str], header: Callable[[str], bool]
) -> tuple[dict[str, bytes], dict[str, str], list[tuple[str, Finding]]]:
    """The text of every unit of the new map that is not its old self: carried when its atoms are one old unit's,
    else composed from the blocks of the old units that held them. Returns the texts, the old text each path is judged
    against and the old units' problems."""
    old = snapshot.layout.units
    held = {n: u for u in old.values() for n in u.members}

    class Texts(dict[str, str]):  # an old unit is read when a unit takes from it, never otherwise
        def __missing__(self, path: str) -> str:
            self[path] = value = snapshot.read(path).decode(errors="replace")
            return value

    text = Texts()
    out: dict[str, bytes] = {}
    before: dict[str, str] = {}
    problems: list[tuple[str, Finding]] = []
    split: dict[str, tuple[list[str], dict[str, str]]] = {}
    fed: defaultdict[str, list[tuple[UnitSpec, list[str]]]] = defaultdict(list)
    for unit in final.units.values():
        names = [renamed(n, was) for n in unit.members]
        feeds = {held[n].path for n in names if n in held}
        same = old.get(unit.path)
        if same and same.members == tuple(names) and same.group == unit.group:
            continue  # nothing about this unit moved: only its include lines can change
        lead = f'#include "{final.groups[unit.group].header.removeprefix("include/")}"'
        if len(feeds) == 1 and set(old[(only := next(iter(feeds)))].members) == set(names):
            before[unit.path] = text[only]
            if old[only].group != unit.group and header(final.groups[unit.group].header):
                out[unit.path] = headers.ordered(
                    headers.included(text[only], [lead.split('"')[1]]), lead.split('"')[1]
                ).encode()
            elif only != unit.path:
                out[unit.path] = text[only].encode()
            continue
        for path in sorted(feeds):
            fed[path].append((unit, [n for n in names if n in held and held[n].path == path]))
    take: dict[tuple[str, str], set[str]] = {}
    feeds_of: defaultdict[str, set[str]] = defaultdict(set)  # per new unit the old units that feed it
    for source, feeding in fed.items():
        for fed_unit, _ in feeding:
            feeds_of[fed_unit.path].add(source)
    scope: dict[str, dict[str, list[tuple[str, str, str]]]] = {}  # per old unit its shared blocks' declarations
    reached: dict[str, set[str]] = {}
    externs: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    definers: dict[str, set[str]] | None = None

    def clash(path: str, heads: Sequence[tuple[str, str]]) -> str:
        """Why a function made external would collide with a definition of its name in another source, or ''."""
        nonlocal definers
        if definers is None:
            definers = headers.definers(snapshot)  # read once, only when a function is made external
        for name, _ in heads:
            if others := sorted(definers.get(name, set()) - {path}):
                return f"{name} made external would collide with its definition in {others[0]}"
        return ""

    for path, units in fed.items():
        try:
            split[path] = headers.split_atoms(text[path], old[path].members)
        except Refusal as refusal:
            problems += [(path, f) for f in refusal.findings]
            continue
        blocks = split[path][1]
        shared = {k: v for k, v in blocks.items() if k.startswith("#")}
        if any(len(feeds_of[u.path]) > 1 for u, _ in units):  # only a unit fed by several old units can disagree
            scope[path] = _scope_rows(shared)
        words_of: dict[str, set[str]] = {}

        def close(shared: Mapping[str, str], words: set[str], got: set[str]) -> None:
            grew = True
            while grew:
                grew = False
                for key, body in shared.items():
                    if key not in got and (not (declared := headers.names(body)) or declared & words):
                        got.add(key)
                        words |= set(re.findall(r"\w+", body))
                        grew = True

        for unit, names in units:
            words_of[unit.path] = set(re.findall(r"\w+", " ".join(blocks.get(n, "") for n in names)))
            take[path, unit.path] = set()
            close(shared, words_of[unit.path], take[path, unit.path])
        owner_of = {n: u.path for u, names in units for n in names}
        made: set[str] = set()
        static = Finding("move.static", "a file-local definition is used by several units", unit=path)
        for key, body in shared.items():
            users = [u.path for u, _ in units if key in take[path, u.path]]
            heads = headers.signatures(body.encode())
            inline = [h for h in heads if _inline(h[1])]
            if len(users) > 1 and (
                headers.variables(body.encode())
                or (inline and len(inline) < len(heads))
                or (inline and "static" in body.partition("{")[2])  # each copy would keep its own static
            ):
                problems.append((path, static))
            elif len(users) > 1 and heads and not inline and (other := clash(path, heads)):
                problems.append((path, replace(static, reason=other)))
            elif len(users) > 1 and heads and not inline:
                # a plain function copied to each unit would emit its code twice: one unit keeps it, the rest call it
                holder = _holder(list(blocks), key, owner_of) or users[0]
                external, protos = _external(body, heads)
                made |= {f for f, _ in heads}
                blocks[key] = shared[key] = "\n".join(protos) + "\n\n" + external
                for user in users:
                    if user != holder:
                        take[path, user].discard(key)
                        externs[path, user] += [f"extern {p}" for p in protos]
                take[path, holder].add(key)
                close(shared, words_of[holder] | set(re.findall(r"\w+", external)), take[path, holder])
            if not users:  # read by no member: an unread object's bytes sit among the old unit's data rows
                rows = [u.path for u, names in units if any(_data_member(final, snapshot, n) for n in names)]
                take[path, rows[0] if len(rows) == 1 and headers.variables(body.encode()) else units[0][0].path].add(
                    key
                )
        for unit, names in units:  # a member static function that another unit calls is made external
            for n in names:
                every = headers.signatures(blocks.get(n, "").encode())
                heads = [h for h in every if not _inline(h[1])]
                called = {f for f, _ in every}
                callers = [u.path for u, _ in units if u.path != unit.path and called & words_of[u.path]]
                if callers and len(heads) < len(every):  # an inline member keeps its bytes in one unit only
                    problems.append((path, replace(static, reason=f"{n} is an inline member another unit calls")))
                elif callers and (other := clash(path, heads)):
                    problems.append((path, replace(static, reason=other)))
                elif heads and callers:
                    external, protos = _external(blocks[n], heads)
                    made |= {f for f, _ in heads}
                    blocks[n] = "\n".join(protos) + "\n\n" + external
                    for caller in callers:
                        externs[path, caller] += [f"extern {p}" for p in protos]
        for name in made:  # a static forward declaration of a function made external would contradict it
            forward = re.compile(rf"\bstatic\s+(?=[^;{{}}]*\b{re.escape(name)}\s*\([^;{{}}]*\)\s*;)")
            for k, block in blocks.items():
                blocks[k] = forward.sub("", block)
    for unit in final.units.values():
        if unit.path in out or not any(unit.path == u.path for units in fed.values() for u, _ in units):
            continue
        includes, preamble, pieces, seen = [], [], [], set()
        group_header = final.groups[unit.group].header
        if header(group_header):
            includes.append(f'#include "{group_header.removeprefix("include/")}"')
        members = [renamed(n, was) for n in unit.members]
        sources = list(dict.fromkeys(held[n].path for n in members if n in held and held[n].path in split))
        local, conflict = _localized(unit, sources, take, scope, split)
        if conflict:  # every old unit's moves made the clash: each is named, so dropping any one's moves is not enough
            found = Finding("move.member_conflict", conflict, unit=", ".join(sources))
            problems += [(p, found) for p in sources]
            local = {}  # the text is still composed as is: a caller that keeps going writes it with the clash
        if local:
            spelled = _header_names(snapshot, [*includes, *(i for p in sources for i in split[p][0])], reached)
            if declared := sorted(local.keys() & spelled):
                reason = f"{declared[0]} is declared differently by {', '.join(sources)} and by a header they include"
                found = Finding("move.member_conflict", reason, unit=", ".join(sources))
                problems += [(p, found) for p in sources]
                local = {}
        emitted: set[str] = set()
        # a shared block that emits bytes (a function that is not inline, an object) keeps its place among the
        # members of its old unit: in the preamble it would move ahead of them and change their code and data order
        placed: dict[str, list[tuple[int, str]]] = {}
        last = {held[n].path: n for n in members if n in held}
        for n in members:
            if n in held and held[n].path in split:
                path = held[n].path
                order = {k: i for i, k in enumerate(split[path][1])}
                if path not in seen:
                    seen.add(path)
                    includes += [i for i in split[path][0] if i not in includes]
                    for k in sorted(take[path, unit.path], key=lambda k: int(k[1:])):
                        block = split[path][1][k]
                        for name in sorted(local):
                            if path in local[name]:
                                block = _without(block, name)[0]
                                if re.search(rf"\b{re.escape(name)}\b", block):  # a shared function (inline) uses it
                                    block = _scoped(block, local[name][path])
                        canon = headers._canon(block)
                        if canon and canon in emitted and not _emits(block):
                            continue  # the same type or declaration from two old units is written once (a repeated
                            # typedef, even a forward one, does not compile)
                        emitted.add(canon)
                        if _emits(block):
                            placed.setdefault(path, []).append((order[k], block))
                        else:
                            preamble.append(block)
                    preamble += list(dict.fromkeys(externs.get((path, unit.path), ())))
                    before[unit.path] = before.get(unit.path, "") + "\n" + text[path]
                at = order.get(n, len(order))
                pieces += [b for i, b in placed.get(path, ()) if i < at]
                placed[path] = [(i, b) for i, b in placed.get(path, ()) if i >= at]
                if n in split[path][1]:
                    piece = split[path][1][n]
                    for name, lines in sorted(local.items()):
                        if path in lines and re.search(rf"\b{re.escape(name)}\b", piece):
                            piece = _scoped(piece, lines[path])
                    pieces.append(piece)
                if last[path] == n:  # the blocks after its last member follow it, before the next old unit's
                    pieces += [b for _, b in placed.pop(path, ())]
        own = group_header.removeprefix("include/")
        out[unit.path] = headers.render(headers.include_order(includes, own), preamble, pieces)
    return out, before, problems


def rename_writes(
    snapshot: Snapshot, table: dict[str, dict[str, Any]], renames: Mapping[str, str]
) -> dict[str, bytes | None]:
    """The symbol files, split rows and configuration the symbol moves change."""
    project = snapshot.config.project
    writes: dict[str, bytes | None] = {}
    if not renames:
        return writes
    for path, data in symbols.files(table, {v: vf.symbols for v, vf in project.version_files.items()}).items():
        if snapshot.peek(path) != data:
            writes[path] = data
    owners: defaultdict[str, list[str]] = defaultdict(list)
    for name in snapshot.layout.members:
        if name.count("/") == 2:
            owners[name.split("/")[1]].append(name)
    for old, new in renames.items():
        for name in [old, *owners.get(old, ())]:
            for holder in snapshot.layout.members[name].holders() if name in snapshot.layout.members else ():
                split = snapshot.versions[holder].split
                current = writes.get(split) or snapshot.read(split)
                writes[split] = edit_split(current, name, rename=renamed(name, {old: new}))
    config_text = snapshot.read("config.toml").decode()
    changed = config_text
    for old, new in renames.items():
        changed = re.sub(rf'(?m)^(\s*function\s*=\s*)"{re.escape(old)}"', rf'\1"{new}"', changed)
    if changed != config_text:
        writes["config.toml"] = changed.encode()
    return writes
