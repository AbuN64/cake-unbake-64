"""Toolchain installation and the generated whole-ROM build files: Makefile, link script, splat yaml."""

from __future__ import annotations

import json
import os
import re
import shlex
import tarfile
import urllib.request
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from unbake import adapters, effort, pool, process, report, store, symbols, tree, versions
from unbake import config as configuration
from unbake.contracts import (
    Config,
    Finding,
    Json,
    NativeResult,
    Project,
    Refusal,
    Segment,
    file_facts,
    macro,
    sha256,
    sha256_file,
)

if TYPE_CHECKING:
    from unbake.tree import Tree

_REPO = configuration.load_resource("repo.toml")
_CODE = frozenset({"func"})  # symbol_addrs types whose bytes are code
_C = "c"  # the splat subsegment type of code
_NOT_DATA = frozenset({"label", "jtbl_label", "alabel"})  # splat types that name a place inside other bytes
_MAKE = {row["host"]: f"$({var})" for var, row in _REPO["make"]["tools"].items()}
_GENERATED = "generated from config.toml; do not edit"  # names no tool: the repo builds without one


def _pins_bad(dest: Path, pins: Json) -> list[str]:
    """Pinned files that are missing or hash wrong. A hash is remembered with the file's size and mtime."""
    manifest = dest / ".pin-manifest.json"
    try:
        known = json.loads(manifest.read_text())
    except (OSError, ValueError):
        known = {}
    bad, fresh = [], {}
    for rel, want in sorted(pins.items()):
        path = dest / rel
        if not path.is_file():
            bad.append(rel)
            continue
        stat, row = path.stat(), known.get(rel)
        have = row[2] if row and row[:2] == [stat.st_mtime_ns, stat.st_size] else sha256_file(path)
        fresh[rel] = [stat.st_mtime_ns, stat.st_size, have]
        if have != want:
            bad.append(rel)
    if fresh != known and dest.is_dir():
        store.write(manifest, json.dumps(fresh).encode())
    return bad


def _fetch(config: Config, row: Json) -> Path:
    cached = config.host.toolchain_root / ".downloads" / str(row["sha256"])
    if cached.is_file() and sha256_file(cached) == row["sha256"]:
        return cached
    try:
        with urllib.request.urlopen(row["url"]) as response:
            data = response.read()
    except OSError as error:
        raise Refusal(Finding("toolchain.download", reason=f"{row['url']}: {error}", path=row["url"])) from error
    if sha256(data) != row["sha256"]:
        raise Refusal(
            Finding("toolchain.download", reason=f"{row['url']} does not match sha256 {row['sha256']}", path=row["url"])
        )
    store.write(cached, data)
    return cached


def _install(item: tuple[Config, str]) -> Path:
    return install_toolchain(*item)


def install_toolchains(config: Config, ids: Sequence[str]) -> list[Path]:
    """Every toolchain installed at once: each is a download and an unpack that waits on the network and one core."""
    return pool.map(config, "build.install_toolchains", _install, [(config, id) for id in ids])


def install_toolchain(config: Config, id: str) -> Path:
    with effort.stage("build.install_toolchain"):
        rows = configuration.toolchains()
        if id not in rows:
            raise Refusal(Finding("toolchain.pin", reason=f"toolchain {id} is not in toolchains.toml"))
        row, dest = rows[id], config.host.toolchain_root / id
        if not _pins_bad(dest, row["pins"]):
            return dest
        for download in row["downloads"]:
            with tarfile.open(_fetch(config, download)) as archive:
                names = set(archive.getnames())
                absent = [f for f in download["files"] if f not in names]
                if absent:
                    raise Refusal(
                        Finding(
                            "toolchain.download",
                            reason=f"{download['url']} lacks {absent}",
                            path=download["url"],
                            missing=tuple(absent),
                        )
                    )
                dest.mkdir(parents=True, exist_ok=True)
                archive.extractall(dest, members=[archive.getmember(f) for f in download["files"]], filter="data")
            for member in download["files"]:
                path = dest / member
                if path.is_file():
                    path.chmod(path.stat().st_mode | 0o111)
        bad = _pins_bad(dest, row["pins"])
        if bad:
            raise Refusal(
                Finding(
                    "toolchain.pin",
                    reason=f"{id}: {', '.join(bad)} do not match their pins",
                    path=str(dest),
                    missing=tuple(bad),
                )
            )
        return dest


def _key(id: str) -> str:
    return id.replace("-", "_").replace(".", "_")


def _shell(argv: Sequence[object]) -> str:
    return " ".join(str(w) if str(w).startswith("$") or "$(" in str(w) else shlex.quote(str(w)) for w in argv)


def _flags(row: Json, config: Config, added: Sequence[str] = (), omit: Sequence[str] = ()) -> list[str]:
    """A toolchain row's flags and the project's, less what a [file] row omits, plus the flags the row adds."""
    return [f for f in (*row["cflags"], *config.project.cflags) if f not in omit] + list(added)


def _recipe(id: str) -> str:
    """The commands that turn one source into an object with a toolchain, spelled in make variables: PREPROCESS_<id>
    writes the preprocessed text next to the object's .i (the rule keeps the .i when the bytes are the same), and
    COMPILE_<id> turns that .i into the object, as one shell command (the object rule runs it only when the .i or
    another input changed)."""
    row, key = configuration.toolchain(id), _key(id)
    out = "$(@:.o=.s)" if row["emits_asm"] else "$@"
    values = {
        **{k: [v] for k, v in _MAKE.items()},
        "cc": [f"$(CC_{key})"],
        "as": [f"$(AS_{key})"],
        # -MP: a phony rule per header, so a header a later commit deletes never stops an incremental build
        "cppflags": ["$(CPPFLAGS) -MMD -MP -MF $(@:.i.ok=.d) -MT $@"],
        "codegen": ["$(CFLAGS)"],
        "asflags": ["$(ASFLAGS) $(ASINC)"],
        "defines": ["$(DEFINES)"],
        "includes": ["$(INCLUDES)"],
    }
    preprocess = _shell(adapters.render(row["preprocess"], {**values, "source": ["$<"]})) + " > $(@:.ok=.new)"
    lines = [_shell(adapters.render(row["compile"], {**values, "source": ["$(@:.o=.i)"], "out": [out]}))]
    if row["emits_asm"]:
        keep = "$(@:.o=.s.prev)"
        lines.insert(0, f"{{ [ ! -f $(@:.o=.s) ] || mv -f $(@:.o=.s) {keep}; }}")  # the last assembly, to compare
        lines.append('$(if $(NOALIGN),python3 -I -c "$$STRIP_ALIGN" $(@:.o=.s),true)')  # data packed at the byte
        if row.get("jump_table_base"):
            entry = r"^([ \t]*\.word[ \t]+\.L[0-9]+)[ \t]*$$"
            lines.append(f"sed -i -E 's/{entry}/\\1-0x{row['jump_table_base']:X}/' $(@:.o=.s)")
        template, _ = adapters.assemble_template(row)
        assemble = [_shell(adapters.render(template, {**values, "source": ["$(@:.o=.s)"], "out": ["$@"]}))]
        if row.get("rdata_alloc"):
            assemble.append("$(OBJCOPY) --set-section-flags .rdata=alloc,load,readonly,data,contents $@")
        # Only the .i moved (a header edit rewrites many) and the compiler's assembly is the last one's, byte for byte,
        # with the same text size: the object stands, and `.same` tells the rule to skip the trim it already holds.
        same = (
            '[ -z "$(filter-out %.i.ok,$?)" ] && [ -f $@ ] && [ -f ' + keep + " ] && "
            '[ "x$(file <$@.textsize)" = "x$(TEXTSIZE)" ] && cmp -s $(@:.o=.s) ' + keep
        )
        lines.append(
            f"if {same}; then rm -f {keep} && touch $@ && : > $@.same; else rm -f {keep} && {' && '.join(assemble)}; fi"
        )
    body = " && ".join(lines)
    return f"define PREPROCESS_{key}\n$(Q){preprocess}\nendef\nCOMPILE_{key} = {body}"


def _file_blocks(config: Config) -> list[str]:
    """Target-specific variables per [file] row: a dir/ key is a pattern over its objects, a path is one object."""
    project, rows = config.project, []
    for path in sorted(project.files, key=lambda p: (not p.endswith("/"), len(p), p)):
        facts = file_facts(project, path)
        id = facts.toolchain or project.toolchain
        row = configuration.toolchain(id)
        target = "$(foreach v,$(VERSIONS),build/$(v)/" + (f"{path}%.o)" if path.endswith("/") else f"{path}.o)")
        cpp, cc = _preprocessor_split(row, facts.cflags)
        rows += [
            f"{target}: TC := {_key(id)}",
            f"{target}: CPPFLAGS := {' '.join([*row.get('cppflags', []), *project.cppflags, *cpp])}",
            f"{target}: CFLAGS := {' '.join(_flags(row, config, cc, facts.omit))}",
        ]
    return rows


def _preprocessor_split(row: Json, flags: Sequence[str]) -> tuple[list[str], list[str]]:
    """(the preprocessor's flags, the compiler's) of a [file] row: `-D`, `-U`, `-I`... reach cpp, never the compiler,
    which may refuse them; a paired option written as two words keeps its value."""
    options, paired = tuple(row.get("preprocessor_options", ())), set(row.get("paired", ()))
    cpp: list[str] = []
    cc: list[str] = []
    words = iter(flags)
    for word in words:
        into = cpp if word.startswith(options) else cc
        into.append(word)
        if word in paired:
            into.append(next(words, ""))
    return cpp, cc


def makefile(config: Config) -> bytes:
    """The one Makefile of a repository: the same body for every game, whatever the number of files."""
    with effort.stage("build.makefile"):
        project, rows = config.project, configuration.toolchains()
        default = rows[project.toolchain]
        used = sorted({project.toolchain, *(f.toolchain for f in project.files.values() if f.toolchain)})
        pinned = _fetched(used)
        tools = [
            f"ifneq ($(filter default undefined,$(origin {var})),)\n{var} := "
            + (f"tools/{row['host']}/{row['binary']}" if var in pinned else str(row["default"]))
            + "\nendif"
            for var, row in sorted(_REPO["make"]["tools"].items())
        ]  # make's own CPP and AS beat a ?= line
        chains, stamps = [], []
        for var in sorted(pinned):  # a pinned tool the project's toolchains run is fetched like a compiler
            row = _REPO["make"]["tools"][var]
            chains.append(_fetch_rule(f"tools/{row['host']}", row["download"], row["pins"]))
            stamps.append(f"tools/{row['host']}/.stamp")
        for id in used:
            key, row = _key(id), rows[id]
            n = len(row["downloads"])
            if n > 1:
                raise Refusal(
                    Finding("toolchain.download", f"{id} has {n} downloads; the generated build supports one", path=id)
                )
            tools.append(
                f"CC_{key} ?= tools/{id}/{row['cc']}" if n else f"CC_{key} ?= $(error CC_{key} must be set in local.mk)"
            )
            tools.append(f"AS_{key} ?= tools/{id}/{row['as']}" if "as" in row else f"AS_{key} ?= $(AS)")
            tools.append(f"TOOLCHAIN_{key} ?= tools/{id}/.stamp" if n else f"TOOLCHAIN_{key} ?=")
            stamps.append(f"$(TOOLCHAIN_{key})")
            if n:
                chains.append(_fetch_rule(f"tools/{id}", row["downloads"][0], row["pins"]))
        platform = configuration.load_resource("rom.toml").get("asflags", [])
        version_vars = []
        for v in project.versions:
            version_vars += [
                f"VMACRO_{v} := {macro(v)}",
                f"SHA1_{v} := {project.version[v].sha1}",
                f"ROM0_{v} := {project.version[v].segments[0].rom}",
            ]
        text = Template(configuration.template("Makefile.in")).substitute(
            title=project.title,
            name=project.name,
            versions=" ".join(project.versions),
            tool_defaults="\n".join(tools),
            version_vars="\n".join(version_vars),
            toolchain=_key(project.toolchain),
            cppflags=" ".join([*default.get("cppflags", []), *project.cppflags]),
            cflags=" ".join(_flags(default, config)),
            asflags=" ".join([*platform, *project.asflags]),
            compile_recipes="\n".join(_recipe(id) for id in used),
            file_blocks="\n".join(_file_blocks(config)),
            stamps=" ".join(stamps),
            toolchain_rules="\n".join(chains),
            label="[A-Za-z_.]*" + re.escape(_REPO["splat"]["options"]["symbol_name_format"].replace("$VRAM", "")),
        )
        return text.encode()


# ---------------------------------------------------------------- the link script (C5)
def _guard(held: Sequence[str], every: Sequence[str]) -> str | None:
    """The `#if` condition for these versions, or None when they are every version the block can appear in."""
    if tuple(held) == tuple(every):
        return None
    return " || ".join(f"defined({macro(v)})" for v in held)


def _blocks(orders: Mapping[str, Sequence[str]]) -> list[tuple[tuple[str, ...], list[str]]]:
    """Module runs with the versions that hold them: one merged order when every version agrees on the relative
    order of the modules it shares, else one run per version."""
    names: list[str] = list(orders)
    merged: list[str] = []
    for v in names:
        at = -1
        for module in orders[v]:
            if module in merged:
                at = merged.index(module)
            else:
                at += 1
                merged.insert(at, module)
    if any([m for m in merged if m in set(orders[v])] != list(orders[v]) for v in names):
        return [((v,), list(orders[v])) for v in names]
    blocks: list[tuple[tuple[str, ...], list[str]]] = []
    for module in merged:
        held = tuple(v for v in names if module in orders[v])
        if blocks and blocks[-1][0] == held:
            blocks[-1][1].append(module)
        else:
            blocks.append((held, [module]))
    return blocks


def _pass(blocks: Sequence[tuple[tuple[str, ...], list[str]]], names: Sequence[str]) -> list[str]:
    """The input-section statements of every block, each run under the `#if` of the versions that hold it."""
    out: list[str] = []
    for held, items in blocks:
        guard = _guard(held, names)
        out += [f"#if {guard}"] if guard else []
        out += [f"        {item};" for item in items]
        out += ["#endif"] if guard else []
    return out


def _spelled(section: str) -> str:
    """The input sections that hold a section's bytes: read-only data also under the `.rdata` spelling the MIPS
    assemblers give it, a stand-in's own section (`.rodata.<name>`) exactly."""
    if section == ".rodata" or section.startswith(".rodata."):
        return f"{section} .rdata{section[len('.rodata') :]}"
    return section


def _piece(config: Config, v: str, piece: str) -> tuple[int, int, int]:
    """(rom start, rom end, vram or 0) of one version's piece, by its name in `versions.pieces`."""
    return next((start, end, vram) for name, start, end, vram in versions.pieces(config, v) if name == piece)


def _rows(config: Config, held: Tree, v: str, piece: str) -> list[tuple[int, str, str]]:
    """(rom, path, section) of every name a file owns in one version's piece (its ROM range: pieces may share a segment
    name), in ROM order."""
    rows: list[tuple[int, str, str]] = []
    table = held.symbols.get(v, {})
    start, end, _ = _piece(config, v, piece)
    for module in held.modules:
        for name in module.owns.get(v, ()):
            fact = table.get(name)
            if fact is None or fact[1] is None or not start <= fact[1] < end:
                continue
            rows.append((fact[1], module.path, module.sections.get(v, {}).get(name, ".text")))
    return sorted(rows)


def text_sizes(config: Config, held: Tree, v: str) -> bytes:
    """versions/<v>/text_sizes.mk: the size each object's `.text` has in the ROM, from its first owned address to the
    next owned bytes of anything else. A compiler or assembler may pad a section past its last function (IDO and
    gas round `.text` up to 16 bytes); the build cuts a zero tail back to this size before the object links. Only an
    object whose code lies in one run and is followed by owned bytes of its segment gets a row."""
    with effort.stage("build.text_sizes"):
        cuts = versions.segments(config, v)
        sizes: dict[str, int | None] = {}
        placed = zip(config.project.version[v].segments, versions.pieces(config, v), strict=True)
        for i, (segment, (piece, *_)) in enumerate(placed):
            if segment.kind == "bin":
                continue
            end = cuts[i + 1][1] if i + 1 < len(cuts) else None  # the ROM end needs the ROM: no row for the last
            runs: list[tuple[int, str, str]] = []
            for rom, path, section in _rows(config, held, v, piece):
                if not runs or runs[-1][1:] != (path, section):
                    runs.append((rom, path, section))
            for k, (rom, path, section) in enumerate(runs):
                if section != ".text":
                    continue
                after = runs[k + 1][0] if k + 1 < len(runs) else end
                sizes[path] = None if path in sizes or after is None else after - rom
        rows = [f"build/{v}/{path}.o: TEXTSIZE := 0x{size:X}" for path, size in sorted(sizes.items()) if size]
        rows += [f"build/{v}/{path}.o: NOALIGN := 1" for path in _unaligned(config, held, v)]
        return "\n".join([f"# {_GENERATED}", *rows, ""]).encode()


def _unaligned(config: Config, held: Tree, v: str) -> list[str]:
    """The files whose data starts a run of their own off a word boundary in the ROM: the ROM packs it at the byte, so
    the build drops the alignment the compiler asks for in that object's data sections (NOALIGN)."""
    found: set[str] = set()
    for segment, (piece, *_) in zip(config.project.version[v].segments, versions.pieces(config, v), strict=True):
        if segment.kind == "bin":
            continue
        last = None
        for rom, path, section in _rows(config, held, v, piece):
            if (path, section) != last and section != ".text":
                vram = rom - segment.rom + (segment.vram or segment.rom)
                found |= {path} if vram % 4 else set()
            last = (path, section)
    return sorted(found)


def _doubled(config: Config, held: Tree, v: str, piece: str) -> set[str]:
    """The files whose compiled read-only data (a `literals_` row) starts on an 8-byte boundary: the ROM keeps the
    alignment the compiler gave the section, which `SUBALIGN(1)` drops, so the link script asks for it where the
    file's data starts (a no-op where the data before it already ends there)."""
    table = held.symbols.get(v, {})
    start, end, _ = _piece(config, v, piece)
    first: dict[str, tuple[int, str]] = {}
    for module in held.modules:
        for name in module.owns.get(v, ()):
            fact = table.get(name)
            if fact is None or fact[1] is None or not start <= fact[1] < end:
                continue
            if module.sections.get(v, {}).get(name, ".text") == ".rodata":
                first[module.path] = min(first.get(module.path, (fact[1], name)), (fact[1], name))
    return {p for p, (_, name) in first.items() if name.startswith(tree.LITERALS) and table[name][0] % 8 == 0}


def _apart(held: Tree, v: str, rows: Sequence[tuple[int, str, str]], config: Config) -> None:
    """A file's object has one read-only section, listed where its first byte is: literals of a function that lie
    later, after other files' data, would be left out of the ROM. Refuse, naming the function."""
    table = held.symbols.get(v, {})
    literal = {
        (table[n][1], m.path): n
        for m in held.modules
        for n in m.owns.get(v, ())
        if n.startswith(tree.LITERALS) and n in table
    }
    seen: set[tuple[str, str]] = set()
    last: tuple[str, str] | None = None
    bad: list[Finding] = []
    for rom, path, section in rows:
        key = (path, section)
        name = literal.get((rom, path))
        if key in seen and key != last and name is not None:
            bad.append(
                Finding(
                    "sync.split",
                    f"{name} in {v} lies after other files' data, apart from the rest of {path}",
                    path=path,
                    unit=name,
                    versions=(v,),
                    action=f"move {name[len(tree.LITERALS) :]} into a file of its own",
                )
            )
        seen.add(key)
        last = key
    if bad:
        raise Refusal(*bad)


def _items(config: Config, held: Tree, v: str, piece: str, per_object: bool) -> list[str]:
    """The input sections of one version's segment in the order their bytes lie in the ROM: for each name a file owns
    in the segment, its section of that file's object (`.text` for code, the source's own section for data), the
    first of each in address order. A file's compiled read-only data sits where its `literals_` fact says; else a
    toolchain that keeps a function's literals with its object (`per-object`) has them follow the code unless the file
    names rodata of its own in the segment."""
    rows = _rows(config, held, v, piece)
    named = {(path, section) for _, path, section in rows}
    doubled = _doubled(config, held, v, piece)
    items: list[tuple[str, str]] = []
    for _, path, section in rows:
        if (path, section) not in items:
            items.append((path, section))
            if per_object and section == ".text" and (path, ".rodata") not in named:
                items.append((path, ".rodata"))
    _apart(held, v, rows, config)
    out: list[str] = []
    for path, section in items:
        if section == ".rodata" and path in doubled:
            out.append(f". = ALIGN(8) /* {path}.o */")
        out.append(f"{path}.o({_spelled(section)})")
    return out


def _header(name: str, by: Mapping[str, Segment], names: Sequence[str]) -> list[str]:
    """`.seg vram : AT(rom)` and the brace: one line, or an `#if` ladder when versions place the segment apart."""
    groups: dict[str, list[str]] = {}
    for v in names:
        s = by[v]
        at = f"0x{s.rom if s.vram is None else s.vram:08X} : AT(0x{s.rom:X}) SUBALIGN(1)"  # the files hold every byte
        groups.setdefault(f"    .{name} {at}", []).append(v)
    if len(groups) == 1:
        return [next(iter(groups)), "    {"]
    out: list[str] = []
    for i, (text, held) in enumerate(groups.items()):
        out += [f"#{'if' if i == 0 else 'elif'} " + " || ".join(f"defined({macro(v)})" for v in held), text]
    return [*out, "#endif", "    {"]


def linker(config: Config, held: Tree) -> bytes:
    """The single link script: one output section per segment whose input sections are in the order their bytes lie in
    the ROM, so a file's rodata may follow its code (one segment) or sit in a data segment of its own, and `#if` per
    version where the versions differ."""
    with effort.stage("build.linker"):
        project = config.project
        per_object = configuration.toolchain(project.toolchain).get("link_sections", "per-section") == "per-object"
        found: dict[str, dict[str, Segment]] = {}
        for v in project.versions:
            for segment, (piece, *_) in zip(project.version[v].segments, versions.pieces(config, v), strict=True):
                found.setdefault(piece, {})[v] = segment
        first = {name: min(s.rom for s in by.values()) for name, by in found.items()}
        out: list[str] = []
        for name in sorted(found, key=lambda n: (first[n], n)):
            by = found[name]
            names = [v for v in project.versions if v in by]
            whole = _guard(names, project.versions)
            binary = all(s.kind == "bin" for s in by.values())
            orders = {v: [] if binary else _items(config, held, v, name, per_object) for v in names}
            blocks = (
                [(tuple(names), [f"assets/{name}.bin.o(.data)"])] if binary else _blocks({v: orders[v] for v in names})
            )
            out += [f"#if {whole}"] if whole else []
            out += _header(name, by, names)
            out += _pass(blocks, names)
            out.append("    }")
            if not binary:
                files = {v: list(dict.fromkeys(i.rpartition(".o(")[0] for i in orders[v] if ".o(" in i)) for v in names}
                bss = _blocks({v: [f"{f}.o(.bss .sbss COMMON)" for f in files[v]] for v in names})
                out += [f"    .{name}.bss (NOLOAD) :", "    {", *_pass(bss, names), "    }"]
            out += ["#endif"] if whole else []
        text = Template(configuration.template("link.ld.in")).substitute(generated=_GENERATED, segments="\n".join(out))
        return text.encode()


# ---------------------------------------------------------------- the splat yaml (C4)
def _scalar(value: object) -> str:
    return json.dumps(value) if isinstance(value, bool | int | float) else json.dumps(str(value))


def splat_yaml(config: Config, v: str, held: Tree) -> bytes:
    """versions/<v>/splat.yaml: the segments of config.toml and the fixed splat options. One `c` subsegment per code
    segment; with `data_split = "subsegment"` also one subsegment `<segment>/<name>` per data symbol of
    symbol_addrs.txt a file owns (a row whose type is not code, an untyped row included: splat accepts no `type:data`;
    a data name no file owns is a label inside the piece before it) and per data name a file owns, in C or behind a
    stand-in, wherever its fact lives (aliases.txt, a row with no rom) and on any byte (the build fills each piece from
    the ROM): a stand-in needs its file, and the piece before C data must end where it starts, `rodata`
    when the tree puts the name in .rodata (a `literals_<function>` fact is one), and the code
    that follows data or a function of declared size starts a `c` subsegment again
    (splat reads no function past a size it was given), as does each function a file owns, in C or behind a stand-in
    (else splat may read it as an entry inside the function before it: the stand-in has no file, the C one a twin)."""
    with effort.stage("build.splat_yaml") as span:
        project = config.project
        row = configuration.toolchain(project.toolchain)
        cuts = versions.segments(config, v)
        placeholders = {"version": v, "name": project.name, "rom": f"roms/baserom.{v}.z64"}
        out = [f"# {_GENERATED}", "options:"]
        for key, value in _REPO["splat"]["options"].items():
            out.append(f"  {key}: {_scalar(value.format(**placeholders) if isinstance(value, str) else value)}")
        out += [f"  compiler: {row['splat']}", "segments:"]
        split = row.get("data_split", "symbol") == "subsegment"
        sections = {n: s for m in held.modules for n, s in m.sections.get(v, {}).items()}
        table = held.symbols.get(v, {})
        stand = {n for m in held.modules for n in m.owns.get(v, ()) if sections.get(n, ".text") != ".text"}
        entries = sorted(  # each function a file owns is its own: splat ends the one before at a `c` start
            fact[1]
            for m in held.modules
            for n in m.owns.get(v, ())
            if sections.get(n, ".text") == ".text" and (fact := table.get(n)) is not None and fact[1] is not None
        )
        every = versions.symbol_addrs(config, v)
        rows = [r for r in every if r.rom is not None]
        for r in every:  # a function fact with no rom is placed by the segment that maps its address
            at = _placed(r, versions.pieces(config, v)) if r.rom is None and r.kind in _CODE else None
            if at is not None:
                rows.append(versions.Row(r.name, r.vram, r.kind, at, r.size, r.line))
                finding = Finding(
                    "symbol.unplaced",
                    f"{v}: {r.name} at 0x{r.vram:08X} names no rom; its segment maps it to 0x{at:X} (add rom:0x{at:X})",
                    path=symbols.path(v),
                    unit=r.name,
                    versions=(v,),
                    blocking=False,
                )
                span.add(findings=[finding])
                effort.say(finding.reason)
        listed = {r.name for r in rows}
        rows += [  # owned data starts its own piece, though aliases.txt alone or no rom of its row places it
            versions.Row(n, fact[0], fact[3], fact[1], fact[2], 0)
            for n in sorted(stand - listed)
            if (fact := table.get(n)) is not None and fact[1] is not None
        ]
        rows.sort(key=lambda r: (r.rom or 0, r.name in stand))  # at one address the owned name wins
        free = set(held.unowned(v))
        labels = {n for n in held.symbols.get(v, {}) if n not in free and held.owned(v, n) is None}
        for segment, (piece, start, end, _) in zip(
            project.version[v].segments, versions.pieces(config, v), strict=True
        ):
            out += [
                f"  - name: {piece}",
                f"    type: {'bin' if segment.kind == 'bin' else 'code'}",
                f"    start: 0x{start:X}",
            ]
            if segment.kind == "bin":
                continue
            out += [f"    vram: 0x{segment.vram:08X}"] if segment.vram is not None else []
            cells: list[tuple[int, str, str]] = [(start, _C, segment.name)]
            for r in rows:
                if not start <= (r.rom or 0) < end or r.kind in _NOT_DATA:
                    continue
                if symbols.inside_data(r.kind, r.vram) and r.name not in stand:
                    finding = Finding(
                        "symbol.folded",
                        f"{v}: {r.name} at 0x{r.vram:08X} is folded into the preceding data chunk "
                        "because it is not 4-byte aligned",
                        path=symbols.path(v),
                        unit=r.name,
                        versions=(v,),
                        blocking=False,
                    )
                    span.add(findings=[finding])
                    effort.say(finding.reason)
                    continue
                if not split:
                    continue
                if r.kind in _CODE:
                    if cells[-1][1] != _C:
                        cells.append((r.rom or 0, _C, segment.name))
                    if r.size and (r.rom or 0) + r.size < end:  # a sized function ends where it says
                        cells.append(((r.rom or 0) + r.size, _C, segment.name))
                    continue
                if r.name in labels:
                    continue  # a label inside the data before it: splat writes it into that piece
                kind = "rodata" if sections.get(r.name, "").startswith(".rodata") else "data"
                cells.append((r.rom or 0, kind, f"{segment.name}/{r.name}"))
            cells += [(rom, _C, segment.name) for rom in entries if start < rom < end]
            code_at = {r.rom: r.name for r in rows if r.kind in _CODE and r.rom is not None}
            for at, kind, name in cells:
                if kind != _C and at in code_at and not name.endswith("/" + code_at[at]):
                    finding = Finding(
                        "symbol.shadowed",
                        f"{v}: {code_at[at]} at rom 0x{at:X} is shadowed by the data piece {name}: splat writes its "
                        "assembly from the next instruction, not from its first",
                        path=symbols.path(v),
                        unit=code_at[at],
                        versions=(v,),
                        blocking=False,
                    )
                    span.add(findings=[finding])
                    effort.say(finding.reason)
            cells.sort(key=lambda c: (c[0], c[1] != _C))  # at one address, data outranks a restart of code
            kept = [c for c, after in zip(cells, [*cells[1:], None], strict=True) if after is None or after[0] > c[0]]
            out += ["    subsegments:", *(f"      - [0x{at:X}, {kind}, {name}]" for at, kind, name in kept)]
        out.append(f"  - [0x{cuts[-1][2]:X}]")
        return ("\n".join(out) + "\n").encode()


def _placed(row: versions.Row, pieces: Sequence[tuple[str, int, int, int]]) -> int | None:
    """The rom the segment that maps `row.vram` gives it, None when no segment with a vram does."""
    for _, start, end, vram in pieces:
        if vram and vram <= row.vram < vram + end - start:
            return start + row.vram - vram
    return None


def _fetched(toolchains: Sequence[str]) -> set[str]:
    """The make variables of the pinned tools (a download in repo.toml) that a command of these toolchains runs: the
    Makefile fetches each into tools/<host>/ like a compiler, and no machine path (local.mk) names it."""
    rows = configuration.toolchains()
    return {
        var
        for var, row in _REPO["make"]["tools"].items()
        if "download" in row and any(f"{{{row['host']}}}" in str(rows.get(id)) for id in toolchains)
    }


def _fetch_rule(folder: str, download: Json, pins: Mapping[str, str]) -> str:
    """The rule that fetches one pinned archive into `folder`, checks it and every pinned file, and stamps it."""
    strip = f"--strip-components={download['strip']} " if download.get("strip") else ""
    return Template(configuration.template("toolchain.mk.in")).substitute(
        stamp=f"{folder}/.stamp",
        dir=folder,
        url=download["url"],
        archive=f"{folder}.tar.gz",
        sha256=download["sha256"],
        files=strip + " ".join(download["files"]),
        stamp_name=".stamp",
        pins=" ".join(f"'{sha}  {file}'" for file, sha in sorted(pins.items())),
    )


def _cached_cc(config: Config, compiler: Path) -> str:
    """The compiler the Makefile runs: behind resources/cc_cache.py, whose results live beside the host content cache,
    so every tree on the host copies the assembly of a text any tree compiled (a commit that changes a header all
    sources include changes every .i, yet its proof tree compiled those texts: 1045 s CPU in a seeded tip setup)."""
    cache = getattr(config.host, "cache_dir", None)
    if cache is None:
        return str(compiler)
    # imported, not run as a script: a script is compiled again on every call, a module keeps its .pyc
    start = "import sys;sys.path.insert(0,sys.argv.pop(1));import cc_cache;sys.exit(cc_cache.main(sys.argv))"
    folder = Path(__file__).parent / "resources"
    parts = ("python3", "-I", "-S", "-c", start, folder, cache.with_name(f"{cache.name}.cc"), compiler)
    return " ".join(shlex.quote(str(p)) for p in parts)


def local_mk(config: Config) -> bytes:
    with effort.stage("build.local_mk"):
        assignments = [
            f"JOBS := {config.host.workers}",
            *(
                f"{var} := {config.host.tools.get(row['host'], '')}"
                for var, row in sorted(_REPO["make"]["tools"].items())
                if "download" not in row  # a pinned tool is the Makefile's own fetch, never a machine path
            ),
        ]
        for id, row in sorted(configuration.toolchains().items()):
            key, base = _key(id), config.host.toolchain_root / id
            assignments.append(f"CC_{key} := {_cached_cc(config, base / row['cc'])}")
            assignments.append(f"AS_{key} := {base / row['as']}" if "as" in row else f"AS_{key} := $(AS)")
            assignments.append(f"TOOLCHAIN_{key} :=")
        return Template(configuration.template("local.mk.in")).substitute(assignments="\n".join(assignments)).encode()


# ---------------------------------------------------------------- running make
def make(config: Config, targets: Sequence[str]) -> NativeResult:
    """A setup make run (extract, check): local.mk first, then one make. The Makefile reads the machine's tool paths
    (splat among them) from local.mk, and a clone has none until this writes it, so no target may run before it is
    current; a land builds only in a set-up clone."""
    store.write(config.project.root / "local.mk", local_mk(config))
    return process.make(config, targets, config.host.slots or config.host.workers)


def _make(config: Config, target: str) -> NativeResult:
    return make(config, [target])


def extract(config: Config, version: str | None = None) -> NativeResult:
    """`make extract`: splat splits one version's baserom (default: all) into asm/<v>/."""
    with effort.stage("build.extract"):
        # splat writes in place; a tree set up from a shared cache holds hard links to it, so no link may survive
        root = config.project.root / "asm"
        store.unshare(root / version if version else root)
        result = _make(config, f"extract-{version}" if version else "extract")
        if result.exit != 0:
            tail = result.stderr.decode(errors="replace").splitlines()
            raise Refusal(Finding("setup.splat", reason="\n".join(tail[-20:]) or f"exit {result.exit}"))
        return result


def inputs(config: Config) -> str:
    """What `make check` reads or builds, by name, size and mtime: while it is unchanged the last check still holds."""
    return project_inputs(config.project)


def project_inputs(project: Project) -> str:
    """Build input/output identity without a host config, shared by make's verdict and the report reader."""
    root = str(project.root)
    roms = (f"build/{v}/{project.name}.z64" for v in project.versions)
    paths = [f"{root}/{p}" for p in ("config.toml", "Makefile", "local.mk", f"{project.name}.ld", *roms)]
    for place in ("src", "include", "asm", "versions", "tools"):
        paths += [f"{top}/{name}" for top, _, names in os.walk(f"{root}/{place}") for name in names]
    rows = []
    for path in sorted(paths):
        try:
            stat = os.stat(path)
            rows.append((path, stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            rows.append((path, 0, 0))
    return sha256(repr(rows).encode())


_FILE_RULE = re.compile(r"^\$\(foreach v,\$\(VERSIONS\),build/\$\(v\)/(.+?\.o)\):")
_INCLUDED = re.compile(rb'^\s*\.include\s+"([^"]+)"', re.M)


def _objects(config: Config, version: str) -> list[Path]:
    """The compiled objects of one version: build/<v>/<source>.o, not the assembled (.s.o) or binary (.bin.o) ones,
    whose rule names their one input file."""
    found = []
    for directory, _, names in os.walk(config.project.root / "build" / version):
        found += [Path(directory) / n for n in names if n.endswith(".o") and not n.endswith((".s.o", ".bin.o"))]
    return found


def _stamps(config: Config, version: str) -> list[Path]:
    """The .i stamps of one version: build/<v>/<source>.i.ok, one per source the .i rule preprocessed."""
    found = []
    for directory, _, names in os.walk(config.project.root / "build" / version):
        found += [Path(directory) / n for n in names if n.endswith(".i.ok")]
    return found


def drop_including(config: Config, version: str, names: set[str]) -> int:
    """Remove the objects of `version` whose compiler output includes one of `names` (paths under asm/<v>/, as
    INCLUDE_ASM spells them) or has no compiler output to read, so make rebuilds exactly those. Returns the count."""
    dropped = 0
    wanted = {n.encode() for n in names}
    for obj in _objects(config, version):
        try:
            included = set(_INCLUDED.findall(obj.with_suffix(".s").read_bytes()))
        except FileNotFoundError:
            included = wanted
        if included & wanted:
            obj.unlink(missing_ok=True)
            dropped += 1
    return dropped


_PREPROCESS_LINE = re.compile(r"^(?:\S.*?: )?(?:CPPFLAGS|DEFINES|INCLUDES) :?= ")


def _preprocess_lines(text: str) -> set[str]:
    """The lines of a Makefile that decide only the preprocessed text: the preprocessor's flags and the bodies of the
    PREPROCESS_ recipes. The .i rule compares the text it makes, so such a line changing needs the .i made again,
    never the object removed."""
    found: set[str] = set()
    inside = False
    for line in text.splitlines():
        if line.startswith("define PREPROCESS_"):
            inside = True
        elif inside and line == "endef":
            inside = False
        elif inside or _PREPROCESS_LINE.match(line):
            found.add(line)
    return found


_ASSIGN = re.compile(r"^(?:(?:export|override)\s+)*([A-Za-z_][\w.-]*)\s*(?::{1,3}=|\?=|\+=|!=|=)")
_EXPORT = re.compile(r"^export\s+([A-Za-z_][\w.-]*)\s*$")
_WORD = re.compile(r"[A-Za-z_][\w]*")
_COMPUTED = re.compile(r"([A-Za-z_][\w]*)\$[({]")  # VMACRO_$(w): a name made up while make expands
_DIRECTIVE = ("ifeq", "ifneq", "ifdef", "ifndef", "else", "endif", "include", "-include", "sinclude", "vpath", "$(")


class _Block:
    """One part of a Makefile: a variable's definition ("var", its name), a rule with its recipe ("rule", its
    targets), a per-file row ("row", its object) or anything else ("other": a conditional, an eval, an include)."""

    def __init__(self, kind: str, name: str, lines: list[str]) -> None:
        self.kind, self.name, self.lines = kind, name, lines
        text = "\n".join(lines)
        self.words = set(_WORD.findall(text))
        self.prefixes = set(_COMPUTED.findall(text))

    def uses(self, name: str) -> bool:
        return name in self.words or any(name.startswith(p) for p in self.prefixes)


def _make_blocks(text: str) -> list[_Block]:
    """The Makefile's parts, each with its physical lines (a continued line keeps every piece)."""
    lines, found, rule, i = text.splitlines(), [], None, 0
    while i < len(lines):
        group = [lines[i]]
        while group[-1].endswith("\\") and i + 1 < len(lines):
            i += 1
            group.append(lines[i])
        i += 1
        first = group[0]
        if first.startswith("define "):
            while group[-1] != "endef" and i < len(lines):
                group.append(lines[i])
                i += 1
            found.append(_Block("var", first.split()[1], group))
            rule = None
        elif rule is not None and first.startswith("\t"):
            rule.lines.extend(group)
            rule.words.update(_WORD.findall("\n".join(group)))
            rule.prefixes.update(_COMPUTED.findall("\n".join(group)))
        elif not first.strip() or first.lstrip().startswith("#"):
            continue  # neither ends a recipe
        elif row := _FILE_RULE.match(first):
            block = _Block("row", row.group(1), group)
            value = "\n".join(group)[row.end() :].partition("=")[2]  # the row sets its variable; only the value uses
            block.words, block.prefixes = set(_WORD.findall(value)), set(_COMPUTED.findall(value))
            found.append(block)
            rule = None
        elif (assign := _ASSIGN.match(first)) or (assign := _EXPORT.match(first)):
            found.append(_Block("var", assign.group(1), group))
            rule = None
        elif ":" in first and not first.startswith(_DIRECTIVE):
            rule = _Block("rule", " ".join(first.split(":", 1)[0].split()), group)
            found.append(rule)
        else:
            found.append(_Block("other", "", group))
            rule = None
    return found


def _closure(blocks: Sequence[_Block], start: Sequence[_Block]) -> list[_Block]:
    """`start` and every variable definition it uses, transitively."""
    seen, todo = list(start), list(start)
    while todo:
        block = todo.pop()
        for other in blocks:
            if other.kind == "var" and other not in seen and block.uses(other.name):
                seen.append(other)
                todo.append(other)
    return seen


def rule_text(text: str, target: str) -> str | None:
    """The rule making `target` with every variable definition it uses: what that recipe writes depends on nothing
    else in the Makefile. None when no rule names the target."""
    blocks = _make_blocks(text)
    rules = [b for b in blocks if b.kind == "rule" and target in b.name.split()]
    if not rules:
        return None
    return "\n".join(sorted({line for b in _closure(blocks, rules) for line in b.lines}))


def _reach(before: str, after: str, moved: set[str]) -> tuple[set[str], set[str]] | None:
    """What the changed lines reach: (the objects of the per-file rows, the targets of the rules) that use one of
    them, directly or through the variables they define. None when a line reaches what cannot be named: a
    conditional, an eval, a variable no rule here uses (make's own rules read CFLAGS), a target make computes."""
    blocks = _make_blocks(before) + _make_blocks(after)
    rows, targets, todo, done = set(), set(), [], set()

    def take(block: _Block) -> bool:
        if block.kind == "row":
            rows.add(block.name)
        elif block.kind == "rule":
            if "$" in block.name:
                return False
            targets.update(block.name.split())
        elif block.kind == "var":
            if block.name not in done:
                done.add(block.name)
                todo.append(block.name)
        else:
            return False
        return True

    for line in moved:
        owners = [b for b in blocks if line in b.lines]
        if not owners or not all(take(b) for b in owners):
            return None
    while todo:
        name = todo.pop()
        users = [b for b in blocks if b.uses(name) and not (b.kind == "var" and b.name == name)]
        if not users or not all(take(b) for b in users):
            return None
    return rows, targets


def _pattern(target: str) -> re.Pattern[str]:
    head, _, tail = target.partition("%")
    return re.compile(re.escape(head) + ("(.+)" + re.escape(tail) if "%" in target else ""))


def _maker(patterns: Mapping[str, re.Pattern[str]], rel: str) -> str | None:
    """The target of the rule make picks for `rel`: an exact one, else the pattern with the shortest stem."""
    best, stem = None, None
    for target, pattern in patterns.items():
        if m := pattern.fullmatch(rel):
            size = len(m.group(1)) if m.groups() else -1
            if stem is None or size < stem:
                best, stem = target, size
    return best


_LONG_AGO = 1_000_000_000  # ns: older than any input, so make remakes the file and nothing is deleted under a rule


def drop_reflagged(config: Config, new: bytes | None = None) -> int:
    """make does not see a rule change, so the Makefile the objects in build/ were made by is kept
    (.unbake/make.Makefile) and compared with the one about to build (`new`, default the tree's Makefile). A changed
    line that decides only the preprocessed text (`_preprocess_lines`) removes the .i stamp (`.i.ok`) of the objects
    it reaches, so their text is made again and only an object whose text changed compiles again; any other changed
    line reaches what the rules using it make (`_reach`): a per-file row its object in every version, a recipe or a
    variable only the files of the rules that use it, the rule make picks for each file deciding. A reached compiled
    object goes with its stamp; any other reached file in build/ is dated before every input, so make makes it again
    (a rule that only checks a file splat wrote still finds it). A line whose reach cannot be named removes every
    compiled object. Comment and blank lines are ignored. The kept copy becomes the new one. Returns the count of
    objects removed; nothing is removed when no copy was kept (objects made before this rule)."""
    root = config.project.root
    kept, current = root / ".unbake/make.Makefile", root / "Makefile"
    try:
        new = current.read_bytes() if new is None else new
    except FileNotFoundError:
        return 0
    try:
        old = kept.read_bytes()
    except FileNotFoundError:
        old = new
    store.write(kept, new)
    before, after = old.decode(), new.decode()
    changed = set(before.splitlines()) ^ set(after.splitlines())
    moved = {line for line in changed if line.strip() and not line.startswith("#")}
    if not moved:
        return 0
    text_only = _preprocess_lines(before) | _preprocess_lines(after)

    def reached(lines: set[str]) -> set[str] | None:
        rules = {_FILE_RULE.match(line) for line in lines}
        return None if None in rules else {m.group(1) for m in rules if m}

    objects = {line for line in moved if line not in text_only}
    reach = _reach(before, after, objects) if objects else (set(), set())
    remade, targets = (None, set()) if reach is None else reach
    stamped = reached(moved - objects) if moved - objects else set()
    patterns = {
        t: _pattern(t)
        for b in (_make_blocks(before) + _make_blocks(after) if targets else [])
        if b.kind == "rule" and "$" not in b.name
        for t in b.name.split()
    }
    dropped = 0
    for version in config.project.versions:
        base = root / "build" / version
        gone = set()
        for obj in _objects(config, version):
            name = str(obj.relative_to(base))
            if remade is None or name in remade or (targets and _maker(patterns, f"build/{version}/{name}") in targets):
                obj.unlink(missing_ok=True)
                gone.add(name)
                dropped += 1
        for stamp in _stamps(config, version):  # an object already gone may still have the stamp of an old text
            name = str(stamp.relative_to(base))[: -len(".i.ok")] + ".o"
            if targets and _maker(patterns, f"build/{version}/{name[:-2]}.i.ok") in targets:
                if (base / name).exists():
                    (base / name).unlink()
                    dropped += 1
                gone.add(name)
            if remade is None or stamped is None or name in gone or name in stamped:
                stamp.unlink(missing_ok=True)
        if targets:
            for directory, _, names in os.walk(base):
                for n in names:
                    path = Path(directory) / n
                    if n.endswith(".i.ok") or (n.endswith(".o") and not n.endswith((".s.o", ".bin.o"))):
                        continue
                    if _maker(patterns, str(path.relative_to(root))) in targets:
                        os.utime(path, ns=(_LONG_AGO, _LONG_AGO))
    return dropped


def make_check(config: Config, held: Tree | None = None) -> Json:
    """`make check`: one make run builds and links every version and verifies its sha1."""
    with effort.stage("build.make_check"):
        stamp, known = config.project.root / ".unbake/make.json", inputs(config)
        done = {"exit": 0, "versions": list(config.project.versions)}
        if stamp.exists() and json.loads(stamp.read_bytes()).get("inputs") == known:
            effort.count("make-check", True)
            report.record(
                config,
                held or tree.capture(config),
                exact=sorted(config.project.versions),
                failing=[],
                build_inputs=known,
                build_failed=None,
            )
            return done
        effort.count("make-check", False)
        drop_reflagged(config)
        result = _make(config, "check")
        if result.exit == 0:
            stamp.parent.mkdir(parents=True, exist_ok=True)
            store.write(stamp, json.dumps({"inputs": inputs(config)}).encode())
            report.record(
                config,
                held or tree.capture(config),
                exact=sorted(config.project.versions),
                failing=[],
                build_inputs=inputs(config),
                build_failed=None,
            )
            return done
        report.record(
            config,
            held or tree.capture(config),
            exact=[],
            build_inputs=inputs(config),
            build_failed=f"make exit {result.exit}",
        )
        lines = result.stderr.decode(errors="replace").splitlines()
        failed = [line for line in lines if "***" in line]  # make -k ran everything it could: every failed target
        raise Refusal(Finding("setup.make", "\n".join(failed or lines[-20:]), missing=(f"make exit {result.exit}",)))


def verify(params: Json) -> Json:
    """Record make's exit and independently hash every built ROM; no host configuration is needed."""
    project = configuration.load_project(Path(params["project"]))
    exact, findings = [], []
    status = params.get("status") or 0
    if status:
        findings.append(Finding("build.failed", f"make exited {status}; no version receives matched credit"))
    else:
        for version in project.versions:
            path = project.root / "build" / version / f"{project.name}.z64"
            try:
                data = path.read_bytes()
            except OSError as error:
                findings.append(Finding("build.rom", str(error), path=str(path), versions=(version,)))
                continue
            expected = project.version[version]
            if len(data) != expected.size or sha1(data).hexdigest() != expected.sha1:
                findings.append(
                    Finding("build.rom", "built ROM size or SHA-1 differs", path=str(path), versions=(version,))
                )
            else:
                exact.append(version)
    report.record_build(project.root, project_inputs(project), exact, f"make exit {status}" if status else None)
    if findings:
        raise Refusal(*findings)
    return {"exact": exact}
