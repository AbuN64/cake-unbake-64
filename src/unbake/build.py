"""Toolchain installation, splat extraction and portable build-file rendering."""

from __future__ import annotations

import json
import os
import pickle
import re
import shlex
import shutil
import tarfile
import time
import urllib.request
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib import metadata
from itertools import pairwise
from pathlib import Path
from string import Template
from typing import Any

import yaml

from unbake import adapters, delta, effort, native, pool, present, process, recipes, store, symbols, view
from unbake import config as configuration
from unbake.contracts import (
    Config,
    Finding,
    Json,
    NativeResult,
    Recipe,
    Refusal,
    Snapshot,
    UnitSpec,
    code_identity,
    digest,
    sha256,
    sha256_file,
)

_REPO = configuration.load_resource("repo.toml")
_WORD = 0xFFFFFFFF
_TAG = "@RULES@"  # stands for the digest of the rule it is in: a changed rule has outputs of its own
_MAKE = {row["host"]: f"$({var})" for var, row in _REPO["make"]["tools"].items()}


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
        manifest.write_text(json.dumps(fresh))
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


def _shell(argv: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(w)) for w in argv)


def _values(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe) -> dict[str, list[str]]:
    project = snapshot.config.project
    key = recipe.toolchain.replace("-", "_").replace(".", "_")
    macros = _REPO["splat"]["options"]["generated_asm_macros_directory"].format(version=version, name=project.name)
    return {
        **{k: [v] for k, v in _MAKE.items()},
        "cc": [f"$(CC_{key})"],
        "as": [f"$(AS_{key})"],
        "cppflags": list(recipe.cppflags),
        "codegen": list(recipe.cflags),
        "asflags": list(recipe.asflags),
        "defines": list(project.version_macros[version]),
        "includes": ["-Iinclude", f"-I{Path(unit.path).parent}", f"-I{macros}"],
        "name": [Path(unit.path).stem],
    }


def _reading(snapshot: Snapshot, unit: UnitSpec, version: str) -> list[str]:
    """The command line that reads the unit's source, whose search flags decide which headers it reaches."""
    recipe = recipes.resolve(snapshot.config, unit, {})
    row = configuration.toolchain(recipe.toolchain)
    phases = configuration.kind(unit.kind)["phases"]
    values = {**_values(snapshot, unit, version, recipe), "source": [unit.path], "out": ["-"]}
    if "preprocess" in phases:
        return adapters.render(row["preprocess"], values)
    if "assemble" in phases:
        return adapters.render(adapters.assemble_template(row if "compile" in phases else None)[0], values)
    return []  # armips: only file-relative includes, and an unfound one is covered by the caller


def _headers(snapshot: Snapshot, unit: UnitSpec, version: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return view.headers(snapshot, unit, _reading(snapshot, unit, version))


def _versions(snapshot: Snapshot, unit: UnitSpec) -> list[str]:
    return sorted(snapshot.layout.held(unit.members) - set(unit.withheld))


def _rule(snapshot: Snapshot, unit: UnitSpec, version: str) -> tuple[str, str, list[str], list[str], list[str]]:
    """(rule text, its output directory digest, outputs, dependencies, link commands) of one unit in one version."""
    cfg, project = snapshot.config, snapshot.config.project
    recipe = recipes.resolve(cfg, unit, {})
    row = configuration.toolchain(recipe.toolchain)
    phases = configuration.kind(unit.kind)["phases"]
    home = Path("build") / version / Path(unit.path).with_suffix("")
    work = home / _TAG
    obj = work / "source.o"
    current: str | Path = unit.path
    commands = [
        f"mkdir -p {shlex.quote(str(work))}",
        f"find {shlex.quote(str(home))} -mindepth 1 -maxdepth 1 ! -name {_TAG} -exec rm -rf {{}} +",
    ]
    key = recipe.toolchain.replace("-", "_").replace(".", "_")
    values = _values(snapshot, unit, version, recipe)
    for phase in phases:
        if phase == "preprocess":
            out = work / "source.i"
            argv = adapters.render(row[phase], {**values, "source": [str(current)], "out": [str(out)]})
            commands.append(_shell(argv) + f" > {out} 2> {work}/preprocess.err")
            if row.get("error_pattern"):
                commands.append(f"! grep -Eq {shlex.quote(row['error_pattern'])} {work}/preprocess.err")
        elif phase == "compile":
            out = work / ("source.s" if row["emits_asm"] else "source.o")
            argv = adapters.render(row[phase], {**values, "source": [str(current)], "out": [str(out)]})
            commands.append(_shell(argv) + f" 2> {work}/compile.err || {{ cat {work}/compile.err >&2; exit 1; }}")
        elif phase == "assemble":
            if "compile" in phases and not row["emits_asm"]:
                continue
            out = obj
            template, tool = adapters.assemble_template(row if "compile" in phases else None)
            commands.append(
                _shell(
                    adapters.render(
                        template, {**values, "as": values[tool], "source": [str(current)], "out": [str(out)]}
                    )
                )
            )
        elif phase == "armips":
            out, wrapper = work / "data.bin", work / "resource.s"
            commands.append(f"printf '%s\\n' '.create \"{out}\", 0' > {wrapper}")
            commands.append(f"sed -E '/^[[:space:]]*\\.(create|close)([[:space:]]|$)/d' {unit.path} >> {wrapper}")
            commands.append(f"printf '%s\\n' '.close' >> {wrapper}")
            commands.append(_shell([_MAKE["armips"], wrapper, "-root", "."]))
        else:
            continue
        current = out
    ranges = native.sections(snapshot, unit, version)
    outputs, patches = [], []
    if "armips" in phases:
        if len(ranges) != 1 or ranges[0].section == ".bss":
            raise Refusal(Finding("link.error", "resource must own one ROM section", unit=unit.path))
        outputs.append(str(current))
        patches.append(f"{current}:{ranges[0].rom_start}")
        commands.append(f'test "$$(wc -c < {current})" -eq {ranges[0].size}')
    else:
        tail = len(commands)
        symbols = Path("versions") / version / "symbols.ld"
        trim = not row["preserves_padding"] or snapshot.layout.members[unit.members[0]].state == "hasm"
        script = adapters.link_script(ranges, symbols, trim)
        commands.append("printf '%s\\n' " + _shell(script.splitlines()) + f" > {work}/unit.ld")
        commands.extend(_shell(argv) for argv in adapters.link_commands(_MAKE, obj, ranges, work, trim))
        for placement in ranges:
            suffix = ".size" if placement.section == ".bss" else ".bin"
            output = str(work / (placement.section[1:] + suffix))
            outputs.append(output)
            if placement.section == ".bss":
                commands.append(f"printf '%s' '{placement.size}' > {output}")
            else:
                commands.append(f'test "$$(wc -c < {output})" -eq {placement.size}')
                patches.append(f"{output}:{placement.rom_start}")
    found, missing = _headers(snapshot, unit, version)
    dependencies = [unit.path, f"versions/{version}/symbols.ld", *found]
    if missing:  # a quoted include found nowhere yet: wherever it appears, and any project header, may be it
        dependencies.append("$(wildcard " + " ".join(missing) + " include/*.h include/*/*.h)")
    if "compile" in phases:
        dependencies.append(f"$(TOOLCHAIN_{key})")
    dependencies.extend(_tool_stamps(" ".join(commands)))
    text = Template(configuration.template("unit.mk.in")).substitute(
        unit=unit.path,
        version=version,
        outputs=" ".join(outputs),
        dependencies=" ".join(dependencies),
        commands=" && ".join(commands),
        rom=f"build/{version}/{project.name}.z64",
        patches=" ".join(patches),
    )
    tag = digest(text)[:12]
    return (
        text.replace(_TAG, tag),
        tag,
        [o.replace(_TAG, tag) for o in outputs],
        [d for d in dependencies if "$" not in d],
        [c.replace(_TAG, tag) for c in (commands[1:2] + commands[tail:] if "armips" not in phases else [])],
    )


_RULE_FORMAT = 2  # bump when _rule writes its make rule differently: the key names inputs, not the tool code


def _rules_context(snapshot: Snapshot) -> str:
    """What every unit's rules read besides the unit: the project, the toolchain and unit tables, the rule template."""
    return effort.memo(
        ("unit-rules", snapshot.config.digest),
        lambda: digest(
            (
                snapshot.config.project,
                configuration.load_resource("toolchains.toml"),
                configuration.load_resource("units.toml"),
                configuration.template("unit.mk.in"),
                _RULE_FORMAT,
            )
        ),
    )


def _unit_key(item: tuple[Snapshot, UnitSpec, str]) -> str:
    """A unit's rules read its own rows, text, headers and recipe, never another unit's: one changed unit, one key."""
    snapshot, unit, version = item
    headers = _headers(snapshot, unit, version)
    return digest(
        (unit, version, [snapshot.layout.members[n] for n in unit.members], headers, _rules_context(snapshot))
    )


def _unit_stamp(item: tuple[Snapshot, UnitSpec, str]) -> str:
    """The inputs of a unit's rules other than files: the unit, the version and its members' rows."""
    snapshot, unit, version = item
    return delta.fingerprint((unit, version, [snapshot.layout.members[n] for n in unit.members]))


def _unit_ident(item: tuple[Snapshot, UnitSpec, str]) -> str:
    return f"{item[1].path}\0{item[2]}"


def _unit_job(item: tuple[Snapshot, UnitSpec, str]) -> tuple[str, dict[str, str]]:
    """A unit's rule text and every file its header walk read or probed, pinned."""
    snapshot, unit, version = item
    return _rule(*item)[0], view.asked(snapshot, unit, _reading(snapshot, unit, version))


_CHECK_FORMAT = 1  # bump when a check reports other or more findings for the same inputs


def check_key(item: tuple[Snapshot, UnitSpec, str]) -> str:
    """Every input of a check of one unit in one version, never the tool's code: the compile cache below it is keyed
    the same way, so a clean verdict survives an install and a header edit that changes no text moves nothing. Only
    clean results are stored under it: a failure runs again, so a fixed tool bug clears at once."""
    snapshot, unit, version = item
    return digest((native.stamp(snapshot, unit, version, code=False), _CHECK_FORMAT))


check_key.stable = True  # type: ignore[attr-defined]


def check_unit(item: tuple[Snapshot, UnitSpec, str]) -> tuple[Finding, ...]:
    """A pool job of check: compile_unit, but an object is reused only when the whole preprocessed text (every header of
    the include closure, reached or not) is the one it was built from, so an edited header recompiles its consumers."""
    with native.exactly():
        return compile_unit(item)


def compile_unit(item: tuple[Snapshot, UnitSpec, str]) -> tuple[Finding, ...]:
    """A pool job of check: the unit's findings and, when it builds, the outputs of its make rule left where make looks,
    made by the rule's own link commands from the object just built, so `make check` links nothing twice."""
    (snapshot, unit, version), root, begun = item, item[0].config.project.root, time.time_ns()  # outputs carry `begun`
    with store.work(snapshot.config) as work:  # so a peer's edit made during this compile still makes make rebuild
        found = native.compiles(item, work)  # leaves the object in `work`
        if found and all(f.key == "journal.changed" for f in found):
            return (
                Finding(
                    "check.moved-during-run",
                    configuration.sentence("check.moved-during-run"),
                    path=unit.path,
                    unit=unit.path,
                    versions=(version,),
                    blocking=False,
                ),
            )
        obj = work / f"{Path(unit.path).stem}.o"
        if found or version in unit.withheld or not obj.is_file():
            return found
        _, tag, outputs, dependencies, commands = pickle.loads(
            store.cached(snapshot.config, "unit-rule", _unit_key(item), lambda: pickle.dumps(_rule(*item)))
        )
        stale = max(((root / d).stat().st_mtime_ns for d in dependencies if (root / d).is_file()), default=0)
        if not commands or all((root / o).is_file() and (root / o).stat().st_mtime_ns >= stale for o in outputs):
            return found
        home = root / "build" / version / Path(unit.path).with_suffix("") / tag
        home.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(obj, home / "source.o")
        proved = work / "proved"
        if proved.is_dir() and {p.name for p in proved.iterdir()} == {Path(o).name for o in outputs}:
            for old in home.parent.iterdir():
                if old != home:
                    if old.is_dir() and not old.is_symlink():
                        shutil.rmtree(old)
                    else:
                        old.unlink()
            for o in outputs:
                shutil.copyfile(proved / Path(o).name, root / o)
                os.utime(root / o, ns=(begun, begun))
            return found
        script = " && ".join(c.replace("$$", "$") for c in commands)
        for host, var in _MAKE.items():
            script = (
                script.replace(var, shlex.quote(str(process.tool(snapshot.config, host)))) if var in script else script
            )
        result = process.run("build.materialize", ["/bin/sh", "-c", script], root, tmp=process.scratch(root))
        for o in outputs:
            if result.exit == 0 and (root / o).is_file():
                os.utime(root / o, ns=(begun, begun))
    return (
        found
        if result.exit == 0
        else (
            Finding(
                "link.error",
                f"{unit.path}: {result.stderr.decode(errors='replace')[-300:]}",
                unit=unit.path,
                versions=(version,),
            ),
        )
    )


def makefile(snapshot: Snapshot) -> bytes:
    with effort.stage("build.makefile"):
        cfg, project = snapshot.config, snapshot.config.project
        rows = configuration.toolchains()
        used = sorted({u.toolchain for u in snapshot.layout.units.values()} | {project.toolchain})
        pinned = _fetched(used)
        tools = [
            f"ifneq ($(filter default undefined,$(origin {var})),)\n{var} := "
            + (f"tools/{row['host']}/{row['binary']}" if var in pinned else str(row["default"]))
            + "\nendif"
            for var, row in sorted(_REPO["make"]["tools"].items())
        ]  # make's own CPP and AS beat a ?= line
        blocks, chains, rom_rules, checks = [], [], [], []
        for var in sorted(pinned):  # a pinned tool the project's toolchains run is fetched like a compiler
            row = _REPO["make"]["tools"][var]
            chains.append(_fetch_rule(f"tools/{row['host']}", row["download"], row["pins"]))
        for id in used:
            key, row = id.replace("-", "_").replace(".", "_"), rows[id]
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
            if n:
                chains.append(_fetch_rule(f"tools/{id}", row["downloads"][0], row["pins"]))
        units = sorted(snapshot.layout.units.values(), key=lambda u: u.path)
        blocks = delta.map(  # a landing renders the rules of the units it changed; the others keep theirs
            cfg,
            "build.units",
            _unit_job,
            [(snapshot, u, v) for u in units for v in _versions(snapshot, u)],
            key=_unit_key,
            ident=_unit_ident,
            stamp=_unit_stamp,
            pin=lambda path: view._pin(snapshot, path),
            context=_rules_context(snapshot),
        )
        for version in project.versions:
            files, rom = project.version_files[version], f"build/{version}/{project.name}.z64"
            checks.append(f"\t$(Q)test \"$$(sha1sum < {rom} | cut -d' ' -f1)\" = {files.baserom_sha1}")
            rom_rules.append(
                f"{rom}: {files.baserom} Makefile\n"
                f"\t$(Q)mkdir -p $(@D) && cp {shlex.quote(files.baserom)} $@ && "
                "for p in $(PATCHES); do dd if=$${p%:*} of=$@ bs=64K oflag=seek_bytes seek=$${p##*:} "
                "conv=notrunc status=none || exit 1; done\n"
                f"\t$(Q)test \"$$({{ sha1sum < $@; }} | cut -d' ' -f1)\" = {files.baserom_sha1}\n"
            )
        return (
            Template(configuration.template("Makefile.in"))
            .substitute(
                title=project.title,
                name=project.name,
                versions=" ".join(project.versions),
                tool_defaults="\n".join(tools),
                toolchain_rules="\n".join(chains),
                units="\n".join(blocks),
                tree_checks="\n".join(
                    f"\t$(Q)test -s {path}"
                    for path in [
                        *(f"versions/{v}/report.json" for v in project.versions),
                        *present.paths(project.versions),
                    ]
                ),
                roms=" ".join(f"build/{v}/{project.name}.z64" for v in project.versions),
                rom_checks="\n".join(checks),
                rom_rules="\n".join(rom_rules),
            )
            .encode()
        )


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


def _tool_stamps(commands: str) -> list[str]:
    """The fetch stamps of the pinned tools a unit's commands run."""
    return [
        f"tools/{row['host']}/.stamp"
        for var, row in sorted(_REPO["make"]["tools"].items())
        if "download" in row and f"$({var})" in commands
    ]


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
            key, base = id.replace("-", "_").replace(".", "_"), config.host.toolchain_root / id
            assignments.append(f"CC_{key} := {base / row['cc']}")
            assignments.append(f"AS_{key} := {base / row['as']}" if "as" in row else f"AS_{key} := $(AS)")
            assignments.append(f"TOOLCHAIN_{key} :=")
        return Template(configuration.template("local.mk.in")).substitute(assignments="\n".join(assignments)).encode()


def symbols_ld(snapshot: Snapshot, version: str) -> bytes:
    with effort.stage("build.symbols_ld"):
        rows, mask = snapshot.versions[version].symbols, snapshot.config.project.build.get("text_mask", _WORD)
        if mask != _WORD:  # the code is linked at its masked address, so a symbol in it is provided there too
            spans = _text_spans(snapshot, version)
            table = symbols.load(snapshot.read, snapshot.config.project.versions, store.content(snapshot.config).cached)
            code = snapshot.versions[version].code
            rows = {
                n: a & mask
                if n in code
                or table.get(n, {}).get("kind") == "function"
                or spans[bisect_right(spans, (a, _WORD)) - 1][1] > a
                else a
                for n, a in rows.items()
            }
        return "".join(f"PROVIDE({name} = 0x{rows[name]:08X});\n" for name in sorted(rows)).encode("utf-8")


def _text_spans(snapshot: Snapshot, version: str) -> list[tuple[int, int]]:
    """The disjoint (start, end) vram runs that text rows cover, sorted, with a sentinel before the first."""

    def spans() -> list[tuple[int, int]]:
        merged = [(-1, 0)]
        for start, end in sorted(
            (p.vram, p.vram + p.rom_end - p.rom_start)
            for m in snapshot.layout.members.values()
            for p in m.placements
            if p.version == version and p.section == ".text"
        ):
            merged.append((start, end)) if start > merged[-1][1] else merged.__setitem__(
                -1, (merged[-1][0], max(end, merged[-1][1]))
            )
        return merged

    return effort.memo(("spans", snapshot.digest, version), spans)


def _split_input(text: str) -> dict[str, Any]:
    document: dict[str, Any] = yaml.load(text, Loader=yaml.CSafeLoader)
    rows = [row for s in document["segments"] if isinstance(s, dict) for row in [s, *s.get("subsegments", [])]]
    for row in rows:
        key = "name" if isinstance(row, dict) else 2
        if (isinstance(row, dict) and key not in row) or (isinstance(row, list) and len(row) < 3):
            continue
        name = row[key]
        if not isinstance(name, str):
            continue
        bounded = "/".join(
            part
            if len(part.encode()) <= 240
            else part.encode()[:180].decode(errors="ignore") + "_" + sha256(part.encode())[:16]
            for part in name.split("/")
        )
        row[key] = bounded
    return document


_READ_PATHS = ("target_path", "symbol_addrs_path", "reloc_addrs_path", "extensions_path")
_UNTOUCHED = ("base_path", "data_path", "nonmatchings_path", "matchings_path")


def _publish(stage: Path, root: Path, owned: Path) -> None:
    """Copy what splat wrote under stage into the project where bytes differ; drop asm files the run did not write."""
    old, new, written = str(stage).encode() + b"/", str(root).encode() + b"/", set()
    for source in stage.rglob("*"):
        if not source.is_file():
            continue
        target = root / source.relative_to(stage)
        written.add(target)
        store.write(target, source.read_bytes().replace(old, new))
    for stale in (p for p in (root / owned).rglob("*") if p.is_file() and p not in written):
        stale.unlink()


def _label_tables(asm: Path, segments: list[Any], mask: int) -> None:
    """A jump table word in a segment under the text mask names its target's label as the function's asm spells it."""

    def start(s: Any) -> int:  # a segment may be written as a list
        return int(s["start"] if isinstance(s, dict) else s[0])

    spans = [
        ((s["vram"] & mask), (s["vram"] & mask) + start(n) - start(s), s["vram"] & ~mask)
        for s, n in pairwise(segments)
        if isinstance(s, dict) and s.get("vram")
    ]

    def label(word: re.Match[str]) -> str:
        value = int(word[2], 16)
        high = next((h for lo, hi, h in spans if lo <= value < hi), None)
        return word[0] if high is None else f"{word[1]}.Lauto_{high | value:08X}"

    def table(block: re.Match[str]) -> str:
        return re.sub(r"(\.word )0x([0-9A-F]{8})\b", label, block[0])

    for path in asm.rglob("*.s"):
        text = path.read_text(errors="replace")
        if "dlabel jtbl_" in text:
            path.write_text(re.sub(r"(?ms)^dlabel jtbl_\w+$.*?^enddlabel", table, text))


def _splat_identity(config: Config) -> str:
    """The installed splat package version; a host whose splat is no package is named by its tool file."""
    try:
        return metadata.version("splat64")
    except metadata.PackageNotFoundError:
        tool = process.tool(config, "splat")
        return f"{tool}:{tool.stat().st_mtime_ns if tool.exists() else 0}"


def _extract_key(config: Config, version: str, document: dict[str, Any]) -> str:
    """What a split reads: rows, ROM, every file of every read-path option, splat and this code."""
    root, options = config.project.root, document.get("options", {})
    defaults = {"symbol_addrs_path": "symbol_addrs.txt", "reloc_addrs_path": "reloc_addrs.txt"}
    reads: set[Path] = set()
    for name in _READ_PATHS:
        value = options.get(name, defaults.get(name))
        for item in value if isinstance(value, list) else [value]:
            if item is not None:
                path = root / item
                reads.update([p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path])
    return digest(
        (
            version,
            yaml.safe_dump(document, sort_keys=True),
            config.project.version_files[version].baserom_sha1,
            [(str(p), p.read_bytes() if p.is_file() else None) for p in sorted(reads)],
            config.project.build.get("text_mask", _WORD),
            _splat_identity(config),
            code_identity(extract),
        )
    )


def extract(config: Config, version: str) -> NativeResult:
    with effort.stage("build.extract"):
        root = config.project.root
        split = root / config.project.version_files[version].split
        document = _split_input(split.read_text())
        stamp = _extract_key(config, version, document)
        facts = root / ".unbake" / "symbols" / version
        marker = root / ".unbake" / "extract" / f"{version}.key"  # per root: the tree it vouches for is this one's
        if marker.is_file() and marker.read_text() == stamp and any(facts.glob("*.csv")):
            return NativeResult(("splat",), str(root), 0, None, b"", b"", 0.0, 0.0, 0, {})  # same inputs, same tree
        with store.work(config) as work:
            stage = work / "tree"
            options = document.setdefault("options", {})
            defaults = dict(
                asm_path="asm",
                src_path="src",
                asset_path="assets",
                build_path="build",
                lib_path="lib",
                o_path="build",
                generated_asm_macros_directory="include",
                symbol_addrs_path="symbol_addrs.txt",
                reloc_addrs_path="reloc_addrs.txt",
                undefined_funcs_auto_path="undefined_funcs_auto.txt",
                undefined_syms_auto_path="undefined_syms_auto.txt",
                ld_script_path=f"{options.get('basename', 'rom')}.ld",
            )
            for key, value in defaults.items():
                options.setdefault(key, value)
            for key, value in list(options.items()):
                if (key.endswith("_path") and key not in _UNTOUCHED) or key == "generated_asm_macros_directory":
                    home = root if key in _READ_PATHS else stage
                    options[key] = [str(home / p) for p in value] if isinstance(value, list) else str(home / value)
            options.update(base_path=str(work), cache_path=str(work / ".splache"), dump_symbols=True)
            private = work / "split.yaml"
            private.write_text(yaml.safe_dump(document, sort_keys=False))
            result = process.run(
                "splat",
                [process.tool(config, "splat"), "split", str(private), "--make-full-disasm-for-code"],
                root,
                tmp=process.scratch(root),
            )
            if result.exit == 0:
                mask = config.project.build.get("text_mask", _WORD)
                if mask != _WORD:
                    _label_tables(Path(options["asm_path"]), document["segments"], mask)
                _publish(stage, root, Path(options["asm_path"]).relative_to(stage))
                facts.mkdir(parents=True, exist_ok=True)
                for path in facts.glob("*.csv"):
                    path.unlink()
                for path in (work / ".splat").glob("*.csv"):
                    shutil.copyfile(path, facts / path.name)
                store.write(marker, stamp.encode())
        if result.exit != 0:
            tail = result.stderr.decode(errors="replace").splitlines()
            raise Refusal(
                Finding("setup.splat", versions=(version,), reason="\n".join(tail[-20:]) or f"exit {result.exit}")
            )
        return result


def inputs(config: Config) -> str:
    """What `make check` reads or builds, by name, size and mtime: while it is unchanged the last check still holds."""
    root, project = str(config.project.root), config.project
    roms = (f"build/{v}/{project.name}.z64" for v in project.versions)
    paths = [f"{root}/{p}" for p in ("Makefile", "local.mk", *roms)]
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


def compile_jobs(snapshot: Snapshot, paths: Sequence[str]) -> list[tuple[Snapshot, UnitSpec, str]]:
    phases = configuration.kinds()
    return [
        (snapshot, u, v)
        for p in sorted(paths)
        if (u := snapshot.layout.units.get(p))
        and "preprocess" in phases[u.kind]["phases"]
        and phases[u.kind]["cacheable"]
        and phases[u.kind]["decompiled"]
        for v in sorted(snapshot.layout.held(u.members))
    ]


def make_check(config: Config, snapshot: Snapshot) -> Json:
    with effort.stage("build.make_check"):
        stamp, known = config.project.root / ".unbake/make.json", inputs(config)
        done = {"exit": 0, "versions": list(config.project.versions)}
        if stamp.exists() and json.loads(stamp.read_bytes()).get("inputs") == known:
            effort.count("make-check", True)
            return done
        effort.count("make-check", False)
        root = config.project.root
        broken = [  # the content cache builds each unit and leaves its make outputs, so make below only links
            f
            for found in pool.map(
                config,
                "land.compile",
                compile_unit,
                [
                    job
                    for job in compile_jobs(snapshot, list(snapshot.layout.units))
                    if job[2] not in job[1].withheld  # prewarm only the holders the generated make rules build
                ],
            )
            for f in found
            if f.blocking
        ]
        if broken:
            raise Refusal(*(replace(f, key="setup.make") for f in broken))
        with pool.jobserver(config.host) as (_, env, fds):  # elastic: grows into the slots other commands free
            argv = [str(process.tool(config, "make")), "-k", "check"]
            result = process.run("make check", argv, root, tmp=process.scratch(root), env=env, pass_fds=fds)
        if result.exit == 0:
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text(json.dumps({"inputs": inputs(config)}))
            return done
        lines = result.stderr.decode(errors="replace").splitlines()
        failed = [line for line in lines if "***" in line]  # make -k ran everything it could: every failed target
        raise Refusal(Finding("setup.make", "\n".join(failed or lines[-20:]), missing=(f"make exit {result.exit}",)))
