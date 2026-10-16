"""ROM validation, repository bootstrap and the setup spine."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import sys
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from functools import cache
from pathlib import Path
from string import Template
from typing import Any

import psutil
import rabbitizer
import yaml

from unbake import (
    adapters,
    build,
    ci,
    effort,
    infer,
    journal,
    layout,
    libasm,
    native,
    pool,
    present,
    process,
    report,
    store,
    symbols,
    types,
    versions,
)
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, NativeResult, Plan, Project, Refusal, Snapshot, digest, sha1


def _mask(word: int) -> int:
    opcode = rabbitizer.Instruction(word).getOpcodeName()
    if opcode in {"lui", "addiu", "ori", "lw", "sw", "lh", "sh", "lb", "sb", "lwc1", "swc1"}:
        return word & 0xFFFF0000
    return word & 0xFC000000 if opcode == "jal" else word


def _windows(data: bytes, row: Json) -> set[tuple[int, ...]]:
    start, length, window = row["start"], row["length"], row["window"]
    return {
        tuple(_mask(int.from_bytes(chunk[j : j + 4], "big")) for j in range(0, len(chunk), 4))
        for i in range(start, min(start + length, len(data)), window)
        if len(chunk := data[i : min(i + window, start + length)]) == window
    }


def validate_roms(roms: Mapping[str, Path], names_from: str) -> Json:
    with effort.stage("repo.validate_roms"):
        rules = configuration.load_resource("rom.toml")
        data: dict[str, bytes] = {}
        result: dict[str, dict[str, Any]] = {}
        orders: list[Finding] = []
        duplicates: list[Finding] = []
        mismatches: list[Finding] = []
        hashes: dict[str, Any] = {}
        for v, path in sorted(roms.items()):
            try:
                data[v] = path.read_bytes()
            except OSError as error:
                raise Refusal(Finding("init.request", str(error), path=str(path))) from error
            word = int.from_bytes(data[v][:4], "big")
            if word != rules["order"]["z64"]:
                name = next((r["name"] for r in rules["order"]["refuse"] if r["word"] == word), "an unknown order")
                orders.append(
                    Finding(
                        "init.rom_order",
                        f"{v}: first word 0x{word:08X} is {name}; convert the dump to big-endian .z64",
                        path=str(path),
                    )
                )
            rom_sha = sha1(data[v])
            if rom_sha in hashes:
                duplicates.append(
                    Finding(
                        "init.rom_duplicate",
                        f"{hashes[rom_sha]} and {v} have identical ROM bytes",
                        versions=(hashes[rom_sha], v),
                    )
                )
            hashes[rom_sha] = v
            fields = {}
            for row in rules["field"]:
                raw = data[v][row["offset"] : row["offset"] + row["size"]]
                fields[row["name"]] = (
                    raw.decode("ascii", "replace").rstrip("\x00 ") if row["encoding"] == "ascii" else raw.hex()
                )
            result[v] = {"sha1": rom_sha, "fields": fields, "similarity": 1.0}
        for findings in (orders, duplicates):
            if findings:
                raise Refusal(*findings)
        if names_from not in result:
            raise Refusal(Finding("init.request", f"{names_from} is not one of the roms"))
        reference = _windows(data[names_from], rules["similarity"])
        for v in result:
            if v == names_from:
                continue
            for row in rules["field"]:
                field = row["name"]
                value, ref = result[v]["fields"][field], result[names_from]["fields"][field]
                if row["equal"] and value != ref:
                    mismatches.append(
                        Finding(
                            "init.rom_mismatch",
                            f"{v}: {field} {value!r} differs from {names_from} {ref!r}",
                            versions=(v,),
                        )
                    )
            similarity = len(reference & _windows(data[v], rules["similarity"])) / len(reference) if reference else 0.0
            result[v]["similarity"] = similarity
            if similarity < rules["similarity"]["min"]:
                mismatches.append(
                    Finding(
                        "init.rom_mismatch",
                        f"{v}: similarity {similarity:.6f} to {names_from} is below {rules['similarity']['min']}",
                        versions=(v,),
                    )
                )
        if mismatches:
            raise Refusal(*mismatches)
        return result


def _render(template_name: str, **values: object) -> bytes:
    return Template(configuration.template(template_name)).substitute(values).encode()


_EMPTY_TYPES = b"schema = 1\n\n[function]\n\n[global]\n\n[struct]\n"


def _extras(rules: Json, title: str, versions: Iterable[str]) -> dict[str, bytes]:
    roms = "\n".join(f"- `roms/baserom.{v}.z64`" for v in versions)
    return {
        "CONTRIBUTING.md": _render("CONTRIBUTING.md.in", title=title, roms=roms),
        "types.toml": _EMPTY_TYPES,
        ".recomp.json": present.recomp(title),
        **{row["path"]: configuration.template(row["template"]).encode() for row in rules["sdk"]},
    }


def _checked(result: NativeResult, key: str, name: str) -> None:
    if result.exit != 0 or result.signal is not None:
        tail = result.stderr.decode(errors="replace").splitlines()
        raise Refusal(Finding(key, f"{name}: {tail[-1] if tail else f'exit {result.exit}, signal {result.signal}'}"))


def _options(rules: Json, version: str, name: str, rom: str) -> Json:
    return {
        k: value.format(version=version, name=name, rom=rom) if isinstance(value, str) else value
        for k, value in rules["splat"]["options"].items()
    }


def _require_options(config: Config) -> None:
    """A split yaml without the options init writes would let splat write C files over the tracked sources."""
    rules, project, found = configuration.load_resource("repo.toml"), config.project, []
    for v, files in project.version_files.items():
        options = yaml.safe_load((project.root / files.split).read_bytes()).get("options") or {}
        wanted = _options(rules, v, project.name, files.baserom)
        found += [
            f"{files.split}: {k}"
            for k, value in wanted.items()
            if k not in options or (not isinstance(value, str) and options[k] != value)
        ]
    if found:
        raise Refusal(
            Finding(
                "setup.options",
                configuration.sentence("setup.options"),
                missing=tuple(found),
                action="set these options in the split yaml to the values unbake init writes",
            )
        )


class _Dumper(yaml.SafeDumper):
    """A split yaml as the boundary editor reads it: a row of plain values on one flow-style line."""


_Dumper.add_representer(
    list,
    lambda dumper, rows: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", rows, flow_style=not any(isinstance(x, (list, dict)) for x in rows)
    ),
)


def _dump_split(document: Json) -> bytes:
    return str(yaml.dump(document, Dumper=_Dumper, sort_keys=False, width=1 << 30)).encode()


def _users(prefix: Path) -> list[str]:
    """Every other process running from this environment (any command, its pool workers, make under setup): a swap
    under one leaves its output made by two tool versions."""
    here, mine = f"{prefix}/", os.getpid()
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        line = proc.info["cmdline"] or []
        if proc.info["pid"] != mine and any(str(arg).startswith(here) for arg in line):
            found.append(f"{proc.info['pid']} {' '.join(map(str, line))[:160]}")
    return found


def _source() -> str | None:
    """The URL this tool was installed from, as pip recorded it."""
    try:
        recorded = importlib.metadata.distribution("unbake").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    url = json.loads(recorded).get("url") if recorded else None
    return url if url and not url.startswith("file:") else None


def upgrade(params: Json) -> Json:
    """Install REF of the tool into the environment running it, refused while any other process runs from it."""
    prefix, ref = Path(sys.prefix), params["ref"]
    if busy := _users(prefix):
        raise Refusal(
            Finding(
                "upgrade.busy",
                f"{len(busy)} processes run from {prefix}",
                missing=tuple(busy),
                action="upgrade when no command of this environment runs",
            )
        )
    source = _source()
    if source is None:
        raise Refusal(Finding("upgrade.source", f"{prefix} was not installed from a URL", action="install it by hand"))
    argv = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--force-reinstall", "--no-deps"]
    result = process.run("pip", [*argv, f"git+{source}@{ref}"], Path.cwd(), tmp=Path.cwd())
    if result.exit != 0 or result.signal is not None:
        raise Refusal(Finding("upgrade.install", result.stderr.decode(errors="replace")[-2000:] or "pip failed"))
    return {"ref": ref, "prefix": str(prefix), "source": source}


def init(params: Json) -> Json:
    with effort.stage("repo.init"):
        root, name, roms = Path(params["dir"]).absolute(), params["name"], {}
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise Refusal(Finding("init.request", f"{root} is not an empty directory", path=str(root)))
        for item in params["rom"]:
            v, sep, path = item.partition("=")
            if not sep or not v or not path or v in roms or not Path(path).is_file():
                raise Refusal(Finding("init.request", f"--rom {item} is not a unique VERSION=PATH naming a file"))
            roms[v] = Path(path)
        toolchains = configuration.toolchains()
        if params["names_from"] not in roms or params["toolchain"] not in toolchains:
            raise Refusal(Finding("init.request", "names_from must name a ROM and toolchain must name a toolchain"))
        host = configuration.load_host(Path(params["host"]))
        versions, rules = validate_roms(roms, params["names_from"]), configuration.load_resource("repo.toml")
        created, blocks = [], ["[resident]\n"]

        def write(path: str, value: bytes) -> None:
            store.write(root / path, value)
            created.append(path)

        write(".gitignore", ("\n".join(rules["ignore"]) + "\n").encode())
        for v, source in sorted(roms.items()):
            path = f"roms/baserom.{v}.z64"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, root / path)
            created.append(path)
        for v in sorted(roms):
            rom, split = f"roms/baserom.{v}.z64", f"versions/{v}/{name}.yaml"
            symbol_file = f"versions/{v}/symbol_addrs.txt"
            before = set(root.glob("*.yaml"))
            result = process.run(
                "splat create_config", [str(host.tools["splat"]), "create_config", rom], root, tmp=process.scratch(root)
            )
            _checked(result, "setup.splat", "splat create_config")
            made = set(root.glob("*.yaml")) - before
            if len(made) != 1:
                raise Refusal(Finding("setup.splat", f"expected one new yaml, found {len(made)}", versions=(v,)))
            (root / split).parent.mkdir(parents=True, exist_ok=True)
            made.pop().rename(root / split)
            document = yaml.safe_load((root / split).read_bytes())
            document["options"].update(_options(rules, v, name, rom))
            write(split, _dump_split(document))
            write(symbol_file, b"")
            blocks.append(
                f'\n[version.{v}]\nbaserom = {json.dumps(rom)}\nbaserom_sha1 = "{versions[v]["sha1"]}"\n'
                f"split = {json.dumps(split)}\nsymbols = {json.dumps(symbol_file)}\nmacros = []\n"
            )
        values = {key: json.dumps(params[key]) for key in ("name", "title", "names_from", "toolchain")}
        write(
            "config.toml",
            _render(
                "config.toml",
                **values,
                id=json.dumps(str(uuid.uuid4())),
                versions=json.dumps(sorted(roms)),
                version_blocks="".join(blocks),
            ),
        )
        write("layout.toml", b"schema = 5\ngroup = []\nunit = []\n")
        write(symbols.path(), symbols.empty())
        write(
            "README.md",
            _render("README.md.in", title=params["title"], begin=rules["readme"]["begin"], end=rules["readme"]["end"]),
        )
        for path, value in _extras(rules, params["title"], sorted(roms)).items():
            write(path, value)
        _checked(
            process.run("git init repository", [str(host.tools["git"]), "init"], root, tmp=process.scratch(root)),
            "native.exit",
            "git init",
        )
        return {"dir": str(root), "created": sorted(created), "versions": versions}


def files(snapshot: Snapshot) -> dict[str, bytes]:
    with effort.stage("repo.files"):
        rules, project = configuration.load_resource("repo.toml"), snapshot.config.project
        wanted = {
            "Makefile": build.makefile(snapshot),
            **report.files(snapshot),
            **{f"versions/{v}/symbols.ld": build.symbols_ld(snapshot, v) for v in project.versions},
            ".gitignore": ("\n".join(rules["ignore"]) + "\n").encode(),
            **ci.files(project.versions),
        }
        # the manifest lists every generated file; the starter files below are the author's once they exist
        wanted[_MANIFEST] = "".join(f"{p}\n" for p in sorted({*wanted, _MANIFEST})).encode()
        for path, value in _extras(rules, project.title, project.versions).items():
            if snapshot.peek(path) is None:
                wanted[path] = value
        return wanted


def extract_job(job: tuple[Config, str]) -> NativeResult:
    return build.extract(*job)


def _prove_unit(job: tuple[Snapshot, str, str]) -> tuple[Any, ...]:
    """One extracted member in one version: its assembly unit is made here, in the worker, not for every member in the
    parent before any of them is dispatched."""
    snapshot, member, version = job
    return native.prove_job((snapshot, layout.asm_unit(snapshot, member, version), version))


def _prove_key(job: tuple[Snapshot, str, str]) -> str:
    snapshot, member, version = job
    return native.stamp(snapshot, layout.asm_unit(snapshot, member, version), version)


def _prove(snapshot: Snapshot, versions: Iterable[str]) -> dict[str, Any]:
    owned = {m for unit in snapshot.layout.units.values() for m in unit.members}
    jobs = [
        (snapshot, m.name, v)
        for m in snapshot.layout.members.values()
        if m.kind == "function" and m.name not in owned
        for v in m.holders()
        if v in versions
    ]
    counts = {v: {"members": 0, "exact": 0, "bytes": 0} for v in versions}
    missing: list[Any] = []
    bad = []
    for (_, member, v), proofs in zip(
        jobs, pool.map(snapshot.config, "repo.prove", _prove_unit, jobs, _prove_key), strict=True
    ):
        exact = bool(proofs) and all(p.exact for p in proofs)
        counts[v]["members"] += 1
        counts[v]["exact"] += int(exact)
        counts[v]["bytes"] += sum(p.size for p in snapshot.layout.members[member].placements if p.version == v)
        if not exact:
            bad.append(v)
            missing.extend(s for p in proofs for s in p.missing)
            if not proofs:
                missing.append(f"{v}: {member}: no proof")
    if bad:
        raise Refusal(
            Finding(
                "setup.not_exact",
                f"{len(bad)} extracted members do not rebuild their ROM bytes",
                missing=tuple(missing[:20]),
                versions=tuple(sorted(set(bad))),
            )
        )
    return counts


def _built(snapshot: Snapshot) -> dict[str, dict[str, int]]:
    """Per version the compiled units built and the ones withheld there."""
    phases = configuration.kinds()
    out = {v: {"built": 0, "withheld": 0} for v in snapshot.config.project.versions}
    for unit in snapshot.layout.units.values():
        if "compile" in phases[unit.kind]["phases"]:
            for v in snapshot.layout.held(unit.members):
                out[v]["withheld" if v in unit.withheld else "built"] += 1
    return out


def _paths(project: Project) -> dict[str, str]:
    return {v: vf.symbols for v, vf in project.version_files.items()}


def _changed(snapshot: Snapshot, wanted: Mapping[str, bytes]) -> dict[str, bytes]:
    return {path: value for path, value in wanted.items() if snapshot.peek(path) != value}


_MANIFEST = "generated.txt"


def _stale(snapshot: Snapshot, wanted: Mapping[str, bytes]) -> dict[str, None]:
    """Files the previous manifest lists that this setup no longer generates, and nothing else."""
    try:
        before = snapshot.read(_MANIFEST).decode().splitlines()
    except FileNotFoundError:
        return {}
    gone: dict[str, Any] = {}
    for path in sorted(set(before) - wanted.keys()):
        if path.startswith("/") or ".." in path.split("/"):
            raise Refusal(
                Finding("config.file", f"{_MANIFEST} lists {path}, which is not inside the project", path=_MANIFEST)
            )
        if snapshot.peek(path) is not None:
            gone[path] = None
    return gone


def _plan(snapshot: Snapshot, writes: Mapping[str, bytes | None], operation: str, message: str) -> Plan:
    return Plan(operation, snapshot.digest, writes, (), (), (), message, digest((snapshot.digest, writes, message)))


@cache
def _splat_version(tool: Path, root: Path) -> bytes:
    result = process.run("splat version", [str(tool), "--version"], tool.parent, tmp=process.scratch(root))
    if result.exit:
        raise Refusal(Finding("setup.splat", result.stderr.decode(errors="replace")))
    return result.stdout


def _stamp(config: Config, v: str, symbol_file: bytes | None = None) -> str:
    root, vf = config.project.root, config.project.version_files[v]
    symbol_file = (root / vf.symbols).read_bytes() if symbol_file is None else symbol_file
    return digest(
        (
            (root / vf.split).read_bytes(),
            symbol_file,
            configuration.load_resource("repo.toml")["splat"],
            _splat_version(config.host.tools["splat"], root),
            sha1((root / vf.baserom).read_bytes()),
        )
    )


def _stamps(config: Config) -> dict[str, Any]:
    """The splat stamps of the last extraction, by version."""
    path = config.project.root / ".unbake/extract.json"
    return json.loads(path.read_bytes()) if path.exists() else {}


def _state(config: Config) -> str:
    """Everything setup reads or writes: the tool, the repository, the ROMs and what was built. Unchanged means done."""
    root = config.project.root
    head = process.git(config, "rev-parse", "HEAD").stdout.decode().strip()
    tool = digest(
        [p.read_bytes() for p in sorted(Path(__file__).parent.rglob("*")) if p.is_file() and p.suffix != ".pyc"]
    )  # the tool's content, not when it was touched
    top = [(p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(root.iterdir()) if p.is_file()]
    return digest((head, config.digest, tool, top, build.inputs(config), _stamps(config)))


def _drop_stale_facts(config: Config) -> None:
    """Extraction output is true only for its inputs: facts of a version whose stamp mismatches are removed."""
    root, project = config.project.root, config.project
    if not (root / symbols.path()).is_file():
        raise Refusal(
            Finding(
                "config.missing",
                f"{symbols.path()} does not exist",
                path=symbols.path(),
                action=f"create {symbols.path()} as the empty symbol table that unbake init writes",
            )
        )
    stamps = _stamps(config)
    table = symbols.parse((root / symbols.path()).read_bytes(), project.versions)
    for v in project.versions:
        if stamps.get(v) != _stamp(config, v, symbols.render(table, v)):
            stale_files: list[str | Path] = [
                *versions.fact_files(config, v)[0],
                *(root / ".unbake/symbols" / v).glob("*.csv"),
            ]
            for path in stale_files:
                (root / path).unlink(missing_ok=True)


def setup(config: Config, params: Json) -> Json:
    wait_s = config.host.move_wait_s
    with effort.stage("repo.setup"), store.exclusive(config, "structure", wait=True, wait_s=wait_s) as held:
        adapters.check_abucache()
        if not held:
            raise Refusal(Finding("setup.busy", configuration.sentence("setup.busy")))
        journal.recover(config)
        _require_options(config)
        done, state = config.project.root / ".unbake/setup.json", _state(config)
        kept = json.loads(done.read_bytes()) if done.exists() else None
        if kept and kept["state"] == state:
            effort.count("setup", True)
            return {**kept["result"], "messages": [], "extracted": [], "files": []}
        effort.count("setup", False)
        _drop_stale_facts(config)
        project = config.project
        messages: list[Any] = []
        written: set[Any] = set()

        snapshot: Snapshot
        moved: set[str] = set()  # versions extracted again since the proof: proved once, when the layout settles
        extracting = False

        def apply(plan: Plan) -> bool:
            """Write the plan. Once extraction has run, a version whose split or symbols file it wrote is extracted
            again and the project captured; true then, else the snapshot is the plan laid over the old one."""
            nonlocal snapshot
            journal.apply(config, plan, snapshot)
            messages.append(plan.message)
            written.update(plan.writes)
            changed = [v for v, vf in project.version_files.items() if {vf.split, vf.symbols} & plan.writes.keys()]
            if extracting and changed:
                moved.update(changed)
                extract(changed)
                snapshot = layout.capture(config)
                return True
            snapshot = replace(layout.overlay(snapshot, plan.writes), reads=snapshot.reads)
            return False

        def extract(versions: Iterable[str]) -> None:
            """Splat runs for the versions whose inputs moved since its last run, wherever setup asks."""
            stale = [v for v in versions if stamps.get(v) != _stamp(config, v)]
            for v in versions:
                effort.count("extract", v not in stale)
            results = pool.map(config, "repo.extract", extract_job, [(config, v) for v in stale])
            for v, result in zip(stale, results, strict=True):
                _checked(result, "setup.splat", f"extract {v}")
                stamps[v] = _stamp(config, v)
            effort.forget("generated", "versions.read", "pin", "includes", "closure")
            store.write(project.root / ".unbake/extract.json", json.dumps(stamps, sort_keys=True).encode())
            extracted.update(stale)

        with effort.stage("repo.setup.toolchains"):
            snapshot = layout.capture(config)
            table = symbols.edit(snapshot)
            writes: dict[str, bytes | None] = {**_changed(snapshot, symbols.files(table, _paths(project)))}
            writes.update(
                {
                    path: value
                    for path, value in _extras(
                        configuration.load_resource("repo.toml"), project.title, project.versions
                    ).items()
                    if not (project.root / path).exists()
                }
            )
            if writes:
                apply(_plan(snapshot, writes, "setup", "setup: extraction inputs"))
            toolchains = sorted({project.toolchain, *(u.toolchain for u in snapshot.layout.units.values())})
            build.install_toolchains(config, toolchains)
        with effort.stage("repo.setup.extract"):
            extracted: set[Any] = set()
            stamps = _stamps(config)
            extract(project.versions)
            extracting = True
        with effort.stage("repo.setup.prove"):
            snapshot = layout.capture(config)
            proof = _prove(snapshot, project.versions)
        joined, recaptured = 0, False

        def converge() -> Plan:
            """One map pass: plan, read the names the versions give one thing and the addresses the ROM code forms, plan
            again, and write it all in one plan; changed versions are extracted again. A second pass must find nothing
            to write, else the layout is not a function of the ROMs and setup refuses naming what moved."""
            nonlocal snapshot, joined, recaptured
            snapshot = layout.capture(config)
            try:
                mapped = infer.plan(snapshot)
            except Refusal as refusal:
                if recaptured or any(f.key != "journal.changed" for f in refusal.findings):
                    raise
                recaptured = True
                return converge()
            if mapped.blocking:  # landed C that does not compile in a version it holds is not debt: setup refuses
                raise Refusal(*mapped.blocking)
            table = symbols.edit(snapshot)  # names the versions give one thing become one symbol
            read, misnamed = infer.addresses(snapshot, mapped.debt)
            mapped = replace(mapped, debt=(*mapped.debt, *misnamed))
            found = symbols.join_pairs(snapshot, table, infer.correspondences(snapshot)) + symbols.fix_rows(table, read)
            writes, message = dict(mapped.writes), mapped.message
            if found:  # plan again on the table the readings give, so the same plan holds the layout it implies
                writes.update(_changed(snapshot, symbols.files(table, _paths(project))))
                settled = infer.plan(layout.overlay(snapshot, writes))
                writes.update(settled.writes)
                message = f"{settled.message}; {found} symbols read from the ROM"
            if not writes:
                return mapped
            split = any(vf.split in writes or vf.symbols in writes for vf in project.version_files.values())
            planned = None if split else layout.overlay(snapshot, writes).digest  # what the writes give, unextracted
            if not apply(_plan(snapshot, writes, "layout", message)):
                snapshot = layout.capture(config)
            joined += found
            if planned is not None and snapshot.digest == planned:  # the plan is its own fixpoint
                return replace(mapped, writes={})
            again = infer.plan(snapshot)  # extraction told something else: the layout must still hold under it
            again = replace(again, debt=(*again.debt, *misnamed))
            if again.writes:
                raise Refusal(
                    Finding(
                        "layout.nonconvergent",
                        "a second map pass still changes "
                        f"{', '.join(sorted(again.writes)[:8])} ({len(again.writes)} files) after the "
                        f"first pass wrote {len(writes)} files: {again.message}",
                    )
                )
            return again

        with effort.stage("repo.setup.map"):
            mapped = converge()
            if moved:
                proof.update(_prove(layout.capture(config), sorted(moved)))
                moved.clear()
            snapshot = layout.capture(config)
            writes = {**_changed(snapshot, {"types.toml": types.scan(snapshot)})}
            if writes:
                apply(_plan(snapshot, writes, "layout", "setup: type map"))
        with effort.stage("repo.setup.boundary"):
            plan, boundary = layout.boundary_plan(snapshot)
            if plan.writes:
                apply(plan)
                changed = [v for v, vf in project.version_files.items() if vf.split in plan.writes]
                proof.update(_prove(snapshot, changed))
                moved.clear()
                if layout.boundary_plan(snapshot)[0].writes:
                    raise Refusal(Finding("layout.nonconvergent", configuration.sentence("layout.nonconvergent")))
                mapped = converge()  # the functions moved: the units and their data follow
                if moved:
                    proof.update(_prove(layout.capture(config), sorted(moved)))
        with effort.stage("repo.setup.library"):
            if writes := libasm.plan(snapshot)[0]:  # foreign routines become library assembly from the ROM bytes
                apply(_plan(snapshot, writes, "layout", "setup: library assembly"))
                if moved:
                    proof.update(_prove(layout.capture(config), sorted(moved)))
                    moved.clear()
                snapshot = layout.capture(config)
        with effort.stage("repo.setup.files"):
            wanted = files(snapshot)
            writes = {**_changed(snapshot, wanted), **_stale(snapshot, wanted)}
            if writes:
                apply(_plan(snapshot, writes, "setup", "setup: repository files"))
            local = project.root / "local.mk"
            store.write(local, build.local_mk(config))
        with effort.stage("repo.setup.make"):
            make = build.make_check(config, layout.capture(config))
        snapshot = layout.capture(config)
        type_map, conflicts = types.load(snapshot), types.conflicts(snapshot)
        result = {
            "toolchains": toolchains,
            "extracted": sorted(extracted),
            "proof": proof,
            "groups": len(snapshot.layout.groups),
            "segments": dict(Counter(g.segment for g in snapshot.layout.groups.values())),
            "types": {
                plural: len(type_map[key])
                for plural, key in (("functions", "function"), ("globals", "global"), ("structs", "struct"))
            },
            "boundary": boundary,
            "joined": joined,
            "units": _built(snapshot),
            "debt": [*(f.reason for f in mapped.debt), *(f"{f.reason}: " + "; ".join(f.missing) for f in conflicts)],
            "messages": messages,
            "files": sorted(written),
            "make": make,
        }
        store.write(done, json.dumps({"state": _state(config), "result": result}).encode())
        return result
