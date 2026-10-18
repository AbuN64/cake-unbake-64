"""ROM validation, repository bootstrap, sync and the one-pass setup."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path
from string import Template
from typing import Any

import rabbitizer
import yaml

from unbake import adapters, api, build, ci, effort, journal, pool, process, report, store, tree
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, NativeResult, Refusal, digest, sha1


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


def _checked(result: NativeResult, key: str, name: str) -> None:
    if result.exit != 0 or result.signal is not None:
        tail = result.stderr.decode(errors="replace").splitlines()
        raise Refusal(Finding(key, f"{name}: {tail[-1] if tail else f'exit {result.exit}, signal {result.signal}'}"))


HEADER = "include/include_asm.h"


_CODE_SEGMENTS = frozenset({"code", "c", "asm"})  # splat segment types that hold code


def _segment_rows(document: Json) -> list[Json]:
    """The segment rows splat create_config detected, as C1 rows: list rows [start, type, name] and dict rows alike."""
    rows: list[Json] = []
    for item in document.get("segments") or ():
        if isinstance(item, dict):
            start, kind, name, vram = item.get("start"), item.get("type"), item.get("name"), item.get("vram")
        elif len(item) >= 2:
            start, kind, name, vram = item[0], item[1], item[2] if len(item) > 2 else None, None
        else:
            continue  # the closing [end] row
        if start is None or name is None:
            continue
        rows.append({"name": str(name), "rom": int(start), "vram": vram if kind in _CODE_SEGMENTS else None})
    return rows


def _toml_rows(rows: Iterable[Json]) -> str:
    out = []
    for row in rows:
        cells = [f"name = {json.dumps(row['name'])}", f"rom = 0x{row['rom']:X}"]
        cells.append(f"vram = 0x{int(row['vram']):08X}" if row["vram"] is not None else 'kind = "bin"')
        out.append("  { " + ", ".join(cells) + " },")
    return "segments = [\n" + "\n".join(out) + "\n]\n"


def init(params: Json) -> Json:
    with effort.stage("repo.init"):
        root, roms = Path(params["dir"]).absolute(), {}
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
        host_path = Path(params["host"])
        host = configuration.load_host(host_path)
        versions, rules = validate_roms(roms, params["names_from"]), configuration.load_resource("repo.toml")
        created, blocks = [], []

        def write(path: str, value: bytes) -> None:
            store.write(root / path, value)
            created.append(path)

        write(".gitignore", ("\n".join(rules["ignore"]) + "\n").encode())
        write(HEADER, configuration.template("include_asm.h.in").encode())
        for v, source in sorted(roms.items()):
            path = f"roms/baserom.{v}.z64"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, root / path)
            created.append(path)
        for v in sorted(roms):
            rom = f"roms/baserom.{v}.z64"
            before = set(root.glob("*.yaml"))
            result = process.run(
                "splat create_config", [str(host.tools["splat"]), "create_config", rom], root, tmp=process.scratch(root)
            )
            _checked(result, "setup.splat", "splat create_config")
            made = set(root.glob("*.yaml")) - before
            if len(made) != 1:
                raise Refusal(Finding("setup.splat", f"expected one new yaml, found {len(made)}", versions=(v,)))
            detected = made.pop()
            rows = _segment_rows(yaml.safe_load(detected.read_bytes()))
            detected.unlink()  # splat's scratch output: the generated versions/<v>/splat.yaml replaces it
            write(f"versions/{v}/symbol_addrs.txt", b"")
            blocks.append(f'\n[version.{v}]\nsha1 = "{versions[v]["sha1"]}"\n{_toml_rows(rows)}')
        values = {key: json.dumps(params[key]) for key in ("name", "title", "names_from", "toolchain")}
        write("config.toml", _render("config.toml", **values, version_blocks="".join(blocks)))
        write(
            "README.md",
            _render("README.md.in", title=params["title"], begin=rules["readme"]["begin"], end=rules["readme"]["end"]),
        )
        roms_list = "\n".join(f"- `roms/baserom.{v}.z64`" for v in sorted(roms))
        write("CONTRIBUTING.md", _render("CONTRIBUTING.md.in", title=params["title"], roms=roms_list))
        _checked(
            process.run("git init repository", [str(host.tools["git"]), "init"], root, tmp=process.scratch(root)),
            "native.exit",
            "git init",
        )
        synced = sync(configuration.load(root, host_path), {"check": False})
        return {"dir": str(root), "created": sorted({*created, *synced["written"]}), "versions": versions}


def _macro(version: str) -> str:
    return "VERSION_" + version.upper().replace("-", "_")


def _segment_of(config: Config, version: str, rom: int) -> str | None:
    rows = sorted(config.project.version[version].segments, key=lambda s: s.rom)
    return next(
        (
            s.name
            for s, after in zip(rows, [*rows[1:], None], strict=True)
            if rom >= s.rom and (after is None or rom < after.rom)
        ),
        None,
    )


def _extension(modules: Iterable[Any]) -> str:
    """The extension most source files of the tree use: where a new stand-in file is written."""
    found = Counter(Path(m.path).suffix for m in modules)
    return found.most_common(1)[0][0] if found else ".c"


def _stand_in(kind: str, segment: str, name: str) -> str:
    return f'{"INCLUDE_ASM" if kind == "func" else "INCLUDE_DATA"}("{segment}", {name});'


def stand_ins(config: Config, current: Any) -> dict[str, bytes]:
    """The files to write so that every name of tree.unowned(v) has a stand-in line: appended to the file owning the
    previous address of its segment, or a new src/<segment>/<ADDR>.<ext> when nothing precedes it."""
    root, ext = config.project.root, _extension(current.modules)
    texts: dict[str, str] = {}
    appended: dict[str, list[str]] = {}
    placed: set[str] = set()

    def text(path: str) -> str:
        if path not in texts:
            file = root / path
            texts[path] = file.read_text() if file.is_file() else ""
        return texts[path]

    for v in config.project.versions:
        unowned = current.unowned(v)
        if not unowned:
            continue
        facts = api.symbols(config, v)
        owners: dict[str, list[tuple[int, str]]] = {}
        for module in current.modules:
            for owned in module.owns.get(v, ()):
                if owned in facts and facts[owned][1] is not None:
                    segment = _segment_of(config, v, facts[owned][1] or 0)
                    if segment is not None:
                        owners.setdefault(segment, []).append((facts[owned][1] or 0, module.path))
        for rows in owners.values():
            rows.sort()
        for name in sorted(
            (n for n in unowned if facts.get(n, (0, None))[1] is not None), key=lambda n: facts[n][1] or 0
        ):
            if name in placed:
                continue
            vram, rom, _size, kind = facts[name]
            rom = rom or 0
            segment = _segment_of(config, v, rom)
            if segment is None:
                raise Refusal(Finding("sync.segment", f"{v}: {name} at rom 0x{rom:X} is in no segment", versions=(v,)))
            rows = owners.setdefault(segment, [])
            elsewhere = next((current.owner(o, name) for o in config.project.versions if current.owner(o, name)), None)
            before = [path for at, path in rows if at < rom]
            path = elsewhere or (before[-1] if before else f"src/{segment}/{vram:08X}{ext}")
            line = _stand_in(kind, segment, name)
            if elsewhere:
                line = f"#if {_macro(v)}\n{line}\n#endif"
            else:
                placed.add(name)
            appended.setdefault(path, []).append(line)
            rows.append((rom, path))
            rows.sort()
    writes: dict[str, bytes] = {}
    for path, lines in appended.items():
        body = text(path)
        if "include_asm.h" not in body:
            body = '#include "include_asm.h"\n\n' + body
        writes[path] = (body.rstrip("\n") + "\n" + "\n".join(lines) + "\n").encode()
    return writes


def _generated(config: Config, current: Any) -> dict[str, bytes]:
    project = config.project
    wanted = {
        "Makefile": build.makefile(config),
        f"{project.name}.ld": build.linker(config, current),
        **{f"versions/{v}/splat.yaml": build.splat_yaml(config, v, current) for v in project.versions},
        **ci.files(),
    }
    if not (project.root / HEADER).is_file():
        wanted[HEADER] = configuration.template("include_asm.h.in").encode()
    return wanted


def sync(config: Config, params: Json) -> Json:
    """Write every generated file and the stand-ins for unowned names; with check, write nothing and refuse naming the
    files that would change."""
    return _sync(config, params)[0]


def _sync(config: Config, params: Json) -> tuple[Json, Any]:
    """`sync`, and the tree it left (the one capture a setup shares with the report that follows)."""
    with effort.stage("repo.sync"):
        root = config.project.root
        current = tree.capture(config)
        appended = stand_ins(config, current)
        if appended:
            current = tree.capture(config, appended)
        changed = {
            path: value
            for path, value in {**appended, **_generated(config, current)}.items()
            if not (root / path).is_file() or (root / path).read_bytes() != value
        }
        if params.get("check"):
            if changed:
                raise Refusal(
                    Finding(
                        "sync.changed",
                        f"{len(changed)} generated or stand-in files differ from `unbake sync`",
                        missing=tuple(sorted(changed)),
                        action="run unbake sync and commit the result",
                    )
                )
            return {"written": [], "stand_ins": {}}, current
        for path, value in changed.items():
            store.write(root / path, value)
        effort.count("sync", not changed)
        return {"written": sorted(changed), "stand_ins": {p: [] for p in sorted(appended)}}, current


def symbols_command(config: Config, params: Json) -> Json:
    wanted = [params["version"]] if params.get("version") else list(config.project.versions)
    return {"symbols": {v: {n: list(row) for n, row in api.symbols(config, v).items()} for v in wanted}}


def extract_job(job: tuple[Config, str]) -> bool:
    """One splat run unless the extract cache holds this input; true when splat ran."""
    config, v = job
    key, ran = _extract_key(config, v), []

    def produce() -> bytes:
        ran.append(v)
        _checked(build.extract(config, v), "setup.splat", f"extract {v}")
        return key.encode()

    if (config.project.root / "asm" / v).is_dir():
        store.cached(config, "extract", key, produce)
    else:
        produce()  # no tree to trust: a stamp alone is not the extraction
        store.cached(config, "extract", key, key.encode)
    return bool(ran)


@cache
def _splat_version(tool: Path, root: Path) -> bytes:
    result = process.run("splat version", [str(tool), "--version"], tool.parent, tmp=process.scratch(root))
    if result.exit:
        raise Refusal(Finding("setup.splat", result.stderr.decode(errors="replace")))
    return result.stdout


def _extract_key(config: Config, v: str) -> str:
    """splat version, yaml, symbol_addrs and ROM: nothing else decides what extraction writes."""
    root = config.project.root
    return digest(
        (
            _splat_version(config.host.tools["splat"], root),
            sha1((root / f"versions/{v}/splat.yaml").read_bytes()),
            sha1((root / f"versions/{v}/symbol_addrs.txt").read_bytes()),
            sha1((root / f"roms/baserom.{v}.z64").read_bytes()),
        )
    )


def _extractable(config: Config) -> list[str]:
    root = config.project.root
    return [v for v in config.project.versions if (root / f"versions/{v}/splat.yaml").is_file()]


def _state(config: Config) -> str:
    """Everything setup reads or writes: the tool, the commit, the working tree and the extraction inputs. Unchanged
    means done."""
    root = config.project.root
    head = process.git(config, "rev-parse", "HEAD").stdout.decode().strip()
    dirty = process.git(config, "status", "--porcelain", "--untracked-files=all").stdout
    tool = digest(
        [p.read_bytes() for p in sorted(Path(__file__).parent.rglob("*")) if p.is_file() and p.suffix != ".pyc"]
    )  # the tool's content, not when it was touched
    facts = [sha1((root / f"versions/{v}/symbol_addrs.txt").read_bytes()) for v in config.project.versions]
    stamps = [_extract_key(config, v) for v in _extractable(config)]
    return digest((head, dirty, config.digest, tool, facts, stamps))


def _extract(config: Config, versions: Sequence[str]) -> list[str]:
    ran = pool.map(config, "repo.extract", extract_job, [(config, v) for v in versions])
    for did in ran:
        effort.count("extract", not did)
    return [v for v, did in zip(versions, ran, strict=True) if did]


def setup(config: Config, params: Json) -> Json:
    wait_s = config.host.move_wait_s
    with effort.stage("repo.setup"), store.exclusive(config, "structure", wait=True, wait_s=wait_s) as held:
        adapters.check_abucache()
        if not held:
            raise Refusal(Finding("setup.busy", configuration.sentence("setup.busy")))
        journal.recover(config)
        project = config.project
        done = project.root / ".unbake/setup.json"
        kept = json.loads(done.read_bytes()) if done.exists() else None
        if kept and kept["state"] == _state(config):
            effort.count("setup", True)
            return {**kept["result"], "extracted": [], "files": []}
        effort.count("setup", False)
        with effort.stage("repo.setup.toolchains"):
            toolchains = sorted(
                {project.toolchain, *(row.toolchain for row in project.files.values() if row.toolchain)}
            )
            build.install_toolchains(config, toolchains)
        with effort.stage("repo.setup.extract"):
            extracted = _extract(config, _extractable(config))
        synced, current = _sync(config, {"check": False})
        written = synced["written"]
        again = [v for v in project.versions if f"versions/{v}/splat.yaml" in written]
        if again:  # the yaml changed under extraction: its output is the output of the old one
            extracted = sorted({*extracted, *_extract(config, again)})
        store.write(project.root / "local.mk", build.local_mk(config))
        with effort.stage("repo.setup.make"):
            make = build.make_check(config, current)
        report.progress(config, current)
        result = {"toolchains": toolchains, "extracted": sorted(extracted), "files": written, "make": make}
        store.write(done, json.dumps({"state": _state(config), "result": result}).encode())
        return result
