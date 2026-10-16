"""Repo tests use synthetic bytes and mocked native/pool/journal operations only."""

from __future__ import annotations

import json
import tomllib
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from unittest.mock import Mock

import fixture
import pytest
import yaml

from unbake import adapters, recipes, repo
from unbake import config as configuration
from unbake.contracts import (
    Config,
    Finding,
    Group,
    LayoutMap,
    Member,
    Placement,
    Plan,
    Project,
    Recipe,
    Refusal,
    Snapshot,
    UnitSpec,
    VersionFiles,
    digest,
    sha1,
)


@pytest.fixture(autouse=True)
def _compiler_row(monkeypatch):
    row = {"never_emits": ["mfc0", "mtc0", "cache", "sync", "eret"], "fpu_control_with": ["trunc.w.s"]}
    monkeypatch.setattr(adapters, "row", lambda id: row)
    monkeypatch.setattr(repo.libasm, "plan", lambda snapshot: ({}, ()))


@pytest.fixture(autouse=True)
def isolate_native(monkeypatch):
    # Any missed mock must fail rather than launch a process or a worker.
    monkeypatch.setattr(repo.process, "run", Mock(side_effect=AssertionError("unmocked process")))
    monkeypatch.setattr(repo.pool, "map", Mock(side_effect=AssertionError("unmocked pool")))
    monkeypatch.setattr(repo.effort, "stage", lambda name: nullcontext())


@pytest.fixture
def rom_pair(tmp_path):
    paths = {v: tmp_path / f"{v}.z64" for v in ("a", "b")}
    for v, region in (("a", b"E"), ("b", b"J")):
        paths[v].write_bytes(fixture.rom_bytes(region=region))
    return paths


def test_validate_roms_ok(rom_pair):
    result = repo.validate_roms(dict(reversed(list(rom_pair.items()))), "a")
    assert list(result) == ["a", "b"]
    for v, path in rom_pair.items():
        assert result[v]["sha1"] == sha1(path.read_bytes())
        assert result[v]["similarity"] == 1.0
        assert result[v]["fields"] == {
            "game_code": "NXX",
            "region": "E" if v == "a" else "J",
            "revision": "00",
            "entrypoint": "00000000",
            "title": "FIXTURE",
        }


@pytest.mark.parametrize(
    "order,name", [("v64", "v64 (byte-swapped)"), ("n64", "n64 (little-endian words)"), ("unknown", "an unknown order")]
)
def test_v64_order_refused_by_name(rom_pair, order, name):
    blob = fixture.rom_bytes(order=order) if order != "unknown" else b"bad!"
    rom_pair["b"].write_bytes(blob)
    with pytest.raises(Refusal) as caught:
        repo.validate_roms(rom_pair, "a")
    (finding,) = caught.value.findings
    assert finding.key == "init.rom_order"
    assert finding.path == str(rom_pair["b"])
    assert finding.reason == (
        f"b: first word 0x{int.from_bytes(blob[:4], 'big'):08X} is {name}; convert the dump to big-endian .z64"
    )


def test_duplicate_refused(rom_pair):
    rom_pair["b"].write_bytes(rom_pair["a"].read_bytes())
    with pytest.raises(Refusal) as caught:
        repo.validate_roms(rom_pair, "a")
    (finding,) = caught.value.findings
    assert finding.key == "init.rom_duplicate"
    assert finding.versions == ("a", "b")
    assert "a and b" in finding.reason


def test_game_code_mismatch_refused(rom_pair):
    rom_pair["b"].write_bytes(fixture.rom_bytes(game_code=b"NYY"))
    with pytest.raises(Refusal) as caught:
        repo.validate_roms(rom_pair, "a")
    (finding,) = caught.value.findings
    assert finding.key == "init.rom_mismatch"
    assert finding.reason == "b: game_code 'NYY' differs from a 'NXX'"


def test_low_similarity_refused_with_value(rom_pair):
    rom_pair["b"].write_bytes(fixture.rom_bytes(body=b"\xff" * 0x100000))
    with pytest.raises(Refusal) as caught:
        repo.validate_roms(rom_pair, "a")
    (finding,) = caught.value.findings
    assert finding.key == "init.rom_mismatch"
    assert "similarity 0.000000" in finding.reason
    assert "below 0.5" in finding.reason


def test_validation_aggregates_same_kind(rom_pair, tmp_path):
    rom_pair["b"].write_bytes(fixture.rom_bytes(order="v64"))
    rom_pair["c"] = tmp_path / "c.z64"
    rom_pair["c"].write_bytes(fixture.rom_bytes(order="n64"))
    with pytest.raises(Refusal) as caught:
        repo.validate_roms(rom_pair, "a")
    assert [f.key for f in caught.value.findings] == ["init.rom_order"] * 2
    rom_pair["b"].write_bytes(fixture.rom_bytes(game_code=b"NYY"))
    rom_pair["c"].write_bytes(fixture.rom_bytes(game_code=b"NZZ"))
    with pytest.raises(Refusal) as caught:
        repo.validate_roms(rom_pair, "a")
    assert [f.key for f in caught.value.findings] == ["init.rom_mismatch"] * 2


@pytest.mark.parametrize("opcode", [0x0F, 0x09, 0x0D, 0x23, 0x2B, 0x21, 0x29, 0x20, 0x28, 0x31, 0x39, 0x03])
def test_similarity_masks_instruction_addresses(rom_pair, opcode):
    # Fill the full similarity range: no common zero windows can conceal a bad mask.
    for v, immediate in (("a", 0x1234), ("b", 0x4321)):
        word = (opcode << 26) | (2 << 21) | (3 << 16) | immediate
        rom_pair[v].write_bytes(fixture.rom_bytes(body=word.to_bytes(4, "big") * 0x40000))
    assert repo.validate_roms(rom_pair, "a")["b"]["similarity"] == 1.0


def test_similarity_uses_unique_windows(rom_pair):
    zero, other = b"\0" * 32, (0x03E00008).to_bytes(4, "big") * 8
    rom_pair["a"].write_bytes(fixture.rom_bytes(body=other + zero * (0x8000 - 1)))
    rom_pair["b"].write_bytes(fixture.rom_bytes(region=b"J", body=zero * 0x8000))
    assert repo.validate_roms(rom_pair, "a")["b"]["similarity"] == 0.5


@pytest.fixture
def init_case(tmp_path, rom_pair, host_file, toolchains, monkeypatch):
    root = tmp_path / "new"
    params = {
        "dir": str(root),
        "rom": [f"{v}={p}" for v, p in rom_pair.items()],
        "names_from": "a",
        "toolchain": "gcc-test",
        "name": "demo",
        "title": "Demo",
        "host": str(host_file),
    }
    calls = []

    def run(name, argv, cwd, **kwargs):
        assert cwd == root
        assert (cwd / ".gitignore").read_text().splitlines() == configuration.load_resource("repo.toml")["ignore"]
        calls.append(tuple(argv[1:]))
        if argv[1] == "create_config":
            (cwd / "generated.yaml").write_text("options:\n  basename: stale\n  custom: keep\nsegments: []\n")
        return fixture.native_result(argv)

    monkeypatch.setattr(repo.process, "run", run)
    return params, root, calls


def test_init_never_stages_or_commits(init_case):
    params, root, calls = init_case
    repo.init(params)
    assert {"/roms/", "/asm/"} <= set((root / ".gitignore").read_text().splitlines())  # splat writes asm/ untracked
    assert calls[-1] == ("init",)  # unbake writes no Git state beyond creating the repository
    assert not any(c[0] in ("add", "commit") for c in calls if c)


def test_init_yaml_options_rewritten(init_case):
    params, root, calls = init_case
    repo.init(params)
    rules = configuration.load_resource("repo.toml")
    for v in ("a", "b"):
        document = yaml.safe_load((root / f"versions/{v}/demo.yaml").read_text())
        assert document == {
            "options": {**repo._options(rules, v, "demo", f"roms/baserom.{v}.z64"), "custom": "keep"},
            "segments": [],
        }
        assert (root / f"versions/{v}/symbol_addrs.txt").read_bytes() == b""
    assert not list(root.glob("*.yaml"))
    assert calls[:2] == [("create_config", "roms/baserom.a.z64"), ("create_config", "roms/baserom.b.z64")]


def test_init_result_validates(init_case, rom_pair):
    params, root, _ = init_case
    result = repo.init(params)
    configuration.validate("result.init", json.loads(json.dumps(result)), "test")
    assert result["dir"] == str(root)
    assert result["created"] == sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    project = tomllib.loads((root / "config.toml").read_text())
    assert project["resident"] == {}
    assert project["project"]["versions"] == ["a", "b"]
    for v in ("a", "b"):
        assert (root / f"roms/baserom.{v}.z64").read_bytes() == rom_pair[v].read_bytes()
        assert project["version"][v] == {
            "baserom": f"roms/baserom.{v}.z64",
            "baserom_sha1": result["versions"][v]["sha1"],
            "split": f"versions/{v}/demo.yaml",
            "symbols": f"versions/{v}/symbol_addrs.txt",
            "macros": [],
        }
    assert tomllib.loads((root / "layout.toml").read_text()) == {"schema": 5, "group": [], "unit": []}
    assert tomllib.loads((root / "types.toml").read_text()) == {"schema": 1, "function": {}, "global": {}, "struct": {}}
    rules = configuration.load_resource("repo.toml")
    assert rules["readme"]["begin"] in (root / "README.md").read_text()
    assert rules["readme"]["end"] in (root / "README.md").read_text()
    assert "- `roms/baserom.b.z64`" in (root / "CONTRIBUTING.md").read_text()
    for row in rules["sdk"]:
        assert (root / row["path"]).read_text() == configuration.template(row["template"])


@pytest.mark.parametrize("bad", ["nonempty", "file", "malformed", "duplicate", "missing", "names", "toolchain"])
def test_init_bad_request(init_case, bad):
    params, root, calls = init_case
    if bad == "nonempty":
        root.mkdir()
        (root / "keep").write_text("keep")
    elif bad == "file":
        root.write_text("keep")
    elif bad == "malformed":
        params["rom"] = ["a"]
    elif bad == "duplicate":
        params["rom"] *= 2
    elif bad == "missing":
        params["rom"] = [f"a={root / 'missing'}"]
    elif bad == "names":
        params["names_from"] = "c"
    else:
        params["toolchain"] = "absent"
    with pytest.raises(Refusal) as caught:
        repo.init(params)
    assert caught.value.findings[0].key == "init.request"
    assert calls == []


@pytest.mark.parametrize("count", [0, 2])
def test_init_requires_one_new_yaml(init_case, monkeypatch, count):
    params, _root, _ = init_case

    def run(name, argv, cwd, **kw):
        for i in range(count):
            (cwd / f"new{i}.yaml").write_text("options: {}")
        return fixture.native_result(argv)

    monkeypatch.setattr(repo.process, "run", run)
    with pytest.raises(Refusal) as caught:
        repo.init(params)
    assert caught.value.findings[0].key == "setup.splat"
    assert f"found {count}" in caught.value.findings[0].reason


@pytest.fixture
def snapshot(tmp_path, host_file):
    root = tmp_path / "project"
    root.mkdir()
    vf = {
        v: VersionFiles(f"roms/{v}.z64", "0" * 40, f"versions/{v}/demo.yaml", f"versions/{v}/symbol_addrs.txt", {})
        for v in ("a", "b")
    }
    project = Project(root, "id", "demo", "Demo", ("a", "b"), "a", "gcc-test", {}, vf, {"a": (), "b": ()}, {}, {}, "p")
    cfg = Config(project, configuration.load_host(host_file), "cfg")
    placements = tuple(Placement(v, ".text", 0x1000, 0x1008, 0x80000400) for v in ("a", "b"))
    members = {"entry": Member("entry", "function", "asm", "code", placements)}
    group = Group("code", "main", ("entry",), "authored", (), False, "src/code.c", "include/main/code.h")
    mapping = LayoutMap({"code": group}, members, {}, "layout", (), {})
    for v, files in vf.items():
        for path, data in ((files.split, b"segments: []\n"), (files.baserom, b"synthetic " + v.encode())):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return Snapshot(cfg, "0" * 40, mapping, {}, {}, "snapshot")


@pytest.mark.parametrize("existing", [False, True])
def test_files_contents(snapshot, monkeypatch, existing):
    rules = configuration.load_resource("repo.toml")
    preserved = ["CONTRIBUTING.md", rules["sdk"][0]["path"]] if existing else []
    snapshot = replace(snapshot, overlays={p: b"authored" for p in preserved})
    monkeypatch.setattr(repo.build, "makefile", lambda s: b"makefile")
    monkeypatch.setattr(repo.build, "symbols_ld", lambda s, v: v.encode())
    monkeypatch.setattr(repo.report, "files", lambda s: {"README.md": b"report"})
    result = repo.files(snapshot)
    assert all(isinstance(value, bytes) for value in result.values())
    assert result["Makefile"] == b"makefile"
    assert result["README.md"] == b"report"
    for v in ("a", "b"):
        assert result[f"versions/{v}/symbols.ld"] == v.encode()
    for path, value in repo.ci.files(("a", "b")).items():  # the CI files: one entry point and its thin callers
        assert result[path] == value
    assert all(p in result["generated.txt"].decode().splitlines() for p in repo.ci.files(("a", "b")))
    assert json.loads(result[".recomp.json"]) == {
        "$schema": "https://recomp.fyi/schema/v1.json",
        "game": snapshot.config.project.title,
        "system": "N64",
        "type": "decomp",
    }
    for path in preserved:
        assert path not in result
        assert snapshot.read(path) == b"authored"
    for row in rules["sdk"]:
        if row["path"] not in preserved:
            assert result[row["path"]] == configuration.template(row["template"]).encode()
    if not existing:
        assert "- `roms/baserom.a.z64`" in result["CONTRIBUTING.md"].decode()


@pytest.fixture
def setup_case(snapshot, monkeypatch, warm):
    stages, events, plans = [], [], []
    real_map = lambda c, n, f, jobs, key=None: repo.pool.gather(c, [(n, f, jobs)])[0]  # noqa: E731  (keys: own test)
    monkeypatch.setattr(repo, "_splat_version", lambda tool, root: b"splat test-version")
    root, cfg = snapshot.config.project.root, snapshot.config
    for vf in cfg.project.version_files.values():
        symbols = root / vf.symbols
        symbols.parent.mkdir(parents=True, exist_ok=True)
        symbols.write_text("")
    (root / "symbols.toml").write_bytes(repo.symbols.empty())
    for path, value in repo._extras(
        configuration.load_resource("repo.toml"), cfg.project.title, cfg.project.versions
    ).items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    for v, vf in cfg.project.version_files.items():
        (root / vf.split).parent.mkdir(parents=True, exist_ok=True)
        (root / vf.split).write_text(
            yaml.safe_dump(
                {"options": repo._options(configuration.load_resource("repo.toml"), v, cfg.project.name, vf.baserom)}
            )
        )
    boundary = {k: {"proposed": 0, "applied": 0, "withheld": 0} for k in ("prelude", "split", "merge")}

    @contextmanager
    def stage(name):
        if not name.startswith(("pool.", "job")):  # the warm pool's own stages are not setup's
            stages.append(name)
        yield

    def plan(writes, operation="layout", message="map"):
        return Plan(operation, snapshot.digest, writes, (), (), (), message, digest((snapshot.digest, writes, message)))

    def apply(config, proposed, current):
        assert config == cfg and current.digest == snapshot.digest
        assert proposed.digest == digest((snapshot.digest, proposed.writes, proposed.message))
        plans.append(proposed)
        events.append(proposed.message)
        for path, data in proposed.writes.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return tuple(proposed.writes)

    def pool_map(config, name, function, jobs, key=None):  # every dispatch on the warm workers, as setup runs them
        assert config == cfg
        events.append((name, list(jobs)))
        return real_map(config, name, function, jobs, key)

    def prove(s, unit, version, recipe, work):
        assert s == snapshot and work == root / ".unbake/work"
        assert recipe.digest == "recipe"
        return (fixture.proof(unit.path, unit.members[0], version, True),)

    monkeypatch.setattr(repo.effort, "stage", stage)
    monkeypatch.setattr(repo.store, "exclusive", lambda config, name, wait, wait_s=None: nullcontext(True))
    monkeypatch.setattr(repo.store, "work", lambda config: nullcontext(root / ".unbake/work"))
    monkeypatch.setattr(repo.journal, "recover", lambda config: events.append("recover"))
    monkeypatch.setattr(repo.journal, "apply", apply)
    monkeypatch.setattr(repo.layout, "capture", lambda config: snapshot)
    monkeypatch.setattr(repo.layout, "overlay", lambda current, writes: current)
    monkeypatch.setattr(
        repo.layout, "asm_unit", lambda s, m, v: UnitSpec(f"asm/{v}/{m}.s", "asm", "code", (m,), "gcc-test", {})
    )
    monkeypatch.setattr(recipes, "resolve", lambda c, u, o: Recipe("gcc-test", (), (), (), "recipe"))
    monkeypatch.setattr(repo.native, "prove", prove)
    monkeypatch.setattr(repo.pool, "map", pool_map)
    monkeypatch.setattr(repo.build, "install_toolchains", lambda c, ids: events.extend(("toolchain", i) for i in ids))
    monkeypatch.setattr(repo.build, "extract", lambda c, v: fixture.native_result())
    planned = []  # the first plan writes the layout, the plan on what it wrote writes nothing

    def infer_plan(s, **kwargs):
        planned.append(s)
        return plan({} if len(planned) > 1 else {"layout.toml": b"inferred"}, message="infer")

    monkeypatch.setattr(repo.infer, "plan", infer_plan)
    monkeypatch.setattr(repo.infer, "correspondences", lambda s: ())
    monkeypatch.setattr(repo.infer, "addresses", lambda s, debt: ({}, []))
    monkeypatch.setattr(repo.types, "scan", lambda s: b"type map")
    monkeypatch.setattr(repo.types, "load", lambda s: {"function": {"entry": {}}, "global": {}, "struct": {}})
    monkeypatch.setattr(repo.types, "conflicts", lambda s: [])
    monkeypatch.setattr(repo.layout, "boundary_plan", lambda s: (plan({}), boundary))
    monkeypatch.setattr(repo, "files", lambda s: {"Makefile": b"generated"})
    monkeypatch.setattr(repo.build, "local_mk", lambda c: b"local")
    states = iter(range(10**6))
    monkeypatch.setattr(repo, "_state", lambda c: str(next(states)))  # every setup finds something changed
    monkeypatch.setattr(repo.build, "make_check", lambda c, s: {"exit": 0, "versions": ["a", "b"]})
    return snapshot, stages, events, plans, plan, boundary


def test_setup_busy_when_land_held(setup_case, monkeypatch):
    snapshot, stages, events, *_ = setup_case
    lock = Mock(return_value=nullcontext(False))
    monkeypatch.setattr(repo.store, "exclusive", lock)
    with pytest.raises(Refusal) as caught:
        repo.setup(snapshot.config, {})
    lock.assert_called_once_with(snapshot.config, "structure", wait=True, wait_s=snapshot.config.host.move_wait_s)
    assert caught.value.findings[0].key == "setup.busy"
    assert events == [] and stages == ["repo.setup"]


@pytest.mark.parametrize("options", [{}, {"create_c_files": True}])
def test_setup_refuses_a_split_yaml_that_would_let_splat_write_c_files(setup_case, options):
    snapshot, _, events, *_ = setup_case
    root, vf = snapshot.config.project.root, snapshot.config.project.version_files["a"]
    document = yaml.safe_load((root / vf.split).read_text())
    document["options"].pop("create_c_files")
    (root / vf.split).write_text(yaml.safe_dump({"options": {**document["options"], **options}}))
    with pytest.raises(Refusal) as caught:
        repo.setup(snapshot.config, {})
    finding = caught.value.findings[0]
    assert finding.key == "setup.options" and finding.missing == (f"{vf.split}: create_c_files",)
    assert ("toolchain", 0) not in events and not [e for e in events if isinstance(e, tuple)]


def test_setup_stage_order(setup_case, warm):
    snapshot, stages, events, plans, *_ = setup_case
    repo.setup(snapshot.config, {})
    assert warm.forks == 1 and warm.dispatches >= 2  # every stage's dispatch on one warm pool
    assert stages == [
        "repo.setup",
        *[f"repo.setup.{s}" for s in ("toolchains", "extract", "prove", "map", "boundary", "library", "files", "make")],
    ]
    assert events[0] == "recover"
    assert [p.message for p in plans] == ["infer", "setup: type map", "setup: repository files"]
    assert [p.operation for p in plans] == ["layout", "layout", "setup"]


def test_setup_joins_correspondences_and_extracts_the_changed_versions_again(setup_case, monkeypatch):
    snapshot, _stages, events, plans, *_ = setup_case
    root, files = snapshot.config.project.root, snapshot.config.project.version_files
    table = {"entry": {"kind": "function", "a": 0x80000400}, "alias": {"kind": "function", "b": 0x80000410}}
    (root / "symbols.toml").write_bytes(repo.symbols.dump(table))
    for v in files:
        (root / files[v].symbols).write_bytes(repo.symbols.render(table, v))
    monkeypatch.setattr(repo.infer, "correspondences", lambda s: (("entry", "a", "alias", "b"),))
    monkeypatch.setattr(repo.symbols, "referenced", lambda s, t: set())
    result = repo.setup(snapshot.config, {})
    (joined,) = [p for p in plans if p.message.endswith("1 symbols read from the ROM")]
    assert repo.symbols.parse(joined.writes["symbols.toml"], ("a", "b")) == {
        "entry": {"kind": "function", "a": 0x80000400, "b": 0x80000410}
    }
    assert set(joined.writes) == {"layout.toml", "symbols.toml", files["b"].symbols}  # one commit; only b gained a row
    assert result["joined"] == 1
    extracts = [[v for _, v in e[1]] for e in events if isinstance(e, tuple) and e[0] == "repo.extract"]
    assert extracts == [["a", "b"], ["b"]]


def test_setup_apply_extracts_before_overlay_on_split_write(setup_case, monkeypatch):
    snapshot, _stages, events, _plans, *_ = setup_case
    root, files = snapshot.config.project.root, snapshot.config.project.version_files
    table = {"entry": {"kind": "function", "a": 0x80000400}, "alias": {"kind": "function", "b": 0x80000410}}
    (root / "symbols.toml").write_bytes(repo.symbols.dump(table))
    for v in files:
        (root / files[v].symbols).write_bytes(repo.symbols.render(table, v))
    monkeypatch.setattr(repo.infer, "correspondences", lambda s: (("entry", "a", "alias", "b"),))
    monkeypatch.setattr(repo.symbols, "referenced", lambda s, t: set())
    overlaid = []
    monkeypatch.setattr(repo.layout, "overlay", lambda current, writes: overlaid.append(set(writes)) or current)
    repo.setup(snapshot.config, {})
    split = {files["b"].symbols}
    # only the plan on the readings lays the split write over a snapshot; applying it extracts and captures instead
    assert len([w for w in overlaid if w & split]) == 1
    assert sum(1 for e in events if isinstance(e, tuple) and e[0] == "repo.extract") == 2


@pytest.mark.parametrize("empty", [False, True])
def test_setup_not_exact_refuses(setup_case, monkeypatch, empty):
    snapshot, stages, _events, plans, *_ = setup_case
    missing = tuple(f"missing {i}" for i in range(25))
    monkeypatch.setattr(
        repo.native,
        "prove",
        lambda s, u, v, r, w: () if empty else (fixture.proof(u.path, "entry", v, False, missing),),
    )
    with pytest.raises(Refusal) as caught:
        repo.setup(snapshot.config, {})
    (finding,) = caught.value.findings
    assert finding.key == "setup.not_exact"
    assert finding.reason == "2 extracted members do not rebuild their ROM bytes"
    assert finding.versions == ("a", "b")
    assert finding.missing == (("a: entry: no proof", "b: entry: no proof") if empty else missing[:20])
    assert stages[-1] == "repo.setup.prove" and plans == []


def test_setup_result_validates(setup_case):
    snapshot, _, _events, _plans, _, boundary = setup_case
    result = repo.setup(snapshot.config, {})
    configuration.validate("result.setup", json.loads(json.dumps(result)), "test")
    assert result == {
        "toolchains": ["gcc-test"],
        "extracted": ["a", "b"],
        "proof": {v: {"members": 1, "exact": 1, "bytes": 8} for v in ("a", "b")},
        "groups": 1,
        "segments": {"main": 1},
        "types": {"functions": 1, "globals": 0, "structs": 0},
        "boundary": boundary,
        "joined": 0,
        "units": {"a": {"built": 0, "withheld": 0}, "b": {"built": 0, "withheld": 0}},
        "debt": [],
        "messages": [p.message for p in _plans],
        "files": ["Makefile", "layout.toml", "types.toml"],
        "make": {"exit": 0, "versions": ["a", "b"]},
    }
    stamps = json.loads((snapshot.config.project.root / ".unbake/extract.json").read_bytes())
    for v in snapshot.config.project.version_files:
        assert stamps[v] == repo._stamp(snapshot.config, v)
    assert (snapshot.config.project.root / "local.mk").read_bytes() == b"local"


@pytest.mark.parametrize("changed", [None, "split", "rom", "symbols"])
def test_setup_extract_uses_content_digests(setup_case, changed):
    snapshot, _, events, *_ = setup_case
    repo.setup(snapshot.config, {})
    if changed:
        vf = snapshot.config.project.version_files["b"]
        path = vf.split if changed == "split" else "symbols.toml" if changed == "symbols" else vf.baserom
        value = (
            repo.symbols.dump({"changed": {"kind": "data", "b": 0x80000400}})
            if changed == "symbols"
            else (snapshot.config.project.root / path).read_bytes() + b"\nchanged: 1\n"
            if changed == "split"
            else b"changed"
        )
        (snapshot.config.project.root / path).write_bytes(value)
    events.clear()
    result = repo.setup(snapshot.config, {})
    assert result["extracted"] == ([] if changed is None else ["b"])
    jobs = next(e[1] for e in events if isinstance(e, tuple) and e[0] == "repo.extract")
    assert [v for _, v in jobs] == result["extracted"]


@pytest.mark.parametrize("nonconvergent", [False, True])
def test_setup_boundary_reextracts_and_reproves(setup_case, monkeypatch, nonconvergent):
    snapshot, _, events, _, plan, boundary = setup_case
    split = snapshot.config.project.version_files["b"].split
    calls = 0

    def boundary_plan(s):
        nonlocal calls
        calls += 1
        return plan({split: b"changed"} if calls == 1 or nonconvergent else {}, message="boundary"), boundary

    monkeypatch.setattr(repo.layout, "boundary_plan", boundary_plan)
    path = snapshot.config.project.root / split
    before = path.read_bytes() if path.exists() else None
    if nonconvergent:
        with pytest.raises(Refusal) as caught:
            repo.setup(snapshot.config, {})
        assert caught.value.findings[0].key == "layout.nonconvergent"
        assert (path.read_bytes() if path.exists() else None) == before  # a refused setup leaves its files as found
    else:
        repo.setup(snapshot.config, {})
    assert calls == 2
    extracts = [e[1] for e in events if isinstance(e, tuple) and e[0] == "repo.extract"]
    proofs = [e[1] for e in events if isinstance(e, tuple) and e[0] == "repo.prove"]
    assert [[v for _, v in jobs] for jobs in extracts] == [["a", "b"], ["b"]]
    assert [[v for _, _, v in jobs] for jobs in proofs] == [["a", "b"], ["b"]]


def test_setup_generates_the_symbol_files_from_the_table_before_extract(setup_case, monkeypatch):
    snapshot, _stages, _events, plans, *_ = setup_case
    root, vf = snapshot.config.project.root, snapshot.config.project.version_files["a"]
    table = {"entry": {"kind": "function", "a": 0x80000400}, "alias": {"kind": "data", "a": 0x80000400}}
    (root / "symbols.toml").write_bytes(repo.symbols.dump(table))

    def extract(cfg, v):
        assert (root / vf.symbols).read_text().count("allow_duplicated:true") == 2
        return fixture.native_result()

    monkeypatch.setattr(repo.build, "extract", extract)
    repo.setup(snapshot.config, {})
    assert plans[0].message == "setup: extraction inputs"
    assert set(plans[0].writes) == {vf.symbols}


def test_setup_installs_missing_splat_extension_before_extract(setup_case, monkeypatch):
    snapshot, _stages, _events, plans, *_ = setup_case
    root = snapshot.config.project.root
    target = root / "tools/splat_ext/resource.py"
    target.unlink()

    def extract(cfg, v):
        assert b"class N64SegResource(CommonSegBin)" in target.read_bytes()
        return fixture.native_result()

    monkeypatch.setattr(repo.build, "extract", extract)
    repo.setup(snapshot.config, {})
    assert set(plans[0].writes) == {"tools/splat_ext/resource.py"}


def test_extract_stamp_uses_real_inputs_not_adapter_source(setup_case, monkeypatch):
    snapshot, *_ = setup_case
    cfg = snapshot.config
    original = repo._stamp(cfg, "a")
    symbols = cfg.project.root / cfg.project.version_files["a"].symbols
    symbols.write_text("alias = 0x80000400;\n")
    changed = repo._stamp(cfg, "a")
    assert changed != original
    adapter = cfg.project.root / "fake-build-adapter.py"
    adapter.write_text("# changed policy")
    monkeypatch.setattr(repo.build, "__file__", str(adapter))
    assert repo._stamp(cfg, "a") == changed
    monkeypatch.setattr(repo, "_splat_version", lambda tool, root: b"changed splat-version")
    assert repo._stamp(cfg, "a") != changed


def test_prove_hands_the_pool_its_stamp_and_refuses_each_time(setup_case, monkeypatch):
    snapshot, *_ = setup_case
    runs = []
    monkeypatch.setattr(
        repo.pool,
        "map",
        lambda config, name, function, jobs, key: (
            runs.append((len(jobs), callable(key)))
            or [(fixture.proof("p", "entry", j[2], False, ("gap",)),) for j in jobs]
        ),
    )
    for _ in range(2):
        with pytest.raises(Refusal) as caught:
            repo._prove(snapshot, ["a", "b"])
        assert caught.value.findings[0].missing == ("gap", "gap")
    assert runs == [(2, True)] * 2  # the pool resolves warm jobs itself, by the proof stamp


def test_setup_drops_the_facts_of_a_stale_extraction_before_reading_them(setup_case):
    snapshot, *_ = setup_case
    context = snapshot.config.project.root / ".unbake/symbols/a/spim_context.csv"
    context.parent.mkdir(parents=True)
    context.write_text("stale")
    repo._drop_stale_facts(snapshot.config)
    assert not context.exists()  # no stamp yet: the facts belong to no extraction
    repo.setup(snapshot.config, {})
    context.write_text("fresh")
    repo._drop_stale_facts(snapshot.config)
    assert context.read_text() == "fresh"  # the stamp matches the inputs: the facts stay


def test_setup_refuses_when_landed_c_does_not_compile_and_reports_the_debt_otherwise(setup_case, monkeypatch):
    snapshot, _stages, _events, plans, plan, _ = setup_case
    broken = Finding("compile.error", "1 unit holders do not compile", missing=("src/f.c a boom",))
    monkeypatch.setattr(repo.infer, "plan", lambda s, **kwargs: replace(plan({}), blocking=(broken,)))
    with pytest.raises(Refusal) as caught:
        repo.setup(snapshot.config, {})
    assert caught.value.findings == (broken,) and plans == []
    debt = Finding("layout.ownership", "2 unit holders are withheld: label table", blocking=False)
    writes = iter([{"layout.toml": b"x"}])  # written once, then settled
    monkeypatch.setattr(repo.infer, "plan", lambda s, **kwargs: replace(plan(next(writes, {})), debt=(debt,)))
    assert repo.setup(snapshot.config, {})["debt"] == [debt.reason]


def test_setup_returns_at_once_when_nothing_it_reads_or_writes_changed(setup_case, monkeypatch):
    snapshot, _stages, events, plans, *_ = setup_case
    monkeypatch.setattr(repo, "_state", lambda c: "same")
    first = repo.setup(snapshot.config, {})
    made = len(plans)
    events.clear()
    again = repo.setup(snapshot.config, {})
    assert again == {**first, "messages": [], "extracted": [], "files": []}
    assert len(plans) == made and not any(isinstance(e, tuple) for e in events)  # no pool, no plan, no make


def test_setup_captures_the_published_files_after_make_to_warm_the_next_command(setup_case, monkeypatch):
    snapshot, _, events, *_ = setup_case
    captures = []

    def capture(config):
        captures.append(tuple(events))
        return snapshot

    def make(config, snap):
        events.append("make finished")
        return {"exit": 0}

    monkeypatch.setattr(repo.layout, "capture", capture)
    monkeypatch.setattr(repo.build, "make_check", make)
    repo.setup(snapshot.config, {})
    assert captures[-1][-1] == "make finished"
    assert "setup: repository files" in captures[-1]


def test_the_manifest_lists_every_generated_file_sorted_and_not_the_starter_files(snapshot, monkeypatch):
    monkeypatch.setattr(repo.build, "makefile", lambda s: b"makefile")
    monkeypatch.setattr(repo.build, "symbols_ld", lambda s, v: v.encode())
    monkeypatch.setattr(repo.report, "files", lambda s: {"README.md": b"report"})
    wanted = repo.files(snapshot)
    lines = wanted["generated.txt"].decode().splitlines()
    assert lines == sorted(lines) and {"generated.txt", "Makefile", "README.md", "versions/a/symbols.ld"} <= set(lines)
    assert all(path in wanted for path in lines)
    assert "CONTRIBUTING.md" in wanted and "CONTRIBUTING.md" not in lines  # written once, then the author's
    assert "memo.sh" not in lines


def _snapshot(files):
    from types import SimpleNamespace

    def read(path):
        if path not in files:
            raise FileNotFoundError(path)
        return files[path]

    return SimpleNamespace(read=read, peek=files.get)


def test_stale_files_are_the_previous_manifest_minus_what_is_generated_now():
    previous = b"Makefile\nold.mk\ngenerated.txt\ngone.mk\n"
    snapshot = _snapshot({"generated.txt": previous, "old.mk": b"x", "Makefile": b"x", "mine.c": b"x"})
    assert repo._stale(snapshot, {"Makefile": b"", "generated.txt": b""}) == {"old.mk": None}  # gone.mk is absent
    assert repo._stale(snapshot, {"Makefile": b"", "old.mk": b"", "generated.txt": b""}) == {}


def test_without_a_manifest_nothing_is_stale():
    assert repo._stale(_snapshot({"memo.sh": b"#!/bin/sh"}), {"Makefile": b""}) == {}


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside", "a/../../b"])
def test_a_manifest_naming_a_path_outside_the_project_is_refused_by_name(path):
    with pytest.raises(Refusal) as caught:
        repo._stale(_snapshot({"generated.txt": f"{path}\n".encode()}), {})
    assert "generated.txt lists" in caught.value.findings[0].reason


def test_setup_without_a_symbol_table_refuses_by_name_with_an_action(setup_case):
    snapshot, *_ = setup_case
    (snapshot.config.project.root / "symbols.toml").unlink()
    with pytest.raises(Refusal) as caught:
        repo.setup(snapshot.config, {})
    (finding,) = caught.value.findings
    assert (finding.key, finding.path) == ("config.missing", "symbols.toml") and finding.action


def test_the_split_yaml_init_writes_is_flow_style_the_boundary_editor_can_edit():
    from unbake import layout

    document = {
        "options": {"basename": "g", "asm_path": ["asm", "us"]},
        "segments": [
            [0, "header"],
            {
                "name": "main",
                "type": "code",
                "start": 4096,
                "subsegments": [[4096, "asm", "func_80001000"], [4104, "asm", "func_80001008"]],
            },
        ],
    }
    text = repo._dump_split(document)
    assert yaml.safe_load(text) == document and b"- [4096, asm, func_80001000]" in text
    edited = layout.edit_split(text, "func_80001008", start=4108, rename="func_8000100C")
    assert yaml.safe_load(edited)["segments"][1]["subsegments"][1] == [4108, "asm", "func_8000100C"]


@pytest.mark.parametrize(
    "keys,failures,attempts",
    [
        (("journal.changed",), 1, 2),
        (("journal.changed",), 2, 2),  # one recapture; a second moving publication refuses
        (("worker.crash",), 1, 1),  # unchanged inputs leave missing files as real faults
        (("compile.error",), 1, 1),
        (("journal.changed", "compile.error"), 1, 1),
    ],
)
def test_setup_converge_recaptures_only_stale_inputs_once(setup_case, monkeypatch, keys, failures, attempts):
    snapshot, stages, events, plans, plan, _ = setup_case
    planned = []
    refusal = Refusal(*(Finding(key, "source missing", path="src/rodata.func_80233DFC_de.800C3028.c") for key in keys))
    captures = Mock(side_effect=lambda cfg: replace(snapshot, reads=dict(snapshot.reads)))
    proof = Mock(return_value={})
    monkeypatch.setattr(repo.layout, "capture", captures)
    monkeypatch.setattr(repo, "_prove", proof)

    def infer(current, **kwargs):
        assert plans == []  # no stale plan was written
        planned.append(current)
        if len(planned) <= failures:
            raise refusal
        return plan({})

    monkeypatch.setattr(repo.infer, "plan", infer)
    if failures < attempts:
        repo.setup(snapshot.config, {})
        assert planned[0] is not planned[1]
    else:
        with pytest.raises(Refusal) as caught:
            repo.setup(snapshot.config, {})
        assert caught.value is refusal and plans == []
    assert len(planned) == attempts
    assert proof.call_count == 1
    assert stages.count("repo.setup.extract") == stages.count("repo.setup.map") == 1
    assert len([e for e in events if isinstance(e, tuple) and e[0] == "repo.extract"]) == 1


def test_prove_jobs_carry_names_and_the_unit_is_made_by_the_worker(setup_case, monkeypatch):
    snapshot, *_ = setup_case
    seen = []
    monkeypatch.setattr(
        repo.pool,
        "map",
        lambda config, name, function, jobs, key: (
            seen.append((function, jobs)) or [(fixture.proof("p", "e", j[2], True),) for j in jobs]
        ),
    )
    repo._prove(snapshot, ["a"])
    function, jobs = seen[0]
    assert function is repo._prove_unit and all(isinstance(j[1], str) and isinstance(j[2], str) for j in jobs)
