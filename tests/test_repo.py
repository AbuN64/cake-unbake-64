"""Repo tests use synthetic bytes and mocked native/pool/journal operations only."""

from __future__ import annotations

import tomllib
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import fixture
import pytest

from unbake import adapters, repo
from unbake import config as configuration
from unbake.contracts import NativeResult, Refusal, sha1

pytestmark = pytest.mark.usefixtures("plain_roms")


@pytest.fixture(autouse=True)
def _compiler_row(monkeypatch):
    row = {"never_emits": ["mfc0", "mtc0", "cache", "sync", "eret"], "fpu_control_with": ["trunc.w.s"]}
    monkeypatch.setattr(adapters, "row", lambda id: row)


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


HEADER_TEXT = "#ifndef INCLUDE_ASM_H\n#define INCLUDE_ASM_H\n#endif\n"


@pytest.fixture(autouse=True)
def _header_template(monkeypatch):
    real = configuration.template
    monkeypatch.setattr(
        configuration, "template", lambda name: HEADER_TEXT if name == "include_asm.h.in" else real(name)
    )


def _write(root, path, text):
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(text)


def _config(root, host=None):
    rows = (
        SimpleNamespace(name="main", rom=0x1000, vram=0x80000400, kind="code"),
        SimpleNamespace(name="data", rom=0x2000, vram=None, kind="bin"),
    )
    version = {v: SimpleNamespace(segments=rows) for v in ("a", "b")}
    project = SimpleNamespace(
        root=root, name="demo", versions=("a", "b"), version=version, files={}, toolchain="gcc-test"
    )
    host = host or SimpleNamespace(slots=4, workers=4, move_wait_s=1, tools={"splat": root / "splat", "git": "git"})
    return SimpleNamespace(project=project, host=host, digest="config-digest")


def _tree(owns, unowned=None, fuzzy=(), stale=None, sections=None):
    modules = [
        SimpleNamespace(
            path=path, owns=names, fuzzy=fuzzy, stand_ins={}, stale=(stale or {}).get(path, {}),
            sections={v: {n: (sections or {}).get(n, ".text") for n in ns} for v, ns in names.items()},
        )
        for path, names in owns.items()
    ]  # fmt: skip
    unowned = unowned or {}
    return SimpleNamespace(
        modules=tuple(modules),
        digest="tree",
        unowned=lambda v: tuple(unowned.get(v, ())),
        owner=lambda v, name: next((m.path for m in modules if name in m.owns.get(v, ())), None),
    )


MADE = {"exit": 0, "versions": ["a", "b"]}  # build.make_check on a clean make

FACTS = {
    "f_main": (0x80000400, 0x1000, 0x40, "func"),
    "f_mid": (0x80000440, 0x1040, 0x40, "func"),
    "f_late": (0x80000480, 0x1080, 0x40, "func"),
    "d_blob": (0x80100000, 0x2000, 0x10, "data"),
}


class World:
    """Everything setup and sync touch outside the project directory, replaced and counted."""

    def __init__(self, root, monkeypatch):
        self.root, self.cfg = root, _config(root)
        self.captures, self.extracts, self.makes, self.cache = [], [], [], {}
        self.tree = _tree({"src/main/a.c": {"a": ("f_main",), "b": ("f_main",)}})
        for v in ("a", "b"):
            _write(root, f"roms/baserom.{v}.z64", v)
            _write(root, f"versions/{v}/symbol_addrs.txt", "f_main = 0x80000400; // type:func rom:0x1000\n")
            _write(root, f"versions/{v}/splat.yaml", "generated\n")
        _write(root, "src/main/a.c", "int f_main(void) { return 1; }\n")

        def capture(config, overlay=None):
            self.captures.append(dict(overlay or {}))
            return self.tree

        def extract(config, v):
            self.extracts.append(v)
            (root / "asm" / v).mkdir(parents=True, exist_ok=True)
            return NativeResult((), "", 0, None, b"", b"", 0.0, 0.0, 0, {})

        def cached(config, kind, key, produce):
            if (kind, key) not in self.cache:
                self.cache[kind, key] = produce()
            return self.cache[kind, key]

        def make(config, targets, jobs):
            self.makes.append((tuple(targets), jobs))
            return fixture.native_result()

        generated = lambda *a: b"generated\n"  # noqa: E731
        patches = {
            (repo.tree, "capture"): capture,
            (repo.api, "symbols"): lambda config, v: FACTS,
            (repo.build, "makefile"): generated,
            (repo.build, "linker"): generated,
            (repo.build, "splat_yaml"): generated,
            (repo.build, "text_sizes"): generated,
            (repo.build, "extract"): extract,
            (repo.build, "install_toolchains"): Mock(),
            (repo.build, "local_mk"): lambda config: b"local\n",
            (repo.process, "make"): make,
            (repo.process, "git"): lambda config, *a: SimpleNamespace(stdout=b"same\n"),
            (repo.process, "run"): Mock(side_effect=AssertionError("unmocked process")),
            (repo.store, "cached"): cached,
            (repo.store, "exclusive"): contextmanager(lambda config, name, **kw: (yield True)),
            (repo.journal, "recover"): Mock(),
            (repo.report, "progress"): Mock(return_value={}),
            (repo.pool, "map"): lambda config, name, fn, items, *a: [fn(i) for i in items],
            (repo, "_splat_version"): lambda tool, root: b"0.50.0",
        }
        for (module, name), value in patches.items():
            monkeypatch.setattr(module, name, value, raising=False)
        self.toolchains = patches[repo.build, "install_toolchains"]
        self.progress = patches[repo.report, "progress"]

    def setup(self):
        return repo.setup(self.cfg, {})


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_setup_captures_once(world):
    result = world.setup()
    assert len(world.captures) == 1 and world.captures[0] == {}
    assert world.makes == [(("check",), 4)] and sorted(world.extracts) == ["a", "b"]
    assert not hasattr(repo, "infer") and not hasattr(repo, "converge") and not hasattr(repo, "upgrade")
    assert result["extracted"] == ["a", "b"] and result["make"] == MADE
    world.toolchains.assert_called_once()
    world.progress.assert_called_once()


def test_setup_runs_in_the_order_toolchains_extract_sync_make_progress(world, monkeypatch):
    order = []
    monkeypatch.setattr(repo.build, "install_toolchains", lambda config, ids: order.append("toolchains"))
    real_extract, real_sync = repo.build.extract, repo._sync
    monkeypatch.setattr(repo.build, "extract", lambda c, v: order.append("extract") or real_extract(c, v))
    monkeypatch.setattr(repo, "_sync", lambda c, p: order.append("sync") or real_sync(c, p))
    monkeypatch.setattr(repo.process, "make", lambda c, t, j: order.append("make") or fixture.native_result())
    monkeypatch.setattr(repo.report, "progress", lambda c, t: order.append("progress"))
    world.setup()
    assert [o for i, o in enumerate(order) if o not in order[:i]] == [
        "toolchains",
        "extract",
        "sync",
        "make",
        "progress",
    ]


def test_noop_setup_runs_nothing(world):
    world.setup()
    before = (len(world.captures), len(world.extracts), len(world.makes), world.progress.call_count)
    again = world.setup()
    assert (len(world.captures), len(world.extracts), len(world.makes), world.progress.call_count) == before
    assert again["extracted"] == [] and again["files"] == [] and again["make"] == MADE


def test_extract_cache(world):
    world.setup()
    assert sorted(world.extracts) == ["a", "b"]
    (world.root / ".unbake/setup.json").unlink()  # state forgotten, the extraction inputs are not
    result = world.setup()
    assert sorted(world.extracts) == ["a", "b"] and result["extracted"] == []  # two cache hits, no splat
    _write(world.root, "versions/a/symbol_addrs.txt", "f_main = 0x80000400; // type:func rom:0x1000 size:0x40\n")
    world.setup()
    assert sorted(world.extracts) == ["a", "a", "b"]  # only the version whose input changed


def test_a_yaml_that_sync_rewrote_is_extracted_again(world):
    (world.root / "versions/a/splat.yaml").write_text("stale\n")
    result = world.setup()
    assert world.extracts.count("a") == 2 and world.extracts.count("b") == 1
    assert "versions/a/splat.yaml" in result["files"]


def _sync_world(world, unowned):
    world.tree = _tree(
        {"src/main/a.c": {"a": ("f_main",), "b": ("f_main",)}, "src/main/z.c": {"a": ("f_late",)}},
        {"a": unowned, "b": unowned},
    )
    _write(world.root, "src/main/z.c", '#include "include_asm.h"\nINCLUDE_ASM("main", f_late);\n')


def test_sync_check_lists_changes_and_writes_nothing(world):
    with pytest.raises(Refusal) as refused:
        repo.sync(world.cfg, {"check": True})
    (finding,) = refused.value.findings
    assert finding.key == "sync.changed"
    assert {"Makefile", "demo.ld", "ci/run", ".gitlab-ci.yml", "include/include_asm.h"} <= set(finding.missing)
    assert sorted(p.name for p in world.root.iterdir()) == ["roms", "src", "versions"]
    repo.sync(world.cfg, {"check": False})
    assert repo.sync(world.cfg, {"check": True}) == {"written": [], "stand_ins": {}}  # now nothing would change


def test_sync_writes_the_generated_files_and_the_header_only_when_missing(world):
    written = repo.sync(world.cfg, {"check": False})["written"]
    assert {"Makefile", "demo.ld", "ci/run", ".github/workflows/ci.yml"} <= set(written)
    assert (world.root / "include/include_asm.h").read_text() == HEADER_TEXT
    (world.root / "include/include_asm.h").write_text("mine\n")
    assert "include/include_asm.h" not in repo.sync(world.cfg, {"check": False})["written"]
    assert (world.root / "include/include_asm.h").read_text() == "mine\n"


def test_sync_appends_a_stand_in_for_an_unowned_name_to_the_file_owning_the_previous_address(world):
    _sync_world(world, ["f_mid"])
    result = repo.sync(world.cfg, {"check": False})
    assert (world.root / "src/main/a.c").read_text().endswith('\nINCLUDE_ASM("main", f_mid);\n')
    assert (world.root / "src/main/z.c").read_text().count("INCLUDE_ASM") == 1  # the later file is untouched
    assert "src/main/a.c" in result["written"]
    assert world.captures[-1]["src/main/a.c"].decode().count("f_mid") == 1  # the link script sees the stand-in


def test_sync_puts_names_nothing_precedes_before_the_next_owner_and_makes_no_new_file(world):
    world.tree = _tree({"src/main/z.c": {"a": ("f_late",)}}, {"a": ["f_main", "f_mid"], "b": ["f_main", "f_mid"]})
    _write(world.root, "src/main/z.c", "int f_late(void) { return 3; }\n")
    repo.sync(world.cfg, {"check": False})
    assert not (world.root / "src/main/80000400.c").exists()
    assert (world.root / "src/main/z.c").read_text() == (
        '#include "include_asm.h"\n\nINCLUDE_ASM("main", f_main);\nINCLUDE_ASM("main", f_mid);\n'
        "int f_late(void) { return 3; }\n"
    )


def test_sync_puts_a_name_nothing_precedes_before_the_next_owners_guarded_stand_in(world):
    world.tree = _tree({"src/main/z.c": {"a": ("f_late",)}}, {"a": ["f_main"], "b": ["f_main"]})
    _write(
        world.root, "src/main/z.c", '#include "include_asm.h"\n\n#if VERSION_A\nINCLUDE_ASM("main", f_late);\n#endif\n'
    )
    repo.sync(world.cfg, {"check": False})
    lines = (world.root / "src/main/z.c").read_text().splitlines()
    assert lines[2:] == ['INCLUDE_ASM("main", f_main);', "#if VERSION_A", 'INCLUDE_ASM("main", f_late);', "#endif"]


def test_sync_makes_a_new_file_only_for_a_segment_without_owners(world):
    world.tree = _tree({}, {"a": ["f_main", "f_mid"], "b": ()})
    repo.sync(world.cfg, {"check": False})
    text = (world.root / "src/main/80000400.c").read_text()
    assert text.startswith('#include "include_asm.h"') and text.count("INCLUDE_ASM") == 2
    assert text.index("f_main") < text.index("f_mid")


def test_sync_stands_in_data_with_include_data(world):
    world.tree = _tree({"src/main/a.c": {"a": ("f_main",)}}, {"a": ["d_blob"], "b": ()})
    repo.sync(world.cfg, {"check": False})
    assert 'INCLUDE_DATA("data", d_blob);' in (world.root / "src/data/80100000.c").read_text()


def test_sync_guards_a_name_another_version_defines(world):
    world.tree = _tree({"src/main/a.c": {"a": ("f_mid",)}}, {"a": (), "b": ["f_mid"]})
    repo.sync(world.cfg, {"check": False})
    assert (world.root / "src/main/a.c").read_text().endswith('#if VERSION_B\nINCLUDE_ASM("main", f_mid);\n#endif\n')


def test_sync_guards_a_new_stand_in_by_the_versions_that_lack_a_file_of_it(world):
    """A version that knows the name only as a second name of owned bytes, or not at all, extracts no file of it."""
    _sync_world(world, ["f_mid"])
    world.tree = _tree({"src/main/z.c": {"a": ("f_late",)}}, {"a": ["f_mid"], "b": ()})
    repo.sync(world.cfg, {"check": False})
    assert '\n#if VERSION_A\nINCLUDE_ASM("main", f_mid);\n#endif\n' in (world.root / "src/main/z.c").read_text()


def test_sync_puts_a_stand_in_right_after_the_previous_name_and_gives_data_the_section_before_it(world):
    world.tree = _tree(
        {"src/main/a.c": {"a": ("f_main", "f_late"), "b": ("f_main", "f_late")}}, {"a": ["f_mid"], "b": ["f_mid"]}
    )
    _write(
        world.root,
        "src/main/a.c",
        '#include "include_asm.h"\nINCLUDE_ASM("main", f_main);\nINCLUDE_ASM("main", f_late);\n',
    )
    repo.sync(world.cfg, {"check": False})
    assert (world.root / "src/main/a.c").read_text().splitlines()[1:] == [
        'INCLUDE_ASM("main", f_main);', 'INCLUDE_ASM("main", f_mid);', 'INCLUDE_ASM("main", f_late);'
    ]  # fmt: skip
    facts = {"d_one": (0x80100000, 0x2000, 0x10, "data"), "d_two": (0x80100008, 0x2008, 0, "data")}
    world.tree = _tree(
        {"src/data/d.c": {"a": ("d_one",)}}, {"a": ["d_two"], "b": ["d_two"]}, sections={"d_one": ".rodata"}
    )
    repo.api.symbols = lambda config, v: facts
    _write(
        world.root, "src/data/d.c", '#include "include_asm.h"\n#if VERSION_A\nINCLUDE_RODATA("data", d_one);\n#endif\n'
    )
    repo.sync(world.cfg, {"check": False})
    assert (world.root / "src/data/d.c").read_text().endswith('#endif\nINCLUDE_RODATA("data", d_two);\n')


def test_sync_puts_a_stand_in_after_the_c_body_of_the_previous_name_not_after_its_other_versions_stand_in(world):
    world.tree = _tree(
        {"src/main/a.c": {"a": ("f_main", "f_late"), "b": ("f_main", "f_late")}}, {"a": ["f_mid"], "b": ["f_mid"]}
    )
    for module in world.tree.modules:
        module.stand_ins = {"a": ("f_late",), "b": ("f_late",)}
    _write(
        world.root,
        "src/main/a.c",
        '#include "include_asm.h"\n#if VERSION_X\nINCLUDE_ASM("main", f_main);\n#endif\n'
        "s32 f_main(void) {\n    return 1;\n}\n"
        'INCLUDE_ASM("main", f_late);\n',
    )
    repo.sync(world.cfg, {"check": False})
    assert (world.root / "src/main/a.c").read_text().splitlines()[-3:] == [
        "}", 'INCLUDE_ASM("main", f_mid);', 'INCLUDE_ASM("main", f_late);'
    ]  # fmt: skip


def test_sync_removes_a_stand_in_a_c_body_supersedes(world):
    text = '#include "include_asm.h"\nint f_main(void) { return 1; }\n#if VERSION_A\nINCLUDE_ASM("main", f_main);\n'
    _write(world.root, "src/main/a.c", text + '#endif\nINCLUDE_ASM("main", f_mid);\nINCLUDE_ASM("main", f_late);\n')
    world.tree = _tree(
        {"src/main/a.c": {"a": ("f_main", "f_late"), "b": ("f_main", "f_late")}},
        stale={"src/main/a.c": {"a": ("f_main", "f_mid"), "b": ("f_mid",)}},
    )
    result = repo.sync(world.cfg, {"check": False})
    assert (world.root / "src/main/a.c").read_text() == (
        '#include "include_asm.h"\nint f_main(void) { return 1; }\nINCLUDE_ASM("main", f_late);\n'
    )
    assert "src/main/a.c" in result["written"]


def test_a_name_in_no_segment_is_refused(world):
    world.tree = _tree({}, {"a": ["f_main"], "b": ()})
    monkey = {"f_main": (0x1, 0x10, 0, "func")}
    repo.api.symbols = lambda config, v: monkey
    with pytest.raises(Refusal) as refused:
        repo.sync(world.cfg, {"check": False})
    assert refused.value.findings[0].key == "sync.segment"


def test_symbols_command_lists_every_version(world):
    out = repo.symbols_command(world.cfg, {"version": None})
    assert set(out["symbols"]) == {"a", "b"} and out["symbols"]["a"]["f_main"] == [0x80000400, 0x1000, 0x40, "func"]
    assert set(repo.symbols_command(world.cfg, {"version": "b"})["symbols"]) == {"b"}


SPLAT_YAML = (
    "options:\n  basename: x\n"
    "segments:\n"
    "  - [0x0, header, header]\n"
    "  - name: main\n    type: code\n    start: 0x1000\n    vram: 0x80000400\n"
    "  - [0x1B5A50]\n"
)


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
        calls.append(tuple(argv[1:]))
        if argv[1] == "create_config":
            (cwd / "generated.yaml").write_text(SPLAT_YAML)
        return fixture.native_result(argv)

    synced = []
    monkeypatch.setattr(repo.process, "run", run)
    monkeypatch.setattr(configuration, "load", lambda root, host: _config(root))
    monkeypatch.setattr(repo, "sync", lambda cfg, params: synced.append(params) or {"written": ["Makefile"]})
    return params, root, calls, synced


def test_init_writes_a_schema_4_config_seeded_from_splat_and_then_syncs(init_case):
    params, root, calls, synced = init_case
    result = repo.init(params)
    document = tomllib.loads((root / "config.toml").read_text())
    assert document["schema"] == 4 and document["name"] == "demo" and set(document["version"]) == {"a", "b"}
    assert document["version"]["a"]["segments"] == [
        {"name": "header", "rom": 0, "kind": "bin"},
        {"name": "main", "rom": 0x1000, "vram": 0x80000400},
    ]
    assert (root / "include/include_asm.h").read_text() == HEADER_TEXT and (root / ".gitignore").is_file()
    assert synced == [{"check": False}] and "Makefile" in result["created"]
    assert not list(root.glob("*.yaml")) and ("--version",) not in calls


def test_init_never_writes_the_retired_files(init_case):
    params, root, calls, _synced = init_case
    created = repo.init(params)["created"]
    banned = (".recomp.json", "boards.yml", "symbols.toml", "types.toml", "layout.toml")
    assert not [p for p in created if p.endswith(banned)]
    assert not [p.name for p in root.rglob("*") if p.name in banned]
    assert ("init",) in calls and all(c[0] in {"create_config", "init"} for c in calls)


@pytest.mark.parametrize(
    "bad", [{"names_from": "zz"}, {"toolchain": "no-such"}, {"rom": ["a"]}, {"rom": ["a=/no/such/file"]}]
)
def test_init_bad_request(init_case, bad):
    params, *_ = init_case
    with pytest.raises(Refusal) as refused:
        repo.init({**params, **bad})
    assert refused.value.findings[0].key == "init.request"


def test_init_refuses_a_directory_that_is_not_empty(init_case):
    params, root, *_ = init_case
    root.mkdir()
    (root / "x").write_text("x")
    with pytest.raises(Refusal) as refused:
        repo.init(params)
    assert refused.value.findings[0].key == "init.request"


def test_sync_refreshes_readme_from_the_current_report_renderer(world, monkeypatch):
    (world.root / "README.md").write_text("old progress")
    calls = []
    monkeypatch.setattr(
        repo.report, "files", lambda config, tree: calls.append(tree) or {"README.md": b"current progress"}
    )
    result = repo.sync(world.cfg, {"check": False})
    assert (world.root / "README.md").read_bytes() == b"current progress"
    assert "README.md" in result["written"] and len(calls) == 1
    repo.sync(world.cfg, {"check": True})
    assert len(calls) == 1
