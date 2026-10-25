"""Repo tests use synthetic bytes and mocked native/pool/journal operations only."""

from __future__ import annotations

import os
import tomllib
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import fixture
import pytest

from unbake import adapters, repo
from unbake import config as configuration
from unbake.contracts import Finding, NativeResult, Refusal, sha1

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
    host = host or SimpleNamespace(slots=4, workers=4, move_wait_s=1, cache_dir=root.parent / "cache", cache_max_bytes=10**9, tools={"splat": root / "splat", "git": "git"})
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
        _write(root, "Makefile", "generated\n")  # as sync writes it: a Makefile change extracts again

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


def test_a_setup_without_make_prepares_and_a_later_setup_still_builds(world):
    result = repo.setup(world.cfg, {"no_make": True})
    assert world.makes == [] and sorted(world.extracts) == ["a", "b"] and result["make"] is None
    world.progress.assert_not_called()  # the caller's build records the verdict
    assert repo.setup(world.cfg, {"no_make": True})["extracted"] == [] and world.makes == []  # nothing moved
    assert world.setup()["make"] == MADE and world.makes == [(("check",), 4)]  # a full setup does not take it as done


def test_extract_cache(world):
    world.setup()
    assert sorted(world.extracts) == ["a", "b"]
    (world.root / ".unbake/setup.json").unlink()  # state forgotten, the extraction inputs are not
    result = world.setup()
    assert sorted(world.extracts) == ["a", "b"] and result["extracted"] == []  # two cache hits, no splat
    _write(world.root, "versions/a/symbol_addrs.txt", "f_main = 0x80000400; // type:func rom:0x1000 size:0x40\n")
    world.setup()
    assert sorted(world.extracts) == ["a", "a", "b"]  # only the version whose input changed


def test_only_the_extract_recipe_of_the_makefile_keys_the_extraction(world):
    """A Makefile change outside the extract recipe (a new assembler flag) extracts nothing again; one in the recipe
    or a variable it uses does."""
    rule = "SED := -e s/a/b/\nCC := cc\nextract-%:\n\tsplat $*\n\tsed $(SED) asm/$*/x.s\n"
    _write(world.root, "Makefile", rule)
    key = repo._extract_key(world.cfg, "a")
    _write(world.root, "Makefile", rule.replace("CC := cc", "CC := gcc") + "build/%.o:\n\t$(CC) -c $<\n")
    assert repo._extract_key(world.cfg, "a") == key
    _write(world.root, "Makefile", rule.replace("s/a/b/", "s/a/c/"))
    assert repo._extract_key(world.cfg, "a") != key


def test_blobs_toml_does_not_key_the_extraction(world):
    key = repo._extract_key(world.cfg, "a")
    _write(world.root, "blobs.toml", 'schema = 1\n\n[blob."x"]\ntype = "world"\n')
    assert repo._extract_key(world.cfg, "a") == key


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
    (world.root / "Makefile").unlink()  # a tree sync has not written yet
    with pytest.raises(Refusal) as refused:
        repo.sync(world.cfg, {"check": True})
    (finding,) = refused.value.findings
    assert finding.key == "sync.changed"
    assert {"Makefile", "demo.ld", "ci/run", ".gitlab-ci.yml", "include/include_asm.h"} <= set(finding.missing)
    assert sorted(p.name for p in world.root.iterdir()) == ["roms", "src", "versions"]
    repo.sync(world.cfg, {"check": False})
    assert repo.sync(world.cfg, {"check": True}) == {"written": [], "stand_ins": {}}  # now nothing would change


def test_sync_leaves_an_existing_makefile_alone_until_asked(world):
    repo.sync(world.cfg, {})
    generated = (world.root / "Makefile").read_bytes()
    (world.root / "Makefile").write_bytes(b"# main's older generation\n" + generated)  # what a newer tool would rewrite
    assert repo.sync(world.cfg, {"check": True}) == {"written": [], "stand_ins": {}}
    assert "Makefile" not in repo.sync(world.cfg, {})["written"]
    assert (world.root / "Makefile").read_bytes().startswith(b"# main's older")
    assert "Makefile" in repo.sync(world.cfg, {"makefile": True})["written"]
    assert (world.root / "Makefile").read_bytes() == generated


def test_sync_settle_writes_then_checks_again_only_after_a_write_and_refuses_a_moving_output(world, monkeypatch):
    (world.root / "Makefile").unlink()
    runs = []
    real = repo._sync
    monkeypatch.setattr(repo, "_sync", lambda c, p: runs.append(bool(p.get("check"))) or real(c, p))
    assert "Makefile" in repo.sync(world.cfg, {"settle": True})["written"] and runs == [False, False, True]
    runs.clear()
    assert repo.sync(world.cfg, {"settle": True})["written"] == [] and runs == [False]  # a fixed point: no recheck
    calls = iter(range(100))
    generated = repo._generated
    monkeypatch.setattr(repo, "_generated", lambda c, t: {**generated(c, t), "moving.txt": b"%d" % next(calls)})
    with pytest.raises(Refusal) as refused:
        repo.sync(world.cfg, {"settle": True})
    assert refused.value.findings[0].key == "sync.changed" and "moving.txt" in refused.value.findings[0].missing


def test_sync_writes_the_generated_files_and_the_header_only_when_missing(world):
    (world.root / "Makefile").unlink()  # a tree sync has not written yet
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


def test_sync_drops_a_word_array_a_literals_row_supersedes_and_the_file_that_held_only_it(world, monkeypatch):
    monkeypatch.setattr(repo.api, "symbols", lambda config, v: {**FACTS, "d_chunk": (0x80100000, 0x2000, 8, "data")})
    array = "#if VERSION_A\nconst unsigned int d_chunk[2] = {\n    0x00000001, 0x00000002,\n};\n#endif\n"
    _write(world.root, "src/main/chunk.c", "\n" + array)
    _write(world.root, "src/main/keep.c", "int f_keep(void) { return 1; }\n" + array)
    world.tree = _tree(
        {"src/main/chunk.c": {}, "src/main/keep.c": {"a": ("f_keep",)}},
        stale={"src/main/chunk.c": {"a": ("d_chunk",)}, "src/main/keep.c": {"a": ("d_chunk",)}},
    )
    monkeypatch.setattr(repo, "_PASSES", 0)  # the stub tree never reflects the first pass
    repo.sync(world.cfg, {"check": False})
    assert not (world.root / "src/main/chunk.c").exists()
    assert (world.root / "src/main/keep.c").read_text() == "int f_keep(void) { return 1; }\n"


def test_sync_keeps_a_stand_in_a_literals_row_supersedes_in_one_version_only_for_the_others(world, monkeypatch):
    # one word array stands for the bytes in a and b; a literals row takes them in b only: b leaves it, a keeps it
    monkeypatch.setattr(repo.api, "symbols", lambda config, v: {**FACTS, "d_chunk": (0x80100000, 0x2000, 8, "data")})
    array = "const unsigned int d_chunk[2] = {\n    0x00000001, 0x00000002,\n};\n"
    _write(world.root, "src/main/keep.c", "int f_keep(void) { return 1; }\n" + array)
    world.tree = _tree(
        {"src/main/keep.c": {"a": ("f_keep", "d_chunk"), "b": ("f_keep",)}},
        stale={"src/main/keep.c": {"b": ("d_chunk",)}},
        sections={"d_chunk": ".rodata"},
    )
    monkeypatch.setattr(repo, "_PASSES", 0)  # the stub tree never reflects the first pass
    repo.sync(world.cfg, {"check": False})
    assert (world.root / "src/main/keep.c").read_text() == (
        "int f_keep(void) { return 1; }\n#if !(defined(VERSION_B))\n" + array + "#endif\n"
    )


def test_sync_fences_a_stand_in_once_however_often_it_runs(world, monkeypatch):
    # 800E2328_2.CPP gained one more `#if !(defined(...))` wrapper per sync run
    monkeypatch.setattr(repo.api, "symbols", lambda config, v: {**FACTS, "d_chunk": (0x80100000, 0x2000, 8, "data")})
    array = "const unsigned int d_chunk[2] = {\n    0x00000001, 0x00000002,\n};\n"
    _write(world.root, "src/main/keep.c", "int f_keep(void) { return 1; }\n" + array)
    world.tree = _tree(
        {"src/main/keep.c": {"a": ("f_keep", "d_chunk"), "b": ("f_keep",)}},
        stale={"src/main/keep.c": {"b": ("d_chunk",)}},
        sections={"d_chunk": ".rodata"},
    )
    monkeypatch.setattr(repo, "_PASSES", 0)  # the stub tree never reflects the first pass
    repo.sync(world.cfg, {"check": False})
    once = (world.root / "src/main/keep.c").read_text()
    for _ in range(2):
        repo.sync(world.cfg, {"check": False})
    assert (world.root / "src/main/keep.c").read_text() == once
    assert once.count("#if !(") == 1


def test_sync_drops_a_data_word_array_a_literals_row_supersedes(world, monkeypatch):
    # a pad the extractor wrote as .data (no const) inside a literals row's bytes goes as a const array does
    monkeypatch.setattr(repo.api, "symbols", lambda config, v: {**FACTS, "d_pad": (0x80100000, 0x2000, 4, "data")})
    pad = "unsigned int d_pad[1] = {\n    0x00000000,\n};\n"
    _write(world.root, "src/main/keep.c", "int f_keep(void) { return 1; }\n" + pad)
    world.tree = _tree(
        {"src/main/keep.c": {"a": ("f_keep",), "b": ("f_keep",)}},
        stale={"src/main/keep.c": {"a": ("d_pad",), "b": ("d_pad",)}},
    )
    monkeypatch.setattr(repo, "_PASSES", 0)  # the stub tree never reflects the first pass
    repo.sync(world.cfg, {"check": False})
    assert (world.root / "src/main/keep.c").read_text() == "int f_keep(void) { return 1; }\n"


def test_sync_shrinks_a_word_array_to_the_words_before_a_literals_row(world, monkeypatch):
    words = "    0x00000000, 0x00000001, 0x00000002, 0x00000003, 0x00000004,\n"
    _write(world.root, "src/main/chunk.c", "const unsigned int d_chunk[5] = {\n" + words + "};\n")
    facts = {"d_chunk": (0x80100000, 0x2000, 0x14, "data"), "literals_f": (0x80100004, 0x2004, None, "data")}
    monkeypatch.setattr(repo.api, "symbols", lambda config, v: facts)
    world.tree = _tree({"src/main/chunk.c": {"a": ("d_chunk",), "b": ("d_chunk",)}}, sections={"d_chunk": ".rodata"})
    repo.sync(world.cfg, {"check": False})
    assert (world.root / "src/main/chunk.c").read_text() == "const unsigned int d_chunk[1] = {\n    0x00000000,\n};\n"


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
    monkeypatch.setattr(repo.report, "publish_hold", lambda config, tree: None)
    monkeypatch.setattr(
        repo.report, "files", lambda config, tree: calls.append(tree) or {"README.md": b"current progress"}
    )
    result = repo.sync(world.cfg, {"check": False})
    assert (world.root / "README.md").read_bytes() == b"current progress"
    assert "README.md" in result["written"] and len(calls) == 1
    repo.sync(world.cfg, {"check": True})
    assert len(calls) == 2  # check renders what sync would write, to compare


def held(key, action):
    return Finding(key, "held", blocking=False, action=action)


def test_sync_without_a_verdict_keeps_the_progress_files_and_says_to_check(world, monkeypatch):
    (world.root / "README.md").write_text("old progress")
    (world.root / "progress.json").write_text("old json")
    monkeypatch.setattr(
        repo.report, "publish_hold", lambda config, tree: held("sync.unchecked", "run make check, then unbake sync")
    )
    monkeypatch.setattr(repo.report, "files", lambda config, tree: pytest.fail("numbers derived without a verdict"))
    result = repo.sync(world.cfg, {"check": False})
    assert (world.root / "README.md").read_text() == "old progress"
    assert (world.root / "progress.json").read_text() == "old json"
    (note,) = result["notes"]
    assert (note["key"], note["blocking"], note["action"]) == (
        "sync.unchecked",
        False,
        "run make check, then unbake sync",
    )


def test_sync_check_refuses_a_progress_file_that_differs_from_what_sync_writes(world, monkeypatch):
    (world.root / "README.md").write_text("zero percent")
    monkeypatch.setattr(repo.report, "publish_hold", lambda config, tree: None)
    monkeypatch.setattr(repo.report, "files", lambda config, tree: {"README.md": b"real", "progress.json": b"{}\n"})
    (world.root / "progress.json").write_text("{}\n")
    repo.sync(world.cfg, {"check": False})
    assert (world.root / "README.md").read_text() == "real"
    (world.root / "README.md").write_text("zero percent")
    with pytest.raises(Refusal) as refused:
        repo.sync(world.cfg, {"check": True})
    (finding,) = refused.value.findings
    assert (finding.key, finding.missing) == ("sync.changed", ("README.md",))
    assert (world.root / "README.md").read_text() == "zero percent"


def test_a_tree_whose_facts_went_back_to_an_earlier_input_is_extracted_again(world):
    """The extraction in asm/<v>/ is the one of the last facts, whatever other trees or earlier runs extracted."""
    original = (world.root / "versions/a/symbol_addrs.txt").read_text()
    world.setup()
    _write(world.root, "versions/a/symbol_addrs.txt", original.replace("rom:0x1000", "rom:0x1000 size:0x40"))
    world.setup()
    _write(world.root, "versions/a/symbol_addrs.txt", original)  # the first input again: a "seen" key, other assembly
    result = world.setup()
    assert world.extracts == ["a", "b", "a", "a"] and result["extracted"] == ["a"]


def test_assembly_a_neighbour_tree_extracted_is_not_this_trees_extraction(world):
    world.setup()
    (world.root / "asm/a/.unbake-extract").unlink()  # this tree's assembly came from some other input
    (world.root / ".unbake/setup.json").unlink()
    world.cache.update({("extract", repo._extract_key(world.cfg, "a")): b"stamped by a neighbour"})
    result = world.setup()
    assert world.extracts == ["a", "b", "a"] and result["extracted"] == ["a"]


def test_an_extraction_that_died_leaves_no_stamp(world, monkeypatch):
    world.setup()
    _write(world.root, "versions/a/symbol_addrs.txt", "f_main = 0x80000400; // type:func rom:0x1000 size:0x40\n")

    def die(config, v):
        raise Refusal(Finding("setup.splat", "splat died"))

    monkeypatch.setattr(repo.build, "extract", die)
    with pytest.raises(Refusal):
        world.setup()
    assert not (world.root / "asm/a/.unbake-extract").exists()


def test_extract_job_extracts_again_when_an_asm_file_changed_under_its_stamp(tmp_path, monkeypatch):
    from types import SimpleNamespace

    asm = tmp_path / "asm" / "us"
    asm.mkdir(parents=True)
    (asm / "a.s").write_bytes(b"good\n")
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    ran = []
    monkeypatch.setattr(repo, "_extract_key", lambda c, v: "K")
    monkeypatch.setattr(repo.build, "extract", lambda c, v: ran.append(v))
    monkeypatch.setattr(repo, "_checked", lambda *a: None)
    repo._write_asm_manifest(config, "us")
    repo.store.write(repo._extract_stamp(config, "us"), b"K")
    assert repo.extract_job((config, "us")) is False
    (asm / "a.s").write_bytes(b"evil\n")
    assert repo._asm_damaged(config, "us") == ["asm/us/a.s"]
    assert repo.extract_job((config, "us")) is True
    assert ran == ["us"]


def test_missing_auto_symbols_file_makes_extract_job_extract_again(tmp_path, monkeypatch):
    from types import SimpleNamespace

    (tmp_path / "asm" / "us").mkdir(parents=True)
    (tmp_path / "asm" / "us" / "a.s").write_bytes(b"x\n")
    auto = tmp_path / "build" / "us" / "undefined_syms_auto.txt"
    auto.parent.mkdir(parents=True)
    auto.write_bytes(b"sym = 0;\n")
    stale = tmp_path / "build" / "us" / "link.ld"
    stale.write_bytes(b"old")
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(repo, "_extract_key", lambda c, v: "K")
    monkeypatch.setattr(repo.build, "extract", lambda c, v: None)
    monkeypatch.setattr(repo, "_checked", lambda *a: None)
    repo._write_asm_manifest(config, "us")
    repo.store.write(repo._extract_stamp(config, "us"), b"K")
    auto.unlink()
    assert repo.extract_job((config, "us")) is True
    assert not stale.exists()


# ---------------------------------------------------------------- jump tables of C functions the link would drop
def _table_tree(monkeypatch, sections):
    module = SimpleNamespace(
        path="src/m.c", owns={"a": ("f", *sections)}, stand_ins={"a": ()}, sections={"a": {"f": ".text", **sections}}
    )
    facts = {
        "f": (0x80400000, 0x1000, 0x40, "func"),
        "g": (0x80400040, 0x1040, None, "func"),
        "jtbl_800E1500": (0x800E1500, 0x2000, None, "data"),
        "D_800E1510": (0x800E1510, 0x2010, None, "data"),
        "literals_f": (0x800E1500, 0x2000, None, "data"),
    }
    if not sections:
        del facts["literals_f"]
    owned = {"f": "src/m.c", **{n: "src/m.c" for n in sections}}
    monkeypatch.setattr(repo.api, "symbols", lambda config, v: facts)
    return SimpleNamespace(
        modules=(module,),
        owner=lambda v, n: owned.get(n) or (owned.get("literals_f") if facts[n][0] == 0x800E1500 else None),
    )


def _rom_with_table(monkeypatch, tmp_path):
    words = b"".join(w.to_bytes(4, "big") for w in (0x00400010, 0x00400018, 0x00400020, 0x00400028))
    (tmp_path / "rom").write_bytes(bytes(0x2000) + words)
    monkeypatch.setattr(repo.configuration, "rom", lambda project, v: tmp_path / "rom")


def test_a_c_functions_jump_table_no_file_holds_names_the_literals_row(monkeypatch, tmp_path):
    # SPEED-4 func_8041D4E0_de: its INCLUDE_RODATA stand-in went, no literals_ row came; sync dropped the object's
    # .rodata from the link (ROM 0x38 short) instead of saying what places it
    _rom_with_table(monkeypatch, tmp_path)
    config = SimpleNamespace(project=SimpleNamespace(versions=("a",)))
    (found,) = repo._unplaced_tables(config, _table_tree(monkeypatch, {}))
    assert found.key == "sync.literals" and found.unit == "f" and found.versions == ("a",)
    assert "literals_f = 0x800E1500; // rom:0x2000" in found.action and "versions/a/aliases.txt" in found.action


def test_a_jump_table_the_files_own_rodata_holds_or_no_rom_is_no_finding(monkeypatch, tmp_path):
    _rom_with_table(monkeypatch, tmp_path)
    config = SimpleNamespace(project=SimpleNamespace(versions=("a",)))
    assert repo._unplaced_tables(config, _table_tree(monkeypatch, {"literals_f": ".rodata"})) == []

    monkeypatch.setattr(repo.configuration, "rom", lambda project, v: tmp_path / "absent")
    assert repo._unplaced_tables(config, _table_tree(monkeypatch, {})) == []


def test_words_inside_two_functions_or_off_code_are_no_jump_table():
    code = [(0x400000, 0x400040, "f"), (0x400040, 0x400080, "g")]
    assert repo._jump_target([0x400010, 0x400020], code) == "f"
    assert repo._jump_target([0x400010, 0x400050], code) is None
    assert repo._jump_target([0x400010, 0x500000], code) is None
    assert repo._jump_target([0x400010], code) is None


# ---------------------------------------------------------------- extraction again: only what changed rebuilds
def _extraction(tmp_path, monkeypatch, files):
    """A tree extracted once from `files` (asm/us/ relative path -> bytes), an object per C unit including one file,
    and a fake splat that writes whatever `files` holds at the time of the call."""
    from types import SimpleNamespace

    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path, versions=["us"]))
    keys = iter(["K1", "K2", "K3"])
    monkeypatch.setattr(repo, "_extract_key", lambda c, v: next(keys))
    monkeypatch.setattr(repo, "_checked", lambda *a: None)

    def splat(c, v):
        for rel, data in files.items():
            _write(tmp_path, f"asm/us/{rel}", data.decode())
        _write(tmp_path, "asm/us/.extracted", "")

    monkeypatch.setattr(repo.build, "extract", splat)
    assert repo.extract_job((config, "us")) is True
    for unit, included in (("a", "nonmatchings/a/f.s"), ("b", "nonmatchings/b/g.s")):
        _write(tmp_path, f"build/us/src/{unit}.c.s", f'    .include "{included}"\n')
        _write(tmp_path, f"build/us/src/{unit}.c.o", "obj")
    old = 1_000_000_000_000_000_000
    for path in (tmp_path / "asm/us").rglob("*"):
        if path.is_file():
            os.utime(path, ns=(old, old))
    repo._write_asm_manifest(config, "us")
    return config, old


def test_extracting_again_rebuilds_only_objects_including_a_changed_file(tmp_path, monkeypatch):
    files = {"macro.inc": b"m\n", "nonmatchings/a/f.s": b"f\n", "nonmatchings/b/g.s": b"g\n", "data/d.s": b"d\n"}
    config, old = _extraction(tmp_path, monkeypatch, files)
    files["nonmatchings/a/f.s"] = b"f renamed\n"
    del files["data/d.s"]  # a file the new facts no longer extract
    assert repo.extract_job((config, "us")) is True
    asm = tmp_path / "asm/us"
    assert (asm / "macro.inc").stat().st_mtime_ns == old and (asm / "nonmatchings/b/g.s").stat().st_mtime_ns == old
    assert (asm / ".extracted").stat().st_mtime_ns == old  # no whole-version rebuild
    assert (asm / "nonmatchings/a/f.s").stat().st_mtime_ns != old
    assert not (asm / "data/d.s").exists()  # a fresh directory: what a cold extraction holds, nothing older
    assert not (tmp_path / "build/us/src/a.c.o").exists() and (tmp_path / "build/us/src/b.c.o").exists()
    assert repo._asm_damaged(config, "us") == []


def test_a_changed_shared_include_rebuilds_the_whole_version(tmp_path, monkeypatch):
    files = {"macro.inc": b"m\n", "nonmatchings/a/f.s": b'.include "macro.inc"\n'}
    config, old = _extraction(tmp_path, monkeypatch, files)
    files["macro.inc"] = b"m2\n"
    assert repo.extract_job((config, "us")) is True
    assert (tmp_path / "asm/us/.extracted").stat().st_mtime_ns != old
    assert (tmp_path / "asm/us/nonmatchings/a/f.s").stat().st_mtime_ns == old
    assert (tmp_path / "build/us/src/a.c.o").exists()  # the fresh mark rebuilds every object of the version


def test_an_extraction_over_damaged_assembly_rebuilds_the_whole_version(tmp_path, monkeypatch):
    files = {"nonmatchings/a/f.s": b"f\n", "nonmatchings/b/g.s": b"g\n"}
    config, old = _extraction(tmp_path, monkeypatch, files)
    (tmp_path / "asm/us/nonmatchings/b/g.s").write_bytes(b"written through a link\n")
    assert repo.extract_job((config, "us")) is True
    assert (tmp_path / "asm/us/.extracted").stat().st_mtime_ns != old


def test_a_literals_row_running_over_another_functions_jump_table_names_the_row_that_splits_it(monkeypatch, tmp_path):
    # SPEED-1c func_8024D160_de: in us and eu the next row came after func_8024D284_de's 60-byte table, so the piece
    # literals_ sized swallowed it and the ROM would come out short with no warning
    own = (0x00400010, 0x00400018, 0x00400020, 0x00400028)
    other = (0x00400048, 0x00400050, 0x00400058)
    words = b"".join(w.to_bytes(4, "big") for w in (*own, *other))
    (tmp_path / "rom").write_bytes(bytes(0x2000) + words + bytes(4))
    monkeypatch.setattr(repo.configuration, "rom", lambda project, v: tmp_path / "rom")
    config = SimpleNamespace(project=SimpleNamespace(versions=("a",)))
    tree_ = _table_tree(monkeypatch, {"literals_f": ".rodata"})
    tree_.modules = (
        *tree_.modules,
        SimpleNamespace(path="src/n.c", owns={"a": ("g",)}, stand_ins={"a": ("g",)}, sections={"a": {"g": ".text"}}),
    )
    facts = repo.api.symbols(config, "a")
    del facts["D_800E1510"]
    facts["D_800E1520"] = (0x800E1520, 0x2020, None, "data")
    (found,) = repo._swallowed_tables(config, tree_)
    assert found.key == "sync.literals-span" and found.unit == "literals_f"
    assert "3-entry jump table of g" in found.reason and "short by 0x10 bytes" in found.reason
    assert "jtbl_800E1510 = 0x800E1510; // rom:0x2010" in found.action
    facts["D_800E1510"] = (0x800E1510, 0x2010, None, "data")  # the row that splits g's table off
    assert repo._swallowed_tables(config, tree_) == []
    del facts["D_800E1510"]
    tree_.modules[0].owns["a"] = (*tree_.modules[0].owns["a"], "g")  # g in C in f's own file: one object's .rodata
    assert repo._swallowed_tables(config, tree_) == []
    tree_.modules[0].stand_ins["a"] = ("g",)  # g behind its stand-in there: the object compiles no table for it
    assert [f.unit for f in repo._swallowed_tables(config, tree_)] == ["literals_f"]


# ---------------------------------------------------------------- extraction again after a rename: no splat
RENAME_FACTS = "func_8021A2D4_de = 0x8021A2D4; // type:func rom:0x1AED4\nD_80100000_de = 0x80100000; // type:data\n"


def test_rename_delta_accepts_only_names_of_symbols_that_keep_address_and_attributes():
    new = RENAME_FACTS.replace("func_8021A2D4_de", "Render_Draw")
    assert repo._rename_delta(RENAME_FACTS, new) == {"func_8021A2D4_de": "Render_Draw"}
    assert (
        repo._rename_delta(RENAME_FACTS, "".join(reversed(RENAME_FACTS.splitlines(keepends=True)))) == {}
    )  # only the order moved
    moved = RENAME_FACTS.replace("0x8021A2D4;", "0x8021A2D8;").replace("func_8021A2D4_de", "Render_Draw")
    assert repo._rename_delta(RENAME_FACTS, moved) is None  # a new address
    assert repo._rename_delta(RENAME_FACTS, new.replace("rom:0x1AED4", "rom:0x1AED4 size:0x40")) is None  # an attribute
    assert repo._rename_delta(RENAME_FACTS, RENAME_FACTS + "extra_de = 0x80100010;\n") is None  # an added fact
    assert repo._rename_delta(RENAME_FACTS, RENAME_FACTS.splitlines()[0] + "\n") is None  # a removed fact
    assert (
        repo._rename_delta(RENAME_FACTS, RENAME_FACTS.replace("func_8021A2D4_de", "func_8021A2D4")) is None
    )  # a splat default name
    assert repo._rename_delta(RENAME_FACTS, RENAME_FACTS.replace("func_8021A2D4_de", "bad-name")) is None
    assert repo._rename_delta(RENAME_FACTS, RENAME_FACTS + "// a comment\n") is None  # a line that is not a fact moved


def test_a_rename_of_facts_renames_the_last_extraction_and_runs_no_splat(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path, versions=["us"]))
    _write(tmp_path, "versions/us/symbol_addrs.txt", RENAME_FACTS)
    monkeypatch.setattr(repo.api, "symbols", lambda config, v: {})  # no fact names a leftover file
    monkeypatch.setattr(
        repo,
        "_extract_key",
        lambda c, v, facts=None, yaml=None: (
            "K" + (facts or (tmp_path / "versions/us/symbol_addrs.txt").read_bytes()).decode()
        ),
    )
    monkeypatch.setattr(repo, "_checked", lambda *a: None)
    ran = []

    def splat(c, v):
        ran.append(v)
        _write(tmp_path, "asm/us/nonmatchings/func_8021A2D4_de.s", "glabel func_8021A2D4_de\njr $ra\n")
        _write(tmp_path, "asm/us/nonmatchings/caller.s", "jal func_8021A2D4_de\nnop\n")
        _write(tmp_path, "asm/us/nonmatchings/other.s", "jal func_80000000_de\n")
        _write(tmp_path, "asm/us/data/d.s", "dlabel D_80100000_de\n")
        _write(tmp_path, "asm/us/.extracted", "")
        _write(tmp_path, "build/us/undefined_syms_auto.txt", "func_8021A2D4_de_x = 0;\nD_1 = 1;\n")
        _write(tmp_path, "build/us/splat.ld", "x = func_8021A2D4_de;\n")

    monkeypatch.setattr(repo.build, "extract", splat)
    assert repo.extract_job((config, "us")) is True and ran == ["us"]
    _write(tmp_path, "build/us/src/c.c.s", '.include "nonmatchings/caller.s"\n')
    _write(tmp_path, "build/us/src/c.c.o", "obj")
    _write(tmp_path, "build/us/src/o.c.s", '.include "nonmatchings/other.s"\n')
    _write(tmp_path, "build/us/src/o.c.o", "obj")
    old = 1_000_000_000_000_000_000
    for path in (tmp_path / "asm/us").rglob("*"):
        if path.is_file():
            os.utime(path, ns=(old, old))
    repo._write_asm_manifest(config, "us")
    other_inode = (tmp_path / "asm/us/nonmatchings/other.s").stat().st_ino
    _write(tmp_path, "versions/us/symbol_addrs.txt", RENAME_FACTS.replace("func_8021A2D4_de", "Render_Draw"))
    assert repo.extract_job((config, "us")) is True
    assert ran == ["us"]  # splat did not run again
    asm = tmp_path / "asm/us"
    assert not (asm / "nonmatchings/func_8021A2D4_de.s").exists()
    assert (asm / "nonmatchings/Render_Draw.s").read_text() == "glabel Render_Draw\njr $ra\n"
    assert (asm / "nonmatchings/caller.s").read_text() == "jal Render_Draw\nnop\n"
    assert (asm / "nonmatchings/other.s").stat().st_mtime_ns == old
    assert (asm / "nonmatchings/other.s").stat().st_ino == other_inode
    assert (tmp_path / "build/us/splat.ld").read_text() == "x = Render_Draw;\n"
    assert (tmp_path / "build/us/undefined_syms_auto.txt").read_text() == "func_8021A2D4_de_x = 0;\nD_1 = 1;\n"
    assert not (tmp_path / "build/us/src/c.c.o").exists() and (tmp_path / "build/us/src/o.c.o").exists()
    assert repo._asm_damaged(config, "us") == []
    assert repo._facts_snapshot(config, "us").read_text() == RENAME_FACTS.replace("func_8021A2D4_de", "Render_Draw")
    assert repo.extract_job((config, "us")) is False


def test_a_name_already_in_the_text_makes_the_rename_extraction_run_splat(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path, versions=["us"]))
    _write(tmp_path, "versions/us/symbol_addrs.txt", RENAME_FACTS)
    monkeypatch.setattr(repo.api, "symbols", lambda config, v: {})  # no fact names a leftover file
    monkeypatch.setattr(
        repo,
        "_extract_key",
        lambda c, v, facts=None, yaml=None: (
            "K" + (facts or (tmp_path / "versions/us/symbol_addrs.txt").read_bytes()).decode()
        ),
    )
    monkeypatch.setattr(repo, "_checked", lambda *a: None)
    ran = []

    def splat(c, v):
        ran.append(v)
        _write(tmp_path, "asm/us/a.s", "jal func_8021A2D4_de\njal Render_Draw\n")
        _write(tmp_path, "asm/us/.extracted", "")

    monkeypatch.setattr(repo.build, "extract", splat)
    repo.extract_job((config, "us"))
    _write(tmp_path, "versions/us/symbol_addrs.txt", RENAME_FACTS.replace("func_8021A2D4_de", "Render_Draw"))
    assert repo.extract_job((config, "us")) is True
    assert ran == ["us", "us"]


DROP_FACTS = RENAME_FACTS + "D_80100004_de = 0x80100004; // rom:0x1AED8\n"
YAML_OLD = "segments:\n  - [0x1000, rodata, data_800C/D_old]\n  - [0x1100, c, span_1000]\n"


def test_fact_delta_reports_a_dropped_row_beside_the_renames():
    new = DROP_FACTS.replace("func_8021A2D4_de", "Render_Draw").replace(
        "D_80100004_de = 0x80100004; // rom:0x1AED8\n", ""
    )
    assert repo._fact_delta(DROP_FACTS, new) == ({"func_8021A2D4_de": "Render_Draw"}, {"D_80100004_de"})
    assert repo._fact_delta(RENAME_FACTS, RENAME_FACTS) == ({}, set())
    assert repo._rename_delta(DROP_FACTS, RENAME_FACTS) is None  # a drop is not a pure rename
    assert repo._fact_delta(RENAME_FACTS, DROP_FACTS) is None  # an added fact
    assert repo._fact_delta(DROP_FACTS, RENAME_FACTS.replace("type:data", "type:data size:0x4")) is None  # an attribute


def test_yaml_delta_accepts_only_new_names_of_the_same_pieces():
    assert repo._yaml_delta(YAML_OLD, YAML_OLD.replace("D_old", "literals_f")) == {"D_old": "literals_f"}
    assert repo._yaml_delta(YAML_OLD, YAML_OLD) == {}
    assert repo._yaml_delta(YAML_OLD, YAML_OLD.replace("0x1000", "0x1004")) is None  # a piece moved
    assert repo._yaml_delta(YAML_OLD, YAML_OLD.replace("rodata", "data")) is None  # a piece changed kind
    assert repo._yaml_delta(YAML_OLD, YAML_OLD + "  - [0x1200, c, span_1000]\n") is None  # a piece added
    assert repo._yaml_delta(YAML_OLD, YAML_OLD.replace("D_old", "bad-name")) is None
    assert repo._yaml_delta(YAML_OLD, YAML_OLD.replace("D_old", "span_1000")) is None  # a name another piece has


def _drop_job(tmp_path, monkeypatch, text):
    from types import SimpleNamespace

    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path, versions=["us"]))
    monkeypatch.setattr(repo.api, "symbols", lambda config, v: {})  # no fact names a leftover file
    _write(tmp_path, "versions/us/symbol_addrs.txt", DROP_FACTS)
    _write(tmp_path, "versions/us/splat.yaml", YAML_OLD)
    monkeypatch.setattr(
        repo,
        "_extract_key",
        lambda c, v, facts=None, yaml=None: (
            "K"
            + (facts or (tmp_path / "versions/us/symbol_addrs.txt").read_bytes()).decode()
            + (yaml or (tmp_path / "versions/us/splat.yaml").read_bytes()).decode()
        ),
    )
    monkeypatch.setattr(repo, "_checked", lambda *a: None)
    ran = []

    def splat(c, v):
        ran.append(v)
        _write(tmp_path, "asm/us/data/D_old.s", "dlabel D_80100000_de\n")
        _write(tmp_path, "asm/us/nonmatchings/f.s", text)
        _write(tmp_path, "asm/us/.extracted", "")
        _write(tmp_path, "build/us/undefined_syms_auto.txt", "D_80100004_de = 0x80100004;\nD_1 = 0x1;\n")
        _write(tmp_path, "build/us/splat.ld", "build/us/asm/us/data/D_old.s.o(.rodata);\n")

    monkeypatch.setattr(repo.build, "extract", splat)
    assert repo.extract_job((config, "us")) is True and ran == ["us"]
    assert repo._yaml_snapshot(config, "us").read_text() == YAML_OLD
    _write(tmp_path, "versions/us/symbol_addrs.txt", RENAME_FACTS)
    _write(tmp_path, "versions/us/splat.yaml", YAML_OLD.replace("D_old", "literals_f"))
    assert repo.extract_job((config, "us")) is True
    return ran


def test_a_dropped_fact_no_text_names_and_a_renamed_piece_run_no_splat(tmp_path, monkeypatch):
    ran = _drop_job(tmp_path, monkeypatch, "jal func_8021A2D4_de\n")
    assert ran == ["us"]  # splat did not run again
    assert not (tmp_path / "asm/us/data/D_old.s").exists()
    assert (tmp_path / "asm/us/data/literals_f.s").read_text() == "dlabel D_80100000_de\n"
    assert (tmp_path / "build/us/undefined_syms_auto.txt").read_text() == "D_1 = 0x1;\n"  # the dropped row went
    assert (tmp_path / "build/us/splat.ld").read_text() == "build/us/asm/us/data/literals_f.s.o(.rodata);\n"


def test_a_dropped_fact_the_text_uses_runs_splat(tmp_path, monkeypatch):
    assert _drop_job(tmp_path, monkeypatch, "lw $a0, %lo(D_80100004_de)($a1)\n") == ["us", "us"]


def test_a_current_extraction_without_a_facts_snapshot_gets_one(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path, versions=["us"]))
    _write(tmp_path, "versions/us/symbol_addrs.txt", RENAME_FACTS)
    _write(tmp_path, "asm/us/a.s", "x\n")
    monkeypatch.setattr(repo, "_extract_key", lambda c, v, facts=None, yaml=None: "K")
    repo._write_asm_manifest(config, "us")
    repo.store.write(repo._extract_stamp(config, "us"), b"K")
    assert repo.extract_job((config, "us")) is False
    assert repo._facts_snapshot(config, "us").read_text() == RENAME_FACTS


def test_a_pass_that_deletes_a_file_runs_again_until_a_pass_writes_nothing(world, monkeypatch):
    world.root.joinpath("gone.txt").write_text("x")
    calls = []

    def fake(config, params):
        calls.append(1)
        if len(calls) == 1:
            world.root.joinpath("gone.txt").unlink()
            return {"written": ["gone.txt"], "stand_ins": {}}, None
        return {"written": [], "stand_ins": {}}, None

    monkeypatch.setattr(repo, "_sync", fake)
    out = repo.sync(world.cfg, {})
    assert len(calls) == 2 and out["written"] == ["gone.txt"]


def test_an_extraction_change_reaches_the_files_that_include_it_through_others():
    includes = {
        "data/a.s": set(),
        "data/b.s": {"data/a.s"},  # b includes a
        "nonmatchings/s/f.s": {"b.s"},  # f includes b by its tail
        "nonmatchings/s/g.s": set(),
    }
    reach = repo._including({"data/a.s"}, includes)
    assert {"data/a.s", "data/b.s", "nonmatchings/s/f.s"} <= reach and "nonmatchings/s/g.s" not in reach
    assert repo._including({"nonmatchings/s/g.s"}, includes) >= {"nonmatchings/s/g.s", "g.s"}


def test_sync_fences_a_stand_in_out_of_every_version_that_does_not_own_the_name(world, monkeypatch):
    # a owns d_chunk; b knows the bytes under another name (no stale row, no ownership): only a may emit the pair
    monkeypatch.setattr(repo.api, "symbols", lambda config, v: {**FACTS, "d_chunk": (0x80100000, 0x2000, 8, "data")})
    array = "const unsigned int d_chunk[2] = {\n    0x00000001, 0x00000002,\n};\n"
    _write(world.root, "src/main/keep.c", "int f_keep(void) { return 1; }\n" + array)
    world.cfg.project.versions = ("a", "b", "c")
    world.tree = _tree(
        {"src/main/keep.c": {"a": ("f_keep", "d_chunk"), "b": ("f_keep",), "c": ("f_keep",)}},
        stale={"src/main/keep.c": {"c": ("d_chunk",)}},
        sections={"d_chunk": ".rodata"},
    )
    monkeypatch.setattr(repo, "_PASSES", 0)  # the stub tree never reflects the first pass
    repo.sync(world.cfg, {"check": False})
    text = (world.root / "src/main/keep.c").read_text()
    assert "#if !(defined(VERSION_B) || defined(VERSION_C))\n" + array in text


def test_extraction_removes_the_assembly_an_earlier_name_left_behind(world, monkeypatch):
    seg = world.root / "asm/a/nonmatchings/main"
    _write(
        world.root,
        "asm/a/nonmatchings/main/f_old.s",
        "glabel f_old\n    /* 1000 80000400 27BDFFE8 */  addiu $sp, $sp, -0x18\n",
    )
    _write(
        world.root,
        "asm/a/nonmatchings/main/f_main.s",
        "glabel f_main\n    /* 1000 80000400 27BDFFE8 */  addiu $sp, $sp, -0x18\n",
    )
    _write(
        world.root,
        "asm/a/nonmatchings/main/func_80009999.s",
        "glabel func_80009999\n    /* 9999 80009999 27BDFFE8 */  nop\n",
    )
    monkeypatch.setattr(repo.api, "symbols", lambda config, v: FACTS)
    assert repo._prune_superseded(world.cfg, "a")
    assert sorted(p.name for p in seg.iterdir()) == ["f_main.s", "func_80009999.s"]  # an auto-named function stays
    assert not repo._prune_superseded(world.cfg, "a")


def test_a_split_another_tree_made_is_copied_not_split_again(world):
    """The extraction of one input is kept in the host cache: a fresh tree with the same input copies it (no splat)."""
    world.setup()
    assert sorted(world.extracts) == ["a", "b"]
    key = repo._extract_key(world.cfg, "a")
    saved = repo.store.get(world.cfg, "split", f"extract-v1:{key}", named=True)
    if saved is None:  # the fake splat writes no .extracted mark: write one the way the recipe does and keep it
        (world.root / "asm/a/.extracted").write_text("")
        repo._save_extraction(world.cfg, "a", key)
    import shutil

    shutil.rmtree(world.root / "asm/a")
    assert repo._restore_extraction(world.cfg, "a", key) is True
    assert (world.root / "asm/a/.extracted").is_file()
    assert repo._restore_extraction(world.cfg, "a", "0" * 64) is False  # another input: a miss
