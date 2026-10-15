"""Portable build rendering and mocked native checks; no ROMs or builds."""

import os
import re
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import fixture
import pytest

from unbake import build, native, process, recipes, store
from unbake import config as configuration
from unbake.contracts import (
    Config,
    Finding,
    Host,
    LayoutMap,
    Member,
    Placement,
    Project,
    Recipe,
    Refusal,
    Snapshot,
    UnitSpec,
    Version,
    VersionFiles,
    digest,
)


@pytest.fixture(autouse=True)
def isolated_toolchains(toolchains):
    # conftest reuses TOOLCHAIN_ROW; keep all mutations local to this test.
    toolchains["toolchain"] = deepcopy(toolchains["toolchain"])


@pytest.fixture
def build_snapshot(tmp_path, toolchains, monkeypatch):
    tools = {name: tmp_path / "host-bin" / name for name in fixture.TOOLS}
    host = Host(
        2,
        3,
        1024,
        1024,
        tmp_path / "cache",
        1024,
        tmp_path / "chains",
        tools,
        None,
        2,
        2,
        0.8,
        2,
        ("test", "test@invalid"),
        {},
        "host",
    )
    files = {
        v: VersionFiles(f"roms/{v}.z64", str(i) * 40, f"versions/{v}/Game.yaml", f"versions/{v}/symbols.txt", {})
        for i, v in enumerate(("a", "b"), 1)
    }
    project = Project(
        tmp_path,
        "fixture",
        "fixture",
        "Fixture title",
        ("b", "a"),
        "a",
        "gcc-test",
        {},
        files,
        {v: (f"-DVERSION_{v.upper()}",) for v in files},
        {v: () for v in files},
        {},
        "project",
    )
    cfg = Config(project, host, "config")
    placements = tuple(Placement(v, ".text", 0x1000, 0x1008, 0x80000400) for v in files)
    member = Member("function", "function", "c", "group", placements)
    unit = UnitSpec("src/group.c", "c", "group", (member.name,), "gcc-test", {})
    layout = LayoutMap({}, {member.name: member}, {unit.path: unit}, "layout", (), {})
    versions = {
        v: Version(
            v,
            tmp_path / files[v].baserom,
            "sha",
            files[v].split,
            files[v].symbols,
            {"z": 0x80000408, "a": 0x80000400},
            (),
        )
        for v in files
    }
    snapshot = Snapshot(cfg, "head", layout, versions, {}, "snapshot")
    monkeypatch.setattr(
        recipes,
        "resolve",
        lambda cfg, unit, overrides: Recipe(
            unit.toolchain, ("-E",), ("-O2",), ("-EB",), digest((unit.toolchain, unit.options))
        ),
    )
    monkeypatch.setattr(
        native,
        "sections",
        lambda snap, unit, version: tuple(
            p for n in unit.members for p in snap.layout.members[n].placements if p.version == version
        ),
    )
    return snapshot


@pytest.fixture(autouse=True)
def native_mock(monkeypatch):
    run = Mock(return_value=fixture.native_result())
    monkeypatch.setattr(process, "run", run)
    monkeypatch.setattr(process, "tool", lambda cfg, name: cfg.host.tools[name])
    return run


@pytest.fixture(autouse=True)
def no_compile_group(request, monkeypatch):
    if "make_check" in request.node.name:
        monkeypatch.setattr(build.pool, "map", lambda config, stage, fn, jobs, *rest, **keys: [() for _ in jobs])


@pytest.fixture(autouse=True)
def fragment_cache(monkeypatch):
    values, calls = {}, []

    def cached(cfg, kind, key, produce):
        hit = (kind, key) in values
        calls.append((kind, key, hit))
        if not hit:
            values[kind, key] = produce()
        return values[kind, key]

    monkeypatch.setattr(store, "cached", cached)
    monkeypatch.setattr(
        build.pool,
        "map",
        lambda cfg, name, function, items, key: [
            cached(cfg, name, key(item), lambda item=item: function(item)) for item in items
        ],
    )
    return calls


def _unit(snapshot):
    return next(iter(snapshot.layout.units.values()))


def _with_unit(snapshot, unit):
    return replace(snapshot, layout=replace(snapshot.layout, units={unit.path: unit}))


def _download():
    return {"url": "https://example.invalid/toolchain.tar.gz", "sha256": "a" * 64, "files": ["cc", "as1"]}


def test_makefile_has_no_host_paths(build_snapshot, native_mock):
    text = build.makefile(build_snapshot).decode()
    host = build_snapshot.config.host
    assert str(host.toolchain_root) not in text
    assert str(build_snapshot.config.project.root) not in text
    assert all(str(path) not in text for path in host.tools.values())
    assert "NAME := fixture" in text
    assert "VERSIONS := b a" in text
    assert "Fixture title" in text
    native_mock.assert_not_called()


@pytest.mark.parametrize("kind", ["asm", "c"])
def test_asm_unit_uses_gnu_assembler_var(build_snapshot, kind):
    unit = replace(
        _unit(build_snapshot), path="src/group.s" if kind == "asm" else "src/group.c", kind=kind, toolchain="ido-7.1"
    )
    snapshot = _with_unit(build_snapshot, unit)
    snapshot = replace(
        snapshot, config=replace(snapshot.config, project=replace(snapshot.config.project, toolchain="ido-7.1"))
    )
    text = build.makefile(snapshot).decode()
    if kind == "asm":
        assert "'$(AS)' -EB -Iinclude -Isrc -Iasm/a/include src/group.s -o" in text
        assert "'$(AS_ido_7_1)'" not in text
        assert "$(TOOLCHAIN_ido_7_1)" not in text
    else:
        assert "'$(CC_ido_7_1)'" in text
        assert "$(TOOLCHAIN_ido_7_1)" in text
        assert "'$(AS)' -EB" not in text


@pytest.mark.parametrize("toolchain", ["gcc-test", "ido-7.1"])
def test_a_pinned_tool_is_fetched_into_tools_and_only_its_users_wait_for_it(build_snapshot, toolchains, toolchain):
    gcc = toolchains["toolchain"]["gcc-test"]  # gcc output assembles with n64link, as the gcc toolchains do
    gcc.update(
        emits_asm=True, as_host="mips_as", assemble=["{n64link}", "asn64", "--as", "{as}", "{source}", "-o", "{out}"]
    )
    unit = replace(_unit(build_snapshot), path="src/group.c", kind="c", toolchain=toolchain)
    snapshot = _with_unit(build_snapshot, unit)
    snapshot = replace(
        snapshot, config=replace(snapshot.config, project=replace(snapshot.config.project, toolchain=toolchain))
    )
    text = build.makefile(snapshot).decode()
    row = configuration.load_resource("repo.toml")["make"]["tools"]["N64LINK"]
    if toolchain == "ido-7.1":  # no command of the project runs it: nothing fetches it
        assert "tools/n64link" not in text and "N64LINK := n64link\n" in text
        return
    assert "N64LINK := tools/n64link/n64link\n" in text and "tools/n64link/.stamp:\n" in text
    rule = text[text.index("tools/n64link/.stamp:\n") :].splitlines()[1]
    assert row["download"]["url"] in rule and f"echo '{row['download']['sha256']}  tools/n64link.tar.gz'" in rule
    assert "-C tools/n64link --strip-components=1 n64link-v0.2.0-linux-x86_64/n64link " in rule
    assert f"'{row['pins']['n64link']}  n64link'" in rule
    deps = next(line for line in text.splitlines() if " &: " in line)
    assert "tools/n64link/.stamp" in deps


def test_makefile_tree_and_rom_checks(build_snapshot):
    text = build.makefile(build_snapshot).decode()
    assert "ifeq ($(CHECK),tree)" in text
    assert "else ifeq ($(CHECK),rom)" in text
    assert "check: build/b/fixture.z64 build/a/fixture.z64" in text
    for version, files in build_snapshot.config.project.version_files.items():
        rom = f"build/{version}/fixture.z64"
        assert f"\t$(Q)test -s versions/{version}/report.json" in text
        assert f"\t$(Q)test \"$$(sha1sum < {rom} | cut -d' ' -f1)\" = {files.baserom_sha1}" in text
        assert f"{rom}: {files.baserom} Makefile" in text
        assert f"cp {files.baserom} $@" in text
    assert "for p in $(PATCHES); do dd if=$${p%:*} of=$@ bs=64K oflag=seek_bytes seek=$${p##*:}" in text
    assert "conv=notrunc status=none || exit 1" in text


def test_makefile_dispatches_each_holder_and_reuses_unaffected_units(build_snapshot, monkeypatch):
    one = _unit(build_snapshot)
    two = replace(one, path="src/other.c")
    snapshot = replace(build_snapshot, layout=replace(build_snapshot.layout, units={one.path: one, two.path: two}))
    cache, jobs = {}, []

    def mapped(cfg, name, function, items, key):
        assert name == "build.units"
        result = []
        for item in items:
            assert len(item) == 3 and item[2] in ("a", "b")
            named = key(item)
            if named not in cache:
                jobs.append((item[1].path, item[2]))
                cache[named] = function(item)
            result.append(cache[named])
        return result

    monkeypatch.setattr(build.pool, "map", mapped)
    before = build.makefile(snapshot)
    assert jobs == [(one.path, "a"), (one.path, "b"), (two.path, "a"), (two.path, "b")]
    assert build.makefile(snapshot) == before and len(jobs) == 4
    edited = replace(one, toolchain="ido-7.1")
    changed = replace(snapshot, layout=replace(snapshot.layout, units={one.path: edited, two.path: two}))
    assert build.makefile(changed) != before
    assert jobs[4:] == [(one.path, "a"), (one.path, "b")]


def test_a_landing_renders_again_only_the_rules_of_units_reaching_the_edited_header(build_snapshot, monkeypatch):
    """One fuzzy landing re-keyed the rules of all 22,905 units: a command after it renders the changed set alone."""
    root = build_snapshot.config.project.root
    for path, text in {
        "src/group.c": '#include "a.h"\n',
        "src/a.h": "int a;\n",
        "src/other.c": '#include "b.h"\n',
        "src/b.h": "int b;\n",
    }.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    one = _unit(build_snapshot)
    two = replace(one, path="src/other.c")
    snapshot = replace(build_snapshot, layout=replace(build_snapshot.layout, units={one.path: one, two.path: two}))
    jobs = []

    def mapped(cfg, name, function, items, key):  # no result cache: every item handed over is a rendered rule
        jobs.extend((item[1].path, item[2]) for item in items)
        return [function(item) for item in items]

    monkeypatch.setattr(build.pool, "map", mapped)
    before = build.makefile(snapshot)
    assert len(jobs) == 4
    build.effort._memo.clear()  # the next command
    assert build.makefile(snapshot) == before and len(jobs) == 4
    build.effort._memo.clear()
    (root / "src/b.h").write_text("int b, c;\n")
    build.makefile(snapshot)
    assert jobs[4:] == [(two.path, "a"), (two.path, "b")]


def test_toolchain_rule_pins(build_snapshot, toolchains):
    row = toolchains["toolchain"]["gcc-test"]
    download = _download()
    row.update(downloads=[download], pins={"z-file": "b" * 64, "a-file": "c" * 64})
    text = build.makefile(build_snapshot).decode()
    assert "CC_gcc_test ?= tools/gcc-test/cc" in text
    assert "ifneq ($(filter default undefined,$(origin CPP)),)\nCPP := cpp\nendif" in text
    assert "AS_gcc_test ?= $(AS)" in text
    assert "TOOLCHAIN_gcc_test ?= tools/gcc-test/.stamp" in text
    assert "tools/gcc-test/.stamp:" in text
    assert f"curl -fsSL {download['url']} -o tools/gcc-test.tar.gz" in text
    assert f"echo '{download['sha256']}  tools/gcc-test.tar.gz'" in text
    assert "tar -xzf tools/gcc-test.tar.gz -C tools/gcc-test cc as1" in text
    assert "'" + "c" * 64 + "  a-file' '" + "b" * 64 + "  z-file'" in text
    assert "cd tools/gcc-test" in text
    assert "touch .stamp" in text


def test_no_download_requires_local_cc(build_snapshot):
    text = build.makefile(build_snapshot).decode()
    assert "CC_gcc_test ?= $(error CC_gcc_test must be set in local.mk)" in text
    assert "TOOLCHAIN_gcc_test ?=\n" in text
    assert "curl -fsSL" not in text


@pytest.mark.parametrize("id", ["gcc-test", "ido-7.1"])
def test_two_downloads_refuse(build_snapshot, toolchains, id):
    toolchains["toolchain"][id]["downloads"] = [_download(), _download()]
    snapshot = _with_unit(build_snapshot, replace(_unit(build_snapshot), toolchain=id))
    with pytest.raises(Refusal) as caught:
        build.makefile(snapshot)
    finding = caught.value.findings[0]
    assert finding.key == "toolchain.download"
    assert finding.reason == f"{id} has 2 downloads; the generated build supports one"
    assert finding.path == id


def test_local_mk_host_paths(build_snapshot, toolchains):
    cfg = build_snapshot.config
    text = build.local_mk(cfg).decode()
    tools = configuration.load_resource("repo.toml")["make"]["tools"]
    for var, row in tools.items():  # a pinned tool is never a machine path: the Makefile fetches it
        assert (f"{var} := {cfg.host.tools[row['host']]}" in text) == ("download" not in row)
        assert ("download" not in row) or f"{var} :=" not in text
    for id, row in toolchains["toolchain"].items():
        key = id.replace("-", "_").replace(".", "_")
        assert f"CC_{key} := {cfg.host.toolchain_root / id / row['cc']}" in text
        expected = str(cfg.host.toolchain_root / id / row["as"]) if "as" in row else "$(AS)"
        assert f"AS_{key} := {expected}" in text
        assert f"TOOLCHAIN_{key} :=\n" in text


@pytest.mark.parametrize("exit", [1, 2, None])
def test_make_check_refuses_on_exit(build_snapshot, native_mock, exit):
    lines = [f"diagnostic {i}" for i in range(25)]
    native_mock.return_value = fixture.native_result(exit=exit, stderr="\n".join(lines).encode())
    with pytest.raises(Refusal) as caught:
        build.make_check(build_snapshot.config, build_snapshot)
    finding = caught.value.findings[0]
    assert finding.key == "setup.make"
    assert finding.reason == "\n".join(lines[-20:])
    assert finding.missing == (f"make exit {exit}",)


def test_make_check_success(build_snapshot, native_mock):
    cfg = build_snapshot.config
    assert build.make_check(cfg, build_snapshot) == {"exit": 0, "versions": ["b", "a"]}
    make = str(cfg.host.tools["make"])
    (name, argv, cwd), keys = native_mock.call_args
    assert (name, argv, cwd, keys["tmp"]) == (
        "make check",
        [make, "-k", "check"],
        cfg.project.root,
        cwd / ".unbake/tmp",
    )
    assert native_mock.call_count == 1 and keys["env"]["MAKEFLAGS"].startswith("-j3")


def test_make_check_is_skipped_while_its_inputs_are_unchanged(build_snapshot, native_mock):
    cfg = build_snapshot.config
    build.make_check(cfg, build_snapshot)
    build.make_check(cfg, build_snapshot)
    native_mock.assert_called_once()  # the second check found the same inputs
    (cfg.project.root / "Makefile").write_text("changed\n")
    build.make_check(cfg, build_snapshot)
    assert native_mock.call_count == 2


def test_rule_outputs_live_in_a_directory_named_by_the_digest_of_the_rule(build_snapshot, monkeypatch, toolchains):
    """A rule has no Makefile dependency: a changed rule has other outputs, an unchanged one keeps its own."""

    def tags(snapshot):
        return re.findall(r"build/[ab]/src/group/(\w{12})/text\.bin", build.makefile(snapshot).decode())

    first = tags(build_snapshot)
    assert len(set(first)) == 2 and tags(build_snapshot) == first  # one per version, the same on every run
    toolchains["toolchain"]["gcc-test"]["compile"].append("-changed")
    build.effort.forget("unit-rules")  # changed installed instructions start with a fresh process memo
    assert not set(tags(build_snapshot)) & set(first)


def test_a_rule_does_not_depend_on_the_host_or_the_project_root(build_snapshot):
    other_host = replace(build_snapshot.config.host, workers=7, digest="another host")
    moved = replace(build_snapshot, config=replace(build_snapshot.config, host=other_host, digest="another config"))
    assert build.makefile(moved) == build.makefile(build_snapshot)


def test_unit_fragment_placeholders_and_dependencies(build_snapshot):
    text = build.makefile(build_snapshot).decode()
    assert "# src/group.c" not in text  # no comment line per unit rule (14k lines in a large project)
    assert text.index("build/a/src/group/") < text.index("build/b/src/group/")
    for v in ("a", "b"):
        assert re.search(rf"build/{v}/src/group/\w{{12}}/text\.bin &: src/group.c versions/{v}/symbols.ld", text)
        rule = re.search(rf"build/{v}/src/group/\w{{12}}/text\.bin &:[^\n]*", text)[0]
        assert rule.endswith(f"versions/{v}/symbols.ld $(TOOLCHAIN_gcc_test)")
        assert f"-Iinclude -Isrc -Iasm/{v}/include" in text
        assert re.search(rf"build/{v}/fixture.z64: PATCHES \+= build/{v}/src/group/\w{{12}}/text.bin:4096", text)
    assert "'$(LD)'" in text and "n64link place" not in text.lower() and " place " not in text and "SUBALIGN(4)" in text
    assert "set-section-flags" not in text and "normal.o" not in text and "placed.o" not in text
    assert all("--rom" not in line and "--map" not in line for line in text.splitlines())


def test_a_unit_depends_on_the_headers_it_can_reach_and_on_no_others(build_snapshot):
    root = build_snapshot.config.project.root
    for path, text in {
        "src/group.c": '#include "a.h"\n',
        "include/a.h": '#include "common/b.h"\n',
        "include/common/b.h": "",
        "include/other.h": "",
    }.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    rule = re.search(r"build/a/src/group/\w{12}/text\.bin &:[^\n]*", build.makefile(build_snapshot).decode())[0]
    assert "src/group.c versions/a/symbols.ld include/a.h include/common/b.h $(TOOLCHAIN_gcc_test)" in rule
    planned = {"include/other.h": b'#include "new.h"\n', "include/new.h": b""}
    unrelated = replace(build_snapshot, overlays=planned, digest="unrelated")
    assert build.makefile(unrelated) == build.makefile(build_snapshot)  # a header it never includes changes nothing
    including = replace(
        build_snapshot, digest="including", overlays={**planned, "src/group.c": b'#include "a.h"\n#include "other.h"\n'}
    )
    assert "include/common/b.h include/new.h include/other.h" in build.makefile(including).decode()


def test_headers_follow_the_search_flags_of_the_compile_line(build_snapshot, monkeypatch):
    root = build_snapshot.config.project.root
    files = {
        "src/group.c": '#include "deep.h"\n#include "nowhere.h"\n',
        "extra/deep.h": "#include <sys.h>\n",
        "sys/sys.h": "",
        "forced/pre.h": "",
        "asm/a/include/macro.inc": "",
    }
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    monkeypatch.setattr(
        recipes,
        "resolve",
        lambda cfg, unit, overrides: Recipe(
            unit.toolchain, ("-I", "extra", "-isystem", "sys", "-include", "forced/pre.h"), ("-O2",), ("-EB",), "r"
        ),
    )
    rule = re.search(r"build/a/src/group/\w{12}/text\.bin &:[^\n]*", build.makefile(build_snapshot).decode())[0]
    for header in ("extra/deep.h", "sys/sys.h", "forced/pre.h"):  # reached only through a recipe flag
        assert f" {header} " in rule
    covered = re.search(r"\$\(wildcard ([^)]*)\)", rule)[1].split()  # an unfound quoted include is never dropped
    assert {"src/nowhere.h", "extra/nowhere.h", "include/*.h", "include/*/*.h"} <= set(covered)


def test_symbols_ld_sorted(build_snapshot):
    assert build.symbols_ld(build_snapshot, "a") == b"PROVIDE(a = 0x80000400);\nPROVIDE(z = 0x80000408);\n"


@pytest.mark.parametrize("exit", [0, 1])
def test_extract_mocked(build_snapshot, native_mock, exit):
    split = build_snapshot.config.project.root / build_snapshot.config.project.version_files["a"].split
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("options: {basename: rom}\nsegments: []\n")
    cfg = build_snapshot.config
    native_mock.return_value = fixture.native_result(exit=exit, stderr=b"splat failed")
    if exit:
        with pytest.raises(Refusal) as caught:
            build.extract(cfg, "a")
        assert caught.value.findings[0].key == "setup.splat"
        assert caught.value.findings[0].versions == ("a",)
    else:
        assert build.extract(cfg, "a") is native_mock.return_value
    argv = native_mock.call_args.args[1]
    assert argv[:2] == [cfg.host.tools["splat"], "split"]
    assert argv[3:] == ["--make-full-disasm-for-code"]
    assert argv[2] != str(split)
    assert native_mock.call_args.args[2] == cfg.project.root


def test_extract_with_unchanged_inputs_never_splits_again(build_snapshot, native_mock, monkeypatch):
    held = {}
    monkeypatch.setattr(store, "get", lambda cfg, kind, key: held.get((kind, key)))
    monkeypatch.setattr(store, "put", lambda cfg, kind, key, value: held.__setitem__((kind, key), value))
    cfg = build_snapshot.config
    split = cfg.project.root / cfg.project.version_files["a"].split
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("options: {basename: rom}\nsegments: []\n")
    native_mock.return_value = fixture.native_result(exit=0)
    (cfg.project.root / ".unbake/symbols/a").mkdir(parents=True, exist_ok=True)
    build.extract(cfg, "a")
    (cfg.project.root / ".unbake/symbols/a/x.csv").write_text("x")  # the facts a split leaves
    build.extract(cfg, "a")
    build.extract(cfg, "a")
    assert native_mock.call_count == 1
    split.write_text("options: {basename: rom}\nsegments: [{name: m, type: code, start: 0}]\n")
    before = native_mock.call_count
    build.extract(cfg, "a")
    assert native_mock.call_count == before + 1


def test_split_input_bounds_long_output_names_and_preserves_logical_input():
    import yaml

    long_name = "rodata/" + "unclaimed_" * 40
    original = {
        "options": {"base_path": "../.."},
        "segments": [
            {
                "name": "m",
                "type": "code",
                "start": 0,
                "subsegments": [[0, "data", long_name], {"start": 4, "type": "data", "name": long_name + "b"}, [8]],
            },
            [8],
        ],
    }
    source = yaml.safe_dump(original)
    document = build._split_input(source)
    rows = document["segments"][0]["subsegments"]
    assert all(len(name.split("/")[-1].encode()) <= 240 for name in [rows[0][2], rows[1]["name"]])
    assert rows[0][2] != rows[1]["name"]
    assert build._split_input(source) == document
    assert yaml.safe_load(source) == original
    assert build._split_input(yaml.safe_dump(document)) == document


def test_split_input_preserves_normal_config():
    assert build._split_input("options: {basename: rom}\nsegments:\n - [0]\n") == {
        "options": {"basename": "rom"},
        "segments": [[0]],
    }


def test_extract_long_names_uses_private_config(build_snapshot, native_mock):
    import yaml

    cfg = build_snapshot.config
    split = cfg.project.root / cfg.project.version_files["a"].split
    split.parent.mkdir(parents=True, exist_ok=True)
    original = yaml.safe_dump(
        {
            "options": {"base_path": "../.."},
            "segments": [
                {"name": "main", "type": "code", "start": 0, "subsegments": [[0, "data", "d" * 300], [4]]},
                [4],
            ],
        }
    )
    split.write_text(original)
    private = []

    def invoke(name, argv, cwd, **kw):
        path = Path(argv[2])
        private.append(path)
        document = yaml.safe_load(path.read_text())
        assert path != split and document["options"]["base_path"] == str(path.parent)
        assert len(document["segments"][0]["subsegments"][0][2]) <= 240
        return fixture.native_result()

    native_mock.side_effect = invoke
    assert build.extract(cfg, "a").exit == 0
    assert split.read_text() == original and not private[0].exists()


def test_extract_isolates_dump_and_preserves_path_destinations(build_snapshot, native_mock):
    import yaml

    cfg = build_snapshot.config
    root = cfg.project.root
    split = root / cfg.project.version_files["a"].split
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text(
        yaml.safe_dump(
            {
                "options": {
                    "base_path": "../..",
                    "asm_path": "asm/a",
                    "symbol_addrs_path": ["symbols/a.txt"],
                    "target_path": "roms/a.z64",
                },
                "segments": [],
            }
        )
    )
    seen = []

    def invoke(name, argv, cwd, **kw):
        path = Path(argv[2])
        options = yaml.safe_load(path.read_text())["options"]
        assert options["base_path"] == str(path.parent)
        assert options["asm_path"] == str(path.parent / "tree" / "asm/a")  # written aside, published when it differs
        assert options["symbol_addrs_path"] == [str(root / "symbols/a.txt")]
        assert options["target_path"] == str(root / "roms/a.z64")
        assert options["dump_symbols"] is True
        dump = path.parent / ".splat"
        dump.mkdir()
        (dump / "spim_context_unksegment.csv").write_text("category,address,getName\nsymbol,0x80000800,target\n")
        seen.append(path.parent)
        return fixture.native_result()

    native_mock.side_effect = invoke
    build.extract(cfg, "a")
    assert not seen[0].exists()
    assert (root / ".unbake/symbols/a/spim_context_unksegment.csv").read_text().endswith("target\n")
    assert not (root / ".splat").exists()


def test_symbols_ld_provides_code_at_the_masked_address_only_when_the_project_says_so(build_snapshot, monkeypatch):
    snapshot = build_snapshot
    record = replace(
        snapshot.versions["a"],
        symbols={"call": 0x802BB550, "data": 0x800D8A24, "label": 0x802BB700},
        code=frozenset({"label"}),
    )
    members = {
        "f": SimpleNamespace(
            placements=(
                Placement("a", ".text", 0, 0x100, 0x802BB500),
                Placement("a", ".text", 0x100, 0x200, 0x802BB600),
            )
        )
    }
    project = replace(snapshot.config.project, build={**snapshot.config.project.build, "text_mask": 0x1FFFFFFF})
    masked = replace(
        snapshot,
        layout=SimpleNamespace(members=members),
        versions={"a": record},
        digest="masked",
        config=replace(snapshot.config, project=project),
    )
    monkeypatch.setattr(build.symbols, "load", lambda reader, versions, cache: {"call": {"kind": "function"}})
    text = build.symbols_ld(masked, "a").decode()
    assert "PROVIDE(call = 0x002BB550);" in text and "PROVIDE(label = 0x002BB700);" in text
    assert "PROVIDE(data = 0x800D8A24);" in text  # data keeps the address it has
    assert build._text_spans(masked, "a") == [(-1, 0), (0x802BB500, 0x802BB700)]  # runs that touch are one run
    unmasked = replace(masked, config=snapshot.config)
    assert "PROVIDE(call = 0x802BB550);" in build.symbols_ld(unmasked, "a").decode()


def test_make_check_reports_every_failed_target(build_snapshot, native_mock):
    lines = ["make: *** [a.o] Error 1", "noise", "make: *** [b.o] Error 2", "make: Target 'check' not remade"]
    native_mock.return_value = fixture.native_result(exit=2, stderr="\n".join(lines).encode())
    with pytest.raises(Refusal) as caught:
        build.make_check(build_snapshot.config, build_snapshot)
    assert caught.value.findings[0].reason == "\n".join([lines[0], lines[2]])


def test_local_mk_carries_the_hosts_worker_count(build_snapshot):
    cfg = build_snapshot.config
    assert f"JOBS := {cfg.host.workers}\n" in build.local_mk(cfg).decode()


def test_makefile_runs_parallel_from_jobs_and_a_given_j_wins(build_snapshot):
    text = build.makefile(build_snapshot).decode()
    assert (
        "-include local.mk\nifneq ($(JOBS),)\nifeq ($(filter -j% --jobserver%,$(MAKEFLAGS)),)\n"
        "MAKEFLAGS += -j$(JOBS)\nendif\nendif\n"
    ) in text


def test_make_uses_jobs_unless_j_is_given(tmp_path):
    head = configuration.template("Makefile.in").split("${tool_defaults}")[0]
    head = head.replace("${title}", "t").replace("${name}", "n").replace("${versions}", "a").replace("$$", "$")
    (tmp_path / "Makefile").write_text(head + "\nshow:\n\t@echo $(filter -j%,$(MAKEFLAGS))\n")
    (tmp_path / "local.mk").write_text("JOBS := 3\n")

    def flags(*args):
        return subprocess.run(["make", "-s", *args], cwd=tmp_path, capture_output=True, text=True).stdout.split()

    assert flags() == ["-j3"] and flags("-j2") == ["-j2"]


def test_a_masked_jump_table_word_names_the_label_of_its_target(tmp_path):
    table = (
        "dlabel jtbl_800DD1F8\n    /* E1E28 800E1228 00414E54 */ .word 0x00414E54\n"
        "    /* E1E2C 800E122C 0000000C */ .word 0x0000000C\nenddlabel jtbl_800DD1F8\n"
        "dlabel D_800E1254\n    /* E1E54 800E1254 00414E98 */ .word 0x00414E98\nenddlabel D_800E1254\n"
    )
    (tmp_path / "t.s").write_text(table)
    segments = [{"start": 0x1000, "vram": 0x80200400}, {"start": 0x400000}, [0x500000]]
    build._label_tables(tmp_path, segments, 0x1FFFFFFF)
    text = (tmp_path / "t.s").read_text()
    assert ".word .Lauto_80414E54" in text and ".word 0x0000000C" in text and ".word 0x00414E98" in text


def _compile_item(build_snapshot, monkeypatch, found=(), run=None, version="a"):
    """compile_unit with the compile mocked to leave an object and the rule's own link script mocked."""
    root = build_snapshot.config.project.root
    for path in ("src/group.c", f"versions/{version}/symbols.ld"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("x")
    monkeypatch.setattr(native, "compiles", lambda item, work: (work / "group.o").write_bytes(b"obj") and found)
    ran = Mock(side_effect=run or (lambda *a, **k: SimpleNamespace(exit=0, stderr=b"")))
    monkeypatch.setattr(build.process, "run", ran)
    return (build_snapshot, build_snapshot.layout.units["src/group.c"], version), ran


def test_the_rule_names_its_own_directory_and_a_link_tail_without_the_compile(build_snapshot):
    text, tag, outputs, dependencies, commands = build._rule(
        build_snapshot, build_snapshot.layout.units["src/group.c"], "a"
    )
    assert text in build.makefile(build_snapshot).decode()  # the Makefile is these very rules
    assert outputs == [f"build/a/src/group/{tag}/text.bin"] and "src/group.c" in dependencies
    assert not any("$(TOOLCHAIN" in d for d in dependencies)
    assert commands[0].startswith("find ") and tag in commands[0] and "-O2" not in " ".join(commands)
    assert any("unit.ld" in c for c in commands) and any("$(LD)" in c for c in commands)


def test_check_leaves_the_rule_outputs_by_running_the_rules_link_tail_once(build_snapshot, monkeypatch):
    item, ran = _compile_item(build_snapshot, monkeypatch)
    assert build.compile_unit(item) == ()
    (call,) = ran.call_args_list
    script = call.args[1][2]
    _, tag, outputs, _, commands = build._rule(*item)
    assert call.args[1][:2] == ["/bin/sh", "-c"] and "$$" not in script and "$(LD)" not in script
    assert str(build_snapshot.config.host.tools["mips_ld"]) in script and f"{tag}/unit.ld" in script
    assert "-O2" not in script and len(script.split(" && ")) == len(
        commands
    )  # the recipe's own link commands, no compile
    home = build_snapshot.config.project.root / Path(outputs[0]).parent
    assert (home / "source.o").read_bytes() == b"obj"


def test_a_second_check_with_nothing_changed_materializes_nothing(build_snapshot, monkeypatch):
    def link(name, argv, cwd, tmp):
        for output in build._rule(*item)[2]:
            (cwd / output).parent.mkdir(parents=True, exist_ok=True)
            (cwd / output).write_bytes(b"bin")
        return SimpleNamespace(exit=0, stderr=b"")

    item, ran = _compile_item(build_snapshot, monkeypatch, run=link)
    for _ in range(3):
        build.compile_unit(item)
    assert ran.call_count == 1  # then the outputs are newer than every dependency
    root = build_snapshot.config.project.root
    (root / "src/group.c").write_text("edited")
    stamp = (root / build._rule(*item)[2][0]).stat().st_mtime_ns + 10**9
    os.utime(root / "src/group.c", ns=(stamp, stamp))
    build.compile_unit(item)
    assert ran.call_count == 2


def test_a_unit_whose_dependencies_are_all_gone_from_disk_still_materializes(build_snapshot, monkeypatch):
    item, ran = _compile_item(build_snapshot, monkeypatch)
    root = build_snapshot.config.project.root
    for dependency in build._rule(*item)[3]:  # deleted by a peer while the object was being built
        (root / dependency).unlink(missing_ok=True)
    assert build.compile_unit(item) == ()
    ran.assert_called_once()


@pytest.mark.parametrize("case", ["findings", "withheld"])
def test_nothing_is_materialized_for_a_unit_that_fails_or_is_withheld_in_the_version(build_snapshot, monkeypatch, case):
    found = (Finding("compile.error", "bad"),) if case == "findings" else ()
    item, ran = _compile_item(build_snapshot, monkeypatch, found=found)
    if case == "withheld":
        item = (build_snapshot, replace(item[1], withheld=("a",)), "a")
    assert build.compile_unit(item) == found
    ran.assert_not_called()


def test_a_failing_link_tail_is_a_finding_of_the_unit_and_version(build_snapshot, monkeypatch):
    item, _ = _compile_item(
        build_snapshot, monkeypatch, run=lambda *a, **k: SimpleNamespace(exit=1, stderr=b"ld: boom")
    )
    (finding,) = build.compile_unit(item)
    assert (finding.key, finding.unit, finding.versions) == (
        "link.error",
        "src/group.c",
        ("a",),
    ) and "boom" in finding.reason


def test_an_edit_made_during_the_compile_leaves_the_outputs_older_so_make_rebuilds(build_snapshot, monkeypatch):
    def link(name, argv, cwd, tmp):
        for output in build._rule(*item)[2]:
            (cwd / output).parent.mkdir(parents=True, exist_ok=True)
            (cwd / output).write_bytes(b"bin")
        return SimpleNamespace(exit=0, stderr=b"")

    item, ran = _compile_item(build_snapshot, monkeypatch, run=link)
    root = build_snapshot.config.project.root

    def compile_while_a_peer_edits(item, work):
        (root / "src/group.c").write_text("peer edit")  # lands after `begun`, before the outputs are written
        (work / "group.o").write_bytes(b"obj")
        return ()

    monkeypatch.setattr(native, "compiles", compile_while_a_peer_edits)
    build.compile_unit(item)
    output = root / build._rule(*item)[2][0]
    assert output.stat().st_mtime_ns < (root / "src/group.c").stat().st_mtime_ns  # make sees the edit as newer
    build.compile_unit(item)
    assert ran.call_count == 2  # and the next check materializes again


def test_make_check_runs_on_the_elastic_jobserver_not_a_fixed_j(build_snapshot, native_mock, monkeypatch):
    cfg = build_snapshot.config
    seen = []
    env = {"UNBAKE_CORES": "2", "MAKEFLAGS": "-j2 --jobserver-auth=7,8"}

    @contextmanager
    def held(host, work=0):
        seen.append(host)
        yield 2, env, (7, 8)

    monkeypatch.setattr(build.pool, "jobserver", held)
    build.make_check(cfg, build_snapshot)
    argv = [str(a) for a in native_mock.call_args.args[1]]
    assert seen == [cfg.host] and not any(a.startswith("-j") for a in argv)  # make -jN would ignore the jobserver
    assert native_mock.call_args.kwargs["env"] == env and native_mock.call_args.kwargs["pass_fds"] == (7, 8)


def test_extract_stamp_is_per_root_and_covers_every_read_path(build_snapshot, native_mock):
    cfg = build_snapshot.config
    root = cfg.project.root
    split = root / cfg.project.version_files["a"].split
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text("options: {basename: rom, extensions_path: ext}\nsegments: []\n")
    (root / "ext").mkdir(exist_ok=True)
    (root / "ext/x.py").write_text("one")
    native_mock.return_value = fixture.native_result(exit=0)
    (root / ".unbake/symbols/a").mkdir(parents=True, exist_ok=True)
    build.extract(cfg, "a")
    assert (root / ".unbake/extract/a.key").is_file()
    (root / ".unbake/symbols/a/x.csv").write_text("x")
    build.extract(cfg, "a")
    assert native_mock.call_count == 1
    (root / "ext/x.py").write_text("two")  # a changed splat extension re-splits
    (root / ".unbake/symbols/a/x.csv").write_text("x")
    build.extract(cfg, "a")
    assert native_mock.call_count == 2


def test_make_check_compiles_every_job_once_then_runs_make(build_snapshot, native_mock, monkeypatch):
    cfg, order = build_snapshot.config, []
    jobs = [(build_snapshot, "u", "a"), (build_snapshot, "u", "b")]
    monkeypatch.setattr(build, "compile_jobs", lambda snap, paths: jobs)

    def mapped(config, stage, fn, given):
        order.append(("pool", stage, fn, list(given)))
        return [()] * len(jobs)

    monkeypatch.setattr(build.pool, "map", mapped)
    native_mock.side_effect = lambda *a, **k: order.append("make") or fixture.native_result()
    build.make_check(cfg, build_snapshot)
    assert order == [("pool", "land.compile", build.compile_unit, jobs), "make"]


def test_make_check_refuses_on_a_blocking_compile_finding_without_running_make(
    build_snapshot, native_mock, monkeypatch
):
    bad = Finding("source.x", "broken", unit="u", blocking=True)
    soft = Finding("source.y", "soft", unit="v", blocking=False)
    monkeypatch.setattr(build.pool, "map", lambda *a: [(soft, bad)])
    with pytest.raises(Refusal) as caught:
        build.make_check(build_snapshot.config, build_snapshot)
    assert [f.key for f in caught.value.findings] == ["setup.make"]
    native_mock.assert_not_called()


def test_make_check_stamp_hit_runs_neither_compiles_nor_make(build_snapshot, native_mock, monkeypatch):
    build.make_check(build_snapshot.config, build_snapshot)
    ran = Mock(return_value=[])
    monkeypatch.setattr(build.pool, "map", ran)
    native_mock.reset_mock()
    build.make_check(build_snapshot.config, build_snapshot)
    ran.assert_not_called()
    native_mock.assert_not_called()


def test_compile_jobs_lists_every_held_version_of_each_cacheable_decompiled_unit(build_snapshot):
    snap = build_snapshot
    jobs = build.compile_jobs(snap, list(snap.layout.units))
    assert all(item[0] is snap and item[1].path in snap.layout.units for item in jobs)
    assert build.compile_jobs(snap, []) == []


def test_check_compiles_with_exact_reuse_and_leaves_the_flag_down(monkeypatch):
    seen = []
    monkeypatch.setattr(build, "compile_unit", lambda item: seen.append(build.native._EXACT.get()) or ())
    assert build.check_unit(("snapshot", "unit", "a")) == ()
    assert seen == [True] and build.native._EXACT.get() is False


def test_toolchains_install_in_one_pool_dispatch(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(
        build.pool,
        "map",
        lambda config, name, function, items, key=None: calls.append((name, items)) or [Path(i[1]) for i in items],
    )
    assert build.install_toolchains(cfg, ["x", "y"]) == [Path("x"), Path("y")]
    assert calls == [("build.install_toolchains", [(cfg, "x"), (cfg, "y")])]


def test_the_rule_text_is_pinned_so_a_change_forces_the_rule_format_bump(build_snapshot):
    """The unit-rule cache key names inputs, not the tool code: change how _rule writes a rule and bump _RULE_FORMAT."""
    text, _tag, outputs, dependencies, commands = build._rule(
        build_snapshot, build_snapshot.layout.units["src/group.c"], "a"
    )
    assert build._RULE_FORMAT == 2
    assert (
        digest([text, outputs, dependencies, commands])
        == "efee2ffe43cf38c895e7855b938526c005956127e574bd7ca8ee354ce2e91d84"
    )


def test_a_check_key_names_the_inputs_and_not_the_tool_code(build_snapshot, monkeypatch, toolchains):
    item = (build_snapshot, build_snapshot.layout.units["src/group.c"], "a")
    monkeypatch.setattr(native.adapters, "host_tools", lambda cfg: {})
    monkeypatch.setattr(native.adapters, "tool_identity", lambda cfg, id: "cc-1")
    first = build.check_key(item)
    monkeypatch.setattr(native, "code_identity", lambda *f: "another install")
    assert build.check_key(item) == first and build.check_key.stable
    assert native.stamp(*item) != first  # the full stamp still names the code


def test_check_findings_are_pinned_so_a_change_forces_the_check_format_bump(build_snapshot, monkeypatch):
    """The check key names inputs, not the tool code: change what a check reports and bump build._CHECK_FORMAT."""
    seen = []
    failing, _ = _compile_item(build_snapshot, monkeypatch, found=(Finding("compile.error", "bad"),))
    seen.append(build.check_unit(failing))
    broken, _ = _compile_item(
        build_snapshot, monkeypatch, run=lambda *a, **k: SimpleNamespace(exit=1, stderr=b"ld: boom")
    )
    seen.append(build.check_unit(broken))
    clean, _ = _compile_item(build_snapshot, monkeypatch)
    seen.append(build.check_unit(clean))
    assert build._CHECK_FORMAT == 1
    assert digest(seen) == "8395a0f4ad2a27dc5585b9d00d47f10ee66d77bf3304f9030e30a3ab81c8c182"


def test_a_header_edit_moves_the_check_key_of_only_the_consumers_whose_preprocessed_text_changed(
    build_snapshot, monkeypatch
):
    """The scoped check of a header dispatches the units whose key is not warm: a consumer whose preprocessed text is
    the same keeps its key, so it compiles nothing."""
    monkeypatch.setattr(native.adapters, "host_tools", lambda cfg: {})
    monkeypatch.setattr(native.adapters, "tool_identity", lambda cfg, id: "cc-1")
    units = [replace(build_snapshot.layout.units["src/group.c"], path=f"src/u{i}.c") for i in range(20)]
    preprocessed = {u.path: f"text of {u.path}" for u in units}
    monkeypatch.setattr(native.view, "reads", lambda snap, unit, version, recipe, whole=False: preprocessed[unit.path])
    warm = {build.check_key((build_snapshot, u, "a")) for u in units}
    preprocessed["src/u3.c"] = "text of u3 after the header edit"
    preprocessed["src/u7.c"] = "text of u7 after the header edit"
    cold = [u.path for u in units if build.check_key((build_snapshot, u, "a")) not in warm]
    assert cold == ["src/u3.c", "src/u7.c"]
