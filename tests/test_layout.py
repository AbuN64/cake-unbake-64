"""Layout regressions with mocked process calls and memory-only ROM material."""

from __future__ import annotations

import re
import tomllib
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import fixture
import pytest
import yaml

from unbake import layout
from unbake.contracts import Claim, Group, LayoutMap, Placement, Refusal, Snapshot, UnitSpec, Version, digest, sha256

BASE = 0x80000400
RETURN = (0x03E00008, 0)


@pytest.fixture
def scene(tmp_path, monkeypatch):
    """Create contract records and YAML files, without fixture.project's git or ROMs."""
    monkeypatch.setattr(layout.effort, "stage", lambda name: nullcontext())
    monkeypatch.setattr(layout.process, "git", Mock(return_value=fixture.native_result(stdout=b"head\n")))
    config = fixture.bare_config(tmp_path, memory=1 << 30, toolchain_root=tmp_path / "toolchains")
    blobs = {}

    def rows(version, reader, cache=None):
        subs = yaml.safe_load(reader(version.split))["segments"][0]["subsegments"]
        return [
            (
                row[2],
                row[1],
                Placement(
                    version.id,
                    ".text" if row[1] in ("asm", "c") else ".data",
                    row[0],
                    subs[i + 1][0],
                    BASE + row[0] - 0x1000,
                ),
            )
            for i, row in enumerate(subs[:-1])
        ]

    monkeypatch.setattr(layout.version_data, "rows", rows)
    monkeypatch.setattr(
        layout.version_data, "rom_bytes", lambda version, start, end: blobs[version.id][start - 0x1000 : end - 0x1000]
    )

    def make(
        entries=(("func_80000400", "asm", RETURN),),
        *,
        other=None,
        third=None,
        units=(),
        authored=(),
        fuzzy=None,
        grouped=True,
    ):
        versions, named = {}, {}
        held = (("a", entries), ("b", entries if other is None else other), *((("c", third),) if third else ()))
        for vid, items in held:
            subs, blob = [], b""
            for name, state, words in items:
                at = 0x1000 + len(blob)
                subs.append(f"      - [0x{at:X}, {state}, {name}]\n")
                named.setdefault(name, {"kind": "function"})[vid] = BASE + len(blob)
                blob += b"".join(word.to_bytes(4, "big") for word in words)
            subs.append(f"      - [0x{0x1000 + len(blob):X}]\n")
            split = f"versions/{vid}/Game.yaml"
            sym = f"versions/{vid}/symbol_addrs.txt"
            (tmp_path / split).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / split).write_text("segments:\n  - name: main\n    subsegments:\n" + "".join(subs))
            (tmp_path / sym).write_bytes(layout.symbols.render(named, vid))
            blobs[vid] = blob
            versions[vid] = Version(
                vid,
                tmp_path / f"unused-{vid}",
                sha256(blob),
                split,
                sym,
                {},
                (("main", 0x1000, 0x1000 + len(blob), BASE),),
            )
        (tmp_path / "symbols.toml").write_bytes(layout.symbols.dump(named))
        groups = {
            "grp": Group(
                "grp", "main", tuple(n for n, _, _ in entries), "authored", (), False, "src/grp.c", "include/main/grp.h"
            )
        }
        initial = LayoutMap(groups if grouped else {}, {}, {u.path: u for u in units}, "", authored, fuzzy or {})
        text = layout.dump_map(initial)
        (tmp_path / "layout.toml").write_bytes(text)
        monkeypatch.setattr(layout.version_data, "read", Mock(return_value=versions))
        bound = replace(config, project=replace(config.project, versions=tuple(versions))) if third else config
        loaded = layout.load_map(bound, versions, text, lambda p: (tmp_path / p).read_bytes())
        return Snapshot(
            bound,
            "head",
            loaded,
            versions,
            {},
            "snapshot",
            {"symbols.toml": sha256((tmp_path / "symbols.toml").read_bytes())},
        )

    return make


def _unit(members=("func_80000400",), path="src/grp.c", kind="c"):
    return UnitSpec(path, kind, "grp", members, "gcc-test", {"add": ["-O1"], "omit": ["-O2"]})


def _load(snapshot, text):
    return layout.load_map(snapshot.config, snapshot.versions, text, snapshot.read)


def test_load_dump_schema_5(scene):
    authored = ({"version": "a", "rom": 0x1000, "kind": "function", "name": "func_80000400"},)
    snapshot = scene(units=(_unit(),), authored=authored)
    result = _load(snapshot, layout.dump_map(snapshot.layout))
    assert result == snapshot.layout
    assert result.authored == authored
    assert result.units["src/grp.c"] == _unit()
    assert result.members["func_80000400"].holders() == ("a", "b")
    assert result.members["func_80000400"].group == "grp"
    doc = tomllib.loads(layout.dump_map(snapshot.layout).decode())
    assert doc["schema"] == 5 and "cap" not in doc
    assert (doc["group"][0]["source"], doc["group"][0]["header"]) == ("src/grp.c", "include/main/grp.h")


def test_a_group_without_its_source_and_header_is_refused_and_the_dump_invents_no_path(scene):
    snapshot = scene()
    text = layout.dump_map(snapshot.layout).decode()
    bare = "\n".join(line for line in text.splitlines() if not line.startswith(("source = ", "header = ")))
    with pytest.raises(Refusal) as caught:
        _load(snapshot, bare.encode())
    assert caught.value.findings[0].key == "config.schema"


def test_schema_3_is_refused_without_a_reader(scene):
    snapshot = scene()
    with pytest.raises(Refusal) as caught:
        _load(snapshot, b"schema = 3\ncap = 32\ngroup = []\nunit = []\n")
    assert caught.value.findings[0].key == "config.schema"


def test_unit_path_gives_the_source_then_numbered_files(scene):
    snapshot = scene()
    mapped = replace(snapshot.layout, groups={"grp": Group("grp", "main", (), "authored", (), False, "src/x/m.c", "")})
    assert layout.unit_path(mapped, "grp", set()) == "src/x/m.c"
    assert layout.unit_path(mapped, "grp", {"src/x/m.c"}) == "src/x/m_2.c"
    assert layout.unit_path(mapped, "grp", {"src/x/m.c", "src/x/m_2.c"}) == "src/x/m_3.c"
    assert layout.unit_path(mapped, "other", set()) == "src/other.c"


def test_standalone_unit_path_comes_from_unit_path(scene, tmp_path):
    entries = (("fa", "c", RETURN), ("fb", "asm", RETURN))
    snapshot = scene(entries, units=(_unit(("fa",)),))
    (tmp_path / "src").mkdir()
    (tmp_path / "src/grp.c").write_bytes(b"fa source")
    options = layout.unit_options(snapshot, "fb", b"x")
    assert [unit.path for unit, _ in options] == ["src/grp.c", "src/grp_2.c"]  # appended, then standalone


def _physical(scene, entries, units, **kw):
    snapshot = scene(entries, units=units, **kw)
    return layout.physical(snapshot, snapshot.layout)


def test_physical_joins_contiguous_atoms_and_numbers_the_files_a_hole_leaves(scene):
    both = (("fa", "c", RETURN), ("fb", "c", RETURN))
    units = (_unit(("fa",), "src/fa.c"), _unit(("fb",), "src/fb.c"))
    joined = _physical(scene, both, units)
    assert list(joined) == ["src/grp.c"] and joined["src/grp.c"].members == ("fa", "fb")
    held = (("fa", "c", RETURN), ("hole", "asm", RETURN), ("fb", "c", RETURN))
    split = _physical(scene, held, units)
    assert {p: u.members for p, u in split.items()} == {"src/grp.c": ("fa",), "src/grp_2.c": ("fb",)}


def test_physical_starts_a_file_at_each_boundary_the_map_records_and_the_map_keeps_them(scene):
    both = (("fa", "c", RETURN), ("fb", "c", RETURN))
    units = (_unit(("fa",), "src/fa.c"), _unit(("fb",), "src/fb.c"))
    snapshot = scene(both, units=units)
    bounded = replace(snapshot.layout, boundaries=("fb",))  # a proof found fb's bytes change after fa in one file
    result = layout.physical(snapshot, bounded)
    assert {p: u.members for p, u in result.items()} == {"src/grp.c": ("fa",), "src/grp_2.c": ("fb",)}
    loaded = _load(snapshot, layout.dump_map(bounded))
    assert loaded.boundaries == ("fb",) and loaded.digest != _load(snapshot, layout.dump_map(snapshot.layout)).digest
    assert tomllib.loads(layout.dump_map(bounded).decode())["boundaries"] == ["fb"]


def test_physical_splits_a_run_where_the_bytes_of_two_atoms_are_not_adjacent(scene, monkeypatch):
    entries = (("fa", "c", RETURN), ("fb", "c", RETURN))
    units = (_unit(("fa",), "src/fa.c"), _unit(("fb",), "src/fb.c"))
    snapshot = scene(entries, units=units)
    rows = layout.version_data.rows

    def gapped(version, reader, cache=None):
        first, second = rows(version, reader, cache)
        return [first, (second[0], second[1], replace(second[2], rom_start=second[2].rom_start + 4))]

    monkeypatch.setattr(layout.version_data, "rows", gapped)
    result = layout.physical(snapshot, snapshot.layout)
    assert {p: u.members for p, u in result.items()} == {"src/grp.c": ("fa",), "src/grp_2.c": ("fb",)}


@pytest.mark.parametrize(("shift", "apart"), [(1, True), (0, False)])
def test_physical_keeps_a_data_unit_off_a_word_boundary_apart(scene, monkeypatch, shift, apart):
    entries = (("da", "data", (1,)), ("db", "data", (2,)))
    units = (_unit(("da",), "src/da.c", kind="data"), _unit(("db",), "src/db.c", kind="data"))
    snapshot = scene(entries, units=units)
    rows = layout.version_data.rows

    def shifted(version, reader, cache=None):  # da starts `shift` bytes in and still ends where db starts
        first, second = rows(version, reader, cache)
        p = first[2]
        return [(first[0], first[1], replace(p, rom_start=p.rom_start + shift, vram=p.vram + shift)), second]

    monkeypatch.setattr(layout.version_data, "rows", shifted)
    result = layout.physical(snapshot, snapshot.layout)
    assert len(result) == (2 if apart else 1)  # a word after it would move by the compiler's alignment


@pytest.mark.parametrize("difference", ["holders", "toolchain", "withheld"])
def test_physical_splits_at_a_difference_and_never_refuses(scene, difference):
    entries = (("fa", "c", RETURN), ("fb", "c", RETURN))
    first, second = _unit(("fa",), "src/fa.c"), _unit(("fb",), "src/fb.c")
    kw = {}
    if difference == "holders":
        kw["other"] = entries[:1]
    else:
        second = replace(second, **{difference: "other" if difference == "toolchain" else ("b",)})
    result = _physical(scene, entries, (first, second), **kw)
    assert [u.members for u in result.values()] == [("fa",), ("fb",)] and list(result) == ["src/grp.c", "src/grp_2.c"]


def test_physical_data_atom_rides_a_run_and_a_far_one_opens_a_unit(scene):
    entries = (("fa", "c", RETURN), ("rodata/fa/80000408", "data", (1,)), ("blob", "data", (2,)), ("dz", "data", (3,)))
    units = (_unit(("fa", "rodata/fa/80000408")), _unit(("dz",), "src/dz.c", kind="data"))
    result = _physical(scene, entries, units)
    assert {p: (u.kind, u.members) for p, u in result.items()} == {
        "src/grp.c": ("c", ("fa", "rodata/fa/80000408")),
        "src/grp_2.c": ("data", ("dz",)),
    }


def test_physical_owned_adjacent_data_joins_the_module_unit_and_a_far_one_stays_apart(scene):
    entries = (("fa", "c", RETURN), ("dz", "data", (3,)))
    units = (_unit(("fa",)), replace(_unit(("dz",), "src/dz.c", kind="data"), toolchain="other"))
    result = _physical(scene, entries, units)
    assert {p: (u.kind, u.members) for p, u in result.items()} == {"src/grp.c": ("c", ("fa", "dz"))}
    far = (("fa", "c", RETURN), ("rodata/fa/80000408", "data", (1,)), ("blob", "data", (2,)), ("dz", "data", (3,)))
    owned = replace(_unit(("fa", "rodata/fa/80000408")))
    kept = _physical(scene, far, (owned, units[1]))
    assert {p: u.members for p, u in kept.items()} == {
        "src/grp.c": ("fa", "rodata/fa/80000408"),
        "src/grp_2.c": ("dz",),
    }


def test_physical_keeps_an_owned_row_named_after_an_old_unit_with_the_function_of_that_unit(scene):
    # G 3d7afb392: src/func_8023F43C_de_2.c held func_8023FD18_de and rodata/func_8023F43C_de_2/800C87F8 (the row was
    # cut for the unit, so it names the unit, not a function); the relayout dropped the row from every unit, and 22
    # units failed "owned .rodata size" in all five versions
    row = "rodata/grp_2/80000408"
    snapshot = scene((("fa", "c", RETURN), (row, "data", (1,))), units=(_unit(("fa", row), "src/grp_2.c"),))
    group = snapshot.layout.groups["grp"]
    unlisted = replace(snapshot.layout, groups={"grp": replace(group, members=("fa",))})  # no group lists the row
    result = layout.physical(snapshot, unlisted)
    assert [u.members for u in result.values()] == [("fa", row)]


def test_physical_keeps_an_owned_member_apart_when_another_units_bytes_lie_between_it_and_its_owner(scene):
    entries = (("fa", "c", RETURN), ("rodata/fa/80000408", "data", (1,)), ("blob", "data", (2,)))
    entries += (("rodata/fa/80000410", "data", (3,)),)
    near, far = "rodata/fa/80000408", "rodata/fa/80000410"
    units = (_unit(("fa", near)), _unit((far,), "src/far.c", kind="data"))  # blob: bytes no unit of grp holds
    result = _physical(scene, entries, units)
    assert {p: u.members for p, u in result.items()} == {"src/grp.c": ("fa", near), "src/grp_2.c": (far,)}


def test_schema_5_round_trips_subsystem_and_library_and_derives_sdk(scene):
    snapshot = scene(units=(_unit(),))
    group = replace(snapshot.layout.groups["grp"], subsystem="main/actor", library="libultra")
    mapped = replace(snapshot.layout, groups={"grp": group})
    text = layout.dump_map(mapped)
    row = tomllib.loads(text.decode())["group"][0]
    assert (row["subsystem"], row["library"], "sdk" in row) == ("main/actor", "libultra", False)
    layout._dumped.clear()  # parse the text, not the dump_map memo
    loaded = _load(snapshot, text).groups["grp"]
    assert (loaded.subsystem, loaded.library, loaded.sdk) == ("main/actor", "libultra", True)
    plain = _load(snapshot, layout.dump_map(snapshot.layout))
    layout._dumped.clear()
    assert plain.groups["grp"].sdk is False


def test_a_schema_4_map_is_refused(scene):
    snapshot = scene()
    text = layout.dump_map(snapshot.layout).replace(b"schema = 5", b"schema = 4")
    layout._dumped.clear()
    with pytest.raises(Refusal) as caught:
        _load(snapshot, text)
    assert caught.value.findings[0].key == "config.schema"


def test_ungrouped_member_and_unit_of(scene):
    snapshot = scene(grouped=False)
    assert snapshot.layout.members["func_80000400"].group == ""
    assert layout.unit_of(snapshot, "func_80000400") is None
    assert layout.unit_of(snapshot, "missing") is None
    with_unit = replace(snapshot, layout=replace(snapshot.layout, units={"src/grp.c": _unit()}))
    assert layout.unit_of(with_unit, "func_80000400") == _unit()


def test_fuzzy_roundtrip(scene):
    entries = (("zeta", "asm", RETURN), ("alpha", "asm", RETURN))
    fuzzy = {name: {"path": f"src/fuzzy/{name}.c", "scores": {"b": 0.7, "a": 0.4}} for name, _, _ in entries}
    snapshot = scene(entries, fuzzy=fuzzy)
    dumped = layout.dump_map(snapshot.layout)
    assert _load(snapshot, dumped) == snapshot.layout
    assert snapshot.layout.fuzzy == fuzzy
    rows = tomllib.loads(dumped.decode())["fuzzy"]
    assert [row["member"] for row in rows] == ["alpha", "zeta"]
    assert all(list(row["scores"]) == ["a", "b"] for row in rows)
    assert "scores = {a = 0.4, b = 0.7}" in dumped.decode()


def test_fuzzy_unknown_member_refuses_layout_map(scene):
    snapshot = scene()
    text = layout.dump_map(
        replace(snapshot.layout, fuzzy={"missing": {"path": "src/fuzzy/missing.c", "scores": {"a": 0.1}}})
    )
    with pytest.raises(Refusal) as caught:
        _load(snapshot, text)
    assert caught.value.findings[0].key == "layout.map"
    assert caught.value.findings[0].missing == ("missing",)


def test_dump_schema_5(scene):
    text = layout.dump_map(scene().layout)
    doc = tomllib.loads(text.decode())
    assert doc["schema"] == 5
    assert doc["unit"] == []
    assert "authored" not in doc and "fuzzy" not in doc


@pytest.mark.parametrize("text", [b"[bad", b"\xff", b"schema = 2\ncap = 200\ngroup = []\nunit = []\n"])
def test_invalid_layout_refuses(scene, text):
    snapshot = scene()
    with pytest.raises(Refusal):
        _load(snapshot, text)


def test_unit_unknown_member_refuses(scene):
    snapshot = scene()
    text = layout.dump_map(replace(snapshot.layout, units={"src/grp.c": _unit(("missing",))}))
    with pytest.raises(Refusal) as caught:
        _load(snapshot, text)
    assert caught.value.findings[0].key == "layout.map"
    assert caught.value.findings[0].missing == ("missing",)


def test_capture_retries_when_head_moves(scene, monkeypatch):
    snapshot = scene()
    git = Mock(side_effect=[fixture.native_result(stdout=f"{head}\n".encode()) for head in ("a", "b", "b", "b")])
    monkeypatch.setattr(layout.process, "git", git)
    captured = layout.capture(snapshot.config)
    assert captured.commit == "b"
    assert git.call_count == 4
    assert layout.version_data.read.call_count == 1  # the retry reads the same inputs, so the capture is cached
    assert all(call.args == (snapshot.config, "rev-parse", "HEAD") for call in git.call_args_list)


def test_capture_digest_ignores_the_commit(scene, monkeypatch):
    snapshot = scene()
    monkeypatch.setattr(
        layout.process,
        "git",
        Mock(side_effect=[fixture.native_result(stdout=f"{h}\n".encode()) for h in ("a", "a", "b", "b")]),
    )
    assert layout.capture(snapshot.config).digest == layout.capture(snapshot.config).digest


def test_capture_busy_after_three(scene, monkeypatch):
    snapshot = scene()
    git = Mock(side_effect=[fixture.native_result(stdout=str(i).encode()) for i in range(6)])
    monkeypatch.setattr(layout.process, "git", git)
    with pytest.raises(Refusal) as caught:
        layout.capture(snapshot.config)
    finding = caught.value.findings[0]
    assert (finding.key, finding.reason, finding.action) == (
        "layout.busy",
        "HEAD moved while reading the project three times",
        "retry the command",
    )
    assert git.call_count == 6
    assert layout.version_data.read.call_count == 1


@pytest.mark.parametrize("version", ["a", "b"])
@pytest.mark.parametrize("grouped", [True, False])
def test_asm_unit_paths_and_kind(scene, monkeypatch, version, grouped):
    snapshot = scene(grouped=grouped)
    path = snapshot.config.project.root / f"versions/{version}/asm/func_80000400.s"
    asm_path = Mock(return_value=path)
    monkeypatch.setattr(layout.version_data, "asm_path", asm_path)
    unit = layout.asm_unit(snapshot, "func_80000400", version)
    assert unit == UnitSpec(
        path.relative_to(snapshot.config.project.root).as_posix(),
        "asm",
        "grp" if grouped else "",
        ("func_80000400",),
        "gcc-test",
        {"add": [], "omit": []},
    )
    asm_path.assert_called_once_with(snapshot.config, version, "func_80000400")


def test_ungrouped_member_is_refused_by_name_when_a_unit_is_built(scene):
    snapshot = scene(grouped=False)
    with pytest.raises(Refusal) as caught:
        layout.unit_options(snapshot, "func_80000400", b"source")
    finding = caught.value.findings[0]
    assert (finding.key, finding.unit, finding.action) == ("layout.member", "func_80000400", "run unbake setup")


@pytest.mark.parametrize("member,version", [("missing", "a"), ("func_80000400", "missing"), ("datum", "a")])
def test_asm_unit_no_text_refuses(scene, member, version):
    snapshot = scene((("func_80000400", "asm", RETURN), ("datum", "data", (0,))))
    with pytest.raises(Refusal) as caught:
        layout.asm_unit(snapshot, member, version)
    assert caught.value.findings[0].key == "layout.member"
    assert layout.edit_split(b"  - [0x1000, .data, old]\n", "old", rename="new") == b"  - [0x1000, .data, new]\n"


def test_overlay_content_deletion_and_digest(scene):
    snapshot = scene()
    writes = {"src/new.c": b"source", "gone": None}
    updated = layout.overlay(snapshot, writes)
    assert updated.read("src/new.c") == b"source"
    with pytest.raises(FileNotFoundError):
        updated.read("gone")
    pins = [("gone", None), ("src/new.c", sha256(b"source"))]
    assert updated.digest == digest((snapshot.digest, pins))
    assert updated.layout is snapshot.layout
    assert snapshot.overlays == {}
    again = layout.overlay(updated, {"src/new.c": b"replacement"})
    assert again.overlays == {"src/new.c": b"replacement", "gone": None}


@pytest.mark.parametrize("watched", ["layout.toml", "versions/a/Game.yaml", "symbols.toml"])
def test_overlay_reloads_watched_files(scene, watched):
    snapshot = scene()
    old_calls = layout.version_data.read.call_count
    updated = layout.overlay(snapshot, {watched: snapshot.read(watched)})
    assert updated.layout == snapshot.layout
    assert layout.version_data.read.call_count == old_calls + 1


@pytest.mark.parametrize("path", ["src/new.c", "layout.toml", "versions/a/Game.yaml", "versions/a/symbol_addrs.txt"])
@pytest.mark.parametrize("table", ["unchanged", "external", "explicit", "inherited", "unpinned"])
def test_overlay_symbol_reuse_requires_matching_pin(scene, monkeypatch, path, table):
    snapshot = scene()
    symbol = layout.symbols.path()
    before = dict(snapshot.reads)
    original = snapshot.read(symbol)
    changed = layout.symbols.dump({"new_symbol": {"kind": "data", "a": BASE, "b": BASE}})
    writes = {path: b"source" if path == "src/new.c" else snapshot.read(path)}
    if table == "external":
        (snapshot.config.project.root / symbol).write_bytes(changed)
    elif table == "explicit":
        writes[symbol] = changed
    elif table == "inherited":
        snapshot = replace(snapshot, overlays={symbol: changed})
    elif table == "unpinned":
        snapshot.reads.pop(symbol)
    expected = original if table in ("unchanged", "unpinned") else changed

    def read_versions(config, reader, only, reuse):
        assert reader(symbol) == expected
        assert reuse is (snapshot.versions if table == "unchanged" else None)
        return {v: replace(snapshot.versions[v], symbols={"fresh": BASE}) for v in only}

    read = Mock(side_effect=read_versions)
    monkeypatch.setattr(layout.version_data, "read", read)
    updated = layout.overlay(snapshot, writes)
    wanted = set(snapshot.versions) if table != "unchanged" else ({"a"} if path.startswith("versions/") else set())
    assert {v for v in updated.versions if updated.versions[v] is not snapshot.versions[v]} == wanted
    assert read.call_count == (0 if table == "unchanged" and path == "src/new.c" else 1)
    assert updated.reads is snapshot.reads
    assert snapshot.reads[symbol] == before[symbol]
    assert layout.overlay(snapshot, writes) is updated
    assert read.call_count == (0 if table == "unchanged" and path == "src/new.c" else 1)


def test_unit_options_standalone_and_overlay(scene):
    snapshot = scene()
    options = layout.unit_options(snapshot, "func_80000400", b"void f(void) {}\n")
    assert len(options) == 1
    unit, writes = options[0]
    assert unit.path == "src/grp.c" and unit.members == ("func_80000400",)
    proposed = layout.overlay(snapshot, writes)
    assert layout.unit_of(proposed, "func_80000400") == unit
    assert proposed.read(unit.path) == b"void f(void) {}\n"
    assert snapshot.layout.units == {}
    for version in snapshot.versions.values():
        assert b", c, func_80000400]" in proposed.read(version.split)
        assert b", asm, func_80000400]" in snapshot.read(version.split)


@pytest.mark.parametrize("adjacent", [True, False])
def test_unit_options_append_requires_all_holders(scene, adjacent):
    entries = (("first", "c", RETURN), ("target", "asm", RETURN), ("gap", "asm", RETURN))
    other = entries if adjacent else (entries[0], entries[2], entries[1])
    unit = _unit(("first",))
    snapshot = scene(entries, other=other, units=(unit,))
    (snapshot.config.project.root / unit.path).parent.mkdir(exist_ok=True)
    (snapshot.config.project.root / unit.path).write_bytes(b"first source")
    options = layout.unit_options(snapshot, "target", b"target source")
    assert len(options) == (2 if adjacent else 1)
    assert options[-1][0].path == "src/grp_2.c"
    if adjacent:
        extended, writes = options[0]
        assert extended.members == ("first", "target") and extended.options == unit.options
        assert writes[unit.path] == b"first source\ntarget source"


@pytest.mark.parametrize("state,member", [("c", "func_80000400"), ("asm", "missing")])
def test_unit_options_refuses(scene, state, member):
    snapshot = scene((("func_80000400", state, RETURN),))
    with pytest.raises(Refusal) as caught:
        layout.unit_options(snapshot, member, b"source")
    assert caught.value.findings[0].key == "layout.member"


@pytest.mark.parametrize("rule", ["prelude", "split", "merge"])
@pytest.mark.parametrize("guard", ["none", "authored", "compiled", "disagree"])
def test_boundary_rules_and_guards(scene, monkeypatch, rule, guard):
    first = ("func_80000400", "asm", RETURN if rule != "merge" else (0x24020001, 0))
    words = {"prelude": (0, 0, *RETURN), "split": (*RETURN, *RETURN), "merge": RETURN}[rule]
    target = ("func_80000408", "c" if guard == "compiled" else "asm", words)
    entries = (first, target)
    authored = ()
    if guard == "authored":
        authored = ({"version": "a", "rom": 0x1008, "kind": "function", "name": target[0]},)
    snapshot = scene(entries, authored=authored)
    refs = frozenset({BASE + 16}) if rule != "merge" else frozenset()

    def decoded(snap, version, rows):
        references = frozenset({BASE + 8}) if guard == "disagree" and version.id == "b" else refs
        return {first[0]: first[2], target[0]: words}, references

    monkeypatch.setattr(layout, "_decoded", decoded)
    plan, counts = layout.boundary_plan(snapshot)
    assert plan.operation == "layout" and plan.base == snapshot.digest
    assert not plan.blocking and not plan.debt and not plan.affected
    if guard != "none":
        assert not plan.writes
        if guard == "disagree":
            assert counts[rule] == {"proposed": 1, "applied": 0, "withheld": 1}
            assert counts["withheld_reasons"] == {"versions disagree": 1}
        return
    assert counts[rule] == {"proposed": 1, "applied": 1, "withheld": 0}
    expected = {
        "prelude": (first[0], "func_80000410"),
        "split": (first[0], target[0], "func_80000410"),
        "merge": (first[0],),
    }[rule]
    assert tomllib.loads(plan.writes["layout.toml"].decode())["group"][0]["members"] == list(expected)
    for version in snapshot.versions.values():
        split = yaml.safe_load(plan.writes[version.split])["segments"][0]["subsegments"]
        assert tuple(row[2] for row in split[:-1]) == expected
        symbols = plan.writes[version.symbols_file]
        assert all(name.encode() in symbols for name in expected)
        if rule != "split":
            assert target[0].encode() not in symbols


@pytest.mark.parametrize(
    "tail,expected", [((0x27BD0018,), ("first",)), ((0x27BD0018, *RETURN), ("first", "func_8000040C"))]
)
def test_boundary_returns_a_delay_slot_to_the_jump_that_owns_it(scene, monkeypatch, tail, expected):
    ending = (0x24020001, 0x03E00008)  # the jump is the last word: its delay slot sits in the next member
    snapshot = scene(
        (("first", "asm", ending), ("func_80000408", "asm", tail)),
        other=None if len(tail) == 1 else (("first", "asm", (*ending, tail[0])), ("func_80000408", "asm", tail[1:])),
    )
    monkeypatch.setattr(
        layout,
        "_decoded",
        lambda snap, version, rows: (
            {"first": ending, "func_80000408": tail}
            if version.id == "a" or len(tail) == 1
            else {"first": (*ending, tail[0]), "func_80000408": tail[1:]},
            frozenset({BASE + 8}),
        ),
    )  # referenced: only the slot rule may move it
    plan, counts = layout.boundary_plan(snapshot)
    assert counts["merge"]["applied"] + counts["prelude"]["applied"] == 1 and not counts["withheld_reasons"]
    assert tomllib.loads(plan.writes["layout.toml"].decode())["group"][0]["members"] == list(expected)
    a = yaml.safe_load(plan.writes[snapshot.versions["a"].split])["segments"][0]["subsegments"]
    assert tuple(row[2] for row in a[:-1]) == expected  # a version whose owner has the slot already is left alone


def test_decoded_references_and_cache(scene):
    jal = (3 << 26) | ((BASE + 8) >> 2 & 0x3FFFFFF)
    snapshot = scene(
        (("caller", "asm", (jal, 0)), ("callee", "asm", RETURN), ("table", "data", (BASE + 8, 0x12345678)))
    )
    version = snapshot.versions["a"]
    rows = layout.version_data.rows(version, snapshot.read)
    words, refs = layout._decoded(snapshot, version, rows)
    assert words == {"caller": (jal, 0), "callee": RETURN}
    assert refs == frozenset({BASE + 8})
    assert layout._decoded(snapshot, version, rows)[0] is words


@pytest.mark.parametrize("prior", [RETURN, (0x08000100, 0)])
def test_boundary_merge_withheld_after_return_or_jump(scene, prior):
    snapshot = scene((("first", "asm", prior), ("second", "asm", RETURN)))
    plan, counts = layout.boundary_plan(snapshot)
    assert not plan.writes
    assert counts["merge"]["proposed"] == 0


@pytest.mark.parametrize("guard", ["previous_c", "following_c", "segment", "gap", "referenced"])
def test_boundary_merge_guards(scene, monkeypatch, guard):
    entries = (
        ("first", "c" if guard == "previous_c" else "asm", (0x24020001, 0)),
        ("second", "asm", RETURN),
        ("third", "c" if guard == "following_c" else "asm", RETURN),
    )
    snapshot = scene(entries)
    if guard == "segment":
        versions = {
            key: replace(v, segments=(*v.segments, ("next", 0x1008, 0x1018, BASE + 8)))
            for key, v in snapshot.versions.items()
        }
        snapshot = replace(snapshot, versions=versions)
    if guard == "gap":
        original = layout.version_data.rows

        def rows(version, reader, cache=None):
            result = original(version, reader)
            name, state, placement = result[0]
            result[0] = (name, state, replace(placement, rom_end=placement.rom_end - 4))
            return result

        monkeypatch.setattr(layout.version_data, "rows", rows)
    if guard == "referenced":
        monkeypatch.setattr(
            layout,
            "_decoded",
            lambda snap, version, rows: ({name: words for name, _, words in entries}, frozenset({BASE + 8})),
        )
    plan, counts = layout.boundary_plan(snapshot)
    assert not plan.writes
    assert counts["merge"]["proposed"] == 0


@pytest.mark.parametrize("ending", ["\n", "\r\n", ""])
def test_edit_preserves_format_and_comments(ending):
    text = f"  - [0x1000, asm, 'named'] # keep{ending}".encode()
    edited = layout.edit_split(text, "named", state="c", start=0x1004, rename="renamed")
    assert edited == f"  - [0x1004, c, 'renamed'] # keep{ending}".encode()
    assert layout.edit_split(text, "named", drop=True) == b""


def test_edit_additions_and_nontext_refusal():
    text = b"  - [4096, asm, old] # original\n"
    result = layout.edit_split(text, "old", additions=((0x1008, "new"),))
    assert result == b"  - [4096, asm, old] # original\n  - [0x1008, asm, new]\n"
    with pytest.raises(Refusal) as caught:
        layout.edit_split(b"  - [0x1000, data, old]\n", "old", state="c")
    assert caught.value.findings[0].key == "layout.member"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("func_80000400", "func_80000408"),
        ("D_80000400_tail", "D_80000408_tail"),
        ("authored", "authored"),
        ("func_80000500", "func_80000500"),
    ],
)
def test_rename_only_matching_auto_addresses(name, expected):
    assert layout._rename(name, BASE, BASE + 8) == expected


def test_capture_cache_reuses_parsed_snapshot_and_invalidates_inputs(scene):
    original = scene()
    cfg = original.config
    first = layout.capture(cfg)
    second = layout.capture(cfg)
    assert second == first
    assert layout.version_data.read.call_count == 1
    path = cfg.project.root / "layout.toml"
    path.write_bytes(path.read_bytes() + b"\n")
    assert layout.capture(cfg).layout == first.layout
    assert layout.version_data.read.call_count == 2


def test_boundary_plan_is_computed_once_per_snapshot(scene, monkeypatch):
    snapshot = scene()
    calls = []
    words = {"func_80000400": RETURN}
    monkeypatch.setattr(layout, "_decoded", lambda s, v, r: calls.append(v.id) or (words, frozenset()))
    first = layout.boundary_plan(snapshot)
    count = len(calls)
    assert layout.boundary_plan(snapshot)[1] == first[1] and len(calls) == count


def test_boundary_key_never_renders_toml_and_ignores_fuzzy_scores(scene, monkeypatch):
    snapshot = scene()
    calls = []
    monkeypatch.setattr(layout, "dump_map", lambda *args: pytest.fail("a boundary cache key must not render TOML"))
    monkeypatch.setattr(
        layout,
        "_boundary",
        lambda s: calls.append(s) or (SimpleNamespace(writes={}), {"computed": len(calls), "fill": 0}),
    )
    scored = replace(snapshot, layout=replace(snapshot.layout, fuzzy={"func_80000400": {"scores": {"a": 0.5}}}))
    first = layout.boundary_plan(scored)
    fuzzy = replace(snapshot, layout=replace(snapshot.layout, fuzzy={"func_80000400": {"scores": {"a": 0.7}}}))
    assert layout.boundary_plan(fuzzy) == first and len(calls) == 1
    changed = replace(snapshot, layout=replace(snapshot.layout, groups={}))
    assert layout.boundary_plan(changed)[1] == {"computed": 2, "fill": 0}


def test_every_copy_of_a_project_shares_the_one_host_cache(scene, tmp_path):
    from unbake import store

    cfg = scene().config
    other = replace(cfg, project=replace(cfg.project, root=tmp_path / "copy"))
    shared = store.content(cfg).directory  # the one folder beside cache.dir named for the entry format
    assert shared.parent == cfg.host.cache_dir.parent and shared.name.startswith(cfg.host.cache_dir.name + ".v")
    assert store.content(other).directory == shared
    assert store.cached(cfg, "capture", "k", lambda: b"one") == b"one"
    assert store.cached(other, "capture", "k", lambda: b"never") == b"one"  # the same content key, one shared entry


HAND = [("a", 0x1000, 0x1020), ("b", 0x1020, 0x1040), ("c", 0x1040, 0x1060)]


def claim_scene(scene, monkeypatch, hand):
    snapshot = scene()
    version = snapshot.versions["a"]
    lines = "".join(f'      - [0x{s:X}, rodata, "{n}"]\n' for n, s, _ in hand)
    text = f"segments:\n  - name: m\n    subsegments:\n{lines}      - [0x{hand[-1][2]:X}]\n".encode()
    rows = [(n, "rodata", Placement("a", ".rodata", s, e, 0x800C0000 + s - 0x1000)) for n, s, e in hand]
    monkeypatch.setattr(layout.version_data, "rows", lambda v, reader, cache=None: rows)
    monkeypatch.setattr(layout.version_data, "section_of", lambda v, kind, where: ".rodata")
    return replace(snapshot, overlays={version.split: text}), version.split


@pytest.mark.parametrize(
    "claims, expected",
    [
        ([(0x1000, 0x1060, "r/x")], ['[0x1000, rodata, "r/x"]']),  # every hand row is replaced by the claim
        (
            [(0x1008, 0x1018, "r/x")],
            [
                '[0x1000, rodata, "a"]',
                '[0x1008, rodata, "r/x"]',
                '[0x1018, rodata, "rodata/unresolved/800C0018"]',
                '[0x1020, rodata, "b"]',
                '[0x1040, rodata, "c"]',
            ],
        ),  # inside one row: its head and its tail stay
        (
            [(0x1010, 0x1050, "r/x")],
            ['[0x1000, rodata, "a"]', '[0x1010, rodata, "r/x"]', '[0x1050, rodata, "rodata/unresolved/800C0050"]'],
        ),  # b dropped, c cut at the end
        (
            [(0x1008, 0x1010, "r/x"), (0x1010, 0x1020, "r/y")],
            [
                '[0x1000, rodata, "a"]',
                '[0x1008, rodata, "r/x"]',
                '[0x1010, rodata, "r/y"]',
                '[0x1020, rodata, "b"]',
                '[0x1040, rodata, "c"]',
            ],
        ),  # a tail never doubles a claim
        ([(0x1020, 0x1040, "r/x")], ['[0x1000, rodata, "a"]', '[0x1020, rodata, "r/x"]', '[0x1040, rodata, "c"]']),
        (
            [(0x1028, 0x1040, "b")],
            [
                '[0x1000, rodata, "a"]',
                '[0x1020, rodata, "rodata/unresolved/800C0020"]',
                '[0x1028, rodata, "b"]',
                '[0x1040, rodata, "c"]',
            ],
        ),  # a claim of the row's own name past its start: the head it leaves is named by address, not twice
    ],
)
def test_claim_rows_makes_each_claim_one_row_and_names_what_it_cuts_by_address(scene, monkeypatch, claims, expected):
    snapshot, split = claim_scene(scene, monkeypatch, HAND)
    made = tuple(Claim("u", "a", ".rodata", s, e, (n,)) for s, e, n in claims)
    got = layout.claim_rows(snapshot, made)
    assert [line.strip(" -") for line in got[split].decode().splitlines()[3:-1]] == expected


def test_claim_rows_changes_nothing_where_the_rows_are_the_claims_and_refuses_a_name_in_use(scene, monkeypatch):
    snapshot, _ = claim_scene(scene, monkeypatch, HAND)
    assert layout.claim_rows(snapshot, (Claim("u", "a", ".rodata", 0x1020, 0x1040, ("b",)),)) == {}
    with pytest.raises(Refusal):
        layout.claim_rows(snapshot, (Claim("u", "a", ".rodata", 0x1000, 0x1010, ("c",)),))
    with pytest.raises(Refusal):
        layout.claim_rows(snapshot, (Claim("u", "a", ".rodata", 0x2000, 0x2010, ("z",)),))


def test_claim_rows_reads_a_split_files_rows_once_for_all_the_units_whose_claims_are_rows_already(scene, monkeypatch):
    snapshot, _ = claim_scene(scene, monkeypatch, HAND)
    reads = []
    real = layout.split_states
    monkeypatch.setattr(layout, "split_states", lambda snap, version: reads.append(version.id) or real(snap, version))
    layout.effort._memo.clear()
    claim = (Claim("u", "a", ".rodata", 0x1020, 0x1040, ("b",)),)
    assert all(layout.claim_rows(snapshot, claim) == {} for _ in range(500))  # 500 units, the same rows
    assert reads == ["a"]


def test_unit_options_data_member_gets_its_own_unit(scene):
    snapshot = scene((("func_80000400", "asm", RETURN), ("tbl", "rodata", (1, 2))))
    unit, writes = layout.unit_options(snapshot, "tbl", b"const int tbl[] = {1, 2};\n")[0]
    assert unit.path == "src/grp.c" and unit.members == ("tbl",) and unit.group == "grp"
    assert writes[unit.path] == b"const int tbl[] = {1, 2};\n"
    for version in snapshot.versions.values():
        assert b", .rodata, tbl]" in writes[version.split] and b", rodata, tbl]" in snapshot.read(version.split)


def test_unit_options_data_member_refuses_compiled_state(scene):
    snapshot = scene((("func_80000400", "asm", RETURN), ("tbl", "c", RETURN)))
    with pytest.raises(Refusal) as caught:
        layout.unit_options(snapshot, "tbl", b"x")
    assert caught.value.findings[0].key == "layout.member"


def test_dump_map_writes_the_documented_layout_text():
    import tomllib

    groups = {"x": Group("x", "s", ("m", "n"), "proven", ("seg",), False, "src/x.c", "include/s/x.h")}
    units = {
        "a.c": UnitSpec("a.c", "c", "x", ("m",), "t", {"add": ["-O1"], "omit": []}, ("eu",)),
        "b.c": UnitSpec("b.c", "c", "x", ("n",), "t", {"add": [], "omit": []}),
    }
    fuzzy = {"m": {"path": 'p"q', "scores": {"us-rev1": 0.5, "de": 1.0}}}
    text = layout.dump_map(LayoutMap(groups, {}, units, "d", (), fuzzy))
    assert text.decode() == (
        'schema = 5\n\n[[group]]\nname = "x"\nsegment = "s"\nmembers = ["m", "n"]\nevidence = "proven"\n'
        'signals = ["seg"]\nsource = "src/x.c"\nheader = "include/s/x.h"\nsubsystem = ""\nlibrary = ""\n\n'
        '[[unit]]\npath = "a.c"\nkind = "c"\ngroup = "x"\n'
        'members = ["m"]\ntoolchain = "t"\nwithheld = ["eu"]\n\n[unit.options]\nadd = ["-O1"]\nomit = []\n\n'
        '[[unit]]\npath = "b.c"\nkind = "c"\ngroup = "x"\nmembers = ["n"]\ntoolchain = "t"\n\n[unit.options]\n'
        'add = []\nomit = []\n\n[[fuzzy]]\nmember = "m"\npath = "p\\"q"\nscores = {de = 1.0, us-rev1 = 0.5}\n'
    )
    assert tomllib.loads(text.decode())["fuzzy"][0]["scores"] == {"de": 1.0, "us-rev1": 0.5}
    empty = tomllib.loads(layout.dump_map(LayoutMap({}, {}, {}, "d", (), {})).decode())
    assert empty == {"schema": 5, "group": [], "unit": []}


def _ungrouped_data(scene, entries):
    snapshot = scene(entries)
    members = {n: replace(m, group="") if m.kind != "function" else m for n, m in snapshot.layout.members.items()}
    groups = {
        k: replace(g, members=tuple(n for n in g.members if members[n].kind == "function"))
        for k, g in snapshot.layout.groups.items()
    }
    text = layout.dump_map(replace(snapshot.layout, members=members, groups=groups))
    (snapshot.config.project.root / "layout.toml").write_bytes(text)
    return layout.overlay(snapshot, {"layout.toml": text})


def _land_data(snapshot, name):
    unit, writes = layout.unit_options(snapshot, name, b"const unsigned int d[2] = {1, 2};\n")[0]
    return unit, layout.overlay(snapshot, writes)


def test_unowned_data_joins_the_module_of_the_row_before_it(scene):
    entries = (
        ("func_80000400", "asm", RETURN),
        ("rodata/unresolved/80000408", "rodata", (1, 2)),
        ("rodata/unresolved/80000410", "rodata", (3, 4)),
    )
    snapshot = _ungrouped_data(scene, entries)
    first, landed = _land_data(snapshot, "rodata/unresolved/80000410")
    assert first.group == "grp" and first.kind == "data" and first.members == ("rodata/unresolved/80000410",)
    assert layout.unit_of(landed, "rodata/unresolved/80000410") == first
    assert first.path == "src/grp.c" and set(landed.layout.groups) == {"grp"}
    assert b", .rodata, rodata/unresolved/80000410]" in landed.read("versions/a/Game.yaml")


def test_data_in_a_segment_without_a_module_opens_one_with_default_paths(scene):
    entries = (("rodata/unresolved/80000408", "rodata", (1, 2)), ("rodata/unresolved/80000410", "rodata", (3, 4)))
    snapshot = _ungrouped_data(scene, entries)
    bare = replace(snapshot, layout=replace(snapshot.layout, groups={}))
    first, landed = _land_data(bare, "rodata/unresolved/80000410")
    module = landed.layout.groups["data_80000408"]
    assert first.group == "data_80000408" and first.path == "src/data_80000408.c"
    assert (module.segment, module.source, module.header) == (
        "main",
        "src/data_80000408.c",
        "include/main/data_80000408.h",
    )


def test_data_named_for_its_function_joins_that_functions_module(scene):
    entries = (("func_80000400", "asm", RETURN), ("rodata/func_80000400/80000408", "rodata", (1, 2)))
    snapshot = _ungrouped_data(scene, entries)
    unit, landed = _land_data(snapshot, "rodata/func_80000400/80000408")
    assert unit.group == "grp" and landed.layout.groups == snapshot.layout.groups


def test_unit_options_keeps_the_signature_recipe_calls() -> None:
    import inspect

    assert str(inspect.signature(layout.unit_options)) == (
        "(snapshot: 'Snapshot', member: 'str', source: 'bytes') -> 'list[tuple[UnitSpec, dict[str, bytes | None]]]'"
    )


def _share(snapshot):
    """One symbol both versions declare at the data row's address."""
    table = layout.symbols.parse(snapshot.read("symbols.toml"), ("a", "b"))
    (snapshot.config.project.root / "symbols.toml").write_bytes(
        layout.symbols.dump({**table, "shared": {"kind": "data", "a": BASE + 8, "b": BASE + 8}})
    )
    shared = {"shared": BASE + 8}
    return replace(snapshot, versions={k: replace(v, symbols=shared) for k, v in snapshot.versions.items()})


def test_boundary_joins_the_data_rows_one_symbol_names_in_two_versions(scene):
    data = (0x11111111, 0x22222222)
    snapshot = scene(
        (("func_80000400", "asm", RETURN), ("rodata_a", "data", data)),
        other=(("func_80000400", "asm", RETURN), ("rodata_b", "data", data)),
    )
    snapshot = _share(snapshot)
    plan, counts = layout.boundary_plan(snapshot)
    assert counts["join"] == 1
    names = {
        k: [r[2] for r in yaml.safe_load(plan.writes[v.split])["segments"][0]["subsegments"][:-1]]
        for k, v in snapshot.versions.items()
        if v.split in plan.writes
    }
    assert names == {"b": ["func_80000400", "rodata_a"]}
    again = layout.overlay(snapshot, {"versions/b/Game.yaml": plan.writes["versions/b/Game.yaml"]})
    assert layout.boundary_plan(again)[1]["join"] == 0


def _joined(scene, landed_names):
    data = (0x11111111, 0x22222222)
    units = tuple(_unit((n,), f"src/{n}.c", "data") for n in landed_names)
    snapshot = scene(
        (("func_80000400", "asm", RETURN), ("rodata_a", "data", data)),
        other=(("func_80000400", "asm", RETURN), ("rodata_b", "data", data)),
        units=units,
    )
    snapshot = _share(snapshot)
    plan, counts = layout.boundary_plan(snapshot)
    return {
        k: [r[2] for r in yaml.safe_load(plan.writes[v.split])["segments"][0]["subsegments"][:-1]]
        for k, v in snapshot.versions.items()
        if v.split in plan.writes
    }, counts


@pytest.mark.parametrize("landed", [("rodata_a",), ("rodata_b",)])
def test_boundary_join_never_gives_a_landed_member_another_version(scene, landed):
    names, counts = _joined(scene, landed)
    assert names == {} and counts["join"] == 0


def test_boundary_join_never_renames_a_row_between_two_landed_names(scene):
    names, counts = _joined(scene, ("rodata_a", "rodata_b"))
    assert names == {} and counts["join"] == 0


def test_boundary_splits_an_unowned_data_row_that_starts_between_words(scene, monkeypatch):
    monkeypatch.setattr(layout.symbols, "edit", lambda snapshot: {})
    snapshot = scene((("func_80000400", "asm", RETURN), ("rodata/unresolved/80000408", "data", (1, 2, 3))))
    for vid in ("a", "b"):
        path = snapshot.config.project.root / f"versions/{vid}/Game.yaml"
        path.write_text(path.read_text().replace("0x1008", "0x1009"))
    plan, counts = layout.boundary_plan(snapshot)
    rows = [r[:3] for r in yaml.safe_load(plan.writes["versions/a/Game.yaml"])["segments"][0]["subsegments"][:-1]]
    assert counts["align"] == 2 and rows[1:] == [
        [0x1009, "data", "rodata/unresolved/80000408"],
        [0x100C, "data", "data/unresolved/8000040C"],
    ]


def test_boundary_does_not_join_rows_on_a_name_only_the_versions_generated(scene):
    data = (0x11111111, 0x22222222)
    snapshot = scene(
        (("func_80000400", "asm", RETURN), ("rodata_a", "data", data)),
        other=(("func_80000400", "asm", RETURN), ("rodata_b", "data", data)),
    )
    generated = {"D_auto_80000408": BASE + 8}
    snapshot = replace(snapshot, versions={k: replace(v, symbols=generated) for k, v in snapshot.versions.items()})
    assert layout.boundary_plan(snapshot)[1]["join"] == 0


def test_boundary_types_an_unowned_data_row_holding_a_jump_table_as_read_only_data(scene, monkeypatch):
    monkeypatch.setattr(layout.symbols, "edit", lambda snapshot: {"jtbl_80000408": {"kind": "data"}})
    snapshot = scene((("func_80000400", "asm", RETURN), ("rodata/unresolved/80000408", "data", (1, 2, 3))))
    snapshot = replace(
        snapshot, versions={k: replace(v, symbols={"jtbl_80000408": BASE + 8}) for k, v in snapshot.versions.items()}
    )
    plan, counts = layout.boundary_plan(snapshot)
    rows = [r[:3] for r in yaml.safe_load(plan.writes["versions/a/Game.yaml"])["segments"][0]["subsegments"][:-1]]
    assert counts["retype"] == 2 and rows[1] == [0x1008, "rodata", "rodata/unresolved/80000408"]


def test_edit_parses_only_rows_containing_the_member_name(monkeypatch):
    text = "segments:\n  - type: code\n    subsegments:\n"
    text += "".join(f"      - [0x{i * 4:X}, asm, f{i}]\n" for i in range(1000))
    actual, parsed = layout._LINE, []

    class Pattern:
        def match(self, line):
            parsed.append(line)
            return actual.match(line)

    monkeypatch.setattr(layout, "_LINE", Pattern())
    changed = layout.edit_split(text.encode(), "f999", rename="renamed")
    assert len(parsed) == 1 and b", asm, renamed]" in changed and b", asm, f998]" in changed


@pytest.mark.parametrize("gap", [False, True])
def test_explicit_shared_source_keeps_prefix_and_requires_contiguous_holders(scene, gap):
    prefix = _unit(("first",))
    entries = (("first", "c", RETURN), ("middle", "asm", RETURN), ("target", "asm", RETURN))
    snapshot = scene(entries, units=(prefix,))
    source = b"void first(void) {}\nvoid middle(void) {}\nvoid target(void) {}\n"
    if gap:
        source = b"void first(void) {}\nvoid target(void) {}\n"
        with pytest.raises(Refusal, match="contiguous"):
            layout.source_unit(snapshot, "target", source)
        return
    result, writes = layout.source_unit(snapshot, "target", source)
    assert result.path == prefix.path and result.members == ("first", "middle", "target")
    assert result.options == prefix.options and writes[result.path] == source
    for vid in ("a", "b"):
        assert b", c, middle]" in writes[snapshot.versions[vid].split]
        assert b", c, target]" in writes[snapshot.versions[vid].split]
    assert snapshot.layout.units == {prefix.path: prefix}


def test_shared_source_cannot_drop_a_landed_prefix_function(scene):
    prefix = _unit(("first", "middle"))
    snapshot = scene((("first", "c", RETURN), ("middle", "c", RETURN), ("target", "asm", RETURN)), units=(prefix,))
    with pytest.raises(Refusal, match="complete landed prefix"):
        layout.source_unit(snapshot, "target", b"void first(void) {}\nvoid target(void) {}\n")


def test_boundary_proposes_no_split_for_a_piece_of_only_nops(scene, monkeypatch):
    first, target = ("func_80000400", "asm", RETURN), ("func_80000408", "asm", (*RETURN, 0, 0))
    snapshot = scene((first, target))
    monkeypatch.setattr(
        layout,
        "_decoded",
        lambda snap, version, rows: ({first[0]: first[2], target[0]: target[2]}, frozenset({BASE + 16})),
    )  # padding is referenced, not a function
    plan, counts = layout.boundary_plan(snapshot)
    assert counts["split"] == {"proposed": 0, "applied": 0, "withheld": 0} and not plan.writes


@pytest.mark.parametrize(
    "guard",
    [
        "none",
        "landed_previous",
        "landed_aligned",
        "branch",
        "jump",
        "indirect",
        "reference",
        "body",
        "authored",
        "compiled",
        "no_return",
        "segment",
        "same_offset",
        "walk_stops",
    ],
)
def test_boundary_nonzero_fill_after_completed_return(scene, monkeypatch, guard):
    name = "func_8000040C"
    body = (0x27BDFFE8, 0x24020001, 0x03E00008, 0x27BD0018)
    if guard == "branch":
        body = (0x1000FFFE, *body[1:])  # +0x10 branches into the fill at +0x0C
    if guard == "jump":
        body = (0x08000103, *body[1:])
    if guard == "indirect":
        body = (0x00600008, *body[1:])
    first = ("first", "c" if guard in ("landed_previous", "no_return") else "asm", (0, *RETURN))
    if guard == "no_return":
        first = (first[0], first[1], (0, 0x24020001, 0))
    if guard == "walk_stops":  # an indirect jump the walk cannot follow, then the member's own final return
        first = (first[0], first[1], (0x00400008, *RETURN))
    target = (name, "c" if guard == "compiled" else "asm", (0x00001821, *body))
    other = (("first", first[1], RETURN), (name, target[1], (0x08000140, 0, *body)))
    if guard == "body":
        other = (other[0], (name, "asm", (0x08000140, 0, 0x27BDFFE8, 0x24030001, *body[2:])))
    if guard == "same_offset":
        other = (first, (name, "asm", (0x27BDFFE8, *body)))
    if guard == "landed_aligned":
        other = (("first", "asm", (0, 0, *RETURN)), (name, "c", body))
    units = (_unit(("first",)),) if guard == "landed_previous" else ()
    if guard == "landed_aligned":
        units = (_unit((name,)),)
    authored = ({"version": "a", "rom": 0x100C, "kind": "function", "name": name},) if guard == "authored" else ()
    fuzzy = {name: {"path": "src/fuzzy/f.c", "scores": {"a": 0.5, "b": 1.0}}} if guard == "landed_aligned" else {}
    snapshot = scene((first, target), other=other, units=units, authored=authored, fuzzy=fuzzy)
    if guard == "reference":
        decode = layout._decoded
        monkeypatch.setattr(layout, "_decoded", lambda s, v, r: (decode(s, v, r)[0], frozenset({BASE + 12})))
    if guard == "segment":
        snapshot = replace(
            snapshot,
            versions={
                v: replace(ver, segments=((*ver.segments, ("next", 0x100C, 0x1100, BASE + 12))))
                for v, ver in snapshot.versions.items()
            },
        )
    plan, counts = layout.boundary_plan(snapshot)
    if guard not in ("none", "landed_previous", "landed_aligned", "walk_stops"):
        assert not any("_padding_" in p for p in plan.writes)
        return
    changed = layout.overlay(snapshot, plan.writes)
    assert counts["prelude"]["applied"] == 1
    for v, version in snapshot.versions.items():
        before = {n: p for n, _, p in layout.split_states(snapshot, version)}
        after = {n: (state, p) for n, state, p in layout.split_states(changed, version)}
        assert after["first"][1] == before["first"]
        kept = name if guard == "landed_aligned" else "func_80000410"
        assert after[kept][1].vram == BASE + 16
        pad = f"{name}_padding_{before[name].rom_start:X}"
        if guard == "landed_aligned" and v == "b":
            assert pad not in after
        else:
            assert after[pad][0] == "data"
            assert after[pad][1].rom_end == after[kept][1].rom_start
            assert f"src/{pad}.c" in changed.layout.units
            assert pad in changed.layout.groups["grp"].members
            from unbake import versions

            source = plan.writes[f"src/{pad}.c"]
            assert source.startswith(b"unsigned char ")
            data = bytes(int(word, 16) for word in re.findall(rb"0x([0-9A-F]{2})", source))
            assert data == versions.rom_bytes(version, after[pad][1].rom_start, after[pad][1].rom_end)
    assert not layout.boundary_plan(changed)[0].writes


@pytest.mark.parametrize("ambiguous", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_retained_code_requires_a_unique_whole_function_fingerprint(monkeypatch, ambiguous, declared):
    from types import SimpleNamespace

    from unbake import infer
    from unbake.contracts import Member, Placement

    code = b"abcdefgh" * 4
    member = Member("fn", "function", "asm", "g", (Placement("a", ".text", 0, 32, 0x80000000),))
    retained = Placement("b", ".data", 64, 104, 0x80001000)
    members = {"fn": member}
    if ambiguous:
        members["twin"] = replace(member, name="twin")
    vs = {
        "a": SimpleNamespace(id="a", symbols={"fn": 0x80000000}, code={"fn"}),
        "b": SimpleNamespace(id="b", symbols={"target": 0x80001008}, code={"target"}),
    }
    snapshot = SimpleNamespace(
        versions=vs,
        layout=SimpleNamespace(members=members, units={}),
        config=SimpleNamespace(project=SimpleNamespace(version_files={})),
        read=lambda p: b"",
    )
    monkeypatch.setattr(layout, "split_states", lambda s, v: [("retained_40", "data", retained)] if v.id == "b" else [])
    monkeypatch.setattr(infer, "_code", lambda s, p: code[: p.size])
    monkeypatch.setattr(infer, "masked_code", lambda raw: raw)
    table = {name: {"kind": "function", "a": 0x80000000} for name in members} if declared else {}
    monkeypatch.setattr(layout.symbols, "edit", lambda _: table)
    claims = []
    monkeypatch.setattr(layout, "claim_rows", lambda s, rows: claims.extend(rows) or {"b.yaml": b"claimed"})
    monkeypatch.setattr(layout.symbols, "files", lambda *_: {})
    monkeypatch.setattr(layout.version_data, "rows", lambda *args: [("fn", "asm", None)])
    monkeypatch.setattr(layout.store, "content", lambda _: SimpleNamespace(cached=None))
    result = layout._retained(snapshot)
    assert bool(result) is not ambiguous
    if not ambiguous:
        assert [(c.start, c.end, c.section) for c in claims] == [(72, 104, ".text")]
        assert table["fn"] == {"kind": "function", "a": 0x80000000, "b": 0x80001008}


JR, STALE_A, STALE_B = 0x03E00008, 0x24848080, 0x000B0000
NEXT = ("next", "asm", (0x27BDFFE8, 0x24020001, *RETURN))


@pytest.fixture
def calls(monkeypatch):
    """Make the aligned member after a fill a called function, as it is in a real ROM."""
    decode = layout._decoded

    def add(*addresses):
        monkeypatch.setattr(layout, "_decoded", lambda s, v, r: (decode(s, v, r)[0], frozenset({*addresses})))

    return add


def _rows(snapshot, plan):
    changed = layout.overlay(snapshot, plan.writes)
    return changed, {
        v: {n: (state, p) for n, state, p in layout.split_states(changed, ver)} for v, ver in snapshot.versions.items()
    }


@pytest.mark.parametrize("referenced", [False, True])
@pytest.mark.parametrize("regroup", ["authored", "same", "renamed"])
def test_boundary_fill_member_between_a_return_and_the_next_aligned_member_becomes_data_per_version(
    scene, calls, monkeypatch, referenced, regroup
):
    from unbake import infer, policy, versions

    entries = (("first", "asm", (0, *RETURN)), ("stale", "asm", (STALE_A,)), NEXT)
    snapshot = scene(entries, other=(entries[0], ("stale", "asm", (STALE_B,)), NEXT))
    if regroup != "authored":
        group = replace(snapshot.layout.groups["grp"], evidence="inferred")
        snapshot = layout.overlay(
            snapshot, {"layout.toml": layout.dump_map(replace(snapshot.layout, groups={group.name: group}))}
        )
    calls(BASE + 16, *([BASE + 12] if referenced else []))
    plan, counts = layout.boundary_plan(snapshot)
    if referenced:
        assert not plan.writes  # a stub something calls stays a function
        return
    changed, rows = _rows(snapshot, plan)
    for vid, word in (("a", STALE_A), ("b", STALE_B)):
        pad = f"stale_padding_100C_{vid}"  # the two versions hold different words, so each has its own
        assert "stale" not in rows[vid] and rows[vid][pad][0] == "data"
        assert versions.rom_bytes(
            snapshot.versions[vid], rows[vid][pad][1].rom_start, rows[vid][pad][1].rom_end
        ) == word.to_bytes(4, "big")
        assert plan.writes[f"src/{pad}.c"].startswith(f"unsigned char D_auto_{pad}[4]".encode())
    assert counts["prelude"]["applied"] == 2 and "stale" not in changed.layout.members
    assert {"stale_padding_100C_a", "stale_padding_100C_b"} <= set(changed.layout.groups["grp"].members)
    assert not layout.boundary_plan(changed)[0].writes
    group = changed.layout.groups["grp"]
    inferred = (
        group
        if regroup == "authored"
        else replace(group, name="code_80000400" if regroup == "renamed" else group.name, members=("first", "next"))
    )
    monkeypatch.setattr(infer, "groups", lambda s, *a: (inferred,))
    monkeypatch.setattr(infer, "sdk", lambda s: frozenset())
    monkeypatch.setattr(infer.ownership, "claims", lambda s, units: (units, (), []))
    monkeypatch.setattr(infer.ownership, "exact", lambda s, units: (units, []))
    settled = layout.overlay(changed, infer.plan(changed).writes)
    for unit in settled.layout.units.values():
        assert unit.group in settled.layout.groups
        assert set(unit.members) <= set(settled.layout.groups[unit.group].members)
        assert all(settled.layout.members[n].group == unit.group for n in unit.members)
        text = settled.read(unit.path).decode()
        assert policy.census(settled, changes={unit.path: ("", text)}) == ((), [])
    assert not infer.plan(settled).writes


def test_boundary_fill_folds_the_padding_row_of_an_earlier_run_with_the_leftover_word(scene, calls):
    first = ("first", "asm", RETURN)
    one = (first, ("first_padding_A", "data", (STALE_A,)), ("left_a", "asm", (STALE_B,)), NEXT)
    two = (first, ("first_padding_B", "data", (STALE_A,)), ("left_b", "asm", (STALE_B,)), NEXT)
    snapshot = scene(one, other=two)
    calls(BASE + 16)
    plan, _ = layout.boundary_plan(snapshot)
    changed, rows = _rows(snapshot, plan)
    for vid, pad, left in (("a", "first_padding_A", "left_a"), ("b", "first_padding_B", "left_b")):
        assert left not in rows[vid] and rows[vid][pad][1].size == 8
        assert plan.writes[f"src/{pad}.c"].startswith(f"unsigned char D_auto_{pad}[8]".encode())
    assert not layout.boundary_plan(changed)[0].writes


@pytest.mark.parametrize("scored", [0.5, 1.0])
def test_boundary_fill_after_the_last_return_of_a_fuzzy_member_is_peeled_as_its_tail(scene, calls, scored):
    owner = ("owner", "asm", (0x24020001, JR, 0, STALE_A))
    fuzzy = {"owner": {"path": "src/fuzzy/owner.c", "scores": {"a": scored, "b": scored}}}
    snapshot = scene((owner, NEXT), fuzzy=fuzzy)
    calls(BASE + 16)
    plan, _counts = layout.boundary_plan(snapshot)
    if scored == 1.0:
        assert not plan.writes  # an exact member includes every word it holds
        return
    changed, rows = _rows(snapshot, plan)
    for vid in "ab":
        assert rows[vid]["owner"][1].size == 12 and rows[vid]["owner_padding_100C"][0] == "data"
    assert not layout.boundary_plan(changed)[0].writes


@pytest.mark.parametrize("scored", [0.5, 1.0])
def test_boundary_fill_peels_the_leading_words_of_a_landed_inexact_member_without_renaming_it(scene, calls, scored):
    name = "func_80000408"
    fuzzy = {name: {"path": "src/fuzzy/f.c", "scores": {"a": scored, "b": scored}}}
    entries = (("first", "asm", (0, *RETURN)), (name, "c", (0x00001821, 0x27BDFFE8, 0x24020001, *RETURN)))
    snapshot = scene(entries, units=(_unit((name,)),), fuzzy=fuzzy)
    calls(BASE + 16)
    plan, _ = layout.boundary_plan(snapshot)
    assert bool(plan.writes) == (scored < 1.0)
    if scored < 1.0:
        changed, rows = _rows(snapshot, plan)
        assert rows["a"][name][0] == "c" and rows["a"][name][1].vram == BASE + 16
        assert rows["a"][f"{name}_padding_100C"][0] == "data" and name in changed.layout.units["src/grp.c"].members


@pytest.mark.parametrize("peeled_body", ["held", "own"])
def test_boundary_fill_peels_a_version_whose_body_another_version_holds_though_a_third_differs(scene, peeled_body):
    # a holds the function after one stale word; b holds the same body bare; c is the function changed elsewhere
    body = (0x27BDFFE8, 0x24020001, *RETURN)
    changed = (0x27BDFFE8, 0x24030002, *RETURN)
    name = "func_8000040C"
    peeled = body if peeled_body == "held" else (0x27BDFFE8, 0x24040003, *RETURN)
    snapshot = scene(
        (("first", "asm", (0, *RETURN)), (name, "asm", (STALE_A, *peeled))),
        other=(("first", "asm", (0, *RETURN)), (name, "asm", body)),
        third=(("first", "asm", (0, *RETURN)), (name, "asm", changed)),
    )
    plan, _ = layout.boundary_plan(snapshot)
    assert any("_padding_" in p for p in plan.writes) == (peeled_body == "held")


@pytest.mark.parametrize("words", [(JR, 0x27BD0018), (JR, 0x24020001)])
def test_boundary_epilogue_fragment_member_with_a_fuzzy_row_is_fill_and_loses_the_row(scene, calls, words):
    fuzzy = {"fragment": {"path": "src/fuzzy/fragment.c", "scores": {"a": 0.5, "b": 0.5}}}
    snapshot = scene((("first", "asm", RETURN), ("fragment", "asm", words), NEXT), fuzzy=fuzzy)
    calls(BASE + 16)
    plan, _ = layout.boundary_plan(snapshot)
    if words[1] == 0x24020001:
        assert not plan.writes  # only a return that gives a frame back is another member's epilogue
        return
    changed, rows = _rows(snapshot, plan)
    assert "fragment" not in changed.layout.members and "fragment" not in changed.layout.fuzzy
    assert rows["a"]["fragment_padding_1008"][0] == "data"


@pytest.mark.parametrize("revision", ["head", "private-index-tree"])
def test_tree_capture_reads_blobs_and_hides_untracked_headers(scene, monkeypatch, revision):
    import io
    import tarfile

    original = scene()
    cfg = original.config
    blobs = {"layout.toml": b"HEAD layout", "include/a.h": b"HEAD header", "src/f.c": b"HEAD source"}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, raw in blobs.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    git = Mock(return_value=fixture.native_result(stdout=stream.getvalue()))
    monkeypatch.setattr(layout.process, "git", git)
    monkeypatch.setattr(layout.configuration, "load_project", lambda root, reader: cfg.project)
    monkeypatch.setattr(layout.version_data, "read", lambda config, reader: {})
    monkeypatch.setattr(layout, "load_map", lambda config, versions, text, reader: original.layout)
    monkeypatch.setattr(layout, "_content", lambda *a: "content")
    (cfg.project.root / "include").mkdir(exist_ok=True)
    (cfg.project.root / "include/a.h").write_bytes(b"dirty")
    (cfg.project.root / "include/extra.h").write_bytes(b"untracked")
    captured = layout.tree(cfg, revision, "head")
    assert captured.read("include/a.h") == b"HEAD header"
    assert captured.peek("include/extra.h") is None
    assert captured.tree and captured.commit == "head"
    git.assert_called_once_with(cfg, "archive", "--format=tar", revision)


@pytest.mark.parametrize(
    "prefix,state", [((0x27BDFFC8, 0xAFB7002C, 0x00A0B821), "data"), ((0x27BDFFC8,), "data"), ((0x27BDFFC8,), "asm")]
)
@pytest.mark.parametrize("guard", ["none", "exact", "authored", "disagree", "gap", "segment", "no_restore"])
def test_boundary_recovers_detached_prologue_with_stable_body_name(scene, monkeypatch, prefix, state, guard):
    name, pad = "func_80000410", "old_padding_1008" if state == "data" else "fragment"
    body = (0xAFBF0030, 0x24020001, JR, 0x27BD0038 if guard != "no_restore" else 0)
    entries = (("previous", "c", RETURN), (pad, state, prefix), (name, "asm", body))
    other = (("previous", "c", RETURN), (name, "asm", (*prefix, *body)))
    if guard == "disagree":
        other = (other[0], (name, "asm", (0xAFBF0030, *body)))
    units = (_unit((pad,), path="src/pad.c", kind="data"),) if state == "data" else ()
    fuzzy = {name: {"path": "src/fuzzy/f.c", "scores": {"a": 1.0, "b": 1.0}}} if guard == "exact" else {}
    authored = (
        ({"version": "a", "rom": 0x1008 + 4 * len(prefix), "kind": "function", "name": name},)
        if guard == "authored"
        else ()
    )
    snapshot = scene(entries, other=other, units=units, fuzzy=fuzzy, authored=authored)
    if guard == "segment":
        snapshot = replace(
            snapshot,
            versions={
                h: replace(
                    v, segments=(*v.segments, ("next", 0x1008 + 4 * len(prefix), 0x1100, BASE + 8 + 4 * len(prefix)))
                )
                for h, v in snapshot.versions.items()
            },
        )
    if guard == "gap":
        original = layout.split_states

        def rows(s, v):
            return [(n, st, replace(p, rom_end=p.rom_end - 4) if n == pad else p) for n, st, p in original(s, v)]

        monkeypatch.setattr(layout, "split_states", rows)
    plan, counts = layout.boundary_plan(snapshot)
    if guard != "none":
        assert not plan.writes
        if guard in ("exact", "authored", "disagree"):
            assert counts["prelude"]["withheld"] >= 1
        return
    changed, rows = _rows(snapshot, plan)
    assert pad not in changed.layout.members and pad not in changed.layout.fuzzy
    assert set(rows["a"]) == set(rows["b"]) == {"previous", name}
    assert rows["a"][name][1].rom_start == 0x1008
    assert rows["a"][name][1].size == 4 * (len(prefix) + len(body))
    assert counts["heads"] == 1
    assert plan.writes.get("src/pad.c") is None
    assert "src/pad.c" not in changed.layout.units
    assert not layout.boundary_plan(changed)[0].writes


@pytest.mark.parametrize("frame", [0x27BDFFE8, 0x27BDFFC8, 0x27BDFDB0])
def test_boundary_never_peels_a_real_prologue_to_alignment(scene, frame):
    name = "func_80000408"
    body = (frame, 0xAFBF0014, 0x24020001, JR, 0x27BD0000 | layout._frame(frame))
    snapshot = scene((("previous", "c", RETURN), (name, "asm", body)))
    plan, _ = layout.boundary_plan(snapshot)
    assert not plan.writes


@pytest.mark.parametrize("terminal", ["return", "tail", "loop", "branch_loop"])
@pytest.mark.parametrize("guard", ["none", "target", "exact", "prologue", "reachable_tail"])
def test_boundary_trims_only_unreachable_terminal_tail(scene, calls, terminal, guard):
    transfer = {"return": JR, "tail": 0x08000200, "loop": 0x08000100, "branch_loop": 0x1000FFFF}[terminal]
    ending = (transfer, 0)
    if guard == "reachable_tail":
        ending = (0x14000001, 0)
    tail = (0x27BDFFE8 if guard == "prologue" else STALE_A, STALE_B)
    fuzzy = {"owner": {"path": "src/fuzzy/owner.c", "scores": {"a": 1.0, "b": 1.0}}} if guard == "exact" else {}
    snapshot = scene((("owner", "asm", (*ending, *tail)), NEXT), fuzzy=fuzzy)
    calls(BASE + 16, *([BASE + 8] if guard == "target" else []))
    plan, counts = layout.boundary_plan(snapshot)
    if guard != "none":
        assert counts["fill"] == 0
        assert not any("_padding_" in p for p in plan.writes)
        return
    changed, rows = _rows(snapshot, plan)
    assert counts["fill"] == 2
    assert rows["a"]["owner"][1].size == 8
    assert rows["a"]["owner_padding_1008"][0] == "data"
    assert not layout.boundary_plan(changed)[0].writes


@pytest.mark.parametrize("kind", ["body", "epilogue"])
def test_boundary_merges_fallthrough_fragments_and_removes_fuzzy_alias(scene, kind):
    first = (0x27BDFFE8, 0xAFBF0014, 0x24020001, 0x24420001, 0)
    body = (0x24420001,) if kind == "body" else ()
    fragment = (*body, 0x8FBF0014, JR, 0x27BD0018)
    fuzzy = {"fragment": {"path": "src/fuzzy/fragment.c", "scores": {"a": 0.5, "b": 0.5}}}
    snapshot = scene((("owner", "asm", first), ("fragment", "asm", fragment)), fuzzy=fuzzy)
    plan, counts = layout.boundary_plan(snapshot)
    changed, rows = _rows(snapshot, plan)
    assert counts["merge"]["applied"] == 1
    assert "fragment" not in changed.layout.members and "fragment" not in changed.layout.fuzzy
    assert rows["a"]["owner"][1].size == 4 * (len(first) + len(fragment))


def test_boundary_stale_branch_before_detached_prologue_is_fill(scene, calls):
    stale = (0x8FA40018, 0x84830000, 0x10720001)
    body = (0xAFBF0014, JR, 0x27BD0018)
    a = (("previous", "c", (0, 0, 0, *RETURN)), ("fragment", "asm", (*stale, 0x27BDFFE8)), ("body", "asm", body))
    b = (a[0], ("fragment", "asm", stale), ("body", "asm", (0x27BDFFE8, *body)))
    snapshot = scene(a, other=b)
    calls(BASE + 32)
    plan, counts = layout.boundary_plan(snapshot)
    changed, rows = _rows(snapshot, plan)
    assert counts["heads"] == 1
    assert rows["a"]["body"][1].rom_start == rows["b"]["body"][1].rom_start == 0x1020
    assert "fragment" not in changed.layout.members
    assert counts["fill"] == 2
    assert not layout.boundary_plan(changed)[0].writes


@pytest.mark.parametrize(
    "code,extent",
    [
        ((JR, 0, STALE_A), 2),
        ((0x10000003, 0, JR, 0, 0x24020001, JR, 0), 7),
        ((0x14000003, 0, JR, 0, 0x24020001, JR, 0), 7),
        ((0x1000FFFF, 0), 2),
        ((0x14000005, 0, JR, 0), None),
        ((0x00600008, 0), None),
        ((JR,), None),
        ((0x0C000200, 0, JR, 0), 4),
        ((0x10000000, 0x24020001, JR, 0), 4),
    ],
)
def test_terminal_extent_follows_all_reachable_paths(code, extent):
    assert layout._end(code, BASE) == extent


@pytest.mark.parametrize("clean_exact", [True, False])
def test_boundary_forward_prologue_uses_a_version_already_separated_from_fill(scene, clean_exact):
    name = "func_80000408"
    body = (0x27BDFFE8, 0xAFBF0014, JR, 0x27BD0018)
    a = (("previous", "c", RETURN), (name, "c", (STALE_A, STALE_B, *body)))
    b = (a[0], ("existing_padding", "data", (STALE_A, STALE_B)), (name, "c", body))
    fuzzy = {name: {"path": "src/fuzzy/f.c", "scores": {"a": 0.9, "b": 1.0 if clean_exact else 0.9}}}
    snapshot = scene(a, other=b, units=(_unit((name,)),), fuzzy=fuzzy)
    plan, counts = layout.boundary_plan(snapshot)
    changed, rows = _rows(snapshot, plan)
    assert counts["prelude"]["applied"] == 1
    assert rows["a"][name][1].rom_start == rows["b"][name][1].rom_start == 0x1010
    assert name in changed.layout.units["src/grp.c"].members
    assert snapshot.read(snapshot.versions["b"].split) == changed.read(snapshot.versions["b"].split)
    assert not layout.boundary_plan(changed)[0].writes


@pytest.mark.parametrize("guard", ["landed", "exact_fuzzy", "indirect"])
def test_boundary_does_not_absorb_an_exact_member_or_follow_an_indirect_exit(scene, guard):
    prior = (0x00600008, 0) if guard == "indirect" else (0x24020001, 0)
    units = (_unit(("second",)),) if guard == "landed" else ()
    fuzzy = {"second": {"path": "src/fuzzy/f.c", "scores": {"a": 1.0, "b": 1.0}}} if guard == "exact_fuzzy" else {}
    snapshot = scene((("first", "asm", prior), ("second", "asm", RETURN)), units=units, fuzzy=fuzzy)
    assert not layout.boundary_plan(snapshot)[0].writes


def test_physical_splits_where_sources_set_other_macro_switches(scene, monkeypatch):
    entries = (("fa", "c", RETURN), ("fb", "c", RETURN))
    units = (_unit(("fa",), "src/fa.c"), _unit(("fb",), "src/fb.c"))
    snapshot = scene(entries, units=units)
    switches = {"src/fa.c": b"#undef F3DEX_GBI_2\nint a;\n", "src/fb.c": b"int b;\n"}
    monkeypatch.setattr(type(snapshot), "peek", lambda self, p: switches.get(p))
    result = layout.physical(snapshot, snapshot.layout)
    assert {p: u.members for p, u in result.items()} == {"src/grp.c": ("fa",), "src/grp_2.c": ("fb",)}
    assert layout._defines(snapshot, snapshot.layout.units["src/fa.c"]) == ("#undef F3DEX_GBI_2",)


def test_the_rows_of_a_version_read_only_its_split_text_so_the_claim_rows_key_names_all_it_reads(scene):
    snapshot = scene((("func_80000400", "asm", RETURN), ("tbl", "rodata", (1, 2))))
    version = snapshot.versions["a"]
    read = []
    reader = snapshot.read
    layout.effort._memo.clear()
    layout.version_data.rows(version, lambda path: read.append(path) or reader(path))
    assert set(read) == {version.split}  # the key's other parts: the version id and its segments


def test_physical_keeps_a_data_unit_whose_members_no_group_lists(scene):
    entries = (("fa", "c", RETURN), ("data/unresolved/80000408", "data", (1,)))
    units = (_unit(("fa",)), _unit(("data/unresolved/80000408",), "src/data.unresolved.80000408.c", kind="data"))
    snapshot = scene(entries, units=units)
    group = snapshot.layout.groups["grp"]
    mapped = replace(snapshot.layout, groups={"grp": replace(group, members=("fa",))})
    result = layout.physical(snapshot, mapped)
    assert {p: u.members for p, u in result.items()} == {
        "src/grp.c": ("fa",),
        "src/data.unresolved.80000408.c": ("data/unresolved/80000408",),
    }


@pytest.mark.parametrize("first", [True, False])
def test_physical_places_a_member_two_groups_list_in_one_unit_the_group_of_its_unit_today(scene, first):
    entries = (("fa", "c", RETURN), ("dz", "data", (3,)))
    units = (_unit(("fa",)), _unit(("dz",), "src/dz.c", kind="data"))
    snapshot = scene(entries, units=units)
    grp = snapshot.layout.groups["grp"]
    two = Group("two", "main", ("dz",), "inferred", (), False, "src/two.c", "")
    groups = {"two": two, "grp": grp} if first else {"grp": grp, "two": two}
    result = layout.physical(snapshot, replace(snapshot.layout, groups=groups))
    assert [m for u in result.values() for m in u.members].count("dz") == 1  # linked twice, its bytes would double
    assert {p: u.members for p, u in result.items()} == {"src/grp.c": ("fa", "dz")}


@pytest.mark.parametrize(
    ("data", "apart"), [("const float D_k[] = { 0.5f };\n", True), ("float D_k[] = { 0.5f };\n", False)]
)
def test_physical_keeps_a_const_apart_from_code_of_another_unit_that_reads_it(scene, monkeypatch, data, apart):
    entries = (("fa", "c", RETURN), ("rodata/fa/80000408", "data", (1,)))
    units = (_unit(("fa",), "src/fa.c"), _unit(("rodata/fa/80000408",), "src/k.c", kind="data"))
    snapshot = scene(entries, units=units)
    files = {"src/fa.c": b"extern const float D_k[];\nfloat fa(void) { return D_k[0]; }\n", "src/k.c": data.encode()}
    monkeypatch.setattr(type(snapshot), "peek", lambda self, p: files.get(p))
    result = layout.physical(snapshot, snapshot.layout)
    assert len(result) == (2 if apart else 1)  # in one file the compiler folds the read into 0.5f


def _declaring(scene, monkeypatch, texts):
    entries = (("fa", "c", RETURN), ("fb", "c", RETURN))
    units = (_unit(("fa",), "src/fa.c"), _unit(("fb",), "src/fb.c"))
    snapshot = scene(entries, units=units)
    files = {p: t.encode() for p, t in texts.items()}
    monkeypatch.setattr(type(snapshot), "peek", lambda self, p: files.get(p))
    split = []
    result = layout.physical(snapshot, snapshot.layout, split)
    return {p: u.members for p, u in result.items()}, split


def test_physical_keeps_apart_a_unit_that_declares_a_name_per_version_and_one_that_declares_it_once(scene, monkeypatch):
    branches = "#ifdef VERSION_EU\nextern struct S X;\n#else\nextern char X[];\n#endif\n"
    units, split = _declaring(
        scene,
        monkeypatch,
        {
            "src/fa.c": branches + "int fa(void) { return X != 0; }\n",
            "src/fb.c": "extern char X[];\nint fb(void) { return X[0]; }\n",
        },
    )
    assert len(units) == 2 and [f.reason.split()[0] for f in split] == ["X"]


def test_physical_keeps_apart_a_unit_that_calls_a_function_undeclared_and_one_that_prototypes_it(scene, monkeypatch):
    units, split = _declaring(
        scene,
        monkeypatch,
        {
            "src/fa.c": "int fa(int *p) { return g(p, 2); }\n",  # implicit: any arguments
            "src/fb.c": "int g(int);\nint fb(void) { return g(1); }\n",
        },
    )
    assert len(units) == 2 and [f.reason.split()[0] for f in split] == ["g"]  # g(int) would refuse g(p, 2)


def test_physical_keeps_apart_a_type_one_unit_defines_and_a_header_of_the_other_defines_the_same_way(
    scene, monkeypatch
):
    pair = "typedef struct Pair { int a; } Pair;\n"
    units, split = _declaring(
        scene,
        monkeypatch,
        {
            "include/h.h": pair,
            "src/fa.c": pair + "int fa(Pair *p) { return p->a; }\n",
            "src/fb.c": '#include "h.h"\nint fb(Pair *p) { return p->a; }\n',
        },
    )
    assert len(units) == 2 and split  # one file would define Pair twice


@pytest.mark.parametrize("other", ["return x + 1.0f;", "return x;"])  # bodies that differ, and the same one
def test_physical_keeps_apart_units_that_each_define_their_own_static_helper_of_one_name(scene, monkeypatch, other):
    helper = "static inline float wrap(float x) {{ {} }}\n"
    units, split = _declaring(
        scene,
        monkeypatch,
        {
            "src/fa.c": helper.format("return x;") + "int fa(void) { return wrap(1.0f) != 0; }\n",
            "src/fb.c": helper.format(other) + "int fb(void) { return wrap(2.0f) != 0; }\n",
        },
    )
    assert len(units) == 2 and [f.reason.split()[0] for f in split] == ["wrap"]  # two definitions in one file


def test_physical_keeps_apart_units_whose_extern_differs_from_a_header_the_other_includes(scene, monkeypatch):
    units, split = _declaring(
        scene,
        monkeypatch,
        {
            "src/fa.c": '#include "h.h"\nint fa(void) { return X != 0; }\n',
            "src/fb.c": "extern struct B *X;\nint fb(void) { return X == 0; }\n",
            "include/h.h": "extern struct A *X;\n",
        },
    )
    assert units == {"src/grp.c": ("fa",), "src/grp_2.c": ("fb",)}
    assert [(f.key, f.reason.split()[0], f.unit, f.blocking) for f in split] == [
        ("layout.decl_split", "X", "src/fa.c, src/fb.c", False)
    ]


def test_physical_keeps_apart_a_definition_and_a_declaration_spelled_otherwise(scene, monkeypatch):
    units, split = _declaring(
        scene,
        monkeypatch,
        {
            "src/fa.c": "int fa(int a) { return a; }\n",
            "src/fb.c": "extern int fa(void);\nint fb(void) { return fa(); }\n",
        },
    )
    assert units == {"src/grp.c": ("fa",), "src/grp_2.c": ("fb",)} and [f.reason.split()[0] for f in split] == ["fa"]


@pytest.mark.parametrize(
    "header, extern",
    [
        ("extern float X;\n", "extern const float X;\n"),  # one file cannot hold both qualifiers
        ("#define X Y.pad[3]\n", "extern char X;\n"),  # the header's macro would rewrite the declaration
        ("extern void (*X)(void);\n", "extern void (*X)(s32 offset);\n"),  # function pointers of two types
    ],
)
def test_physical_keeps_apart_units_that_one_header_would_break(scene, monkeypatch, header, extern):
    units, split = _declaring(
        scene,
        monkeypatch,
        {
            "src/fa.c": '#include "h.h"\nint fa(void) { return 1; }\n',
            "src/fb.c": extern + "int fb(void) { return X != 0; }\n",
            "include/h.h": header,
        },
    )
    assert units == {"src/grp.c": ("fa",), "src/grp_2.c": ("fb",)} and [f.reason.split()[0] for f in split] == ["X"]


def test_physical_reads_what_a_unit_builds_with_today_without_the_module_header_it_never_includes(scene, monkeypatch):
    entries = (("fa", "c", RETURN), ("fb", "c", RETURN))
    snapshot = scene(entries, units=(_unit(("fa",), "src/fa.c"), _unit(("fb",), "src/fb.c")))
    module = snapshot.layout.groups["grp"].header
    files = {
        "src/fa.c": f'#include "{module.removeprefix("include/")}"\nint fa(void) {{ return X != 0; }}\n',
        "src/fb.c": "extern struct B *X;\nint fb(void) { return X == 0; }\n",
        module: "extern void *X;\n",
    }
    monkeypatch.setattr(type(snapshot), "peek", lambda self, p: files.get(p, b"").encode() if p in files else None)
    split = []
    result = layout.physical(snapshot, snapshot.layout, split)
    assert {p: u.members for p, u in result.items()} == {"src/grp.c": ("fa",), "src/grp_2.c": ("fb",)}
    assert [f.reason.split()[0] for f in split] == ["X"]


def test_physical_joins_siblings_whose_externs_differ_and_no_header_declares_the_name(scene, monkeypatch):
    units, split = _declaring(
        scene,
        monkeypatch,
        {
            "src/fa.c": "extern struct A *X;\nint fa(void) { return X != 0; }\n",
            "src/fb.c": "extern struct B *X;\nint fb(void) { return X == 0; }\n",
        },
    )
    assert units == {"src/grp.c": ("fa", "fb")} and split == []  # compose moves each extern to block scope


def test_spellings_count_block_scope_declarations_and_drop_declarations_nothing_uses():
    text = "extern int unused;\nextern struct A *X;\nint f(void) {\n    s32 g();\n    extern void h(int a);\n"
    text += "    return X != 0;\n}\n"
    found = layout._spellings(text, True)
    assert set(found) == {"X", "f", "g", "h"}  # `unused` is dropped as compose drops it
    assert found["g"] == {"int@()": {"block"}} and found["h"] == {"void@(int)": {"block"}}


def test_declaration_tables_are_parsed_once_per_text_and_never_again_by_a_later_command(scene, monkeypatch):
    texts = {
        "src/fa.c": '#include "h.h"\nint fa(void) { return X != 0; }\n',
        "src/fb.c": '#include "h.h"\nextern struct B *X;\nint fb(void) { return X == 0; }\n',
        "include/h.h": "extern struct A *X;\n",
    }
    parsed = []
    spellings = layout._spellings
    monkeypatch.setattr(layout, "_spellings", lambda text, source: parsed.append(text) or spellings(text, source))
    _declaring(scene, monkeypatch, texts)
    assert parsed.count(texts["include/h.h"]) == 1  # both units reach the header: one parse
    parsed.clear()
    layout.effort._memo.clear()  # the next command
    _declaring(scene, monkeypatch, texts)
    assert parsed == []  # every table comes from the host cache
