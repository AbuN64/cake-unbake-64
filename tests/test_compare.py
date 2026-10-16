"""compare: holders, gaps, bind, measure and run with native, pool and land mocked."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import fixture
import pytest

from unbake import compare, journal, moved, ownership, store
from unbake import config as configuration
from unbake.contracts import (
    Config,
    Finding,
    Group,
    LayoutMap,
    Member,
    Placement,
    Recipe,
    Refusal,
    Snapshot,
    UnitSpec,
    digest,
)

RECIPE = Recipe("gcc-test", (), (), (), "recipe")
REAL_DERIVE, REAL_PAIRS = ownership.derive, ownership.derive_pairs
UNIT = "src/a.c"


@pytest.fixture(autouse=True)
def _no_ownership(monkeypatch):  # the units here have no source to compile: they own what they list
    monkeypatch.setattr(ownership, "derive", lambda snapshot, unit, loose=False, recipe=None: (snapshot, unit))
    monkeypatch.setattr(ownership, "derive_pairs", lambda pairs, loose=False, recipe=None: list(pairs))


def _each(submit: Any) -> Any:
    """A land.submit_many over a fake land.submit: each call's Refusal is its own result."""
    keys = ("request", "unit", "proofs", "source", "origin", "extras", "withhold", "snapshot")

    def many(config: Config, calls: Any) -> list[Any]:
        out = []
        for call in calls:
            try:
                out.append(submit(config, *(call[k] for k in keys)))
            except Refusal as refusal:
                out.append(refusal)
        return out

    return many


def member(name: str, versions: tuple[str, ...] = ("a", "b")) -> Member:
    places = tuple(Placement(v, ".text", 0x1000, 0x1008, 0x80000400) for v in versions)
    return Member(name, "function", "c", "g", places)


def unit(*names: str, path: str = UNIT) -> UnitSpec:
    return UnitSpec(path, "c", "g", tuple(names), "gcc-test", {})


def snap(config: Config, members: dict[str, Member], units: tuple[UnitSpec, ...] = ()) -> Snapshot:
    layout = LayoutMap({}, members, {u.path: u for u in units}, "d", (), {})
    return Snapshot(config, "c" * 40, layout, {}, {}, "s")


def test_holders_union_sorted(cfg: Config) -> None:
    s = snap(cfg, {"f": member("f", ("b",)), "g": member("g", ("a", "c"))})
    assert compare.holders(s, unit("f", "g")) == ("a", "b", "c")


@pytest.mark.parametrize("exact_b", [True, False])
def test_gaps_per_member(cfg: Config, exact_b: bool) -> None:
    s = snap(cfg, {"f": member("f")})
    proofs = [fixture.proof(UNIT, "f", "a", True)]
    proofs.append(fixture.proof(UNIT, "f", "b", exact_b, () if exact_b else ("bytes differ",)))
    found = compare.gaps(s, unit("f"), proofs)
    assert bool(found) is (not exact_b)
    if found:
        assert found[0].key == "land.not_exact"
        assert found[0].versions == ("b",)


def test_gap_missing_measurement(cfg: Config) -> None:
    s = snap(cfg, {"f": member("f")})
    found = compare.gaps(s, unit("f"), [fixture.proof(UNIT, "f", "a", True)])
    assert found[0].versions == ("b",)
    assert found[0].missing == ("version b: no measurement",)
    assert found[0].symptoms == {"no_measurement": True}


def test_gap_symptoms_merged(cfg: Config) -> None:
    s = snap(cfg, {"f": member("f")})
    bad = fixture.proof(UNIT, "f", "a", False, ("size 8 != 12",), symptoms={"register_dominant": True})
    found = compare.gaps(s, unit("f"), [bad])
    facts = found[0].symptoms
    assert facts["register_dominant"] is True  # from the proof
    assert facts["size_delta_bytes"] == -4  # from the missing text
    assert facts["no_measurement"] is True  # version b was never measured


def test_bind_known_unit(cfg: Config) -> None:
    u = unit("f", path="src/a.c")
    s = snap(cfg, {"f": member("f")}, (u,))
    file = cfg.project.root / "src/a.c"
    assert compare.bind(s, file, None) == (u, s)


def test_bind_new_text_for_a_landed_member_replaces_its_unit_file(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    u = unit("f", "g", path="src/a.c")
    s = snap(cfg, {"f": member("f"), "g": member("g")}, (u,))
    file = tmp_path / "f.c"
    file.write_bytes(b"int f;")
    over = replace(s, digest="overlaid")
    writes = []
    monkeypatch.setattr(compare.layout, "overlay", lambda snapshot, w: writes.append(w) or over)
    monkeypatch.setattr(compare.layout, "unit_options", lambda *a: pytest.fail("a landed member needs no new unit"))
    assert compare.bind(s, file, None) == (u, over)
    assert writes == [{"src/a.c": b"int f;"}]


def test_bind_respells_a_candidate_written_before_a_move_through_the_chains(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    u = unit("f", path="src/a.c")
    s = snap(cfg, {"f": member("f")}, (u,))
    moved.record(
        cfg,
        [
            ("path", "include/old/a.h", "include/mid/a.h"),
            ("path", "include/mid/a.h", "include/new/a.h"),
            ("symbol", "old_fn", "mid_fn"),
            ("symbol", "mid_fn", "new_fn"),
        ],
        "d",
    )
    file = tmp_path / "f.c"
    file.write_bytes(b'#include "old/a.h"\nint f(void) { return old_fn() + mid_fn() + old_fn_x(); }\n')
    writes: list[Any] = []
    monkeypatch.setattr(compare.layout, "overlay", lambda snapshot, w: writes.append(w) or snapshot)
    notes: list[str] = []
    assert compare.bind(s, file, None, notes) == (u, s)
    assert writes[0]["src/a.c"] == b'#include "new/a.h"\nint f(void) { return new_fn() + new_fn() + old_fn_x(); }\n'
    assert notes == ["include/old/a.h->include/new/a.h", "mid_fn->new_fn", "old_fn->new_fn"]


def test_a_candidate_untouched_by_the_moves_is_not_rewritten(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    u = unit("f", path="src/a.c")
    s = snap(cfg, {"f": member("f")}, (u,))
    moved.record(cfg, [("symbol", "old_fn", "new_fn")], "d")
    file = tmp_path / "f.c"
    file.write_bytes(b"int f;")
    writes: list[Any] = []
    monkeypatch.setattr(compare.layout, "overlay", lambda snapshot, w: writes.append(w) or snapshot)
    notes: list[str] = []
    compare.bind(s, file, None, notes)
    assert notes == [] and writes == [{"src/a.c": b"int f;"}]


def test_the_moved_table_is_read_once_until_the_stream_grows(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    moved.record(cfg, [("symbol", "a", "b")], "d")
    reads = []
    real = moved.store.rows
    monkeypatch.setattr(moved.store, "rows", lambda c, s: reads.append(s) or real(c, s))
    for _ in range(5):
        assert moved.table(cfg) == ({}, {"a": "b"})
    assert reads == ["moved"]
    moved.record(cfg, [("symbol", "b", "c")], "d2")
    assert moved.table(cfg) == ({}, {"a": "c", "b": "c"}) and reads == ["moved", "moved"]


def test_an_old_name_that_exists_again_is_never_respelled(cfg: Config) -> None:
    moved.record(cfg, [("path", "include/a.h", "include/b.h"), ("symbol", "f", "g"), ("symbol", "h", "k")], "d")
    s = replace(snap(cfg, {"f": member("f")}), overlays={"include/a.h": b"reborn\n"})
    text = b'#include "a.h"\nint x(void) { return f() + h(); } /* h */ char *y = "h";\n'
    assert moved.rewrite(s, text, "include/x.c") == (
        b'#include "a.h"\nint x(void) { return f() + k(); } /* h */ char *y = "h";\n',
        ["h->k"],
    )


def test_a_name_moved_and_moved_back_is_live_again(cfg: Config) -> None:
    moved.record(cfg, [("symbol", "a", "b"), ("symbol", "b", "a"), ("path", "x.h", "y.h")], "d")
    assert moved.table(cfg) == ({"x.h": "y.h"}, {"b": "a"})


def test_bind_missing_file_refused_by_name(cfg: Config, tmp_path: Path) -> None:
    s = snap(cfg, {"f": member("f")})
    with pytest.raises(Refusal) as error:
        compare.bind(s, tmp_path / "absent.c", "f")
    assert error.value.findings[0].key == "land.request" and "absent.c" in error.value.findings[0].reason


def test_bind_unknown_member_refused(cfg: Config) -> None:
    s = snap(cfg, {"f": member("f")})
    with pytest.raises(Refusal) as error:
        compare.bind(s, Path("/elsewhere/nope.c"), None)
    assert error.value.findings[0].key == "land.request"


def test_bind_function_names_member_and_overlays(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = snap(cfg, {"f": member("f")})
    file = tmp_path / "draft.c"
    file.write_bytes(b"int f;")
    u = unit("f")
    over = replace(s, digest="overlaid")
    monkeypatch.setattr(compare.layout, "unit_options", lambda snapshot, name, source: [(None, {}), (u, {"x": b"1"})])
    monkeypatch.setattr(compare.layout, "overlay", lambda snapshot, writes: over)
    assert compare.bind(s, file, "f") == (u, over)


def test_bind_no_options_refused(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = snap(cfg, {"f": member("f")})
    file = tmp_path / "f.c"
    file.write_bytes(b"")
    monkeypatch.setattr(compare.layout, "unit_options", lambda *a: [])
    with pytest.raises(Refusal) as error:
        compare.bind(s, file, "f")
    assert "no standalone unit option" in error.value.findings[0].reason


def test_bind_a_c_source_never_takes_a_landed_assembly_unit(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asm = UnitSpec("src/sdk/f.s", "asm", "g", ("f",), "gcc-test", {})
    s = snap(cfg, {"f": member("f")}, (asm,))
    file = tmp_path / "f.c"
    file.write_bytes(b"void f(void) {}")
    own = unit("f", path="src/f.c")
    over = replace(s, digest="overlaid")
    monkeypatch.setattr(compare.layout, "unit_options", lambda snapshot, name, source: [(own, {"src/f.c": source})])
    monkeypatch.setattr(compare.layout, "overlay", lambda snapshot, writes: over)
    assert compare.bind(s, file, None) == (own, over)  # its own C unit, never C text at the .s path
    monkeypatch.setattr(compare.layout, "unit_options", lambda *a: [])
    with pytest.raises(Refusal) as error:
        compare.bind(s, file, None)
    assert "src/sdk/f.s (asm) holds it" in error.value.findings[0].reason


def test_bind_names_the_storage_a_source_defines_when_no_member_matches(cfg: Config, tmp_path: Path) -> None:
    s = snap(cfg, {"f": member("f")})
    file = tmp_path / "d.c"
    file.write_bytes(b"int table[4];")
    with pytest.raises(Refusal) as error:
        compare.bind(s, file, None)
    assert error.value.findings[0].reason.endswith("it defines table")


def mock_measure(
    monkeypatch: pytest.MonkeyPatch, seen: list[Path], exact: bool = True, keys: list[Any] | None = None
) -> None:
    keys = [] if keys is None else keys
    monkeypatch.setattr(compare.recipes, "resolve", lambda *a: RECIPE)

    def fake_map(config: Config, name: str, function: Any, items: list[Any], key: Any = None) -> list[Any]:
        if name == "compare.stamps":
            return [digest((item[0].digest, item[1], item[2], item[3])) for item in items]
        keys.append(key)
        if function is compare._export:
            return [function(item) for item in items]
        seen.extend(item[4] for item in items)
        return [(fixture.proof(UNIT, "f", item[2], exact, () if exact else ("bytes differ",)),) for item in items]

    monkeypatch.setattr(compare.pool, "map", fake_map)


def test_measure_refuses_existing_out(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = snap(cfg, {"f": member("f")})
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(Refusal) as error:
        compare.measure(s, unit("f"), {"add": [], "omit": []}, out)
    assert error.value.findings[0].key == "compare.out"


def test_work_dir_removed_after_measure(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []
    mock_measure(monkeypatch, seen)
    s = snap(cfg, {"f": member("f")})
    proofs = compare.measure(s, unit("f"), {"add": [], "omit": []}, None)
    assert [p.version for p in proofs] == ["a", "b"]
    assert [p.name for p in seen] == ["a", "b"]  # per-holder job dirs under one work dir
    assert seen[0].parent == seen[1].parent
    assert not seen[0].parent.exists()


def test_measure_exports_before_work_removed(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_measure(monkeypatch, [])
    s = snap(cfg, {"f": member("f")})
    exported: list[Path] = []

    def export(item: tuple) -> None:
        _snapshot, _u, _version, _recipe, work, _proofs, target = item
        exported.append(work.parent)
        assert work.parent.exists()
        target.mkdir(parents=True)
        (target / "result.json").write_text("{}")

    monkeypatch.setattr(compare, "_export", export)
    calls = []
    inner = compare.pool.map
    monkeypatch.setattr(compare.pool, "map", lambda *a: calls.append((a[1], len(a[3]))) or inner(*a))
    out = tmp_path / "out"
    compare.measure(s, unit("f"), {"add": [], "omit": []}, out)
    assert sorted(p.name for p in out.iterdir()) == ["a", "b"]
    assert calls == [("compare.measure", 2), ("compare.export", 2)]
    assert not (out.parent / ".tmp-out").exists()
    assert not exported[0].exists()


def mock_run(
    monkeypatch: pytest.MonkeyPatch, cfg: Config, u: UnitSpec, exact: bool, submitted: Any, calls: dict[str, Any]
) -> Snapshot:
    s = snap(cfg, {n: member(n) for n in u.members}, (u,))
    seen: list[Path] = []
    mock_measure(monkeypatch, seen, exact)
    monkeypatch.setattr(compare.layout, "capture", lambda config: s)
    monkeypatch.setattr(compare, "bind", lambda snapshot, file, function, notes=None: (u, snapshot))

    def fake_submit(
        config: Config,
        request: dict[str, Any],
        spec: UnitSpec,
        proofs: Any,
        source: bytes,
        origin: str,
        extras: dict[str, bytes],
        withhold: bool = False,
        snapshot: Any = None,
    ) -> Any:
        calls["submit"] = (request, origin, source)
        calls["extras"] = extras
        calls["withhold"] = withhold
        calls["snapshot"] = snapshot
        return submitted

    def fake_drain(config: Config, only: str | None = None) -> dict[str, Any]:
        calls["drain"] = calls.get("drain", 0) + 1
        calls["only"] = only
        return {"running": False, "landed": [], "refused": []}

    monkeypatch.setattr(compare.land, "submit", fake_submit)
    monkeypatch.setattr(compare.land, "submit_many", _each(fake_submit))
    monkeypatch.setattr(compare.land, "drain", fake_drain)
    return s


def params(tmp_path: Path, **over: Any) -> dict[str, Any]:
    file = tmp_path / "f.c"
    file.write_bytes(b"int f;")
    base = {"file": file, "function": None, "toolchain": None, "flag": ["-g"], "omit_flag": [], "out": None}
    base["note"] = None
    base["withhold"] = False
    return base | over


def validate(result: dict[str, Any]) -> None:
    configuration.validate("result.compare", json.loads(json.dumps(result)), "result.compare")


def test_run_exact_submits_and_drains(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    s = mock_run(monkeypatch, cfg, unit("f"), True, fixture.submission(id="sub1"), calls)
    result = compare.run(cfg, params(tmp_path, note="trick"))
    assert result["submitted"] == "sub1"
    assert calls["drain"] == 1 and calls["only"] == "sub1"  # it lands its own submission, not the whole inbox
    assert calls["snapshot"] is s  # the project it measured against is read once
    assert result["drain"] == {"running": False, "landed": [], "refused": []}
    assert result["exact"] is True
    assert result["score"] == {"a": 1.0, "b": 1.0}
    request, origin, source = calls["submit"]
    assert origin == "compare" and source == b"int f;"
    assert request["function"] == "f" and request["note"] == "trick"
    assert request["overrides"] == {"add": ["-g"], "omit": []}


def test_run_withhold_goes_through_submit_and_the_same_drain(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    mock_run(monkeypatch, cfg, unit("f"), False, fixture.submission(id="sub2"), calls)
    result = compare.run(cfg, params(tmp_path, withhold=True))
    assert calls["withhold"] is True and calls["drain"] == 1 and result["submitted"] == "sub2"


def test_run_not_exact_submits_nothing(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    mock_run(monkeypatch, cfg, unit("f"), False, None, calls)
    result = compare.run(cfg, params(tmp_path, toolchain="gcc-test"))
    assert "drain" not in calls
    assert result["submitted"] is None and result["drain"] is None
    assert result["exact"] is False
    assert result["gaps"]
    assert calls["submit"][0]["overrides"]["toolchain"] == "gcc-test"


@pytest.mark.parametrize("exact,submitted", [(True, "sub"), (False, None)])
def test_run_result_validates_schema(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exact: bool, submitted: str | None
) -> None:
    sub = fixture.submission(id=submitted) if submitted else None
    mock_run(monkeypatch, cfg, unit("f"), exact, sub, {})
    validate(compare.run(cfg, params(tmp_path)))


def test_multi_member_file_needs_function(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    mock_run(monkeypatch, cfg, unit("f", "g"), True, None, calls)
    with pytest.raises(Refusal) as error:
        compare.run(cfg, params(tmp_path))
    assert error.value.findings[0].key == "land.request"
    assert "--function" in error.value.findings[0].reason
    result = compare.run(cfg, params(tmp_path, function="g"))
    assert calls["submit"][0]["function"] == "g"
    assert result["members"] == ["f", "g"]


def test_run_audits_the_derived_claim_rows(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = unit("f", "old_row")
    calls: dict[str, Any] = {}
    mock_run(monkeypatch, cfg, original, True, None, calls)
    claimed = unit("f", "rodata/f/00001000")
    after = snap(cfg, {"f": member("f"), "rodata/f/00001000": member("rodata/f/00001000", ("a",))}, (claimed,))
    monkeypatch.setattr(ownership, "derive", lambda snapshot, spec, **_: (after, claimed))
    proofs = [fixture.proof(UNIT, "f", v, True) for v in ("a", "b")]
    proofs.append(fixture.proof(UNIT, "rodata/f/00001000", "a", True))
    monkeypatch.setattr(compare, "measure", lambda snapshot, spec, overrides, out: tuple(proofs))
    result = compare.run(cfg, params(tmp_path, function="f"))
    assert result["exact"] and result["gaps"] == []
    assert result["members"] == list(claimed.members)


def test_gaps_require_only_the_claim_rows_holders(cfg: Config) -> None:
    row = member("rodata/f/00001000", ("b",))
    snapshot = snap(cfg, {"f": member("f"), row.name: row})
    proofs = [fixture.proof(UNIT, "f", v, True) for v in ("a", "b")]
    assert compare.gaps(snapshot, unit("f", row.name), proofs)[0].versions == ("b",)
    proofs.append(fixture.proof(UNIT, row.name, "b", True))
    assert compare.gaps(snapshot, unit("f", row.name), proofs) == ()


def test_measure_many_runs_all_variants_in_one_pool_map(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []
    calls: list[int] = []
    mock_measure(monkeypatch, seen)
    inner = compare.pool.map
    monkeypatch.setattr(compare.pool, "map", lambda *a: calls.append(len(a[3])) or inner(*a))
    s = snap(cfg, {"f": member("f")})
    proofs = compare.measure_many(s, unit("f"), [{"add": [], "omit": []}, {"add": ["-O1"], "omit": []}])
    assert calls == [4] and [[p.version for p in each] for each in proofs] == [["a", "b"], ["a", "b"]]
    assert len({p.parent / p.name for p in seen}) == 4  # each variant has its own job directory


def test_submit_lands_an_exact_file_with_its_change_set_and_validates(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    mock_run(monkeypatch, cfg, unit("f"), True, fixture.submission(id="sub3", operation="publish"), calls)
    extra = cfg.project.root / "include/h.h"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"extern int f;")
    result = compare.submit_command(
        cfg,
        {"files": [params(tmp_path)["file"], extra], "function": None, "note": None, "withhold": False, "bulk": False},
    )
    row = result["submissions"][0]
    assert row["id"] == "sub3" and row["exact"] and row["operation"] == "publish" and calls["drain"] == 1
    assert calls["submit"][1] == "submit" and calls["extras"] == {"include/h.h": b"extern int f;"}
    configuration.validate("result.submit", json.loads(json.dumps(result)), "result.submit")


def test_submit_bulk_groups_a_source_with_its_path_equals_file_extras_and_a_refusal_blocks_only_its_group(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    mock_run(monkeypatch, cfg, unit("f"), False, fixture.submission(id="sub4", operation="fuzzy"), calls)
    header = cfg.project.root / "include/h.h"
    header.parent.mkdir(parents=True)
    header.write_bytes(b"old")
    new = tmp_path / "new.h"
    new.write_bytes(b"new")
    first, bad = params(tmp_path)["file"], tmp_path / "bad.c"
    inner = compare.bind

    def bind(snapshot: Snapshot, file: Path, function: str | None, notes: Any = None) -> Any:
        if file == bad:
            raise Refusal(Finding("land.request", "no such member"))
        return inner(snapshot, file, function)

    monkeypatch.setattr(compare, "bind", bind)
    extra = Path(f"include/h.h={new}")
    result = compare.submit_command(
        cfg,
        {
            "files": [bad, extra, first, extra, first],
            "function": None,
            "note": None,
            "withhold": False,
            "bulk": True,
            "toolchain": None,
            "flag": ["-g"],
            "omit_flag": [],
        },
    )
    rows = result["submissions"]
    assert [r["id"] for r in rows] == [None, "sub4", "sub4"] and rows[0]["findings"][0]["key"] == "land.request"
    assert [len(r["files"]) for r in rows] == [2, 2, 1]  # the extra travels with its source, never alone
    assert rows[1]["operation"] == "fuzzy" and result["drain"] is None and "drain" not in calls
    assert calls["extras"] == {} and calls["submit"][0]["overrides"] == {"add": ["-g"], "omit": []}  # last group alone
    compare.submit_command(
        cfg,
        {
            "files": [first, extra],
            "function": None,
            "note": None,
            "withhold": False,
            "bulk": True,
            "toolchain": None,
            "flag": [],
            "omit_flag": [],
        },
    )
    assert calls["extras"] == {"include/h.h": b"new"}
    configuration.validate("result.submit", json.loads(json.dumps(result)), "result.submit")


def test_trial_options_make_the_recipe_that_places_the_emitted_data(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_run(monkeypatch, cfg, unit("f"), True, None, {})
    seen: list[Recipe] = []
    monkeypatch.setattr(
        compare.recipes, "resolve", lambda config, spec, o: Recipe("gcc-test", (), tuple(o["add"]), (), "r")
    )
    monkeypatch.setattr(
        ownership, "derive", lambda snapshot, spec, loose=False, recipe=None: seen.append(recipe) or (snapshot, spec)
    )
    compare.run(cfg, params(tmp_path))
    compare.measure_many(snap(cfg, {"f": member("f")}), unit("f"), [{"add": ["-g"], "omit": []}])
    assert seen and all(r.cflags == ("-g",) for r in seen)


def test_submit_withhold_asks_land_to_withhold_a_partial_file(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    mock_run(monkeypatch, cfg, unit("f"), False, None, calls)
    result = compare.submit_command(
        cfg, {"files": [params(tmp_path)["file"]], "function": None, "note": None, "withhold": True, "bulk": False}
    )
    row = result["submissions"][0]
    assert calls["withhold"] is True and row["id"] is None and row["operation"] == "none" and not row["exact"]
    assert row["findings"] and result["drain"] is None
    configuration.validate("result.submit", json.loads(json.dumps(result)), "result.submit")


def test_measure_names_each_job_by_what_it_reads_unless_it_exports(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys: list[Any] = []
    mock_measure(monkeypatch, [], keys=keys)
    monkeypatch.setattr(compare, "_export", lambda item: item[6].mkdir(parents=True))
    s = snap(cfg, {"f": member("f")})
    compare.measure(s, unit("f"), {"add": [], "omit": []}, None)
    assert keys == [compare._key]  # a warm measurement never travels to a worker
    compare.measure(s, unit("f"), {"add": [], "omit": []}, tmp_path / "out")
    assert keys[1] is None  # an export links, cached or not, so its jobs always run


def test_the_key_of_a_measurement_is_the_stamp_of_its_own_recipe(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple] = []
    monkeypatch.setattr(compare.native, "stamp", lambda *a: seen.append(a) or "stamp")
    item = ("snapshot", "unit", "a", RECIPE, Path("work"), False)
    assert compare._key(item) == "stamp" and seen == [("snapshot", "unit", "a", RECIPE)]


def test_an_export_links_again_so_a_cached_proof_still_leaves_the_disassembly(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs: list[tuple] = []

    def fake_map(config: Config, name: str, function: Any, items: list[Any], key: Any = None) -> list[Any]:
        if name == "compare.export":
            for item in items:
                item[6].mkdir(parents=True)
            return []
        jobs.extend(items)
        return [(fixture.proof(UNIT, "f", item[2], True),) for item in items]

    monkeypatch.setattr(compare.recipes, "resolve", lambda *a: RECIPE)
    monkeypatch.setattr(compare.pool, "map", fake_map)
    s = snap(cfg, {"f": member("f")})
    compare.measure(s, unit("f"), {"add": [], "omit": []}, tmp_path / "out")
    assert [job[5] for job in jobs] == [True, True]  # the keep flag asks native.prove to link even on a warm proof
    jobs.clear()
    compare.measure_many(s, unit("f"), [{"add": [], "omit": []}, {"add": ["-O1"], "omit": []}])
    assert [job[5] for job in jobs] == [False] * 4


def test_export_writes_the_disassembly_the_diff_and_the_target_data(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    place = Placement("a", ".text", 0x1000, 0x1008, 0x80000400)
    data = Placement("a", ".rodata", 0x2000, 0x2004, 0x80001000)
    s = snap(cfg, {"f": Member("f", "function", "c", "g", (place,)), "t": Member("t", "data", "data", "g", (data,))})
    work, target = tmp_path / "work", tmp_path / "target"
    work.mkdir()
    (work / "text.bin").write_bytes(b"built")
    monkeypatch.setattr(compare.versions, "rom_bytes", lambda v, lo, hi: b"orig" + bytes([lo & 255]))
    monkeypatch.setattr(compare.process, "tool", lambda config, name: "objdump")
    monkeypatch.setattr(compare.process, "scratch", lambda root: tmp_path)

    def run(phase: str, argv: list[str], cwd: Path, *, stdout_path: Path, tmp: Path) -> None:
        stdout_path.write_text("add\n" if "text.bin" in argv[-1] else "sub\n")

    monkeypatch.setattr(compare.process, "run", run)
    s = replace(s, versions={"a": object()})
    compare._export((s, unit("f", "t"), "a", RECIPE, work, [], target))
    assert (target / "linked.dis").read_text() == "add\n" and (target / "original.dis").read_text() == "sub\n"
    assert "-sub" in (target / "diff.txt").read_text() and "+add" in (target / "diff.txt").read_text()
    assert (target / "original.rodata.bin").read_bytes() == b"orig" + bytes([0x00])  # the data the build must reach
    assert json.loads((target / "result.json").read_text())["proofs"] == []


def _score_scene(
    cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    built: bytes,
    target: bytes,
    source: bytes = b"int f;\n",
):
    calls: dict[str, Any] = {}
    u = unit("f")
    s = snap(cfg, {"f": member("f")}, (u,))
    s = replace(
        s,
        versions={"a": object(), "b": object()},
        layout=replace(
            s.layout, groups={"g": Group("g", "main", ("f",), "authored", (), False, "src/g.c", "include/main/g.h")}
        ),
    )
    bound = replace(s, overlays={UNIT: source})
    calls["captures"] = []
    monkeypatch.setattr(compare.layout, "capture", lambda config: calls["captures"].append(1) or s)
    monkeypatch.setattr(compare.view, "get", lambda *a, **k: None)
    monkeypatch.setattr(compare, "bind", lambda snapshot, file, function, notes=None: (u, bound))
    monkeypatch.setattr(compare.recipes, "resolve", lambda *a: RECIPE)
    monkeypatch.setattr(compare.versions, "rom_bytes", lambda v, lo, hi: target)

    def prove(snapshot: Snapshot, spec: UnitSpec, version: str, recipe: Recipe, work: Path, keep: bool) -> tuple:
        calls["prove"] = (version, keep)
        calls["proved"] = calls.get("proved", 0) + 1
        (work / "text.bin").write_bytes(built)
        return (fixture.proof(UNIT, "f", version, built == target, () if built == target else ("bytes differ",)),)

    monkeypatch.setattr(compare.native, "prove", prove)
    monkeypatch.setattr(compare.land, "submit", lambda *a, **k: pytest.fail("score never submits"))
    monkeypatch.setattr(compare.land, "submit_many", lambda *a, **k: pytest.fail("score never submits"))
    monkeypatch.setattr(compare.land, "drain", lambda *a, **k: pytest.fail("score never lands"))
    return calls, params(tmp_path, version=None, lines=None) | {"flag": [], "omit_flag": [], "alloc": False}


def test_score_measures_one_version_writes_nothing_and_validates(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same = (0x03E00008).to_bytes(4, "big") + bytes(4)
    calls, given = _score_scene(cfg, tmp_path, monkeypatch, same, same)
    result = compare.score_command(cfg, given)
    configuration.validate("result.score", json.loads(json.dumps(result)), "result.score")
    assert calls["prove"] == (cfg.project.names_from, True)  # the names-from version, linked fresh for its bytes
    assert result["exact"] and result["words"] == {"built": 2, "target": 2, "matched": 2} and result["diff"] == []


def test_score_places_data_where_the_roms_code_reads_it_and_scores_the_code_alone(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same = (0x03E00008).to_bytes(4, "big") + bytes(4)
    _calls, given = _score_scene(cfg, tmp_path, monkeypatch, same, same)
    seen: list[bool] = []
    monkeypatch.setattr(
        ownership, "derive", lambda snapshot, unit, loose=False, recipe=None: seen.append(loose) or (snapshot, unit)
    )
    assert compare.score_command(cfg, given)["score"] == 1.0 and seen == [True]


def test_score_shows_the_differing_instruction_and_honors_version_and_lines(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = (0x24020001).to_bytes(4, "big") + (0x03E00008).to_bytes(4, "big")  # addiu v0, zero, 1; jr ra
    built = (0x24020002).to_bytes(4, "big") + (0x03E00008).to_bytes(4, "big")  # addiu v0, zero, 2; jr ra
    calls, given = _score_scene(cfg, tmp_path, monkeypatch, built, target)
    result = compare.score_command(cfg, given | {"version": "b", "lines": 40})
    assert calls["prove"] == ("b", True) and not result["exact"]
    assert result["words"] == {"built": 2, "target": 2, "matched": 1}
    assert any(line.startswith("-") and "0x1" in line for line in result["diff"])
    assert any(line.startswith("+") and "0x2" in line for line in result["diff"])
    with pytest.raises(Refusal) as error:
        compare.score_command(cfg, given | {"version": "z"})
    assert error.value.findings[0].key == "land.request" and "z" in error.value.findings[0].reason


def test_score_reports_a_file_local_define_without_refusing(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same = (0x03E00008).to_bytes(4, "big") + bytes(4)
    _, given = _score_scene(cfg, tmp_path, monkeypatch, same, same, b"#define PAD(n) pad(n)\nint f;\n")
    result = compare.score_command(cfg, given)
    assert result["exact"] and result["score"] == 1.0  # the rules never change the score
    assert [(f["key"], f["line"]) for f in result["source"]] == [("source.local-define", 1)]


def test_score_does_the_least_work_one_capture_one_unit_nothing_written(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same = (0x03E00008).to_bytes(4, "big") + bytes(4)
    calls, given = _score_scene(cfg, tmp_path, monkeypatch, same, same)
    for owner, name in (
        (compare.policy, "census"),
        (compare.land, "inbox"),
        (journal, "apply"),
        (store, "write"),
        (compare.native, "dumps"),
    ):
        monkeypatch.setattr(owner, name, lambda *a, _n=name, **k: pytest.fail(f"score called {_n}"))
    compare.score_command(cfg, given)
    assert calls["captures"] == [1] and calls["proved"] == 1  # no allocation compile unless --alloc asks
    dumps: list[str] = []
    monkeypatch.setattr(compare.native, "dumps", lambda *a: dumps.append(a[2]) or ("", "", "", {}))
    assert compare.score_command(cfg, given | {"alloc": True})["alloc"] == [] and dumps == [cfg.project.names_from]
    assert calls["captures"] == [1, 1] and calls["proved"] == 2


def test_score_variants_capture_once_dispatch_once_and_report_each_candidate(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same = (0x03E00008).to_bytes(4, "big") + bytes(4)
    calls, given = _score_scene(cfg, tmp_path, monkeypatch, same, same)
    rows = [{"file": "one.c", "overlays": []}, {"file": "two.c", "overlays": ["include/h.h=new.h"]}, {"file": "gone.c"}]
    listing = tmp_path / "variants.json"
    listing.write_text(json.dumps(rows), encoding="utf-8")
    seen: list[Any] = []

    def bind(snapshot: Snapshot, file: Path, function: str | None, notes: Any = None) -> tuple:
        if file.name == "gone.c":
            raise Refusal(Finding("land.request", reason="gone"))
        return unit("f"), replace(snapshot, overlays={UNIT: b"int f;\n"})

    derived: list[int] = []
    monkeypatch.setattr(compare, "bind", bind)
    monkeypatch.setattr(compare, "_extra", lambda snapshot, spec: ("include/h.h", b"x"))
    monkeypatch.setattr(compare.layout, "overlay", lambda snapshot, files: snapshot)
    monkeypatch.setattr(compare.ownership, "derive", lambda s, u, loose=False, recipe=None: derived.append(1) or (s, u))
    monkeypatch.setattr(
        compare.pool, "map", lambda c, name, fn, items, key=None: seen.append(len(items)) or [fn(i) for i in items]
    )
    result = compare.score_command(cfg, given | {"file": None, "variants": listing})
    configuration.validate("result.score", json.loads(json.dumps(result)), "result.score")
    assert calls["captures"] == [1] and seen == [3]  # one capture and one pool dispatch for every candidate
    assert derived == [1, 1] and calls["proved"] == 2  # one derive and one build per candidate that binds
    assert [(v["file"], v["ok"]) for v in result["variants"]] == [("one.c", True), ("two.c", True), ("gone.c", False)]
    assert result["variants"][0]["result"]["exact"] and "proofs" not in result["variants"][0]["result"]
    assert result["variants"][2]["findings"][0]["reason"] == "gone"
    with pytest.raises(Refusal):
        compare.score_command(cfg, given | {"file": None})
    assert calls["captures"] == [1]  # missing FILE also refuses before snapshot capture


def test_score_variants_refuses_a_row_with_an_unknown_key_or_overlays_given_as_a_map(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = Mock(side_effect=AssertionError("a malformed manifest captured the project"))
    monkeypatch.setattr(compare.layout, "capture", capture)
    for row, said in (
        ({"file": "one.c", "overlay": ["include/a.h=a.h"]}, "not overlay"),
        ({"file": "one.c", "overlays": {"include/a.h": "a.h"}}, "overlays must be a list"),
    ):
        listing = tmp_path / "variants.json"
        listing.write_text(json.dumps([row]), encoding="utf-8")
        with pytest.raises(Refusal) as error:
            compare.score_command(cfg, {"variants": listing})
        assert said in error.value.findings[0].reason and error.value.findings[0].reason.startswith("row 1: ")
    capture.assert_not_called()


@pytest.mark.parametrize("text,line,column", [("", 1, 1), ('[\n {"file": "one.c"},\n]', 2, 19)])
def test_score_variants_malformed_json_names_input_and_location(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, line: int, column: int
) -> None:
    listing = tmp_path / "variants.json"
    listing.write_text(text, encoding="utf-8")
    capture = Mock(side_effect=AssertionError("invalid manifest captured the project"))
    monkeypatch.setattr(compare.layout, "capture", capture)
    dispatch = Mock(side_effect=AssertionError("invalid manifest scheduled workers"))
    monkeypatch.setattr(compare.pool, "map", dispatch)
    with pytest.raises(Refusal) as error:
        compare.score_command(cfg, {"variants": listing})
    finding = error.value.findings[0]
    assert finding.key == "land.request" and finding.path == str(listing) and finding.line == line
    assert f"line {line}, column {column}" in finding.reason and "JSON" in finding.action
    capture.assert_not_called()
    dispatch.assert_not_called()


@pytest.mark.parametrize("content", [None, b"\xff"])
def test_score_variants_unreadable_input_is_actionable(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes | None
) -> None:
    listing = tmp_path / "variants.json"
    if content is not None:
        listing.write_bytes(content)
    capture = Mock(side_effect=AssertionError("unreadable manifest captured the project"))
    monkeypatch.setattr(compare.layout, "capture", capture)
    dispatch = Mock(side_effect=AssertionError("unreadable manifest scheduled workers"))
    monkeypatch.setattr(compare.pool, "map", dispatch)
    with pytest.raises(Refusal) as error:
        compare.score_command(cfg, {"variants": listing})
    finding = error.value.findings[0]
    assert finding.key == "land.request" and finding.path == str(listing)
    assert "cannot read" in finding.reason and "UTF-8" in finding.action
    capture.assert_not_called()
    dispatch.assert_not_called()


@pytest.mark.parametrize("document", [{"file": "one.c"}, None])
def test_score_variants_requires_outer_list(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: Any
) -> None:
    listing = tmp_path / "variants.json"
    listing.write_text(json.dumps(document), encoding="utf-8")
    capture = Mock(side_effect=AssertionError("invalid manifest captured the project"))
    monkeypatch.setattr(compare.layout, "capture", capture)
    dispatch = Mock(side_effect=AssertionError("invalid manifest scheduled workers"))
    monkeypatch.setattr(compare.pool, "map", dispatch)
    with pytest.raises(Refusal) as error:
        compare.score_command(cfg, {"variants": listing})
    finding = error.value.findings[0]
    assert finding.key == "land.request" and finding.path == str(listing)
    assert "JSON list" in finding.reason
    capture.assert_not_called()
    dispatch.assert_not_called()


@pytest.mark.parametrize(
    "row",
    [
        "two.c",
        {},
        {"file": 2},
        {"file": ""},
        {"file": "two.c", "overlays": "include/h.h=new.h"},
        {"file": "two.c", "overlays": ["include/h.h=new.h", 2]},
        {"file": "two.c", "overlays": [""]},
    ],
)
def test_score_variants_validates_every_row_before_dispatch(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: Any
) -> None:
    listing = tmp_path / "variants.json"
    listing.write_text(json.dumps([{"file": "one.c"}, row]), encoding="utf-8")
    capture = Mock(side_effect=AssertionError("invalid row captured the project"))
    monkeypatch.setattr(compare.layout, "capture", capture)
    dispatch = Mock(side_effect=AssertionError("invalid row scheduled workers"))
    monkeypatch.setattr(compare.pool, "map", dispatch)
    with pytest.raises(Refusal) as error:
        compare.score_command(cfg, {"variants": listing})
    finding = error.value.findings[0]
    assert finding.key == "land.request" and finding.path == str(listing)
    assert "row 2:" in finding.reason and finding.action
    capture.assert_not_called()
    dispatch.assert_not_called()


def test_score_variants_preserves_valid_overlays_and_command_options(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = tmp_path / "variants.json"
    listing.write_text(
        json.dumps(
            [
                {"file": "one.c"},
                {"file": "two.c", "overlays": None},
                {"file": "three.c", "overlays": []},
                {"file": "four candidate.c", "overlays": ["include/shared h.h=new h.h", " extra.h "]},
            ]
        ),
        encoding="utf-8",
    )
    snapshot = snap(cfg, {})
    capture = Mock(return_value=snapshot)
    monkeypatch.setattr(compare.layout, "capture", capture)
    dispatch = Mock(return_value=[])
    monkeypatch.setattr(compare.pool, "map", dispatch)
    result = compare.score_command(
        cfg, {"variants": listing, "function": "chosen", "version": "us", "alloc": True, "extras": [Path("old.h")]}
    )
    assert result == {"variants": []}
    capture.assert_called_once_with(cfg)
    dispatch.assert_called_once()
    config, name, worker, jobs = dispatch.call_args.args
    assert config is cfg and name == "compare.variants" and worker is compare._variant_job
    assert all(s is snapshot for s, params in jobs)
    expected_files = [Path(name) for name in ("one.c", "two.c", "three.c", "four candidate.c")]
    expected_extras = [[], [], [], [Path("include/shared h.h=new h.h"), Path(" extra.h ")]]
    assert [params["file"] for s, params in jobs] == expected_files
    assert [params["extras"] for s, params in jobs] == expected_extras
    assert all(params["function"] == "chosen" and params["version"] == "us" and params["alloc"] for s, params in jobs)


def test_score_compile_failure_shows_every_error_line_and_no_warning(tmp_path: Path) -> None:
    from unbake import adapters

    err = "\n".join(
        ["a.c: In function `f':"]
        + [f"a.c:{n}: warning: x" for n in range(30)]
        + [f"a.c:{n}: no member {n}" for n in range(9)]
    )
    shown = adapters._tail(SimpleNamespace(stderr=err.encode(), exit=1)).splitlines()
    assert shown == [f"a.c:{n}: no member {n}" for n in range(9)]
    proof = fixture.proof(UNIT, "f", "a", False, ("version a: compile.error: " + "\n".join(shown),))
    assert len(compare._compile_failure(unit("f"), "a", [proof]).reason.split("; ")) == 9


def test_a_change_set_file_may_name_the_project_file_it_replaces(cfg: Config, tmp_path: Path) -> None:
    header = cfg.project.root / "include/h.h"
    header.parent.mkdir(parents=True)
    header.write_bytes(b"extern int f;")
    text = tmp_path / "new_h.txt"
    text.write_bytes(b"extern long f;")
    s = snap(cfg, {"f": member("f")})
    assert compare._extra(s, Path(f"include/h.h={text}")) == ("include/h.h", b"extern long f;")
    for spec in (f"../escape.h={text}", f"include/h.h={tmp_path}/none"):
        with pytest.raises(Refusal):
            compare._extra(s, Path(spec))
    with pytest.raises(Refusal):
        compare._extra(s, tmp_path / "stranger.c")  # outside the project and no landed unit's source


def test_a_unit_of_one_function_and_its_data_rows_names_its_function(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    s = mock_run(monkeypatch, cfg, unit("f", "rodata/f/00001000"), True, fixture.submission(id="sub4"), calls)
    s.layout.members["rodata/f/00001000"] = replace(s.layout.members["rodata/f/00001000"], kind="data")
    compare.run(cfg, params(tmp_path))
    assert calls["submit"][0]["function"] == "f"


def test_library_candidate_keeps_its_declared_folder_and_replaces_the_old_owner(cfg, monkeypatch):
    cfg = replace(cfg, project=replace(cfg.project, library_dirs=("src/vendor",)))
    old = unit("f", path="src/f.c")
    snapshot = snap(cfg, {"f": member("f")}, (old,))
    file = cfg.project.root / "src/vendor/f.c"
    file.parent.mkdir(parents=True)
    file.write_text("int f(void) { return 1; }\n")
    maps, writes = [], []
    monkeypatch.setattr(compare.layout, "dump_map", lambda mapping: maps.append(mapping) or b"layout")
    monkeypatch.setattr(compare.layout, "overlay", lambda s, rows: writes.append(rows) or s)
    bound, _ = compare.bind(snapshot, file, "f")
    assert bound.path == "src/vendor/f.c"
    assert set(maps[0].units) == {bound.path}
    assert writes[0][bound.path] == file.read_bytes()


def test_a_candidate_binds_by_the_member_it_defines_never_by_its_file_name(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = snap(cfg, {"func_A": member("func_A")})
    file = tmp_path / "85c78-accessor.c"
    file.write_bytes(b"int func_A(void) { return 1; }\n")
    u, asked = unit("func_A"), []
    monkeypatch.setattr(compare.layout, "unit_options", lambda snapshot, name, source: asked.append(name) or [(u, {})])
    monkeypatch.setattr(compare.layout, "overlay", lambda snapshot, writes: snapshot)
    assert compare.bind(s, file, None)[0] == u and asked == ["func_A"]


def test_a_candidate_that_defines_no_member_refuses_naming_what_it_defines(cfg: Config, tmp_path: Path) -> None:
    s = snap(cfg, {"func_A": member("func_A")})
    file = tmp_path / "func_A.c"  # the name is no longer a member
    file.write_bytes(b"int other(void) { return 1; }\n")
    with pytest.raises(Refusal) as error:
        compare.bind(s, file, None)
    finding = error.value.findings[0]
    assert finding.key == "land.request" and "defines no member" in finding.reason and "other" in finding.reason


def _admit_new_header(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, includers: tuple[str, ...]) -> Any:
    s = mock_run(monkeypatch, cfg, unit("f"), True, fixture.submission(id="sub9"), {})
    new = tmp_path / "new.h"
    new.write_bytes(b"extern int f;")
    measured: list[list[str]] = []
    asked: list[list[str]] = []
    monkeypatch.setattr(
        compare.headers, "consumers", lambda snapshot, changed: asked.append(list(changed)) or includers
    )
    proofs = tuple(fixture.proof(UNIT, "f", v, True, ()) for v in ("a", "b"))
    monkeypatch.setattr(
        compare, "measure_units", lambda snapshot, units, over: measured.append([u.path for u in units]) or proofs
    )
    given = params(tmp_path) | {"prove_extras": True}
    spec = Path(f"include/shared/new.h={new}")
    return (
        lambda: compare._admit(cfg, s, [given["file"], spec], given, {"add": [], "omit": []}, None, "test"),
        asked,
        measured,
    )


def test_a_new_header_with_an_includer_is_admitted_and_its_consumers_are_proved(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admit, asked, measured = _admit_new_header(cfg, tmp_path, monkeypatch, (UNIT,))
    assert admit()["submission"].id == "sub9"
    assert asked == [["include/shared/new.h"]] * 2 and measured == [[UNIT]]  # once to find includers, once to prove


def test_a_new_header_no_source_includes_is_refused(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admit, _, measured = _admit_new_header(cfg, tmp_path, monkeypatch, ())
    with pytest.raises(Refusal) as error:
        admit()
    assert "included by no source" in error.value.findings[0].reason and measured == []


def test_a_change_set_entry_never_creates_a_source(cfg: Config, tmp_path: Path) -> None:
    text = tmp_path / "t.c"
    text.write_bytes(b"int f;")
    s = snap(cfg, {"f": member("f")})
    with pytest.raises(Refusal) as error:
        compare._extra(s, Path(f"src/new.c={text}"))
    assert "new source" in error.value.findings[0].reason
    assert compare._extra(s, Path(f"include/new/h.h={text}")) == ("include/new/h.h", b"int f;")


def test_score_overlays_change_set_files_without_writing_the_project(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same = (0x03E00008).to_bytes(4, "big") + bytes(4)
    calls, given = _score_scene(cfg, tmp_path, monkeypatch, same, same)
    header = cfg.project.root / "include/shared.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_bytes(b"old\n")
    fresh = tmp_path / "new-shared.h"
    fresh.write_bytes(b"new\n")
    bound: list[Snapshot] = []
    monkeypatch.setattr(
        compare.layout, "overlay", lambda snapshot, writes: replace(snapshot, overlays={**snapshot.overlays, **writes})
    )
    monkeypatch.setattr(
        compare,
        "bind",
        lambda snapshot, file, function, notes=None: (
            bound.append(snapshot) or (unit("f"), replace(snapshot, overlays={**snapshot.overlays, UNIT: b"int f;\n"}))
        ),
    )
    compare.score_command(cfg, given | {"extras": [Path(f"include/shared.h={fresh}")]})
    assert bound[0].overlays["include/shared.h"] == b"new\n"  # the score input carries the overlay
    assert header.read_bytes() == b"old\n"  # the project is untouched
    assert calls["captures"] == [1]


def test_score_refuses_an_unknown_overlay_path_by_name(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same = (0x03E00008).to_bytes(4, "big") + bytes(4)
    _, given = _score_scene(cfg, tmp_path, monkeypatch, same, same)
    fresh = tmp_path / "x.h"
    fresh.write_bytes(b"x\n")
    with pytest.raises(Refusal) as error:
        compare.score_command(cfg, given | {"extras": [Path(f"include/missing/x.c={fresh}")]})
    assert error.value.findings[0].path == "include/missing/x.c"


def test_score_names_a_compile_failure_as_a_finding_with_the_compilers_lines(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    same = (0x03E00008).to_bytes(4, "big") + bytes(4)
    _, given = _score_scene(cfg, tmp_path, monkeypatch, same, same)
    monkeypatch.setattr(
        compare.native,
        "prove",
        lambda snapshot, spec, version, recipe, work, keep: (
            fixture.proof(UNIT, "f", version, False, ("cc1: error: first\\nsecond line\\n",)),
        ),
    )
    with pytest.raises(Refusal) as error:
        compare.score_command(cfg, given)
    found = error.value.findings[0]
    assert found.key == "score.compile" and "cc1: error: first" in found.reason and "second line" in found.reason


def test_short_score_shows_every_hunk_compactly_within_the_line_limit(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nop, other = bytes(4), (0x24020001).to_bytes(4, "big")
    target, built = other + nop * 10 + other, nop * 12
    _, given = _score_scene(cfg, tmp_path, monkeypatch, built, target)
    full = compare.score_command(cfg, given)["diff"]
    short = compare.score_command(cfg, given | {"short": True})["diff"]
    assert sum(line.startswith("@@") for line in full) == 2 == sum(line.startswith("@@") for line in short)
    assert all(line.startswith(("@@", "-", "+")) for line in short) and len(short) < len(full)
    assert len(compare.score_command(cfg, given | {"short": True, "lines": 3})["diff"]) == 3


CONSUMER = "src/c.c"


def _admit_header_change(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after: dict[str, bool]
) -> tuple[Any, list[tuple[bool, list[str]]]]:
    """A header extra whose consumer c.c is measured before (without the overlay) and after (with it)."""
    primary, other = unit("f"), unit("g", path=CONSUMER)
    s = mock_run(monkeypatch, cfg, primary, True, fixture.submission(id="sub9"), {})
    s = snap(cfg, {"f": member("f"), "g": member("g")}, (primary, other))
    monkeypatch.setattr(compare.layout, "capture", lambda config: s)
    header = tmp_path / "h.h"
    header.write_bytes(b"extern int f;")
    monkeypatch.setattr(compare.headers, "consumers", lambda snapshot, changed: (UNIT, CONSUMER))
    monkeypatch.setattr(compare.land, "declaration_users", lambda before, after, affected, changed: affected)
    calls: list[tuple[bool, list[str]]] = []

    def measured(snapshot: Snapshot, units: list[UnitSpec], over: Any) -> tuple[Any, ...]:
        calls.append((snapshot is s, [u.path for u in units]))
        exact = {UNIT: True, CONSUMER: snapshot is s or after[CONSUMER]}
        return tuple(
            fixture.proof(u.path, u.members[0], v, exact[u.path], () if exact[u.path] else ("bytes differ",))
            for u in units
            for v in ("a", "b")
        )

    monkeypatch.setattr(compare, "measure_units", measured)
    given = params(tmp_path)
    spec = Path(f"include/shared/h.h={header}")
    return (lambda: compare._admit(cfg, s, [given["file"], spec], given, {"add": [], "omit": []}, None, "t"), calls)


def test_a_header_extra_proves_every_consumer_and_reports_their_count(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admit, calls = _admit_header_change(cfg, tmp_path, monkeypatch, {CONSUMER: True})
    done = admit()
    assert done["submission"].id == "sub9" and done["found"] == ()
    assert done["consumers"]["count"] == 1 and done["consumers"]["wall"] >= 0
    assert calls == [(False, [UNIT, CONSUMER]), (True, [CONSUMER])]  # with the overlays, then the baseline once
    assert {p.unit for p in done["proofs"]} == {UNIT}  # only the change set's own proofs are submitted


def test_groups_of_one_chunk_measure_a_shared_consumer_in_the_tree_as_it_is_once(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admit, calls = _admit_header_change(cfg, tmp_path, monkeypatch, {CONSUMER: True})
    for _ in range(4):  # four groups of one bulk chunk, every one changing a header c.c includes
        assert admit()["found"] == ()
    assert calls.count((True, [CONSUMER])) == 1  # the baseline: once per command, not once per group
    assert calls.count((False, [UNIT, CONSUMER])) == 4  # each group's own overlays are still proved


def test_a_consumer_that_builds_worse_with_the_header_refuses_the_group_naming_unit_and_version(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admit, _ = _admit_header_change(cfg, tmp_path, monkeypatch, {CONSUMER: False})
    done = admit()
    assert done["submission"] is None
    assert [(f.key, f.unit, f.versions) for f in done["found"]] == [
        ("land.consumer_regressed", CONSUMER, ("a",)),
        ("land.consumer_regressed", CONSUMER, ("b",)),
    ]


def test_submit_proves_only_the_includers_that_use_an_altered_declaration(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    users = [f"src/u{i}.c" for i in range(90)]
    s = snap(cfg, {})
    s = replace(s, layout=replace(s.layout, units={p: None for p in users}))
    unit = UnitSpec("src/own.c", "c", "g", (), "test", {})
    asked: list[Any] = []
    monkeypatch.setattr(compare.headers, "consumers", lambda _s, ch: tuple(users) if ch else ())
    monkeypatch.setattr(
        compare.land, "declaration_users", lambda b, a, affected, ch: asked.append(affected) or set(users[:12])
    )
    proved = compare._proof_set(s, s, unit, {"include/a.h": b""}, False)
    assert proved["consumers"] == sorted(users[:12]) and asked == [set(users)]
    asked.clear()
    assert compare._proof_set(s, s, unit, {"src/x.c": b""}, False)["consumers"] == [] and asked == []


def test_a_bulk_group_refused_at_entry_is_its_own_row_with_every_finding_and_the_next_group_still_enters(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    mock_run(monkeypatch, cfg, unit("f"), True, fixture.submission(id="sub"), calls)
    first, second = tmp_path / "f.c", tmp_path / "g.c"
    first.write_bytes(b"int f;")
    second.write_bytes(b"int g;")
    entered: list[str] = []
    rules = (Finding("source.shared-declarations", "repeated extern"), Finding("land.view-conversion", "warned"))

    def fake_submit(config: Config, request: dict[str, Any], *args: Any) -> Any:
        entered.append(Path(request["file"]).name)
        if request["file"] == str(first):
            raise Refusal(*rules)
        return fixture.submission(id="sub")

    monkeypatch.setattr(compare.land, "submit", fake_submit)
    monkeypatch.setattr(compare.land, "submit_many", _each(fake_submit))
    base = {"function": None, "note": None, "withhold": False, "bulk": True, "toolchain": None}
    result = compare.submit_command(cfg, {**base, "files": [first, second], "flag": [], "omit_flag": []})
    rows = result["submissions"]
    assert entered == ["f.c", "g.c"]  # the refusal of the first entry never stops the second
    assert [(r["id"], r["refused"]) for r in rows] == [(None, True), ("sub", False)]
    assert [f["key"] for f in rows[0]["findings"]] == ["source.shared-declarations", "land.view-conversion"]
    configuration.validate("result.submit", json.loads(json.dumps(result)), "result.submit")


def test_bulk_submit_of_a_multi_member_file_is_an_error_not_a_silent_row(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refusal = Finding("land.request", "the file holds several members; pass --function")  # a row, then the error

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise Refusal(refusal)

    monkeypatch.setattr(compare, "_stage", refuse)  # the bulk path stages each group, then measures them at once
    monkeypatch.setattr(compare, "_prepare", lambda snapshot, files, params: ({}, unit("a", "b"), snapshot, []))
    monkeypatch.setattr(compare.layout, "capture", lambda config: None)
    source = tmp_path / "two.c"
    source.write_text("int a; int b;")
    with pytest.raises(Refusal) as raised:
        compare.submit_command(
            cfg, {"files": [source], "function": None, "note": None, "withhold": False, "bulk": True}
        )
    assert [f.key for f in raised.value.findings] == ["land.request"]


def _keyed(snapshot, monkeypatch, stamps, text=b"int f(void);"):
    unit = UnitSpec("src/fuzzy/f.c", "c", "g", ("f",), "test", {})
    monkeypatch.setattr(compare.recipes, "resolve", lambda cfg, u, o: "recipe")
    monkeypatch.setattr(compare.ownership, "derive_pairs", lambda pairs, loose=False, recipe=None: list(pairs))
    monkeypatch.setattr(compare, "holders", lambda snap, u: ("a", "b"))
    seen = []

    def stamp(snap, u, version, recipe, *, pinned=False):
        seen.append((version, pinned))
        return stamps[version]

    monkeypatch.setattr(compare.native, "stamp", stamp)
    monkeypatch.setattr(type(snapshot), "read", lambda self, path: text, raising=False)
    return compare.fuzzy_keys(snapshot.config, [(snapshot, unit)])[0], seen


def test_the_fuzzy_key_is_the_pinned_stamp_per_held_version_plus_the_candidate_text(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = Mock(config=cfg)
    key, seen = _keyed(s, monkeypatch, {"a": "A", "b": "B"})
    assert sorted(key) == ["a", "b"] and seen == [("a", True), ("b", True)]  # closure pins, not every header
    assert _keyed(s, monkeypatch, {"a": "A", "b": "B"})[0] == key  # an unrelated header edit leaves the stamps alone
    moved = _keyed(s, monkeypatch, {"a": "A", "b": "B2"})[0]  # a placement or recipe change moves that version only
    assert moved["a"] == key["a"] and moved["b"] != key["b"]
    assert _keyed(s, monkeypatch, {"a": "A", "b": "B"}, text=b"int f(int);")[0] != key  # the text moves it


def test_measuring_n_units_derives_them_in_one_call_with_each_units_own_recipe(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    units = [UnitSpec(f"src/u{i}.c", "c", "g", (f"f{i}",), "test", {}) for i in range(30)]
    snapshot = Mock(config=cfg)
    calls = []
    monkeypatch.setattr(compare.recipes, "resolve", lambda c, u, o: f"recipe-{u.path}")
    monkeypatch.setattr(
        compare.ownership,
        "derive_pairs",
        lambda pairs, loose=False, recipe=None: calls.append((list(pairs), recipe)) or list(pairs),
    )
    monkeypatch.setattr(compare, "holders", lambda snap, u: ())
    monkeypatch.setattr(compare.pool, "map", lambda *a, **k: [])
    assert compare.measure_units(snapshot, units, {}) == ()
    ((pairs, recipe),) = calls  # N units are one derive_pairs, not N derives each with its own dispatch
    assert pairs == [(snapshot, u) for u in units] and recipe == [f"recipe-{u.path}" for u in units]


def test_bulk_submit_of_n_units_derives_ownership_in_one_dispatch_and_admission_never_derives_again(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    count = 6
    names = [f"f{i}" for i in range(count)]
    calls: dict[str, Any] = {}
    s = mock_run(monkeypatch, cfg, unit(*names), True, fixture.submission(id="sub"), calls)
    files = []
    for name in names:
        files.append(tmp_path / f"{name}.c")
        files[-1].write_bytes(b"int x;")
    trees = {f.stem: replace(s, digest=f"tree-{f.stem}") for f in files}  # each candidate is its own tree state
    monkeypatch.setattr(
        compare,
        "bind",
        lambda snapshot, file, function, notes=None: (unit(file.stem, path=f"src/{file.stem}.c"), trees[file.stem]),
    )
    monkeypatch.setattr(ownership, "derive", REAL_DERIVE)
    monkeypatch.setattr(ownership, "derive_pairs", REAL_PAIRS)
    monkeypatch.setattr(ownership, "_emits_data", lambda snapshot, u: True)
    monkeypatch.setattr(ownership, "_derived", lambda snapshot, u, result: (snapshot, u))
    measured = compare.pool.map
    dispatched: list[list[str]] = []

    def fake_map(config: Config, name: str, function: Any, items: list[Any], key: Any = None) -> list[Any]:
        if name != "ownership.derive":
            return measured(config, name, function, items, key)
        dispatched.append(sorted({item[1].path for item in items}))
        return [{item[2]: {}} for item in items]

    monkeypatch.setattr(compare.pool, "map", fake_map)
    base = {"function": None, "note": None, "withhold": False, "bulk": True, "toolchain": None}
    result = compare.submit_command(cfg, {**base, "files": files, "flag": [], "omit_flag": []})
    assert [r["id"] for r in result["submissions"]] == ["sub"] * count
    assert dispatched == [[f"src/{n}.c" for n in names]]  # one derive for N units; admission and measure reuse it


def test_a_bulk_admission_of_a_two_function_source_labels_it_with_its_first_member_and_a_single_file_refuses(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    two = unit("f", "g")
    calls: dict[str, Any] = {}
    s = mock_run(monkeypatch, cfg, two, True, fixture.submission(id="sub2"), calls)
    given = params(tmp_path)
    compare._admit(cfg, s, [given["file"]], {**given, "bulk": True}, {"add": [], "omit": []}, None, "submit")
    assert calls["submit"][0]["function"] == "f"
    with pytest.raises(Refusal) as raised:
        compare._admit(cfg, s, [given["file"]], given, {"add": [], "omit": []}, None, "submit")
    assert "holds several members" in raised.value.findings[0].reason


def _bulk_pair(
    cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, bytes],
    fits: Any,
    dispatched: list[Any] | None = None,
    consumers: tuple[str, ...] = ("src/a.c", "src/b.c"),
) -> list[dict[str, Any]]:
    """A bulk of a.c and b.c, landed fuzzy units including one header each group rewrites (`headers`: the text
    each group brings). `fits(path, header)` scores a draft against a header; an old source scores 0.5 with the old
    header and 0.2 with any new one."""
    a, b = unit("f", path="src/a.c"), unit("g", path="src/b.c")
    s = replace(snap(cfg, {"f": member("f"), "g": member("g")}, (a, b)), digest="bulk-pair")
    monkeypatch.setattr(compare.layout, "capture", lambda config: s)
    monkeypatch.setattr(
        compare.layout,
        "overlay",
        lambda tree, writes: replace(tree, overlays={**tree.overlays, **writes}, digest=digest((tree.digest, writes))),
    )
    owned = {"a": a, "b": b}
    monkeypatch.setattr(
        compare,
        "bind",
        lambda tree, file, function, notes=None: (
            owned[file.stem],
            compare.layout.overlay(tree, {owned[file.stem].path: file.read_bytes()}),
        ),
    )
    monkeypatch.setattr(compare.headers, "consumers", lambda tree, changed: consumers)
    monkeypatch.setattr(compare.land, "declaration_users", lambda before, after, affected, changed: affected)
    old = cfg.project.root / "include/h.h"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")

    def measured(config: Config, pairs: Any, over: Any) -> list[tuple[Any, ...]]:
        if dispatched is not None:
            dispatched.append([(u.path, dict(tree.overlays)) for tree, u in pairs])
        out = []
        for tree, u in pairs:
            header, source = tree.overlays.get("include/h.h", b"old"), tree.overlays.get(u.path)
            score = fits(u.path, header) if source else (0.5 if header == b"old" else 0.2)
            exact = score == 1.0
            out.append(
                tuple(
                    fixture.proof(u.path, u.members[0], v, exact, () if exact else ("bytes differ",), score)
                    for v in ("a", "b")
                )
            )
        return out

    monkeypatch.setattr(compare, "measure_bound", measured)
    entered = lambda config, request, spec, *rest, **kw: fixture.submission(  # noqa: E731
        id=f"sub-{spec.path}", member=spec.members[0]
    )
    monkeypatch.setattr(compare.land, "submit", entered)
    monkeypatch.setattr(compare.land, "submit_many", _each(entered))
    files: list[Path] = []
    for name, text in headers.items():
        source, header = tmp_path / f"{name}.c", tmp_path / f"{name}.h"
        source.write_bytes(f"int {name};".encode())
        header.write_bytes(text)
        files += [source, Path(f"include/h.h={header}")]
    base = {"function": None, "note": None, "withhold": False, "bulk": True, "toolchain": None}
    return compare.submit_command(cfg, {**base, "files": files, "flag": [], "omit_flag": []})["submissions"]


def test_a_bulk_member_is_proved_as_its_draft_so_a_header_its_sibling_needs_never_refuses_the_pair(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _bulk_pair(cfg, tmp_path, monkeypatch, {"a": b"new", "b": b"new"}, lambda path, header: 1.0)
    assert [(r["id"], r["findings"]) for r in rows] == [("sub-src/a.c", []), ("sub-src/b.c", [])]
    assert [r["consumers"]["count"] for r in rows] == [1, 1]  # the sibling is still a proved consumer


def test_a_bulk_member_that_really_regresses_its_siblings_draft_is_refused_naming_both(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # b's draft needs b's header; a's header (which a lands) makes it worse, so a is refused naming a and b, and
    # b, whose proof assumed a's draft, keeps a's old-source regression once a is refused
    rows = _bulk_pair(
        cfg,
        tmp_path,
        monkeypatch,
        {"a": b"newA", "b": b"newB"},
        lambda path, header: 0.3 if (path, header) == ("src/b.c", b"newA") else 1.0,
    )
    assert [r["id"] for r in rows] == [None, None]
    first = rows[0]["findings"]
    assert {(f["key"], f["path"], f["unit"]) for f in first} == {("land.consumer_regressed", "src/a.c", "src/b.c")}
    assert all("src/a.c" in f["reason"] and "src/b.c" in f["reason"] for f in first)
    second = rows[1]["findings"]
    assert {(f["unit"], f["versions"][0]) for f in second} == {("src/a.c", "a"), ("src/a.c", "b")}
    assert all("src/a.c was refused" in f["reason"] for f in second)


def test_a_change_set_lets_another_unit_keep_a_gap_head_already_has_and_blocks_new_ones(monkeypatch):
    NS = SimpleNamespace
    primary, other = NS(members=("f",), path="src/f.c"), NS(members=("g", "h"), path="src/g.c")
    monkeypatch.setattr(compare.layout, "unit_of", lambda s, m: other if m in other.members else primary)
    head = (
        fixture.proof("src/g.c", "g", "eu", False, missing=("unresolved",), score=0.0),
        fixture.proof("src/g.c", "h", "eu", True),
    )
    monkeypatch.setattr(compare, "_baseline", lambda s, paths, o: head)
    after = (
        fixture.proof("src/g.c", "g", "eu", False, missing=("unresolved",), score=0.0),
        fixture.proof("src/g.c", "h", "eu", False, missing=("bytes",)),
        fixture.proof("src/f.c", "f", "eu", False, missing=("bytes",)),
    )
    gaps = [Finding("land.not_exact", "x", unit=n, versions=("eu",), missing=("m",)) for n in ("g", "h", "f")]
    blocking, standing = compare._standing(None, primary, gaps, after, {})
    assert [g.unit for g in blocking] == ["h", "f"]  # h was exact at HEAD, f is the primary
    assert standing == {("g", "eu")}


def test_a_retype_gates_its_primary_on_no_regression_too(monkeypatch):
    # real case: func_8021B468_de (eu: unresolved D_auto_800E0D24 at HEAD) sank every retype group it sat in
    NS = SimpleNamespace
    primary = NS(members=("f", "g"), path="src/f.c")
    monkeypatch.setattr(compare.layout, "unit_of", lambda s, m: primary)
    head = (fixture.proof("src/f.c", "g", "eu", False, missing=("unresolved",), score=0.0),)
    monkeypatch.setattr(compare, "_baseline", lambda s, paths, o: head)
    after = (
        fixture.proof("src/f.c", "g", "eu", False, missing=("unresolved",), score=0.0),
        fixture.proof("src/f.c", "f", "eu", False, missing=("bytes",)),
    )
    gaps = [Finding("land.not_exact", "x", unit=n, versions=("eu",), missing=("m",)) for n in ("g", "f")]
    assert [g.unit for g in compare._standing(None, primary, gaps, after, {})[0]] == ["g", "f"]
    blocking, standing = compare._standing(None, primary, gaps, after, {}, everyone=True)
    assert [g.unit for g in blocking] == ["f"] and standing == {("g", "eu")}  # f was exact at HEAD: it regressed


def test_prove_groups_measures_every_alternative_in_one_dispatch(monkeypatch):
    NS = SimpleNamespace
    units = {"src/a.c": NS(path="src/a.c", members=("a",)), "src/b.c": NS(path="src/b.c", members=("b",))}
    held = NS(holders=lambda: ("eu",))
    bound = NS(layout=NS(units=units, members={"a": held, "b": held}))
    monkeypatch.setattr(compare, "_prepare", lambda s, g, p: ({}, units[str(g[0])], bound, []))
    monkeypatch.setattr(compare.ownership, "derive_pairs", lambda pairs, recipe: list(pairs))
    monkeypatch.setattr(compare.recipes, "resolve", lambda c, u, o: None)
    monkeypatch.setattr(
        compare,
        "_proof_set",
        lambda s, b, u, e, strict, every=False: {"changed": [u.path], "consumers": [], "proving": True},
    )
    calls = []

    def measure(config, pairs, overrides):
        calls.append(len(pairs))
        exact = {"src/a.c": True}
        return [
            (fixture.proof(u.path, u.members[0], "eu", u.path in exact, () if u.path in exact else ("bytes",)),)
            for _, u in pairs
        ]

    monkeypatch.setattr(compare, "measure_bound", measure)
    monkeypatch.setattr(compare, "_baseline", lambda s, paths, o: ())
    monkeypatch.setattr(compare.layout, "unit_of", lambda s, m: units[f"src/{m}.c"])
    monkeypatch.setattr(compare, "_source_rules", lambda s, done: ())
    found = compare.prove_groups(None, NS(), [[Path("src/a.c")], [Path("src/b.c")]], {})
    assert calls == [2] and found[0] == () and [f.unit for f in found[1]] == ["b"]


def test_prove_groups_proves_the_head_of_every_set_in_one_baseline_before_any_gate(monkeypatch):
    # one baseline per set re-derived and re-measured the same consumers per alternative: a bisect level crawled
    NS = SimpleNamespace
    units = {p: NS(path=p, members=(p[4],)) for p in ("src/a.c", "src/b.c", "src/c.c", "src/d.c")}
    held = NS(holders=lambda: ("eu",))
    bound = NS(layout=NS(units=units, members={m: held for m in "abcd"}))
    monkeypatch.setattr(compare, "_prepare", lambda s, g, p: ({}, units[str(g[0])], bound, []))
    monkeypatch.setattr(compare.ownership, "derive_pairs", lambda pairs, recipe: list(pairs))
    monkeypatch.setattr(compare.recipes, "resolve", lambda c, u, o: None)
    consumers = {"src/a.c": ["src/c.c"], "src/b.c": ["src/d.c", "src/c.c"]}
    monkeypatch.setattr(
        compare,
        "_proof_set",
        lambda s, b, u, e, strict, every=False: {"changed": [u.path], "consumers": consumers[u.path], "proving": True},
    )
    monkeypatch.setattr(
        compare,
        "measure_bound",
        lambda config, pairs, overrides: [(fixture.proof(u.path, u.members[0], "eu", True),) for _, u in pairs],
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(compare, "_baseline", lambda s, paths, o: calls.append(list(paths)) or ())
    monkeypatch.setattr(compare.layout, "unit_of", lambda s, m: units[f"src/{m}.c"])
    monkeypatch.setattr(compare, "_source_rules", lambda s, done: ())
    compare.prove_groups(None, NS(), [[Path("src/a.c")], [Path("src/b.c")]], {})
    assert calls[0] == ["src/c.c", "src/d.c"]  # the union of both sets' consumers, first and once


def test_bind_spells_a_draft_copy_defining_an_older_address_name_as_its_member(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, fuzzy = "func_80201AD0_de.c", {"func_80201AD0_de": {"path": "src/fuzzy/func_80201ACC_de.c"}}
    u = unit("func_80201AD0_de", path="src/a.c")
    s = snap(cfg, {"func_80201AD0_de": member("func_80201AD0_de")}, (u,))
    s = replace(s, layout=replace(s.layout, fuzzy=fuzzy))
    file = tmp_path / name
    file.write_bytes(b"void *func_80201ACC_de(void) { return func_80201ACC_de; }\n")
    writes: list[Any] = []
    monkeypatch.setattr(compare.layout, "overlay", lambda snapshot, w: writes.append(w) or snapshot)
    monkeypatch.setattr(compare.layout, "source_unit", lambda *a: None)
    notes: list[str] = []
    assert compare.bind(s, file, None, notes)[0] == u
    assert notes == ["func_80201ACC_de->func_80201AD0_de"]
    assert writes[0]["src/a.c"] == b"void *func_80201AD0_de(void) { return func_80201AD0_de; }\n"


def test_bind_still_refuses_a_source_of_unrelated_names(cfg: Config, tmp_path: Path) -> None:
    s = snap(cfg, {"f": member("f")})
    file = tmp_path / "other.c"
    file.write_bytes(b"int g(void) { return 0; }\nint h(void) { return 1; }\n")
    with pytest.raises(Refusal) as error:
        compare.bind(s, file, None)
    assert "defines g, h" in error.value.findings[0].reason


def test_bulk_submit_of_n_groups_measures_them_all_in_one_dispatch(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # RW submit --bulk b0d5a298: 84 groups in 1,972 s at 3.1 cores, each group its own small measure dispatch in turn
    # (~33 s a group). Groups proving without sibling drafts are measured together, on every core at once.
    count = 6
    names = [f"f{i}" for i in range(count)]
    calls: dict[str, Any] = {}
    s = mock_run(monkeypatch, cfg, unit(*names), True, fixture.submission(id="sub"), calls)
    files = []
    for name in names:
        files.append(tmp_path / f"{name}.c")
        files[-1].write_bytes(b"int x;")
    trees = {f.stem: replace(s, digest=f"tree-{f.stem}") for f in files}
    monkeypatch.setattr(
        compare,
        "bind",
        lambda snapshot, file, function, notes=None: (unit(file.stem, path=f"src/{file.stem}.c"), trees[file.stem]),
    )
    monkeypatch.setattr(ownership, "derive", REAL_DERIVE)
    monkeypatch.setattr(ownership, "derive_pairs", REAL_PAIRS)
    monkeypatch.setattr(ownership, "_emits_data", lambda snapshot, u: True)
    monkeypatch.setattr(ownership, "_derived", lambda snapshot, u, result: (snapshot, u))
    measured = compare.pool.map
    dispatched: list[int] = []

    def fake_map(config: Config, name: str, function: Any, items: list[Any], key: Any = None) -> list[Any]:
        if name == "ownership.derive":
            return [{item[2]: {}} for item in items]
        if function is compare.native.prove_args:
            dispatched.append(len(items))
        return measured(config, name, function, items, key)

    monkeypatch.setattr(compare.pool, "map", fake_map)
    base = {"function": None, "note": None, "withhold": False, "bulk": True, "toolchain": None}
    result = compare.submit_command(cfg, {**base, "files": files, "flag": [], "omit_flag": []})
    assert [r["id"] for r in result["submissions"]] == ["sub"] * count
    assert len(dispatched) == 1 and dispatched[0] >= count  # one measure dispatch holds every group


@pytest.mark.parametrize("regresses", [False, True])
def test_bulk_batches_sibling_drafts_without_merging_their_trees_or_verdicts(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, regresses: bool
) -> None:
    dispatched: list[Any] = []
    rows = _bulk_pair(
        cfg,
        tmp_path,
        monkeypatch,
        {"a": b"newA", "b": b"newB"},
        lambda path, header: 0.3 if regresses and (path, header) == ("src/b.c", b"newA") else 1.0,
        dispatched,
    )
    # One HEAD union and one candidate dispatch: every draft still has its own and applied tree.
    assert dispatched[0] == [("src/a.c", {}), ("src/b.c", {})]
    assert len(dispatched) == 2
    assert dispatched[1] == [
        ("src/a.c", {"include/h.h": b"newA", "src/a.c": b"int a;"}),
        ("src/b.c", {"include/h.h": b"newA", "src/a.c": b"int a;"}),
        ("src/b.c", {"include/h.h": b"newB", "src/b.c": b"int b;"}),
        ("src/b.c", {"include/h.h": b"newA", "src/a.c": b"int a;", "src/b.c": b"int b;"}),
        ("src/b.c", {"include/h.h": b"newB", "src/b.c": b"int b;"}),
        ("src/a.c", {"include/h.h": b"newB", "src/b.c": b"int b;"}),
        ("src/a.c", {"include/h.h": b"newA", "src/a.c": b"int a;"}),
        ("src/a.c", {"include/h.h": b"newB", "src/b.c": b"int b;", "src/a.c": b"int a;"}),
    ]
    assert [r["member"] for r in rows] == ["f", "g"]
    assert [r["refused"] for r in rows] == [regresses, regresses]
    if regresses:
        assert [r["id"] for r in rows] == [None, None]
        assert all(f["path"] == "src/a.c" and f["unit"] == "src/b.c" for f in rows[0]["findings"])
        assert all("src/a.c was refused" in f["reason"] for f in rows[1]["findings"])
    else:
        assert [r["id"] for r in rows] == ["sub-src/a.c", "sub-src/b.c"]
        assert all(not r["findings"] for r in rows)


@pytest.mark.parametrize("consumer_recipe_refused", [False, True])
def test_bulk_prewarms_actual_overlapping_consumers_and_keeps_atomic_group_results(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer_recipe_refused: bool
) -> None:
    names = "abcxyz"
    specs = {name: unit(f"f_{name}", path=f"src/{name}.c") for name in names}
    snapshot = replace(
        snap(cfg, {f"f_{name}": member(f"f_{name}") for name in names}, tuple(specs.values())),
        digest="overlapping-consumers",
    )
    monkeypatch.setattr(compare.layout, "capture", lambda config: snapshot)
    monkeypatch.setattr(
        compare.layout,
        "overlay",
        lambda tree, writes: replace(tree, overlays={**tree.overlays, **writes}, digest=digest((tree.digest, writes))),
    )

    def bind(tree: Snapshot, file: Path, function: Any, notes: Any = None) -> tuple[UnitSpec, Snapshot]:
        if file.stem == "bad":
            raise Refusal(Finding("land.request", "candidate has no layout member", path=str(file)))
        spec = specs[file.stem]
        return spec, compare.layout.overlay(tree, {spec.path: file.read_bytes()})

    monkeypatch.setattr(compare, "bind", bind)
    affected = {b"A": ("src/x.c", "src/y.c"), b"B": ("src/y.c", "src/z.c"), b"C": ("src/x.c", "src/y.c")}
    monkeypatch.setattr(compare.headers, "consumers", lambda tree, changed: affected[tree.overlays["include/h.h"]])
    monkeypatch.setattr(compare.land, "declaration_users", lambda before, after, paths, changed: paths)
    header = cfg.project.root / "include/h.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_bytes(b"old")
    files: list[Path] = []
    for name in "abc":
        source, extra = tmp_path / f"{name}.c", tmp_path / f"{name}.h"
        source.write_bytes(f"int f_{name};".encode())
        extra.write_bytes(name.upper().encode())
        files.extend([source, Path(f"include/h.h={extra}")])
        if name == "a":
            files.append(tmp_path / "bad.c")
    dispatches: list[Any] = []

    def measure(config: Config, pairs: Any, overrides: Any) -> list[tuple[Any, ...]]:
        assert overrides == {"add": ["-O1"], "omit": []}
        dispatches.append([(u.path, dict(tree.overlays)) for tree, u in pairs])
        return [
            tuple(
                fixture.proof(
                    u.path,
                    u.members[0],
                    version,
                    not (tree.overlays.get("include/h.h") == b"C" and u.path == "src/x.c"),
                    ("consumer bytes differ",)
                    if tree.overlays.get("include/h.h") == b"C" and u.path == "src/x.c"
                    else (),
                    0.25 if tree.overlays.get("include/h.h") == b"C" and u.path == "src/x.c" else 1.0,
                )
                for version in ("a", "b")
            )
            for tree, u in pairs
        ]

    monkeypatch.setattr(compare, "measure_bound", measure)
    resolve = compare.recipes.resolve

    def recipe(config: Config, spec: UnitSpec, overrides: Any) -> Recipe:
        if consumer_recipe_refused and spec.path == "src/x.c":
            raise Refusal(Finding("recipe.option", "unsupported consumer option", unit=spec.path))
        return resolve(config, spec, overrides)

    monkeypatch.setattr(compare.recipes, "resolve", recipe)
    accepted: list[Any] = []

    def submit_many(config: Config, calls: Any) -> list[Any]:
        accepted.extend(calls)
        return [fixture.submission(id=f"sub-{call['unit'].path}") for call in calls]

    monkeypatch.setattr(compare.land, "submit_many", submit_many)
    audited: list[str] = []
    monkeypatch.setattr(compare, "_source_rules", lambda tree, done: audited.append(done["unit"].path) or ())
    result = compare.submit_command(
        cfg, {"files": files, "bulk": True, "function": None, "note": None, "withhold": False, "flag": ["-O1"]}
    )
    rows = result["submissions"]
    assert len(dispatches) == 2
    assert [Path(row["file"]).stem for row in rows] == ["a", "bad", "b", "c"]
    assert rows[1]["findings"][0]["key"] == "land.request"
    if consumer_recipe_refused:
        assert dispatches[0] == [("src/y.c", {}), ("src/z.c", {})]
        assert [path for path, _ in dispatches[1]] == ["src/b.c", "src/y.c", "src/z.c"]
        assert [row["refused"] for row in rows] == [True, True, False, True]
        assert [row["id"] for row in rows] == [None, None, "sub-src/b.c", None]
        assert [rows[n]["findings"][0]["key"] for n in (0, 3)] == ["recipe.option", "recipe.option"]
        assert [call["extras"] for call in accepted] == [{"include/h.h": b"B"}]
        assert [tuple(p.unit for p in call["proofs"]) for call in accepted] == [("src/b.c",) * 2]
        assert audited == []
    else:
        assert dispatches[0] == [("src/x.c", {}), ("src/y.c", {}), ("src/z.c", {})]
        assert [p for p, _ in dispatches[1]] == [
            "src/a.c",
            "src/x.c",
            "src/y.c",
            "src/b.c",
            "src/y.c",
            "src/z.c",
            "src/c.c",
            "src/x.c",
            "src/y.c",
        ]
        assert [writes["include/h.h"] for _, writes in dispatches[1]] == [b"A"] * 3 + [b"B"] * 3 + [b"C"] * 3
        assert [row["refused"] for row in rows] == [False, True, False, True]
        assert [row["id"] for row in rows] == ["sub-src/a.c", None, "sub-src/b.c", None]
        assert {(f["key"], f["unit"], f["versions"][0]) for f in rows[3]["findings"]} == {
            ("land.consumer_regressed", "src/x.c", "a"),
            ("land.consumer_regressed", "src/x.c", "b"),
        }
        assert [call["extras"] for call in accepted] == [{"include/h.h": b"A"}, {"include/h.h": b"B"}]
        assert [tuple(p.unit for p in call["proofs"]) for call in accepted] == [("src/a.c",) * 2, ("src/b.c",) * 2]
        assert audited == ["src/c.c"]
    assert header.read_bytes() == b"old"
    configuration.validate("result.submit", json.loads(json.dumps(result)), "result.submit")


def test_empty_bulk_has_no_capture_baseline_or_measurement(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("empty bulk scheduled work")

    monkeypatch.setattr(compare.layout, "capture", unexpected)
    monkeypatch.setattr(compare, "_baseline", unexpected)
    monkeypatch.setattr(compare, "measure_bound", unexpected)
    monkeypatch.setattr(compare.land, "submit_many", unexpected)
    assert compare._bulk(cfg, [], {"bulk": True}) == {"submissions": [], "drain": None}
    # The public command still rejects a request without candidates.
    with pytest.raises(Refusal, match="Pass source files"):
        compare.submit_command(cfg, {"files": [], "bulk": True})


def test_entirely_refused_bulk_chunk_never_warms_or_measures(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compare.layout, "capture", lambda config: snap(cfg, {}))

    def refuse(snapshot: Any, files: Any, params: Any) -> Any:
        raise Refusal(Finding("land.request", "no member", path=str(files[0])))

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("refused groups scheduled proof work")

    monkeypatch.setattr(compare, "_prepare", refuse)
    monkeypatch.setattr(compare, "_baseline", unexpected)
    monkeypatch.setattr(compare, "measure_bound", unexpected)
    monkeypatch.setattr(compare.land, "submit_many", lambda config, calls: [] if not calls else unexpected())
    files = [tmp_path / "a.c", tmp_path / "b.c"]
    rows = compare.submit_command(cfg, {"files": files, "bulk": True})["submissions"]
    assert [row["file"] for row in rows] == [str(f) for f in files]
    assert all(row["refused"] and row["findings"][0]["key"] == "land.request" for row in rows)


def test_bulk_mixes_sibling_and_standalone_candidates_in_one_measurement_batch(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatched: list[Any] = []
    rows = _bulk_pair(
        cfg, tmp_path, monkeypatch, {"a": b"newA", "b": b"newB"}, lambda path, header: 1.0, dispatched, ("src/b.c",)
    )
    assert len(dispatched) == 2
    assert dispatched[0] == [("src/b.c", {})]
    assert [path for path, _ in dispatched[1]] == ["src/a.c", "src/b.c", "src/b.c", "src/b.c", "src/b.c"]
    assert dispatched[1][-1][1] == {"include/h.h": b"newB", "src/b.c": b"int b;"}
    assert [row["id"] for row in rows] == ["sub-src/a.c", "sub-src/b.c"]
    assert [row["consumers"]["count"] for row in rows] == [1, 0]


@pytest.mark.parametrize("option", ["flag", "omit_flag"])
def test_bulk_primary_recipe_refusal_keeps_its_row_and_excludes_its_draft_from_shared_work(
    cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    flag = "-funroll-loops" if option == "flag" else "-O1"
    specs = {
        name: replace(unit(name, path=f"src/{name}.c"), options={"add": [flag]} if name != "b" else {})
        for name in "abc"
    }
    snapshot = snap(cfg, {name: member(name) for name in specs}, tuple(specs.values()))
    monkeypatch.setattr(compare.layout, "capture", lambda config: snapshot)
    files = [tmp_path / f"{name}.c" for name in specs]
    for file in files:
        file.write_text(f"int {file.stem}(void) {{ return 1; }}")
    monkeypatch.setattr(
        compare,
        "bind",
        lambda tree, file, function, notes=None: (
            specs[file.stem],
            replace(tree, overlays={specs[file.stem].path: file.read_bytes()}, digest=file.stem),
        ),
    )
    monkeypatch.setattr(compare.headers, "consumers", lambda tree, changed: ())
    derived: list[Any] = []

    def derive(pairs: Any, loose: bool = False, recipe: Any = None) -> Any:
        derived.append(([u.path for _, u in pairs], recipe))
        return list(pairs)

    monkeypatch.setattr(ownership, "derive_pairs", derive)
    stage = compare._stage
    staged: list[Any] = []

    def staging(config: Any, tree: Any, group: Any, params: Any, overrides: Any, prepared: Any, kin: Any) -> Any:
        staged.append((prepared[1].path, tuple(kin)))
        return stage(config, tree, group, params, overrides, prepared, kin)

    monkeypatch.setattr(compare, "_stage", staging)
    measured: list[Any] = []

    def measure(config: Any, pairs: Any, overrides: Any) -> Any:
        measured.append([u.path for _, u in pairs])
        return [tuple(fixture.proof(u.path, u.members[0], v, True, ()) for v in ("a", "b")) for _, u in pairs]

    monkeypatch.setattr(compare, "measure_bound", measure)
    entered: list[Any] = []

    def submit_many(config: Any, calls: Any) -> Any:
        entered.extend(calls)
        return [fixture.submission(id=f"sub-{call['unit'].path}") for call in calls]

    monkeypatch.setattr(compare.land, "submit_many", submit_many)
    result = compare.submit_command(
        cfg, {"files": files, "bulk": True, "function": None, "note": None, "withhold": False, option: [flag]}
    )
    rows = result["submissions"]
    assert [Path(row["file"]).stem for row in rows] == ["a", "b", "c"]
    assert [row["id"] for row in rows] == ["sub-src/a.c", None, "sub-src/c.c"]
    assert [row["refused"] for row in rows] == [False, True, False]
    assert [(f["key"], f["unit"]) for f in rows[1]["findings"]] == [("recipe.option", "src/b.c")]
    action = "add" if option == "flag" else "omit"
    assert f"cannot {action} {flag}" in rows[1]["findings"][0]["reason"]
    assert len(derived) == 1 and derived[0][0] == ["src/a.c", "src/c.c"] and len(derived[0][1]) == 2
    assert staged == [("src/a.c", ("src/c.c",)), ("src/c.c", ("src/a.c",))]
    assert measured == [["src/a.c", "src/c.c"]]
    assert [call["unit"].path for call in entered] == ["src/a.c", "src/c.c"]
    assert all(call["request"]["overrides"]["add" if option == "flag" else "omit"] == [flag] for call in entered)
    configuration.validate("result.submit", json.loads(json.dumps(result)), "result.submit")


def test_measure_bound_coalesces_native_stamps_across_trees_and_restores_every_pair(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bulk union used to dispatch duplicate cold misses: identical consumers in distinct overlays
    # lost the reuse that separate groups got when the earlier group completed its proof.
    a, b = unit("f"), unit("g", path="src/b.c")
    alternate = replace(a, toolchain="gcc-alternative")
    base = snap(cfg, {"f": member("f"), "g": member("g", ("b",))}, (a, b))
    unrelated = replace(base, digest="unrelated", overlays={"src/unrelated.c": b"changed"})
    changed = replace(base, digest="changed", overlays={"include/shared.h": b"bad"})
    pairs = [(base, a), (unrelated, a), (base, b), (changed, a), (base, alternate)]
    monkeypatch.setattr(
        compare.recipes,
        "resolve",
        lambda config, spec, overrides: replace(RECIPE, toolchain=spec.toolchain, digest=spec.toolchain),
    )
    named: list[Any] = []
    built: list[Any] = []
    dispatches: list[tuple[str, int]] = []

    def stamp(tree: Snapshot, spec: UnitSpec, version: str, recipe: Recipe) -> str:
        inputs = (spec.path, version, recipe.digest, tree.overlays.get("include/shared.h"))
        named.append(inputs)
        return digest(inputs)

    def prove(item: tuple[Any, ...]) -> tuple[Any, ...]:
        tree, spec, version, recipe, work, export = item
        built.append((spec.path, version, recipe.digest, tree.overlays.get("include/shared.h")))
        assert not export and work.is_relative_to(cfg.project.root)
        exact = "include/shared.h" not in tree.overlays
        return (fixture.proof(spec.path, spec.members[0], version, exact, () if exact else ("compile.error",)),)

    def mapped(config: Config, name: str, function: Any, items: Any, key: Any = None) -> list[Any]:
        dispatches.append((name, len(items)))
        if key:
            # Reading precomputed stamps must not evaluate native.stamp again, even with no cache hits.
            assert len({key(item) for item in items}) == len(items)
        return [function(item) for item in items]

    monkeypatch.setattr(compare.native, "stamp", stamp)
    monkeypatch.setattr(compare.native, "prove_args", prove)
    monkeypatch.setattr(compare.pool, "map", mapped)
    results = compare.measure_bound(cfg, pairs, {})
    assert dispatches == [("compare.stamps", 9), ("compare.measure", 7)]
    assert len(named) == 9 and len(built) == len(set(named)) == 7
    assert results[0] == results[1]
    assert [tuple(p.version for p in group) for group in results] == [
        ("a", "b"),
        ("a", "b"),
        ("b",),
        ("a", "b"),
        ("a", "b"),
    ]
    assert all(p.unit == b.path for p in results[2])
    assert all(not p.exact and p.missing == ("compile.error",) for p in results[3])
    assert sum(row[2] == "gcc-alternative" for row in built) == 2


def test_measure_bound_without_held_versions_dispatches_nothing(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("an empty proof request scheduled work")

    monkeypatch.setattr(compare.pool, "map", unexpected)
    snapshot = snap(cfg, {})
    assert compare.measure_bound(cfg, [], {}) == []
    assert compare.measure_bound(cfg, [(snapshot, unit())], {}) == [()]


def test_a_variants_row_carries_its_own_flag_set_and_function(tmp_path):
    row = {"file": "a.c", "flags": ["-ffast-math"], "omit_flags": ["-O2"], "function": "f"}
    got = compare._variant_params({"flag": ["-g"], "omit_flag": [], "function": None}, row)
    assert (got["flag"], got["omit_flag"], got["function"]) == (["-ffast-math"], ["-O2"], "f")
    plain = compare._variant_params({"flag": ["-g"], "omit_flag": [], "function": None}, {"file": "a.c"})
    assert (plain["flag"], plain["function"]) == (["-g"], None)  # the command's own set stands
    with pytest.raises(Refusal):
        compare._variant_params({}, {"file": "a.c", "flags": "-ffast-math"})


def test_bind_resolves_an_explicit_function_name_renamed_after_submission(cfg, tmp_path, monkeypatch):
    u = unit("new_fn")
    s = snap(cfg, {"new_fn": member("new_fn")}, (u,))
    moved.record(cfg, [("symbol", "old_fn", "new_fn")], "after")
    source = tmp_path / "old_fn.c"
    source.write_bytes(b"int old_fn(void) { return 1; }\n")
    writes = []
    monkeypatch.setattr(compare.layout, "overlay", lambda snapshot, changed: writes.append(changed) or snapshot)
    notes = []
    assert compare.bind(s, source, "old_fn", notes)[0] == u
    assert b"int new_fn(void)" in writes[0][u.path]
    assert "old_fn->new_fn" in notes
