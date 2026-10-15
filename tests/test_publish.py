"""Publish lane: real contract records, mocked native/pool/journal boundaries."""

from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock

import fixture
import pytest

from unbake import ownership, publish
from unbake import policy as _policy
from unbake.contracts import (
    Finding,
    Group,
    LayoutMap,
    Member,
    Placement,
    Receipt,
    Recipe,
    Refusal,
    Snapshot,
    SourceView,
    UnitSpec,
    Version,
    digest,
)


@pytest.fixture
def lane(tmp_path, monkeypatch):
    config = fixture.bare_config(tmp_path, memory=1000)
    unit = UnitSpec("src/f.c", "c", "group", ("f",), "gcc-test", {"add": [], "omit": []})
    group = Group("group", "main", ("f",), "authored", (), False, "src/group.c", "include/main/group.h")
    placements = tuple(Placement(v, ".text", 4096, 4104, 0x80000400) for v in ("a", "b"))
    member = Member("f", "function", "asm", "group", placements)
    mapping = LayoutMap({"group": group}, {"f": member}, {}, "map", (), {})
    versions = {
        v: Version(v, tmp_path / f"{v}.rom", "rom", f"versions/{v}/Game.yaml", f"versions/{v}/symbols.txt", {}, ())
        for v in ("a", "b")
    }
    files = {
        "src/f.c": b"int f(void) { return 1; }\n",
        "types.toml": b"old types\n",
        "Makefile": b"same make\n",
        "README.md": b"old readme\n",
    }
    snapshot = Snapshot(config, "current-head", mapping, versions, files, "snapshot")
    proposed = replace(snapshot, layout=replace(mapping, units={unit.path: unit}), digest="proposed")
    recipe = Recipe("gcc-test", (), (), (), "recipe")
    proofs = tuple(fixture.proof(unit.path, "f", v, True) for v in ("a", "b"))
    source = tmp_path / "candidate.c"
    source.write_bytes(files[unit.path])
    work = tmp_path / ".unbake" / "work"
    work.mkdir(parents=True)
    dumps = {}

    def dump_map(value):
        data = digest(value).encode()
        dumps[data] = value
        return data

    def overlay(snap, writes):
        merged = {**snap.overlays, **writes}
        mapped = dumps.get(merged.get("layout.toml"), snap.layout)
        return replace(snap, overlays=merged, layout=mapped, digest=digest((snap.digest, writes)))

    monkeypatch.setattr(publish.effort, "stage", Mock(side_effect=lambda name: nullcontext()))
    monkeypatch.setattr(publish.effort, "invocation", Mock(return_value="invocation"))
    monkeypatch.setattr(publish.layout, "capture", Mock(return_value=snapshot))
    monkeypatch.setattr(publish.layout, "overlay", Mock(side_effect=overlay))
    monkeypatch.setattr(publish.layout, "dump_map", Mock(side_effect=dump_map))
    monkeypatch.setattr(
        publish.layout,
        "unit_of",
        Mock(side_effect=lambda snap, name: next((u for u in snap.layout.units.values() if name in u.members), None)),
    )
    monkeypatch.setattr(publish.layout, "unit_options", Mock(return_value=[(unit, {})]))
    monkeypatch.setattr(publish.compare, "bind", Mock(return_value=(unit, proposed)))
    monkeypatch.setattr(publish.compare, "holders", Mock(return_value=("a", "b")))
    monkeypatch.setattr(publish.compare, "gaps", Mock(return_value=()))
    monkeypatch.setattr(publish.compare, "fuzzy_keys", Mock(return_value=[{"a": "ka", "b": "kb"}]))
    monkeypatch.setattr(publish.rescore, "writes", Mock(return_value={}))
    monkeypatch.setattr(publish.compare, "measure", Mock(return_value=proofs))
    monkeypatch.setattr(publish.recipes, "resolve", Mock(return_value=recipe))
    views = {v: SourceView(unit.path, v, v, "source", (), ()) for v in ("a", "b")}
    monkeypatch.setattr(publish.view, "get", Mock(return_value=views["a"]))
    monkeypatch.setattr(publish.view, "all_versions", Mock(return_value=views))
    monkeypatch.setattr(publish.policy, "evaluate", Mock(return_value=()))
    monkeypatch.setattr(publish.policy, "scope", Mock(return_value=((), ())))
    monkeypatch.setattr(publish.headers, "fold", Mock(return_value=({unit.path: files[unit.path]}, ())))
    monkeypatch.setattr(publish.headers, "reconcile", Mock(side_effect=lambda snap, over, writes: (writes, [])))
    monkeypatch.setattr(publish.headers, "dangling", Mock(return_value=()))
    monkeypatch.setattr(publish.types, "scan", Mock(return_value=files["types.toml"]))
    monkeypatch.setattr(publish.report, "files", Mock(return_value={"README.md": b"readme"}))
    monkeypatch.setattr(publish.report, "progress", Mock(return_value={"versions": {}}))
    monkeypatch.setattr(publish.report, "readme", Mock(side_effect=lambda text, snap, current: text))
    monkeypatch.setattr(publish.present, "files", Mock(return_value={}))
    monkeypatch.setattr(publish.repo, "files", Mock(return_value={"Makefile": files["Makefile"]}))
    monkeypatch.setattr(publish.store, "work", Mock(side_effect=lambda cfg: nullcontext(work)))
    monkeypatch.setattr(
        publish.pool, "map", Mock(side_effect=lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    )
    monkeypatch.setattr(publish.native, "objects", Mock(side_effect=lambda *args: (args[-1] / "out.o", ())))
    monkeypatch.setattr(publish.native, "measure", Mock(side_effect=lambda *args: (proofs[("a", "b").index(args[2])],)))
    monkeypatch.setattr(publish.native, "prove", Mock(side_effect=lambda *args: (proofs[("a", "b").index(args[2])],)))
    monkeypatch.setattr(publish.native, "warnings", Mock(return_value=()))
    monkeypatch.setattr(publish.versions, "undefined", Mock(return_value=()))
    monkeypatch.setattr(publish.versions, "resolve", Mock(return_value=()))
    monkeypatch.setattr(publish.journal, "apply", Mock(return_value=("src/f.c",)))
    submission = fixture.submission(member="f", source="candidate.c", proofs=proofs)
    return {
        "snapshot": snapshot,
        "proposed": proposed,
        "config": config,
        "unit": unit,
        "proofs": proofs,
        "source": source,
        "work": work,
        "dumps": dumps,
        "submission": submission,
    }


REAL_SCOPE = _policy.scope


@pytest.fixture(autouse=True)
def _no_ownership(monkeypatch):  # the units here have no source to compile: they own what they list
    derive = Mock(side_effect=lambda snapshot, unit: (snapshot, unit))
    monkeypatch.setattr(ownership, "derive", derive)
    monkeypatch.setattr(ownership, "derive_many", lambda snapshot, units: [derive(snapshot, u) for u in units])
    return derive


def _request(lane, **changes):
    return {"file": "candidate.c", "function": "f", "overrides": {"add": [], "omit": []}, "note": "", **changes}


def _gap(reason="changed bytes"):
    return Finding("land.not_exact", reason, unit="f", missing=("bytes",), symptoms={"bytes_differ": True})


def test_admission_order(lane):
    tracker = Mock()
    for name, mock in (
        ("bind", publish.compare.bind),
        ("policy", publish.policy.evaluate),
        ("recipe", publish.recipes.resolve),
        ("derive", publish.ownership.derive),
        ("views", publish.view.all_versions),
        ("objects", publish.native.objects),
        ("undefined", publish.versions.undefined),
        ("resolve", publish.versions.resolve),
        ("measure", publish.native.measure),
        ("gaps", publish.compare.gaps),
    ):
        tracker.attach_mock(mock, name)
    overrides = {"toolchain": "alternative", "add": ["-O1"], "omit": []}
    publish.recipes.resolve.return_value = Recipe("alternative", (), (), (), "alternative")
    unit, snap, proofs = publish.admit(lane["snapshot"], _request(lane, overrides=overrides))
    assert [c[0] for c in tracker.mock_calls] == [
        "bind",
        "policy",
        "recipe",
        "derive",
        "views",
        "policy",
        "objects",
        "objects",
        "undefined",
        "resolve",
        "undefined",
        "resolve",
        "measure",
        "measure",
        "gaps",
    ]
    assert unit.toolchain == "alternative"
    assert unit.options == {"add": ["-O1"], "omit": []}
    assert snap is lane["proposed"] and proofs == lane["proofs"]
    assert publish.policy.evaluate.call_args_list[0].args[3] is None
    assert publish.policy.evaluate.call_args_list[1].args[3].version == "a"
    assert [c.args[-1] for c in publish.native.objects.call_args_list] == [lane["work"] / "0", lane["work"] / "1"]
    assert [c.args[-1] for c in publish.native.measure.call_args_list] == [lane["work"] / "0", lane["work"] / "1"]
    stages = [c.args[0] for c in publish.effort.stage.call_args_list]
    assert stages == ["publish.admit", *[f"publish.admit.{n}" for n in (1, 3, 4, 5, 6, 7, 8)]]


def test_admission_proves_the_unit_with_the_data_its_source_emits(lane, _no_ownership):
    owned = replace(lane["unit"], members=(*lane["unit"].members, "rodata/f/80000000"))
    derived = replace(lane["proposed"], digest="derived")
    _no_ownership.side_effect = lambda snapshot, unit: (derived, owned)
    unit, snap, _ = publish.admit(lane["snapshot"], _request(lane))
    assert unit is owned and snap is derived
    assert all(c.args[0] is derived and c.args[1] is owned for c in publish.native.measure.call_args_list)


@pytest.mark.parametrize("step", ["owner", "policy3", "policy5", "unresolved", "gaps"])
def test_admission_refuses_at_gate(lane, step):
    snapshot = lane["snapshot"]
    request = _request(lane)
    finding = _gap()
    if step == "owner":
        owner = replace(lane["unit"], path="src/other.c")
        snapshot = replace(
            snapshot,
            layout=replace(
                snapshot.layout,
                units={owner.path: owner},
                members={"f": replace(snapshot.layout.members["f"], state="c")},
            ),
        )
    elif step.startswith("policy"):
        publish.policy.scope.side_effect = REAL_SCOPE
        finding = replace(finding, path=lane["unit"].path)
        publish.policy.evaluate.side_effect = [(finding,), (), ()] if step == "policy3" else [(), (finding,), ()]
    elif step == "unresolved":
        publish.versions.resolve.return_value = (finding,)
    else:
        publish.compare.gaps.return_value = (finding,)
    with pytest.raises(Refusal) as exc:
        publish.admit(snapshot, request)
    assert exc.value.findings[0].key == ("land.request" if step == "owner" else finding.key)
    if step == "owner":
        publish.native.objects.assert_not_called()
    if step in ("owner", "unresolved"):  # what does not link measures nothing; a source rule still measures
        publish.native.measure.assert_not_called()
    publish.journal.apply.assert_not_called()


def test_one_admission_refusal_names_every_class_it_finds_each_once(lane, monkeypatch):
    """A source rule no longer hides the view warning and the inexact bytes behind it: one submit lists all three."""
    publish.policy.scope.side_effect = REAL_SCOPE
    rule = Finding("source.shared-declarations", "repeated extern", path=lane["unit"].path, line=8)
    warned = Finding("land.view-conversion", "view warning", path=lane["unit"].path, line=1)
    publish.policy.evaluate.side_effect = [(rule,), (), (rule,), ()]  # the view pass repeats the text rule
    monkeypatch.setattr(publish, "view_findings", Mock(return_value=(warned,)))
    publish.compare.gaps.return_value = (_gap(),)
    with pytest.raises(Refusal) as exc:
        publish.admit(lane["snapshot"], _request(lane))
    assert [f.key for f in exc.value.findings] == [
        "source.shared-declarations",
        "land.view-conversion",
        "land.not_exact",
    ]
    assert publish.native.objects.call_count == 2 and publish.native.measure.call_count == 2


def test_admission_allows_nonblocking_debt(lane):
    publish.policy.evaluate.return_value = (Finding("land.request", "existing debt", blocking=False),)
    assert publish.admit(lane["snapshot"], _request(lane))[2] == lane["proofs"]


def test_admission_ignores_findings_the_landed_text_already_carries(lane):
    publish.policy.scope.side_effect = REAL_SCOPE
    old = Finding("source.version-guard", "old guard", path=lane["unit"].path)
    new = Finding("source.volatile-storage", "new qualifier", path=lane["unit"].path)
    publish.policy.evaluate.side_effect = [(old,), (old,), (), (old,), (old,), ()]
    assert publish.admit(lane["snapshot"], _request(lane))[2] == lane["proofs"]
    publish.policy.evaluate.side_effect = [(old, new), (old,), ()]
    with pytest.raises(Refusal) as exc:
        publish.admit(lane["snapshot"], _request(lane))
    assert [f.reason for f in exc.value.findings] == ["new qualifier"]


def test_plan_removes_fuzzy_row_when_exact(lane):
    snapshot = lane["snapshot"]
    fuzzy_path = "src/fuzzy/f.c"
    snapshot = replace(
        snapshot,
        layout=replace(
            snapshot.layout,
            fuzzy={
                "f": {"path": fuzzy_path, "scores": {"a": 0.7, "b": 0.6}},
                "other": {"path": "src/fuzzy/other.c", "scores": {"a": 0.5}},
            },
        ),
    )
    (plan,) = publish.plans(snapshot, lane["unit"], lane["proposed"], {})
    assert plan.writes[fuzzy_path] is None
    mapped = lane["dumps"][plan.writes["layout.toml"]]
    assert "f" not in mapped.fuzzy and "other" in mapped.fuzzy
    assert mapped.units[lane["unit"].path] == lane["unit"]


def test_plan_replaces_the_file_of_a_member_that_is_already_landed(lane):
    snapshot, unit = lane["snapshot"], lane["unit"]
    snapshot = replace(
        snapshot,
        layout=replace(
            snapshot.layout, units={unit.path: unit}, members={"f": replace(snapshot.layout.members["f"], state="c")}
        ),
    )
    publish.layout.unit_options.side_effect = AssertionError("a landed member needs no new unit option")
    (plan,) = publish.plans(snapshot, unit, lane["proposed"], {})
    assert plan.writes[unit.path] == lane["source"].read_bytes()
    publish.headers.fold.assert_not_called()
    assert lane["dumps"][plan.writes["layout.toml"]].units[unit.path] == unit


def test_plan_carries_only_the_readme_block_and_scans_nothing(lane):
    publish.types.scan.side_effect = publish.repo.files.side_effect = AssertionError("generated once per drain")
    publish.report.readme.side_effect = lambda text, snap, current: b"block of the landed tree\n".decode()
    (plan,) = publish.plans(lane["snapshot"], lane["unit"], lane["proposed"], {})
    assert not {"types.toml", "Makefile"} & plan.writes.keys()
    assert plan.writes["README.md"] == b"block of the landed tree\n"  # committed with the landing, never stale
    assert plan.affected == ("src/f.c",) and plan.operation == "publish"
    assert plan.digest == digest(
        (plan.operation, plan.base, plan.writes, plan.affected, plan.blocking, plan.debt, plan.message)
    )


def test_a_generate_over_the_inputs_the_last_one_derived_from_derives_nothing_again(lane, monkeypatch):
    """A drain that landed nothing (a refused or fuzzy-held entry) re-derived 31,638 jobs in 645 s for no change."""
    monkeypatch.setattr(
        publish.headers, "sources", lambda snapshot, *a: [p for p in snapshot.overlays if p.endswith(".c")]
    )
    publish.repo.files.return_value = {"README.md": b"new readme"}
    publish.types.scan.return_value = b"old types\n"
    publish.generate(lane["config"])
    assert publish.rescore.writes.call_count == publish.types.scan.call_count == publish.repo.files.call_count == 1
    assert publish.generate(lane["config"]) == ()  # the same tree: nothing scanned, scored or derived
    assert publish.rescore.writes.call_count == publish.types.scan.call_count == publish.repo.files.call_count == 1
    lane["snapshot"].overlays["src/f.c"] = b"int f(void) { return 2; }\n"  # an input moved: derived again
    publish.generate(lane["config"])
    assert publish.types.scan.call_count == publish.repo.files.call_count == 2


def test_generate_writes_changed_repo_files_and_the_type_map_once(lane):
    changed = {"README.md": b"new readme", "versions/a/report.json": b"{}"}
    publish.repo.files.return_value = {"Makefile": b"same make\n", **changed}
    publish.types.scan.return_value = b"new types"
    publish.generate(lane["config"])
    plan = publish.journal.apply.call_args.args[1]
    assert "Makefile" not in plan.writes and plan.operation == "generate"
    assert all(plan.writes[p] == data for p, data in changed.items()) and plan.writes["types.toml"] == b"new types"
    publish.types.scan.assert_called_once()
    publish.repo.files.assert_called_once()
    assert publish.repo.files.call_args.args[0].read("types.toml") == b"new types"


def test_plan_falls_through_an_option_that_refuses_outright(lane):
    second = replace(lane["unit"], path="src/second.c")
    publish.layout.unit_options.return_value = [(lane["unit"], {}), (second, {})]
    calls = iter([Refusal(Finding("headers.parse", "no parse")), None])

    def reconcile(snap, over, writes):
        if isinstance(error := next(calls), Refusal):
            raise error
        return writes, []

    publish.headers.reconcile.side_effect = reconcile
    (plan,) = publish.plans(lane["snapshot"], lane["unit"], lane["proposed"], {})
    assert plan.affected == (second.path,)


def test_plan_refuses_with_the_last_refusal_when_every_option_refuses(lane):
    publish.headers.reconcile.side_effect = Refusal(Finding("headers.parse", "no parse"))
    with pytest.raises(Refusal) as exc:
        list(publish.plans(lane["snapshot"], lane["unit"], lane["proposed"], {}))
    assert exc.value.findings[0].key == "headers.parse"


def test_land_publish_writes_once(lane):
    receipt = publish.land(lane["config"], lane["submission"])
    plan = publish.journal.apply.call_args.args[1]
    publish.journal.apply.assert_called_once_with(lane["config"], plan, lane["snapshot"])
    assert receipt == Receipt("publish", ("src/f.c",), plan.message, plan.digest, lane["proofs"], 0, "invocation")
    publish.compare.bind.assert_called_once_with(lane["snapshot"], lane["source"], "f", [])
    assert publish.native.prove.call_count == 2
    assert [c.args[-1] for c in publish.native.prove.call_args_list] == [lane["work"] / "0", lane["work"] / "1"]


def _change_set(lane, tmp_path_files):
    snapshot, g = lane["snapshot"], replace(lane["unit"], path="src/g.c", members=("g",))
    units = {lane["unit"].path: lane["unit"], g.path: g}
    publish.layout.capture.return_value = replace(snapshot, layout=replace(snapshot.layout, units=units))
    extras = {}
    for path, data in tmp_path_files.items():
        copy = f"set.{len(extras)}"
        (lane["config"].project.root / copy).write_bytes(data)
        extras[path] = copy
    return replace(lane["submission"], extras=extras)


def test_land_change_set_is_one_plan_and_one_commit(lane):
    entry = _change_set(lane, {"src/g.c": b"int g;\n", "include/x.h": b"extern int g;\n"})
    publish.land(lane["config"], entry)
    (plan,) = (c.args[1] for c in publish.journal.apply.call_args_list)
    assert plan.writes["src/g.c"] == b"int g;\n" and plan.writes["include/x.h"] == b"extern int g;\n"
    assert {"src/f.c", "src/g.c"} <= set(plan.affected)


def test_land_change_set_refuses_every_file_when_one_unit_breaks_where_head_is_exact(lane, monkeypatch):
    entry = _change_set(lane, {"src/g.c": b"int g;\n", "include/x.h": b"extern int g;\n"})
    bad = Finding("land.not_exact", "g differs", unit="g", versions=("a",), missing=("bytes",))
    head = (*lane["proofs"], fixture.proof("src/g.c", "g", "a", True))
    monkeypatch.setattr(publish, "_proofs", Mock(side_effect=[(lane["proofs"], (bad,)), (head, ())]))
    with pytest.raises(Refusal) as exc:
        publish.land(lane["config"], entry)
    assert [(f.key, f.unit) for f in exc.value.findings] == [("land.breaks_dependent", "g")]
    publish.journal.apply.assert_not_called()


def test_land_change_set_lands_over_a_gap_head_already_has_in_another_unit(lane, monkeypatch):
    entry = _change_set(lane, {"src/g.c": b"int g;\n", "include/x.h": b"extern int g;\n"})
    standing = Finding("land.not_exact", "g unresolved", unit="g", versions=("a",), missing=("unresolved",))
    head = (*lane["proofs"], fixture.proof("src/g.c", "g", "a", False, missing=("unresolved",)))
    monkeypatch.setattr(publish, "_proofs", Mock(side_effect=[(lane["proofs"], (standing,)), (head, ())]))
    publish.land(lane["config"], entry)
    publish.journal.apply.assert_called_once()


def test_land_change_set_refuses_its_own_unit_gap_even_where_head_is_inexact(lane, monkeypatch):
    entry = _change_set(lane, {"src/g.c": b"int g;\n", "include/x.h": b"extern int g;\n"})
    own = Finding("land.not_exact", "f differs", unit="f", versions=("a",), missing=("bytes",))
    monkeypatch.setattr(publish, "_proofs", Mock(return_value=(lane["proofs"], (own,))))
    with pytest.raises(Refusal) as exc:
        publish.land(lane["config"], entry)
    assert [f.unit for f in exc.value.findings] == ["f"]


@pytest.mark.parametrize("same,decompiled", [(True, True), (False, True), (True, False)])
def test_land_duplicate_refuses(lane, same, decompiled):
    snapshot, unit = lane["snapshot"], lane["unit"]
    unit = replace(unit, kind="c" if decompiled else "asm")
    snapshot = replace(snapshot, layout=replace(snapshot.layout, units={unit.path: unit}))
    publish.layout.capture.return_value = snapshot
    if not same:
        lane["source"].write_bytes(b"changed candidate")
    if same and decompiled:
        with pytest.raises(Refusal) as exc:
            publish.land(lane["config"], lane["submission"])
        assert exc.value.findings[0].key == "land.duplicate"
        assert exc.value.findings[0].unit == "f"
        publish.compare.bind.assert_not_called()
        publish.journal.apply.assert_not_called()
    else:
        assert publish.land(lane["config"], lane["submission"]).operation == "publish"


@pytest.mark.parametrize("same", [True, False])
def test_land_replaces_the_untyped_source_of_a_landed_data_member_with_a_typed_one(lane, same):
    snapshot, unit = lane["snapshot"], lane["unit"]
    unit = replace(unit, kind="c")
    landed = replace(snapshot.layout.members["f"], kind="rodata", state=".rodata")  # landed data
    members = {**snapshot.layout.members, "f": landed}
    snapshot = replace(snapshot, layout=replace(snapshot.layout, units={unit.path: unit}, members=members))
    publish.layout.capture.return_value = snapshot
    if not same:
        lane["source"].write_bytes(b"const unsigned int f = 1;")  # the typed text of the same bytes
    publish.layout.unit_options.side_effect = AssertionError("a landed member needs no new unit option")
    if same:
        with pytest.raises(Refusal) as exc:
            publish.land(lane["config"], lane["submission"])
        assert exc.value.findings[0].key == "land.duplicate"
        publish.journal.apply.assert_not_called()
    else:
        assert publish.land(lane["config"], lane["submission"]).operation == "publish"
        publish.compare.bind.assert_called_once_with(snapshot, lane["source"], "f", [])


def test_land_not_exact_after_head_moved(lane, monkeypatch):
    snapshot = lane["snapshot"]
    plan = publish.make_plan("publish", snapshot, {"src/f.c": b"candidate"}, ("src/f.c",), (), "publish f")
    monkeypatch.setattr(publish, "admit", Mock(return_value=(lane["unit"], lane["proposed"], lane["proofs"])))
    monkeypatch.setattr(publish, "plans", Mock(return_value=[plan, replace(plan, message="alternate")]))
    stale = fixture.proof("src/f.c", "f", "a", False, ("bytes",))
    first, last = _gap("first option"), _gap("last option")
    monkeypatch.setattr(publish, "_proofs", Mock(side_effect=[((stale,), (first,)), ((stale,), (last,))]))
    with pytest.raises(Refusal) as exc:
        publish.land(lane["config"], replace(lane["submission"], base="old-head"))
    assert exc.value.findings == (last,)
    assert publish._proofs.call_count == 2
    publish.journal.apply.assert_not_called()


def test_land_first_exact_plan_wins(lane, monkeypatch):
    snapshot = lane["snapshot"]
    plans = [
        publish.make_plan("publish", snapshot, {"src/f.c": str(i).encode()}, ("src/f.c",), (), str(i)) for i in range(3)
    ]
    monkeypatch.setattr(publish, "admit", Mock(return_value=(lane["unit"], lane["proposed"], lane["proofs"])))
    monkeypatch.setattr(publish, "plans", Mock(return_value=plans))
    monkeypatch.setattr(publish, "_proofs", Mock(side_effect=[((), (_gap(),)), (lane["proofs"], ())]))
    receipt = publish.land(lane["config"], lane["submission"])
    assert receipt.plan == plans[1].digest
    assert publish._proofs.call_count == 2
    publish.journal.apply.assert_called_once_with(lane["config"], plans[1], snapshot)


@pytest.mark.parametrize("exact_at_head, lands", [(False, True), (True, False)])
def test_land_a_consumers_gap_blocks_only_when_head_proves_it_exact(lane, monkeypatch, exact_at_head, lands):
    snapshot = lane["snapshot"]
    plan = publish.make_plan("publish", snapshot, {"src/f.c": b"0"}, ("src/f.c", "src/g.c"), (), "0")
    error = "src/g.c:3: conflicting types for 'x'"
    consumer = Finding("land.not_exact", "g differs", unit="g", versions=("a",), missing=(error,))
    monkeypatch.setattr(publish, "admit", Mock(return_value=(lane["unit"], lane["proposed"], lane["proofs"])))
    monkeypatch.setattr(publish, "plans", Mock(return_value=[plan]))
    unit_of = publish.layout.unit_of.side_effect
    g = replace(lane["unit"], path="src/g.c", members=("g",))
    pick = Mock(side_effect=lambda snap, name: g if name == "g" else unit_of(snap, name))
    monkeypatch.setattr(publish.layout, "unit_of", pick)
    head = (fixture.proof("src/g.c", "g", "a", exact_at_head, () if exact_at_head else ("bytes",)),)
    monkeypatch.setattr(publish, "_proofs", Mock(side_effect=[(lane["proofs"], (consumer,)), (head, ())]))
    if lands:
        assert publish.land(lane["config"], lane["submission"]).plan == plan.digest
    else:
        with pytest.raises(Refusal) as exc:
            publish.land(lane["config"], lane["submission"])
        (finding,) = exc.value.findings
        assert (finding.key, finding.path, finding.unit, finding.missing) == (
            "land.breaks_dependent",
            "src/g.c",
            "g",
            (error,),
        )
        assert finding.reason == f"{lane['unit'].path} breaks src/g.c: {error}" and "src/g.c" in finding.action
    assert publish._proofs.call_args_list[1].args[1] == ["src/g.c"]


def _fuzzy_proofs(lane, scores, symptoms=None, missing=("bytes",)):
    return tuple(
        fixture.proof(lane["unit"].path, "f", v, False, missing, score, symptoms)
        for v, score in zip(("a", "b"), scores, strict=True)
    )


def test_land_fuzzy_writes_row_and_file(lane, monkeypatch):
    monkeypatch.setattr(publish.ownership, "derive_many", Mock(side_effect=AssertionError("unrelated proof")))
    proofs = _fuzzy_proofs(lane, (0.8, 0.6))
    publish.compare.measure.return_value = proofs
    publish.report.files.return_value = {"README.md": b"fuzzy readme"}
    submission = replace(lane["submission"], operation="fuzzy")
    receipts, refused = publish.land_fuzzy(lane["config"], [submission])
    receipt, plan = receipts[submission.id], publish.journal.apply.call_args.args[1]
    assert refused == {}
    publish.repo.files.assert_not_called()
    publish.ownership.derive_many.assert_not_called()
    assert plan.operation == "fuzzy" and plan.affected == () and plan.message == "fuzzy f 60.0%"
    assert plan.writes["src/fuzzy/f.c"] == lane["source"].read_bytes()
    mapped = lane["dumps"][plan.writes["layout.toml"]]
    assert mapped.fuzzy["f"] == {"path": "src/fuzzy/f.c", "scores": {"a": 0.8, "b": 0.6}, "key": {"a": "ka", "b": "kb"}}
    assert "Makefile" not in plan.writes and plan.writes["README.md"] == b"fuzzy readme"
    assert receipt == Receipt("fuzzy", ("src/f.c",), plan.message, plan.digest, (), 0, "invocation")
    publish.compare.measure.assert_called_once_with(lane["proposed"], lane["unit"], lane["submission"].overrides, None)
    assert publish.policy.evaluate.call_count == 2
    publish.native.objects.assert_not_called()
    publish.journal.apply.assert_called_once()


@pytest.mark.parametrize("built_size,score", [(4, 0.5), (12, 1.0 - 2**-53), (8, 1.0)])
@pytest.mark.parametrize("data_first", [False, True])
def test_land_fuzzy_keeps_member_score_with_exact_data(lane, built_size, score, data_first):
    exact = built_size == 8
    missing = () if exact else (f".text size {built_size} != 8",)
    proofs = []
    for version in ("a", "b"):
        code = fixture.proof(lane["unit"].path, "f", version, exact, missing, score)
        data = fixture.proof(lane["unit"].path, "rodata/f/table", version, True)
        proofs.extend((data, code) if data_first else (code, data))
    publish.compare.measure.return_value = tuple(proofs)
    submission = replace(lane["submission"], operation="fuzzy")
    receipts, refused = publish.land_fuzzy(lane["config"], [submission])
    assert not refused
    plan = publish.journal.apply.call_args.args[1]
    scores = lane["dumps"][plan.writes["layout.toml"]].fuzzy["f"]["scores"]
    assert scores == {"a": score, "b": score}
    assert all((value == 1.0) == exact for value in scores.values())
    assert receipts[submission.id].message == f"fuzzy f {score * 100:.1f}%"
    publish.compare.measure.assert_called_once_with(lane["proposed"], lane["unit"], submission.overrides, None)


@pytest.mark.parametrize(
    "prior,scores,allowed",
    [
        (None, (0.5, 0.0), False),
        ({"a": 0.8, "b": 0.5}, (0.9, 0.51), False),
        ({"a": 0.8, "b": 0.5}, (0.9, 0.55), True),
        ({"a": 0.8, "b": 0.5}, (0.9, 0.49), False),
    ],
)
def test_land_fuzzy_no_gain(lane, prior, scores, allowed, monkeypatch):
    real = publish.configuration.load_resource
    monkeypatch.setattr(
        publish.configuration,
        "load_resource",
        lambda name: {"fuzzy": {"min_gain": 0.05, "chunk": 150}} if name == "flow.toml" else real(name),
    )
    snapshot = lane["snapshot"]
    if prior is not None:
        snapshot = replace(
            snapshot, layout=replace(snapshot.layout, fuzzy={"f": {"path": "src/fuzzy/f.c", "scores": prior}})
        )
    publish.layout.capture.return_value = snapshot
    publish.compare.measure.return_value = _fuzzy_proofs(lane, scores)
    submission = replace(lane["submission"], operation="fuzzy")
    receipts, refused = publish.land_fuzzy(lane["config"], [submission])
    if allowed:
        assert receipts[submission.id].operation == "fuzzy"
    else:
        assert refused[submission.id][0].key == "fuzzy.no_gain" and not receipts
        publish.journal.apply.assert_not_called()


@pytest.mark.parametrize("symptom", ["compile_failed", "unresolved", "no_measurement"])
def test_land_fuzzy_compile_failed_refuses(lane, symptom):
    proofs = _fuzzy_proofs(lane, (0.9, 0.8), {symptom: True}, (symptom,))
    publish.compare.measure.return_value = proofs
    submission = replace(lane["submission"], operation="fuzzy")
    receipts, refused = publish.land_fuzzy(lane["config"], [submission])
    (finding,) = refused[submission.id]
    assert not receipts and finding.key == "land.not_exact" and finding.unit == "f"
    assert set(finding.missing) == {symptom} and finding.symptoms[symptom]
    publish.journal.apply.assert_not_called()


def test_land_fuzzy_writes_the_admitted_candidates_as_one_plan_and_refuses_the_others_alone(lane, monkeypatch):
    first = replace(lane["submission"], operation="fuzzy")
    second = replace(first, id="2", member="g")
    publish.compare.measure.return_value = _fuzzy_proofs(lane, (0.8, 0.6))
    monkeypatch.setattr(
        publish,
        "_fuzzy",
        Mock(
            side_effect=[
                ("src/fuzzy/f.c", b"f", {"a": 0.8, "b": 0.6}, {"a": "k"}),
                Refusal(Finding("fuzzy.no_gain", "none", unit="g")),
            ]
        ),
    )
    receipts, refused = publish.land_fuzzy(lane["config"], [first, second])
    (plan,) = (c.args[1] for c in publish.journal.apply.call_args_list)
    assert set(receipts) == {first.id} and refused["2"][0].key == "fuzzy.no_gain"
    assert plan.message == "fuzzy f 60.0%" and plan.writes["src/fuzzy/f.c"] == b"f"
    publish._fuzzy.side_effect = [
        ("src/fuzzy/f.c", b"f", {"a": 0.8}, {"a": "k"}),
        ("src/fuzzy/g.c", b"g", {"a": 0.5}, {"a": "k"}),
    ]
    publish.land_fuzzy(lane["config"], [first, second])
    plan = publish.journal.apply.call_args.args[1]
    assert plan.message == "fuzzy 2 members: f 80.0%, g 50.0%" and {"src/fuzzy/f.c", "src/fuzzy/g.c"} <= set(
        plan.writes
    )


def test_a_later_layout_option_is_planned_only_when_the_first_does_not_prove(lane, monkeypatch):
    first, second = Mock(name="first"), Mock(name="second")
    monkeypatch.setattr(publish, "_option_plan", Mock(side_effect=[first, second]))
    monkeypatch.setattr(publish, "Plan", type(first))
    monkeypatch.setattr(publish.layout, "unit_options", Mock(return_value=[(lane["unit"], {}), (lane["unit"], {})]))
    monkeypatch.setattr(publish.headers, "fold", Mock(return_value=({lane["unit"].path: b"x"}, ())))
    monkeypatch.setattr(publish.view, "get", Mock())
    monkeypatch.setattr(publish.compare, "holders", Mock(return_value=("a",)))
    taken = publish.plans(lane["snapshot"], lane["unit"], lane["proposed"], {})
    assert next(taken) is first and publish._option_plan.call_count == 1


def test_admission_refuses_a_view_conversion_in_new_text_only(lane):
    text = lane["snapshot"].read(lane["unit"].path).decode()
    publish.native.warnings.return_value = (f"{lane['unit'].path}: 1: warning: makes pointer from integer",)
    snapshot = replace(lane["snapshot"], overlays={lane["unit"].path: b"int f(void) { return 0; }\n"})
    publish.compare.bind.return_value = (
        lane["unit"],
        replace(lane["proposed"], overlays={lane["unit"].path: text.encode()}),
    )
    with pytest.raises(Refusal) as exc:
        publish.admit(snapshot, _request(lane))
    assert [(f.key, f.line) for f in exc.value.findings] == [("land.view-conversion", 1)]
    same = replace(lane["snapshot"], overlays={lane["unit"].path: text.encode()})
    assert publish.admit(same, _request(lane))[2] == lane["proofs"]  # the line was already landed text


def _bad_in(*versions):
    return Finding("land.not_exact", "bytes", unit="f", versions=versions, missing=("bytes",))


def test_partial_admission_withholds_the_inexact_versions(lane):
    publish.compare.gaps.return_value = (_bad_in("b"),)
    with pytest.raises(Refusal):
        publish.admit(lane["snapshot"], _request(lane))
    unit, _, _ = publish.admit(lane["snapshot"], _request(lane), True)
    assert unit.withheld == ("b",)


def test_partial_admission_refuses_when_every_version_is_inexact(lane):
    publish.compare.gaps.return_value = (_bad_in("a", "b"),)
    with pytest.raises(Refusal):
        publish.admit(lane["snapshot"], _request(lane), True)


def test_partial_landing_writes_the_withheld_unit_and_names_the_exact_versions(lane):
    publish.compare.gaps.side_effect = lambda snap, unit, proofs: (_bad_in("b"),) if "b" not in unit.withheld else ()
    entry = replace(lane["submission"], withheld=("b",))
    publish.land(lane["config"], entry)
    plan = publish.journal.apply.call_args.args[1]
    assert plan.message == "publish f (a exact; b withheld)"
    assert lane["dumps"][plan.writes["layout.toml"]].units["src/f.c"].withheld == ("b",)
    assert [c.args[2] for c in publish.native.prove.call_args_list] == ["a"]


def test_partial_change_set_lands_extras_with_the_main_unit_withheld(lane):
    publish.compare.gaps.side_effect = lambda snap, unit, proofs: (_bad_in("b"),) if "b" not in unit.withheld else ()
    entry = replace(_change_set(lane, {"include/x.h": b"extern int g;\n"}), withheld=("b",))
    publish.land(lane["config"], entry)
    plan = publish.journal.apply.call_args.args[1]
    assert plan.writes["include/x.h"] == b"extern int g;\n"
    assert lane["dumps"][plan.writes["layout.toml"]].units["src/f.c"].withheld == ("b",)


def test_shared_prefix_claim_rows_and_retired_providers_are_published_atomically(lane, monkeypatch):
    snapshot, prefix = lane["snapshot"], lane["unit"]
    new = replace(snapshot.layout.members["f"], name="g")
    pool = Member("old_pool", "data", ".rodata", "group", ())
    old = replace(prefix, path="src/pool.c", kind="data", members=(pool.name,))
    members = {
        **snapshot.layout.members,
        "f": replace(snapshot.layout.members["f"], state="c"),
        new.name: new,
        pool.name: pool,
    }
    snapshot = replace(
        snapshot, layout=replace(snapshot.layout, members=members, units={prefix.path: prefix, old.path: old})
    )
    combined = replace(prefix, members=("f", "g"))
    split = snapshot.versions["a"].split
    proposed = replace(
        snapshot,
        layout=replace(snapshot.layout, units={combined.path: combined}),
        overlays={**snapshot.overlays, split: b"renamed rows", combined.path: b"shared source"},
    )
    original = publish.layout.overlay.side_effect

    def atomic(snap, writes):
        got = original(snap, writes)
        if got.overlays.get(split) == b"renamed rows":
            assert old.path not in got.layout.units
            assert got.layout.units[combined.path].members == ("f", "g")
        return got

    monkeypatch.setattr(publish.layout, "overlay", Mock(side_effect=atomic))
    plan = next(publish.plans(snapshot, combined, proposed, {}))
    assert plan.writes[split] == b"renamed rows" and plan.writes[old.path] is None
    assert plan.writes[combined.path] == b"shared source"
    assert "publish g" in plan.message


def test_the_proof_of_a_landing_is_named_by_what_it_reads_so_a_warm_one_never_travels(monkeypatch):
    seen = []
    monkeypatch.setattr(publish.native, "stamp", lambda *a: seen.append(a) or "stamp")
    assert publish._stamp_key(("snapshot", "unit", "a", "recipe", "work")) == "stamp"
    assert seen == [("snapshot", "unit", "a", "recipe")]  # the recipe of the landing, not the unit's own


def test_plan_gives_the_unit_that_takes_the_source_the_data_rows_its_compile_claimed(lane):
    option = replace(lane["unit"], members=("f",))
    claimed = replace(lane["unit"], members=("f", "rodata/f/00000100"))
    publish.layout.unit_options.return_value = [(option, {})]
    row = Member("rodata/f/00000100", "data", "rodata", "group", ())
    layout_map = replace(lane["proposed"].layout, members={**lane["proposed"].layout.members, row.name: row})
    proposed = replace(lane["proposed"], overlays={"versions/a/Game.yaml": b"renamed rows"}, layout=layout_map)
    (plan,) = publish.plans(lane["snapshot"], claimed, proposed, {})
    assert plan.writes["versions/a/Game.yaml"] == b"renamed rows"
    assert lane["dumps"][plan.writes["layout.toml"]].units[option.path].members == ("f", "rodata/f/00000100")


def test_a_change_set_lands_on_the_type_map_its_files_make(lane, monkeypatch):
    """A definition the set rewrites is the type its callers are judged against: the map is read once from the set."""
    snapshot = lane["snapshot"]
    scans = []
    monkeypatch.setattr(publish.types, "scan", lambda s: scans.append(s.peek("src/g.c")) or b"new types\n")
    assert publish._changed(snapshot, {}) is snapshot and scans == []
    landed = publish._changed(snapshot, {"src/g.c": b"int g(int, int);\n"})
    assert scans == [b"int g(int, int);\n"] and landed.peek("types.toml") == b"new types\n"
    assert landed.peek("src/g.c") == b"int g(int, int);\n"
    monkeypatch.setattr(publish.types, "scan", lambda s: s.peek("types.toml"))  # an unchanged map is not rewritten
    assert publish._changed(snapshot, {"src/g.c": b"x\n"}).peek("types.toml") == b"old types\n"


@pytest.mark.parametrize("changed", [(), ("a",), ("a", "b")])
def test_land_extracts_only_versions_with_changed_split_rows(lane, monkeypatch, changed):
    snapshot = lane["snapshot"]
    old = {v.split: b"old split" for v in snapshot.versions.values()}
    for path, data in old.items():
        target = snapshot.config.project.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def apply(config, plan, original):
        for path, data in plan.writes.items():
            (config.project.root / path).write_bytes(data)
        return tuple(plan.writes)

    monkeypatch.setattr(publish.journal, "apply", Mock(side_effect=apply))
    monkeypatch.setattr(publish.layout, "capture", Mock(return_value=snapshot))
    writes = {v.split: b"new split" if v.id in changed else b"old split" for v in snapshot.versions.values()}
    plan = publish.make_plan("publish", snapshot, writes, (), (), "split promotion")
    monkeypatch.setattr(publish, "admit", Mock(return_value=(lane["unit"], lane["proposed"], lane["proofs"])))
    monkeypatch.setattr(publish, "plans", Mock(side_effect=lambda *a: iter((plan,))))
    monkeypatch.setattr(publish, "_proofs", Mock(return_value=(lane["proofs"], ())))
    extract = Mock()
    monkeypatch.setattr(publish.repo, "extract_job", extract)
    monkeypatch.setattr(publish.effort, "forget", Mock())
    publish.land(lane["config"], lane["submission"])
    publish.land(lane["config"], lane["submission"])  # a second entry on the same versions
    assert extract.call_args_list == []
    publish.settle()
    assert [call.args[0][1] for call in extract.call_args_list] == list(changed)  # each version extracted once
    assert publish.effort.forget.called == bool(changed)


def test_exact_plan_only_proves_consumers_whose_live_declarations_changed(lane, monkeypatch):
    snapshot, proposed = lane["snapshot"], lane["proposed"]
    g, h = [replace(lane["unit"], path=f"src/{n}.c", members=(n,)) for n in ("g", "h")]
    units = {u.path: u for u in (g, h)}
    snapshot = replace(
        snapshot,
        layout=replace(snapshot.layout, units=units),
        overlays={**snapshot.overlays, "include/new.h": b"old", g.path: b"g", h.path: b"h"},
    )
    proposed = replace(proposed, layout=replace(proposed.layout, units={**units, lane["unit"].path: lane["unit"]}))
    publish.headers.fold.return_value = ({lane["unit"].path: b"folded", "include/new.h": b"new"}, ())
    monkeypatch.setattr(publish.headers, "consumers", lambda *_: (g.path, h.path))
    monkeypatch.setattr(
        publish.view, "reads", lambda s, u, *args: s.peek("include/new.h") if u.path == g.path else b"unchanged"
    )
    (plan,) = publish.plans(snapshot, lane["unit"], proposed, {})
    assert set(plan.affected) == {lane["unit"].path, g.path}
    publish.headers.fold.assert_called_once()
    stages = [c.args[0] for c in publish.effort.stage.call_args_list]
    assert stages.count("publish.plans") == stages.count("publish.option") == 1


def test_a_header_left_naming_a_dropped_declaration_puts_each_includer_in_the_proof(lane, monkeypatch):
    snapshot, proposed = lane["snapshot"], lane["proposed"]
    g, h = [replace(lane["unit"], path=f"src/{n}.c", members=(n,)) for n in ("g", "h")]
    units = {u.path: u for u in (g, h)}
    snapshot = replace(
        snapshot,
        layout=replace(snapshot.layout, units=units),
        overlays={**snapshot.overlays, "include/new.h": b"old", g.path: b"g", h.path: b"h"},
    )
    proposed = replace(proposed, layout=replace(proposed.layout, units={**units, lane["unit"].path: lane["unit"]}))
    publish.headers.fold.return_value = ({lane["unit"].path: b"folded", "include/new.h": b"new"}, ())
    monkeypatch.setattr(publish.headers, "consumers", lambda _s, changed: (g.path, h.path) if changed else ())
    monkeypatch.setattr(publish.headers, "dangling", lambda *_: ("include/span.h",))
    monkeypatch.setattr(publish.view, "reads", lambda *_: b"unchanged")  # no reached declaration moved
    (plan,) = publish.plans(snapshot, lane["unit"], proposed, {})
    assert set(plan.affected) == {lane["unit"].path, g.path, h.path}


def test_landing_must_prove_the_source_after_project_declarations_are_folded(lane, monkeypatch):
    publish.headers.fold.return_value = ({lane["unit"].path: b"project declarations"}, ())
    seen = []

    def prove(snapshot, affected):
        seen.append(snapshot.read(lane["unit"].path))
        return lane["proofs"], (Finding("land.not_exact", "changed ABI", unit="f", versions=("a",)),)

    monkeypatch.setattr(publish, "_proofs", prove)
    with pytest.raises(Refusal):
        publish.land(lane["config"], lane["submission"])
    assert seen == [b"project declarations"]
    publish.journal.apply.assert_not_called()


def test_option_plan_stages_every_block_it_runs(lane):
    list(publish.plans(lane["snapshot"], lane["unit"], lane["proposed"], {}))
    stages = {c.args[0] for c in publish.effort.stage.call_args_list}
    assert {"publish.reconcile", "layout.overlay", "publish.consumers", "publish.policy"} <= stages
    assert "publish.plan" in stages


def test_proof_extracts_exactly_the_stale_versions_it_reads_before_it_proves(lane, monkeypatch):
    order = []
    monkeypatch.setattr(publish.repo, "extract_job", Mock())
    base = publish.pool.map.side_effect
    monkeypatch.setattr(
        publish.pool,
        "map",
        Mock(side_effect=lambda cfg, name, fn, items, key=None: order.append(name) or base(cfg, name, fn, items, key)),
    )
    monkeypatch.setattr(publish.compare, "holders", Mock(return_value=("a",)))
    monkeypatch.setattr(publish.ownership, "derive_many", lambda s, units: [(s, u) for u in units])
    publish._stale.update({"a": lane["config"], "z": lane["config"]})
    try:
        publish._proofs(lane["proposed"], ["src/f.c"])
        assert order == ["repo.extract", "publish.prove"]  # one extract, before the proof reads it
        assert [c.args[0][1] for c in publish.repo.extract_job.call_args_list] == ["a"] and list(publish._stale) == [
            "z"
        ]
    finally:
        publish._stale.clear()


def test_consumers_whose_reads_are_named_alike_share_one_answer(lane, monkeypatch):
    snapshot = lane["snapshot"]
    monkeypatch.setattr(publish.view, "closure", lambda s, u, v: (("h", "pin"),))
    units = replace(snapshot.layout, units={"src/f.c": lane["unit"]})
    job = (replace(snapshot, layout=units), replace(snapshot, layout=units), "src/f.c")
    assert publish._moved_key(job) == publish._moved_key(job)
    monkeypatch.setattr(publish.view, "reads", Mock(return_value=("same",)))
    assert publish._moved(job) is False
    assert publish.view.reads.call_count == 4  # both sides, two versions


class _Source:
    def __init__(self, files):
        self.files = files
        self.reads = []

    def read(self, path):
        self.reads.append(path)
        return self.files[path].encode()


def test_a_stale_version_is_extracted_for_a_proof_only_when_the_proved_source_names_a_moved_member(monkeypatch):
    monkeypatch.setattr(publish, "_stale", {"us": object(), "eu": object()})
    monkeypatch.setattr(publish, "_changed_names", {"us": {"func_A"}, "eu": {"func_B"}})
    source = _Source({"src/x.c": "void f(void) { func_A(); }", "src/y.c": "void g(void) { }"})
    x, y = Mock(path="src/x.c"), Mock(path="src/y.c")
    assert publish._read_versions(source, [(x, "us"), (x, "eu"), (y, "us"), (y, "eu")]) == {"us"}
    assert publish._read_versions(source, [(y, "us"), (y, "eu"), (x, "ne")]) == set()


def test_a_stale_version_with_no_recorded_move_is_always_extracted_and_each_source_is_read_once(monkeypatch):
    monkeypatch.setattr(publish, "_stale", {"us": object(), "eu": object()})
    monkeypatch.setattr(publish, "_changed_names", {"eu": {"func_B"}})
    source = _Source({"src/x.c": "func_B"})
    x = Mock(path="src/x.c")
    assert publish._read_versions(source, [(x, "us"), (x, "eu")]) == {"us", "eu"}
    assert source.reads == ["src/x.c"]


def test_accept_is_the_landing_check_and_writes_nothing(lane):
    plan, proofs, _ = publish.accept(
        lane["snapshot"], {"file": str(lane["source"]), "function": "f", "overrides": {}, "note": ""}, {}, False
    )
    assert proofs == lane["proofs"] and plan.operation == "publish"
    publish.journal.apply.assert_not_called()


def _proof_world(monkeypatch, tmp_path, count):
    units = {f"src/u{i}.c": UnitSpec(f"src/u{i}.c", "c", "g", (f"f{i}",), "t", {}, ()) for i in range(count)}
    snapshot = Mock(layout=Mock(units=units), config=Mock())
    derived, mapped = [], []

    def derive(s, us, keep=True):
        derived.append((len(us), keep))
        return [(s, u) for u in us]

    def run(config, stage, fn, jobs, key=None):  # a dispatch, its jobs run here
        mapped.append((stage, len(jobs)))
        return [fn(j) for j in jobs] if stage == "publish.units" else [[j[1].path] for j in jobs]

    monkeypatch.setattr(publish, "_PROOF_CHUNK", 2)
    monkeypatch.setattr(publish.store, "work", lambda config: nullcontext(tmp_path))
    monkeypatch.setattr(publish.ownership, "derive_many", derive)
    monkeypatch.setattr(publish.recipes, "resolve", lambda config, unit, options: "r")
    monkeypatch.setattr(publish.compare, "holders", lambda own, unit: ["a"])
    monkeypatch.setattr(publish.compare, "gaps", lambda own, unit, proofs: ())
    monkeypatch.setattr(publish, "settle", lambda value: value)
    monkeypatch.setattr(publish, "_read_versions", lambda snapshot, pairs: ())
    monkeypatch.setattr(publish.pool, "map", run)
    return snapshot, units, derived, mapped


def test_proofs_of_many_units_go_to_one_dispatch_whose_jobs_derive_prove_and_judge_one_unit(monkeypatch, tmp_path):
    snapshot, units, derived, mapped = _proof_world(monkeypatch, tmp_path, 5)
    proofs, gaps = publish._proofs(snapshot, sorted(units))
    # one dispatch over the units (never a barrier per chunk); each job derives its own unit, kept nowhere
    assert mapped[0] == ("publish.units", 5) and [m for m in mapped if m[0] == "publish.units"] == [mapped[0]]
    assert derived == [(1, False)] * 5 and [m for m in mapped if m[0] == "publish.prove"] == [("publish.prove", 1)] * 5
    assert proofs == tuple(sorted(units)) and gaps == ()


def test_proofs_of_a_chunk_of_units_derive_in_one_dispatch_and_prove_in_one_more(monkeypatch, tmp_path):
    snapshot, units, derived, mapped = _proof_world(monkeypatch, tmp_path, 2)
    proofs, _ = publish._proofs(snapshot, sorted(units))
    assert derived == [(2, True)] and mapped == [("publish.prove", 2)] and proofs == tuple(sorted(units))


def _refusing_derive(monkeypatch, derived, bad="f1"):
    """derive_many refusing whenever the units include the one holding `bad`, as a row write it cannot make does."""

    def derive(s, us, keep=True):
        derived.append((len(us), keep))
        if any(bad in u.members for u in us):
            raise Refusal(Finding("layout.map", "row has no editable subsegment line", unit="retained_100"))
        return [(s, u) for u in us]

    monkeypatch.setattr(publish.ownership, "derive_many", derive)


@pytest.mark.parametrize("count", [5, 2])  # one job per unit, and a chunk derived in one dispatch
def test_a_unit_whose_derive_refuses_is_its_own_gap_and_the_other_units_still_prove(monkeypatch, tmp_path, count):
    snapshot, units, derived, _ = _proof_world(monkeypatch, tmp_path, count)
    _refusing_derive(monkeypatch, derived)
    proofs, gaps = publish._proofs(snapshot, sorted(units))
    assert proofs == tuple(p for p in sorted(units) if p != "src/u1.c")
    (gap,) = gaps  # named by the unit's member, so a move keeps that one unit; the refused row stays a symptom
    assert (gap.key, gap.unit, gap.versions) == ("layout.map", "f1", ("a",))
    assert gap.missing == ("row has no editable subsegment line",) and gap.symptoms == {"refused": "retained_100"}


def test_a_header_clash_already_on_the_base_tree_never_refuses_and_a_new_one_does(monkeypatch):
    def clash(name, a, b):
        return Finding("headers.conflict", f"{name}: include/a.h:3 '{a}' vs include/u.h:9 '{b}'", path="include/a.h")

    debt, fresh, respelled = (
        clash("D_1", "s32 []", "char *[]"),
        clash("D_2", "s32", "u8"),
        clash("D_1", "u16", "s32 []"),
    )
    base = Mock(return_value=({}, [clash("D_1", "char *[]", "s32 []")]))
    monkeypatch.setattr(publish.headers, "reconcile", base)
    texts = {"include/a.h": b"base a", "include/u.h": b"base u"}
    snapshot = Mock(digest=f"base-{id(base)}", peek=texts.get, read=texts.__getitem__)
    written = {"include/a.h": b"edited", "include/new.h": b"new"}
    assert publish._new_clashes(snapshot, written, [debt, fresh]) == (fresh,)
    assert publish._new_clashes(snapshot, written, [debt, respelled]) == (respelled,)  # a new spelling of debt refuses
    assert publish._new_clashes(snapshot, written, []) == ()
    base.assert_called_once_with(snapshot, snapshot, {"include/a.h": b"base a"})  # one base judgement per tree state


def _credit_case(lane):
    """The README a plan leaves: its block is rendered from the report of the tree the writes make."""
    snapshot = lane["snapshot"]
    publish.report.progress.return_value = {"versions": {"a": {"code_matched": 2}}}
    publish.report.readme.side_effect = lambda text, snap, current: f"{text}{current['versions']['a']['code_matched']}"
    return snapshot


@pytest.mark.parametrize("path", ["src/f.c", "include/main/group.h", "layout.toml"])
def test_a_plan_that_moves_credit_carries_the_readme_block_of_the_tree_it_leaves(lane, path):
    """G main showed 463,468 B for a tree a fresh report credits 518,144: folds and lands never rewrote the README."""
    snapshot = _credit_case(lane)
    for operation in ("publish", "move", "repair"):
        plan = publish.make_plan(operation, snapshot, {path: b"new"}, (), (), operation)
        assert plan.writes == {path: b"new", "README.md": b"old readme\n2"}
        assert plan.digest == digest(
            (plan.operation, plan.base, plan.writes, plan.affected, plan.blocking, plan.debt, plan.message)
        )
    (after,) = {call.args[0].overlays[path] for call in publish.report.progress.call_args_list}
    assert after == b"new"  # the report of the tree after the writes, not the one before


@pytest.mark.parametrize(
    "writes, readme",
    [
        ({"types.toml": b"new"}, b"old readme\n"),  # nothing that credits
        ({"src/f.c": b"new", "README.md": b"own"}, b"old readme\n"),  # the plan renders the README itself
        ({"src/f.c": b"new"}, None),  # no README
    ],
)
def test_a_plan_leaves_the_readme_alone_when_no_credit_moves_or_it_renders_it(lane, writes, readme):
    snapshot = _credit_case(lane)
    snapshot.overlays["README.md"] = readme
    plan = publish.make_plan("publish", snapshot, writes, (), (), "x")
    assert plan.writes == writes
    publish.report.progress.assert_not_called()


def test_a_plan_that_moves_credit_carries_the_presentation_files_that_changed(lane):
    """The site reads progress.json, the maps and the symbol lines live: a landing commits them with its change."""
    snapshot = _credit_case(lane)
    snapshot.overlays["progress.json"] = b"same\n"
    publish.present.files.return_value = {"progress.json": b"same\n", "docs/memory-map-a.svg": b"<svg/>"}
    plan = publish.make_plan("publish", snapshot, {"src/f.c": b"new"}, (), (), "x")
    assert plan.writes == {"src/f.c": b"new", "README.md": b"old readme\n2", "docs/memory-map-a.svg": b"<svg/>"}
    after, current = publish.present.files.call_args.args
    assert after.overlays["src/f.c"] == b"new" and current == publish.report.progress.return_value  # one report


def test_an_unchanged_block_writes_no_readme(lane):
    snapshot = lane["snapshot"]  # the default readme mock renders the text it is given
    assert publish.make_plan("move", snapshot, {"src/f.c": b"new"}, (), (), "x").writes == {"src/f.c": b"new"}
    publish.report.progress.assert_called_once()
