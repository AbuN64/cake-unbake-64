"""Batch publication preserves complete proofs and commits no partially proved change set."""

import json
from contextlib import nullcontext
from unittest.mock import Mock

import fixture
import pytest
from test_layout import scene as scene

from unbake import batch_data, config, layout
from unbake.contracts import Finding, Recipe, Refusal


@pytest.mark.parametrize(
    "value", [[], {}, [{"member": "m"}], [{"member": "m", "file": "f"}] * 2, [{"member": 1, "file": "f"}]]
)
def test_manifest_rejects_invalid_and_duplicate_members(tmp_path, value):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    with pytest.raises(Refusal, match="Invalid data manifest"):
        batch_data._manifest(path)


def test_manifest_paths_are_relative_to_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('[{"member":"m","file":"candidates/m.c"}]')
    assert batch_data._manifest(path)[0]["file"] == str(tmp_path / "candidates/m.c")


def batch(scene, tmp_path, long=False):
    names = ["data/unresolved/80000400", "data/unresolved/80000404" + ("_unclaimed_ABC" * 40 if long else "")]
    snapshot = scene(tuple((n, "data", (i + 1,)) for i, n in enumerate(names)), grouped=False)
    rows = []
    for i, name in enumerate(names):
        source = tmp_path / f"candidate{i}.c"
        source.write_text(f"unsigned int datum{i} = {i + 1};\n")
        rows.append({"member": name, "file": str(source)})
    return snapshot, rows


def test_prepare_combines_all_units_and_splits_without_disk_writes(scene, tmp_path):
    snapshot, rows = batch(scene, tmp_path, long=True)
    old = {v.split: snapshot.read(v.split) for v in snapshot.versions.values()}
    proposed, units = batch_data._prepare(snapshot, rows)
    assert len(units) == 2 and all(len(u.path.rsplit("/", 1)[1].encode()) < 255 for u in units)
    for version in snapshot.versions.values():
        assert (tmp_path / version.split).read_bytes() == old[version.split]
        assert proposed.read(version.split).count(b", .data, ") == 2
    assert all(layout.unit_of(proposed, r["member"]) for r in rows)
    assert all(not (tmp_path / u.path).exists() for u in units)


@pytest.mark.parametrize("name", ["missing", "func_80000400"])
def test_prepare_rejects_non_data_without_writes(scene, tmp_path, name):
    snapshot = scene()
    source = tmp_path / "source.c"
    source.write_text("unsigned int x = 1;")
    with pytest.raises(Refusal, match="unpublished data"):
        batch_data._prepare(snapshot, [{"member": name, "file": str(source)}])


@pytest.mark.parametrize("bad,withhold", [(False, False), (True, False), (True, True)])
def test_batch_proves_all_members_and_versions_before_atomic_commit(scene, tmp_path, monkeypatch, bad, withhold):
    snapshot, rows = batch(scene, tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows))
    monkeypatch.setattr(batch_data.layout, "capture", lambda cfg: snapshot)
    locks = []
    monkeypatch.setattr(batch_data.store, "exclusive", lambda c, n, **k: locks.append(("ex", n)) or nullcontext(True))
    monkeypatch.setattr(batch_data.store, "shared", lambda c, n: locks.append(("sh", n)) or nullcontext())
    monkeypatch.setattr(batch_data.store, "work", lambda cfg: nullcontext(tmp_path / "work"))
    monkeypatch.setattr(batch_data.recipes, "resolve", lambda *a: Recipe("gcc-test", (), (), (), "recipe"))
    seen = []

    def dispatch(cfg, stage, function, jobs):
        seen.extend((u.members[0], v) for _, u, v, *_ in jobs)
        return [
            (
                fixture.proof(
                    u.path,
                    u.members[0],
                    v,
                    not (bad and v == "b"),
                    missing=("bytes differ",) if bad and v == "b" else (),
                ),
            )
            for _, u, v, *_ in jobs
        ]

    monkeypatch.setattr(batch_data.pool, "map", dispatch)
    monkeypatch.setattr(batch_data.policy, "evaluate", lambda *a: ())
    monkeypatch.setattr(batch_data.types, "scan", lambda s: b"types")
    monkeypatch.setattr(batch_data.repo, "files", lambda s: {})
    commit = Mock(return_value="new-head")
    monkeypatch.setattr(batch_data.journal, "apply", commit)
    logs = Mock()
    monkeypatch.setattr(batch_data.store, "log", logs)
    params = {"files": [], "function": None, "withhold": withhold, "batch_data": manifest, "note": None}
    if bad and not withhold:
        with pytest.raises(Refusal, match=r"land\.not_exact"):
            batch_data.run(snapshot.config, params)
        commit.assert_not_called()
        logs.assert_not_called()
    else:
        result = batch_data.run(snapshot.config, params)
        config.validate("result.submit", result, "batch result")
        assert commit.call_count == 1
        assert len(result["drain"]["landed"]) == 2
        committed = commit.call_args.args[1]
        proposed = layout.overlay(snapshot, committed.writes)
        assert all(tuple(u.withheld) == (("b",) if bad else ()) for u in proposed.layout.units.values())
        assert all(p["exact"] for call in logs.call_args_list for p in call.args[2]["proofs"])
    assert set(seen) == {(r["member"], v) for r in rows for v in ["a", "b"]}
    assert locks == [("ex", "land"), ("sh", "structure")]  # the drain's order, each once


def test_active_lander_refuses_before_capture(tmp_path, monkeypatch):
    path = tmp_path / "batch.json"
    path.write_text('[{"member":"m","file":"m.c"}]')
    monkeypatch.setattr(batch_data.store, "exclusive", lambda *a, **k: nullcontext(False))
    monkeypatch.setattr(batch_data.store, "shared", lambda *a: pytest.fail("structure is taken only with land held"))
    capture = Mock()
    monkeypatch.setattr(batch_data.layout, "capture", capture)
    with pytest.raises(Refusal, match="landing is running"):
        batch_data.run(None, {"files": [], "function": None, "withhold": False, "batch_data": path})
    capture.assert_not_called()


def test_compiled_policy_finding_is_not_silenced_by_proposed_source(scene, tmp_path, monkeypatch):
    snapshot, rows = batch(scene, tmp_path)
    proposed, units = batch_data._prepare(snapshot, rows)
    finding = Finding("source.volatile-storage", "new rule violation", path=units[0].path)
    monkeypatch.setattr(batch_data.view, "get", lambda *a: None)
    monkeypatch.setattr(batch_data.policy, "evaluate", lambda *a: (finding,))
    native = Mock()
    monkeypatch.setattr(batch_data.native, "prove_args", native)
    with pytest.raises(Refusal, match="new rule violation"):
        batch_data._prove((proposed, units[0], "a", None, tmp_path, snapshot))
    native.assert_not_called()
