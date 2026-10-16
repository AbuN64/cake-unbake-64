"""Land intake, ordering, draining and check orchestration without native work."""

import fcntl
import inspect
import json
import os
import re
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import fixture
import pytest

from unbake import config as configuration
from unbake import land
from unbake.contracts import (
    Finding,
    Group,
    LayoutMap,
    Member,
    Placement,
    Receipt,
    Refusal,
    Snapshot,
    UnitSpec,
    digest,
    sha256,
)


@pytest.fixture
def lane(tmp_path, monkeypatch):
    # Do not use cfg/project: those fixtures create ROMs and invoke git.
    cfg = fixture.bare_config(tmp_path, name="test", title="Test")
    unit = UnitSpec("src/f.c", "c", "group", ("f",), "gcc-test", {})
    placements = tuple(Placement(v, ".text", 4096, 4104, 0x80000400) for v in ("a", "b"))
    member = Member("f", "function", "asm", "group", placements)
    mapping = LayoutMap({}, {"f": member}, {}, "layout", (), {})
    snapshot = Snapshot(cfg, "a" * 40, mapping, {}, {}, "snapshot")
    capture = Mock(return_value=snapshot)
    logs = Mock()
    stages = []

    def stage(name):
        stages.append(name)
        return nullcontext()

    monkeypatch.setattr(land.layout, "capture", capture)
    monkeypatch.setattr(land.effort, "stage", stage)
    monkeypatch.setattr(land.effort, "invocation", lambda: "test-invocation")
    monkeypatch.setattr(land.store, "log", logs)
    publisher = Mock(
        side_effect=lambda cfg, entry: Receipt(
            entry.operation, ("f.c",), "publish f", "plan", entry.proofs, 0, "test-invocation"
        )
    )
    monkeypatch.setattr(land.publish, "land", publisher)
    monkeypatch.setattr(land.publish, "generate", Mock(return_value=()))
    accept = Mock(return_value=None)
    monkeypatch.setattr(land.publish, "accept", accept)
    return cfg, unit, snapshot, capture, logs, None, publisher, stages


def _request(function=None, note="technique"):
    return {"file": "candidate.c", "function": function, "overrides": {"add": [], "omit": []}, "note": note}


def _proofs(unit, exact=True, score=None, symptoms=None):
    return tuple(
        fixture.proof(unit.path, unit.members[0], v, exact, () if exact else ("bytes differ",), score, symptoms)
        for v in ("a", "b")
    )


def _submit(lane, source=b"void f(void) {}\n", **kwargs):
    cfg, unit = lane[:2]
    return land.submit(cfg, _request(), unit, _proofs(unit), source, "submit", **kwargs)


def _paths(cfg, entry, directory=""):
    root = cfg.project.root / ".unbake/inbox" / directory
    return root / f"{entry.id}.json", root / f"{entry.id}.c"


def _order(cfg, entries, times=None):
    for index, entry in enumerate(entries):
        at = times[index] if times else index + 1
        os.utime(_paths(cfg, entry)[0], ns=(at, at))


def test_submit_exact_writes_inbox_pair(lane, monkeypatch):
    cfg, unit, snapshot = lane[:3]
    source = b"void f(void) {}\n"
    replacements = []
    real_replace = os.replace

    def track(before, after):
        replacements.append((before, after))
        real_replace(before, after)

    monkeypatch.setattr(land.store.Path, "replace", lambda self, target: track(self, target))
    entry = _submit(lane, source)
    expected_id = digest(("f", sha256(source), _request()["overrides"], "publish"))
    assert entry.id == expected_id
    assert entry.member == "f" and entry.function is None
    assert entry.base == snapshot.commit
    assert entry.proofs == _proofs(unit)
    assert entry.origin == "submit" and entry.note == "technique"
    assert entry.invocation == "test-invocation"
    record, copied = _paths(cfg, entry)
    assert copied.read_bytes() == source
    assert json.loads(record.read_text()) == json.loads(json.dumps(asdict(entry)))
    configuration.validate("submission", json.loads(record.read_text()), str(record))
    assert [after for _, after in replacements] == [copied, record]  # each written atomically, by name
    assert not list(record.parent.glob("*.part"))
    assert "land.submit" in lane[-1]


@pytest.mark.parametrize("directory", ["", "done"])
def test_submit_idempotent_same_id(lane, directory):
    cfg = lane[0]
    first = _submit(lane)
    paths = _paths(cfg, first)
    if directory:
        destination = paths[0].parent / directory
        destination.mkdir()
        for path in paths:
            os.replace(path, destination / path.name)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in _paths(cfg, first, directory)}
    second = land.submit(cfg, _request(note="changed"), lane[1], _proofs(lane[1]), b"void f(void) {}\n", "compare")
    assert second == first
    assert isinstance(second.proofs, tuple)
    assert all(isinstance(p.missing, tuple) for p in second.proofs)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in _paths(cfg, first, directory)}
    if directory:
        assert not _paths(cfg, first)[0].exists()


def test_submit_of_a_refused_source_is_queued_again(lane):
    cfg = lane[0]
    first = _submit(lane)
    destination = _paths(cfg, first)[0].parent / "refused"
    destination.mkdir()
    for path in _paths(cfg, first):
        os.replace(path, destination / path.name)
    second = land.submit(cfg, _request(), lane[1], _proofs(lane[1]), b"void f(void) {}\n", "compare")
    assert second.id == first.id
    assert all(path.is_file() for path in _paths(cfg, first))


@pytest.mark.parametrize("kind,same,expected_none", [("c", True, True), ("c", False, False), ("asm", True, False)])
def test_submit_already_landed_returns_none(lane, kind, same, expected_none):
    cfg, unit, snapshot, capture = lane[:4]
    current = replace(unit, kind=kind)
    capture.return_value = replace(snapshot, layout=replace(snapshot.layout, units={unit.path: current}))
    path = cfg.project.root / unit.path
    path.parent.mkdir()
    path.write_bytes(b"void f(void) {}\n" if same else b"old source")
    result = _submit(lane)
    assert (result is None) == expected_none
    if expected_none:
        assert not (cfg.project.root / ".unbake/inbox").exists()


@pytest.mark.parametrize(
    "score,symptoms,accepted",
    [
        (0.49, {"bytes_differ": True}, False),
        (0.5, {"bytes_differ": True}, True),
        (0.7, {"compile_failed": False}, False),
        (0.7, {"unresolved": []}, False),
        (0.7, {"no_measurement": False}, False),
    ],
)
def test_submit_fuzzy_needs_gain_and_compile(lane, score, symptoms, accepted):
    cfg, unit, snapshot, capture = lane[:4]
    gain = configuration.load_resource("flow.toml")["fuzzy"]["min_gain"]
    fuzzy = {"f": {"path": "src/fuzzy/f.c", "scores": {"a": 0.5 - gain, "b": 0.8}}}
    capture.return_value = replace(snapshot, layout=replace(snapshot.layout, fuzzy=fuzzy))
    proofs = (_proofs(unit, False, 0.9)[0], _proofs(unit, False, score, symptoms)[1])
    result = land.submit(cfg, _request(), unit, proofs, b"candidate", "compare")
    assert (result is not None) == accepted
    if accepted:
        assert result.operation == "fuzzy"
        assert result.id == digest(("f", sha256(b"candidate"), _request()["overrides"], "fuzzy"))


def test_submit_fuzzy_without_previous_or_proofs(lane):
    cfg, unit = lane[:2]
    assert land.submit(cfg, _request(), unit, (), b"candidate", "compare") is None
    assert land.submit(cfg, _request(), unit, _proofs(unit, False, 0.5), b"candidate", "compare").operation == "fuzzy"


@pytest.mark.parametrize("corrupt", ["{bad json", '{"id": 3}'])
def test_inbox_order_and_corrupt_entry_moved(lane, corrupt):
    cfg = lane[0]
    entries = [_submit(lane, source) for source in (b"one", b"two", b"three")]
    _order(cfg, entries, [20, 10, 10])
    root = _paths(cfg, entries[0])[0].parent
    (root / "broken.json").write_text(corrupt)
    (root / "broken.c").write_bytes(b"broken")
    expected = sorted(entries, key=lambda entry: (_paths(cfg, entry)[0].stat().st_mtime_ns, entry.id))
    assert land.inbox(cfg) == expected
    assert not (root / "broken.json").exists()
    assert (root / "refused/broken.json").read_text() == corrupt
    assert (root / "refused/broken.c").read_bytes() == b"broken"
    body = lane[4].call_args.args[2]
    assert lane[4].call_args.args[1] == "refusal"
    assert body["id"] == "broken"
    assert body["findings"][0]["key"] == "inbox.corrupt"
    assert body["findings"][0]["path"] == ".unbake/inbox/broken.json"


def test_drain_lands_in_order_and_moves_files(lane):
    cfg = lane[0]
    entries = [_submit(lane, source) for source in (b"one", b"two")]
    _order(cfg, entries)
    result = land.drain(cfg)
    assert result == {
        "running": False,
        "refused": [],
        "generated": [],
        "landed": [
            {"id": e.id, "member": "f", "operation": "publish", "files": ["f.c"], "message": "publish f"}
            for e in entries
        ],
    }
    assert [call.args[1] for call in lane[6].call_args_list] == entries
    for entry in entries:
        for path in _paths(cfg, entry):
            assert not path.exists()
        assert all(path.is_file() for path in _paths(cfg, entry, "done"))
    assert [call.args[1] for call in lane[4].call_args_list] == ["receipt", "receipt"]
    assert lane[4].call_args_list[0].args[2] == {
        "id": entries[0].id,
        "member": "f",
        **asdict(Receipt("publish", ("f.c",), "publish f", "plan", entries[0].proofs, 0, "test-invocation")),
    }
    assert "land.drain" in lane[-1] and lane[-1].count("land.entry") == 2


def test_a_drain_of_many_entries_generates_once_and_a_breaking_entry_is_refused_by_name(lane):
    cfg = lane[0]
    entries = [_submit(lane, source) for source in (b"one", b"two", b"three")]
    _order(cfg, entries)
    ok = Receipt("publish", ("f.c",), "publish f", "plan", (), 0, "test")
    lane[6].side_effect = [ok, Refusal(Finding("land.not_exact", "breaks")), ok]
    result = land.drain(cfg)
    assert [r["id"] for r in result["landed"]] == [entries[0].id, entries[2].id]
    assert [r["id"] for r in result["refused"]] == [entries[1].id]
    land.publish.generate.assert_called_once_with(cfg)


def test_a_drain_that_landed_a_publish_returns_and_logs_the_generated_files(lane):
    cfg = lane[0]
    _submit(lane)
    land.publish.generate.return_value = ("Makefile", "types.toml")
    result = land.drain(cfg)
    assert result["generated"] == ["Makefile", "types.toml"]
    row = [c.args for c in lane[4].call_args_list if c.args[2].get("id") == "generate"]
    assert row and row[0][1] == "receipt" and row[0][2]["files"] == ["Makefile", "types.toml"]
    assert row[0][2]["operation"] == "generate"


def test_a_drain_that_landed_nothing_writes_nothing(lane):
    """`unbake land` with landed: [] rewrote Makefile, layout, types and reports into a junk commit."""
    cfg = lane[0]
    entry = _submit(lane)
    lane[6].side_effect = Refusal(Finding("land.not_exact", "breaks"))
    assert land.drain(cfg)["generated"] == []
    land.publish.generate.assert_not_called()
    assert all(p.is_file() for p in _paths(cfg, entry, "refused"))
    land.drain(cfg)  # an empty inbox: nothing
    land.publish.generate.assert_not_called()


def test_a_landing_whose_generate_never_ran_is_generated_by_the_next_drain(lane):
    cfg = lane[0]
    land.store.write(cfg.project.root / land._PENDING, b"id")  # a drain stopped after landing, before generating
    land.drain(cfg)
    land.publish.generate.assert_called_once_with(cfg)
    assert not (cfg.project.root / land._PENDING).exists()


def test_drain_refusal_recorded_and_continues(lane):
    cfg = lane[0]
    first, second = [_submit(lane, source) for source in (b"one", b"two")]
    _order(cfg, [first, second])
    findings = (Finding("land.not_exact", "mismatch", missing=("bytes",)), Finding("land.request", "invalid request"))
    lane[6].side_effect = [Refusal(*findings), Receipt("publish", ("f.c",), "publish f", "plan", (), 0, "test")]
    result = land.drain(cfg)
    refused = {"id": first.id, "member": "f", "findings": [asdict(f) for f in findings]}
    assert result["refused"] == [refused]
    assert result["landed"][0]["id"] == second.id
    assert not result["running"]
    assert all(p.is_file() for p in _paths(cfg, first, "refused"))
    assert all(p.is_file() for p in _paths(cfg, second, "done"))
    assert lane[4].call_args_list[0].args == (cfg, "refusal", refused)
    kept = json.loads(_paths(cfg, first, "refused")[0].read_text())
    assert [f["key"] for f in kept["refusal"]] == ["land.not_exact", "land.request"]
    assert land._read(_paths(cfg, first, "refused")[0]).id == first.id  # the schema accepts the refusal key


def test_drain_moves_a_crashed_entry_to_refused_naming_the_exception(lane):
    cfg = lane[0]
    first, second = [_submit(lane, source) for source in (b"one", b"two")]
    _order(cfg, [first, second])
    lane[6].side_effect = [KeyError("boom"), Receipt("publish", ("f.c",), "publish f", "plan", (), 0, "test")]
    result = land.drain(cfg)
    (finding,) = result["refused"][0]["findings"]
    assert finding["key"] == "internal.error" and "KeyError" in finding["reason"] and "boom" in finding["reason"]
    assert all(p.is_file() for p in _paths(cfg, first, "refused"))
    assert result["landed"][0]["id"] == second.id and not result["running"]


def test_drain_running_returns_immediately(lane):
    cfg = lane[0]
    entry = _submit(lane)
    # Independent open file descriptions contend under real flock in one process.
    # A child process is forbidden for this module by the phase-2 lane rules.
    with land.store.exclusive(cfg, "land", wait=False) as held:
        assert held
        assert land.drain(cfg) == {"running": True, "landed": [], "refused": [], "generated": []}
    lane[6].assert_not_called()
    assert all(p.exists() for p in _paths(cfg, entry))


def test_drain_takes_the_structure_lock_shared_once_per_entry_and_generates_once(lane, monkeypatch):
    cfg = lane[0]
    taken = []

    @contextmanager
    def shared(cfg, name):
        taken.append(name)
        yield

    monkeypatch.setattr(land.store, "shared", shared)
    for source in (b"a", b"b", b"c"):
        _submit(lane, source)
    result = land.drain(cfg)
    assert len(result["landed"]) == 3
    assert taken == ["structure"] * 4  # three entries, then the one generate of the drain
    assert land.publish.generate.call_count == 1


def test_an_entry_waits_for_a_move_holding_the_structure_lock_then_lands(lane, monkeypatch):
    cfg = lane[0]
    entry = _submit(lane)
    real_flock = fcntl.flock
    lock = cfg.project.root / ".unbake" / "structure.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    holder = lock.open("a+b")
    real_flock(holder.fileno(), fcntl.LOCK_EX)
    waits = []

    def flock(fd, operation):
        if not holder.closed and operation & fcntl.LOCK_SH:
            if operation & fcntl.LOCK_NB:
                raise BlockingIOError  # the move still holds it
            waits.append(entry.id)  # the entry blocks here until the move is done
            real_flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()
        return real_flock(fd, operation)

    monkeypatch.setattr(land.store.fcntl, "flock", flock)
    result = land.drain(cfg)
    assert waits == [entry.id] and result["refused"] == []
    assert [row["id"] for row in result["landed"]] == [entry.id]


def test_drain_picks_up_entry_added_after_listing(lane):
    cfg = lane[0]
    first = _submit(lane, b"first")
    added = []

    def publish(cfg, entry):
        if entry.id == first.id:
            added.append(_submit(lane, b"added"))
        return Receipt("publish", ("f.c",), "publish f", "plan", (), 0, "test")

    lane[6].side_effect = publish
    result = land.drain(cfg)
    assert [row["id"] for row in result["landed"]] == [first.id, added[0].id]
    assert land.inbox(cfg) == []
    assert all(p.exists() for p in _paths(cfg, added[0], "done"))


def test_drain_rescans_after_lock_release(lane, monkeypatch):
    cfg = lane[0]
    first = _submit(lane, b"first")
    added = []
    real_exclusive = land.store.exclusive
    from contextlib import contextmanager

    @contextmanager
    def exclusive(cfg, name, *, wait):
        with real_exclusive(cfg, name, wait=wait) as held:
            yield held
        if not added:
            added.append(_submit(lane, b"after unlock"))

    monkeypatch.setattr(land.store, "exclusive", exclusive)
    result = land.drain(cfg)
    assert [row["id"] for row in result["landed"]] == [first.id, added[0].id]
    assert lane[6].call_count == 2


def test_drain_measures_again_when_head_moved_under_it(lane):
    cfg = lane[0]
    entry = _submit(lane)
    moved = Refusal(Finding("journal.changed", "HEAD moved"))
    lane[6].side_effect = [moved, Receipt("publish", ("f.c",), "publish f", "plan", (), 0, "test")]
    result = land.drain(cfg)
    assert [row["id"] for row in result["landed"]] == [entry.id] and result["refused"] == []
    assert lane[6].call_count == 2


def test_drain_refuses_after_the_attempts_when_head_keeps_moving(lane):
    cfg = lane[0]
    _submit(lane)
    lane[6].side_effect = Refusal(Finding("journal.changed", "HEAD moved"))
    assert len(land.drain(cfg)["refused"]) == 1
    assert lane[6].call_count == land._ATTEMPTS


def _flags(**overrides):
    return {"strict": False, "repair": False, **overrides}


def test_check_result_validates(lane, monkeypatch):
    monkeypatch.setattr(land.policy, "census", lambda snap, also=(), only=None: ((), [[]]))
    result = land.check_command(lane[0], _flags())
    assert set(result) == {"commit", "counts", "withholding"}
    configuration.validate("result.check", json.loads(json.dumps(result)), "result.check")


def test_check_rows_locate_each_finding_and_put_the_withholding_ones_first(lane):
    snapshot = SimpleNamespace(layout=SimpleNamespace(units={"src/a.c": None}))
    found = [
        Finding("source.union-view", "u", path="include/a.h", line=3),
        Finding("source.fake-type", "f", path="src/a.c", line=68),
        Finding("source.raw-gfx", "g", path="src/a.c", line=2),
    ]
    held, rows = land._located(snapshot, found)
    assert held == [found[1]]  # only an unclean rule in the unit's own text withholds credit
    assert [(r["at"], r["withholding"]) for r in rows] == [
        ("src/a.c:68", True),
        ("include/a.h:3", False),
        ("src/a.c:2", False),
    ]


def test_land_result_validates_and_reports_refusals(lane):
    _submit(lane)
    lane[6].side_effect = Refusal(Finding("land.request", "refused"))
    result = land.land_command(lane[0], {})
    assert len(result["refused"]) == 1
    configuration.validate("result.land", json.loads(json.dumps(result)), "result.land")


def test_check_strict_reports_debt_after_census_written(lane, monkeypatch):
    finding = Finding("check.debt", "debt", blocking=False)
    monkeypatch.setattr(land.policy, "census", lambda snap, also=(), only=None: ((finding,), [[]]))
    with pytest.raises(Refusal) as error:
        land.check_command(lane[0], _flags(strict=True))
    assert error.value.findings[0].key == "check.debt"
    assert error.value.findings[0].missing == ("check.debt",)
    assert (lane[0].project.root / ".unbake/check.json").is_file()


def test_drain_measures_again_when_another_lane_moved_head(lane):
    cfg = lane[0]
    entry = _submit(lane)
    moved = Refusal(Finding("journal.changed", "HEAD moved"))
    lane[6].side_effect = [moved, moved, Receipt("publish", ("f.c",), "publish f", "plan", (), 0, "test")]
    result = land.drain(cfg)
    assert [r["id"] for r in result["landed"]] == [entry.id] and result["refused"] == []
    assert lane[6].call_count == land._ATTEMPTS == 3


def test_drain_refuses_after_the_attempts_run_out_and_never_retries_other_refusals(lane):
    cfg = lane[0]
    first, second = [_submit(lane, source) for source in (b"one", b"two")]
    _order(cfg, [first, second])
    moved = Refusal(Finding("journal.changed", "HEAD moved"))
    lane[6].side_effect = [moved] * land._ATTEMPTS + [Refusal(Finding("land.request", "bad"))]
    result = land.drain(cfg)
    assert [r["findings"][0]["key"] for r in result["refused"]] == ["journal.changed", "land.request"]
    assert lane[6].call_count == land._ATTEMPTS + 1


def test_change_set_is_one_entry_with_its_files_and_moves_as_one(lane):
    cfg = lane[0]
    entry = _submit(lane, extras={"include/x.h": b"extern int g;\n", "src/g.c": b"int g;\n"})
    assert sorted(entry.extras) == ["include/x.h", "src/g.c"]
    assert {(cfg.project.root / copy).read_bytes() for copy in entry.extras.values()} == {
        b"extern int g;\n",
        b"int g;\n",
    }
    assert _submit(lane).id != entry.id
    assert [r["id"] for r in land.drain(cfg)["landed"]].count(entry.id) == 1
    done = cfg.project.root / ".unbake/inbox/done"
    assert all(
        (cfg.project.root / copy).exists() is False and (done / Path(copy).name).exists()
        for copy in entry.extras.values()
    )


def _one_exact(unit):
    return (fixture.proof(unit.path, "f", "a", True), fixture.proof(unit.path, "f", "b", False, ("bytes differ",), 0.9))


def test_submit_with_withhold_lands_the_exact_versions_and_withholds_the_rest(lane):
    cfg, unit = lane[:2]
    entry = land.submit(cfg, _request(), unit, _one_exact(unit), b"candidate", "submit", withhold=True)
    assert entry.operation == "publish" and entry.withheld == ("b",)
    assert land.submit(cfg, _request(), unit, _one_exact(unit), b"candidate", "submit", withhold=True).id == entry.id
    assert land._read(cfg.project.root / f".unbake/inbox/{entry.id}.json").withheld == ("b",)


def test_submit_withhold_keeps_the_withheld_versions_of_a_change_set(lane):
    cfg, unit = lane[:2]
    extras = {"include/x.h": b"extern int g;\n"}
    entry = land.submit(cfg, _request(), unit, _one_exact(unit), b"candidate", "submit", extras, withhold=True)
    assert entry.operation == "publish" and entry.withheld == ("b",) and list(entry.extras) == ["include/x.h"]


def test_submit_without_withhold_keeps_the_fuzzy_rule(lane):
    cfg, unit = lane[:2]
    entry = land.submit(cfg, _request(), unit, _one_exact(unit), b"candidate", "submit")
    assert entry.operation == "fuzzy" and entry.withheld == ()


def test_submit_withhold_with_no_exact_version_is_the_plain_not_exact_result(lane):
    cfg, unit = lane[:2]
    plain = land.submit(cfg, _request(), unit, _proofs(unit, False, 0.9), b"candidate", "submit")
    held = land.submit(cfg, _request(), unit, _proofs(unit, False, 0.9), b"candidate", "submit", withhold=True)
    assert held == plain and (held is None or held.withheld == ())


def test_only_the_last_attempt_holds_the_journal_so_concurrent_landings_cannot_invalidate_it(lane, monkeypatch):
    cfg = lane[0]
    entry = _submit(lane)
    held, seen = [], []

    @contextmanager
    def hold(config):
        held.append("enter")
        yield
        held.append("exit")

    monkeypatch.setattr(land.journal, "held", hold)
    moved = Refusal(Finding("journal.changed", "HEAD moved"))

    def publish(config, submission):
        seen.append(bool(held) and held[-1] == "enter")  # whether the journal is held while this attempt plans
        if len(seen) < land._ATTEMPTS:
            raise moved
        return Receipt("publish", ("f.c",), "publish f", "plan", (), 0, "test")

    lane[6].side_effect = publish
    result = land.drain(cfg)
    assert [row["id"] for row in result["landed"]] == [entry.id]
    assert seen == [False, False, True] and held == ["enter", "exit"]


def test_a_drain_for_one_submission_leaves_the_other_lanes_entries_in_the_inbox(lane):
    cfg = lane[0]
    first, second = [_submit(lane, source) for source in (b"one", b"two")]
    _order(cfg, [first, second])
    result = land.drain(cfg, second.id)
    assert [row["id"] for row in result["landed"]] == [second.id] and result["refused"] == []
    assert [call.args[1] for call in lane[6].call_args_list] == [second]
    assert all(path.is_file() for path in _paths(cfg, first)) and not any(p.exists() for p in _paths(cfg, second))
    assert [row["id"] for row in land.drain(cfg)["landed"]] == [first.id]  # the full drain still lands the rest


def test_drain_lands_the_fuzzy_entries_together_and_a_refusal_blocks_no_other(lane, monkeypatch):
    cfg, unit = lane[:2]
    entries = [
        land.submit(cfg, _request(), unit, _proofs(unit, False, 0.5 + i / 10), b"v%d" % i, "submit") for i in range(3)
    ]
    _order(cfg, entries)
    assert {e.operation for e in entries} == {"fuzzy"}
    seen = []

    def land_fuzzy(config, batch):
        seen.append([e.id for e in batch])
        return (
            {e.id: Receipt("fuzzy", ("src/fuzzy/f.c",), "fuzzy 3", "plan", (), 0, "i") for e in batch[:2]},
            {batch[2].id: (Finding("fuzzy.no_gain", "none", unit="f"),)},
        )

    monkeypatch.setattr(land.publish, "land_fuzzy", land_fuzzy)
    result = land.drain(cfg)
    assert seen == [[e.id for e in entries]] and lane[6].call_count == 0  # one plan, not one landing each
    assert [row["id"] for row in result["landed"]] == [entries[0].id, entries[1].id]
    assert [row["id"] for row in result["refused"]] == [entries[2].id]
    assert all(path.is_file() for path in _paths(cfg, entries[2], "refused"))


MACRO = b"#define BUMP(x) \\\n    do { (x)++; \\\n    } while (0)\nextern int g;\n"


@pytest.fixture
def repair(lane, monkeypatch):
    """A repair over one header with a multi-line macro whose consumer is src/f.c; the preprocessor is a stub that
    refuses a header whose macro lost its continuation lines."""
    cfg, unit, snapshot = lane[:3]
    (cfg.project.root / "include").mkdir()
    (cfg.project.root / "include/abi.h").write_bytes(MACRO)
    (cfg.project.root / "config.toml").write_text("schema = 3\n")
    monkeypatch.setattr(land.policy, "_derived_rows", lambda snap: [])
    snapshot = replace(snapshot, layout=replace(snapshot.layout, units={unit.path: unit}))
    lane[3].return_value = snapshot
    applied = Mock()

    def preprocess(snap, spec, version, recipe, lines=True):
        if b"while" not in snap.read("include/abi.h"):
            return (Finding("preprocess.error", "abi.h:2: unterminated #define", path="include/abi.h"),)
        return ()

    monkeypatch.setattr(land.headers, "sources", lambda snap: ["include/abi.h"])
    monkeypatch.setattr(land.headers, "consumers", lambda snap, changed: ("src/f.c",))
    monkeypatch.setattr(
        land.layout, "overlay", lambda snap, writes: replace(snap, overlays={**snap.overlays, **writes})
    )
    monkeypatch.setattr(land.native, "compiles", lambda item: preprocess(*item, None))
    monkeypatch.setattr(land.pool, "map", lambda cfg, name, function, items, key=None: [function(i) for i in items])
    monkeypatch.setattr(land.journal, "apply", applied)
    monkeypatch.setattr(land, "_live_words", lambda snap: set(re.findall(r"\w+", MACRO.decode())))
    monkeypatch.setattr(land, "_prototypes", lambda snap: ({}, []))
    monkeypatch.setattr(land.policy, "census", lambda snap, also=(), only=None: ((), [[]]))
    return cfg, applied


def test_repair_writes_nothing_and_refuses_when_a_changed_header_no_longer_preprocesses(repair, monkeypatch):
    cfg, applied = repair
    mangled = MACRO.replace(b"    } while (0)\n", b"")  # the continuation lines of the macro were cut away
    monkeypatch.setattr(land.headers, "reconcile", lambda snap, over, texts: ({"include/abi.h": mangled}, []))
    with pytest.raises(Refusal) as refused:
        land.check_command(cfg, _flags(repair=True))
    assert [(f.key, f.path) for f in refused.value.findings] == [("preprocess.error", "include/abi.h")]
    applied.assert_not_called()
    assert not (cfg.project.root / ".unbake/check.json").exists()  # nothing was written, not even the census


def test_repair_applies_the_headers_that_all_still_preprocess(repair, monkeypatch):
    cfg, applied = repair
    folded = MACRO.replace(b"extern int g;\n", b"")  # the repeated declaration folded away, the macro whole
    monkeypatch.setattr(land.headers, "reconcile", lambda snap, over, texts: ({"include/abi.h": folded}, []))
    assert land.check_command(cfg, _flags(repair=True))["repaired"] == ["include/abi.h"]
    assert applied.call_count == 1 and applied.call_args.args[1].writes == {"include/abi.h": folded}


CLOSED = "extern f32 func_8024D398_de(Actor_func_80214624_de *);"
LOCAL = "extern f32 func_8024D398_de(void *);"


def test_check_lists_the_unit_whose_local_extern_conflicts_with_a_header_prototype(lane, monkeypatch):
    """The real src/func_80220ED4_de.c against include/shared/func_80214624_de_closed.h: every version fails."""
    cfg, unit, snapshot = lane[:3]
    unit = replace(unit, path="src/func_80220ED4_de.c")
    snapshot = replace(snapshot, layout=replace(snapshot.layout, units={unit.path: unit}))
    lane[3].return_value = snapshot
    monkeypatch.setattr(
        land.policy, "census", lambda snap, also=(), only=None: ((), [[g[1](i) for i in g[2]] for g in also])
    )

    def compiles(item):
        _, spec, version = item
        if LOCAL != CLOSED:
            return (
                Finding(
                    "compile.error", "conflicting types for `func_8024D398_de'", unit=spec.path, versions=(version,)
                ),
            )
        return ()

    monkeypatch.setattr(land.build, "compile_unit", compiles)
    with pytest.raises(Refusal) as refused:
        land.check_command(cfg, _flags())
    assert [(f.unit, f.versions) for f in refused.value.findings] == [("src/func_80220ED4_de.c", ("a", "b"))]
    assert json.loads((cfg.project.root / ".unbake/check.json").read_text())["failing"] == ["src/func_80220ED4_de.c"]


def test_a_check_of_a_header_compiles_only_the_units_that_read_it(lane, monkeypatch):
    cfg, unit, snapshot = lane[:3]
    units = {p: replace(unit, path=p) for p in (f"src/u{n}.c" for n in range(5))}
    snapshot = replace(snapshot, layout=replace(snapshot.layout, units=units), overlays={"include/h.h": b"int h;"})
    lane[3].return_value = snapshot
    readers = ("src/u0.c", "src/u2.c", "src/u4.c")
    monkeypatch.setattr(land.headers, "consumers", lambda snap, paths: (*readers, "include/other.h"))
    monkeypatch.setattr(
        land.build, "compile_jobs", lambda snap, paths: [(snap, snap.layout.units[p], "a") for p in paths]
    )
    checked = Mock(return_value=())
    monkeypatch.setattr(land.build, "check_unit", checked)
    monkeypatch.setattr(
        land.policy, "census", lambda snap, also=(), only=None: ((), [[g[1](i) for i in g[2]] for g in also])
    )
    land.check_command(cfg, _flags(paths=["include/h.h"]))
    assert sorted(c.args[0][1].path for c in checked.call_args_list) == list(readers)  # 3 readers, not 5 units


@pytest.mark.parametrize(
    ("status", "blocked"), [("added", False), ("existing", False), ("existing", True), ("empty", False)]
)
def test_check_staged_reads_index_and_head_without_builds(lane, monkeypatch, status, blocked):
    cfg, _, snapshot = lane[:3]
    path = "include/a header.h"
    old, staged = "union { int a; float b; };", "union { int a; void *b; };"
    selected = ["include/a header.h"]
    calls = []

    def git(config, *args):
        calls.append(args)
        if args[0] in ("rev-parse", "write-tree"):
            text = snapshot.commit if args[0] == "rev-parse" else "index"
        elif args[0] == "diff":
            assert args[-2:] == ("--", *selected)
            text = path + "\0note.txt\0" if status != "empty" else ""
        elif args[0] == "ls-tree":
            text = path + "\0" if status == "existing" else ""
        else:
            assert args[0] == "show"
            text = staged if args[1] == ":" + path else old
        return Mock(stdout=text.encode())

    monkeypatch.setattr(land.process, "git", git)
    monkeypatch.setattr(
        land.layout,
        "tree",
        lambda config, revision, commit: replace(
            snapshot,
            overlays={
                path: (staged if revision == "index" else old).encode()
                if revision == "index" or status == "existing"
                else None
            },
        ),
    )
    for target, name in ((land, "_repair"), (land.build, "compile_jobs")):
        monkeypatch.setattr(target, name, Mock(side_effect=AssertionError("staged check must not build or repair")))
    finding = Finding("source.union-view", "changed union", path=path, line=1, blocking=blocked)

    def census(snap, *, changes, baseline):
        assert baseline.peek(path) == (old.encode() if status == "existing" else None)
        assert changes == ({} if status == "empty" else {path: (old if status == "existing" else "", staged)})
        if changes:
            assert snap.read(path) == staged.encode()  # the working tree need not even contain this file
        return (() if status == "empty" else (finding,)), []

    monkeypatch.setattr(land.policy, "census", census)
    flags = _flags(staged=True, paths=selected, strict=True, repair=True)
    if blocked:
        with pytest.raises(Refusal) as error:
            land.check_command(cfg, flags)
        assert error.value.findings == (finding,)
    else:
        result = land.check_command(cfg, flags)
        configuration.validate("result.check", result, "result.check")
        assert result["counts"] == ({} if status == "empty" else {"source.union-view": 1})
    assert not (cfg.project.root / ".unbake/check.json").exists()
    if status != "existing":
        assert not any(a == ("show", snapshot.commit + ":" + path) for a in calls)


@pytest.mark.parametrize("strict,blocked", [(False, False), (True, False), (False, True), (True, True)])
def test_check_records_one_moved_unit_without_blocking(lane, monkeypatch, strict, blocked):
    cfg, unit = lane[:2]
    moved = Finding(
        "check.moved-during-run",
        "the source changed while check ran; it is checked next run",
        path=unit.path,
        unit=unit.path,
        blocking=False,
    )
    fault = Finding("compile.error", "bad code", unit="src/bad.c", versions=("a",))
    compiled = [(replace(moved, versions=(v,)),) for v in ("a", "b")]
    if blocked:
        compiled.append((fault,))
    monkeypatch.setattr(land.policy, "census", lambda snap, also=(), only=None: ((), [compiled]))
    if blocked:
        with pytest.raises(Refusal) as refused:
            land.check_command(cfg, _flags(strict=strict))
        assert refused.value.findings == (fault,)
    else:
        assert land.check_command(cfg, _flags(strict=strict))["counts"] == {moved.key: 1}
    report = json.loads((cfg.project.root / ".unbake/check.json").read_text())
    assert report["counts"] == {moved.key: 1}
    assert report["findings"] == [json.loads(json.dumps(asdict(replace(moved, versions=("a", "b")))))]
    assert report["failing"] == ([fault.unit] if blocked else [])


def test_staged_shared_record_change_scores_every_consumer_and_refuses_a_broken_user(lane, monkeypatch):
    cfg, unit, snapshot = lane[:3]
    snapshot = replace(snapshot, layout=replace(snapshot.layout, units={unit.path: unit}))
    lane[3].return_value = snapshot
    monkeypatch.setattr(land, "_consumers", lambda before, after, paths, changed: sorted(paths))
    path = "include/shared.h"
    monkeypatch.setattr(
        land.layout,
        "tree",
        lambda config, revision, commit: replace(
            snapshot,
            overlays={
                path: (
                    "struct Shared { int a; int b; };" if revision == "index" else "struct Shared { int a; };"
                ).encode(),
                "include/dependency.h": b"clean HEAD dependency",
            },
        ),
    )
    monkeypatch.setattr(
        land.process,
        "git",
        lambda config, *args: Mock(
            stdout=(
                snapshot.commit
                if args[0] == "rev-parse"
                else "index"
                if args[0] == "write-tree"
                else path + "\0"
                if args[0] in ("diff", "ls-tree")
                else "struct Shared { int a; int b; };"
                if args[1].startswith(":")
                else "struct Shared { int a; };"
            ).encode()
        ),
    )
    monkeypatch.setattr(land.policy, "census", lambda *args, **kwargs: ((), []))
    monkeypatch.setattr(land.headers, "consumers", lambda *args: (unit.path,))
    gap = Finding(
        "land.not_exact",
        "shared layout broke caller",
        unit=unit.members[0],
        versions=("a",),
        missing=("a: changed bytes",),
    )
    proof = Mock(side_effect=[((), (gap,)), ((Mock(member=unit.members[0], version="a", exact=True),), ())])
    monkeypatch.setattr(land.publish, "_proofs", proof)
    with pytest.raises(Refusal) as caught:
        land.check_command(cfg, _flags(staged=True))
    assert [f.key for f in caught.value.findings] == ["land.breaks_dependent"]
    assert proof.call_args.args[1] == [unit.path]
    assert proof.call_args.args[0].read("include/dependency.h") == b"clean HEAD dependency"


def test_staged_declaration_check_does_not_refuse_preexisting_consumer_debt(lane, monkeypatch):
    cfg, unit, snapshot = lane[:3]
    snapshot = replace(snapshot, layout=replace(snapshot.layout, units={unit.path: unit}))
    lane[3].return_value = snapshot
    monkeypatch.setattr(land, "_consumers", lambda before, after, paths, changed: sorted(paths))
    path = "include/shared.h"
    monkeypatch.setattr(
        land.layout,
        "tree",
        lambda config, revision, commit: replace(
            snapshot,
            overlays={
                path: (
                    "struct Shared { int a; int b; };" if revision == "index" else "struct Shared { int a; };"
                ).encode(),
                "include/dependency.h": b"clean HEAD dependency",
            },
        ),
    )
    monkeypatch.setattr(
        land.process,
        "git",
        lambda config, *args: Mock(
            stdout=(
                snapshot.commit
                if args[0] == "rev-parse"
                else "index"
                if args[0] == "write-tree"
                else path + "\0"
                if args[0] in ("diff", "ls-tree")
                else "struct Shared { int a; int b; };"
                if args[1].startswith(":")
                else "struct Shared { int a; };"
            ).encode()
        ),
    )
    monkeypatch.setattr(land.policy, "census", lambda *args, **kwargs: ((), []))
    monkeypatch.setattr(land.headers, "consumers", lambda *args: (unit.path,))
    gap = Finding("land.not_exact", "existing mismatch", unit=unit.members[0], versions=("a",))
    proof = Mock(return_value=((Mock(member=unit.members[0], version="a", exact=False),), (gap,)))
    monkeypatch.setattr(land.publish, "_proofs", proof)
    assert land.check_command(cfg, _flags(staged=True))["counts"] == {}
    assert proof.call_count == 2


@pytest.mark.parametrize("definition", [b"struct Shared { int a; };", b"typedef struct { int a; } Shared;"])
def test_staged_baseline_and_candidate_use_git_trees_with_private_index(lane, monkeypatch, definition):
    cfg, _unit, snapshot = lane[:3]
    path = "include/shared.h"
    old, new = definition, definition + definition
    (cfg.project.root / "include").mkdir()
    (cfg.project.root / path).write_bytes(b"dirty header")
    monkeypatch.setenv("GIT_INDEX_FILE", str(cfg.project.root / "private-index"))
    snapshots = {
        "head": replace(snapshot, overlays={path: old}, tree=True),
        "index": replace(snapshot, overlays={path: new}, tree=True),
    }
    trees = Mock(side_effect=lambda config, revision, commit: snapshots[revision])
    monkeypatch.setattr(land.layout, "tree", trees)
    monkeypatch.setattr(
        land.process,
        "git",
        lambda config, *args: Mock(
            stdout=(
                b"head"
                if args[0] == "rev-parse"
                else b"index"
                if args[0] == "write-tree"
                else (path + "\0").encode()
                if args[0] in ("diff", "ls-tree")
                else old
            )
        ),
    )
    monkeypatch.setattr(land.policy, "census", lambda *a, **kw: ((), []))
    monkeypatch.setattr(land.headers, "consumers", lambda *a: ())
    with pytest.raises(Refusal, match="duplicate complete"):
        land.check_command(cfg, _flags(staged=True))
    assert [(c.args[1], c.args[2]) for c in trees.call_args_list] == [("head", "head"), ("index", "head")]
    lane[3].assert_not_called()


@pytest.mark.parametrize("reference", ["shared->value", "READ(shared)", "helper()", "VALUE"])
def test_staged_consumer_filter_keeps_reached_types_and_caches_unchanged_users(cfg, monkeypatch, reference):
    from unbake import effort, pool

    monkeypatch.setattr(pool, "_in_worker", True)
    units = {p: UnitSpec(p, "c", "g", (p,), "gcc-test", {}) for p in ("src/changed.c", "src/user.c", "src/unrelated.c")}
    members = {p: Member(p, "function", "c", "g", (Placement("a", ".text", 0, 8, 0x80000000),)) for p in units}
    raw = {
        "src/changed.c": b"int changed(void) { return 1; }",
        "src/user.c": ("int user(void) { return " + reference + "; }").encode(),
        "src/unrelated.c": b"int unrelated(void) { return 0; }",
    }
    suffix = (
        b"\n#define READ(x) (((struct X *)(x))->value)\nstatic int helper(void) { return shared->value; }\n"
        b"enum { VALUE = sizeof(struct X) };\n"
    )
    raw = {p: b'#include "shared.h"\n' + text for p, text in raw.items()}
    before = Snapshot(
        cfg,
        "head",
        LayoutMap({}, members, units, "map", (), {}),
        {},
        {**raw, "include/shared.h": b"struct X { int value; }; extern struct X *shared;" + suffix},
        "before",
    )
    after = replace(
        before,
        overlays={
            **before.overlays,
            "include/shared.h": b"struct X { int pad; int value; }; extern struct X *shared;" + suffix,
        },
        digest="after",
    )
    monkeypatch.setattr(land.layout, "tree", lambda config, revision, commit: before if revision == "head" else after)
    monkeypatch.setattr(
        land.process,
        "git",
        lambda config, *args: Mock(
            stdout=(
                b"head"
                if args[0] == "rev-parse"
                else b"index"
                if args[0] == "write-tree"
                else b"src/changed.c\0include/shared.h\0"
            )
        ),
    )
    monkeypatch.setattr(land.policy, "census", lambda *a, **kw: ((), []))
    proof = Mock(return_value=((), ()))
    monkeypatch.setattr(land.publish, "_proofs", proof)
    jobs = []
    monkeypatch.setattr(effort, "record_pool", lambda name, items, **kw: jobs.append(kw["jobs"]))
    for run in range(2):
        land.check_command(cfg, _flags(staged=True))
        assert proof.call_args.args[1] == ["src/changed.c", "src/user.c"]
        assert (sum(jobs) > 0) == (run == 0)
        jobs.clear()


@pytest.mark.parametrize(
    "change,blocked",
    [
        ("pair", False),
        ("config-only", True),
        ("invalid-partition", True),
        ("inexact", True),
        ("unstaged-config", False),
        ("removed", False),
        ("new", False),
        ("unheld", False),
        ("removed-with-source", False),
    ],
)
def test_staged_regional_policy_uses_each_states_config(lane, monkeypatch, change, blocked):
    from unbake import ownership, policy

    cfg, unit, snapshot = lane[:3]
    row = {"function": "f", "sides": [["a"], ["b"]]}
    project = replace(cfg.project, version_macros={"a": ("-DVERSION_A",), "b": ()})
    old_config = replace(cfg, project=project)
    documented = replace(cfg, project=replace(project, version_differences=(row,), digest="regional"))
    old = b"int f(void) { return 0; }"
    new = b"int f(void) {\n#if defined(VERSION_A)\nreturn 1;\n#else\nreturn 0;\n#endif\n}"
    mapping = replace(
        snapshot.layout,
        units={unit.path: unit},
        groups={"group": Group("group", "main", ("f",), "authored", (), False, "src/group.c", "include/main/group.h")},
    )
    before = replace(
        snapshot,
        config=old_config,
        layout=mapping,
        versions={"a": "a", "b": "b"},
        overlays={unit.path: old},
        digest="before",
        tree=True,
    )
    after = replace(before, config=documented, overlays={unit.path: new}, digest="after")
    paths = [unit.path, "config.toml"]
    if change == "config-only":
        after = replace(after, overlays={unit.path: old})
        paths = ["config.toml"]
    elif change == "invalid-partition":
        row["sides"] = [["a"]]
    elif change == "unstaged-config":
        after = replace(before, digest="after")
        paths = ["notes.txt"]
    elif change == "removed":
        before, after = (
            after,
            replace(before, layout=replace(mapping, units={}, members={}), overlays={unit.path: None}),
        )
    elif change == "new":
        before = replace(before, layout=replace(mapping, units={}, members={}), overlays={unit.path: None})
    elif change == "unheld":
        before = replace(before, layout=replace(mapping, units={}))
        after = replace(after, layout=replace(mapping, units={}))
    elif change == "removed-with-source":
        before, after = after, replace(after, layout=replace(mapping, units={}), config=old_config)
    monkeypatch.setattr(
        land.process,
        "git",
        lambda config, *args: Mock(
            stdout=(
                b"head"
                if args[0] == "rev-parse"
                else b"index"
                if args[0] == "write-tree"
                else ("\0".join(paths) + "\0").encode()
            )
        ),
    )
    monkeypatch.setattr(land.layout, "tree", lambda config, revision, commit: before if revision == "head" else after)
    monkeypatch.setattr(
        land.layout,
        "overlay",
        lambda snap, writes: replace(snap, overlays={**snap.overlays, **writes}, digest=digest((snap.digest, writes))),
    )
    monkeypatch.setattr(policy.versions, "rom_bytes", lambda version, *args: version.encode())
    monkeypatch.setattr(policy.recipes, "resolve", lambda *args: {})
    monkeypatch.setattr(policy.native, "stamp", lambda snap, unit, version, recipe: version)
    monkeypatch.setattr(ownership, "derive", lambda snap, unit: (snap, unit))
    monkeypatch.setattr(policy.native, "prove_job", lambda job: [Mock(version=job[2], exact=change != "inexact")])
    proof = Mock(return_value=((), ()))
    monkeypatch.setattr(land.publish, "_proofs", proof)
    if blocked:
        with pytest.raises(Refusal) as caught:
            land.check_command(documented, _flags(staged=True))
        assert {f.key for f in caught.value.findings} == {"source.file-version-guard"}
    else:
        assert land.check_command(documented, _flags(staged=True))["counts"] == {}
        assert proof.call_count == (change in ("pair", "new"))
        if proof.called:
            assert proof.call_args.args == (after, [unit.path])


@pytest.mark.parametrize("baseline", ["existing", "new-unit", "new-holder"])
def test_staged_proof_regressions_require_a_baseline_holder(lane, monkeypatch, baseline):
    cfg, unit, snapshot = lane[:3]
    before = replace(snapshot, layout=replace(snapshot.layout, units={unit.path: unit}))
    after = replace(before, overlays={unit.path: b"int f(void) { return 1; }"}, digest="after")
    if baseline == "new-unit":
        before = replace(before, layout=replace(before.layout, units={}, members={}))
    elif baseline == "new-holder":
        member = before.layout.members["f"]
        before = replace(
            before,
            layout=replace(
                before.layout,
                members={"f": replace(member, placements=tuple(p for p in member.placements if p.version == "a"))},
            ),
        )
    monkeypatch.setattr(
        land.process,
        "git",
        lambda config, *args: Mock(
            stdout=(
                b"head"
                if args[0] == "rev-parse"
                else b"index"
                if args[0] == "write-tree"
                else (unit.path + "\0").encode()
            )
        ),
    )
    monkeypatch.setattr(land.layout, "tree", lambda config, revision, commit: before if revision == "head" else after)
    monkeypatch.setattr(land.policy, "census", lambda *args, **kwargs: ((), []))
    gap = Finding("land.not_exact", "changed bytes", unit="f", versions=("b",))
    monkeypatch.setattr(land.publish, "_proofs", Mock(return_value=((), (gap,))))
    if baseline == "existing":
        with pytest.raises(Refusal) as caught:
            land.check_command(cfg, _flags(staged=True))
        assert caught.value.findings == (gap,)
    else:
        assert land.check_command(cfg, _flags(staged=True))["counts"] == {}


@pytest.mark.parametrize("revision", ["head", "private-index"])
def test_staged_tree_loads_project_inputs_from_its_own_blobs(cfg, monkeypatch, revision):
    import io
    import tarfile

    raw = (cfg.project.root / "config.toml").read_bytes()
    blobs = {"config.toml": raw.replace(b'title = "Fixture"', b'title = "Blobs"'), "layout.toml": b"layout"}
    for files in cfg.project.version_files.values():
        for path in (files.split, files.symbols):
            blobs[path] = (cfg.project.root / path).read_bytes()
            (cfg.project.root / path).write_bytes(b"dirty project input")
    (cfg.project.root / "config.toml").write_bytes(b"dirty config")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for path, data in blobs.items():
            info = tarfile.TarInfo(path)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(land.process, "git", Mock(return_value=Mock(stdout=stream.getvalue())))
    seen = []

    def versions(config, read):
        seen.append(config)
        assert all(read(path) == data for path, data in blobs.items())
        return {}

    monkeypatch.setattr(land.layout.version_data, "read", versions)
    monkeypatch.setattr(land.layout, "load_map", lambda config, *args: LayoutMap({}, {}, {}, "map", (), {}))
    monkeypatch.setattr(land.layout, "_content", lambda *args: "content")
    snap = land.layout.tree(cfg, revision, "head")
    assert snap.config.project.title == "Blobs"
    assert snap.config.project.digest != cfg.project.digest
    assert snap.config.host is cfg.host and seen == [snap.config]
    assert snap.read("config.toml") == blobs["config.toml"]


def test_repair_drops_dead_header_statements(repair, monkeypatch):
    cfg, applied = repair
    header = (
        b"/* unbake: generated */\nstruct func_8022C6D4_S1;\ntypedef int Used;\nextern Used live_value;\n"
        b"#define KEEP(x) ((x) + 1)\n"
    )
    (cfg.project.root / "include/abi.h").write_bytes(header)
    monkeypatch.setattr(land, "_live_words", lambda snap: {"live_value"})
    monkeypatch.setattr(land.headers, "reconcile", lambda snap, over, texts: ({}, []))
    monkeypatch.setattr(land.native, "compiles", lambda item: ())
    assert land.check_command(cfg, _flags(repair=True))["repaired"] == ["include/abi.h"]
    written = applied.call_args.args[1].writes["include/abi.h"]
    assert written == b"typedef int Used;\nextern Used live_value;\n#define KEEP(x) ((x) + 1)\n"
    assert applied.call_count == 1


def test_staged_own_excludes_project_units(lane, monkeypatch):
    cfg, unit, snapshot = lane[:3]
    snapshot = replace(snapshot, layout=replace(snapshot.layout, units={unit.path: unit}))
    lane[3].return_value = snapshot
    staged = replace(
        snapshot, config=replace(snapshot.config, project=replace(snapshot.config.project, toolchain="x", digest="new"))
    )
    monkeypatch.setattr(
        land.layout, "tree", lambda config, revision, commit: staged if revision == "index" else snapshot
    )
    monkeypatch.setattr(
        land.process,
        "git",
        lambda config, *args: Mock(
            stdout=(snapshot.commit if args[0] == "rev-parse" else "index" if args[0] == "write-tree" else "").encode()
        ),
    )
    monkeypatch.setattr(land, "_consumers", lambda before, after, paths, changed: sorted(paths))
    monkeypatch.setattr(land.policy, "census", lambda *args, **kwargs: ((), []))
    monkeypatch.setattr(land.headers, "consumers", lambda *args: (unit.path,))
    gap = Finding("land.not_exact", "existing mismatch", unit=unit.members[0], versions=("a",))
    proof = Mock(return_value=((Mock(member=unit.members[0], version="a", exact=False),), (gap,)))
    monkeypatch.setattr(land.publish, "_proofs", proof)
    assert land.check_command(cfg, _flags(staged=True))["counts"] == {}


def test_repair_writes_the_derived_version_rows_into_config(repair, monkeypatch):
    cfg, applied = repair
    rows = [{"function": "f", "sides": [["a"], ["b"]]}]
    monkeypatch.setattr(land.policy, "_derived_rows", lambda snap: rows)
    monkeypatch.setattr(land.headers, "sources", lambda snap: [])
    monkeypatch.setattr(land.headers, "reconcile", lambda *args: ({}, []))
    assert land.check_command(cfg, _flags(repair=True))["repaired"] == ["config.toml"]
    written = applied.call_args.args[1].writes["config.toml"].decode()
    assert written == 'schema = 3\n[[project.version_differences]]\nfunction = "f"\nsides = [["a"], ["b"]]\n'


def test_prototype_repair_skips_the_name_whose_proof_regressed_and_writes_the_rest(monkeypatch):
    snapshot = SimpleNamespace(config="cfg", digest="d", layout=SimpleNamespace(units={"src/a.c": 1, "src/b.c": 1}))
    plan = {"keep": (False, frozenset({"src/a.c"})), "typed": (True, frozenset({"src/b.c"}))}

    def prototypes(snap, skip):
        names = {n: v for n, v in plan.items() if n not in skip}
        return names, {p: b"x" for _, paths in names.values() for p in paths}, []

    applied = Mock()
    monkeypatch.setattr(land.headers, "prototypes", prototypes)
    monkeypatch.setattr(land.headers, "consumers", lambda snap, paths: ())
    monkeypatch.setattr(land.layout, "overlay", lambda snap, writes: snap)
    monkeypatch.setattr(land.layout, "unit_of", lambda snap, name: None)
    monkeypatch.setattr(land, "failing", lambda snap, paths: [])
    monkeypatch.setattr(land.publish, "_proofs", lambda snap, units: ((), ("gap",)))
    monkeypatch.setattr(
        land.publish,
        "_regressions",
        lambda snap, gaps, own, who: (Finding("land.breaks_dependent", "breaks b", path="src/b.c", unit="m"),),
    )
    monkeypatch.setattr(land.journal, "apply", applied)
    monkeypatch.setattr(land.publish, "progress_block", lambda snap, writes: {})
    assert land._prototypes(snapshot) == ({"src/a.c": b"x"}, [])
    assert applied.call_args.args[1].writes == {"src/a.c": b"x"}


def test_prototype_repair_skips_the_name_that_adds_a_conversion_finding(monkeypatch):
    snapshot = SimpleNamespace(config="cfg", digest="d", layout=SimpleNamespace(units={"src/a.c": 1}))
    calls = []

    def prototypes(snap, skip):
        return (
            ({} if "n" in skip else {"n": (False, frozenset({"src/a.c"}))}),
            ({} if "n" in skip else {"src/a.c": b"x"}),
            [],
        )

    def failing(snap, paths):
        calls.append(snap is snapshot)
        return [] if snap is snapshot else [Finding("land.view-conversion", "int as pointer", unit="src/a.c")]

    monkeypatch.setattr(land.headers, "prototypes", prototypes)
    monkeypatch.setattr(land.headers, "consumers", lambda snap, paths: ())
    monkeypatch.setattr(land.layout, "overlay", lambda snap, writes: SimpleNamespace(layout=snap.layout))
    monkeypatch.setattr(land, "failing", failing)
    monkeypatch.setattr(land.journal, "apply", Mock(side_effect=AssertionError("nothing may be written")))
    assert land._prototypes(snapshot) == ({}, [])


def test_scoped_check_judges_only_the_given_units_and_keeps_the_project_census(lane, monkeypatch):
    cfg = lane[0]
    seen = {}

    def census(snap, also=(), only=None):
        seen["only"], seen["jobs"] = only, [job for g in also for job in g[2]]
        return (), [[]]

    monkeypatch.setattr(land.policy, "census", census)
    with pytest.raises(Refusal) as unknown:
        land.check_command(cfg, _flags(paths=("src/none.c",)))
    assert unknown.value.findings[0].key == "check.path" and "src/none.c" in unknown.value.findings[0].reason
    assert "only" not in seen  # an unknown path judges nothing and everything: it is refused by name
    assert not (cfg.project.root / ".unbake/check.json").exists()
    land.check_command(cfg, _flags())
    assert seen["only"] is None
    assert (cfg.project.root / ".unbake/check.json").exists()


def test_a_check_path_gives_the_same_scope_relative_or_absolute(tmp_path, monkeypatch):
    monkeypatch.setattr(land.headers, "consumers", lambda snap, paths: ("src/a.c",))
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(root=tmp_path)),
        layout=SimpleNamespace(units={"src/a.c": 1}),
        peek=lambda path: b"" if path == "include/a.h" else None,
    )
    assert land._scope(snapshot, [str(tmp_path / "src/a.c")]) == land._scope(snapshot, ["src/a.c"]) == {"src/a.c"}
    assert land._scope(snapshot, [str(tmp_path / "include/a.h")]) == land._scope(snapshot, ["include/a.h"])
    with pytest.raises(Refusal):
        land._scope(snapshot, ["/elsewhere/src/a.c"])


def test_a_header_left_naming_a_dropped_declaration_makes_every_includer_a_declaration_user(monkeypatch):
    monkeypatch.setattr(land.pool, "map", lambda _c, _n, fn, items, _k: [fn(i) for i in items])
    monkeypatch.setattr(land.headers, "pool", land.pool)
    monkeypatch.setattr(land.headers, "sources", lambda s: [p for p in s.files if p.endswith(".h")])
    rest = {
        "include/span.h": b'#include "types.h"\nstruct S { T1 unk0; };\n',
        "src/a.c": b'#include "span.h"\nint f(void) { return 0; }\n',
        "src/b.c": b'#include "other.h"\nint g(void) { return 0; }\n',
        "include/other.h": b"int unrelated;\n",
    }

    def tree(types_h: bytes, digest: str):
        files = {**rest, "include/types.h": types_h}
        units = {p: SimpleNamespace(path=p) for p in files if p.endswith(".c")}
        return SimpleNamespace(
            peek=files.get, read=files.__getitem__, files=files, digest=digest, overlays={},
            layout=SimpleNamespace(units=units), config=None,
        )  # fmt: skip

    before, after = tree(b"typedef int T1;\ntypedef int T2;\n", "du-before"), tree(b"typedef int T2;\n", "du-after")
    users = land.declaration_users(before, after, {"src/a.c", "src/b.c"}, ["include/types.h"])
    assert users == {"src/a.c"}  # a.c never names T1 but its header does: it cannot compile; b.c is untouched


GUARDED = b"#ifndef %s\n#define %s\n%s#endif\n"


def _includes_tree(monkeypatch, span: bytes, extra: bytes, other: bytes, digest: str):
    """span.h is included by every source; a.c names ALSynth, b.c names T1, c.c also includes other.h, d.c none."""
    monkeypatch.setattr(land.pool, "map", lambda _c, _n, fn, items, _k: [fn(i) for i in items])
    monkeypatch.setattr(land.headers, "pool", land.pool)
    monkeypatch.setattr(land.headers, "sources", lambda s: [p for p in s.files if p.endswith(".h")])
    files = {
        "include/span.h": span,
        "include/extra.h": extra,
        "include/other.h": other,
        "src/a.c": b'#include "span.h"\nint f(ALSynth *s) { return 0; }\n',
        "src/b.c": b'#include "span.h"\nT1 g(void) { return 0; }\n',
        "src/c.c": b'#include "span.h"\n#include "other.h"\nint h(void) { return 0; }\n',
        "src/d.c": b'#include "span.h"\nint k(void) { return 0; }\n',
    }
    units = {p: SimpleNamespace(path=p) for p in files if p.endswith(".c")}
    return SimpleNamespace(
        peek=files.get, read=files.__getitem__, files=files, digest=digest, overlays={},
        layout=SimpleNamespace(units=units), config=None,
    )  # fmt: skip


INCLUDERS = {"src/a.c", "src/b.c", "src/c.c", "src/d.c"}
EXTRA = GUARDED % (b"EXTRA_H", b"EXTRA_H", b"typedef struct ALSynth { int x; } ALSynth;\n")


@pytest.mark.parametrize(
    ("span_after", "other", "users"),
    [
        # A2c: span.h gains an include: only the includer naming what it reaches reads anything new
        (b'#include "extra.h"\ntypedef int T1;\n', b"int unrelated;\n", {"src/a.c"}),
        # and a changed declaration b.c uses still proves b.c
        (b'#include "extra.h"\ntypedef long T1;\n', b"int unrelated;\n", {"src/a.c", "src/b.c"}),
        # another header declaring the reached name may clash in c.c, which never names it
        (b'#include "extra.h"\ntypedef int T1;\n', b"struct ALSynth { int y; };\n", {"src/a.c", "src/c.c"}),
        # a conditional directive change can change anything: every includer
        (b'#include "extra.h"\n#if 0\ntypedef int T1;\n#endif\n', b"int unrelated;\n", INCLUDERS),
    ],
    ids=["gained-include", "used-declaration-changed", "clash", "conditional"],
)
def test_an_include_a_header_gains_proves_only_the_includers_naming_what_it_reaches(
    monkeypatch, span_after, other, users
):
    span_before = GUARDED % (b"SPAN_H", b"SPAN_H", b"typedef int T1;\n")
    before = _includes_tree(monkeypatch, span_before, EXTRA, other, f"inc-before-{users}")
    after = _includes_tree(monkeypatch, GUARDED % (b"SPAN_H", b"SPAN_H", span_after), EXTRA, other, f"inc-{users}")
    assert land.declaration_users(before, after, set(INCLUDERS), ["include/span.h"]) == users


def test_an_unguarded_or_unfound_reached_header_proves_every_includer(monkeypatch):
    span_before = GUARDED % (b"SPAN_H", b"SPAN_H", b"typedef int T1;\n")
    for n, (extra, name) in enumerate(((b"typedef int ALSynth;\n", b"extra.h"), (EXTRA, b"missing.h"))):
        span_after = GUARDED % (b"SPAN_H", b"SPAN_H", b'#include "' + name + b'"\ntypedef int T1;\n')
        before = _includes_tree(monkeypatch, span_before, extra, b"", f"ung-before-{n}")
        after = _includes_tree(monkeypatch, span_after, extra, b"", f"ung-after-{n}")
        assert land.declaration_users(before, after, set(INCLUDERS), ["include/span.h"]) == INCLUDERS


@pytest.mark.parametrize("key", ["headers.conflict", "source.fake-type", "land.view-conversion", "land.request"])
def test_submit_runs_the_landing_check_and_writes_nothing_when_it_refuses(lane, key):
    cfg, _, snapshot = lane[:3]
    check = land.publish.accept
    check.side_effect = Refusal(Finding(key, "refused by the landing check"))
    with pytest.raises(Refusal) as raised:
        _submit(lane)
    assert [f.key for f in raised.value.findings] == [key]
    assert not list((cfg.project.root / ".unbake/inbox").glob("*"))
    (snapshot_arg, request, extras, partial), named = check.call_args
    assert snapshot_arg is snapshot and request["function"] == "f" and extras == {} and partial is False
    assert named == {"present": False}  # admission asks only whether land refuses: no progress block


def test_a_drain_under_memory_pressure_trims_the_memo_after_each_entry(lane, monkeypatch):
    cfg = lane[0]
    trims, real = [], land._land

    class Span:
        def add(self, *, items=0, findings=()):
            trims.append(items)

    @contextmanager
    def stage(name):
        yield Span() if name == "effort.trimmed" else None

    def land_one(config, entry):
        land.effort.memo(("overlay", entry.id), lambda: 1)
        return real(config, entry)

    entries = [_submit(lane, source) for source in (b"one", b"two", b"three")]
    _order(cfg, entries)
    monkeypatch.setattr(land.effort, "stage", stage)
    monkeypatch.setattr(land.effort, "_pressure", lambda: (900, 1000))
    monkeypatch.setattr(land, "_land", land_one)
    land.drain(cfg)
    assert trims == [1, 1, 1]
    assert not [k for k in land.effort._memo if k[0] == "overlay"]


def test_check_dispatches_the_exact_compile_job():
    source = inspect.getsource(land.check_command)
    assert "build.check_unit" in source and "build.compile_unit" not in source
    assert "build.check_key" in source  # a warm unit dispatches nothing, even after an install
