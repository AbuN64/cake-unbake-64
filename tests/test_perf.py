"""The shared performance mechanisms count work: warm items never dispatch, readers run once, the watchdog speaks."""

from dataclasses import replace
from pathlib import Path

import pytest
from test_effort import _config

from unbake import effort, pool, process
from unbake.contracts import Origin, Refusal


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    config = _config(tmp_path)
    origins = {
        **config.host.origins,
        **{
            f"budgets.{k}": Origin(k, "/host.toml", "0" * 64)
            for k in ("serial_seconds", "serial_cores", "pool_fill", "pool_fanout")
        },
    }
    monkeypatch.setattr(pool, "_in_worker", True)  # jobs run in this process: the count is what is under test
    monkeypatch.setattr(effort, "_listeners", [])
    return replace(config, host=replace(config.host, origins=origins))


def test_a_keyed_map_dispatches_only_its_misses(cfg):
    ran = []

    def job(item):
        ran.append(item)
        return item * 2

    with effort.command("check", []):
        assert pool.map(cfg, "doubles", job, [1, 2, 3], str) == [2, 4, 6]
        assert pool.map(cfg, "doubles", job, [1, 2, 3, 4], str) == [2, 4, 6, 8]
        assert pool.map(cfg, "doubles", job, [1, 2, 3, 4], str) == [2, 4, 6, 8]
    assert ran == [1, 2, 3, 4]  # the warm map dispatched 0 items
    assert [r.jobs for r in effort.closed() if r.kind == "pool"] == [3, 1, 0]


def test_a_none_key_always_dispatches(cfg):
    ran = []
    with effort.command("check", []):
        for _ in range(2):
            pool.map(cfg, "open", ran.append, [1], lambda item: None)
    assert ran == [1, 1]


def test_gather_runs_independent_groups_in_one_dispatch_loop(cfg):
    with effort.command("check", []):
        found = pool.gather(cfg, [("squares", lambda v: v * v, [1, 2, 3]), ("names", str, ["a"], str)])
    assert found == [[1, 4, 9], ["a"]]
    assert len([r for r in effort.closed() if r.path[-1] == "pool.map"]) == 1


def test_memo_computes_once_per_command_and_resets(cfg):
    calls = []

    def produce():
        calls.append(1)
        return len(calls)

    with effort.command("check", []):
        assert [effort.memo("k", produce), effort.memo("k", produce)] == [1, 1]
    with effort.command("check", []):
        assert effort.memo("k", produce) == 2


def _budgeted(cfg, *changes):
    """Each change is applied to one finished stage of a bound command and run through the watchdog in turn."""
    with effort.command("check", []):
        effort.bind(cfg)
        with effort.stage("work"):
            pass
        base = next(r for r in effort.closed() if r.path[-1] == "work")
        return [effort._budget(replace(base, **change), True) for change in changes]


def test_budget_flags_a_single_core_stage(cfg):
    (record,) = _budgeted(cfg, {"wall_seconds": 5.0, "cores": 0.5})
    assert [f.key for f in record.findings] == ["budget.single_core"]
    assert not record.findings[0].blocking


def test_budget_flags_an_underused_pool_and_a_warm_dispatch(cfg):
    (record,) = _budgeted(
        cfg, {"kind": "pool", "jobs": 40, "workers_admitted": 2, "cores": 0.5, "cache": {"x": (3, 0)}}
    )
    assert {f.key for f in record.findings} == {"budget.pool_underused", "budget.warm_dispatch"}
    (inline,) = _budgeted(cfg, {"kind": "pool", "jobs": 1, "workers_admitted": 0, "cache": {"x": (3, 0)}})
    assert not inline.findings


def test_budget_flags_a_stage_that_closes_twice(cfg):
    first, second = _budgeted(cfg, {"wall_seconds": 5.0, "cores": 3.0}, {"wall_seconds": 5.0, "cores": 3.0})
    assert not first.findings and [f.key for f in second.findings] == ["budget.repeated"]


def test_two_slow_dispatches_of_different_work_are_not_repeated_but_the_same_dispatch_twice_is(cfg):
    def gather(span, *names):
        effort._dispatched[span] = names
        return {"path": ("land", "pool.map"), "span": span, "wall_seconds": 5.0, "cores": 3.0}

    keys = _budgeted(cfg, gather("a", "rescore.keys"), gather("b", "rescore.measure"), gather("c", "rescore.keys"))
    assert [[f.key for f in r.findings] for r in keys] == [[], [], ["budget.repeated"]]


def test_a_gather_is_not_the_repeat_of_its_own_dispatch_record(cfg):
    # the dispatch record's path is the gather's path then its name: the gather's key once read the same
    effort._dispatched["g"] = ("policy.unclean.keys",)
    pool = {"kind": "pool", "path": ("report", "pool.map", "policy.unclean.keys"), "wall_seconds": 5.0, "cores": 3.0}
    gather = {"path": ("report", "pool.map"), "span": "g", "wall_seconds": 5.0, "cores": 3.0}
    assert [[f.key for f in r.findings] for r in _budgeted(cfg, pool, gather)] == [[], []]


def test_process_refuses_make_without_jobs(tmp_path):
    tool = "make"
    executable = tmp_path / tool
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    with pytest.raises(Refusal) as refused:
        process.run("tool", [str(executable), "arg"], tmp_path, tmp=tmp_path / "tmp")
    assert tool in refused.value.findings[0].reason and Path(executable).is_file()


def test_peek_reads_a_file_or_says_it_is_absent(tmp_path):
    from unbake.contracts import Snapshot

    config = _config(tmp_path)
    (config.project.root / "here.txt").write_bytes(b"x")
    snapshot = Snapshot(config, "head", None, {}, {"over.txt": b"y", "gone.txt": None}, "digest")
    assert [snapshot.peek(p) for p in ("here.txt", "over.txt", "gone.txt", "absent.txt")] == [b"x", b"y", None, None]


def test_budget_does_not_flag_a_stage_shipped_from_many_jobs_as_closed_twice(cfg):
    """Every job of a pool closes the same stage path: that is the design, not a repeat."""
    with effort.command("check", []):
        effort.bind(cfg)
        with effort.stage("work"):
            pass
        base = next(r for r in effort.closed() if r.path[-1] == "work")
        slow = replace(base, wall_seconds=5.0, cores=3.0)
        assert [effort._budget(slow, True, True).findings for _ in range(3)] == [(), (), ()]
        assert not effort._seen


@pytest.mark.parametrize(("held", "flagged"), [(6.8, False), (12.0, True)])
def test_a_pool_is_judged_against_the_slots_it_held_not_the_most_it_was_admitted(cfg, held, flagged):
    """Move round 1 (db552f2 slice): 501 jobs, 12 admitted at most, 6.6 cores over 162 s while a peer command held
    half the host. Busy for the share it held is not underused; the same cores with all 12 held are."""
    (record,) = _budgeted(
        cfg,
        {
            "kind": "pool",
            "jobs": 501,
            "workers_admitted": 12,
            "wall_seconds": 162.0,
            "cores": 6.6,
            "slot_seconds": 162.0 * held,
            "cache": {"x": (0, 501)},
        },
    )
    assert ("budget.pool_underused" in {f.key for f in record.findings}) is flagged
