"""Bounded ordered maps, metrics, and progress using futures and forked workers."""

import fcntl
import importlib
import io
import os
import pickle
import resource
import sys
import threading
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_effort import _config

from unbake import config as configuration
from unbake import effort, pool
from unbake.contracts import Finding, Refusal, sha256

_RAISED = ValueError("unset")


def _fail(value):  # module level: items and functions travel to workers through a pickle
    raise _RAISED


_RAN: list[int] = []


def _record(value):
    _RAN.append(value)  # inline in the parent here: the pool runs a group of two in this process or a worker
    return value


def _name(value):
    return str(value)


def _square(value):
    return value * value


@pytest.fixture(autouse=True)
def _state(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path_factory.mktemp("run")))
    monkeypatch.setattr(effort, "_listeners", [])
    monkeypatch.setattr(pool, "_in_worker", False)
    yield
    pool._drop_executor()


@pytest.fixture
def mocked_parallel(monkeypatch):
    active = set()

    class Widths(list):
        executor = None

    widths = Widths()

    class Executor:
        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            active.add(future)
            widths.append(len(active))
            return future

    def wait(pending, return_when, timeout=None):
        selected = next(reversed(pending))
        active.discard(selected)
        return {selected}, set(pending) - {selected}

    monkeypatch.setattr(pool, "_get_executor", lambda config, needed=(), work=0: Executor())
    monkeypatch.setattr(pool, "wait", wait)
    widths.executor = Executor
    return widths


@pytest.mark.parametrize("serial", [True, False])
def test_progress_counts(tmp_path, monkeypatch, mocked_parallel, serial):
    monkeypatch.setattr(pool, "_in_worker", serial)
    monkeypatch.setenv("UNBAKE_CORES", "2")  # two cores passed down by a holder, whatever the runner has
    for name, value in (("_dir", []), ("_cmd", []), ("_want", 0), ("_size", 0), ("_share", 0)):
        monkeypatch.setattr(pool, name, value)  # the slot state another test on this worker left behind
    seen = []
    effort.listen(lambda event, body: seen.append(body) if event == "progress" else None)
    with effort.command("check", []):
        assert pool.map(_config(tmp_path), "units", _square, [1, 2, 3, 4]) == [1, 4, 9, 16]
    assert seen == [{"name": "units", "done": n, "total": 4} for n in range(1, 5)]
    if not serial:
        assert max(mocked_parallel) == 2  # only admitted jobs hold slots; no queued work escapes rebalancing


@pytest.mark.parametrize("items", [[], [3]])
def test_inline_no_executor(tmp_path, monkeypatch, items):
    def forbidden(config):
        pytest.fail("inline map started an executor")

    monkeypatch.setattr(pool, "_get_executor", forbidden)
    with effort.command("check", []):
        assert pool.map(_config(tmp_path), "units", _square, items) == [v * v for v in items]
    record = next(r for r in effort.closed() if r.kind == "pool")
    assert record.items == record.jobs == len(items)
    assert record.workers_admitted == 0


@pytest.mark.parametrize(
    "exc,key",
    [
        (Refusal(Finding("effort.log", "refused")), "effort.log"),
        (MemoryError("limit"), "worker.memory"),
        (ValueError("crash"), "worker.crash"),
    ],
)
def test_worker_failure_mapping(tmp_path, mocked_parallel, monkeypatch, exc, key):
    monkeypatch.setattr(sys.modules[__name__], "_RAISED", exc)
    with effort.command("check", []), pytest.raises(Refusal) as caught:
        pool.map(_config(tmp_path), "units", _fail, [1, 2])
    assert caught.value.findings[0].key == key
    if key == "effort.log":
        assert caught.value is exc
    if key == "worker.memory":
        assert caught.value.findings[0].origin == _config(tmp_path).host.origins["resources.memory_worker_bytes"]
    assert next(r for r in effort.closed() if r.kind == "pool").jobs == 2


def test_job_cache_and_metrics(tmp_path):
    def work(value):
        effort.count("object", True)
        effort.count("view", False)
        return value

    ok, value, envelope = pool._job(work, 7)
    assert (ok, value) == (True, 7)
    assert envelope["cache"] == {"object": (1, 0), "view": (0, 1)}
    assert envelope["worker_cpu"] >= 0
    assert envelope["rss"] > 0
    assert [r.kind for r in envelope["records"]] == ["job"]  # every record: the chunk folds, the parent floors


def test_broken_pool_dropped(tmp_path, monkeypatch, mocked_parallel):
    dropped = []

    def broken(pending, return_when, timeout=None):
        raise BrokenProcessPool("worker gone")

    monkeypatch.setattr(pool, "wait", broken)
    monkeypatch.setattr(pool, "_drop_executor", lambda: dropped.append(True))
    with effort.command("check", []), pytest.raises(Refusal) as caught:
        pool.map(_config(tmp_path), "units", _square, [1, 2])
    assert caught.value.findings[0].key == "worker.crash"
    assert dropped == [True]


def test_executor_reuse_and_host_change(tmp_path, monkeypatch):
    made = []

    class Executor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.shutdowns = []
            made.append(self)

        def submit(self, function):
            f = Future()
            f.set_result(None)
            return f

        def shutdown(self, **kwargs):
            self.shutdowns.append(kwargs)

    monkeypatch.setattr(pool, "ProcessPoolExecutor", Executor)
    config = _config(tmp_path)
    first = pool._get_executor(config)
    assert pool._get_executor(config) is first
    second = pool._get_executor(replace(config, host=replace(config.host, digest="other")))
    assert second is not first
    assert first.shutdowns == [{"wait": True, "cancel_futures": True}]
    assert second.kwargs["max_workers"] == config.host.workers
    assert second.kwargs["mp_context"].get_start_method() == "fork"
    assert second.kwargs["initargs"] == (config.host.memory_worker_bytes,)


def test_threads_asking_at_once_create_one_executor(tmp_path, monkeypatch):
    made = []

    class Executor:
        def __init__(self, **kwargs):
            time.sleep(0.05)  # long enough for every other thread to arrive while this one is creating
            made.append(self)

        def submit(self, function):
            f = Future()
            f.set_result(None)
            return f

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(pool, "ProcessPoolExecutor", Executor)
    config = _config(tmp_path)
    got = []
    threads = [threading.Thread(target=lambda: got.append(pool._get_executor(config))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made) == 1 and got == made * 8


def test_real_forked_map(tmp_path, monkeypatch):
    seen = []
    effort.listen(lambda event, body: seen.append(body["done"]) if event == "progress" else None)
    with effort.command("check", []):
        assert pool.map(_config(tmp_path), "units", _square, [2, 3, 4]) == [4, 9, 16]
    assert seen == [1, 2, 3]
    record = next(r for r in effort.closed() if r.kind == "pool")
    assert record.jobs == 3
    assert 1 <= record.workers_used <= record.workers_admitted == 2


def test_large_map_batches_transport_and_preserves_every_metric(tmp_path, mocked_parallel):
    items = list(range(256))
    with effort.command("check", []):
        values = pool.map(_config(tmp_path), "units", _square, items)
    assert values == [v * v for v in items]
    record = next(r for r in effort.closed() if r.kind == "pool")
    assert record.items == record.jobs == len(items)
    assert record.workers_admitted == 2


def _identify(value):
    return value, os.getpid()


def test_real_batched_work_count_order_and_worker_pids(tmp_path, monkeypatch):
    items = list(range(256))
    with effort.command("check", []):
        values = pool.map(_config(tmp_path), "units", _identify, items)
    assert [value for value, pid in values] == items
    assert all(pid != os.getpid() for value, pid in values)
    metrics = next(r for r in effort.closed() if r.kind == "pool")
    assert metrics.jobs == metrics.items == len(items)
    assert metrics.workers_used == len({pid for value, pid in values})
    assert metrics.workers_admitted == 2
    assert sum(r.spans for r in effort.closed() if r.kind == "job") == len(items)  # every job, folded


def test_job_ships_only_notable_records(tmp_path):
    def work(value):
        with effort.stage("quick"):
            pass
        with effort.stage("waited"):
            effort.waited(0.5)
        return value

    _, _, everything = pool._job(work, 1)
    assert {r.path[-1] for r in everything["records"]} == {"quick", "waited", "job", "(own)"}


def test_pool_record_measures_the_parent_and_the_workers(tmp_path, mocked_parallel):
    with effort.command("check", []):
        pool.map(_config(tmp_path), "units", _square, list(range(80)))
    record = next(r for r in effort.closed() if r.kind == "pool")
    assert record.dispatch_seconds >= 0 and record.busy_seconds >= 0 and record.jobs == 80


def test_worker_dies_with_its_parent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pool.resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(pool.ctypes, "CDLL", lambda name: type("L", (), {"prctl": lambda self, *a: calls.append(a)})())
    monkeypatch.setattr(pool, "_in_worker", False)
    dropped = []
    monkeypatch.setattr(pool.store, "drop_inherited", lambda: dropped.append("store"))
    monkeypatch.setattr(pool.journal, "drop_inherited", lambda: dropped.append("journal"))
    pool._init(1 << 30)
    assert calls == [(1, pool.signal.SIGKILL)]
    assert dropped == ["store", "journal"]  # the parent's store flocks and landing lock are let go in every worker


def test_a_warm_keyed_group_costs_one_store_read(tmp_path, monkeypatch):
    config, reads = _config(tmp_path), []
    real = pool.store.get
    monkeypatch.setattr(
        pool.store, "get", lambda cfg, kind, key, **kw: reads.append(kind) or real(cfg, kind, key, **kw)
    )
    items = [1, 2, 3, 4]
    with effort.command("check", []):
        assert pool.map(config, "units", _square, items, key=str) == [1, 4, 9, 16]
    assert reads.count("units") == 4  # the cold run asks for every item
    reads.clear()
    with effort.command("check", []):
        assert pool.map(config, "units", _square, items, key=str) == [1, 4, 9, 16]
        assert effort.counters()["units"] == (4, 0)  # all four still count as hits
    assert reads == ["units.group"]
    reads.clear()
    assert pool.map(config, "units", _square, [1, 2, 3, 5], key=str) == [1, 4, 9, 25]  # one new item: per-item reads
    assert reads.count("units") == 4


def test_whoever_ran_a_keyed_chunk_stores_its_results(tmp_path):
    config = _config(tmp_path)
    outcomes = pool._chunk(_square, [2, 3, 4], (config, "units", ["a", None, "c"]))
    assert [o[1] for o in outcomes] == [4, 9, 16]

    def get(key):
        return pool.store.get(config, "units", key, named=True)  # a pool key names its job's code itself

    assert pickle.loads(get("a")) == 4 and get("c") is not None
    assert get("b") is None  # an unkeyed item is not stored
    failed = pool._chunk(_fail, [1], (config, "units", ["f"]))
    assert failed[0][0] is False and get("f") is None
    assert [o[1] for o in pool._chunk(_square, [5], None)] == [25]


def test_a_worker_crash_names_the_type_and_the_innermost_unbake_frame(tmp_path):
    ok, error, _ = pool._job(sha256, "not bytes")  # hashlib refuses inside contracts.sha256
    assert not ok and error.__notes__[0].startswith("at contracts:sha256:")
    with pytest.raises(Refusal) as raised:
        pool._raise_failure(_config(tmp_path), "stage", 3, error)
    assert "TypeError" in raised.value.findings[0].reason and "at contracts:sha256:" in raised.value.findings[0].reason


def test_an_equal_key_from_a_changed_module_is_cold(tmp_path, monkeypatch, mocked_parallel):
    config, ran = _config(tmp_path), _RAN
    ran.clear()
    monkeypatch.setattr(pool, "code_identity", lambda function: "one")
    pool.map(config, "units", _record, [1, 2], key=_name)
    pool.map(config, "units", _record, [1, 2], key=_name)
    assert ran == [1, 2]  # the second run is warm
    monkeypatch.setattr(pool, "code_identity", lambda function: "two")
    pool.map(config, "units", _record, [1, 2], key=_name)
    assert ran == [1, 2, 1, 2]


def _stable_name(value):
    return str(value)


_stable_name.stable = True


def _clean(value):
    _RAN.append(value)
    return ()


def test_a_stable_key_survives_a_changed_module(tmp_path, monkeypatch, mocked_parallel):
    """A key that names every input of the job keeps its stored results across an install."""
    config, ran = _config(tmp_path), _RAN
    ran.clear()
    monkeypatch.setattr(pool, "code_identity", lambda function: "one")
    pool.map(config, "units", _clean, [1, 2], key=_stable_name)
    monkeypatch.setattr(pool, "code_identity", lambda function: "two")
    pool.map(config, "units", _clean, [1, 2], key=_stable_name)
    assert ran == [1, 2]  # warm under the other code


def _coded_name(value):
    return str(value)


_coded_name.coded = True


def test_a_coded_key_keeps_its_results_warm_across_an_install(tmp_path, monkeypatch, mocked_parallel):
    """A key that names the code its result depends on stores every result (not only clean ones) and pool adds no
    identity of the job's whole reach: another install leaves it warm."""
    config, ran = _config(tmp_path), _RAN
    ran.clear()
    monkeypatch.setattr(pool, "code_identity", lambda function: "one")
    pool.map(config, "units", _record, [1, 2], key=_coded_name)
    monkeypatch.setattr(pool, "code_identity", lambda function: "two")
    pool.map(config, "units", _record, [1, 2], key=_coded_name)
    assert ran == [1, 2]  # nonempty results, warm under the other code


def test_the_key_pass_dispatches_chunks_and_the_parent_computes_no_key(tmp_path, monkeypatch, mocked_parallel):
    config, chunks, parent = _config(tmp_path), [], []
    monkeypatch.setattr(sys.modules[__name__], "_KEY_CHUNKS", chunks)
    monkeypatch.setattr(pool, "_keys_job", _record_key_chunk)
    monkeypatch.setattr(pool, "_keys", lambda key, items, identity: parent.append(len(items)))
    items = list(range(640))
    pool.map(config, "units", _square, items, key=_name)
    assert sum(chunks) == 640 and 1 < len(chunks) < 640 and parent == []


_KEY_CHUNKS = []


def _record_key_chunk(item):
    items = item[1]
    _KEY_CHUNKS.append(len(items))
    return [str(value) for value in items]


def test_the_keys_of_a_few_items_are_computed_here_and_many_go_to_the_workers(tmp_path, monkeypatch):
    config = _config(tmp_path)
    asked = []
    monkeypatch.setattr(
        pool,
        "gather",
        lambda c, groups: (
            asked.append([g[0] for g in groups]) or [[[_name(item) for item in group[2][0][1]]] for group in groups]
        ),
    )
    few = [n for n in range(config.host.workers * pool._KEYS_INLINE)]
    many = [n for n in range(config.host.workers * pool._KEYS_INLINE + 1)]
    with effort.command("check", []):
        run = pool._Run("few", _square, few, _name)
        pool._resolve(config, [run])
        assert asked == [] and run.keys == [pool.digest((_name(n), pool.code_identity(_square))) for n in few]
        pool._resolve(config, [pool._Run("many", _square, many, _name)])
    assert asked == [["many.keys"]]  # a dispatch costs more than a handful of keys


def _slots_dir(runtime):
    path = Path(runtime) / pool._POOL
    path.mkdir(exist_ok=True)
    return path


def _holder(runtime, count, start=0):
    held = []
    for i in range(start, start + count):  # another command's slots: closing one is what a release does
        f = (_slots_dir(runtime) / f"slot-{i}.lock").open("a")
        pool.fcntl.flock(f, pool.fcntl.LOCK_EX)
        held.append(f)
    return held


def test_a_pool_takes_only_the_free_slots(tmp_path, monkeypatch):
    held = _holder(os.environ["XDG_RUNTIME_DIR"], 10)
    pool._acquire(12)
    assert len(pool._slots) == 2
    pool._drop_executor()
    assert pool._slots == []
    for f in held:
        f.close()


def test_a_pool_with_every_slot_taken_takes_whichever_frees_first(tmp_path):
    held = _holder(os.environ["XDG_RUNTIME_DIR"], 3)
    thread = threading.Thread(target=pool._acquire, args=(3,))
    thread.start()
    thread.join(0.2)
    assert thread.is_alive() and pool._slots == []
    held.pop().close()  # any one of them, not a slot picked beforehand
    thread.join(5)
    assert not thread.is_alive() and [f.name.rsplit("-", 1)[1] for f in pool._slots] == ["2.lock"]
    for f in held:
        f.close()


def test_slots_are_released_between_phases_and_the_executor_stays(tmp_path):
    with effort.command("check", []):
        assert pool.map(_config(tmp_path), "units", _square, [1, 2, 3, 4]) == [1, 4, 9, 16]
        assert pool._slots == [] and pool._executor is not None  # idle between pools: others may take the slots
        held = _holder(os.environ["XDG_RUNTIME_DIR"], _config(tmp_path).host.workers)  # all free for another command
        for f in held[:1]:
            f.close()  # one slot comes back: the next pool works with what it can take
        assert pool.map(_config(tmp_path), "units", _square, [5, 6]) == [25, 36]
        assert pool._slots == []
        for f in held[1:]:
            f.close()


def test_a_pool_that_waits_for_a_slot_runs_on_workers_that_wait_did_not_stop(tmp_path, monkeypatch):
    with effort.command("check", []):
        assert pool.map(_config(tmp_path), "units", _square, [1, 2, 3, 4]) == [1, 4, 9, 16]
        assert pool._slots == [] and pool._executor is not None  # idle between pools, its slots back with the host

        def wait_free(*args):  # every slot held by another command: the wait stops this process's idle workers
            pool._stop_workers()
            pool._slots.append(open(os.devnull))  # noqa: SIM115

        monkeypatch.setattr(pool, "_wait_free", wait_free)
        held = _holder(os.environ["XDG_RUNTIME_DIR"], _config(tmp_path).host.workers)
        assert pool.map(_config(tmp_path), "units", _square, [5, 6, 7]) == [25, 36, 49]  # not "after shutdown"
        for f in held:
            f.close()


def test_slots_are_released_after_the_executor_is_dropped(tmp_path, monkeypatch):
    class Executor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def submit(self, function):
            f = Future()
            f.set_result(None)
            return f

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(pool, "ProcessPoolExecutor", Executor)
    config = _config(tmp_path)
    executor = pool._get_executor(config)
    assert executor.kwargs["max_workers"] == len(pool._slots) == config.host.workers
    pool._drop_executor()
    held = _holder(os.environ["XDG_RUNTIME_DIR"], config.host.workers)  # all free again
    assert len(held) == config.host.workers


def _snapshot(tmp_path, digest="digest-1"):
    from unbake.contracts import Snapshot

    files = {f"src/m{i}.c": b"int m%d;\n" % i * 20 for i in range(2000)}  # a tree big enough to be worth sharing
    return Snapshot(_config(tmp_path), "head", files, {}, {}, digest)


def test_items_travel_without_the_snapshot_and_workers_read_the_shared_copy(tmp_path, monkeypatch):
    snapshot, needed = _snapshot(tmp_path), set()
    monkeypatch.setattr(pool, "_SHARED", {})
    stream = io.BytesIO()
    pool._Pickler(stream, needed).dump((_square, [(snapshot, 1), (snapshot, 2)]))
    assert len(stream.getvalue()) < 500 and len(needed) == 1 and list(pool._SHARED.values()) == [snapshot]
    function, items = pool._Unpickler(io.BytesIO(stream.getvalue())).load()
    assert function is _square and items[0][0] is snapshot and items[1][0] is snapshot  # no copy is made


def test_a_snapshot_the_warm_workers_were_not_forked_with_travels_in_one_spill_file_and_forks_nothing(
    tmp_path, monkeypatch
):
    made = []

    class Executor:
        def __init__(self, **kwargs):
            made.append(self)
            self.seen = set(pool._SHARED)  # what a fork at this moment would inherit

        def submit(self, function):
            f = Future()
            f.set_result(None)
            return f

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(pool, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(pool, "_SHARED", {})
    monkeypatch.setattr(pool, "_laid", [])
    config, needed = _config(tmp_path), set()
    one, two = _snapshot(tmp_path, "one"), _snapshot(tmp_path, "two")
    pool._Pickler(io.BytesIO(), needed).dump(one)
    first = pool._get_executor(config, frozenset(needed))
    assert pool._get_executor(config, frozenset(needed)) is first  # the same snapshot reuses the workers
    blobs = []
    for _ in range(3):  # three tasks of one dispatch, then the next dispatch: one file, written once
        stream, needed = io.BytesIO(), set()
        pickler = pool._Pickler(stream, needed, ship=pool._warm(config))
        pickler.dump((two, 1))
        blobs.append(stream.getvalue())
    assert pool._get_executor(config, frozenset(needed)) is first and len(made) == 1 and first.seen == {pool._key(one)}
    assert len(blobs[0]) < 500 and set(pool._SHARED) == {pool._key(one)}  # a reference, never the snapshot
    files = list((tmp_path / ".unbake/pool" / str(os.getpid())).iterdir())
    assert len(files) == 1 and len(pool._spilled) == 1
    back = pool._Unpickler(io.BytesIO(blobs[0])).load()[0]  # as a worker reads it: once
    assert back.digest == "two" and pool._Unpickler(io.BytesIO(blobs[1])).load()[0] is back
    pool._forget_spills()
    assert not list(files[0].parent.iterdir())


def test_real_forked_workers_see_the_parent_snapshot_unpickled(tmp_path):
    snapshot = _snapshot(tmp_path)
    with effort.command("check", []):
        found = pool.map(snapshot.config, "units", _digest_of, [(snapshot, 1), (snapshot, 2), (snapshot, 3)])
    assert [f[:2] for f in found] == [("digest-1", 1), ("digest-1", 2), ("digest-1", 3)] and all(f[2] for f in found)


def _digest_of(item):
    return item[0].digest, item[1], os.getpid() != _PARENT


_PARENT = os.getpid()


def _commands(runtime, count):
    held = []
    for i in range(count):  # other commands: each holds its own lock file
        f = (_slots_dir(runtime) / f"cmd-{90000 + i}.lock").open("a")
        pool.fcntl.flock(f, pool.fcntl.LOCK_EX)
        held.append(f)
    return held


@pytest.mark.parametrize(("others", "share"), [(0, 12), (1, 6), (3, 3), (5, 2), (11, 1)])
def test_a_command_takes_its_fair_share_of_the_slots(others, share):
    held = _commands(os.environ["XDG_RUNTIME_DIR"], others)
    pool._acquire(12)
    assert len(pool._slots) == share
    pool._drop_executor()
    for f in held:
        f.close()


@pytest.mark.parametrize(
    ("demands", "share"),
    [((1, 1, 1, 1, 1), 7), ((2, 1), 9), ((12, 12), 4), ((1, 12, 12), 4), ((3, 3, 12), 3), (("", "x"), 4)],
)
def test_a_share_peers_cannot_use_goes_to_the_command_that_can(demands, share, monkeypatch):
    """Five peers that each run one job on twelve slots leave seven to this command, never an idle split of two."""
    monkeypatch.setattr(os, "getpid", lambda: 7)  # an odd split ranks commands by lock file name: fix this one first
    held = _commands(os.environ["XDG_RUNTIME_DIR"], len(demands))
    for f, demand in zip(held, demands, strict=True):
        Path(f.name).write_text(f"{demand} 12" if isinstance(demand, int) else demand)
    pool._acquire(12)
    assert len(pool._slots) == share
    pool._drop_executor()
    for f in held:
        f.close()


def test_a_command_tells_its_peers_the_slots_it_can_use():
    pool._acquire(12, cap=3)
    mine = _slots_dir(os.environ["XDG_RUNTIME_DIR"]) / f"cmd-{os.getpid()}.lock"
    assert mine.read_text() == "3 12" and len(pool._slots) == 3  # what it can use now, and its budget
    pool._acquire(12, active=5, cap=2)  # chunks in flight keep their slots: they are part of what it uses
    assert mine.read_text() == "5 12"
    pool._drop_executor()


def test_a_finished_command_no_longer_counts():
    held = _commands(os.environ["XDG_RUNTIME_DIR"], 3)
    for f in held:
        f.close()  # their files remain but nobody holds them
    pool._acquire(12)
    assert len(pool._slots) == 12


def test_a_long_finished_commands_lock_file_is_removed_once_and_never_opened_again():
    held = _commands(os.environ["XDG_RUNTIME_DIR"], 3)
    for f in held:
        f.close()
    old = time.time() - 3600
    paths = sorted(_slots_dir(os.environ["XDG_RUNTIME_DIR"]).glob("cmd-9000*.lock"))
    for path in paths:
        os.utime(path, (old, old))
    opened = []
    real = os.open
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pool.os, "open", lambda path, *a, **k: (opened.append(Path(path).name), real(path, *a, **k))[1])
        pool._acquire(12)
        first = [n for n in opened if n.startswith("cmd-9000")]
        pool._acquire(12)
    second = [n for n in opened if n.startswith("cmd-9000")]
    assert len(first) == 3 and len(second) == 3  # the second acquire opened none of them
    assert not any(p.exists() for p in paths)
    assert len(pool._slots) == 12


def test_each_group_of_a_gather_is_recorded_with_its_own_span_not_the_gathers(tmp_path, mocked_parallel, monkeypatch):
    """A one-job group beside a big one must not show the whole dispatch as its own wall."""
    recorded = []
    monkeypatch.setattr(effort, "record_pool", lambda name, items, **kw: recorded.append((name, kw)))
    with effort.command("check", []):
        pool.gather(_config(tmp_path), [("small", _name, [1]), ("big", _square, list(range(80)))])
    spans = {name: (kw["start_ns"], kw["end_ns"]) for name, kw in recorded}
    assert all(end is not None and start <= end for start, end in spans.values())
    assert spans["small"] != spans["big"]  # each group has the span of its own chunks


def test_a_warm_group_is_recorded_as_instant(tmp_path, mocked_parallel, monkeypatch):
    recorded = []
    monkeypatch.setattr(effort, "record_pool", lambda name, items, **kw: recorded.append((name, kw)))
    with effort.command("check", []):
        pool.gather(_config(tmp_path), [("none", _name, []), ("big", _square, list(range(80)))])
    assert (
        dict(recorded)["none"]["end_ns"] == dict(recorded)["none"]["start_ns"] and dict(recorded)["none"]["jobs"] == 0
    )


def test_slots_grow_and_shrink_without_duplicate_locks_or_releasing_active_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(pool.fcntl, "flock", lambda *args: None)
    peer = _slots_dir(tmp_path) / "cmd-1.lock"
    peer.touch()
    busy = []
    monkeypatch.setattr(pool, "_busy", lambda path: bool(busy))
    pool._acquire(12)
    assert len(pool._slots) == len(set(pool._slots)) == 12
    busy.append(True)
    pool._acquire(12, active=9)
    assert len(pool._slots) == 9
    pool._acquire(12, active=4)
    assert len(pool._slots) == 6
    busy.clear()
    pool._acquire(12, active=4)
    assert len(pool._slots) == len(set(pool._slots)) == 12


def test_pool_workers_run_at_nice_10(monkeypatch):
    niced = []
    monkeypatch.setattr(pool.os, "nice", niced.append)
    monkeypatch.setattr(pool.resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(pool.ctypes, "CDLL", lambda name: type("L", (), {"prctl": lambda self, *a: None})())
    monkeypatch.setattr(pool, "_in_worker", False)
    pool._init(1 << 30)
    assert niced == [10]


def _host(tmp_path, slots=16, workers=16):
    config = _config(tmp_path)
    return replace(config.host, slots=slots, workers=workers)


def test_two_holders_split_the_host_budget_and_leave_no_file(tmp_path):
    host, runtime = _host(tmp_path), Path(os.environ["XDG_RUNTIME_DIR"])
    held = _commands(runtime, 1)  # the other holder
    with pool.cores(host) as mine:
        assert mine == 8 and len(pool._slots) == 8
        assert (runtime / pool._POOL / f"cmd-{os.getpid()}.lock").exists()
    assert pool._slots == [] and not (runtime / pool._POOL / f"cmd-{os.getpid()}.lock").exists()
    for f in held:
        f.close()


@pytest.mark.parametrize(("work", "yielded"), [(1, 1), (5, 5), (40, 16)])
def test_cores_never_exceeds_the_work(tmp_path, work, yielded):
    with pool.cores(_host(tmp_path), work) as j:
        assert j == yielded == len(pool._slots)


def test_workers_honour_host(tmp_path):
    host = _host(tmp_path, slots=6, workers=2)
    assert pool._budget(host) == pool._fan(host) == 2
    with pool.cores(host):
        assert len(pool._slots) == 2


def test_a_gather_in_one_process_gives_slots_back_but_never_those_in_flight(tmp_path):
    pool._acquire(12)
    held = _commands(os.environ["XDG_RUNTIME_DIR"], 5)  # six commands now: a share of 2
    pool._acquire(12, active=4)
    assert len(pool._slots) == 4
    pool._acquire(12, active=0)
    assert len(pool._slots) == 2
    for f in held:
        f.close()


def test_a_command_file_nobody_holds_or_gone_is_no_peer(tmp_path):
    runtime = Path(os.environ["XDG_RUNTIME_DIR"])
    (runtime / "unbake-cmd-1.lock").write_text("")
    assert not pool._busy(runtime / "unbake-cmd-1.lock") and not pool._busy(runtime / "unbake-cmd-2.lock")
    assert not (runtime / "unbake-cmd-2.lock").exists()  # probing a missing file does not create it


def test_waiting_for_a_slot_never_sleeps(tmp_path, monkeypatch):
    monkeypatch.setattr(pool.time, "sleep", lambda s: pytest.fail("a full host blocks on a lock, it does not poll"))
    test_a_pool_with_every_slot_taken_takes_whichever_frees_first(tmp_path)


def test_the_executor_forks_the_host_fan_once_and_never_again_for_more_work(tmp_path, monkeypatch):
    made = []

    class Executor:
        def __init__(self, **kwargs):
            made.append(kwargs["max_workers"])

        def submit(self, function):
            f = Future()
            f.set_result(None)
            return f

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(pool, "ProcessPoolExecutor", Executor)
    config = replace(_config(tmp_path), host=_host(tmp_path, slots=12, workers=9))
    first = pool._get_executor(config, work=3)
    assert made == [9]  # the host's worker count within its slots, not the first dispatch's three chunks
    assert pool._get_executor(config, work=3) is first and pool._get_executor(config, work=40) is first
    assert made == [9]


def test_the_shell_form_passes_the_held_share_and_runs_at_nice_10(tmp_path, monkeypatch):
    host_file = tmp_path / "host.toml"
    monkeypatch.setenv("UNBAKE_HOST", str(host_file))
    monkeypatch.setattr(configuration, "load_host", lambda p: _host(tmp_path, slots=8, workers=8))
    ran = []
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: ran.append((argv, kw)) or SimpleNamespace(returncode=3))
    held = _commands(os.environ["XDG_RUNTIME_DIR"], 1)
    assert pool.main(["--", "pytest", "-n", "{j}"]) == 3
    argv, kw = ran[0]
    assert argv == ["pytest", "-n", "4"] and kw["env"]["UNBAKE_CORES"] == "4"
    niced = []
    monkeypatch.setattr(pool.os, "nice", niced.append)
    kw["preexec_fn"]()
    assert niced == [10]
    for f in held:
        f.close()


def test_a_command_under_a_holder_uses_its_cores_and_takes_no_slots_of_its_own(tmp_path, monkeypatch):
    monkeypatch.setenv("UNBAKE_CORES", "4")
    held = _holder(os.environ["XDG_RUNTIME_DIR"], 16)  # every real slot is taken: a second acquisition would block
    with pool.cores(_host(tmp_path)) as j:
        assert j == 4
    with pool.cores(_host(tmp_path), 2) as j:
        assert j == 2
    assert not list(_slots_dir(os.environ["XDG_RUNTIME_DIR"]).glob("cmd-*"))
    for f in held:
        f.close()


def test_a_keyed_group_records_its_store_write_as_one_pool_sink_stage_and_its_cpu(tmp_path, monkeypatch):
    spent = iter(range(0, 1000, 10))  # every rusage read advances the clock by ten cpu seconds

    def usage(who):
        return SimpleNamespace(ru_utime=float(next(spent)), ru_stime=0.0, ru_maxrss=1)

    monkeypatch.setattr(pool.resource, "getrusage", usage)
    outcomes = pool._chunk(_square, [1, 2, 3], (_config(tmp_path), "units", ["a", "b", "c"]))
    stages = [r for o in outcomes for r in o[2]["records"] if r["path"][-1] == "pool.sink"]
    assert len(stages) == 1  # one write for the chunk, however many items
    assert all(not o[2]["records"] for o in outcomes[:-1])  # the chunk's records travel once, folded
    first, last = outcomes[0][2]["worker_cpu"], outcomes[-1][2]["worker_cpu"]
    assert first > 0 and last > first  # every sub-floor job counts, and the last envelope also carries the sink


def test_a_worker_closes_the_parents_slot_and_command_locks(tmp_path, monkeypatch):
    slot, mine = (tmp_path / "s").open("a"), (tmp_path / "c").open("a")
    monkeypatch.setattr(pool, "_slots", [slot])
    monkeypatch.setattr(pool, "_cmd", [mine])
    monkeypatch.setattr(pool.os, "nice", lambda n: 0)
    monkeypatch.setattr(pool.resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(pool.ctypes, "CDLL", lambda n: SimpleNamespace(prctl=lambda *a: 0))
    monkeypatch.setattr(pool, "_in_worker", False)
    pool._init(1 << 30)
    assert slot.closed and mine.closed


def _stall_run(tmp_path, monkeypatch, cpu, awaits=lambda pid, executor: True):
    class Executor:
        _processes = dict.fromkeys((1, 2))

        def submit(self, function, *args):
            return Future()  # never completes

    def timed_out(pending, return_when, timeout=None):
        return set(), set(pending)

    class Proc:
        def __init__(self, pid):
            self.pid = pid

        def status(self):
            return "sleeping"

        def children(self):
            return []

        def cpu_times(self):
            return SimpleNamespace(user=cpu(self.pid), system=0.0)

    drops, rounds = [], []
    monkeypatch.setattr(pool, "_get_executor", lambda config, needed=(), work=0: Executor())
    real = pool.psutil.Process
    monkeypatch.setattr(pool.psutil, "Process", lambda pid: Proc(pid) if pid in (1, 2) else real(pid))
    monkeypatch.setattr(pool, "_drop_executor", lambda: drops.append(True))
    monkeypatch.setattr(pool, "_awaits_work", awaits)

    def wait(pending, return_when, timeout=None):
        rounds.append(timeout)
        if len(rounds) > 5:
            raise BrokenProcessPool("test ends")
        return timed_out(pending, return_when)

    monkeypatch.setattr(pool, "wait", wait)
    with effort.command("check", []), pytest.raises(Refusal) as caught:
        pool.map(_config(tmp_path), "units", _square, [1, 2])
    return caught.value.findings[0].key, drops, len(rounds)


def test_stalled_pool_fails_loudly_and_frees_slots(tmp_path, monkeypatch):
    key, drops, rounds = _stall_run(tmp_path, monkeypatch, lambda pid: 1.0)
    assert (key, drops, rounds) == ("worker.stalled", [True], 2)
    assert pool._slots == []


def test_a_worker_sleeping_on_a_lock_is_not_stalled(tmp_path, monkeypatch):
    key, drops, rounds = _stall_run(tmp_path, monkeypatch, lambda pid: 1.0, lambda pid, executor: pid != 2)
    assert (key, drops, rounds) == ("worker.crash", [True], 6)


@pytest.mark.parametrize(
    ("syscall", "expected"),
    [
        ("0 0x5 0x1", True),
        ("202 0x7000 0x109", True),
        ("0 0x9 0x1", False),
        ("202 0x8000 0x0", False),
        ("73 0x5 0x6", False),
        ("running", False),
    ],
)
def test_awaits_work_reads_the_blocking_syscall(monkeypatch, syscall, expected):
    monkeypatch.setattr(pool, "_READ_FUTEX", (0, 202))
    monkeypatch.setattr(pool, "Path", lambda path: SimpleNamespace(read_text=lambda: syscall))
    calls = SimpleNamespace(
        _reader=SimpleNamespace(fileno=lambda: 5), _rlock=SimpleNamespace(_semlock=SimpleNamespace(handle=0x7000))
    )
    assert pool._awaits_work(7, SimpleNamespace(_call_queue=calls)) is expected


def test_busy_worker_keeps_the_wait_going(tmp_path, monkeypatch):
    ticks = iter(range(100))
    key, drops, rounds = _stall_run(tmp_path, monkeypatch, lambda pid: next(ticks) if pid == 2 else 1.0)
    assert (key, drops, rounds) == ("worker.crash", [True], 6)


def _shell_pool(monkeypatch, tmp_path, *, share=4):
    import subprocess

    calls, acquired = [], []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(returncode=7))
    monkeypatch.setattr(configuration, "load_host", lambda path: _host(tmp_path))
    monkeypatch.setattr(
        pool,
        "_acquire",
        lambda want, active=0, cap=0: acquired.append(cap) or pool._slots.extend([None] * min(share, cap or share)),
    )
    monkeypatch.setattr(pool, "_release", lambda: pool._slots.clear())
    return calls, acquired


def test_a_command_never_told_a_count_takes_no_slot(tmp_path, monkeypatch):
    calls, acquired = _shell_pool(monkeypatch, tmp_path)
    assert pool.main(["--", "echo", "x"]) == 7
    assert acquired == [] and len(calls) == 1 and calls[0][0] == ["echo", "x"] and "env" not in calls[0][1]


def test_a_command_told_the_count_holds_the_share_and_gets_it(tmp_path, monkeypatch):
    calls, acquired = _shell_pool(monkeypatch, tmp_path)
    pool.main(["--", "make", "-j{j}"])
    assert acquired == [0] and calls[0][0] == ["make", "-j4"] and calls[0][1]["env"]["UNBAKE_CORES"] == "4"


def test_work_caps_the_share_of_a_counted_command(tmp_path, monkeypatch):
    calls, acquired = _shell_pool(monkeypatch, tmp_path)
    pool.main(["--work", "3", "--", "tool", "-j{j}"])
    assert acquired == [3] and calls[0][0] == ["tool", "-j3"]


def test_a_job_under_memory_pressure_leaves_only_the_memo_it_started_with(monkeypatch):
    monkeypatch.setattr(effort, "_pressure", lambda: (900, 1000))
    monkeypatch.setattr(effort, "_memo", {("base", 0): 0})

    def work(value):
        effort.memo(("overlay", value), lambda: value)
        return value

    outcomes = [pool._job(work, i) for i in (1, 2, 3)]
    assert [o[1] for o in outcomes] == [1, 2, 3]
    assert set(effort._memo) == {("base", 0)}


def test_a_refused_host_file_runs_the_command_alone_at_nice_ten(monkeypatch, tmp_path):
    calls, acquired = _shell_pool(monkeypatch, tmp_path)

    def refuse(path):
        raise Refusal(Finding("config.schema", reason="missing"))

    monkeypatch.setattr(configuration, "load_host", refuse)
    assert pool.main(["--work", "3", "--", "tool", "-j{j}"]) == 7
    assert acquired == [] and len(calls) == 1 and calls[0][0] == ["tool", "-j1"]
    assert calls[0][1]["env"]["UNBAKE_CORES"] == "1" and calls[0][1]["preexec_fn"]


def test_an_unreadable_cache_entry_is_a_miss_in_a_keyed_group(cfg) -> None:
    pool.store.put(cfg, "g", "k0", b"\x01 not a pickle", named=True)
    pool.store.put(cfg, "g", "k1", pickle.dumps(7, protocol=5), named=True)
    run = pool._Run("g", abs, [1, 2], lambda item: None)
    run.resolve(cfg, ["k0", "k1"])
    assert run.todo == [0] and run.outcomes[1] == (True, 7, {})


def test_every_keyed_pool_key_carries_the_identity_of_the_job_function() -> None:
    def one(item):
        return item

    def other(item):
        return item + 1

    keys = [pool._keys(str, [1], pool.code_identity(f)) for f in (one, other)]
    assert keys[0] != keys[1]  # a job whose result changed shape never reads the old entries


def test_a_free_host_slot_is_taken_for_a_running_build_and_none_when_all_are_held(tmp_path, monkeypatch):
    (other,) = _holder(os.environ["XDG_RUNTIME_DIR"], 1, start=2)  # a peer holds the third
    pool._acquire(3, cap=1)
    (first,) = pool._slots
    extra = pool._extra_slot()
    assert extra is not None and extra.name.endswith("slot-1.lock") and pool._slots == [first, extra]
    assert pool._extra_slot() is None and len(pool._slots) == 2
    pool._fed.clear()
    other.close()


def test_the_feeder_writes_one_job_token_per_slot_that_frees(tmp_path, monkeypatch):
    read, write = os.pipe()
    freed = iter(["s1", None, "s2", None])
    monkeypatch.setattr(pool, "_extra_slot", lambda: next(freed, None))
    monkeypatch.setattr(pool, "_dir", [tmp_path])
    monkeypatch.setattr(pool, "_cmd", [(tmp_path / "cmd-1.lock").open("a")])
    attempts = []

    def until(pool_dir, mine, attempt, stop=None, also=None, interval=None):  # two checks, then make returns
        assert interval == pool._REBALANCE
        attempts.extend([attempt(), attempt()])

    monkeypatch.setattr(pool, "_until", until)
    monkeypatch.setattr(pool, "_shed", lambda spare, taken, last: None)
    monkeypatch.setattr(pool, "_idle", lambda spare: 0)  # make consumed every earlier token
    taken = []
    pool._feed(write, -1, -1, taken)
    assert attempts == [False, False]  # the feeder never ends on its own, only on its stop
    os.close(write)
    assert os.read(read, 10) == b"++" and taken == ["s1", "s2"]
    os.close(read)


def test_a_make_build_runs_on_a_jobserver_fed_from_the_slot_pool(tmp_path, monkeypatch):
    calls, _ = _shell_pool(monkeypatch, tmp_path)
    monkeypatch.setattr(pool, "_dir", [tmp_path])
    monkeypatch.setattr(pool, "_cmd", [(tmp_path / "cmd-1.lock").open("a")])
    monkeypatch.setattr(pool, "_feed", lambda tokens, spare, stop, taken: None)
    pool.main(["--", "/usr/bin/make", "-j{j}", "check"])
    command, options = calls[0]
    assert command[-2:] == ["/usr/bin/make", "check"]  # no fixed -j: the first grant is only where the build starts
    assert command[1:3] == ["-n", "10"] and "preexec_fn" not in options  # the feeder thread runs: no preexec_fn
    flags = options["env"]["MAKEFLAGS"]
    assert flags.startswith("-j4 --jobserver-auth=") and options["pass_fds"]
    assert options["env"]["UNBAKE_CORES"] == "4"


def test_the_jobserver_holds_the_grant_as_tokens_and_closes_its_pipe(tmp_path, monkeypatch):
    _shell_pool(monkeypatch, tmp_path, share=3)
    monkeypatch.setattr(pool, "_dir", [tmp_path])
    monkeypatch.setattr(pool, "_cmd", [(tmp_path / "cmd-1.lock").open("a")])
    monkeypatch.setattr(pool, "_feed", lambda tokens, spare, stop, taken: None)
    with pool.jobserver(_host(tmp_path)) as (j, env, fds):
        read, write = fds
        os.set_blocking(read, False)
        assert j == 3 and os.read(read, 10) == b"++"  # make holds the first token itself
        assert env == {"UNBAKE_CORES": "3", "MAKEFLAGS": f"-j3 --jobserver-auth={read},{write}"}
    with pytest.raises(OSError):
        os.fstat(read)


def test_exactly_the_slots_the_feeder_added_go_back_when_make_returns(tmp_path, monkeypatch):
    _shell_pool(monkeypatch, tmp_path, share=2)
    files = [(tmp_path / f"slot{i}").open("a") for i in range(3)]
    monkeypatch.setattr(pool, "_dir", [tmp_path])
    monkeypatch.setattr(pool, "_cmd", [(tmp_path / "cmd-1.lock").open("a")])

    def feed(tokens, spare, stop, taken):
        fcntl.flock(files[1], fcntl.LOCK_EX)
        pool._slots.append(files[1])  # the feeder takes one more slot while make runs
        taken.append(files[1])

    monkeypatch.setattr(pool, "_feed", feed)
    with pool.cores(_host(tmp_path)):  # an enclosing block keeps its grant
        with pool.jobserver(_host(tmp_path)):
            fcntl.flock(files[2], fcntl.LOCK_EX)
            pool._slots.append(files[2])  # another thread of this process takes one after the feeder
        assert files[1] not in pool._slots and files[2] in pool._slots  # only the feeder's went back
        peer = (tmp_path / "slot1").open("a")
        fcntl.flock(peer, fcntl.LOCK_EX | fcntl.LOCK_NB)  # free for a peer again before the command ends
        peer.close()


def test_the_jobserver_without_a_slot_pool_pins_j_to_the_grant(tmp_path, monkeypatch):
    _shell_pool(monkeypatch, tmp_path, share=2)
    monkeypatch.setattr(pool, "_dir", [])
    with pool.jobserver(_host(tmp_path)) as (j, env, fds):
        assert (j, env, fds) == (2, {"UNBAKE_CORES": "2", "MAKEFLAGS": "-j2"}, ())


def test_a_held_slot_shows_up_as_its_own_wait_stage_naming_the_holders(tmp_path, monkeypatch):
    stages, recorded, peer = [], [], 424242

    @contextmanager
    def stage(name):
        stages.append(name)
        yield SimpleNamespace(add=lambda items=0, findings=(): recorded.extend(f.reason for f in findings))

    monkeypatch.setattr(pool.effort, "stage", stage)
    monkeypatch.setattr(pool, "_wait_free", lambda *a: pool._slots.append(open(os.devnull)))  # noqa: SIM115
    monkeypatch.setattr(pool, "_slots", [])
    monkeypatch.setattr(pool, "_cmd", [])
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("UNBAKE_CORES", raising=False)
    held = [(_slots_dir(tmp_path) / f"slot-{i}.lock").open("a") for i in range(2)]
    held.append((_slots_dir(tmp_path) / f"cmd-{peer}.lock").open("a"))  # the live command that holds them
    for f in held:
        fcntl.flock(f, fcntl.LOCK_EX)
    pool._acquire(2)
    assert stages == ["pool.wait_slot"]
    assert recorded == [f"held by pids {peer}"]


def test_a_share_short_of_the_work_shows_in_the_tree_once_naming_the_holders(tmp_path, monkeypatch):
    stages, recorded = [], []

    @contextmanager
    def stage(name):
        stages.append(name)
        yield SimpleNamespace(add=lambda items=0, findings=(): recorded.extend(f.reason for f in findings))

    monkeypatch.setattr(pool.effort, "stage", stage)
    monkeypatch.setattr(pool, "_slots", [])
    monkeypatch.setattr(pool, "_cmd", [])
    monkeypatch.setattr(pool, "_short", [])
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("UNBAKE_CORES", raising=False)
    held = [(_slots_dir(tmp_path) / f"slot-{i}.lock").open("a") for i in (0, 1)]
    for f in held:
        fcntl.flock(f, fcntl.LOCK_EX)
    pool._acquire(4, cap=4)
    pool._acquire(4, pool._flying, 4)  # asked again with the same shortfall: reported once
    assert stages == ["pool.short_slots"] and recorded == ["2 of 4 slots; the rest are held by pids none live"]


def test_a_free_host_grants_make_every_worker_never_one_job(tmp_path):
    with pool.cores(_host(tmp_path, slots=6, workers=3)) as j:  # the -j for make: host.workers, not -j1
        assert j == 3


def test_every_key_function_given_to_a_pool_map_is_a_module_function_that_pickles():
    import ast

    import unbake

    seen = 0
    for path in sorted(p for p in Path(unbake.__file__).parent.glob("*.py") if p.stem != "__main__"):
        module = importlib.import_module(f"unbake.{path.stem}")
        for call in (n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call)):
            keyed = [
                k.value for k in call.keywords if k.arg == "key" and getattr(call.func, "attr", "") in {"map", "gather"}
            ]
            for value in keyed:
                assert isinstance(value, ast.Name), f"{path.name}:{call.lineno} key must be a named module function"
                function = getattr(module, value.id)
                assert pickle.loads(pickle.dumps(function)) is function
                seen += 1
    assert seen >= 1


def test_a_pool_holds_no_slot_its_workers_cannot_use_and_never_waits_while_holding(
    tmp_path, monkeypatch, mocked_parallel
):
    """Three holders on one host: this one holds a share of 4 of the 12 workers it forked, so a long gather keeps 4
    chunks in flight on the same workers, never forks again, and holds slots only for chunks that remain; it never
    blocks while holding."""
    peers = _commands(os.environ["XDG_RUNTIME_DIR"], 2)
    config = replace(_config(tmp_path), host=_host(tmp_path, slots=12, workers=12))
    monkeypatch.setattr(pool, "_REBALANCE", 0.0)  # look at the peers after every chunk
    monkeypatch.setattr(pool, "_size", 12)  # the executor forked the host fan
    pool._acquire(12)  # an earlier phase took the whole share of 4
    assert len(pool._slots) == 4
    held, forks = [], []
    real = pool._acquire

    def acquire(want, active=0, cap=0):
        real(want, active, cap)
        held.append(len(pool._slots))

    monkeypatch.setattr(pool, "_acquire", acquire)
    monkeypatch.setattr(pool, "_get_executor", lambda *a, **k: forks.append(1) or mocked_parallel.executor())
    monkeypatch.setattr(pool, "_wait_free", lambda *a: pytest.fail("waited for a slot while holding some"))
    with effort.command("check", []):
        assert pool.map(config, "units", _square, list(range(8))) == [v * v for v in range(8)]
    assert forks == [1]  # the gather asks for the executor once: no refork as the share moves
    assert max(held) <= 4 and held[-1] <= 2  # never beyond the share, and fewer as the chunks run out
    assert max(mocked_parallel) <= 4
    for f in peers:
        f.close()


def test_a_rebalance_never_gives_back_a_slot_a_make_feeder_took(monkeypatch):
    pool._acquire(4, 0, 2)
    taken = pool._extra_slot()
    assert taken is not None and taken in pool._fed
    monkeypatch.setattr(pool, "_wait_free", lambda *a: pytest.fail("waited"))
    pool._acquire(4, 0, 1)  # the share shrinks to one slot: only the pool's own are returned
    assert taken in pool._slots
    pool._fed.clear()


def _verdict(value):
    _RAN.append(value)
    return () if value % 2 else ("fail",)


def test_a_stable_key_stores_only_clean_results(tmp_path, monkeypatch, mocked_parallel):
    config, ran = _config(tmp_path), _RAN
    ran.clear()
    monkeypatch.setattr(pool, "code_identity", lambda function: "one")
    pool.map(config, "units", _verdict, [1, 2], key=_stable_name)
    monkeypatch.setattr(pool, "code_identity", lambda function: "two")  # a changed module, an unchanged key
    pool.map(config, "units", _verdict, [1, 2], key=_stable_name)
    assert ran == [1, 2, 2]  # the clean unit dispatched 0 times; the failing one ran again
    pool.map(config, "units", _verdict, [2], key=_stable_name)
    assert ran == [1, 2, 2, 2]  # and again under an unchanged module


def test_a_worker_is_refused_memory_only_well_above_the_budget_that_sizes_the_pool(monkeypatch):
    limits, marks = [], []
    monkeypatch.setattr(pool.resource, "setrlimit", lambda which, value: limits.append(value))
    monkeypatch.setattr(pool.effort, "mark_fork", lambda data, budget=0: marks.append((data, budget)))
    monkeypatch.setattr(pool.os, "nice", lambda n: 0)
    monkeypatch.setattr(pool.ctypes, "CDLL", lambda name: type("L", (), {"prctl": lambda self, *a: 0})())
    monkeypatch.setattr(pool, "_slots", [])
    monkeypatch.setattr(pool, "_cmd", [])
    monkeypatch.setattr(pool, "_in_worker", False)
    pool._init(1 << 30)
    ((soft, hard),) = limits
    (inherited, budget) = marks[0]
    assert budget == 1 << 30 and soft == hard == inherited + 4 * (1 << 30)


def test_one_pool_serves_every_round_and_holds_only_the_snapshot_it_forked_with(tmp_path, monkeypatch):
    made = []

    class Executor:
        def __init__(self, **kwargs):
            made.append(set(pool._SHARED))

        def submit(self, function):
            f = Future()
            f.set_result(None)
            return f

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(pool, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(pool, "_SHARED", {})
    config = _config(tmp_path)
    for name in ("one", "two", "three"):  # one snapshot per round
        needed: set[str] = set()
        pool._Pickler(io.BytesIO(), needed, ship=pool._warm(config)).dump(_snapshot(tmp_path, name))
        pool._get_executor(config, frozenset(needed))
    assert [len(m) for m in made] == [1]  # forked once; no later round's map is held by the parent or copied
    assert len(pool._SHARED) == 1 and len(pool._spilled) == 2
    pool._forget_spills()


def _peer(runtime, pid, budget, slots):
    """Another live command of the host: its lock file (what it can use, its budget) and the slots it holds."""
    held = [(_slots_dir(runtime) / f"cmd-{pid}.lock").open("a")]
    fcntl.flock(held[0], fcntl.LOCK_EX)
    Path(held[0].name).write_text(f"{budget} {budget}")
    return held + _holder(runtime, len(slots), start=slots[0])


def _waiter(monkeypatch, want):
    """`_acquire(want)` on a thread, with every probe and every sleep counted: (thread, probes, sleeping)."""
    probes, sleeping = [], threading.Event()
    real_take, real_select = pool._take, pool.select.select
    monkeypatch.setattr(pool, "_take", lambda *a: probes.append(a) or real_take(*a))

    def select(*args):
        sleeping.set()
        return real_select(*args)

    monkeypatch.setattr(pool.select, "select", select)
    thread = threading.Thread(target=pool._acquire, args=(want,), daemon=True)
    thread.start()
    return thread, probes, sleeping


def test_a_small_budget_takes_a_free_slot_a_larger_budget_made_instead_of_waiting_on_held_ones():
    """R 19:16: a report with a budget of 3 waited on slots 0-2 that a 14-slot command held while slots 3-13 were free.
    Every command sees as many slot files as the largest live budget, so it takes slot 4 at once."""
    runtime = os.environ["XDG_RUNTIME_DIR"]
    held = _peer(runtime, 90001, 14, range(4))
    pool._acquire(3)
    assert [Path(f.name).name for f in pool._slots] == ["slot-4.lock", "slot-5.lock", "slot-6.lock"]
    for f in held:
        f.close()


def test_a_full_host_waiter_holds_no_slot_wait_and_takes_the_slot_a_release_frees(monkeypatch):
    runtime = os.environ["XDG_RUNTIME_DIR"]
    held = _peer(runtime, 90001, 3, range(3))
    thread, probes, sleeping = _waiter(monkeypatch, 3)
    assert sleeping.wait(5) and thread.is_alive() and pool._slots == []
    blocked = [
        line for line in Path("/proc/locks").read_text().splitlines() if "->" in line and f" {os.getpid()} " in line
    ]
    assert blocked == []  # nobody queues on one particular slot: the waiter sleeps on the host's events
    asked = len(probes)
    assert asked == 2  # the acquire's own pass and one before sleeping: no probe without an event
    held.pop().close()  # slot 2 is released (a closed write descriptor, like every release)
    thread.join(5)
    assert not thread.is_alive() and [Path(f.name).name for f in pool._slots] == ["slot-2.lock"]
    assert len(probes) == asked + 2  # the woken probe, then the rest of the share
    for f in held:
        f.close()


def test_a_holder_that_dies_without_a_release_wakes_the_waiter_through_its_pidfd(monkeypatch):
    runtime = os.environ["XDG_RUNTIME_DIR"]
    held = _peer(runtime, 90001, 2, range(2))
    died, dies = os.pipe()  # the peer's pidfd: readable once it is gone
    monkeypatch.setattr(pool.os, "pidfd_open", lambda pid: os.dup(died) if pid == 90001 else pytest.fail(pid))
    thread, _, sleeping = _waiter(monkeypatch, 2)
    assert sleeping.wait(5) and thread.is_alive()
    fcntl.flock(held[1], fcntl.LOCK_UN)  # the kernel drops a dead process's locks without any write-close
    os.write(dies, b"x")
    thread.join(5)
    assert not thread.is_alive() and [Path(f.name).name for f in pool._slots] == ["slot-0.lock"]
    for fd in (died, dies):
        os.close(fd)
    for f in held:
        f.close()


def test_a_release_and_a_shrunk_share_each_wake_the_waiters_once(monkeypatch):
    signals = []
    real = pool._signal
    monkeypatch.setattr(pool, "_signal", lambda path: signals.append(path) or real(path))
    pool._acquire(12)
    peers = _commands(os.environ["XDG_RUNTIME_DIR"], 1)
    pool._acquire(12)  # a second command arrived: half the slots go back in one wake
    assert len(pool._slots) == 6 and len(signals) == 1
    pool._release()
    assert len(signals) == 2 and (_slots_dir(os.environ["XDG_RUNTIME_DIR"]) / "wake").exists()
    for f in peers:
        f.close()


def _other_protocol(monkeypatch, lock, capsys=None):
    path = Path(os.environ["XDG_RUNTIME_DIR"]) / lock
    path.parent.mkdir(exist_ok=True)
    other = path.open("a")
    fcntl.flock(other, fcntl.LOCK_EX)
    sleeping, real = threading.Event(), pool.select.select

    def select(*args):
        sleeping.set()
        return real(*args)

    monkeypatch.setattr(pool.select, "select", select)
    thread = threading.Thread(target=pool._acquire, args=(4,), daemon=True)
    thread.start()
    assert sleeping.wait(5) and thread.is_alive() and pool._slots == []  # no slot while protocol 1 may book them
    return path, other, thread


@pytest.mark.parametrize("lock", ["unbake-cmd-77.lock", "unbake-pool-1/cmd-77.lock", "unbake-pool-3/cmd-77.lock"])
def test_a_command_waits_for_live_commands_of_another_slot_protocol_to_release(monkeypatch, lock):
    said = []
    monkeypatch.setattr(pool, "_pidfd", lambda pid: None)
    monkeypatch.setattr(pool.effort, "say", said.append)
    stages = []
    real_stage = pool.effort.stage

    @contextmanager
    def stage(name):
        stages.append([name, "open"])
        with real_stage(name) as span:
            yield span
        stages[-1][1] = "closed"

    monkeypatch.setattr(pool.effort, "stage", stage)
    path, other, thread = _other_protocol(monkeypatch, lock)
    assert said == ["unbake: waiting for pids 77 hold host slots under another slot protocol than 2 to finish"]
    assert stages == [["pool.protocol", "open"]]  # the wait shows as its own stage while it lasts
    path.unlink()  # what a release of protocol 1 does
    other.close()
    thread.join(5)
    assert not thread.is_alive() and len(pool._slots) == 4
    assert stages == [["pool.protocol", "closed"]]


def test_a_command_of_another_protocol_that_dies_wakes_the_waiter_through_its_pidfd(monkeypatch):
    died, dies = os.pipe()
    monkeypatch.setattr(pool.os, "pidfd_open", lambda pid: os.dup(died) if pid == 77 else pytest.fail(pid))
    _, other, thread = _other_protocol(monkeypatch, "unbake-cmd-77.lock")
    fcntl.flock(other, fcntl.LOCK_UN)  # the kernel drops a dead command's lock: no write-close, no unlink
    os.write(dies, b"x")
    thread.join(5)
    assert not thread.is_alive() and len(pool._slots) == 4
    for fd in (died, dies):
        os.close(fd)
    other.close()


def test_a_jobs_native_cpu_counts_every_program_it_waited_for_even_under_an_inline_pool(monkeypatch):
    """A nested pool runs inline in a worker and keeps only its slow records: its programs' CPU still counts."""
    real = resource.getrusage
    reaped = iter([0.0, 5.0])

    def getrusage(who):
        if who == resource.RUSAGE_CHILDREN:
            value = next(reaped)
            return SimpleNamespace(ru_utime=value, ru_stime=0.0)
        return real(who)

    monkeypatch.setattr(pool.resource, "getrusage", getrusage)
    ok, _, envelope = pool._job(lambda value: value, 1)
    assert ok and envelope["native_cpu"] == 5.0


def test_a_jobs_native_cpu_includes_what_a_nested_pool_record_rolled_up(monkeypatch):
    def nested(value):
        effort.record_pool(
            "inner",
            1,
            jobs=1,
            admitted=1,
            start_ns=0,
            end_ns=0,
            envelopes=[
                {
                    "worker_cpu": 0,
                    "native_cpu": 3,
                    "rss": 0,
                    "pid": 1,
                    "start_ns": 0,
                    "end_ns": 0,
                    "cache": {},
                    "records": [],
                }
            ],
        )
        return value

    ok, _, envelope = pool._job(nested, 1)
    assert ok and envelope["native_cpu"] >= 3


def test_a_waiter_woken_by_a_release_takes_its_whole_share_not_one_slot(monkeypatch):
    """R proof 19:36: a 3-slot command woke when a holder released 4 slots and took 1, leaving 3 idle."""
    runtime = os.environ["XDG_RUNTIME_DIR"]
    held = _peer(runtime, 90001, 12, range(12))
    thread, _, sleeping = _waiter(monkeypatch, 3)
    assert sleeping.wait(5) and thread.is_alive()
    for f in held:
        f.close()  # the holder ends: every slot frees at once
    thread.join(5)
    assert not thread.is_alive() and len(pool._slots) == 3


_PARSES: list[int] = []


def _parse_on(item):
    """A job that parses one shared header per content, as the views and proofs of a round do."""
    snapshot, _ = item
    effort.memo(("words", "shared header text"), lambda: _PARSES.append(1) or 1)
    return snapshot.digest, len(snapshot.overlays), os.getpid(), len(_PARSES)


def test_every_chunk_is_ready_at_once_so_the_workers_stay_full_while_work_remains(
    tmp_path, monkeypatch, mocked_parallel
):
    config = replace(_config(tmp_path), host=_host(tmp_path, slots=4, workers=4))
    monkeypatch.setattr(pool, "_size", 4)  # the fan the executor forked
    with effort.command("check", []):
        assert pool.map(config, "units", _square, list(range(64))) == [v * v for v in range(64)]
    assert mocked_parallel[:4] == [1, 2, 3, 4] and set(mocked_parallel[3:]) == {4}  # in flight == workers to the end


def test_a_spilled_tree_carries_the_parents_indexes_so_warm_workers_never_build_them(tmp_path, monkeypatch):
    """Forked workers inherited the catalog the parent warmed; warm ones read it (and the scans the parent made since
    the last file) from the spill file instead of building them each."""
    scans: dict[str, str] = {"text-1": "scan-1"}
    monkeypatch.setattr(pool, "_carried", {"scans": scans})
    monkeypatch.setattr(pool, "_sent", {})
    monkeypatch.setattr(pool, "_SHARED", {})
    monkeypatch.setattr(pool, "_laid", [])
    with effort.command("check", []):
        effort.memo(("catalog", "root", "pins-1"), lambda: {"Vec3": ("include/m.h", 1, "typedef ...")})
        blobs = []
        for name in ("one", "two"):
            stream = io.BytesIO()
            pool._Pickler(stream, set(), ship=True).dump(_snapshot(tmp_path, name))
            blobs.append(stream.getvalue())
            scans["text-2"] = "scan-2"  # the parent scans more before the next round's file
        monkeypatch.setattr(effort, "_memo", {})  # a worker that holds none of it
        monkeypatch.setattr(pool, "_carried", {"scans": {}})
        built = []
        pool._Unpickler(io.BytesIO(blobs[0])).load()
        assert effort.memo(("catalog", "root", "pins-1"), lambda: built.append(1)) == {
            "Vec3": ("include/m.h", 1, "typedef ...")
        }
        pool._Unpickler(io.BytesIO(blobs[1])).load()
        assert built == [] and pool._carried["scans"] == {"text-1": "scan-1", "text-2": "scan-2"}
    pool._forget_spills()


def _laid_tree(item):
    snapshot, _ = item
    return snapshot.digest, dict(snapshot.overlays)


@pytest.mark.parametrize(
    "change", ["relaid", "new_capture", "same_writes_new_base", "new_middle", "large_writes", "unchanged"]
)
def _versions_read(item):
    return effort.memo(("versions.read", "split"), lambda: "read now")


def test_a_warm_worker_forgets_what_the_parent_forgot_before_the_next_dispatch(tmp_path, monkeypatch, warm):
    # setup's extract rewrote the asm and forgot versions.read in the parent; warm workers kept the facts they read
    # before and laid the next plan's overlays from them: 'an overlay laid again in a worker has digest'.
    config = _config(tmp_path)
    monkeypatch.setattr(effort, "_epochs", {})
    with effort.command("check", []):
        effort.memo(("versions.read", "split"), lambda: "before extract")  # in the memo the workers fork holding
        assert pool.map(config, "units", _versions_read, [1, 2]) == ["before extract"] * 2
        effort.forget("versions.read")  # extract: the files these values were read from are rewritten
        assert pool.map(config, "units", _versions_read, [1, 2]) == ["read now"] * 2
        assert pool.map(config, "units", _versions_read, [1, 2]) == ["read now"] * 2  # one forget, followed once
    assert warm.forks == 1 and warm.dispatches == 6


def test_follow_forgets_only_the_kinds_the_parent_forgot_since(monkeypatch):
    monkeypatch.setattr(effort, "_memo", {("a", 1): 1, ("b", 1): 1})
    monkeypatch.setattr(effort, "_epochs", {"a": 2, "b": 1})
    monkeypatch.setattr(effort, "_applied", {"a": 2, "b": 0})
    effort.follow({"a": 2, "b": 1})
    assert effort._memo == {("a", 1): 1} and effort._applied == {"a": 2, "b": 1}
    effort._memo[("b", 2)] = 1
    effort.follow({"a": 2, "b": 1})  # nothing new: nothing forgotten
    assert ("b", 2) in effort._memo


def _two_stages(value):
    with effort.stage("native.objects"), effort.stage("view.get"):
        sum(i * i for i in range(20000))  # cpu the fold must keep, job by job
    return value


def test_a_chunk_ships_every_jobs_records_folded_by_path_once(tmp_path):
    # setup layout.ownership: 5,439 jobs each under the 2 s floor shipped no record, so 1,258 of 1,594 busy
    # worker-seconds had no stage to name what they waited on. A chunk now ships all of them, folded by path.
    outcomes = pool._chunk(_two_stages, list(range(50)), None)
    shipped = outcomes[-1][2]["records"]
    assert all(not o[2]["records"] for o in outcomes[:-1])
    by_path = {tuple(r["path"]): r for r in shipped}
    assert {p[-1] for p in by_path} == {"job", "native.objects", "view.get", "(own)"}
    assert by_path[("job", "native.objects")]["spans"] == 50 and by_path[("job",)]["spans"] == 50
    jobs = by_path[("job",)]
    total = sum(o[2]["worker_cpu"] for o in outcomes)
    assert jobs["parent_cpu_seconds"] >= 0.5 * total  # a workers own cpu is summed, not the first jobs alone
    view = by_path[("job", "native.objects", "view.get")]
    assert view["parent"] == by_path[("job", "native.objects")]["span"]


_CLOCK = [0.0]


def _tick(value):
    """A job of one second on the test clock."""
    _CLOCK[0] += 1.0
    return value


def test_a_pool_records_the_slot_seconds_it_held_as_its_share_shrank(tmp_path, monkeypatch, mocked_parallel):
    """Move round 1 kept 6.6 of the 12 workers it was once admitted busy while another command took half the host:
    the pool record carries the slots it held over time, so its fill is judged against that share."""
    _CLOCK[0] = 0.0
    clock = SimpleNamespace(
        monotonic=lambda: _CLOCK[0],
        monotonic_ns=lambda: int(_CLOCK[0] * 1e9),
        perf_counter=time.perf_counter,
        perf_counter_ns=time.perf_counter_ns,
        sleep=time.sleep,
    )
    monkeypatch.setattr(pool, "time", clock)
    monkeypatch.setattr(pool, "_REBALANCE", 0.0)
    monkeypatch.setattr(pool, "_size", 12)
    monkeypatch.setenv("UNBAKE_CORES", "12")
    real = pool._acquire

    def acquire(want, active=0, cap=0):
        real(want, active, cap)
        os.environ["UNBAKE_CORES"] = "3"  # a peer command takes three quarters of the host after the first look

    monkeypatch.setattr(pool, "_acquire", acquire)
    config = replace(_config(tmp_path), host=_host(tmp_path, slots=12, workers=12))
    with effort.command("check", []):
        assert pool.map(config, "units", _tick, list(range(48))) == list(range(48))
    record = next(r for r in effort.closed() if r.kind == "pool")
    held = record.slot_seconds / record.wall_seconds
    assert record.workers_admitted == 12 and 3.0 < held < 6.0


def test_a_task_header_ships_the_landing_locks_the_parent_holds_now(tmp_path, monkeypatch):
    # a warm worker forked inside journal.held kept the parent's _HELD and skipped its reading locks for life
    root = tmp_path / "project"
    monkeypatch.setattr(pool.journal, "_HELD", {root})
    inside = pool._header()
    pool.journal._HELD.clear()
    outside = pool._header()
    pool.journal._HELD.add(root)  # the worker's copy, forked inside the landing
    pool._follow(io.BytesIO(outside))
    assert not pool.journal._HELD
    pool._follow(io.BytesIO(inside))
    assert sorted(pool.journal._HELD) == [root]


def _holds(root):
    return root in pool.journal._HELD


def test_warm_workers_hold_the_landing_lock_only_for_dispatches_sent_inside_it(tmp_path, monkeypatch, warm):
    config = _config(tmp_path)
    root = config.project.root
    monkeypatch.setattr(pool.journal, "_HELD", set())
    with effort.command("check", []):
        assert pool.map(config, "units", _holds, [root, root]) == [False, False]
        with pool.journal.held(config):
            assert pool.map(config, "units", _holds, [root, root]) == [True, True]
        assert pool.map(config, "units", _holds, [root, root]) == [False, False]
    assert warm.forks == 1


def test_a_command_declares_the_work_it_has_not_its_whole_budget():
    """Fair share hands slots to commands that can use them: a command with 3 chunks says 3, with 2 in flight and
    a larger target it keeps the in-flight count, and only a command with no known work says its whole budget."""
    for cap, active, said in [(3, 0, "3 12"), (3, 5, "5 12"), (0, 0, "12 12")]:
        pool._acquire(12, active, cap)
        assert Path(pool._cmd[0].name).read_text() == said
        pool._release()


def test_a_make_in_a_serial_phase_gives_its_idle_tokens_above_its_share_back(tmp_path, monkeypatch):
    read, write = os.pipe()
    os.write(write, b"+++")  # make runs 2 jobs on 5 held slots: 3 tokens lie idle in its pipe
    spare = os.open(f"/proc/self/fd/{read}", os.O_RDONLY | os.O_NONBLOCK)
    slots = [(tmp_path / f"slot{i}").open("a") for i in range(5)]
    given = []
    monkeypatch.setattr(pool, "_dir", [tmp_path])
    monkeypatch.setattr(pool, "_cmd", [(tmp_path / "cmd-1.lock").open("a+")])
    monkeypatch.setattr(pool, "_slots", list(slots))
    monkeypatch.setattr(pool, "_fed", [slots[4]])
    monkeypatch.setattr(pool, "_want", 6)
    peers = [tmp_path / "cmd-1.lock", tmp_path / "cmd-2.lock"]
    monkeypatch.setattr(pool, "_divide", lambda *a: (peers, 6, 3))  # a peer wants work: this command's share is 3
    monkeypatch.setattr(pool, "_give", given.append)
    signals = []
    monkeypatch.setattr(pool, "_signal", signals.append)
    taken, last = [slots[4]], []
    pool._shed(spare, taken, last)
    assert given == [] and last == [3]  # tokens make may not have read yet: none is idle at a first look
    pool._shed(spare, taken, last)
    assert given == [slots[4], slots[3]] and pool._slots == slots[:3]  # the feeder's own first, down to the share
    assert pool._fed == [] and taken == []
    assert os.read(read, 10) == b"+"  # make keeps its 2 running jobs and one idle token within the share
    assert Path(pool._cmd[0].name).read_text() == "3 6"  # two busy slots plus one token to detect growing demand
    assert signals  # the change woke the peers
    os.write(write, b"+")
    pool._shed(spare, taken, last)  # within its share now: nothing more goes back
    assert len(given) == 2 and os.read(read, 10) == b"+"
    os.write(write, b"++")
    peers[1:] = []  # the peer ended: no command could take a slot, so nothing goes back
    pool._shed(spare, taken, last)
    assert len(given) == 2 and os.read(read, 10) == b"++"
    for fd in (read, write, spare):
        os.close(fd)


def test_a_serial_make_keeps_one_probe_token_without_reacquiring_idle_grants(monkeypatch):
    """A feeder's own release wake must not take and return the same slots forever. Consuming its last idle
    token during a later parallel phase must still grow the grant on the next check, without any peer event."""
    pool._acquire(4)
    peers = _commands(os.environ["XDG_RUNTIME_DIR"], 1)
    Path(peers[0].name).write_text("1 4")
    read, write = os.pipe()
    os.write(write, b"+++")  # one serial job on four held slots
    spare = os.open(f"/proc/self/fd/{read}", os.O_RDONLY | os.O_NONBLOCK)
    taken, extra_calls = [], []
    real_extra = pool._extra_slot

    def extra():
        extra_calls.append(len(pool._slots))
        return real_extra()

    def until(pool_dir, mine, attempt, stop=None, also=None, interval=None):
        assert interval == pool._REBALANCE  # consumption has no readable pipe or host event
        assert attempt() is False  # first look does not mistake startup tokens for idle work
        assert len(pool._slots) == 4
        assert attempt() is False  # two observations establish idle capacity
        assert len(pool._slots) == 2 and pool._idle(spare) == 1
        assert Path(pool._cmd[0].name).read_text() == "2 4"
        for _ in range(4):  # own release wake, peer wakes, or periodic checks cannot reacquire an idle grant
            assert attempt() is False
            assert len(pool._slots) == 2 and pool._idle(spare) == 1
        assert extra_calls == []
        assert os.read(read, 1) == b"+"  # make's next phase starts a second parallel job
        assert attempt() is False
        assert len(pool._slots) == 3 and pool._idle(spare) == 1
        assert Path(pool._cmd[0].name).read_text() == "3 4"
        assert len(extra_calls) == 2  # one new grant, then the peer's one-slot share stops growth

    monkeypatch.setattr(pool, "_extra_slot", extra)
    monkeypatch.setattr(pool, "_until", until)
    try:
        pool._feed(write, spare, -1, taken)
    finally:
        for slot in taken:
            pool._fed.remove(slot)
        pool._release()
        for peer in peers:
            peer.close()
        for fd in (read, write, spare):
            os.close(fd)


@pytest.mark.parametrize("interval", [None, 0.5])
def test_a_slot_waiter_only_rechecks_without_an_event_when_its_caller_requests_it(tmp_path, monkeypatch, interval):
    calls = []
    monkeypatch.setattr(pool, "_inotify", lambda *paths: 501)
    monkeypatch.setattr(pool, "_peers", lambda *args: [])
    monkeypatch.setattr(pool.os, "close", lambda fd: None)

    def select(readers, writers, errors, timeout):
        calls.append(timeout)
        return [502], [], []  # make returned: the stop fd is readable

    monkeypatch.setattr(pool.select, "select", select)
    pool._until(tmp_path, tmp_path / "cmd-1.lock", lambda: False, stop=502, interval=interval)
    assert calls == [interval]


def test_a_refusal_while_keys_resolve_reaches_the_caller_not_an_unbound_width(tmp_path, monkeypatch):
    # G73: unbake map crashed with UnboundLocalError on `width` when a key job refused (no assembly for a row)
    refusal = Refusal(Finding("version.split", "no assembly for osScAddClient in de", action="run unbake setup"))

    def refuse(config, runs):
        raise refusal

    monkeypatch.setattr(pool, "_resolve", refuse)
    with effort.command("t", ()), pytest.raises(Refusal) as caught:
        pool.gather(_config(tmp_path), [("job", len, [b"a"], len)])
    assert caught.value is refusal
