"""Effort spans, accounting, and revision-4 events."""

import json
from dataclasses import asdict

import fixture
import pytest

from unbake import effort
from unbake.contracts import Finding, Origin, Refusal


def _config(tmp_path):
    keys = ("resources.memory_parent_bytes", "resources.memory_worker_bytes", "budgets.serial_seconds")
    origins = {key: Origin(key, "/host.toml", "0" * 64) for key in keys}
    return fixture.bare_config(
        tmp_path, versions=(), toolchain="test", memory=1 << 30, origins=origins, author=("test", "test@invalid")
    )


@pytest.fixture(autouse=True)
def _listeners(monkeypatch):
    monkeypatch.setattr(effort, "_listeners", [])


def test_listen_events():
    events = []
    effort.listen(lambda event, body: events.append((event, body)))
    with effort.command("check", []), effort.stage("outer"), effort.capture():
        effort.record_native("tool", fixture.native_result(cpu=2), 0)
    assert [(event, body["path"]) for event, body in events] == [
        ("open", ["check"]),
        ("open", ["check", "outer"]),
        ("close", ["check", "outer"]),
        ("close", ["check"]),
    ]
    for event, body in events:
        if event == "close":
            assert body["wall"] >= 0
            assert body["status"] == "ok"


def test_progress_event():
    events = []
    effort.listen(lambda event, body: events.append((event, body)))
    effort.progress("units", 2, 7)
    assert events == [("progress", {"name": "units", "done": 2, "total": 7})]


def test_closed_records():
    with effort.command("first", []), effort.stage("child"):
        pass
    records = effort.closed()
    assert isinstance(records, tuple)
    assert [r.kind for r in records] == ["stage", "own", "command"]
    assert records[0].parent == records[-1].span
    assert records[1].path == ("first", "(own)")
    assert effort.tree()["span"] == records[-1].span
    with effort.command("second", []):
        pass
    assert [r.path for r in effort.closed()] == [("second",)]
    assert records[-1].invocation != effort.invocation()


def test_sink_under_dot_unbake(tmp_path, capsys):
    with effort.command("check", []):
        with effort.stage("before.bind"):
            pass
        effort.bind(_config(tmp_path))
    rows = [json.loads(line) for line in (tmp_path / ".unbake/stages.jsonl").read_text().splitlines()]
    closed = effort.closed()
    assert {r["span"] for r in rows} <= {r.span for r in closed}
    assert rows[-1]["span"] == closed[-1].span  # the root is kept
    assert not (tmp_path / "build/stages.jsonl").exists()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("event", ["open", "close", "progress"])
def test_listener_failure_is_finding(event):
    def fail(kind, body):
        if kind == event:
            raise ValueError("listener broke")

    seen = []
    effort.listen(fail)
    effort.listen(lambda kind, body: seen.append(kind))
    with effort.command("check", []):
        effort.progress("work", 1, 1)
    findings = effort.closed()[-1].findings
    assert len(findings) == 1
    assert findings[0].key == "effort.log"
    assert not findings[0].blocking
    assert "listener broke" in findings[0].reason
    assert seen == ["open", "progress", "close"]


def test_listeners_survive_commands():
    events = []
    effort.listen(lambda event, body: events.append(event))
    for name in ("one", "two"):
        with effort.command(name, []):
            pass
    assert events == ["open", "close", "open", "close"]


@pytest.mark.parametrize(
    "exc,status",
    [
        (Refusal(Finding("effort.log", "stop")), "refused"),
        (ValueError("stop"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_span_status_and_stack_restore(exc, status):
    with pytest.raises(type(exc)), effort.command("check", []), effort.stage("work"):
        raise exc
    assert all(r.status == status for r in effort.closed())
    assert effort._stack.get() == ()
    if isinstance(exc, Refusal):
        assert effort.closed()[-1].findings == exc.findings


def test_own_wall_and_cpu(monkeypatch):
    wall = iter([0, 1_000_000_000, 3_000_000_000, 5_000_000_000])
    cpu = iter([0.0, 1.0, 3.0, 6.0])
    monkeypatch.setattr(effort.time, "perf_counter_ns", lambda: next(wall))
    monkeypatch.setattr(effort, "_cpu", lambda: next(cpu))
    with effort.command("check", []), effort.stage("work"):
        pass
    child, own, root = effort.closed()
    assert (child.wall_seconds, child.parent_cpu_seconds) == (2, 2)
    assert (own.wall_seconds, own.parent_cpu_seconds) == (3, 4)
    assert (root.wall_seconds, root.parent_cpu_seconds, root.cores) == (5, 6, 1.2)


def test_native_and_pool_cpu_cache(tmp_path):
    with effort.command("check", []):
        effort.bind(_config(tmp_path))
        effort.record_native("tool", fixture.native_result(cpu=3), effort.time.perf_counter_ns())
        effort.record_pool(
            "units",
            2,
            2,
            2,
            effort.time.perf_counter_ns(),
            [
                {
                    "worker_cpu": 4,
                    "native_cpu": 5,
                    "rss": 1234,
                    "pid": 10,
                    "start_ns": 0,
                    "end_ns": 2_000_000_000,
                    "cache": {"object": (2, 1)},
                    "records": [],
                    "wait_seconds": 1.0,
                },
                {
                    "worker_cpu": 6,
                    "native_cpu": 7,
                    "rss": 2345,
                    "pid": 11,
                    "start_ns": 0,
                    "end_ns": 3_000_000_000,
                    "cache": {"object": (1, 2)},
                    "records": [],
                },
            ],
        )
    root = effort.closed()[-1]
    assert (root.worker_cpu_seconds, root.native_cpu_seconds) == (10, 15)
    assert (root.items, root.jobs, root.workers_used, root.workers_admitted) == (2, 2, 2, 2)
    assert root.worker_rss_peak_bytes == 2345
    assert root.cache["object"] == (3, 3)
    held = next(r for r in effort.closed() if r.kind == "pool")
    assert held.wait_seconds == 1.0 and held.busy_seconds == 4.0


def test_pool_job_records_rebased():
    with effort.capture() as captured, effort.stage("work"):
        effort.count("object", True)
    raws = [asdict(r) for r in captured]
    with effort.command("check", []):
        effort.record_pool(
            "units",
            1,
            1,
            1,
            effort.time.perf_counter_ns(),
            [
                {
                    "worker_cpu": 1,
                    "native_cpu": 0,
                    "rss": 1,
                    "pid": 1,
                    "start_ns": 0,
                    "end_ns": 1,
                    "cache": {"object": (1, 0)},
                    "records": raws,
                }
            ],
        )
    jobs = [r for r in effort.closed() if r.path[:2] == ("check", "units") and r.kind != "pool"]
    assert [r.kind for r in jobs] == [r["kind"] for r in raws]
    assert all(r.parent_cpu_seconds == 0 for r in jobs)
    assert [r.worker_cpu_seconds for r in jobs] == [r["parent_cpu_seconds"] for r in raws]
    assert all(r.path[:2] == ("check", "units") for r in jobs)
    assert all(r.invocation == effort.invocation() for r in jobs)
    assert not {r.span for r in jobs} & {r["span"] for r in raws}


def test_capture_restores_and_null_stage():
    with effort.stage("outside") as span:
        span.add(items=3)
    with effort.command("check", []):
        effort.count("object", True)
        with effort.capture() as records:
            effort.count("view", False)
        assert effort.counters() == {"object": (1, 0)}
        assert records[-1].kind == "job"
    assert effort.closed()[-1].cache == {"object": (1, 0)}


def test_log_failure_is_debt(tmp_path, capsys):
    (tmp_path / ".unbake").write_text("cannot be a directory")
    with effort.command("check", []):
        effort.bind(_config(tmp_path))
    assert any(f.key == "effort.log" and not f.blocking for f in effort.closed()[-1].findings)
    assert capsys.readouterr().err == ""


def test_nested_cpu_inclusive_spans_exclusive_own(monkeypatch):
    cpu = iter([0.0, 1.0, 2.0, 4.0, 7.0, 10.0])
    monkeypatch.setattr(effort, "_cpu", lambda: next(cpu))
    with effort.command("check", []), effort.stage("outer"), effort.stage("inner"):
        pass
    rows = {r.path: r for r in effort.closed()}
    assert rows[("check",)].parent_cpu_seconds == 10
    assert rows[("check", "outer")].parent_cpu_seconds == 6
    assert rows[("check", "outer", "inner")].parent_cpu_seconds == 2
    assert rows[("check", "(own)")].parent_cpu_seconds == 4
    assert rows[("check", "outer", "(own)")].parent_cpu_seconds == 4


def test_imported_native_own_and_cpu_classification():
    with effort.capture() as captured, effort.stage("work"):
        effort.record_native("tool", fixture.native_result(cpu=2), effort.time.perf_counter_ns())
    raw = [asdict(r) for r in captured]
    with effort.command("check", []):
        effort.record_pool(
            "units",
            1,
            1,
            1,
            effort.time.perf_counter_ns(),
            [
                {
                    "worker_cpu": 1,
                    "native_cpu": 2,
                    "rss": 1,
                    "pid": 1,
                    "start_ns": 0,
                    "end_ns": 1,
                    "cache": {},
                    "records": raw,
                }
            ],
        )
    imported = [r for r in effort.closed() if r.path[:3] == ("check", "units", "job")]
    assert [r.kind for r in imported] == [r["kind"] for r in raw]
    native = next(r for r in imported if r.kind == "native")
    assert native.native_cpu_seconds == 2 and native.worker_cpu_seconds == 0
    root = next(r for r in imported if r.kind == "job")
    assert root.worker_cpu_seconds > 0 and root.parent_cpu_seconds == 0
    assert root.parent_rss_peak_bytes == 0 and root.worker_rss_peak_bytes > 0


def test_wait_is_charged_to_the_innermost_stage_only(tmp_path):
    effort.bind(_config(tmp_path))
    with effort.command("check", []), effort.stage("outer"), effort.stage("inner"):
        effort.waited(1.5)
    waits = {r.path[-1]: r.wait_seconds for r in effort.closed() if r.kind in ("stage", "command")}
    assert waits == {"inner": 1.5, "outer": 0.0, "check": 0.0}


def test_forget_drops_only_the_named_kinds():
    with effort.command("check", []):
        effort.memo(("a", 1), lambda: 1)
        effort.memo(("b", 1), lambda: 2)
        effort.forget("a")
        assert effort.memo(("a", 1), lambda: 3) == 3 and effort.memo(("b", 1), lambda: 4) == 2


def test_a_stage_that_fails_names_the_exception_and_the_frame_once():
    from pathlib import Path

    from unbake import land

    with pytest.raises(FileNotFoundError), effort.command("land", []), effort.stage("outer"), effort.stage("inner"):
        land._read(Path("/nonexistent/entry.json"))
    failed = {r.path[-1]: r for r in effort._closed if r.status == "failed"}
    note = failed["inner"].findings[0]
    assert note.key == "internal.error" and "FileNotFoundError" in note.reason and " at land:_read:" in note.reason
    assert failed["outer"].findings == ()  # the stage that raised names it, its parents do not repeat it


def test_close_events_carry_cores_and_jobs_and_open_events_the_previous_wall():
    events = []
    effort.listen(lambda event, body: events.append((event, body)))
    effort._walls["check > outer"] = 42.0
    with effort.command("check", []), effort.stage("outer"):
        pass
    opens = {tuple(b["path"]): b["expected"] for e, b in events if e == "open"}
    assert opens == {("check",): None, ("check", "outer"): 42.0}
    closes = [b for e, b in events if e == "close"]
    assert all({"cores", "jobs", "wall"} <= set(b) for b in closes)


def test_walls_persist_beside_the_stage_log_for_shallow_stages(tmp_path, monkeypatch):
    sink = tmp_path / "stages.jsonl"
    monkeypatch.setattr(effort, "_walls", {})
    with effort.command("check", []):
        monkeypatch.setattr(effort, "_sink", sink)
        with effort.stage("a"), effort.stage("b"), effort.stage("c"):
            pass
    saved = effort._load_walls(tmp_path / "walls.json")
    assert set(saved) == {"check", "check > a", "check > a > b"}  # deeper than three levels is not remembered
    assert effort._load_walls(tmp_path / "missing.json") == {}


def test_the_stage_log_keeps_only_its_newest_whole_lines_past_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(effort, "LOG_BYTES", 100)
    log = tmp_path / "stages.jsonl"
    log.write_text("".join(f"line{i:03d}\n" for i in range(40)))
    effort._trim(log)
    kept = log.read_text().splitlines()
    assert kept[-1] == "line039" and all(k.startswith("line") and len(k) == 7 for k in kept) and len(kept) <= 7
    before = log.read_bytes()
    effort._trim(log)
    assert log.read_bytes() == before


def _pressure(monkeypatch, resident):
    monkeypatch.setattr(effort, "_pressure", lambda: (resident, 1000))


@pytest.mark.parametrize(("resident", "kept"), [(400, True), (600, False)])
def test_scoped_drops_only_what_the_scope_added_and_only_under_pressure(monkeypatch, resident, kept):
    _pressure(monkeypatch, resident)
    monkeypatch.setattr(effort, "_memo", {("base", 1): 1})
    with effort.scoped():
        effort.memo(("overlay", 1), lambda: 2)
    assert (("overlay", 1) in effort._memo) is kept
    assert ("base", 1) in effort._memo


def test_scoped_records_the_trim_count(monkeypatch):
    _pressure(monkeypatch, 900)
    monkeypatch.setattr(effort, "_memo", {})
    with effort.command("t", []):
        with effort.scoped():
            effort.memo(("a", 1), lambda: 1)
            effort.memo(("a", 2), lambda: 2)
        trimmed = [r for r in effort._closed if r.path[-1] == "effort.trimmed"]
    assert [r.items for r in trimmed] == [2] and not trimmed[0].findings


def _mock_data(monkeypatch, vmdata_kib, soft, fork):
    reads = []

    def status():
        reads.append("status")
        return f"Name:\tx\nRssAnon:\t500 kB\nVmData:\t{vmdata_kib} kB\nVmRSS:\t999999999 kB\n"

    real = effort.Path.read_text
    monkeypatch.setattr(
        effort.Path,
        "read_text",
        lambda self, *a, **k: status() if str(self) == "/proc/self/status" else real(self, *a, **k),
    )
    monkeypatch.setattr(effort.resource, "getrlimit", lambda which: (soft, soft))
    monkeypatch.setattr(effort, "_fork_data", fork)
    return reads


def test_a_worker_measures_data_growth_since_the_fork_never_resident_memory(monkeypatch):
    reads = _mock_data(monkeypatch, 3000, 1_000_000 + 2_048_000, -1)
    effort.mark_fork(2_048_000)  # forked at 2000 KiB
    assert effort._pressure() == (3000 * 1024 - 2_048_000, 1_000_000)  # VmRSS in the status is ignored
    assert reads == ["status"]  # one /proc read per check


def test_a_parent_compares_its_anonymous_resident_memory_to_the_parent_budget(monkeypatch, tmp_path):
    _mock_data(monkeypatch, 3000, 0, -1)
    monkeypatch.setattr(effort, "_config", _config(tmp_path))
    assert effort._pressure() == (500 * 1024, effort._config.host.memory_parent_bytes)  # RssAnon, not VmData


@pytest.mark.parametrize(("vmdata_kib", "kept"), [(2000 + 400, True), (2000 + 600, False)])
def test_scoped_fires_on_growth_past_half_the_worker_limit(monkeypatch, vmdata_kib, kept):
    _mock_data(monkeypatch, vmdata_kib, 2000 * 1024 + 1000 * 1024, 2000 * 1024)
    monkeypatch.setattr(effort, "_memo", {})
    with effort.scoped():
        effort.memo(("overlay", 1), lambda: 1)
    assert (("overlay", 1) in effort._memo) is kept


def test_a_worker_with_a_budget_trims_against_the_budget_not_the_higher_data_limit(monkeypatch):
    _mock_data(monkeypatch, 3000, 4_000_000 + 2_048_000, -1)
    effort.mark_fork(2_048_000, 1_000_000)
    try:
        assert effort._pressure() == (3000 * 1024 - 2_048_000, 1_000_000)
    finally:
        effort.mark_fork(-1)


def test_release_drops_every_value_keyed_by_a_snapshot_digest_and_the_snapshot_itself(monkeypatch):
    from types import SimpleNamespace

    laid = SimpleNamespace(digest="wave1")
    monkeypatch.setattr(
        effort,
        "_memo",
        {
            ("parent", "wave1"): 1,
            ("pool.writes", "wave1"): b"blob",
            ("pin", "wave1", "a.h"): "x",
            ("overlay", "base", "w"): laid,  # keyed by its base: found by its value
            ("pin", "wave2", "a.h"): "y",
            ("catalog", "/root", "k"): 2,
        },
    )
    assert effort.release("wave1") == 4
    assert set(effort._memo) == {("pin", "wave2", "a.h"), ("catalog", "/root", "k")}


def test_a_worker_trims_to_what_it_was_forked_holding_but_keeps_content_keyed_kinds(monkeypatch):
    monkeypatch.setattr(effort, "_memo", {("catalog", "base"): 1})
    monkeypatch.setattr(effort, "_inherited", frozenset())
    effort.mark_fork(0)
    effort._memo.update({("catalog", "wave"): 2, ("words", "h"): 3, ("spellings", "wave"): 4})
    assert effort.trim_to_fork(keep=("words",)) == 2
    assert set(effort._memo) == {("catalog", "base"), ("words", "h")}


def test_the_stage_log_keeps_the_head_of_a_long_finding_and_the_record_keeps_it_whole(tmp_path, monkeypatch):
    sink = tmp_path / "stages.jsonl"
    reason = "error: " + "x" * 50_000  # a compile error quoting whole headers
    with effort.command("check", []):
        monkeypatch.setattr(effort, "_sink", sink)
        with pytest.raises(Refusal), effort.stage("native.objects"):
            raise Refusal(Finding("compile.error", reason))
    row = next(json.loads(line) for line in sink.read_text().splitlines() if "native.objects" in line)
    logged = row["findings"][0]["reason"]
    assert logged.startswith("error: x") and len(logged) < effort.REASON_CHARS + 40 and "50007 chars" in logged
    kept = next(r for r in effort.closed() if r.path[-1] == "native.objects")
    assert kept.findings[0].reason == reason
