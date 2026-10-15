"""Invocation spans, exclusive parent work, and worker/native accounting."""

from __future__ import annotations

import json
import os
import resource
import time
import traceback
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil

from unbake.contracts import Config, Finding, Json, NativeResult, Origin, Refusal, Span, StageRecord

_stack: ContextVar[tuple[_Open, ...]] = ContextVar("effort_stack", default=())
_closed: list[StageRecord] = []
_counters: dict[str, list[int]] = {}
_config: Config | None = None
_sink: Path | None = None
_buffer: list[StageRecord] = []
_invocation = ""
_memo: dict[object, Any] = {}  # pure readers' values for this command, by input digest
_dispatched: dict[str, tuple[str, ...]] = {}  # gather stage span -> the dispatches it ran
_seen: set[tuple[str, ...]] = set()  # slow stage paths already closed in this command
_listeners: list[Callable[[str, Json], None]] = []
_walls: dict[str, float] = {}  # the previous command's wall seconds by stage path, for "expected" on start lines
LOG_BYTES = 64 << 20  # the stage log keeps its newest half once it grows past this
WALL_DEPTH = 3  # stages this shallow report live progress and are remembered in .unbake/walls.json


def expected(path: Sequence[str]) -> float | None:
    """How long the previous command spent in this stage path, None when it never ran or was never recorded."""
    return _walls.get(" > ".join(path))


def _load_walls(path: Path) -> dict[str, float]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _save_walls() -> None:
    """Keep this command's shallow stage walls beside the stage log: a few lines, not a scan of the log."""
    if _sink is None:
        return
    walls = {" > ".join(r.path): r.wall_seconds for r in _closed if len(r.path) <= WALL_DEPTH and r.kind != "own"}
    path = _sink.with_name("walls.json")
    try:
        part = path.with_name(f".{uuid4().hex}.part")
        part.write_text(json.dumps({**_load_walls(path), **walls}, sort_keys=True), encoding="utf-8")
        part.replace(path)
    except OSError:
        _log_failure()


def memo[T](key: object, produce: Callable[[], T]) -> T:
    """A pure reader's value for the running command: computed once per key, dropped when the command ends."""
    if key not in _memo:
        _memo[key] = produce()
    value: T = _memo[key]
    return value


def peek(key: object) -> Any:
    """The memoised value of a key, None when none is kept."""
    return _memo.get(key)


def memo_valid[T](key: object, produce: Callable[[], T], valid: Callable[[T], bool]) -> T:
    """memo, but a stored value that no longer passes `valid` is made again."""
    if key in _memo and valid(_memo[key]):
        value: T = _memo[key]
        return value
    value = _memo[key] = produce()
    return value


def forget(*kinds: str) -> None:
    """Drops the memoised values of these kinds: the files they read were rewritten by this command."""
    for key in [k for k in _memo if isinstance(k, tuple) and k[0] in kinds]:
        del _memo[key]


def release(digest_: str) -> int:
    """Drops every memoised value whose key names this snapshot digest, and the snapshot itself: one no later stage
    meets (a wave's overlay once the next one is laid) takes its pins, pool blobs and overlay with it. Returns how
    many went."""
    gone = [
        k for k, v in _memo.items() if (isinstance(k, tuple) and digest_ in k) or getattr(v, "digest", 0) == digest_
    ]
    for k in gone:
        del _memo[k]
    return len(gone)


_inherited: frozenset[Any] = frozenset()  # the memo keys a forked worker was born holding (shared with the parent)


def trim_to_fork(keep: Collection[str] = ()) -> int:
    """A worker drops what it memoised since the fork, except the kinds in `keep` (small values keyed by content):
    what it built for one overlay is never read on the next. Returns how many went."""
    gone = [k for k in _memo if k not in _inherited and not (isinstance(k, tuple) and k and k[0] in keep)]
    for k in gone:
        del _memo[k]
    return len(gone)


_fork_data = -1  # a worker's VmData at fork, set by pool._init; -1 in the parent
_fork_budget = 0  # the growth a worker plans to stay within (host memory_worker_bytes); 0 = its data limit is the plan


def mark_fork(data: int, budget: int = 0) -> None:
    """A forked worker records the data segment it inherited: its pressure counts only growth past it, against the
    planned budget (the hard data limit sits above it, so a job over budget trims its memo before it is refused)."""
    global _fork_data, _fork_budget, _inherited
    _fork_data, _fork_budget = data, budget
    _inherited = frozenset(_memo)


def _status_kib(field: str) -> int:
    """A kB field of /proc/self/status in bytes, 0 when absent."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    return 0


def _pressure() -> tuple[int, int]:
    """(used, allowed): a worker's VmData growth since the fork against its data limit, else the parent's RssAnon
    against its budget (the parent has no data limit, and its VmData counts reserved arenas it never touches)."""
    if _fork_data >= 0:
        soft = resource.getrlimit(resource.RLIMIT_DATA)[0]
        allowed = _fork_budget or (0 if soft == resource.RLIM_INFINITY else soft - _fork_data)
        return _status_kib("VmData") - _fork_data, allowed
    return _status_kib("RssAnon"), (_config.host.memory_parent_bytes if _config else 0)


@contextmanager
def scoped(always: bool = False, keep: Collection[str] = ()) -> Iterator[None]:
    """Values memoised inside are dropped on exit when the process is past half its memory limit, or `always` (a
    job whose intermediate snapshots no later job meets), except the kinds in `keep`.

    The memo only holds pure values by input, so dropping one costs a recompute and never a wrong answer. Keys present
    on entry (the base snapshot's) stay, so reuse across scopes survives while memory is fine.
    """
    before = set(_memo)
    try:
        yield
    finally:
        used, limit = _pressure()
        if always or (limit and used > limit // 2):
            added = [k for k in _memo if k not in before and not (isinstance(k, tuple) and k and k[0] in keep)]
            for k in added:
                del _memo[k]
            if added:
                with stage("effort.trimmed") as span:
                    span.add(items=len(added))


def listen(callback: Callable[[str, Json], None]) -> None:
    _listeners.append(callback)


def _notify(event: str, body: Json) -> None:
    for callback in tuple(_listeners):
        try:
            callback(event, body)
        except Exception as exc:
            if _stack.get():
                _stack.get()[-1].findings.append(Finding("effort.log", reason=f"Listener: {exc}", blocking=False))


def say(text: str) -> None:
    """One line for the person watching the command now (stderr), outside any stage's totals."""
    _notify("say", {"text": text})


def progress(name: str, done: int, total: int) -> None:
    _notify("progress", {"name": name, "done": done, "total": total})


def closed() -> tuple[StageRecord, ...]:
    return tuple(_closed)


def _cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


class _Open:
    def __init__(self, name: str, kind: str, parent: _Open | None = None):
        self.span, self.parent = uuid4().hex, parent.span if parent else None
        self.path: tuple[str, ...] = (parent.path if parent else ()) + (name,)
        self.kind = kind
        self.start, self.cpu = time.perf_counter_ns(), _cpu()
        self.rss = psutil.Process(os.getpid()).memory_info().rss if kind in ("command", "pool") else 0
        self.cache, self.items, self.wait = counters(), 0, 0.0
        self.findings: list[Finding] = []
        self.children: list[StageRecord] = []
        self.remote_cache: dict[str, tuple[int, int]] = {}

    def add(self, *, items: int = 0, findings: Sequence[Finding] = ()) -> None:
        self.items += items
        self.findings.extend(findings)


def waited(seconds: float) -> None:
    if _stack.get():
        _stack.get()[-1].wait += seconds


def _limits() -> dict[str, Any]:
    keys = ("resources.memory_parent_bytes", "resources.memory_worker_bytes")
    return (
        {
            "memory_limits": {k: _config.host.origins[k] for k in keys},
            "memory_limit_values": dict(
                zip(keys, (_config.host.memory_parent_bytes, _config.host.memory_worker_bytes), strict=True)
            ),
        }
        if _config
        else {}
    )


def _record(
    opened: _Open, wall: float, parent: float = 0, worker: float = 0, native: float = 0, **values: Any
) -> StageRecord:
    cores = (parent + worker + native) / wall if wall else 0
    execution = "parallel" if _config and cores >= _config.host.serial_cores else "serial"
    values.setdefault("wait_seconds", opened.wait)
    return StageRecord(
        _invocation,
        opened.span,
        opened.parent,
        opened.path,
        opened.kind,
        "ok",
        execution,
        opened.start,
        wall,
        parent,
        worker,
        native,
        cores,
        values.pop("items", opened.items),
        values.pop("jobs", 0),
        values.pop("workers_used", 0),
        values.pop("workers_admitted", 0),
        values.pop("parent_rss_peak_bytes", 0),
        values.pop("worker_rss_peak_bytes", 0),
        **(_limits() if opened.parent is None else {}),
        **values,
    )


def _budget(record: StageRecord, leaf: bool, shipped: bool = False) -> StageRecord:
    """The watchdog: a stage that breaks a [budgets] rule carries a non-blocking budget.* finding naming it."""
    if not _config or record.kind not in ("stage", "own", "pool"):
        return record
    host, found, cores, pool = _config.host, [], f"{record.cores:.2f} cores", record.kind == "pool"
    slow = record.wall_seconds >= host.serial_seconds
    hits, misses = (sum(v[i] for v in record.cache.values()) for i in (0, 1))
    if leaf and slow and record.cores < host.serial_cores:
        found.append(("single_core", "serial_seconds", f"ran {record.wall_seconds:.1f}s at {cores}"))
    if (
        pool
        and record.jobs >= host.pool_fanout * record.workers_admitted
        and (record.cores < host.pool_fill * record.workers_admitted)
    ):
        found.append(("pool_underused", "pool_fill", f"kept {cores} of {record.workers_admitted} workers busy"))
    if pool and record.workers_admitted and record.jobs and hits and not misses:  # inline jobs dispatch nothing
        found.append(("warm_dispatch", "pool_fill", f"dispatched {record.jobs} jobs with every cache lookup warm"))
    # a job's stage repeats once per job by design; a dispatch repeats when a later phase needs the earlier results
    if slow and record.kind != "own" and not shipped:
        # a gather is the same stage only when it ran the same dispatches: a later phase needs the earlier results
        # the kind keeps a gather's key (its path, then the names of its dispatches) apart from the key of the
        # dispatch record itself, whose path is the gather's path then the same name
        key = (record.kind, *record.path, *_dispatched.get(record.span, ()))
        if key in _seen:
            found.append(("repeated", "serial_seconds", "closed twice in one command"))
        _seen.add(key)
    return replace(
        record,
        findings=(
            *record.findings,
            *(
                Finding(
                    f"budget.{key}",
                    reason=f"{' > '.join(record.path)} {why}",
                    blocking=False,
                    origin=host.origins[f"budgets.{name}"],
                )
                for key, name, why in found
            ),
        ),
    )


def _log_failure() -> Finding:
    finding = Finding("effort.log", reason="Stage log could not be written.", blocking=False)
    if _stack.get():
        _stack.get()[0].findings.append(finding)
    return finding


REASON_CHARS = 2000  # a finding's reason in the stage log; a compile error quoting whole headers is 100 KB per unit


def _logged(record: StageRecord) -> Json:
    """The stage-log row of a record: each finding's reason and missing lines cut to their head (the command's result
    keeps them whole)."""

    def cut(text: str) -> str:
        return text if len(text) <= REASON_CHARS else text[:REASON_CHARS] + f" ... ({len(text)} chars)"

    row = asdict(record)
    for finding in row["findings"]:
        finding["reason"] = cut(finding["reason"])
        finding["missing"] = [cut(m) for m in finding.get("missing", ())]
    return row


def _emit(record: StageRecord, leaf: bool = False, shipped: bool = False) -> StageRecord:
    record = _budget(record, leaf, shipped)
    if _sink and (
        record.parent is None
        or record.kind == "pool"
        or record.findings
        or (_config is not None and record.wall_seconds >= _config.host.serial_seconds / 4)
    ):
        try:
            with _sink.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_logged(record)) + "\n")
        except OSError:
            finding = _log_failure()
            if record.parent is None:
                record = replace(record, findings=(*record.findings, finding))
            _buffer.append(record)
    elif not _sink:
        _buffer.append(record)
    _closed.append(record)
    return record


def _close(opened: _Open, status: str) -> StageRecord:
    end = time.perf_counter_ns()
    children = opened.children
    wall = max(0, end - opened.start) / 1e9
    parent = max(0, _cpu() - opened.cpu)
    own_cpu = max(0, parent - sum(c.parent_cpu_seconds for c in children))
    cache = {
        k: (v[0] - opened.cache.get(k, (0, 0))[0], v[1] - opened.cache.get(k, (0, 0))[1]) for k, v in counters().items()
    }
    _merge_cache(cache, opened.remote_cache)
    record = _record(
        opened,
        wall,
        parent,
        sum(c.worker_cpu_seconds for c in children),
        sum(c.native_cpu_seconds for c in children),
        cache=cache,
        items=opened.items + sum(c.items for c in children if c.kind == "pool"),
        jobs=sum(c.jobs for c in children),
        workers_used=max((c.workers_used for c in children), default=0),
        workers_admitted=max((c.workers_admitted for c in children), default=0),
        parent_rss_peak_bytes=max(
            opened.rss,
            psutil.Process(os.getpid()).memory_info().rss,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        ),
        worker_rss_peak_bytes=max((c.worker_rss_peak_bytes for c in children), default=0),
    )
    if children:
        covered, cursor = 0, opened.start
        for start, stop in sorted(
            (max(opened.start, c.start_ns), min(end, c.start_ns + int(c.wall_seconds * 1e9))) for c in children
        ):
            covered += max(0, stop - max(cursor, start))
            cursor = max(cursor, stop)
        own = _Open.__new__(_Open)
        own.span, own.parent, own.path = uuid4().hex, opened.span, (*opened.path, "(own)")
        own.kind, own.start, own.items, own.wait = "own", opened.start, 0, 0.0
        _emit(replace(_record(own, max(0, wall - covered / 1e9), own_cpu), status=status), True)
    return replace(record, status=status)


@contextmanager
def _scope(opened: _Open) -> Iterator[_Open]:
    token = _stack.set((*_stack.get(), opened))
    status = "ok"
    try:
        if opened.kind in ("command", "stage", "pool"):
            _notify("open", {"path": list(opened.path), "expected": expected(opened.path)})
        yield opened
    except BaseException as exc:
        status = (
            "refused" if isinstance(exc, Refusal) else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        if isinstance(exc, Refusal):
            opened.findings.extend(exc.findings)
        elif status == "failed" and not vars(exc).get("_named", False):  # the stage that raised names the cause once
            own = [f for f in traceback.extract_tb(exc.__traceback__) if "/unbake/" in f.filename]
            frames = [f for f in own if Path(f.filename).stem != "effort"]
            at = f" at {Path(frames[-1].filename).stem}:{frames[-1].name}:{frames[-1].lineno}" if frames else ""
            opened.findings.append(Finding("internal.error", f"{type(exc).__name__}: {exc}{at}"))
            vars(exc)["_named"] = True
        raise
    finally:
        try:
            record = _close(opened, status)
            if opened.kind in ("command", "stage", "pool"):
                _notify(
                    "close",
                    {
                        "path": list(opened.path),
                        "wall": record.wall_seconds,
                        "status": record.status,
                        "cores": record.cores,
                        "jobs": record.jobs,
                    },
                )
            _dispatched[record.span] = tuple(c.path[-1] for c in opened.children if c.kind == "pool")
            record = _emit(replace(record, findings=tuple(opened.findings)), not opened.children)
            _dispatched.pop(record.span, None)
            stack = _stack.get()
            if len(stack) > 1:
                stack[-2].children.append(record)
                _merge_cache(stack[-2].remote_cache, opened.remote_cache)
        finally:
            _stack.reset(token)


@contextmanager
def command(name: str, argv: Sequence[str]) -> Iterator[None]:
    global _closed, _counters, _config, _sink, _buffer, _invocation, _memo, _seen
    _closed, _counters, _config, _sink, _buffer, _invocation = [], {}, None, None, [], uuid4().hex
    _memo, _seen = {}, set()
    token = _stack.set(())
    try:
        with _scope(_Open(name, "command")):
            yield None
    finally:
        _stack.reset(token)
        _save_walls()


def _trim(path: Path) -> None:
    """Drops the oldest lines of a stage log past `LOG_BYTES`, keeping the newest half whole-line."""
    if path.exists() and path.stat().st_size > LOG_BYTES:
        with path.open("rb") as stream:
            stream.seek(-LOG_BYTES // 2, 2)
            stream.readline()
            kept = stream.read()
        temp = path.with_name(f"{path.name}.{os.getpid()}")
        temp.write_bytes(kept)
        temp.replace(path)


def bind(config: Config) -> None:
    global _config, _sink, _buffer
    with stage("effort.bind"):
        _config, _sink = config, config.project.root / ".unbake" / "stages.jsonl"
        _walls.clear()
        _walls.update(_load_walls(_sink.with_name("walls.json")))
        try:
            _sink.parent.mkdir(parents=True, exist_ok=True)
            _trim(_sink)
            with _sink.open("a", encoding="utf-8") as stream:
                while _buffer:
                    stream.write(json.dumps(asdict(_buffer[0])) + "\n")
                    stream.flush()
                    _buffer.pop(0)
        except OSError:
            _log_failure()


@contextmanager
def stage(name: str) -> Iterator[Span]:
    if not _stack.get():
        yield _Null()
    else:
        with _scope(_Open(name, "stage", _stack.get()[-1])) as span:
            yield span


def record_native(name: str, result: NativeResult, start_ns: int) -> None:
    if _stack.get():
        opened = _Open(name, "native", _stack.get()[-1])
        opened.start = start_ns
        record = _emit(
            _record(opened, result.wall_seconds, native=result.cpu_seconds, worker_rss_peak_bytes=result.max_rss_bytes)
        )
        _stack.get()[-1].children.append(record)


def record_pool(
    name: str,
    items: int,
    jobs: int,
    admitted: int,
    start_ns: int,
    envelopes: Sequence[Json],
    dispatch_seconds: float = 0.0,
    end_ns: int | None = None,
) -> None:
    if not _stack.get():
        return
    opened = _Open(name, "pool", _stack.get()[-1])
    opened.start = start_ns
    cache: dict[str, tuple[int, int]] = {}
    for envelope in envelopes:
        _merge_cache(cache, envelope.get("cache", {}))
        records = envelope.get("records", [])
        ids = {r["span"]: uuid4().hex for r in records}
        parents = {r["parent"] for r in records}  # a shipped stage nothing else hangs under is a worker's leaf
        for raw in records:
            values = dict(raw)
            values.update(
                invocation=_invocation,
                span=ids[raw["span"]],
                parent=ids.get(raw["parent"], opened.span),
                path=opened.path + tuple(raw["path"]),
                parent_cpu_seconds=0,
                worker_cpu_seconds=raw["parent_cpu_seconds"] + raw["worker_cpu_seconds"],
                parent_rss_peak_bytes=0,
                worker_rss_peak_bytes=max(raw["parent_rss_peak_bytes"], raw["worker_rss_peak_bytes"]),
            )
            values["memory_limits"] = {k: Origin(**v) for k, v in raw.get("memory_limits", {}).items()}
            values.update(_limits())
            values["cache"] = {k: tuple(v) for k, v in raw.get("cache", {}).items()}
            values["findings"] = tuple(
                Finding(
                    **{
                        **f,
                        "versions": tuple(f.get("versions", ())),
                        "missing": tuple(f.get("missing", ())),
                        "origin": Origin(**f["origin"]) if f.get("origin") else None,
                    }
                )
                for f in raw.get("findings", [])
            )
            _emit(StageRecord(**values), raw["span"] not in parents, True)
    record = _emit(
        _record(
            opened,
            max(0, (end_ns or time.perf_counter_ns()) - start_ns) / 1e9,
            worker=sum(e["worker_cpu"] for e in envelopes),
            native=sum(e["native_cpu"] for e in envelopes),
            items=items,
            jobs=jobs,
            workers_admitted=admitted,
            cache=cache,
            workers_used=len({e["pid"] for e in envelopes}),
            dispatch_seconds=dispatch_seconds,
            wait_seconds=sum(e.get("wait_seconds", 0.0) for e in envelopes),
            busy_seconds=sum((e["end_ns"] - e["start_ns"]) / 1e9 - e.get("wait_seconds", 0.0) for e in envelopes),
            worker_rss_peak_bytes=max((e["rss"] for e in envelopes), default=0),
        ),
        True,
    )
    _stack.get()[-1].children.append(record)
    _merge_cache(_stack.get()[-1].remote_cache, cache)


class _Null:
    def add(self, *, items: int = 0, findings: Sequence[Finding] = ()) -> None:
        pass


def _merge_cache(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, pair in source.items():
        previous = target.get(key, (0, 0))
        target[key] = (previous[0] + pair[0], previous[1] + pair[1])


def count(kind: str, hit: bool) -> None:
    _counters.setdefault(kind, [0, 0])[0 if hit else 1] += 1


def counters() -> dict[str, tuple[int, int]]:
    return {k: (v[0], v[1]) for k, v in _counters.items()}


@contextmanager
def capture() -> Iterator[list[StageRecord]]:
    global _closed, _counters, _sink, _buffer, _invocation, _seen
    saved = _closed, _counters, _sink, _buffer, _invocation, _seen
    _closed, _counters, _sink, _buffer, _invocation, _seen = [], {}, None, [], uuid4().hex, set()
    token = _stack.set(())
    try:
        with _scope(_Open("job", "job")):
            yield _closed
    finally:
        _stack.reset(token)
        _closed, _counters, _sink, _buffer, _invocation, _seen = saved


def tree() -> Json:
    nodes = {r.span: {**asdict(r), "children": []} for r in _closed}
    roots: list[Any] = []
    for record in _closed:
        (nodes[record.parent]["children"] if record.parent in nodes else roots).append(nodes[record.span])
    return roots[-1] if roots else {}


def invocation() -> str:
    return _invocation
