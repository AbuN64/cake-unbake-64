"""Bounded worker execution with ordered results and retained job metrics."""

from __future__ import annotations

import atexit
import ctypes
import fcntl
import gc
import io
import os
import pickle
import resource
import select
import shutil
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import psutil
from abucache.store import absorb_touched, take_touched

from unbake import effort, store
from unbake.contracts import Config, Finding, Host, Json, Refusal, Snapshot, code_identity, digest, sha256

_KEYS_INLINE = 1  # keys of up to this many items per worker are computed here: a dispatch costs more than they do
_INLINE_AT = 1  # a gather of up to this many chunks runs here: a dispatch costs more than one chunk does
_REBALANCE = 0.5  # seconds between looks at the other commands while a pool works
_executor: ProcessPoolExecutor | None = None
_host_digest: str | None = None
_in_worker = False
_creating = threading.Lock()
_slots: list[io.IOBase] = []  # the host-wide worker slots this process holds now: one flock each, freed on death
_dir: list[Path] = []  # the slot directory this process takes from; empty under a holder that passed its cores down
_want = 0  # this command's host budget, as the last acquisition declared it
_users = 0  # gathers using the pool now: the slots are released when the last one ends
_users_lock = threading.Lock()
_fed: list[io.IOBase] = []  # slots a jobserver feeder added for a running make: a pool rebalance never gives them back
_size = 0  # workers of the current executor: the host's whole fan, forked once per command
_share = 0  # the share the last acquisition computed
_flying = 0  # chunks every gather of this process has in flight: their slots are never given back
_WINDOW = _REBALANCE * 20  # seconds without a finished chunk before the workers are looked at
_cmd: list[io.TextIOWrapper] = []  # this command's own lock file: its presence is what divides the slots fairly
_SHARED: dict[str, Snapshot] = {}  # the snapshots by key: workers are forked holding them, items name them by key
_spilled: dict[str, str] = {}  # snapshot or writes key -> the file a live pool's workers read it from, oldest first
_spill_users: dict[str, int] = {}  # spill key -> gathers of this process whose tasks name it now
_SPILL_KEEP = 8  # spill files kept past their last gather: the next dispatch of a round names the same snapshot
_spill_lock = threading.Lock()
_short: list[int] = []  # (slots got, slots wanted) of the last short share reported
_laid: list[str] = []  # in a worker, the overlay keys it laid again itself (not forked holding)
KEEP = ("declarations", "words", "spelled", "typed")  # memo kinds keyed by content alone: valid for every overlay
WARM = ("catalog", "spellings", "authority", "spellers")  # content-keyed indexes the parent builds for a round's tree:
# a spill file carries them, so warm workers read them instead of building them each (forked workers inherited them)
_carried: dict[str, dict[Any, Any]] = {}  # name -> a module's content-keyed table a spill file carries the news of
_sent: dict[str, set[Any]] = {}  # name -> the keys of that table a spill file of this command carried already
_forked: set[str] = set()  # the keys of `_SHARED` the current workers were forked with


_HARD_LIMIT = 4  # a worker is refused memory at this many times host memory_worker_bytes; that budget sizes the pool


def _init(limit: int) -> None:
    """Runs in each forked worker: its memory limit counts what it inherited, so it covers only what it adds. The host
    memory_worker_bytes is the budget that sizes the pool and trims a job's memo; the refusal sits well above it, so
    a job that needs more than the plan finishes instead of dying (the host guard freezes a runaway)."""
    global _in_worker
    os.nice(10)  # pool work yields to whatever else the host runs
    inherited = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmData:")
    )
    hard = inherited + limit * _HARD_LIMIT
    resource.setrlimit(resource.RLIMIT_DATA, (hard, hard))
    effort.mark_fork(inherited, limit)
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG: a worker never outlives the process that forked it
    for f in _slots + _cmd:  # the parent's locks are the parent's: a worker holding one pins it for life
        f.close()
    _in_worker = True


_DEAD_AFTER = 60.0  # seconds a command's lock file must be old before a free one is known to be a finished command's


_PROTOCOL = 2  # the host slot protocol: a command of any other one books other files, so it never runs beside this one
_POOL = f"unbake-pool-{_PROTOCOL}"  # the slot directory of this protocol, under XDG_RUNTIME_DIR
_IN_CHANGES = 0x8 | 0x80 | 0x100 | 0x200  # IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE | IN_DELETE: a release or a peer
_IN_FLAGS = 0o4000 | 0o2000000  # IN_NONBLOCK | IN_CLOEXEC


def _busy(path: Path) -> bool:
    """Whether a live command holds `path`. A free lock of a command that finished long ago is removed, so the files of
    thousands of past commands are never opened again by every later acquire."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        if time.time() - os.fstat(fd).st_mtime > _DEAD_AFTER:
            path.unlink(missing_ok=True)
        return False
    finally:
        os.close(fd)


def _budget(host: Host) -> int:
    """The one host fact every caller divides: the slots of this machine (workers only for a host without it)."""
    return host.slots or host.workers


def _foreign(base: Path) -> tuple[list[Path], list[Path]]:
    """(the directories other slot protocols keep their command files in, the live command files there). Protocol 1
    kept its files flat in the runtime directory."""
    dirs = [base, *(d for d in base.glob("unbake-pool-*") if d.name != _POOL and d.is_dir())]
    files = [*base.glob("unbake-cmd-*.lock"), *(p for d in dirs[1:] for p in d.glob("cmd-*.lock"))]
    return dirs, sorted(p for p in files if _busy(p))


def _pool(base: Path) -> Path:
    """The slot directory of this protocol, once no command of another protocol is live: their slots are other files,
    so running beside one would book the same cores twice. Until they finish this command sleeps on their pidfds and
    on their command files (a release unlinks one), never on a timer; they never see this protocol, so they drain."""
    if live := _foreign(base)[1]:
        named = ", ".join(str(_pid(p)) for p in live)
        reason = f"pids {named} hold host slots under another slot protocol than {_PROTOCOL}"
        effort.say(f"unbake: waiting for {reason} to finish")
        with effort.stage("pool.protocol") as span:  # the wait is a stage of its own in the tree
            span.add(items=len(live), findings=[Finding("pool.protocol", reason, blocking=False)])
            _drain(base)
    pool = base / _POOL
    pool.mkdir(exist_ok=True)
    return pool


def _drain(base: Path) -> None:
    while True:
        dirs, _ = _foreign(base)
        watch = _inotify(*dirs)  # watched before the look: a command that ends between the two is not missed
        try:
            _, live = _foreign(base)
            if not live:
                return
            fds = [fd for p in live if (fd := _pidfd(_pid(p))) is not None]
            try:
                select.select([watch, *fds], [], [])
            finally:
                for fd in fds:
                    os.close(fd)
        finally:
            os.close(watch)


def _register(pool: Path) -> Path:
    """This command's lock file, held exclusively while it has work: its presence divides the slots fairly. The
    file is unlinked on release, so one a peer unlinked in between is made again."""
    mine = pool / f"cmd-{os.getpid()}.lock"
    while True:
        if not _cmd:
            _cmd.append(mine.open("a"))
        fcntl.flock(_cmd[0], fcntl.LOCK_EX)
        try:
            if os.fstat(_cmd[0].fileno()).st_ino == mine.stat().st_ino:
                return mine
        except FileNotFoundError:
            pass
        _cmd.pop().close()


def _peers(pool: Path, mine: Path) -> list[Path]:
    """The lock files of every live command of this protocol, this one's included."""
    return sorted([mine, *(p for p in pool.glob("cmd-*.lock") if p != mine and _busy(p))])


def _pid(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[1])


def _demand(count: int, budget: int) -> None:
    """Writes the slots this command could use now and its host budget into its lock file: peers leave it no more than
    the first, and the largest budget of any live command is how many slot files there are."""
    if _cmd:
        fd = _cmd[0].fileno()
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{count} {budget}".encode(), 0)


def _declared(path: Path, want: int) -> tuple[int, int]:
    """(the slots a peer said it could use, its budget); a peer that said nothing yet is taken to want all of `want`."""
    try:
        words = path.read_text().split()
    except OSError:
        return want, want
    if len(words) != 2 or not all(w.isdigit() and int(w) > 0 for w in words):
        return want, want
    return int(words[0]), int(words[1])


def _fair(want: int, demands: dict[Path, int], mine: Path) -> int:
    """This command's max-min fair share of `want` slots: a peer that can use fewer than an equal split keeps only
    what it can use and the rest is split among the others, so no slot idles while a command has work for it."""
    left, order = want, sorted(demands, key=lambda p: (demands[p], p))
    for at, path in enumerate(order):
        others = order[at:]
        quotient, remainder = divmod(left, len(others))
        if demands[path] > quotient:  # every command from here on can use more than the equal split: split evenly
            ranked = sorted(others)
            return max(1, quotient + (ranked.index(mine) < remainder)) if mine in others else max(1, demands[mine])
        if path == mine:
            return max(1, demands[mine])
        left -= demands[path]
    return max(1, demands[mine])


def _divide(pool: Path, mine: Path, want: int) -> tuple[list[Path], int, int]:
    """(live peers, slot files, this command's fair share). Every command of the host sees the same slot files: as
    many as the largest budget any live command declared, so a command with a small budget takes any free slot of
    them (at most its budget) and never waits on a few low slots while higher ones are free."""
    peers = _peers(pool, mine)
    declared = {p: (want, want) if p == mine else _declared(p, want) for p in peers}
    size = max(budget for _, budget in declared.values())
    return peers, size, _fair(size, {p: demand for p, (demand, _) in declared.items()}, mine)


def _signal(pool: Path) -> None:
    """Wakes every waiter of the host: a write-close in the slot directory is the one event they sleep on."""
    os.close(os.open(pool / "wake", os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC, 0o600))


def _give(slot: io.IOBase) -> None:
    fcntl.flock(slot, fcntl.LOCK_UN)
    slot.close()


def _try(path: Path) -> io.FileIO | None:
    """The slot at `path` held on a read-only descriptor of its own, or None when another command holds it. A probe
    that loses closes a read-only file, which wakes nobody."""
    try:
        slot = io.FileIO(str(path), "r")
    except FileNotFoundError:
        os.close(os.open(path, os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC, 0o600))
        slot = io.FileIO(str(path), "r")
    try:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        slot.close()
        return None
    return slot


def _take(pool: Path, size: int, take: int) -> list[io.IOBase]:
    """Takes free slots of the `size` slot files without waiting until this process holds `take`: the ones it got."""
    held = {getattr(f, "name", None) for f in _slots}
    got: list[io.IOBase] = []
    for index in range(size):
        if len(_slots) >= take:
            break
        path = pool / f"slot-{index}.lock"
        if str(path) in held:
            continue
        if (slot := _try(path)) is not None:
            _slots.append(slot)
            got.append(slot)
    return got


def _pidfd(pid: int) -> int | None:
    try:
        return os.pidfd_open(pid)
    except (ProcessLookupError, PermissionError):  # gone already: what it held is free for the next attempt
        return None


def _inotify(*paths: Path) -> int:
    """One descriptor that turns readable on a write-close, a new file or a removal in any of `paths`."""
    libc = ctypes.CDLL(None, use_errno=True)
    fd = int(libc.inotify_init1(_IN_FLAGS))
    for path in paths:
        if fd < 0 or libc.inotify_add_watch(fd, os.fsencode(path), _IN_CHANGES) < 0:
            number = ctypes.get_errno()
            if fd >= 0:
                os.close(fd)
            raise OSError(number, os.strerror(number), str(path))
    return fd


def _until(pool: Path, mine: Path, attempt: Callable[[], bool], stop: int | None = None) -> None:
    """Calls `attempt` until it is true or `stop` is readable, sleeping between attempts only until something that can
    free a slot happens: a release (the write-close every holder makes), a new or finished command (its lock file),
    or the death of a live command (its pidfd). Nothing is polled and no descriptor waits on one particular slot.
    Idle workers stay up through the wait: they hold no slot (each closed the parent's locks when it forked) and
    their warm state is what the next dispatch runs on."""
    watch = _inotify(pool)  # watched before the first attempt: no release between the two is missed
    try:
        while True:
            pids = [fd for p in _peers(pool, mine) if p != mine and (fd := _pidfd(_pid(p))) is not None]
            try:  # opened before the attempt: a death after it shows as a readable pidfd
                if attempt():
                    return
                ready = select.select([watch, *pids, *(() if stop is None else (stop,))], [], [])[0]
            finally:
                for fd in pids:
                    os.close(fd)
            if stop is not None and stop in ready:
                return
            if watch in ready:
                while True:
                    try:
                        if not os.read(watch, 65536):
                            break
                    except BlockingIOError:
                        break
    finally:
        os.close(watch)


def _wait_free(pool: Path, mine: Path, want: int) -> None:
    """Blocks until any slot of the host frees and takes it."""

    def attempt() -> bool:
        return bool(_take(pool, _divide(pool, mine, want)[1], 1))

    _until(pool, mine, attempt)


def _acquire(want: int, active: int = 0, cap: int = 0) -> None:
    """All unbake processes of this user share the host slots (flock files): a command takes its share of the free
    ones (at most `cap`, the work it has, and `want`, its budget), or blocks until any one of them frees."""
    global _share, _want
    if inherited := int(os.environ.get("UNBAKE_CORES") or 0):  # a holder above this process took them for the command
        _share = min(want, inherited)
        _slots[:] = [Path(os.devnull).open() for _ in range(min(_share, cap or _share))]  # noqa: SIM115
        return
    base, start = Path(os.environ.get("XDG_RUNTIME_DIR", "")), time.perf_counter()
    if not base.is_dir():
        raise Refusal(
            Finding("worker.crash", "XDG_RUNTIME_DIR is not a directory: the worker slots live there", path=str(base))
        )
    if not _cmd or not _dir:
        _dir[:] = [_pool(base)]
    pool, _want = _dir[0], want
    mine = _register(pool)
    _demand(max(cap, active) if cap else want, want)  # what this command can use now, told to its peers
    peers, size, share = _divide(pool, mine, want)
    _share, take = share, min(share, cap) if cap else share
    # In-flight chunks retain their slots until completion; never oversubscribe a released slot.
    shed = False
    while len(_slots) > max(take, active):
        spare = next((f for f in reversed(_slots) if f not in _fed), None)
        if spare is None:  # what is left backs a make that still holds its token
            break
        _slots.remove(spare)
        _give(spare)
        shed = True
    if shed:
        _signal(pool)
    _take(pool, size, take)
    if 0 < len(_slots) < take and _short != [len(_slots), take]:  # a share short of the work shows in the tree once
        _short[:] = [len(_slots), take]
        with effort.stage("pool.short_slots") as span:
            others = [str(_pid(p)) for p in peers if p != mine]
            span.add(
                items=take - len(_slots),
                findings=[
                    Finding(
                        "pool.short_slots",
                        f"{len(_slots)} of {take} slots; the rest are held by pids {', '.join(others) or 'none live'}",
                        blocking=False,
                    )
                ],
            )
    if not _slots:  # every slot is held: wait for whichever frees first, as a stage of its own
        holders = [_pid(p) for p in peers if p != mine]
        with effort.stage("pool.wait_slot") as span:
            held = f"held by pids {', '.join(str(p) for p in holders)}" if holders else "held by no live command"
            span.add(items=len(holders), findings=[Finding("pool.wait_slot", held, blocking=False)])
            _wait_free(pool, mine, want)
        _, size, share = _divide(pool, mine, want)  # a release frees whole shares: take the rest of this one too
        _share = share
        _take(pool, size, min(share, cap) if cap else share)
    effort.waited(time.perf_counter() - start)


def _release() -> None:
    """A command holds slots only while its pool works: between phases other commands may take them."""
    for f in _slots:
        _give(f)
    for f in _cmd:
        fcntl.flock(f, fcntl.LOCK_UN)
    pooled = bool(_slots) and bool(_dir)
    _slots.clear()
    for f in _cmd:  # a finished command leaves no lock file behind
        Path(f.name).unlink(missing_ok=True)
        f.close()
    _cmd.clear()
    if pooled:
        _signal(_dir[0])


@contextmanager
def cores(host: Host, work: int = 0) -> Iterator[int]:
    """The one way anything heavy takes cores: holds this command's share of the host slots (at most `work`) for the
    block and yields how many to use, for a pool, `make -j`, a permuter or a test runner alike."""
    with _users_lock:
        if not _slots:
            _acquire(_budget(host), cap=work)
        _users_add(1)
    try:
        yield max(1, min(work or len(_slots), len(_slots)))
    finally:
        with _users_lock:
            if _users_add(-1) == 0:
                _release()


def _stop_workers() -> None:
    """Ends the current executor and its workers; the slot files stay as they are."""
    global _executor, _host_digest
    executor, _executor = _executor, None
    _host_digest = None
    _forked.clear()
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


_READ_FUTEX = {"x86_64": (0, 202), "aarch64": (63, 98)}.get(os.uname().machine)


def _awaits_work(pid: int, executor: Any) -> bool:
    """Whether the worker is blocked taking its next work item: reading the call queue pipe, or queued on that
    pipe's read lock. A worker sleeping anywhere else (a store lock, sqlite, the journal) is busy, not stalled."""
    calls: Any = getattr(executor, "_call_queue", None)
    try:
        number, first = Path(f"/proc/{pid}/syscall").read_text().split()[:2]
        reader, lock = calls._reader.fileno(), calls._rlock._semlock.handle
    except (OSError, ValueError, AttributeError):
        return False
    read, futex = _READ_FUTEX or (None, None)
    return (int(number), int(first, 16)) in {(read, reader), (futex, lock)}


def _census(executor: Any) -> dict[int, float] | None:
    """CPU seconds of each worker with everything it spawned, or None when any of them is doing something else than
    waiting for its next work item with no children (running, on a lock, a child, a disk) or is gone."""
    seconds: dict[int, float] = {}
    try:
        for pid in list(getattr(executor, "_processes", None) or ()):
            worker = psutil.Process(pid)
            if worker.status() != psutil.STATUS_SLEEPING or worker.children() or not _awaits_work(pid, executor):
                return None
            times = worker.cpu_times()
            seconds[pid] = times.user + times.system
    except psutil.Error:
        return None
    return seconds


def _stalled(executor: Any, seen: dict[int, float]) -> bool:
    """Whether every worker slept without using CPU since the last call (the first call only records)."""
    now = _census(executor)
    before = dict(seen)
    seen.clear()
    if now is None:
        return False
    seen.update(now)
    return bool(now) and now == before


def _drop_executor() -> None:
    """Ends the workers and gives the slots back; their spill files and what they were sent go with them."""
    _stop_workers()
    _forget_spills()
    _sent.clear()
    for mine in _cmd:
        Path(mine.name).unlink(missing_ok=True)
        mine.close()
    for slot in _slots:  # freed only once the workers are gone
        slot.close()
    if _slots and _dir:
        _signal(_dir[0])
    _slots.clear()
    _cmd.clear()


def _users_add(step: int) -> int:
    global _users
    _users += step
    return _users


def _flight(step: int) -> int:
    global _flying
    with _users_lock:
        _flying += step
        return _flying


def _fan(host: Host) -> int:
    """The workers one command forks: the host's worker count within its slots, whatever its share is now."""
    return max(1, min(host.workers, _budget(host)))


def _warm(config: Config) -> bool:
    """Whether this command's workers are up for this host: a dispatch then ships what they lack, never reforks."""
    return _executor is not None and _host_digest == config.host.digest


def _get_executor(config: Config, needed: frozenset[str] = frozenset(), work: int = 0) -> ProcessPoolExecutor:
    """One pool per command, forked once at the host's fan and kept warm across rounds, levels and stages: a new
    snapshot travels to the workers that lack it (`_Pickler`), and a share that grows or shrinks changes how many
    chunks are in flight, never the workers. Only a host change (or a broken pool) forks again. Workers fork
    holding the snapshots in `needed` that are in `_SHARED` then."""
    global _executor, _host_digest, _size
    held = _executor
    if held is not None and _host_digest == config.host.digest:
        return held
    with _creating:  # two threads asking at once make one executor, not two
        if _executor is not None and _host_digest != config.host.digest:
            _drop_executor()
        if _executor is None:
            if not _slots:
                _acquire(_budget(config.host), cap=work)
            _size = _fan(config.host)  # idle workers cost no slot and no CPU: the fan is forked once
            for key in [k for k in _SHARED if k not in needed]:  # workers fork holding only what they run against
                del _SHARED[key]
            _executor = ProcessPoolExecutor(
                max_workers=_size,
                mp_context=get_context("fork"),
                initializer=_init,
                initargs=(config.host.memory_worker_bytes,),
            )
            _host_digest = config.host.digest
            _forked.update(_SHARED)
            gc.freeze()  # the workers' copy-on-write pages stay shared: a collection would touch every object
            try:
                _executor.submit(int).result()  # the first job forks every worker
            finally:
                gc.unfreeze()
        return _executor


def _job(function: Callable[[Any], Any], item: Any, floor: float) -> tuple[bool, Any, Json]:
    start_ns = time.monotonic_ns()
    before = resource.getrusage(resource.RUSAGE_SELF)
    spawned = resource.getrusage(resource.RUSAGE_CHILDREN)
    with effort.capture() as records:
        try:
            with effort.scoped():
                value, ok = function(item), True
        except Exception as exc:
            here = [f for f in traceback.extract_tb(exc.__traceback__) if "/unbake/" in f.filename]
            if here:  # the traceback does not cross the process boundary, a note does
                exc.add_note(f"at {Path(here[-1].filename).stem}:{here[-1].name}:{here[-1].lineno}")
            value, ok = exc, False
    after = resource.getrusage(resource.RUSAGE_SELF)
    reaped = resource.getrusage(resource.RUSAGE_CHILDREN)
    # every program the job ran and waited for, however deep (a nested pool runs inline here and keeps only its slow
    # records): the job's native work is all of it, not only the native records that reached this capture
    native = reaped.ru_utime + reaped.ru_stime - spawned.ru_utime - spawned.ru_stime
    return (
        ok,
        value,
        {
            "pid": os.getpid(),
            "worker_cpu": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
            "native_cpu": max(native, records[-1].native_cpu_seconds),
            "rss": after.ru_maxrss * 1024,
            "cache": dict(records[-1].cache),
            "wait_seconds": records[-1].wait_seconds,
            "records": [asdict(r) for r in records if r.wall_seconds >= floor or r.findings or r.wait_seconds],
            "start_ns": start_ns,
            "end_ns": time.monotonic_ns(),
        },
    )


_INLINE_BYTES = 64 << 10  # an overlay's writes up to this travel inside each task; larger ones in one spill file


def _spill_dir(config: Config) -> Path:
    """This command's spill files, under the project's own state (never a RAM tmpfs). A dead command's are removed
    the first time a later one spills."""
    base = config.project.root / ".unbake/pool"
    mine = base / str(os.getpid())
    if not mine.is_dir():
        mine.mkdir(parents=True, exist_ok=True)
        atexit.register(shutil.rmtree, mine, True)
        for other in base.iterdir():
            if other != mine and other.name.isdigit() and not psutil.pid_exists(int(other.name)):
                shutil.rmtree(other, ignore_errors=True)
    return mine


def _spill(config: Config, name: str, value: Callable[[], bytes]) -> str:
    """The file holding `value()` for the workers, written once per name per command: every task names it, each
    worker reads it once."""
    with _spill_lock:
        if (path := _spilled.pop(name, None)) is None or not Path(path).exists():
            target = _spill_dir(config) / f"{name}.pickle"
            part = target.with_suffix(".part")
            part.write_bytes(value())
            part.replace(target)
            path = str(target)
            effort.count("pool.spill", False)
        else:
            effort.count("pool.spill", True)
        _spilled[name] = path  # newest last
        return path


def carry(name: str, table: dict[Any, Any]) -> None:
    """Registers a module's content-keyed table (a scan per text): each spill file carries the entries the parent
    made since the last one, so warm workers never redo what the parent did."""
    _carried[name] = table


def _warmth() -> dict[str, Any]:
    """What a spill file hands warm workers beside the snapshot: the parent's indexes of the WARM kinds and the news of
    every carried table."""
    news = {}
    for name, table in _carried.items():
        sent = _sent.setdefault(name, set())
        news[name] = {k: v for k, v in list(table.items()) if k not in sent}
        sent.update(news[name])
    return {"memo": effort.held(WARM), "tables": news}


def _adopt(warmth: dict[str, Any]) -> None:
    effort.adopt(warmth["memo"], WARM)
    for name, news in warmth["tables"].items():
        if name in _carried:
            for key, value in news.items():
                _carried[name].setdefault(key, value)


def _hold_spills(names: Collection[str], step: int) -> None:
    """A gather names its spill files while its tasks may read them; at its end, files no live gather names go,
    except the newest few (the next dispatch of a round names the same snapshot)."""
    with _spill_lock:
        for name in names:
            _spill_users[name] = _spill_users.get(name, 0) + step
            if _spill_users[name] <= 0:
                del _spill_users[name]
        if step > 0:
            return
        idle = [n for n in _spilled if n not in _spill_users]
        for name in idle[: max(0, len(idle) - _SPILL_KEEP)]:
            Path(_spilled.pop(name)).unlink(missing_ok=True)


def _forget_spills() -> None:
    """Every spill file of this command goes: its workers are gone."""
    with _spill_lock:
        for path in _spilled.values():
            Path(path).unlink(missing_ok=True)
        _spilled.clear()
        _spill_users.clear()


class _Pickler(pickle.Pickler):
    """Items travel without the snapshot they run against. Before the pool forks (`ship` false), a snapshot goes into
    `_SHARED` and the workers fork holding it; an overlay of one travels as its base and its writes, laid again in
    each worker. Once the workers are up (`ship`), nothing forks again: an overlay travels as its base's reference and
    its writes (inline when small, else one spill file), any other snapshot they were not forked with (the next
    round's capture) as one spill file of the whole of it; each file is written once and read once per worker, so
    rounds, levels and stages all run on the same warm workers."""

    def __init__(self, stream: io.BytesIO, needed: set[str], blobs: dict[str, Any] | None = None, ship: bool = False):
        super().__init__(stream, protocol=5)
        self.needed = needed
        self.blobs = {} if blobs is None else blobs  # per dispatch: each overlay's writes and each spill's file, once
        self.ship = ship
        self.spills: set[str] = set()  # the spill files this pickler's tasks name

    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, Snapshot):
            return self._ref(obj)
        return None

    def _ref(self, obj: Snapshot) -> Any:
        key = _key(obj)
        if key in _forked:
            self.needed.add(key)  # a refork after a broken pool forks holding it again
            return key
        made = effort.peek(("parent", obj.digest))  # (snapshot, writes) it was laid over with
        if made is not None:
            blob = self.blobs.get(obj.digest)
            if blob is None:
                blob = self.blobs[obj.digest] = pickle.dumps(dict(made[1]), protocol=5)
            if len(blob) <= _INLINE_BYTES:  # small writes travel in each task over the base's reference
                return ("over", self._ref(made[0]), key, obj.digest, blob)
            if self.ship:  # large writes: one file, laid again once per worker over the base it holds
                return ("over", self._ref(made[0]), key, obj.digest, self._spill(obj, f"w{obj.digest}", made[1]))
        if self.ship:  # a snapshot of its own (a capture): the whole of it, read once per worker
            return ("snap", key, self._spill(obj, f"s{key}", obj))
        _SHARED[key] = obj
        self.needed.add(key)
        return key

    def _spill(self, obj: Snapshot, name: str, value: Any) -> str:
        """The file of one snapshot or writes, with the parent's warm indexes beside it."""
        self.spills.add(name)
        if (path := self.blobs.get(f"file:{name}")) is None:
            produce = lambda: pickle.dumps((dict(value) if isinstance(value, Mapping) else value, _warmth()), 5)  # noqa: E731
            path = self.blobs[f"file:{name}"] = _spill(obj.config, name, produce)
        return str(path)


def _key(snapshot: Snapshot) -> str:
    return sha256((str(snapshot.config.project.root) + snapshot.digest).encode())


def _chain(ref: Any) -> list[tuple[str, str | None]]:
    """(key, digest) of every snapshot a reference lays, top first; a forked key has no digest to name."""
    if isinstance(ref, str):
        return [(ref, None)]
    if ref[0] == "snap":
        return [(ref[1], None)]
    return [(ref[2], ref[3]), *_chain(ref[1])]


def _lay(ref: Any) -> Snapshot:
    """The snapshot a reference names, laid in this worker once and held in `_SHARED` until another one comes."""
    if isinstance(ref, str):
        return _SHARED[ref]
    if ref[0] == "snap":
        _, key, path = ref
        if key not in _SHARED:
            _SHARED[key], warmth = pickle.loads(Path(path).read_bytes())
            _adopt(warmth)
            _laid.append(key)
        return _SHARED[key]
    from unbake import layout

    _, base, key, digest_, payload = ref
    if key not in _SHARED:
        under = _lay(base)  # first: the writes' file carries the newer indexes
        if isinstance(payload, bytes):
            writes = pickle.loads(payload)
        else:
            writes, warmth = pickle.loads(Path(payload).read_bytes())
            _adopt(warmth)
        _SHARED[key] = layout.overlay(under, writes)
        _laid.append(key)
        if _SHARED[key].digest != digest_:
            raise RuntimeError(f"an overlay laid again in a worker has digest {_SHARED[key].digest}")
    return _SHARED[key]


class _Unpickler(pickle.Unpickler):
    def persistent_load(self, pid: Any) -> Snapshot:
        if isinstance(pid, tuple):  # a snapshot the worker was not forked holding: laid again once per worker
            chain = _chain(pid)
            if chain[0][0] not in _SHARED:
                # a new one (the next wave's writes, the next round's tree): what this worker laid before and built
                # for it (its indexes, pins and views) is never read again, unless the new one is laid over it
                keep = {k for k, _ in chain}
                for old in [k for k in _laid if k not in keep]:
                    gone = _SHARED.pop(old, None)
                    if gone is not None:
                        effort.release(gone.digest)
                _laid[:] = [k for k in _laid if k in keep]
                effort.trim_to_fork(keep=(*KEEP, *WARM), digests={_SHARED[k].digest for k in keep if k in _SHARED})
                # an overlay this worker laid for an earlier dispatch is never laid from its memo: the same base and
                # writes may lay another one now (the files it reads changed on disk between dispatches, as setup
                # writes them), and the parent's digest is the one that holds
                effort.forget("overlay")
            return _lay(pid)
        return _SHARED[pid]


def _chunk(
    function: Callable[..., Any], items: Sequence[Any], floor: float, sink: tuple[Any, ...] | None
) -> list[tuple[bool, Any, Json]]:
    outcomes = [_job(function, item, floor) for item in items]
    if sink and outcomes:
        clean = len(sink) > 3 and sink[3]  # a stable key stores only a result with nothing to report
        before = resource.getrusage(resource.RUSAGE_SELF)
        with effort.capture() as written, effort.stage("pool.sink"):
            store.put_many(
                sink[0],
                sink[1],
                [
                    (key, pickle.dumps(value, protocol=5))
                    for (ok, value, _), key in zip(outcomes, sink[2], strict=True)
                    if ok and key is not None and not (clean and value)
                ],
                named=True,
            )
        after = resource.getrusage(resource.RUSAGE_SELF)
        ok, value, last = outcomes[-1]  # the write belongs to no job: the chunk's last envelope carries its cpu
        outcomes[-1] = (
            ok,
            value,
            {
                **last,
                "worker_cpu": last["worker_cpu"] + after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                "records": [*last["records"], *(asdict(r) for r in written if r.kind == "stage")],
            },
        )
    if outcomes:  # the keys this chunk read or wrote travel back, so the parent writes one manifest per command
        outcomes[-1] = (*outcomes[-1][:2], {**outcomes[-1][2], "touched": take_touched()})
    return outcomes


def _dispatch(blob: bytes, floor: float) -> list[tuple[bool, Any, Json]]:
    """One chunk travels as its own pickle, so a worker only unpacks the items it runs and stores their results."""
    function, items, sink = _Unpickler(io.BytesIO(blob)).load()
    return _chunk(function, items, floor, sink)


def _raise_failure(config: Config, name: str, index: int, exc: Any) -> None:
    if isinstance(exc, Refusal):
        raise exc
    memory = isinstance(exc, MemoryError)
    raise Refusal(
        Finding(
            "worker.memory" if memory else "worker.crash",
            reason=f"Stage {name} item {index} "
            + (f"exceeded worker memory: {exc!r}" if memory else f"failed: {exc!r}")
            + "".join(f" {note}" for note in getattr(exc, "__notes__", ())),
            origin=config.host.origins["resources.memory_worker_bytes"] if memory else None,
        )
    ) from exc


def _stall(pending: dict[Future[Any], tuple[Any, list[int]]]) -> None:
    """Pending work that no worker is running will never come back: fail loudly and give the slots back."""
    run, picked = next(iter(pending.values()))
    for worker in list(
        (getattr(_executor, "_processes", None) or {}).values()
    ):  # idle by proof: a shutdown would wait on them
        worker.kill()
    _drop_executor()
    raise Refusal(
        Finding(
            "worker.stalled",
            reason=f"Stage {run.name} item {picked[0]}: {len(pending)} chunks are pending and every worker waited "
            f"for work without using CPU for {_WINDOW:.0f} seconds",
        )
    )


def _identity(run_function: Callable[..., Any], key: Callable[[Any], str | None]) -> str:
    """What a stored result also depends on: the job's code, unless its key names every input (`stable`)."""
    return "" if getattr(key, "stable", False) else code_identity(run_function)


def _keys(key: Callable[[Any], str | None], items: Sequence[Any], identity: str) -> list[str | None]:
    return [None if (named := key(item)) is None else digest((named, identity)) for item in items]


def _keys_job(item: tuple[Any, ...]) -> list[str | None]:
    return _keys(*item)


class _Run:
    """One group of a gather: its items, the cache keys of those, and the indexes the workers must still compute."""

    def __init__(
        self,
        name: str,
        function: Callable[..., Any],
        items: Sequence[Any],
        key: Callable[[Any], str | None] | None = None,
    ):
        self.name, self.function, self.items, self.key = name, function, items, key
        self.sealed = False
        self.outcomes: list[tuple[bool, Any, Json] | None] = [None] * len(items)
        self.keys: list[str | None] = []
        self.whole: str | None = None
        self.todo = list(range(len(items)))
        self.submitted, self.done = 0, 0
        self.first, self.last = (
            0,
            0,
        )  # when its first chunk went out and its last came back: its own wall, not the gather's

    def resolve(self, config: Config, keys: list[str | None]) -> None:
        """With its keys known, the parent resolves the hits itself: a warm item never travels to a worker."""
        self.keys = keys
        self.whole = digest(keys) if None not in keys else None
        # a fully warm group is one entry
        blob = store.get(config, self.name + ".group", self.whole, named=True) if self.whole else None
        self.sealed = bool(blob)  # a group warm entry by entry is sealed too, so the next run reads one entry
        whole, rows = store.loads(blob)
        if blob and not whole:
            self.sealed = False
        warm = rows if whole else [k and store.get(config, self.name, k, named=True) for k in keys]
        for index, found in enumerate(warm):
            ok, value = store.loads(found) if found else (False, None)
            effort.count(self.name, ok)
            if ok:
                self.outcomes[index] = (True, value, {})
        self.todo = [i for i, outcome in enumerate(self.outcomes) if outcome is None]
        self.done = len(self.items) - len(self.todo)

    def results(self) -> list[tuple[bool, Any, Json]]:
        return [o for o in self.outcomes if o is not None]

    def seal(self, config: Config) -> None:
        stable = bool(getattr(self.key, "stable", False))  # a failure is never sealed under a code-free key
        if (
            self.whole
            and (self.todo or not self.sealed)
            and all(o and o[0] and not (stable and o[1]) for o in self.outcomes)
        ):
            blobs = [pickle.dumps(o[1], protocol=5) for o in self.results()]
            store.put(config, self.name + ".group", self.whole, pickle.dumps(blobs), named=True)

    def sink(self, config: Config, picked: Sequence[int]) -> tuple[Any, ...] | None:
        if not self.key:
            return None
        return (config, self.name, [self.keys[i] for i in picked], bool(getattr(self.key, "stable", False)))

    def settle(self, picked: Sequence[int], outcomes: Sequence[tuple[bool, Any, Json]]) -> None:
        for index, outcome in zip(picked, outcomes, strict=True):
            meta = dict(outcome[2])
            absorb_touched(meta.pop("touched", {}))
            self.outcomes[index] = (outcome[0], outcome[1], meta)
        self.done, self.last = self.done + len(picked), time.monotonic_ns()
        effort.progress(self.name, self.done, len(self.items))


def _resolve(config: Config, runs: Sequence[_Run]) -> None:
    """The keys of every keyed group, computed by the workers in chunks (the parent when few), then the hits."""
    keyed = [(r, r.key) for r in runs if r.key]
    if sum(len(r.items) for r, _ in keyed) <= _KEYS_INLINE * config.host.workers or _in_worker:
        for run, key in keyed:
            run.resolve(config, _keys(key, run.items, _identity(run.function, key)))
        return
    groups = [
        (
            run.name + ".keys",
            _keys_job,
            [(key, run.items[lo : lo + size], _identity(run.function, key)) for lo in range(0, len(run.items), size)],
        )
        for run, key in keyed
        for size in [max(1, len(run.items) // (config.host.workers * 32))]
    ]
    for (run, _), chunks in zip(keyed, gather(config, groups), strict=True):
        run.resolve(config, [key for chunk in chunks for key in chunk])


def map(
    config: Config,
    name: str,
    function: Callable[[Any], Any],
    items: Sequence[Any],
    key: Callable[[Any], str | None] | None = None,
) -> list[Any]:
    return gather(config, [(name, function, items, key)])[0]


def gather(config: Config, groups: Sequence[tuple[Any, ...]]) -> list[list[Any]]:
    """Runs independent (name, function, items[, key]) groups in one dispatch loop, so a small group never idles the
    workers. A keyed group resolves its cache hits in the parent and dispatches only its misses. Every chunk is
    ready at once: as many are in flight as the slots held allow, and a finished one is replaced at once."""
    with effort.stage("pool.map"):
        start_ns, waited, current, rebalanced = time.monotonic_ns(), 0.0, (0, 0), time.monotonic()
        runs = [_Run(*group) for group in groups]
        total, admitted, joined = 0, 0, False
        pending: dict[Future[list[tuple[bool, Any, Json]]], tuple[_Run, list[int]]] = {}
        quiet: dict[int, float] = {}  # the workers' CPU at the last window that finished no chunk
        spills: set[str] = set()  # the spill files this gather's tasks name: kept while it runs
        try:
            _resolve(config, runs)
            total = sum(len(r.todo) for r in runs)
            inline = total <= _INLINE_AT or _in_worker  # nothing to fan out: the jobs run here, one at a time
            width = 1
            floor = config.host.serial_seconds
            fan = max(1, min(config.host.workers, _budget(config.host)))
            size = {r: 1 if inline else max(1, len(r.todo) // (fan * 32)) for r in runs}
            needed: set[str] = set()
            tasks = [(r, r.todo[lo : lo + size[r]]) for r in runs for lo in range(0, len(r.todo), size[r])]
            blobs = [b""] * len(tasks)
            if not inline:  # the snapshots in the items are known before the workers are forked
                writes: dict[str, Any] = {}  # an overlay's writes are pickled once for every task of this dispatch
                ship = _warm(config)  # workers up: what they lack travels to them, they never fork again
                for n, (run, picked) in enumerate(tasks):
                    stream = io.BytesIO()
                    pickler = _Pickler(stream, needed, writes, ship)
                    pickler.dump((run.function, [run.items[i] for i in picked], run.sink(config, picked)))
                    blobs[n] = stream.getvalue()
                    spills |= pickler.spills
                _hold_spills(spills, 1)
                with _users_lock:
                    if not _slots:  # released after the last gather: take the share again, or wait for any free slot
                        _acquire(_budget(config.host), cap=len(tasks))
                    _users_add(1)
                    joined = True
                # the executor is taken once the slots are held: waiting for a slot stops idle workers, and a held
                # executor taken before that wait would be the one stopped
                executor = _get_executor(config, frozenset(needed), len(tasks))
                width = min(len(_slots), _size or len(_slots))
            admitted = 0 if inline else width
            queue = iter(zip(tasks, blobs, strict=True))
            left = len(tasks)  # chunks not submitted yet
            while True:
                while len(pending) < width and (task := next(queue, None)):
                    (run, picked), blob = task
                    left -= 1
                    if inline:
                        future: Future[list[tuple[bool, Any, Json]]] = Future()
                        pending[future] = (run, picked)
                        future.set_result(
                            _chunk(run.function, [run.items[i] for i in picked], floor, run.sink(config, picked))
                        )
                    else:
                        pending[executor.submit(_dispatch, blob, floor)] = (run, picked)
                    _flight(1)
                    run.submitted, run.first = run.submitted + len(picked), run.first or time.monotonic_ns()
                if not pending:
                    break
                blocked = time.perf_counter()
                completed, _ = wait(pending, timeout=None if inline else _WINDOW, return_when=FIRST_COMPLETED)
                waited += time.perf_counter() - blocked
                if not completed and _stalled(executor, quiet):
                    _stall(pending)
                if completed:
                    quiet.clear()
                for future in completed:
                    run, picked = pending.pop(future)
                    _flight(-1)
                    current = (runs.index(run), picked[0])
                    try:
                        outcomes = future.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as exc:
                        outcomes = [(False, exc, {})] * len(picked)
                    run.settle(picked, outcomes)
                if not inline:
                    if time.monotonic() - rebalanced >= _REBALANCE:  # peers are probed now and then, not per chunk
                        rebalanced = time.monotonic()
                        with _users_lock:  # never hold a slot no worker of this pool can use: more than the
                            # executor forked, or more than the chunks still to run, go back to the other commands;
                            # a share that grew is used at once by the same workers (they were forked at the fan)
                            more = len(pending) + left
                            _acquire(_budget(config.host), _flying, max(1, min(more, _size or len(tasks))))
                            width = min(len(_slots), _size or len(_slots))
                    admitted = max(admitted, width)
        except BrokenProcessPool as exc:
            _drop_executor()
            _raise_failure(config, runs[current[0]].name, current[1], exc)
        finally:
            _flight(-len(pending))
            _hold_spills(spills, -1)
            if joined:
                with _users_lock:
                    if _users_add(-1) == 0:
                        _release()
            for run in runs:
                effort.record_pool(
                    run.name,
                    len(run.items),
                    jobs=run.submitted,
                    admitted=admitted,
                    start_ns=run.first or start_ns,
                    end_ns=run.last or start_ns,
                    envelopes=[o[2] for o in run.outcomes if o and o[2]],
                    dispatch_seconds=(time.monotonic_ns() - start_ns) / 1e9 - waited,
                )
        for run in runs:
            run.seal(config)
            for index, outcome in enumerate(run.outcomes):
                if outcome is not None and not outcome[0]:
                    _raise_failure(config, run.name, index, outcome[1])
        return [[outcome[1] for outcome in run.results()] for run in runs]


def _extra_slot() -> io.IOBase | None:
    """Takes one more free host slot for this process if one is free now and its fair share has room (never waits):
    its file, None otherwise."""
    with _users_lock:
        if not _dir or not _cmd:
            return None
        pool = _dir[0]
        _, size, share = _divide(pool, Path(_cmd[0].name), _want)
        if len(_slots) >= share:
            return None
        got = _take(pool, size, len(_slots) + 1)
        _fed.extend(got)
        return got[0] if got else None


def _feed(tokens: int, stop: int, taken: list[io.IOBase]) -> None:
    """Gives a running GNU make one more job token whenever a host slot frees, so a build that started on a small share
    grows into the cores other commands release instead of staying at its first grant. Sleeps on the host's slot
    events and on `stop` (readable once make returned), never on a timer."""

    def attempt() -> bool:
        while (slot := _extra_slot()) is not None:
            taken.append(slot)  # the feeder's own: exactly these go back when make returns
            os.write(tokens, b"+")
        return False

    _until(_dir[0], Path(_cmd[0].name), attempt, stop)


@contextmanager
def jobserver(host: Host, work: int = 0) -> Iterator[tuple[int, dict[str, str], tuple[int, ...]]]:
    """make's share of the host slots as a GNU make jobserver: yields (first grant, env, fds to pass). The grant is
    only where the build starts: a feeder adds a job token whenever another host slot frees. With no slot files
    (no host pool) MAKEFLAGS pins -j to the grant and no fds pass. make itself must get no -j of its own."""
    with cores(host, work) as j:
        if not _dir or not _cmd:
            yield j, {"UNBAKE_CORES": str(j), "MAKEFLAGS": f"-j{j}"}, ()
            return
        read, write = os.pipe()
        os.write(write, b"+" * (j - 1))  # make itself holds the first job token
        stopped, stop = os.pipe()
        taken: list[io.IOBase] = []
        feeder = threading.Thread(target=_feed, args=(write, stopped, taken), daemon=True)
        feeder.start()
        try:
            yield j, {"UNBAKE_CORES": str(j), "MAKEFLAGS": f"-j{j} --jobserver-auth={read},{write}"}, (read, write)
        finally:
            os.close(stop)  # the feeder's end reads EOF: it returns
            feeder.join()
            os.close(stopped)
            os.close(read)
            os.close(write)
            with _users_lock:  # the slots the feeder added go back as make returns, not when the command ends
                given = False
                for slot in taken:
                    if slot in _fed:
                        _fed.remove(slot)
                    if slot in _slots:  # a pool of this process may have given it back already
                        _slots.remove(slot)
                        _give(slot)
                        given = True
                if given:
                    _signal(_dir[0])


def _is_make(program: str) -> bool:
    return Path(program).name in ("make", "gmake")


def main(argv: Sequence[str]) -> int:
    """`python -m unbake.pool [--work N] -- CMD...`: the shell form of `cores`. Only a CMD whose arguments contain `{j}`
    holds a share of the host slots (as `{j}` and UNBAKE_CORES); any other CMD runs directly at nice 10, taking none."""
    import subprocess

    from unbake import config as configuration

    args, work = list(argv), 0
    if args[:1] == ["--work"]:
        work, args = int(args[1]), args[2:]
    command = args[1:] if args[:1] == ["--"] else args
    if not any("{j}" in a for a in command):
        return subprocess.run(command, preexec_fn=lambda: os.nice(10)).returncode
    try:
        host = configuration.load_host(Path(os.environ.get("UNBAKE_HOST", Path.home() / ".config/unbake/unbake.toml")))
    except Refusal:  # no usable host file: run the command alone at nice 10, holding no slots
        return subprocess.run(
            [a.replace("{j}", "1") for a in command],
            env={**os.environ, "UNBAKE_CORES": "1"},
            preexec_fn=lambda: os.nice(10),
        ).returncode
    if _is_make(command[0]) and "-j{j}" in command:  # elastic: the jobserver is fed from the slot pool
        with jobserver(host, work) as (j, env, fds):
            if not fds:
                return subprocess.run(
                    [a.replace("{j}", str(j)) for a in command],
                    env={**os.environ, **env},
                    preexec_fn=lambda: os.nice(10),
                ).returncode
            nice = shutil.which("nice")  # the feeder thread is alive: no preexec_fn, it is not fork-safe with threads
            return subprocess.run(
                [*([nice, "-n", "10"] if nice else []), *(a for a in command if a != "-j{j}")],
                env={**os.environ, **env},
                pass_fds=fds,
            ).returncode
    with cores(host, work) as j:
        child = subprocess.run(
            [a.replace("{j}", str(j)) for a in command],
            env={**os.environ, "UNBAKE_CORES": str(j)},
            preexec_fn=lambda: os.nice(10),
        )
    return child.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
