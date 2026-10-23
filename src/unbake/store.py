"""Lock-free content cache, append-only streams, exclusive project locks and scratch dirs."""

from __future__ import annotations

import fcntl
import functools
import importlib.util
import json
import os
import pickle
import re
import shutil
import stat
import tempfile
import time
import traceback
import uuid
import zlib
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from abucache.store import CacheFull, ContentCache, absorb_touched, everything, open_cache

from unbake import effort, process
from unbake.contracts import Config, Finding, Host, Json, Refusal, pickled, sha256


def write(path: Path, content: bytes) -> bool:
    """The one way a file reaches the project: written atomically and only when its bytes differ (mtime kept)."""
    try:
        if path.read_bytes() == content:
            return False
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{uuid.uuid4().hex}.part")  # the name alone may already sit near the 255-byte limit
    part.write_bytes(content)
    part.replace(path)
    effort.forget("pin", "includes", "closure", "listing")  # the memoised reads of project files are stale now
    return True


def unshare(root: Path) -> int:
    """Give every file under `root` that other trees may share (a hard link, or a read-only file from a shared setup
    cache) a private writable copy, made beside it and renamed over, so a tool that writes in place (splat) can never
    write through a link into the cache or a neighbouring tree. Returns the number of files replaced."""
    replaced = 0
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or (info.st_nlink == 1 and info.st_mode & 0o200):
                continue
            part = path.with_name(f".{uuid.uuid4().hex}.part")
            shutil.copyfile(path, part)
            part.chmod(info.st_mode & 0o7777 | 0o200)
            part.replace(path)
            replaced += 1
    return replaced


def _packages() -> list[Path]:
    """The directories of the installed code every cache entry depends on: unbake, and recipemaker when present."""
    found = [Path(__file__).resolve().parent]
    spec = importlib.util.find_spec("recipemaker")
    if spec and spec.origin:
        found.append(Path(spec.origin).resolve().parent)
    return found


def _digest_of(folders: Iterable[Path]) -> str:
    return sha256(b"".join(f.read_bytes() for folder in folders for f in sorted(folder.rglob("*.py"))))


@functools.cache
def code() -> str:
    """The content digest of the installed tool code, read once per process."""
    return _digest_of(_packages())


def identity() -> str:
    """This process's code identity: the content digest of the installed packages, then their directories."""
    return ";".join([code(), *map(str, _packages())])


def installed(code: str) -> bool:
    """Whether an identity still exists anywhere: its directories are there and still hold the same code."""
    digest, *folders = code.split(";")
    paths = [Path(f) for f in folders]
    return bool(paths) and all(p.is_dir() for p in paths) and _digest_of(paths) == digest


def finish(config: Config) -> None:
    """End of a command: the one manifest of the keys it touched, and the collect of unreachable entries."""
    content(config).finish(config.project.root, identity(), installed, config.host.cache_keep_commits)


def finish_command(config: Config) -> None:
    """The one boundary every command runner (unbake, recipe) calls on success, refusal or crash alike: a failure of the
    bookkeeping is reported on stderr and never changes the command's result."""
    try:
        finish(config)
    except Exception:
        traceback.print_exc()


_PACKED = b"\x02"  # format tag, also the cache folder suffix: a value without it (an older entry) reads as a miss


def _pack(value: bytes) -> bytes:
    return _PACKED + zlib.crc32(value).to_bytes(4, "big") + zlib.compress(value, 1)


def loads(blob: bytes | None) -> tuple[bool, Any]:
    """(True, the value) of a pickled cache entry; (False, None) when it is absent or another build's shape."""
    if not blob:
        return False, None
    try:
        return True, pickle.loads(blob)
    except Exception:
        return False, None


@contextmanager
def _refusing() -> Iterator[None]:
    """A write the full cache refuses becomes a plain finding."""
    try:
        yield
    except CacheFull as full:
        raise Refusal(
            Finding("cache.full", str(full), action="raise cache.max_bytes in the host config or free the disk")
        ) from full


# a tool's output named by all its inputs: every unbake build shares it. A view is named by its reads, the preprocessor
# command, the tool bytes and view.py's own bytes (never the rest of the tool), its chunks by content; a file's parse
# (tree.file) by its text, flags, headers, preprocessor and tree.py's own bytes above its tree section; the
# preprocessor's output (tree.expanded, its texts tree.text by content) by the same inputs and the preprocessing
# functions' source alone, a scan of it (tree.scan) by its text
# the compiled extents of a module's data (report) by its text and version, so a tool update keeps them
_NAMED = frozenset(
    {"unit", "object", "bytes", "view", "view.chunk", "tree.file", "tree.expanded", "tree.text", "tree.scan", "extents"}
)


class Packed(ContentCache):
    """The host's content cache with every value zlib level 1 packed behind a format tag, so reads stay write free.
    A key is this tool code's own unless its kind or caller names every input: another build never reads it."""

    def __init__(self, inner: ContentCache, named: bool = False) -> None:
        self._inner = inner
        self.directory = inner.directory
        self._named = named

    def _key(self, kind: str, key: str) -> str:
        if self._named or kind in _NAMED:
            return key
        return sha256(f"{code()}:{key}".encode())

    def get(self, kind: str, key: str) -> bytes | None:
        raw = self._inner.get(kind, self._key(kind, key))
        if raw is None or raw[:1] != _PACKED or len(raw) < 5:
            return None
        try:
            value = zlib.decompress(raw[5:])
        except zlib.error:
            return None
        return value if zlib.crc32(value).to_bytes(4, "big") == raw[1:5] else None  # a foreign or torn entry is a miss

    def put(self, kind: str, key: str, value: bytes) -> None:
        with _refusing():
            self._inner.put(kind, self._key(kind, key), _pack(value))

    def put_many(self, items: Iterable[tuple[str, str, bytes]]) -> None:
        with _refusing():
            self._inner.put_many((kind, self._key(kind, key), _pack(value)) for kind, key, value in items)

    def count(self, kind: str, hit: bool) -> None:
        self._inner.count(kind, hit)

    def finish(
        self, root: Path, identity: str, installed: Callable[[str], bool] = everything, keep_commits: int = 20
    ) -> None:
        self._inner.finish(root, identity, installed, keep_commits)

    def cached(self, kind: str, key: str, produce: Callable[[], bytes]) -> bytes:
        value = self.get(kind, key)
        hit = value is not None
        if value is None:
            value = produce()
            self.put(kind, key, value)
        self.count(kind, hit)
        return value


def content_directory(host: Host) -> Path:
    """The host content cache's folder, beside cache.dir and named for the entry format."""
    return host.cache_dir.with_name(f"{host.cache_dir.name}.v{_PACKED[0]}")


def content(config: Config, named: bool = False) -> Packed:
    """The host content cache beside cache.dir, in a folder named for the entry format: older installs never read it.
    `named`: the caller's keys name its code (pool jobs), so they are kept as given."""
    return Packed(open_cache(content_directory(config.host), config.host.cache_max_bytes, effort.count), named)


_CLOCK: Callable[[], float] = time.monotonic  # replaced by tests so a deadline passes without waiting
_PAUSE: Callable[[float], None] = time.sleep
_POLL_S = 0.2


BEAT = 15.0  # seconds between the lines a blocked command prints


def _holders(path: Path) -> str:
    """Who holds a flock on `path` now: "pid N (command)" per holder from /proc/locks, "" when none is found."""
    try:
        inode = path.stat().st_ino
        lines = Path("/proc/locks").read_text().splitlines()
    except OSError:
        return ""
    found = []
    for line in lines:
        words = line.split()
        if len(words) > 5 and words[1] == "FLOCK" and words[5].rpartition(":")[2] == str(inode):
            pid = words[4]
            try:
                command = (Path("/proc") / pid / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                command = "?"
            found.append(f"pid {pid} ({command.strip()[:100]})")
    return ", ".join(found)


def _blocked(fd: int, flags: int, path: Path, wait_s: float | None) -> bool:
    """Take the contended lock, printing who holds it at once and every BEAT seconds; False when `wait_s` ran out."""
    began = last = _CLOCK()
    first = True
    while True:
        try:
            fcntl.flock(fd, flags | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            now = _CLOCK()
            if wait_s is not None and now - began >= wait_s:
                return False
            if first or now - last >= BEAT:
                first, last = False, now
                effort.say(f"waiting for lock {path.name} held by {_holders(path) or 'unknown'} ({now - began:.0f}s)")
            _PAUSE(_POLL_S)


@contextmanager
def _flock(path: Path, flags: int, *, wait: bool = True, wait_s: float | None = None) -> Iterator[int | None]:
    """Hold the lock; a contended lock yields None when `wait` is false or `wait_s` ran out, else blocks and records
    the wait."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fd = handle.fileno()
        try:
            fcntl.flock(fd, flags | fcntl.LOCK_NB)
        except BlockingIOError:
            if not wait:
                handle.close()
                yield None
                return
            start = time.perf_counter()
            with effort.stage("store.wait_lock"):  # a blocked command shows what it waited on, never a silent gap
                taken = _blocked(fd, flags, path, wait_s)
            effort.waited(time.perf_counter() - start, "lock", path.name)
            if not taken:
                yield None
                return
        _HELD_FDS.add(fd)
        try:
            yield fd
        finally:
            _HELD_FDS.discard(fd)


_HELD_FDS: set[int] = set()  # the descriptors whose flock this process holds now


def drop_inherited() -> None:
    """A forked worker lets go of the flocks it inherited without unlocking them for the parent: each descriptor is
    pointed at /dev/null, so the lock lives exactly as long as the parent holds it. A warm worker kept them for life,
    and the parent then waited forever on its own journal lock."""
    if not _HELD_FDS:
        return
    null = os.open(os.devnull, os.O_RDWR)
    try:
        for fd in _HELD_FDS:
            os.dup2(null, fd)
    finally:
        os.close(null)
    _HELD_FDS.clear()


def cached(config: Config, kind: str, key: str, produce: Callable[[], bytes]) -> bytes:
    return content(config).cached(kind, key, produce)


def cached_object[T](config: Config, kind: str, key: str, produce: Callable[[], T]) -> T:
    return pickled(partial(cached, config), kind, key, produce)


def get(config: Config, kind: str, key: str, named: bool = False) -> bytes | None:
    return content(config, named).get(kind, key)


def put(config: Config, kind: str, key: str, value: bytes, named: bool = False) -> None:
    content(config, named).put(kind, key, value)


def touch(config: Config, kind: str, keys: Iterable[str], named: bool = False) -> None:
    """Keep these entries reachable for this command's manifest without reading them: a value this command may not need
    (a preprocessor output behind a parse that hits) must outlive the install whose command last used it."""
    cache = content(config, named)
    absorb_touched({str(cache.directory): [f"{kind}:{cache._key(kind, key)}" for key in keys]})


def put_many(config: Config, kind: str, rows: Iterable[tuple[str, bytes]], named: bool = False) -> None:
    """Entries in one write transaction: tens of thousands of single writes queue on the cache's one writer."""
    content(config, named).put_many((kind, key, value) for key, value in rows)


def stem(member: str) -> str:
    """One file-name segment per member: slashes become dots, a name too long for a file keeps head and digest."""
    flat = member.replace("/", ".")
    return flat if len(flat) <= 120 else f"{flat[:80]}-{sha256(member.encode())[:16]}"


_LOG_KINDS = frozenset({"receipt", "refusal", "drain"})
_STREAM = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?$")


def _stream_path(config: Config, stream: str) -> Path:
    if not _STREAM.match(stream) or any(part in {".", ".."} for part in stream.split("/")):
        raise Refusal(Finding("store.corrupt", f"bad stream name {stream!r}"))
    return config.project.root / ".unbake" / f"{stream}.jsonl"


def listing(config: Config, folder: str) -> frozenset[str]:
    """The file names in .unbake/<folder>, read once per command: a thousand questions about files are one listing."""
    path = f"{config.project.root}/.unbake/{folder}"
    return effort.memo(("listing", path), lambda: frozenset(os.listdir(path)) if os.path.isdir(path) else frozenset())


def append(config: Config, stream: str, row: Json) -> None:
    """Append one JSON line to .unbake/<stream>.jsonl as a single O_APPEND write under an exclusive lock."""
    path = _stream_path(config, stream)
    line = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _flock(path.with_name(path.name + ".lock"), fcntl.LOCK_EX), path.open("ab") as handle:
        handle.write(line.encode())
    effort.forget("listing")


def rows(config: Config, stream: str) -> list[Json]:
    """Read every line of .unbake/<stream>.jsonl; a line that does not parse refuses with store.corrupt."""
    path = _stream_path(config, stream)
    if not path.exists():
        return []
    result: list[Json] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            result.append(json.loads(raw))
        except ValueError as error:
            raise Refusal(Finding("store.corrupt", f"{path}:{number}: {error}", path=str(path), line=number)) from error
    return result


def log(config: Config, kind: str, body: Json) -> None:
    """Append one operational ledger row (schema 2) of a known kind."""
    if kind not in _LOG_KINDS:
        raise Refusal(Finding("store.corrupt", f"invalid ledger kind {kind!r}"))
    append(
        config,
        "ledger",
        {
            "schema": 2,
            "kind": kind,
            "invocation": effort.invocation(),
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "body": body,
        },
    )


@contextmanager
def exclusive(config: Config, name: str, *, wait: bool, wait_s: float | None = None) -> Iterator[bool]:
    """Take the named project lock: block for it when `wait` (at most `wait_s` seconds when given), else yield False at
    once when another process holds it. A bounded wait first takes the lock's gate, so readers arriving meanwhile queue
    behind the waiting writer instead of starving it."""
    root = config.project.root / ".unbake"
    with ExitStack() as stack:
        start = _CLOCK()
        if (
            wait
            and wait_s is not None
            and stack.enter_context(_flock(root / f"{name}.gate.lock", fcntl.LOCK_EX, wait_s=wait_s)) is None
        ):
            yield False
            return
        left = None if wait_s is None else max(0.0, wait_s - (_CLOCK() - start))
        fd = stack.enter_context(_flock(root / f"{name}.lock", fcntl.LOCK_EX, wait=wait, wait_s=left))
        if fd is not None:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {effort.invocation()}\n".encode())
        yield fd is not None


@contextmanager
def shared(config: Config, name: str) -> Iterator[None]:
    """Take the shared side of a project lock: many readers at once, never while a writer holds it exclusive. A writer
    waiting with a deadline goes first: its gate is passed before the lock is asked for."""
    root = config.project.root / ".unbake"
    gate = root / f"{name}.gate.lock"
    if gate.exists():
        with _flock(gate, fcntl.LOCK_SH):
            pass
    with _flock(root / f"{name}.lock", fcntl.LOCK_SH):
        yield


@contextmanager
def work(config: Config) -> Iterator[Path]:
    """Yield a fresh scratch directory under the host cache and remove it on exit."""
    path = Path(tempfile.mkdtemp(dir=process.scratch(config.project.root)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
