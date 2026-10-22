"""The work tree's reader/writer lock: `.unbake/tree.lock`, shared by every reader of the tree (a probe, a score, a
compare) and held exclusively by whoever rewrites tracked files or the extraction under it (a ship's sync, check and
commit). A writer announces itself first (`tree.writer`, holding its name and time): new readers wait behind the
announcement, readers already inside finish, so a stream of readers never starves it. A reader that waits says for whom.
A process that holds the lock marks its children (`UNBAKE_TREE_LOCK`): they run inside the hold, never taking it. Each
exclusive hold ends by bumping `tree.epoch`, so a command that spans a writer's window can tell it happened."""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from unbake import effort

ENV = "UNBAKE_TREE_LOCK"
BEAT = 15.0  # seconds between the lines a waiting process prints
inflight: list[str] = []  # what this process read while a writer held the tree, one line per hold taken without waiting


def epoch(root: Path) -> int:
    """How many exclusive holds of the tree have ended."""
    try:
        return int((Path(root) / ".unbake" / "tree.epoch").read_text() or 0)
    except (OSError, ValueError):
        return 0


def _take(handle: int, mode: int, say: Callable[[str], None], note: Callable[[], str]) -> float:
    """Take the flock in `mode`, printing `note()` at once and every BEAT while it waits; the seconds waited."""
    began = last = time.monotonic()
    first = True
    while True:
        try:
            fcntl.flock(handle, mode | fcntl.LOCK_NB)
            return time.monotonic() - began
        except BlockingIOError:
            now = time.monotonic()
            if first or now - last >= BEAT:
                first, last = False, now
                say(f"{note()} ({now - began:.0f}s)")
            time.sleep(0.2)


def _free(intent: int, tree: int) -> bool:
    """Whether no writer holds or announces the tree now (a probe: it takes and gives both shared locks)."""
    for handle in (intent, tree):
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
    return True


@contextmanager
def hold(
    root: Path, *, exclusive: bool = False, who: str = "", say: Callable[[str], None] = effort.say, wait: bool = True
) -> Iterator[float]:
    """Hold the tree lock of `root` (shared by default); yields the seconds waited. A shared hold with `wait` false
    never waits for a writer: it says who writes, reads the tree as the writer has it (every file is whole, a file is
    replaced in one rename) and records that in `inflight`; the ROM check of a build stays the proof of a ship."""
    if os.environ.get(ENV):  # a parent of ours holds it
        yield 0.0
        return
    folder = Path(root) / ".unbake"
    folder.mkdir(parents=True, exist_ok=True)
    tree, intent = (os.open(folder / name, os.O_RDWR | os.O_CREAT, 0o644) for name in ("tree.lock", "tree.writer"))
    try:
        if exclusive:  # announce: readers that come now queue behind us
            waited = _take(intent, fcntl.LOCK_EX, say, lambda: "waiting for another writer of the tree")
            os.ftruncate(intent, 0)
            os.write(intent, f"{who or 'writer'} {time.time():.0f} pid {os.getpid()}".encode())
            waited += _take(tree, fcntl.LOCK_EX, say, lambda: f"waiting for readers of the tree to finish ({who})")
        else:

            def note() -> str:
                text = os.pread(intent, 200, 0).decode(errors="replace").strip().split()
                return (
                    f"waiting for {text[0]} since {time.strftime('%T', time.localtime(int(text[1])))}"
                    + (f" (pid {text[3]})" if len(text) > 3 else "")
                    if len(text) > 1
                    else ("waiting for the tree lock")
                )

            if not wait and not _free(intent, tree):
                line = f"{note().removeprefix('waiting for ')}: read in flight, not waiting"
                inflight.append(line)
                say(f"unbake: tree writer {line}")
                yield 0.0
                return
            holder = note()
            waited = _take(intent, fcntl.LOCK_SH, say, note)  # a writer announced: wait it out
            fcntl.flock(intent, fcntl.LOCK_UN)
            waited += _take(tree, fcntl.LOCK_SH, say, lambda: "waiting for the tree lock")
            if waited >= 1.0:  # the command's lock wait, with whom it waited for
                effort.waited(waited, "lock", holder.removeprefix("waiting for "))
        os.environ[ENV] = "1"
        try:
            yield waited
        finally:
            os.environ.pop(ENV, None)
            if exclusive:
                (folder / "tree.epoch").write_text(str(epoch(root) + 1))
            fcntl.flock(tree, fcntl.LOCK_UN)
            fcntl.flock(intent, fcntl.LOCK_UN)
    finally:
        os.close(tree)
        os.close(intent)
