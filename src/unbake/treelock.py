"""The work tree's reader/writer lock: one file, `.unbake/tree.lock`, shared by every reader of the tree (a try, a
probe, a score, a work) and held exclusively by whoever rewrites tracked files or the extraction under it (a ship's
sync, check and commit). A reader that finds a writer waits and says so; it never fails for the writer's changes.
A process that holds the lock marks its children (`UNBAKE_TREE_LOCK`), which then run inside the hold without taking it again."""

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


@contextmanager
def hold(
    root: Path, *, exclusive: bool = False, who: str = "", say: Callable[[str], None] = effort.say
) -> Iterator[float]:
    """Hold the tree lock of `root` (shared by default); yields the seconds it waited."""
    if os.environ.get(ENV):  # a parent of ours holds it
        yield 0.0
        return
    path = Path(root) / ".unbake" / "tree.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    began = last = time.monotonic()
    with path.open("a") as handle:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        while True:
            try:
                fcntl.flock(handle, mode | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if now - last >= BEAT or last == began:
                    last = now
                    kind = "a reader" if exclusive else "a ship or extraction"
                    say(f"waiting for the tree lock ({kind} holds it) {now - began:.0f}s {who}".rstrip())
                time.sleep(0.5)
        os.environ[ENV] = "1"
        try:
            yield time.monotonic() - began
        finally:
            os.environ.pop(ENV, None)
            fcntl.flock(handle, fcntl.LOCK_UN)
