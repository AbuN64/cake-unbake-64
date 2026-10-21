"""One build, two make runs: the compile commands a build needs are answered in a few batch processes between them,
so make starts no python front per source (about ten thousand of them after a header edit)."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from unbake import effort, process
from unbake.contracts import Host


def cache(host: Host) -> Path | None:
    """The directory of the compiler front's assembly cache, None on a host without a content cache."""
    content = getattr(host, "cache_dir", None)
    return None if content is None else content.with_name(f"{content.name}.cc")


def build[Result](start: Callable[[str | None], Result], root: Path, store: Path | None, workers: int) -> Result:
    """Run the make that `start(record)` starts, in `root`, with the compile commands answered first. `start` runs it
    once; with a `record` file name it sets ABU_CC_COLLECT to it and keeps going past failed units (make -k).
    The front (resources/cc_front.sh) first only records each compile command the build needs and fails it, so that
    run does the preprocessing and finds the stale units; one batch of `workers` processes (resources/cc_cache.py)
    then answers every unit the cache can; the second run is the build, taking those answers and compiling the rest.
    A first run that recorded nothing was the build, and a host without a cache runs once."""
    if store is None:
        return start(None)
    manifest = process.scratch(root) / f"cc-commands.{os.getpid()}"
    manifest.unlink(missing_ok=True)
    try:
        first = start(str(manifest))
        if not manifest.exists():
            return first
        with effort.stage("make.front"):
            process.batch(store, root, manifest, workers)
        return start(None)
    finally:
        _clear(manifest, root)


def _clear(manifest: Path, root: Path) -> None:
    """Remove the record and any answer a unit left unclaimed, so a later build never takes one for another text."""
    from unbake.resources import cc_cache

    if manifest.exists():
        for _, args in cc_cache.commands(str(manifest)):
            if (parsed := cc_cache.parse(args)) is not None:
                (root / (parsed[1] + ".cc")).unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
