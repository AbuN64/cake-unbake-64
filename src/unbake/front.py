"""One build, two make runs: the compile commands a build needs are answered in a few batch processes between them,
so make starts no python front per source (about ten thousand of them after a header edit)."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from unbake import effort, process
from unbake.contracts import Host
from unbake.resources import cc_cache


def cache(host: Host) -> Path | None:
    """The directory of the compiler front's assembly cache, None on a host without a content cache."""
    content = getattr(host, "cache_dir", None)
    return None if content is None else content.with_name(f"{content.name}.cc")


def build[Result](start: Callable[[str | None, bool], Result], root: Path, store: Path | None, workers: int) -> Result:
    """Run the make that `start(record, preprocess)` starts, in `root`, with the compile commands answered first.
    `start` runs it once; with a `record` file name it sets ABU_CC_COLLECT to it and keeps going past failed units
    (make -k), and with `preprocess` it also sets ABU_PP_COLLECT to `<record>.pp`.
    The front (resources/cc_front.sh) first only records each compile command the build needs and fails it, so that
    run does the preprocessing and finds the stale units; one batch of `workers` processes (resources/cc_cache.py)
    then answers every unit the cache can; the second run is the build, taking those answers and compiling the rest.
    A first run that recorded nothing was the build, and a host without a cache runs once. Preprocessing is batched
    the same way before that: a run that finds many stale stamps records their commands (past the first few, which it
    makes itself), one batch makes them all, and the compile commands are recorded by the run after it."""
    if store is None:
        return start(None, False)
    manifest = process.scratch(root) / f"cc-commands.{os.getpid()}"
    stamps = manifest.with_name(manifest.name + ".pp")
    _clear(manifest, root)
    try:
        first = start(str(manifest), True)
        if stamps.exists():
            with effort.stage("make.front.preprocess") as span:
                process.batch(store, root, stamps, workers, "--units")
                span.add(items=len(cc_cache.commands(str(stamps))))
            stamps.unlink()
            manifest.unlink(missing_ok=True)
            first = start(str(manifest), False)
        if not manifest.exists():
            return first
        with effort.stage("make.front") as span:
            process.batch(store, root, manifest, workers)
            answered, total = _answered(manifest, root)
            span.add(items=total)
            effort.say(f"make.front: {answered} of {total} compiles from the cache, {total - answered} compile")
        return start(None, False)
    finally:
        _clear(manifest, root)


def _answered(manifest: Path, root: Path) -> tuple[int, int]:
    """(compiles the batch answered, compiles the build recorded): the rest run the compiler in the second make."""
    parsed = [p for _, args in cc_cache.commands(str(manifest)) if (p := cc_cache.parse(args)) is not None]
    return sum((root / (out + ".cc")).exists() for _, out in parsed), len(parsed)


def _clear(manifest: Path, root: Path) -> None:
    """Remove the record and any answer a unit left unclaimed, so a later build never takes one for another text."""
    if manifest.exists():
        for _, args in cc_cache.commands(str(manifest)):
            if (parsed := cc_cache.parse(args)) is not None:
                (root / (parsed[1] + ".cc")).unlink(missing_ok=True)
    for name in (manifest, manifest.with_name(manifest.name + ".pp"), manifest.with_name(manifest.name + ".pp.n")):
        name.unlink(missing_ok=True)
