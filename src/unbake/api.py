"""The read-only surface Recipe Maker imports: the source tree, the symbol facts, cached scores and the caches and
parallel map the tree's readers share. Nothing else of unbake is public."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from unbake import config as _config
from unbake import effort, human, pool, store
from unbake import symbols as _symbols
from unbake import versions as _versions
from unbake.contracts import Config, Finding, Json, Refusal, Snapshot, Span, StageRecord
from unbake.tree import Module, Tree, capture

__all__ = [
    "Config",
    "Finding",
    "Json",
    "Module",
    "Refusal",
    "Tree",
    "asm_path",
    "bind",
    "cached",
    "capture",
    "closed",
    "command",
    "load",
    "memo",
    "pool_map",
    "score",
    "snapshot",
    "stage",
    "summary",
    "symbols",
]


def load(root: Path, host: Path) -> Config:
    return _config.load(root, host)


def symbols(config: Config, v: str) -> dict[str, tuple[int, int | None, int | None, str]]:
    """Name -> (vram, rom, size, type) of one version's versions/<v>/symbol_addrs.txt."""
    return _symbols.load(config, v)


def asm_path(config: Config, v: str, name: str) -> str:
    """The project-relative path of the extracted assembly of a symbol: nonmatchings/ for code, data/ for data."""
    return _symbols.asm_path(config, v, name)


def score(config: Config, name: str) -> Mapping[str, float]:
    """The cached fuzzy score of a function per version (the `score` store kind); report is loaded on first use."""
    from unbake import report

    return report.scores(config, capture(config)).get(name, {})


def snapshot(config: Config) -> Snapshot:
    """A contracts.Snapshot of the working tree, holding the captured Tree."""
    held = capture(config)
    root = config.project.root
    found = _versions.read(config, lambda path: (root / path).read_bytes())
    fields: dict[str, Any] = {"files": held, "versions": found, "overlays": {}, "digest": held.digest}
    return Snapshot(config=config, commit=_head(config), **fields)


def _head(config: Config) -> str:
    from unbake import process

    return process.git(config, "rev-parse", "HEAD").stdout.decode().strip()


def cached(config: Config, kind: str, key: str, produce: Callable[[], bytes]) -> bytes:
    return store.cached(config, kind, key, produce)


def memo[T](key: object, produce: Callable[[], T]) -> T:
    return effort.memo(key, produce)


def pool_map(
    config: Config,
    name: str,
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    key: Callable[[Any], str | None] | None = None,
) -> list[Any]:
    return pool.map(config, name, fn, items, key)


# ---------------------------------------------------------------- stage timing
def command(name: str, argv: Sequence[str]) -> AbstractContextManager[None]:
    """The span of one command: every stage opened inside it is recorded under it."""
    return effort.command(name, argv)


def bind(config: Config) -> None:
    """Record the stages of this process to the project's stage stream."""
    effort.bind(config)


def stage(name: str) -> AbstractContextManager[Span]:
    """One timed stage, a child of the one open around it."""
    return effort.stage(name)


def closed() -> tuple[StageRecord, ...]:
    """The stage records closed so far in this process."""
    return effort.closed()


def summary(records: Sequence[StageRecord]) -> list[str]:
    """The stage tree and the cache lines of closed records, as the human lines a command prints with --stages."""
    return human.summary(records)
