"""Fuzzy scores kept current: the NON_MATCHING bodies of the source tree measured once per content, never twice."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from unbake import effort, pool, store, symptoms
from unbake.contracts import code_identity, digest, sha256

Fact = tuple[int, int, int, str]  # (vram, rom, size, type) of one symbol_addrs row
Facts = Mapping[str, Mapping[str, Fact]]  # version -> name -> fact
KIND = "score"


class _Tree(Protocol):
    digest: str

    @property
    def modules(self) -> tuple[Any, ...]: ...


def _measure(config: Any, path: str, name: str, version: str) -> float:
    """The score of the NON_MATCHING body of `name` in the module `path`, linked alone at the function's address in
    `version` (compare.py owns the measuring)."""
    scored: float = import_module("unbake.compare").score_function(config, path, name, version)
    return scored


def _lookup(config: Any, key: str) -> float | None:
    blob = store.get(config, KIND, key)
    return None if blob is None else float(json.loads(blob))


def _record(config: Any, key: str, value: float) -> None:
    store.put(config, KIND, key, json.dumps(value).encode())


@lru_cache(maxsize=1)
def _scorer() -> str:
    """The scorer's code identity: the key moves when `symptoms.score` or `symptoms.literals` change."""
    return code_identity(symptoms.score, symptoms.literals)


def _target_sha(config: Any, version: str, rom: int, size: int) -> str:
    path: Path = config.project.root / "roms" / f"baserom.{version}.z64"
    with path.open("rb") as handle:
        handle.seek(rom)
        return sha256(handle.read(size))


def _job(item: tuple[Any, str, str, str, str]) -> float:
    config, path, name, version, key = item
    value = _measure(config, path, name, version)
    _record(config, key, value)
    return value


def scores(config: Any, tree: _Tree, facts: Facts) -> dict[str, dict[str, float]]:
    """name -> version -> score of every NON_MATCHING function. A score is keyed by the module text, the name, the
    target bytes and the scorer, so an unchanged body measures nothing and a changed one measures exactly once."""
    with effort.stage("rescore.scores"):
        found: dict[str, dict[str, float]] = {}
        misses: list[tuple[Any, str, str, str, str]] = []
        texts: dict[str, str] = {}
        for module in tree.modules:
            for name in module.fuzzy:
                for version, owned in module.owns.items():
                    fact = facts.get(version, {}).get(name)
                    if name not in owned or fact is None:
                        continue
                    if module.path not in texts:
                        texts[module.path] = sha256((config.project.root / module.path).read_bytes())
                    key = digest(
                        (texts[module.path], name, version, _target_sha(config, version, fact[1], fact[2]), _scorer())
                    )
                    value = _lookup(config, key)
                    if value is None:
                        misses.append((config, module.path, name, version, key))
                    else:
                        found.setdefault(name, {})[version] = value
        for item, value in zip(misses, pool.map(config, "rescore.measure", _job, misses), strict=True):
            found.setdefault(item[2], {})[item[3]] = float(value)
        return found


def rows(config: Any, tree: _Tree, facts: Facts) -> dict[str, dict[str, float]]:
    """The scores of the tree's NON_MATCHING bodies, measured only where the cache has no entry."""
    return scores(config, tree, facts)
