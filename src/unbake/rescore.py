"""Fuzzy scores kept current: the NON_MATCHING bodies of the source tree measured once per content, never twice."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib import import_module
from typing import Any, Protocol

from unbake import config as configuration
from unbake import effort, pool, store, symptoms
from unbake.contracts import code_identity, digest, sha256

Fact = tuple[int, int, int, str]  # (vram, rom, size, type) of one symbol_addrs row
Facts = Mapping[str, Mapping[str, Fact]]  # version -> name -> fact
KIND = "score"


class _Tree(Protocol):
    digest: str

    @property
    def modules(self) -> tuple[Any, ...]: ...


def _measure(config: Any, path: str, names: Sequence[str], version: str) -> dict[str, float]:
    """The score of the NON_MATCHING body of each of `names` in the module `path`, linked alone at the function's
    address in `version`; the module compiles once for all (compare.py owns the measuring)."""
    scored: dict[str, float] = import_module("unbake.compare").score_functions(config, path, names, version)
    return scored


def _lookup(config: Any, key: str) -> float | None:
    blob = store.get(config, KIND, key)
    return None if blob is None else float(json.loads(blob))


def _record(config: Any, key: str, value: float) -> None:
    store.put(config, KIND, key, json.dumps(value).encode())


@lru_cache(maxsize=1)
def _scorer() -> str:
    """The scorer and native candidate identity: old assembly-backed scores must never survive a scoring fix."""
    native = import_module("unbake.native")
    return code_identity(symptoms.score, symptoms.literals, native.candidate, native.measure)


def _target_sha(config: Any, version: str, rom: int, size: int) -> str:
    with configuration.rom(config.project, version).open("rb") as handle:
        handle.seek(rom)
        return sha256(handle.read(size))


def _job(item: tuple[Any, str, str, tuple[tuple[str, str], ...]]) -> dict[str, float]:
    """One module in one version: every missing function of it measured together, each recorded under its key."""
    config, path, version, wanted = item
    values = _measure(config, path, [name for name, _ in wanted], version)
    for name, key in wanted:
        _record(config, key, values[name])
    return values


def scores(config: Any, tree: _Tree, facts: Facts) -> dict[str, dict[str, float]]:
    """name -> version -> score of every NON_MATCHING function. A score is keyed by the module text, the name, the
    target bytes and the scorer, so an unchanged body measures nothing and a changed one measures exactly once."""
    with effort.stage("rescore.scores"):
        found: dict[str, dict[str, float]] = {}
        misses: dict[tuple[str, str], list[tuple[str, str]]] = {}
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
                        misses.setdefault((module.path, version), []).append((name, key))
                    else:
                        found.setdefault(name, {})[version] = value
        items = [(config, path, version, tuple(wanted)) for (path, version), wanted in misses.items()]
        for item, values in zip(items, pool.map(config, "rescore.measure", _job, items), strict=True):
            for name, value in values.items():
                found.setdefault(name, {})[item[2]] = float(value)
        return found


def rows(config: Any, tree: _Tree, facts: Facts) -> dict[str, dict[str, float]]:
    """The scores of the tree's NON_MATCHING bodies, measured only where the cache has no entry."""
    return scores(config, tree, facts)
