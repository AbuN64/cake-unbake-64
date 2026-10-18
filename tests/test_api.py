"""api: the one surface Recipe Maker imports."""

from __future__ import annotations

from pathlib import Path

import fixture

from unbake import api, pool, store, tree
from unbake.contracts import Refusal

EXPECTED = [
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


def test_all_is_exactly_the_surface() -> None:
    assert sorted(api.__all__) == EXPECTED
    assert all(hasattr(api, name) for name in EXPECTED)
    assert api.Tree is tree.Tree and api.Module is tree.Module and api.Refusal is Refusal


def test_symbols_and_asm_path(tmp_path: Path) -> None:
    config = fixture.config(tmp_path)
    assert api.symbols(config, "a")["c_func"] == (0x80000400, 0x1000, 0x10, "func")
    assert api.asm_path(config, "b", "fuzzy") == "asm/b/nonmatchings/ovl/fuzzy.s"


def test_cached_memo_and_pool_map_are_the_shared_ones(tmp_path: Path) -> None:
    config = fixture.config(tmp_path)
    runs = []
    produce = lambda: runs.append(1) or b"v"  # noqa: E731
    assert api.cached(config, "tmp", "k", produce) == b"v" == store.cached(config, "tmp", "k", produce)
    assert runs == [1]
    count = []
    assert api.memo("key", lambda: count.append(1) or 5) == 5 == api.memo("key", lambda: 6) and count == [1]
    assert api.pool_map(config, "api.test", str, [1, 2]) == pool.map(config, "api.test", str, [1, 2]) == ["1", "2"]


def test_score_reads_report_lazily(tmp_path: Path, monkeypatch) -> None:
    from unbake import report

    monkeypatch.setattr(report, "scores", lambda config, held: {"fuzzy": {"a": 0.5}}, raising=False)
    assert api.score(fixture.config(tmp_path), "fuzzy") == {"a": 0.5}


def test_stage_timing_is_the_shared_one(tmp_path: Path) -> None:
    """Recipe prints its --stages from these: a stage opened through the api closes into the same record list."""
    config = fixture.config(tmp_path)
    api.bind(config)
    with api.command("recipe test", ["x"]), api.stage("recipe.work"):
        pass
    records = api.closed()
    assert [r.span for r in records if r.kind == "stage"] and any(r.kind == "command" for r in records)
    assert any("recipe.work" in line for line in api.summary(records))
