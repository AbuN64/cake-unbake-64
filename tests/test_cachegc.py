"""cachegc: a size-capped trim that removes the least recently used entries by manifest time."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import fixture
from abucache import store as cachestore

from unbake import cachegc, config, store

BIG = 100_000


def _cache(tmp_path: Path, limit: int = 10**9) -> cachestore.ContentCache:
    return cachestore.ContentCache(tmp_path / "c", limit)


def _fill(cache: cachestore.ContentCache, names: list[str]) -> None:
    for name in names:
        cache.put("k", name, bytes([len(name)]) * BIG)


def test_trim_removes_the_oldest_used_and_keeps_manifests_and_young_entries(tmp_path):
    cache = _cache(tmp_path)
    _fill(cache, ["a", "bb", "ccc", "dddd"])
    now = time.time() + 2 * cachestore.GRACE
    cache._write_manifest("/p", "id", "t1", ["k:a", "k:bb"], at=now - 1000)
    cache._write_manifest("/p", "id", "t2", ["k:bb", "k:ccc"], at=now - 10)  # bb is used again, a is not
    # dddd is in no manifest: its store time (now - 2 grace) is older than every manifest
    before = cachegc.trim(cache.directory, 1, now=now, dry_run=True)["before"]
    assert before >= 4 * BIG
    cap = int((before - 2 * BIG) / cachegc.TARGET) + 1  # room for two files less than there are
    result = cachegc.trim(cache.directory, cap, now=now)
    assert (
        result["deleted"] == 3 and result["freed"] >= 2 * BIG
    )  # dddd (no manifest), a, and the manifest naming only a and bb that predates them (oldest manifest)
    kept = [n for n in ("a", "bb", "ccc", "dddd") if cache.get("k", n) is not None]
    assert kept == ["bb", "ccc"]
    assert len(cache._manifest_keys()) == 1  # the older manifest named only what went
    assert result["after"] <= int(cap * cachegc.TARGET)


def test_a_manifest_big_enough_to_be_a_value_file_still_counts(tmp_path):
    cache = _cache(tmp_path)
    _fill(cache, ["a", "bb"])
    now = time.time() + 2 * cachestore.GRACE
    cache._write_manifest("/p", "id", "t1", ["k:bb"] + [f"x:{i:040d}" for i in range(2000)], at=now)
    cache._write_manifest("/q", "id", "t1", ["k:a"], at=now - 5000)
    before = cachegc.trim(cache.directory, 1, now=now, dry_run=True)["before"]
    cachegc.trim(cache.directory, int((before - BIG) / cachegc.TARGET) + 1, now=now)
    assert cache.get("k", "a") is None and cache.get("k", "bb") is not None


def test_trim_under_the_cap_deletes_nothing_and_dry_run_deletes_nothing(tmp_path):
    cache = _cache(tmp_path)
    _fill(cache, ["a", "bb"])
    now = time.time() + 2 * cachestore.GRACE
    assert cachegc.trim(cache.directory, 10**9, now=now)["deleted"] == 0
    dry = cachegc.trim(cache.directory, BIG, now=now, dry_run=True)
    assert dry["deleted"] == 2 and cache.get("k", "a") is not None


def test_young_entries_survive_a_trim(tmp_path):
    cache = _cache(tmp_path)
    _fill(cache, ["a", "bb"])
    assert cachegc.trim(cache.directory, 1, now=time.time())["deleted"] == 0


def test_a_running_collect_is_not_disturbed(tmp_path):
    import fcntl

    cache = _cache(tmp_path)
    with open(cache.directory / "collect.lock", "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        assert "skipped" in cachegc.trim(cache.directory, 1)


def test_main_reads_the_cap_from_the_host_config(tmp_path, monkeypatch, capsys):
    cfg = fixture.config(tmp_path)
    monkeypatch.setattr(config, "load_host", lambda path: replace(cfg.host, cache_gc_bytes=10**12))
    cachestore.ContentCache(store.content_directory(cfg.host), 10**9)
    assert cachegc.main(["--host", str(tmp_path / "h.toml")]) == 0
    assert json.loads(capsys.readouterr().out)["cap"] == 10**12


def test_a_host_without_gc_bytes_trims_to_eight_gigabytes(tmp_path, monkeypatch, capsys):
    cfg = fixture.config(tmp_path)
    unset = type(cfg.host).cache_gc_bytes
    monkeypatch.setattr(config, "load_host", lambda path: replace(cfg.host, cache_gc_bytes=unset))
    cachestore.ContentCache(store.content_directory(cfg.host), 10**9)
    assert cachegc.main(["--host", str(tmp_path / "h.toml")]) == 0
    assert json.loads(capsys.readouterr().out)["cap"] == 8 * 10**9
