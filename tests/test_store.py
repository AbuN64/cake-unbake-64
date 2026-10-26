"""store: lock-free cache, append-only streams, exclusive locks and scratch dirs."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import pickle
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path

import pytest

from unbake import effort, pool, store
from unbake.contracts import Config, Refusal


def _hold_child(config: Config, started: Path, release: Path) -> None:
    with store.exclusive(config, "land", wait=False) as held:
        started.write_text(str(held))
        while not release.exists():
            time.sleep(0.02)


def _wait(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)


def test_cached_pool_cached_holds_no_lock(cfg: Config) -> None:
    """A miss may fan out to the pool whose jobs read and write the same cache: nothing may be held meanwhile."""

    def job(item: int) -> bytes:
        return store.cached(cfg, "inner", str(item), lambda: b"inner")

    def produce() -> bytes:
        return b"+".join(pool.map(cfg, "inner", job, [1]))

    assert store.cached(cfg, "outer", "k", produce) == b"inner"
    assert store.cached(cfg, "outer", "k", lambda: pytest.fail("a second call must hit")) == b"inner"


def test_cached_counts_and_put_overwrites(cfg: Config) -> None:
    before = effort.counters().get("kind", (0, 0))
    assert store.cached(cfg, "kind", "a", lambda: b"one") == b"one"
    assert store.cached(cfg, "kind", "a", lambda: b"two") == b"one"
    after = effort.counters()["kind"]
    assert (after[0] - before[0], after[1] - before[1]) == (1, 1)
    store.put(cfg, "kind", "a", b"three")
    assert store.cached(cfg, "kind", "a", lambda: b"four") == b"three"
    assert effort.counters()["kind"] == (after[0] + 1, after[1])


def test_append_rows_roundtrip(cfg: Config) -> None:
    store.append(cfg, "attempts/f", {"b": 1, "a": "é"})
    store.append(cfg, "attempts/f", {"n": [1, 2]})
    path = cfg.project.root / ".unbake" / "attempts" / "f.jsonl"
    assert path.read_text().splitlines()[0] == '{"a":"\\u00e9","b":1}'
    assert store.rows(cfg, "attempts/f") == [{"a": "é", "b": 1}, {"n": [1, 2]}]


def test_rows_missing_is_empty(cfg: Config) -> None:
    assert store.rows(cfg, "nothing") == []


def test_rows_bad_line_refuses_with_line(cfg: Config) -> None:
    store.append(cfg, "s", {"ok": 1})
    path = cfg.project.root / ".unbake" / "s.jsonl"
    with path.open("a") as handle:
        handle.write("{broken\n")
    with pytest.raises(Refusal) as caught:
        store.rows(cfg, "s")
    finding = caught.value.findings[0]
    assert finding.key == "store.corrupt"
    assert finding.line == 2 and finding.path == str(path) and f"{path}:2:" in finding.reason


@pytest.mark.parametrize(
    ("stream", "ok"),
    [
        ("attempts/f", True),
        ("ledger", True),
        ("a.b-c_d/e.f", True),
        ("../x", False),
        ("a/b/c", False),
        ("", False),
        ("a b", False),
        ("/abs", False),
        ("a/", False),
        ("a/..", False),
    ],
)
def test_stream_name_rules(cfg: Config, stream: str, ok: bool) -> None:
    if ok:
        store.append(cfg, stream, {"v": 1})
        assert store.rows(cfg, stream) == [{"v": 1}]
    else:
        with pytest.raises(Refusal) as caught:
            store.append(cfg, stream, {"v": 1})
        assert caught.value.findings[0].key == "store.corrupt"


def test_log_kinds(cfg: Config) -> None:
    store.log(cfg, "receipt", {"x": 1})
    (row,) = store.rows(cfg, "ledger")
    assert row["schema"] == 2 and row["kind"] == "receipt" and row["body"] == {"x": 1}
    assert row["invocation"] == effort.invocation() and row["time"]
    with pytest.raises(Refusal) as caught:
        store.log(cfg, "bogus", {})
    assert caught.value.findings[0].key == "store.corrupt"
    assert len(store.rows(cfg, "ledger")) == 1


def test_exclusive_second_holder_gets_false(cfg: Config, tmp_path: Path) -> None:
    started, release = tmp_path / "started", tmp_path / "release"
    proc = multiprocessing.get_context("fork").Process(target=_hold_child, args=(cfg, started, release))
    proc.start()
    try:
        _wait(started)
        assert started.read_text() == "True"
        with store.exclusive(cfg, "land", wait=False) as held:
            assert held is False
    finally:
        release.write_text("go")
        proc.join(10)
    assert proc.exitcode == 0
    with store.exclusive(cfg, "land", wait=False) as held:
        assert held is True
        text = (cfg.project.root / ".unbake" / "land.lock").read_text()
        assert text.strip().split()[0].isdigit() and text.endswith("\n")


def test_exclusive_waits_for_a_held_lock_and_records_the_wait(cfg: Config, tmp_path: Path, monkeypatch) -> None:
    started, release = tmp_path / "started", tmp_path / "release"
    proc = multiprocessing.get_context("fork").Process(target=_hold_child, args=(cfg, started, release))
    proc.start()
    waits: list[float] = []
    monkeypatch.setattr(effort, "waited", lambda seconds, *rest: waits.append(seconds))
    try:
        _wait(started)
        threading.Timer(0.3, release.write_text, ["go"]).start()
        with store.exclusive(cfg, "land", wait=True) as held:
            assert held is True
    finally:
        proc.join(10)
    assert len(waits) == 1


def test_work_dir_removed(cfg: Config) -> None:
    with store.work(cfg) as path:
        assert path.is_dir() and path.parent == cfg.project.root / ".unbake" / "tmp"
        (path / "f").write_text("x")
        kept = path
    assert not kept.exists()
    with pytest.raises(RuntimeError), store.work(cfg) as path:
        failed = path
        raise RuntimeError
    assert not failed.exists()


def test_flock_charges_only_blocked_time(tmp_path, monkeypatch):
    calls = []

    def flock(fd, flags):
        calls.append(flags)
        if len(calls) == 1 and flags & store.fcntl.LOCK_NB:
            raise BlockingIOError

    monkeypatch.setattr(store.fcntl, "flock", flock)
    with effort.command("check", []):
        with effort.stage("blocked"), store._flock(tmp_path / "a.lock", store.fcntl.LOCK_EX):
            pass
        with effort.stage("free"), store._flock(tmp_path / "b.lock", store.fcntl.LOCK_EX):
            pass
    waits = {r.path[-1]: r.wait_seconds for r in effort.closed() if r.kind == "stage"}
    assert waits["blocked"] > 0 and waits["free"] == 0
    assert calls[1] == store.fcntl.LOCK_EX | store.fcntl.LOCK_NB


def test_stem_flattens_slashes_and_digests_long_names():
    assert store.stem("src/a.c") == "src.a.c"
    long = "x/" + "y" * 200
    assert len(store.stem(long)) <= 100 and store.stem(long) != store.stem(long + "z")


def test_write_keeps_a_name_near_the_filesystem_limit(tmp_path: Path) -> None:
    path = tmp_path / ("x" * 240 + ".s")  # the temporary file must not lengthen the name past 255 bytes
    assert store.write(path, b"a") and path.read_bytes() == b"a" and [p.name for p in tmp_path.iterdir()] == [path.name]


_CFG: list[Config] = []


def _inner(item: int) -> bytes:  # module level: jobs travel to forked workers by pickle
    return store.cached(_CFG[0], "inner", str(item), lambda: b"v")


def _manifests(cfg: Config) -> list[dict]:
    import json

    cache = store.content(cfg)._inner._cache
    return [json.loads(cache.get(k)) for k in cache.iterkeys() if k.startswith("manifest:")]


def test_one_command_writes_one_manifest_with_the_keys_of_its_workers(cfg: Config) -> None:
    """A pool job's reads and writes travel back with its result, so finish writes one manifest holding them all."""

    _CFG[:] = [cfg]
    store.put(cfg, "unit", "p", b"1")  # a named kind keeps its key as given
    pool.map(cfg, "inner", _inner, [1, 2])
    store.finish(cfg)
    store.finish(cfg)  # nothing new was touched: no second manifest
    (manifest,) = _manifests(cfg)
    assert manifest["root"] == str(cfg.project.root) and manifest["identity"] == store.identity()
    inner = store.content(cfg)._key
    assert {"unit:p", f"inner:{inner('inner', '1')}", f"inner:{inner('inner', '2')}"} <= set(manifest["keys"])


def test_installed_is_true_for_this_code_and_false_once_it_changes_or_vanishes(tmp_path: Path) -> None:
    assert store.installed(store.identity())
    digest, *folders = store.identity().split(";")
    assert not store.installed(";".join(["0" * 64, *folders]))
    assert not store.installed(";".join([digest, str(tmp_path / "gone")]))


def test_values_are_packed_and_read_back_equal(cfg: Config) -> None:
    value = b"int main(void) { return 0; }\n" * 400
    store.put(cfg, "kind", "packed", value)
    raw = store.content(cfg)._inner.get("kind", store.content(cfg)._key("kind", "packed"))
    assert raw is not None and raw[:1] == b"\x02" and len(raw) < len(value) // 4
    assert store.get(cfg, "kind", "packed") == value
    store.put_many(cfg, "kind", [("m1", value), ("m2", b"")])
    assert store.get(cfg, "kind", "m1") == value and store.get(cfg, "kind", "m2") == b""


def test_each_entry_format_version_has_its_own_cache_folder(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    first = store.content(cfg).directory
    monkeypatch.setattr(store, "_PACKED", b"\x03")
    assert store.content(cfg).directory != first and first != cfg.host.cache_dir


def test_a_legacy_raw_entry_is_a_miss_and_the_producer_runs_once(cfg: Config) -> None:
    store.content(cfg)._inner.put("kind", "legacy", b'{"text": "raw"}')
    assert store.get(cfg, "kind", "legacy") is None
    calls: list[int] = []

    def produce() -> bytes:
        calls.append(1)
        return b"fresh"

    assert store.cached(cfg, "kind", "legacy", produce) == b"fresh"
    assert store.cached(cfg, "kind", "legacy", produce) == b"fresh"
    assert len(calls) == 1


def _expansion(name: str, lines: int) -> str:
    return "\n".join(f'# {n} "{name}"\nint {name.replace(".", "_")}_{n};' for n in range(lines))


def test_a_view_keeps_shared_expansions_once_and_reads_back_the_same_text(cfg: Config) -> None:
    from unbake import view

    shared = _expansion("types.h", 3000)
    own = ["unit_a", "unit_b"]
    texts = [f"int {name};\n{shared}\nint {name}_end;\n" for name in own]
    for text in texts:
        assert "\n".join(view._chunks(text)) == text
    for number, text in enumerate(texts):
        view._keep(cfg, f"{number:016x}", False, text, [("src/a.c", "sha")])
    chunk = lambda text: set(view._chunks(text))  # noqa: E731
    assert len(chunk(texts[0]) | chunk(texts[1])) < len(chunk(texts[0])) + len(chunk(texts[1])) - 20  # shared chunks
    for number, text in enumerate(texts):
        assert view._kept(cfg, f"{number:016x}") == (text, [["src/a.c", "sha"]])


def test_an_overlay_directory_name_is_not_stored_and_a_lost_chunk_is_a_miss(cfg: Config) -> None:
    from unbake import view

    key = "0123456789abcdef" + "0" * 48
    text = "\n".join(f'# {n} "build/views/0123456789abcdef/include/h.h"\nint x{n};' for n in range(400))
    view._keep(cfg, key, True, text, [])
    names = json.loads(store.get(cfg, "view", key))["chunks"]
    assert all(b"0123456789abcdef" not in store.get(cfg, "view.chunk", name) for name in names)
    assert view._kept(cfg, key) == (text, [])
    store.content(cfg)._inner._cache.delete(f"view.chunk:{store.content(cfg)._key('view.chunk', names[0])}")
    assert view._kept(cfg, key) is None


def test_finish_hands_the_host_commit_window_to_the_collect(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    inner = store.content(cfg)._inner
    monkeypatch.setattr(inner, "_start_collect", lambda installed, keep: seen.append(keep))
    store.finish(cfg)
    assert seen == [cfg.host.cache_keep_commits]


def test_finish_command_is_the_one_boundary_and_never_changes_the_result(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[Config] = []
    monkeypatch.setattr(store, "finish", calls.append)
    store.finish_command(cfg)
    assert calls == [cfg]

    def broken(config: Config) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(store, "finish", broken)
    store.finish_command(cfg)  # does not raise
    assert "disk gone" in capsys.readouterr().err


def test_a_full_cache_refuses_a_write_with_one_finding(cfg: Config) -> None:
    inner = store.content(cfg)._inner
    inner._full = True
    with pytest.raises(Refusal) as caught:
        store.put(cfg, "kind", "k", b"1")
    assert [f.key for f in caught.value.findings] == ["cache.full"]
    with pytest.raises(Refusal):
        store.content(cfg).put_many([("kind", "k", b"1")])
    inner._full = False
    store.put(cfg, "kind", "k", b"1")
    assert store.get(cfg, "kind", "k") == b"1"


def test_a_foreign_tag_a_torn_entry_and_an_unreadable_pickle_are_misses_never_crashes(cfg: Config) -> None:
    inner, key = store.content(cfg)._inner, partial(store.content(cfg)._key, "kind")
    store.put(cfg, "kind", "torn", b"payload")
    good = inner.get("kind", key("torn"))
    assert good is not None
    inner.put("kind", key("torn"), good[:-1])  # the compressed bytes cut short
    inner.put("kind", key("foreign"), b"\x09" + good[1:])
    inner.put("kind", key("crc"), good[:1] + b"\0\0\0\0" + good[5:])
    assert [store.get(cfg, "kind", k) for k in ("torn", "foreign", "crc")] == [None, None, None]
    store.put(cfg, "kind", "bad", b"\x01 not a pickle")
    assert store.cached_object(cfg, "kind", "bad", lambda: {"fresh": 1}) == {"fresh": 1}
    assert store.loads(b"\x01 not a pickle") == (False, None) and store.loads(None) == (False, None)
    assert store.loads(pickle.dumps([1])) == (True, [1])


def test_a_contended_lock_is_waited_for_inside_a_stage_of_its_own(tmp_path, monkeypatch) -> None:
    from contextlib import contextmanager

    stages: list[str] = []

    @contextmanager
    def stage(name):
        stages.append(name)
        yield None

    monkeypatch.setattr(store.effort, "stage", stage)
    path = tmp_path / "held.lock"
    with store._flock(path, fcntl.LOCK_EX), store._flock(path, fcntl.LOCK_EX, wait_s=0.0) as taken:
        assert taken is None
    assert stages == ["store.wait_lock"]


def test_two_tool_builds_writing_one_logical_key_keep_two_entries_and_never_misread(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "code", lambda: "build-one")
    store.put(cfg, "header-proof", "same", b"one")
    store.put(cfg, "unit", "named", b"object")
    store.put(cfg, "pool.job", "keyed", b"result", named=True)
    monkeypatch.setattr(store, "code", lambda: "build-two")
    assert store.get(cfg, "header-proof", "same") is None  # a side venv's entry is a miss, never this build's value
    store.put(cfg, "header-proof", "same", b"two")
    assert store.get(cfg, "header-proof", "same") == b"two"
    assert store.get(cfg, "unit", "named") == b"object"  # a compiler output named by every input is shared
    assert store.get(cfg, "pool.job", "keyed", named=True) == b"result"  # a pool key names its code itself
    monkeypatch.setattr(store, "code", lambda: "build-one")
    assert store.get(cfg, "header-proof", "same") == b"one"


def test_a_forked_worker_drops_an_inherited_flock_without_unlocking_it(tmp_path) -> None:
    # setup forked its warm workers inside journal.reading: each kept the shared journal lock for life, and the
    # boundary apply then waited forever for the exclusive side. Dropping the inherited descriptor frees the lock
    # once the parent lets go; the holder here plays the worker's copy of the parent's descriptor.
    path = tmp_path / "journal.lock"
    with store._flock(path, fcntl.LOCK_SH) as fd:
        assert fd in store._HELD_FDS
        store.drop_inherited()
        assert not store._HELD_FDS
        with store._flock(path, fcntl.LOCK_EX, wait_s=0.0) as taken:
            assert taken is not None  # no copy of the descriptor holds the lock any more
    assert not store._HELD_FDS


def test_flock_tracks_the_held_descriptor_only_while_held(tmp_path) -> None:
    with store._flock(tmp_path / "a.lock", fcntl.LOCK_EX) as fd:
        assert list(store._HELD_FDS) == [fd]
    assert not store._HELD_FDS


@pytest.mark.parametrize(("kind", "shared"), [("view", True), ("view.chunk", True), ("unit", True), ("kind", False)])
def test_content_named_kinds_survive_an_install_and_code_keyed_kinds_do_not(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, kind: str, shared: bool
) -> None:
    monkeypatch.setattr(store, "code", lambda: "install-one")
    store.put(cfg, kind, "key", b"value")
    monkeypatch.setattr(store, "code", lambda: "install-two")
    assert (store.get(cfg, kind, "key") == b"value") is shared


def test_write_leaves_the_other_hard_link_untouched(tmp_path):
    shared, mine = tmp_path / "cache" / "a.s", tmp_path / "tree" / "a.s"
    shared.parent.mkdir()
    mine.parent.mkdir()
    shared.write_bytes(b"old\n")
    shared.chmod(0o444)
    os.link(shared, mine)
    assert store.write(mine, b"new\n")
    assert mine.read_bytes() == b"new\n"
    assert shared.read_bytes() == b"old\n"
    assert shared.stat().st_nlink == 1


def test_unshare_breaks_links_so_an_in_place_writer_cannot_reach_the_cache(tmp_path):
    shared, mine, private = (
        tmp_path / "cache" / "a.s",
        tmp_path / "tree" / "asm" / "a.s",
        tmp_path / "tree" / "asm" / "b.s",
    )
    shared.parent.mkdir()
    mine.parent.mkdir(parents=True)
    shared.write_bytes(b"old\n")
    shared.chmod(0o444)
    os.link(shared, mine)
    private.write_bytes(b"own\n")
    inode = private.stat().st_ino
    assert store.unshare(tmp_path / "tree" / "asm") == 1
    mine.write_bytes(b"splat\n")  # what splat does: open, truncate, write
    assert shared.read_bytes() == b"old\n"
    assert shared.stat().st_nlink == 1
    assert private.stat().st_ino == inode
    assert not list(mine.parent.glob(".*.part"))


def test_unshare_copies_a_readonly_file_even_when_unlinked(tmp_path):
    only = tmp_path / "asm" / "a.s"
    only.parent.mkdir()
    only.write_bytes(b"x")
    only.chmod(0o444)
    assert store.unshare(tmp_path / "asm") == 1
    assert os.access(only, os.W_OK)
    assert only.read_bytes() == b"x"


def test_a_blocked_lock_says_who_holds_it(tmp_path, monkeypatch):
    path = tmp_path / "held.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl,sys,time;f=open(sys.argv[1],'a+b');fcntl.flock(f,2);print(1,flush=True);time.sleep(2)",
            str(path),
        ],
        stdout=subprocess.PIPE,
    )
    assert holder.stdout and holder.stdout.readline()
    said = []
    monkeypatch.setattr(effort, "say", said.append)
    with store._flock(path, store.fcntl.LOCK_EX):
        pass
    holder.wait()
    assert said and f"pid {holder.pid}" in said[0] and "held.lock" in said[0]


def test_a_journal_undoes_what_was_written_in_it(tmp_path: Path) -> None:
    kept, changed, made = tmp_path / "kept", tmp_path / "changed", tmp_path / "made"
    kept.write_bytes(b"k")
    changed.write_bytes(b"old")
    with store.journal() as seen:
        store.write(kept, b"k")  # same bytes: nothing written, nothing recorded
        store.write(changed, b"one")
        store.write(changed, b"two")  # the first write holds the original
        store.write(made, b"new")
    assert seen == {changed: b"old", made: None}
    store.undo(seen)
    assert (kept.read_bytes(), changed.read_bytes(), made.exists()) == (b"k", b"old", False)
    store.write(changed, b"after")  # outside a journal: recorded nowhere
    assert seen == {changed: b"old", made: None}
