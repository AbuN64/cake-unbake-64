"""Journal filesystem tests; Git is mocked so this lane never starts Git, and any Git call fails the test."""

from __future__ import annotations

import base64
import fcntl
import json
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import fixture
import pytest

from unbake import journal, process
from unbake.contracts import Finding, Plan, Refusal, Snapshot, digest, sha256


def _sha(content):
    return sha256(content) if content is not None else None


@pytest.fixture
def repo(tmp_path, monkeypatch):
    config = fixture.bare_config(
        tmp_path,
        versions=(),
        name="test",
        title="Test",
        toolchain="test",
        cores=1,
        workers=1,
        author=("Writer", "writer@example.invalid"),
    )
    git_mock = Mock(side_effect=AssertionError("apply must not run Git"))
    monkeypatch.setattr(process, "git", git_mock)
    monkeypatch.setattr(process, "run", git_mock)
    stage = Mock(side_effect=lambda name: nullcontext())
    monkeypatch.setattr(journal.effort, "stage", stage)
    return config, git_mock, stage


def _snapshot(config, files=None):
    """A snapshot that read these files (None = it read the file absent)."""
    return Snapshot(config, "h" * 40, None, {}, {}, "snapshot", {name: _sha(c) for name, c in (files or {}).items()})


def _plan(writes, **changes):
    return replace(Plan("repair", "snapshot", writes, (), (), (), "Install exact bytes", digest(writes)), **changes)


def _seed(config, files):
    for name, content in files.items():
        path = config.project.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _manifest(config, plan, before):
    path = config.project.root / ".unbake/journal.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "plan": plan,
                "before": {
                    name: base64.b64encode(content).decode() if content is not None else None
                    for name, content in before.items()
                },
            }
        )
    )
    return path


def _refused(error, key, missing=None):
    assert len(error.value.findings) == 1
    finding = error.value.findings[0]
    assert finding.key == key
    if missing is not None:
        assert finding.missing == missing


def test_apply_writes_exact_bytes_calls_no_git_and_returns_the_paths(repo):
    config, git, stage = repo
    old = {"a.bin": b"old\x00\xff", "deleted.bin": b"delete", "unrelated.bin": b"keep"}
    _seed(config, old)
    writes = {"new/sub.bin": b"\x00\xff\r\n", "deleted.bin": None, "a.bin": b"new\x80"}
    snapshot = _snapshot(config, {**old, "new/sub.bin": None})
    assert journal.apply(config, _plan(writes), snapshot) == ("a.bin", "deleted.bin", "new/sub.bin")
    assert (config.project.root / "a.bin").read_bytes() == b"new\x80"
    assert (config.project.root / "new/sub.bin").read_bytes() == b"\x00\xff\r\n"
    assert not (config.project.root / "deleted.bin").exists()
    assert (config.project.root / "unrelated.bin").read_bytes() == b"keep"
    assert not (config.project.root / ".unbake/journal.json").exists()
    git.assert_not_called()
    stage.assert_called_once_with("journal.apply")


def test_two_applies_in_a_row_without_a_commit_both_succeed(repo):
    config, _, _ = repo
    _seed(config, {"a": b"0"})
    snapshot = _snapshot(config, {"a": b"0", "b": None})
    journal.apply(config, _plan({"a": b"1", "b": b"made"}), snapshot)
    assert journal.apply(config, _plan({"a": b"2"}), snapshot) == ("a",)
    assert (config.project.root / "a").read_bytes() == b"2"
    assert (config.project.root / "b").read_bytes() == b"made"


@pytest.mark.parametrize("current", [b"someone else", None])
def test_a_path_changed_after_the_snapshot_is_refused(repo, current):
    config, _, _ = repo
    snapshot = _snapshot(config, {"planned": b"read", "fresh": None, "other": b"x"})
    _seed(config, {"other": b"x"})
    if current is not None:
        _seed(config, {"planned": current})
    _seed(config, {"fresh": b"appeared"} if current is None else {})
    with pytest.raises(Refusal) as error:
        journal.apply(config, _plan({"planned": b"new", "fresh": b"mine", "other": b"y"}), snapshot)
    _refused(error, "journal.changed", ("fresh", "planned") if current is None else ("planned",))
    assert (config.project.root / "other").read_bytes() == b"x"
    assert not (config.project.root / ".unbake/journal.json").exists()


def test_a_path_the_snapshot_never_read_is_written(repo):
    config, _, _ = repo
    _seed(config, {"a": b"old"})
    journal.apply(config, _plan({"a": b"new"}), _snapshot(config))
    assert (config.project.root / "a").read_bytes() == b"new"


def test_interrupt_then_recover_restores_a_modified_and_a_new_file(repo, monkeypatch):
    config, _, stage = repo
    before = {"a": b"original\x00\xff", "deleted": b"restore"}
    _seed(config, before)
    real = os.replace

    def interrupt(source, target):
        if Path(target) == config.project.root / "new/nested":
            raise InterruptedError("killed mid write")
        real(source, target)

    monkeypatch.setattr(journal.os, "replace", interrupt)
    with pytest.raises(InterruptedError):
        journal.apply(config, _plan({"a": b"changed", "deleted": None, "new/nested": b"new"}), _snapshot(config))
    monkeypatch.setattr(journal.os, "replace", real)
    assert (config.project.root / "a").read_bytes() == b"changed"
    assert not (config.project.root / "deleted").exists()
    assert (config.project.root / ".unbake/journal.json").exists()
    (config.project.root / "new").mkdir(exist_ok=True)
    (config.project.root / "new/nested").write_bytes(b"new")  # the file the crash had made
    assert journal.recover(config) is True
    for path, content in before.items():
        assert (config.project.root / path).read_bytes() == content
    assert not (config.project.root / "new/nested").exists()
    assert not (config.project.root / ".unbake/journal.json").exists()
    assert [call.args[0] for call in stage.call_args_list] == ["journal.apply", "journal.recover"]


def test_the_journal_holds_old_bytes_never_a_reference_to_git(repo, monkeypatch):
    config, _, _ = repo
    _seed(config, {"big": b"old bytes"})
    real = os.replace

    def interrupt(source, target):
        if Path(target) == config.project.root / "big":
            raise InterruptedError()
        real(source, target)

    monkeypatch.setattr(journal.os, "replace", interrupt)
    with pytest.raises(InterruptedError):
        journal.apply(config, _plan({"big": b"new", "loose": b"new"}), _snapshot(config))
    manifest = json.loads((config.project.root / ".unbake/journal.json").read_bytes())
    assert manifest["before"] == {"big": base64.b64encode(b"old bytes").decode(), "loose": None}


def test_no_partial_install(repo, monkeypatch):
    config, _, _ = repo
    before = {"a": b"old a", "b": b"old b"}
    _seed(config, before)
    original_replace = os.replace
    installed = []

    def fail_second(source, target):
        target = Path(target)
        if target.parent == config.project.root:
            installed.append(target.name)
            if len(installed) == 2:
                raise OSError("second replacement failed")
        return original_replace(source, target)

    monkeypatch.setattr(journal.os, "replace", fail_second)
    with pytest.raises(OSError, match="second replacement failed"):
        journal.apply(config, _plan({"b": b"new b", "a": b"new a"}), _snapshot(config))
    assert installed == ["a", "b"]
    assert (config.project.root / "a").read_bytes() == b"new a"
    assert (config.project.root / "b").read_bytes() == b"old b"
    assert journal.recover(config) is True
    assert {name: (config.project.root / name).read_bytes() for name in before} == before
    assert not (config.project.root / ".unbake/journal.json").exists()
    assert not list(config.project.root.rglob(".unbake-*"))


def test_a_write_that_reads_back_differently_is_refused_and_keeps_the_journal(repo, monkeypatch):
    config, _, _ = repo
    _seed(config, {"a": b"original"})
    real = journal._stage
    monkeypatch.setattr(
        journal, "_stage", lambda path, content: real(path, b"corrupt" if path.name == "a" else content)
    )
    with pytest.raises(Refusal) as error:
        journal.apply(config, _plan({"a": b"wanted"}), _snapshot(config))
    _refused(error, "journal.readback", ("a",))
    assert (config.project.root / ".unbake/journal.json").exists()
    assert journal.recover(config) is True
    assert (config.project.root / "a").read_bytes() == b"original"


def test_recover_without_manifest(repo):
    config, git, stage = repo
    assert journal.recover(config) is False
    git.assert_not_called()
    stage.assert_called_once_with("journal.recover")


def test_apply_rolls_back_a_crashed_earlier_apply_first(repo):
    config, _, _ = repo
    _seed(config, {"a": b"half written"})
    _manifest(config, "crashed", {"a": b"original"})
    journal.apply(config, _plan({"b": b"new"}), _snapshot(config))
    assert (config.project.root / "a").read_bytes() == b"original"


def test_blocking_plan_is_programming_error(repo):
    config, git, _ = repo
    plan = _plan({"a": b"new"}, blocking=(Finding("journal.changed", "blocked"),))
    with pytest.raises(ValueError, match="blocking plan"):
        journal.apply(config, plan, _snapshot(config))
    git.assert_not_called()
    assert not (config.project.root / "a").exists()


def test_apply_holds_blocking_lock_while_writing(repo, monkeypatch):
    config, _, _ = repo
    real_flock = fcntl.flock
    lock_calls = []

    def flock(fd, operation):
        lock_calls.append(operation)
        return real_flock(fd, operation)

    monkeypatch.setattr(journal.store.fcntl, "flock", flock)
    real = os.replace

    def check_lock(source, target):
        if Path(target) == config.project.root / "a":
            with (config.project.root / ".unbake/journal.lock").open("a+b") as other, pytest.raises(BlockingIOError):
                real_flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        real(source, target)

    monkeypatch.setattr(journal.os, "replace", check_lock)
    journal.apply(config, _plan({"a": b"new"}), _snapshot(config))
    assert lock_calls == [fcntl.LOCK_EX | fcntl.LOCK_NB]
    with (config.project.root / ".unbake/journal.lock").open("a+b") as other:
        real_flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("content", [b"", b"\x00\xff\n", None])
def test_create_empty_binary_or_delete_absent(repo, content):
    config, _, _ = repo
    name = "nested/space [literal] 雪"
    assert journal.apply(config, _plan({name: content}), _snapshot(config)) == (name,)
    path = config.project.root / name
    if content is None:
        assert not path.exists()
    else:
        assert path.read_bytes() == content


def test_holding_the_journal_blocks_other_writers_and_apply_writes_inside_it(repo):
    config, _, _ = repo
    path = config.project.root / ".unbake/journal.lock"
    with journal.held(config):
        with path.open("a+b") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert journal.apply(config, _plan({"a": b"new"}), _snapshot(config)) == ("a",)  # no second lock
        assert journal.recover(config) is False
    with path.open("a+b") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # released once the landing is over
    assert not journal._HELD


def test_the_swap_renames_only_after_every_new_content_is_staged(repo, monkeypatch):
    config, _, _ = repo
    _seed(config, {"a": b"old a", "b": b"old b"})
    seen = []
    real = os.replace

    def replace(source, target):
        seen.append(sorted(p.name for p in config.project.root.glob(".unbake-*")))
        real(source, target)

    monkeypatch.setattr(journal.os, "replace", replace)
    journal.apply(config, _plan({"a": b"new a", "b": b"new b"}), _snapshot(config))
    assert seen[0] == seen[0][:2] and len(seen[0]) == 2  # both staged files existed at the first rename
    assert not list(config.project.root.rglob(".unbake-*"))


def test_a_refused_apply_leaves_no_staged_files(repo):
    config, _, _ = repo
    _seed(config, {"a": b"old"})
    with pytest.raises(Refusal):
        journal.apply(config, _plan({"a": b"new"}), _snapshot(config, {"a": b"other"}))
    assert not list(config.project.root.rglob(".unbake-*"))


def test_a_reader_waits_for_a_swap_and_a_crashed_swap_is_rolled_back_before_it_reads(repo):
    config, _, _ = repo
    _seed(config, {"a": b"old"})
    with journal.held(config):
        pass
    lock = config.project.root / ".unbake/journal.lock"
    with lock.open("a+b") as writer:
        fcntl.flock(writer.fileno(), fcntl.LOCK_EX)
        with lock.open("a+b") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)  # a reader would wait here
        fcntl.flock(writer.fileno(), fcntl.LOCK_UN)
    _manifest(config, "p", {"a": b"old"})
    (config.project.root / "a").write_bytes(b"half")  # a crash left a half swap
    with journal.reading(config):
        assert (config.project.root / "a").read_bytes() == b"old"
        assert not (config.project.root / ".unbake/journal.json").exists()
        with lock.open("a+b") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # writers wait while it reads
