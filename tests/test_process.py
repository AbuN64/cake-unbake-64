"""Process contract tests: tiny shell children, exact wait4 accounting, and mocked git."""

import ast
import os
import signal
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Thread, get_ident
from time import perf_counter, perf_counter_ns
from types import SimpleNamespace
from unittest.mock import Mock

import fixture
import psutil
import pytest

from unbake import effort, process
from unbake.contracts import Config, Host, Project, Refusal


@pytest.fixture
def process_config(tmp_path):
    """Build contract records without project(), which would invoke git."""
    host = Host(
        cores=2,
        workers=2,
        memory_parent_bytes=1 << 30,
        memory_worker_bytes=1 << 30,
        cache_dir=tmp_path / "cache",
        cache_max_bytes=1 << 30,
        toolchain_root=tmp_path / "tools",
        tools={"git": tmp_path / "git"},
        sdk_catalog=None,
        serial_seconds=2,
        serial_cores=2,
        pool_fill=0.8,
        pool_fanout=2,
        author=("test", "test@example.invalid"),
        origins={},
        digest="host",
    )
    project = Project(
        root=tmp_path,
        name="fixture",
        title="Fixture",
        toolchain="",
        names_from="",
        libraries=(),
        cppflags=(),
        cflags=(),
        asflags=(),
        readme_map=None,
        readme_map_link=None,
        versions=(),
        version={},
        files={},
        digest="project",
    )
    return Config(project, host, "config")


def test_exit_and_streams_kept(tmp_path):
    argv = ("/bin/sh", "-c", "echo a; echo b >&2; exit 3")
    result = process.run("streams", argv, tmp_path, tmp=tmp_path / "tmp")
    assert result.argv == argv
    assert result.cwd == str(tmp_path)
    assert result.exit == 3
    assert result.signal is None
    assert result.stdout == b"a\n"
    assert result.stderr == b"b\n"
    assert result.wall_seconds > 0
    assert result.cpu_seconds >= 0
    assert result.max_rss_bytes > 0


def test_concurrent_cpu_attributed(tmp_path, monkeypatch):
    barrier = Barrier(2)
    results, usages, errors = {}, {}, []
    real_wait4 = os.wait4

    def wait4(pid, options):
        value = real_wait4(pid, options)
        usages[get_ident()] = value[2].ru_utime + value[2].ru_stime
        return value

    monkeypatch.setattr(process.os, "wait4", wait4)

    def spin(count):
        try:
            barrier.wait(timeout=5)
            script = f"i=0; while [ $i -lt {count} ]; do i=$((i+1)); done"
            result = process.run("spin", ("/bin/sh", "-c", script), tmp_path, timeout=5, tmp=tmp_path / "tmp")
            results[count] = (result, usages[get_ident()])
        except BaseException as error:
            errors.append(error)

    threads = [Thread(target=spin, args=(count,)) for count in (30000, 180000)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=7)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors
    short, long = (results[count][0] for count in (30000, 180000))
    assert short.exit == long.exit == 0
    assert short.cpu_seconds > 0
    assert long.cpu_seconds > short.cpu_seconds * 2
    for result, own_cpu in results.values():
        assert result.cpu_seconds == pytest.approx(own_cpu, abs=1e-9)
    assert short.cpu_seconds + long.cpu_seconds == pytest.approx(sum(usages.values()), abs=1e-9)


@pytest.mark.parametrize("stdin", [b"", b"x" * (1 << 20)], ids=["empty-stdin", "blocked-stdin"])
def test_timeout_kills_group(tmp_path, stdin):
    # The deadline starts when the shell is spawned, so under load it may expire before the shell has started its
    # children: the group is found by its unique sleep duration, and may be empty only when the kill came first.
    marker = f"{30 + os.getpid() % 1000}.{perf_counter_ns() % 100000}"
    script = f"sleep {marker} & sleep {marker} & wait"
    started = perf_counter()
    result = process.run("timeout", ("/bin/sh", "-c", script), tmp_path, stdin=stdin, timeout=0.5, tmp=tmp_path / "tmp")
    elapsed = perf_counter() - started
    assert result.exit is None
    assert result.signal == signal.SIGKILL
    assert elapsed < 2
    assert 0.45 <= result.wall_seconds < 2
    for proc in psutil.process_iter(["cmdline", "status"]):
        if marker in " ".join(proc.info["cmdline"] or ()):
            # Orphans may await the container's reaper; zombies cannot execute.
            assert proc.info["status"] == psutil.STATUS_ZOMBIE, proc.info


def test_native_recorded(tmp_path):
    with effort.command("test-process", ()):
        result = process.run("native-child", ("/bin/sh", "-c", "echo recorded"), tmp_path, tmp=tmp_path / "tmp")
    records = effort.closed()
    native = [r for r in records if r.kind == "native"]
    assert len(native) == 1
    assert native[0].path[-1] == "native-child"
    assert native[0].native_cpu_seconds == result.cpu_seconds
    assert native[0].wall_seconds == result.wall_seconds
    assert native[0].worker_rss_peak_bytes == result.max_rss_bytes
    assert any(record.path[-1] == "process.run" for record in records if record.kind == "stage")
    command = next(r for r in records if r.kind == "command")
    assert command.native_cpu_seconds == result.cpu_seconds


@pytest.mark.parametrize("kind", ["missing", "nonexecutable", "directory"])
def test_missing_tool(tmp_path, monkeypatch, kind):
    path = tmp_path / "tool"
    if kind == "nonexecutable":
        path.write_text("#!/bin/sh\n")
        path.chmod(0o644)
    elif kind == "directory":
        path.mkdir()
    popen = Mock(side_effect=AssertionError("missing tool must not spawn"))
    monkeypatch.setattr(process.subprocess, "Popen", popen)
    with pytest.raises(Refusal) as caught:
        process.run("missing", (str(path),), tmp_path, tmp=tmp_path / "tmp")
    assert len(caught.value.findings) == 1
    finding = caught.value.findings[0]
    assert finding.key == "native.missing_tool"
    assert finding.path == str(path)
    popen.assert_not_called()


def test_no_creation_bypass():
    allowed = {"effort.py", "pool.py", "process.py", "runslice.py"}
    banned = ("subprocess", "multiprocessing", "concurrent.futures", "threading")
    violations = []
    for path in sorted(Path(process.__file__).parent.rglob("*.py")):
        if path.name in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            violations.extend(
                f"{path.name}:{node.lineno}: import {n}"
                for n in names
                if any(n == b or n.startswith(b + ".") for b in banned)
            )
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("redirect", [False, True])
def test_large_stdin_and_both_streams(tmp_path, redirect):
    payload = b"input\x00\xff\n" * 40000
    stdout_path = tmp_path / "stdout" if redirect else None
    outputs = {"artifact": tmp_path / "artifact"}
    result = process.run(
        "duplex",
        ("/bin/sh", "-c", "printf '%070000d' 0 >&2; cat"),
        tmp_path,
        stdin=payload,
        stdout_path=stdout_path,
        outputs=outputs,
        timeout=2,
        tmp=tmp_path / "tmp",
    )
    assert result.exit == 0
    assert result.signal is None
    assert result.stderr == b"0" * 70000
    assert result.outputs == outputs
    if redirect:
        assert result.stdout == b""
        assert stdout_path.read_bytes() == payload
    else:
        assert result.stdout == payload


def test_a_grandchild_the_child_leaves_behind_is_killed_and_reaped_by_us(tmp_path, monkeypatch):
    real_wait4, reaped = process.os.wait4, []

    def wait4(pid, options):
        found = real_wait4(pid, options)
        reaped.append((pid, found[0]))
        return found

    monkeypatch.setattr(process.os, "wait4", wait4)
    argv = ("/bin/sh", "-c", "sleep 300 >/dev/null 2>&1 & exit 0")
    result = process.run("orphan", argv, tmp_path, tmp=tmp_path / "tmp")
    child = reaped[0][0]
    assert result.exit == 0 and [pid for pid, _ in reaped] == [child, -child]


def test_wait4_accounting_and_record_callback(tmp_path, monkeypatch):
    real_popen, real_wait4 = process.subprocess.Popen, process.os.wait4
    seen = {}

    def popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        child.wait = Mock(side_effect=AssertionError("the runner reaps with wait4 only"))
        seen["child"], seen["kwargs"] = child, kwargs
        return child

    def wait4(pid, options):
        if pid < 0:
            return real_wait4(pid, options)
        seen["waited"] = pid
        reaped, status, _ = real_wait4(pid, options)
        return reaped, status, SimpleNamespace(ru_utime=0.5, ru_stime=0.25, ru_maxrss=4096)

    record = Mock()
    monkeypatch.setattr(process.subprocess, "Popen", popen)
    monkeypatch.setattr(process.os, "wait4", wait4)
    monkeypatch.setattr(effort, "record_native", record)
    before = perf_counter()
    result = process.run("accounting", ("/bin/sh", "-c", "exit 7"), tmp_path, tmp=tmp_path / "tmp")
    after = perf_counter()
    assert seen["kwargs"]["start_new_session"] is True
    assert seen["kwargs"]["cwd"] == tmp_path
    assert seen["waited"] == seen["child"].pid
    assert seen["child"].returncode == result.exit == 7
    seen["child"].wait.assert_not_called()
    assert result.cpu_seconds == 0.75
    assert result.max_rss_bytes == 4096 * 1024
    record.assert_called_once()
    name, recorded, start = record.call_args.args
    assert name == "accounting" and recorded is result
    assert before * 1e9 <= start <= after * 1e9


def test_tool_lookup(process_config):
    assert process.tool(process_config, "git") == process_config.host.tools["git"]
    with pytest.raises(Refusal) as caught:
        process.tool(process_config, "unconfigured")
    finding = caught.value.findings[0]
    assert finding.key == "native.missing_tool"
    assert finding.reason == "host tools.unconfigured is not configured"


@pytest.mark.parametrize("exit_code", [0, 1, None], ids=["success", "nonzero", "signal"])
def test_git_delegates_and_refuses(process_config, monkeypatch, exit_code):
    result = fixture.native_result(exit=exit_code, stderr=b"first\nlast error\n")
    if exit_code is None:
        result = replace(result, signal=9)
    run = Mock(return_value=result)
    monkeypatch.setattr(process, "run", run)
    with effort.command("mock-git", ()):
        if exit_code == 0:
            assert process.git(process_config, "status", "--short") is result
        else:
            with pytest.raises(Refusal) as caught:
                process.git(process_config, "status", "--short")
            finding = caught.value.findings[0]
            assert finding.key == "native.exit"
            assert finding.reason == "last error"
            assert finding.missing == (f"git status --short exit {exit_code}",)
    root = process_config.project.root
    run.assert_called_once_with(
        "git status",
        [
            process_config.host.tools["git"],
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.fsmonitor=false",
            "status",
            "--short",
        ],
        root,
        tmp=process.scratch(root),
    )
    assert any(r.path[-1] == "process.git" for r in effort.closed() if r.kind == "stage")


def test_children_get_the_project_tmpdir(tmp_path):
    scratch = process.scratch(tmp_path)
    result = process.run("env", ("/bin/sh", "-c", 'printf %s "$TMPDIR"'), tmp_path, tmp=scratch)
    assert result.stdout.decode() == str(tmp_path / ".unbake" / "tmp") and scratch.is_dir()


def test_a_spawned_tool_keeps_the_priority_of_its_caller(tmp_path):
    result = process.run("nice", ("/usr/bin/nice",), tmp_path, tmp=tmp_path / "tmp")
    assert int(result.stdout) == os.nice(0)


@pytest.mark.parametrize(
    ("argv", "env", "runs"),
    [(["-j2"], {}, True), ([], {"MAKEFLAGS": "-j2 --jobserver-auth=3,4"}, True), ([], {}, False)],
)
def test_make_runs_only_with_a_j_or_a_jobserver(tmp_path, monkeypatch, argv, env, runs):
    make = tmp_path / "make"
    make.write_text("#!/bin/sh\n")
    make.chmod(0o755)
    ran = []
    monkeypatch.setattr(process, "_execute", lambda *a: ran.append(a) or "ok")
    if runs:
        assert process.run("make", [make, *argv], tmp_path, tmp=tmp_path / "t", env=env) == "ok" and len(ran) == 1
    else:
        with pytest.raises(Refusal):
            process.run("make", [make, *argv], tmp_path, tmp=tmp_path / "t", env=env)
        assert ran == []
