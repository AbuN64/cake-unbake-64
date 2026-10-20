"""Native execution with per-child accounting and process-group timeouts."""

import ctypes
import hashlib
import os
import selectors
import signal
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, suppress
from functools import cache
from pathlib import Path
from time import perf_counter_ns
from typing import IO

from unbake import effort
from unbake.contracts import Config, Finding, NativeResult, Refusal


def _kill(pid: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


_PR_SET_CHILD_SUBREAPER = 36  # linux/prctl.h


@cache
def _subreaper() -> None:
    """Once per process (the setting is process-wide): orphans of a killed child come to us to be reaped."""
    ctypes.CDLL(None).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)


def _reap(group: int) -> float:
    """Kills what the child left in its process group and reaps it, returning the CPU seconds of every member: a
    killed child never waited for its own children, which are ours (subreaper) and only we can account for."""
    _kill(group)
    cpu = 0.0
    while True:
        try:
            _, _, usage = os.wait4(-group, 0)
        except ChildProcessError:
            return cpu
        cpu += usage.ru_utime + usage.ru_stime


def scratch(root: Path) -> Path:
    """The temporary directory every child of a project gets (TMPDIR), created when first needed."""
    path = root / ".unbake" / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _execute(
    name: str,
    argv: Sequence[str | Path],
    cwd: Path,
    stdin: bytes,
    stdout_path: Path | None,
    timeout: float | None,
    outputs: Mapping[str, Path],
    tmp: Path,
    env: Mapping[str, str] = {},
    pass_fds: Sequence[int] = (),
) -> NativeResult:
    start = perf_counter_ns()
    _subreaper()
    deadline = None if timeout is None else start + timeout * 1_000_000_000
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    with ExitStack() as stack:
        destination: int | IO[bytes] = subprocess.PIPE
        if stdout_path is not None:
            destination = stack.enter_context(stdout_path.open("wb"))
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=destination,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env={**os.environ, **env, "TMPDIR": str(tmp)},
            pass_fds=tuple(pass_fds),
        )
        try:
            selector = stack.enter_context(selectors.DefaultSelector())
            pipes: dict[str, IO[bytes]] = {}
            for role, out in (("stdout", proc.stdout), ("stderr", proc.stderr)):
                if out is not None:
                    pipes[role] = stack.enter_context(out)
                    os.set_blocking(out.fileno(), False)
                    selector.register(out, selectors.EVENT_READ, role)
            assert proc.stdin is not None  # opened with stdin=PIPE
            pipes["stdin"] = stack.enter_context(proc.stdin)
            pending = memoryview(stdin)
            if pending:
                os.set_blocking(proc.stdin.fileno(), False)
                selector.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                proc.stdin.close()
            while selector.get_map():
                remaining = None
                if deadline is not None:
                    remaining = (deadline - perf_counter_ns()) / 1_000_000_000
                    if remaining <= 0:
                        _kill(proc.pid)
                        deadline = None
                        remaining = None
                for key, _ in selector.select(remaining):
                    pipe = pipes[key.data]
                    if key.data == "stdin":
                        try:
                            pending = pending[os.write(pipe.fileno(), pending[:65536]) :]
                        except BrokenPipeError:
                            pending = pending[:0]
                        except BlockingIOError:
                            continue
                        if not pending:
                            selector.unregister(pipe)
                            pipe.close()
                    else:
                        try:
                            chunk = os.read(pipe.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        if chunk:
                            streams[key.data].extend(chunk)
                        else:
                            selector.unregister(pipe)
                            pipe.close()
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            orphans = _reap(proc.pid)
        finally:
            if proc.returncode is None:
                _kill(proc.pid)
                _, status, _ = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(status)
                _reap(proc.pid)
    code = proc.returncode
    result = NativeResult(
        tuple(str(arg) for arg in argv),
        str(cwd),
        code if code >= 0 else None,
        -code if code < 0 else None,
        bytes(streams["stdout"]),
        bytes(streams["stderr"]),
        (perf_counter_ns() - start) / 1_000_000_000,
        usage.ru_utime + usage.ru_stime + orphans,
        usage.ru_maxrss * 1024,
        dict(outputs),
    )
    effort.record_native(name, result, start)
    return result


def run(
    name: str,
    argv: Sequence[str | Path],
    cwd: Path,
    *,
    tmp: Path,
    stdin: bytes = b"",
    stdout_path: Path | None = None,
    timeout: float | None = None,
    outputs: Mapping[str, Path] = {},
    env: Mapping[str, str] = {},
    pass_fds: Sequence[int] = (),
) -> NativeResult:
    with effort.stage("process.run"):
        executable = str(argv[0]) if argv else ""
        if not executable or not Path(executable).is_file() or not os.access(executable, os.X_OK):
            raise Refusal(Finding("native.missing_tool", reason="the tool is not an executable file", path=executable))
        jobs = "-j" in [str(a)[:2] for a in argv[1:]] or "-j" in env.get("MAKEFLAGS", "").split(" ")[0][:2]
        if Path(executable).name == "make" and not jobs:
            raise Refusal(Finding("native.exit", reason="make runs only with -j from pool.cores or pool.jobserver"))
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            return _execute(name, argv, cwd, stdin, stdout_path, timeout, outputs, tmp, env, pass_fds)
        except OSError as error:
            raise Refusal(Finding("native.exit", reason=str(error))) from error


@cache
def _pin(name: str) -> tuple[str, str] | None:
    """(binary, sha256) of the host tool `name` when the project build fetches a pinned release of it (repo.toml
    make.tools with pins), else None."""
    from unbake import config as configuration

    for row in configuration.load_resource("repo.toml")["make"]["tools"].values():
        binary = row.get("binary", name)
        if row.get("host") == name and binary in (row.get("pins") or {}):
            return binary, row["pins"][binary]
    return None


@cache
def _digest(path: Path, size: int, mtime: int) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matches(path: Path | None, sha256: str) -> bool:
    if path is None or not path.is_file():
        return False
    stat = path.stat()
    return _digest(path, stat.st_size, stat.st_mtime_ns) == sha256


def tool(config: Config, name: str, *, kind: str | None = None) -> Path:
    """The host tool `name`. For a tool the project build fetches as a pinned release (n64link) the tree's own
    fetched copy (tools/<name>/) comes first when it is the pinned file: a measure assembles with the rules the build
    uses, not with another build of the tool that happens to sit at the host path."""
    if pin := _pin(name):
        binary, sha256 = pin
        own = config.project.root / "tools" / name / binary
        if _matches(own, sha256):
            return own
    try:
        return config.host.tools[name]
    except KeyError:
        raise Refusal(
            Finding(
                "native.missing_tool",
                reason=f"host tools.{name} is not configured" + (f" for unit kind {kind}" if kind else ""),
            )
        ) from None


def git(config: Config, *args: str) -> NativeResult:
    with effort.stage("process.git"):
        result = run(
            "git " + args[0],
            [tool(config, "git"), "-c", "core.untrackedCache=false", "-c", "core.fsmonitor=false", *args],
            config.project.root,
            tmp=scratch(config.project.root),
        )
        if result.exit != 0:
            lines = result.stderr.decode("utf-8", errors="replace").splitlines()
            raise Refusal(
                Finding(
                    "native.exit",
                    reason=lines[-1] if lines else "git exited with an error",
                    missing=(f"git {' '.join(args)} exit {result.exit}",),
                )
            )
        return result


def make(config: Config, targets: Sequence[str], jobs: int = 0) -> NativeResult:
    """One `make -j<jobs> <targets>` under the pool jobserver: the only process a build spawns, whatever its steps.
    `jobs` is the work the build expects (0: unknown); the jobserver grants the slots and grows into freed ones."""
    from unbake import pool  # pool imports store, which imports this module

    root = config.project.root
    with effort.stage("make"), pool.jobserver(config.host, jobs) as (_, env, fds):
        return run("make", [tool(config, "make"), "-k", *targets], root, tmp=scratch(root), env=env, pass_fds=fds)


def tracked(root: Path) -> set[str]:
    """The files git tracks under root; empty when root is no repository."""
    done = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=False)
    return {name for name in done.stdout.decode().split("\0") if name}
