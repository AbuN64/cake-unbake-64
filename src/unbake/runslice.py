"""Start `unbake-run` inside the host's run slice, so a host that splits cores between users of the machine splits them
by slice weight and not by how many ssh sessions happen to be busy.

The host file names the slice (`resources.run_slice`, a systemd user slice such as `decomp.slice`). When the process
is not already in it, it re-execs itself under `systemd-run --user --scope`; make, gcc and the pools it starts inherit
the slice. A host without systemd-run, or one that refuses, runs the command in place with one warning line."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ENTERED = "UNBAKE_RUN_SLICE_ENTERED"  # the slice a parent already tried, so a refusal can never loop


def in_slice(slice_name: str, cgroup: str) -> bool:
    """Whether a /proc/<pid>/cgroup text puts the process under a cgroup directory named `slice_name`."""
    return any(slice_name in line.partition(":")[2].partition(":")[2].split("/") for line in cgroup.splitlines())


def _read_cgroup() -> str:
    try:
        return Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError:
        return ""


def _systemd_run(slice_name: str, command: Sequence[str]) -> list[str]:
    return ["systemd-run", "--user", "--scope", "--quiet", f"--slice={slice_name}", "--", *command]


def _probe(slice_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _systemd_run(slice_name, ["true"]), capture_output=True, text=True, timeout=15, check=False, env=_clean(None)
    )


def _clean(slice_name: str | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != ENTERED}
    if slice_name:
        env[ENTERED] = slice_name
    return env


def enter(
    slice_name: str | None,
    argv: Sequence[str],
    *,
    cgroup: Callable[[], str] = _read_cgroup,
    which: Callable[[str], str | None] = shutil.which,
    probe: Callable[[str], subprocess.CompletedProcess[str]] = _probe,
    execvpe: Callable[..., None] = os.execvpe,
    warn: Callable[[str], object] = lambda text: sys.stderr.write(text + "\n"),
) -> bool:
    """Re-execs `argv` inside `slice_name` and does not return when it does. Returns False when the command should
    run in place: no slice named, already in it, already tried by a parent, or the host cannot do it (one warning)."""
    if not slice_name or os.environ.get(ENTERED) == slice_name or in_slice(slice_name, cgroup()):
        return False
    if which("systemd-run") is None:
        warn(f"unbake-run: run slice {slice_name} not entered: systemd-run is missing; running in place")
        return False
    try:
        tried = probe(slice_name)
        reason = (
            "" if tried.returncode == 0 else (tried.stderr.strip().splitlines() or [f"exit {tried.returncode}"])[-1]
        )
    except (OSError, subprocess.SubprocessError) as exc:
        reason = str(exc)
    if reason:
        warn(f"unbake-run: run slice {slice_name} not entered: systemd-run refused ({reason}); running in place")
        return False
    command = _systemd_run(slice_name, argv)
    try:
        execvpe(command[0], command, _clean(slice_name))
    except OSError as exc:
        warn(f"unbake-run: run slice {slice_name} not entered: {exc}; running in place")
    return False


def start(host_file: Path) -> None:
    """The one call at the top of `unbake-run`: enter the host file's run slice for this very command line. A host file
    that is absent or unreadable is the command's own business (it runs alone, holding no slots); never fails here."""
    from unbake import config as configuration
    from unbake.contracts import Refusal

    try:
        name = configuration.load_host(host_file).run_slice
    except (Refusal, OSError, ValueError):
        return
    if name:
        enter(name, [sys.executable, *sys.orig_argv[1:]])
