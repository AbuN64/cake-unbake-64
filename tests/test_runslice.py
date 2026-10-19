"""The run slice: decide whether to re-exec under systemd-run, with every host fact mocked."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import fixture
import pytest

from unbake import config as configuration
from unbake import runslice
from unbake.contracts import Refusal

IN = "0::/user.slice/user-1000.slice/user@1000.service/decomp.slice/run-r1.scope\n"
OUT = "0::/user.slice/user-1000.slice/session-3.scope\n"


def _done(code: int, err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, "", err)


def _enter(cgroup: str, which: bool = True, probe: Any = None, **extra: Any) -> tuple[bool, list[Any], list[str]]:
    execs: list[Any] = []
    warned: list[str] = []
    result = runslice.enter(
        "decomp.slice",
        ["/venv/bin/python", "-m", "unbake.pool", "make"],
        cgroup=lambda: cgroup,
        which=lambda name: "/usr/bin/systemd-run" if which else None,
        probe=probe or (lambda name: _done(0)),
        execvpe=lambda *call: execs.append(call),
        warn=warned.append,
        **extra,
    )
    return result, execs, warned


def test_a_process_already_in_the_slice_runs_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runslice.ENTERED, raising=False)
    assert _enter(IN) == (False, [], [])


def test_a_process_outside_the_slice_re_execs_its_own_argv_under_systemd_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runslice.ENTERED, raising=False)
    _, execs, warned = _enter(OUT)
    assert warned == []
    [(program, command, env)] = execs
    assert program == "systemd-run"
    assert command == [
        "systemd-run", "--user", "--scope", "--quiet", "--slice=decomp.slice", "--",
        "/venv/bin/python", "-m", "unbake.pool", "make",
    ]  # fmt: skip
    assert env[runslice.ENTERED] == "decomp.slice"


def test_a_slice_a_parent_already_tried_is_not_tried_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runslice.ENTERED, "decomp.slice")
    assert _enter(OUT) == (False, [], [])


def test_a_host_without_systemd_run_runs_in_place_with_one_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runslice.ENTERED, raising=False)
    result, execs, warned = _enter(OUT, which=False)
    assert (result, execs) == (False, [])
    assert len(warned) == 1 and "systemd-run is missing" in warned[0]


def test_a_refusing_systemd_run_runs_in_place_with_one_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runslice.ENTERED, raising=False)
    result, execs, warned = _enter(OUT, probe=lambda name: _done(1, "Failed to connect to bus\n"))
    assert (result, execs) == (False, [])
    assert len(warned) == 1 and "Failed to connect to bus" in warned[0]


def test_an_exec_that_fails_runs_in_place_with_one_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runslice.ENTERED, raising=False)

    def boom(*call: Any) -> None:
        raise OSError("exec format error")

    warned: list[str] = []
    result = runslice.enter(
        "decomp.slice", ["x"], cgroup=lambda: OUT, which=lambda n: "/x", probe=lambda n: _done(0), execvpe=boom,
        warn=warned.append,
    )  # fmt: skip
    assert result is False and len(warned) == 1


def test_in_slice_matches_a_whole_cgroup_component_only() -> None:
    assert runslice.in_slice("decomp.slice", IN)
    assert not runslice.in_slice("decomp.slice", OUT)
    assert not runslice.in_slice("decomp.slice", "0::/user.slice/notdecomp.slice/x.scope\n")


def test_a_host_file_names_the_slice_and_a_bad_name_is_refused(tmp_path: Path) -> None:
    path = fixture.host(tmp_path)
    assert configuration.load_host(path).run_slice is None
    path.write_text(path.read_text().replace("move_wait_s = 5\n", 'move_wait_s = 5\nrun_slice = "decomp.slice"\n', 1))
    assert configuration.load_host(path).run_slice == "decomp.slice"
    path.write_text(path.read_text().replace("decomp.slice", "decomp", 1))
    with pytest.raises(Refusal):
        configuration.load_host(path)


def test_start_enters_the_slice_the_host_file_names_and_ignores_a_missing_host_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fixture.host(tmp_path)
    path.write_text(path.read_text().replace("move_wait_s = 5\n", 'move_wait_s = 5\nrun_slice = "decomp.slice"\n', 1))
    entered: list[Any] = []
    monkeypatch.setattr(runslice, "enter", lambda *call: entered.append(call))
    runslice.start(tmp_path / "absent.toml")
    assert entered == []
    runslice.start(path)
    assert [call[0] for call in entered] == ["decomp.slice"]
