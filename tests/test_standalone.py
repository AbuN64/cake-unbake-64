"""A game repo builds with make alone: nothing unbake ships into it names a tool."""

import subprocess
import tomllib
from importlib import resources

import fixture

from unbake import config as configuration
from unbake import process


def _data(name):
    return tomllib.loads(resources.files("unbake.resources.data").joinpath(name).read_text())


def test_toolchain_flags_and_ignore_rules_name_no_tool():
    rows = _data("toolchains.toml")["toolchain"].values()
    assert all("unbake" not in " ".join(row.get("cppflags", ())).lower() for row in rows)
    assert all("unbake" not in line.lower() for line in _data("repo.toml")["ignore"])


def test_shipped_sdk_headers_guard_without_the_tool_name():
    headers = [
        h for h in resources.files("unbake.resources.templates").joinpath("sdk").iterdir() if h.name.endswith(".h")
    ]
    assert headers and all("UNBAKE_" not in h.read_text() for h in headers)


def test_state_directory_ignores_itself(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = process.state(tmp_path)
    (path / "check.json").write_text("{}")
    (process.scratch(tmp_path) / "x").write_text("")
    assert (path / ".gitignore").read_text() == "*\n"
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout == ""


def test_loading_a_project_makes_its_state_directory_ignored(tmp_path, toolchains):
    root = fixture.project(tmp_path)
    configuration.load(root, fixture.host(tmp_path))
    assert (root / ".unbake/.gitignore").read_text() == "*\n"
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True)
    assert status.stdout == ""
