"""The generated CI: one forge-neutral entry point (check and audit only) and thin wrappers that only call it."""

import re
from pathlib import Path

import pytest
import yaml

import unbake
from unbake import ci
from unbake import config as configuration
from unbake.contracts import Refusal


def _files():
    return {path: data.decode() for path, data in ci.files().items()}


def test_rendering_is_a_pure_function_of_the_rules_and_release():
    assert ci.files() == ci.files()
    run = _files()[ci.RUN]
    assert f"releases/download/v{unbake.__version__}/unbake-{unbake.__version__}-py3-none-any.whl" in run


def test_one_wrapper_per_forge_row_and_the_entry_point():
    rows = configuration.load_resource("repo.toml")["forge"]
    assert set(_files()) == {ci.RUN, *(row["path"] for row in rows)}


def test_every_wrapper_only_calls_the_one_entry_point():
    files = _files()
    github = yaml.safe_load(files[".github/workflows/ci.yml"])
    assert set(github[True]) == {"pull_request", "push", "workflow_dispatch"}  # yaml reads the key `on` as True
    assert set(github["jobs"]) == {"check", "audit"}
    runs = [step["run"] for job in github["jobs"].values() for step in job["steps"] if "run" in step]
    assert runs == ["sh ci/run check", "sh ci/run audit"]
    gitlab = yaml.safe_load(files[".gitlab-ci.yml"])
    assert {name: job["script"][-1].split("sh ci/run ")[1] for name, job in gitlab.items()} == {
        "check": "check",
        "audit": "audit",
    }
    assert [path for path, text in files.items() if "unbake audit" in text] == [ci.RUN]


def test_the_entry_point_has_only_the_check_and_audit_stages():
    run = _files()[ci.RUN]
    assert re.findall(r"^(\w+)\)$", run, re.M) == ["check", "audit"]
    assert "make check CHECK=tree" in run


def test_nothing_generated_mentions_boards_or_the_external_site():
    text = "".join(_files().values()).lower()
    assert not [word for word in ("board", "bytematch", "recomp", "report.json", "progress.json") if word in text]
    for name in ("run.in", "ci.yml.in", "gitlab-ci.yml.in"):
        assert not [w for w in ("board", "bytematch", "recomp") if w in configuration.template(name).lower()]


def test_audit_base_falls_back_for_a_new_branch_and_a_shallow_base():
    run = _files()[ci.RUN]
    assert "*0000000000000000000000000000000000000000) b=4b825dc642cb6eb9a060e54bf8d69288fbee4904" in run
    assert 'git cat-file -e "$b^{tree}" 2>/dev/null || b=HEAD' in run
    assert "set -eu" in run


def test_pins_are_full_digests():
    text = "".join(_files().values())
    for line in text.splitlines():
        if "uses:" in line:
            assert "@" in line and len(line.split("@")[1].split()[0]) == 40
    for row in configuration.load_resource("repo.toml")["forge"]:
        for value in row["values"].values():
            assert "@" not in value or re.search(r"@(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})\b", value)


def test_no_forge_name_in_the_python_and_no_game_names():
    source = ci.__file__ or ""
    code = Path(source).read_text().lower()
    assert not [w for w in ("github", "gitlab") if w in code.replace("forge-neutral", "")]
    text = "".join(_files().values()).lower()
    assert not [word for word in ("ragewars", "battletanx", "turok") if word in text]


def test_the_check_stage_installs_unbake_before_it_runs_make_in_every_forge():
    """`make check CHECK=tree` calls `unbake sync --check`: each check job needs Python and the wheel."""
    files = _files()
    check = re.search(r"^check\)\n(.*?)\n    ;;", files[ci.RUN], re.M | re.S).group(1)
    assert check.splitlines() == [
        '    pip install "$WHEEL" "splat64[mips]==0.50.0" "spimdisasm==1.42.4"',
        "    host=$(mktemp -d)",
        '    unbake host --write "$host/host.toml"',
        '    export UNBAKE_HOST="$host/host.toml"',
        "    make check CHECK=tree",
    ]
    steps = yaml.safe_load(files[".github/workflows/ci.yml"])["jobs"]["check"]["steps"]
    assert steps[0]["uses"].startswith("actions/checkout") and steps[1]["uses"].startswith("actions/setup-python")
    assert steps[1]["with"]["python-version"] == "3.13" and steps[-1]["run"] == "sh ci/run check"
    gitlab = yaml.safe_load(files[".gitlab-ci.yml"])
    assert gitlab["check"]["image"].startswith("python:3.13")


def test_host_writes_a_loadable_host_config_with_auto_cores_beside_the_file(tmp_path, monkeypatch):
    tools = {name: tmp_path / "bin" / name for name in ("git", "make", "splat")}
    for tool in tools.values():
        tool.parent.mkdir(exist_ok=True)
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    result = ci.host({"write": tmp_path / "out" / "host.toml"})
    assert result["tools"] == ["git", "make", "splat"]
    host = configuration.load_host(tmp_path / "out" / "host.toml")
    assert 1 <= host.workers <= host.cores
    assert "auto" in host.origins["resources.cores"].note
    assert host.cache_dir == tmp_path / "out" / "cache" and set(host.tools) == {"git", "make", "splat"}


def test_host_refuses_a_runner_without_a_required_tool_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(ci.sys, "executable", str(tmp_path / "python"))
    with pytest.raises(Refusal) as caught:
        ci.host({"write": tmp_path / "host.toml"})
    assert [f.path for f in caught.value.findings] == ["git", "make", "splat"]
    assert not (tmp_path / "host.toml").exists()
