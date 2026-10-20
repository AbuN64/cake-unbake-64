"""The generated CI: one forge-neutral entry point (the standalone check only) and thin wrappers that only call it."""

import re
from pathlib import Path

import yaml

import unbake
from unbake import ci
from unbake import config as configuration


def _files():
    return {path: data.decode() for path, data in ci.files().items()}


def test_rendering_is_a_pure_function_of_the_rules_and_never_names_a_release():
    assert ci.files() == ci.files()
    text = "".join(_files().values())
    assert "releases/download" not in text and unbake.__version__ not in text


def test_one_wrapper_per_forge_row_and_the_entry_point():
    rows = configuration.load_resource("repo.toml")["forge"]
    assert set(_files()) == {ci.RUN, *(row["path"] for row in rows)}


def test_every_wrapper_only_calls_the_one_entry_point():
    files = _files()
    github = yaml.safe_load(files[".github/workflows/ci.yml"])
    assert set(github[True]) == {"pull_request", "push", "workflow_dispatch"}  # yaml reads the key `on` as True
    assert set(github["jobs"]) == {"check"}
    runs = [step["run"] for job in github["jobs"].values() for step in job["steps"] if "run" in step]
    assert runs == ["sh ci/run check"]
    gitlab = yaml.safe_load(files[".gitlab-ci.yml"])
    assert {name: job["script"][-1].split("sh ci/run ")[1] for name, job in gitlab.items()} == {"check": "check"}


def test_the_entry_point_has_only_the_standalone_check_stage():
    run = _files()[ci.RUN]
    assert re.findall(r"^(\w+)\)$", run, re.M) == ["check"]
    check = re.search(r"^check\)\n(.*?)\n    ;;", run, re.M | re.S).group(1)
    assert check.splitlines() == ["    make check CHECK=tree"]
    assert "set -eu" in run


def test_the_game_repository_ci_never_installs_or_calls_our_tools():
    """The game repository builds standalone: no wheel, no pip, no tool command, no tool URL in any CI file."""
    text = "".join(_files().values()).lower()
    words = ("unbake", "recipe", "taste", "abu-decomp", "abun64", "pip install", "setup-python")
    assert not [w for w in words if w in text]


def test_nothing_generated_mentions_boards_or_the_external_site():
    text = "".join(_files().values()).lower()
    assert not [word for word in ("board", "bytematch", "recomp", "report.json", "progress.json") if word in text]
    for name in ("run.in", "ci.yml.in", "gitlab-ci.yml.in"):
        assert not [w for w in ("board", "bytematch", "recomp") if w in configuration.template(name).lower()]


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
