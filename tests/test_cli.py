"""cli: handlers resolve from commands.toml, one stdout envelope, exit codes 0/1/2/3, human stderr."""

from __future__ import annotations

import ast
import json
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import unbake
from unbake import cli, config, repo, store
from unbake.contracts import Finding, Refusal

Capture = pytest.CaptureFixture[str]
Patch = pytest.MonkeyPatch
COMMANDS = config.load_resource("commands.toml")["command"]


def _envelope(out: str) -> dict[str, Any]:
    lines = out.splitlines()
    assert len(lines) == 1
    document = json.loads(lines[0])
    config.validate("envelope", document, "envelope")
    return document


@pytest.fixture
def argv(tmp_path: Path, monkeypatch: Patch) -> list[str]:
    monkeypatch.delenv("UNBAKE_HOST", raising=False)
    cfg = SimpleNamespace(host=SimpleNamespace(workers=4), project=SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(config, "load", lambda root, host: cfg)
    monkeypatch.setattr(cli.effort, "bind", lambda cfg: None)
    monkeypatch.setattr(cli.store, "finish_command", lambda cfg: None)
    return ["setup", "--host", str(tmp_path / "host.toml"), "--project", str(tmp_path)]


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_every_command_has_handler(name: str) -> None:
    module, function = COMMANDS[name]["handler"].split(".")
    assert callable(getattr(import_module(f"unbake.{module}"), function))


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_every_declared_argument_option_and_flag_reaches_the_handler_params(name: str, monkeypatch: Patch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "_run", lambda command, params, argv: seen.update(params) or 0)
    given: list[str] = []
    for p in COMMANDS[name]["params"]:
        value = "1" if p["type"] == "int" else "x"
        given += [value] if p["kind"] == "argument" else [f"--{p['name']}"] + ([value] if p["kind"] == "option" else [])
    assert cli.main([name, *given]) == 0
    assert {p["name"].replace("-", "_") for p in COMMANDS[name]["params"]} <= seen.keys()
    assert {"host", "project"} <= seen.keys() or (name == "init" and "host" in seen)
    module, function = COMMANDS[name]["handler"].split(".")  # and the handler reads nothing undeclared
    tree = ast.parse(Path(import_module(f"unbake.{module}").__file__ or "").read_text())
    handler = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    read = {
        n.slice.value
        for n in ast.walk(handler)
        if isinstance(n, ast.Subscript)
        and isinstance(n.value, ast.Name)
        and n.value.id == "params"
        and isinstance(n.slice, ast.Constant)
    }
    read |= {
        n.args[0].value
        for n in ast.walk(handler)
        if isinstance(n, ast.Call)
        and n.args
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "params"
        and n.func.attr == "get"
        and isinstance(n.args[0], ast.Constant)
    }
    assert read <= seen.keys(), f"{function} reads {sorted(read - seen.keys())} which {name} never delivers"


def test_stdout_one_envelope(argv: list[str], monkeypatch: Patch, capsys: Capture) -> None:
    monkeypatch.setattr(repo, "setup", lambda cfg, params: {})
    assert cli.main(argv) == 0
    document = _envelope(capsys.readouterr().out)
    assert document["ok"] is True
    assert document["result"] == {}


def test_refusal_exit_1_and_human_stderr(argv: list[str], monkeypatch: Patch, capsys: Capture) -> None:
    def refuse(cfg: Any, params: Any) -> None:
        raise Refusal(Finding("config.missing", "no value"))

    monkeypatch.setattr(repo, "setup", refuse)
    assert cli.main(argv) == 1
    captured = capsys.readouterr()
    assert "Refused:" in captured.err
    document = _envelope(captured.out)
    assert document["ok"] is False
    assert document["findings"][0]["key"] == "config.missing"


def test_usage_exit_2(capsys: Capture) -> None:
    assert cli.main(["no-such-command"]) == 2
    assert capsys.readouterr().err.startswith("usage:")


def test_internal_exit_3(argv: list[str], monkeypatch: Patch, capsys: Capture) -> None:
    def explode(cfg: Any, params: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(repo, "setup", explode)
    assert cli.main(argv) == 3
    captured = capsys.readouterr()
    assert "Traceback" in captured.err
    document = _envelope(captured.out)
    assert document["findings"][0]["key"] == "internal.error"


def test_refused_landing_entries_fail_the_command_with_their_findings(
    argv: list[str], monkeypatch: Patch, capsys: Capture
) -> None:
    refused = {
        "id": "x",
        "member": "f",
        "findings": [Finding("land.not_exact", "f differs", unit="f", versions=("a",), missing=("size",)).__dict__],
    }
    drain = {"running": False, "landed": [], "refused": [refused]}
    monkeypatch.setattr(repo, "setup", lambda cfg, params: {"drain": drain})
    assert cli.main(argv) == 1
    document = _envelope(capsys.readouterr().out)
    assert document["ok"] is False
    assert [(f["key"], f["unit"]) for f in document["findings"]] == [("land.not_exact", "f")]
    assert document["result"]["drain"]["refused"]


def test_every_refused_bulk_row_fails_the_command_with_all_its_findings_at_once(
    argv: list[str], monkeypatch: Patch, capsys: Capture
) -> None:
    def row(member: str, refused: bool, *keys: str) -> dict[str, Any]:
        found = [Finding(k, f"{member} {k}", unit=member).__dict__ for k in keys]
        return {"member": member, "id": None if refused else "s", "refused": refused, "findings": found}

    rows = [
        row("f", True, "source.shared-declarations", "land.not_exact"),
        row("g", False, "land.not_exact"),  # a fuzzy entry carries its gaps but was not refused
        row("h", True, "land.rebase"),
    ]
    monkeypatch.setattr(repo, "setup", lambda cfg, params: {"submissions": rows, "drain": None})
    assert cli.main(argv) == 1
    document = _envelope(capsys.readouterr().out)
    assert [(f["unit"], f["key"]) for f in document["findings"]] == [
        ("f", "source.shared-declarations"),
        ("f", "land.not_exact"),
        ("h", "land.rebase"),
    ]
    assert [r["member"] for r in document["result"]["submissions"]] == ["f", "g", "h"]  # the rows stay in the result


@pytest.mark.parametrize("nested", [False, True])
def test_a_dry_or_grouped_submission_result_is_read_not_crashed_on(
    argv: list[str], monkeypatch: Patch, capsys: Capture, nested: bool
) -> None:
    # a submit returns submissions None when dry and the whole submit result ({"submissions": [...]}) when not
    bad = {"member": "f", "id": None, "refused": True, "findings": [Finding("land.not_exact", "f").__dict__]}
    held = {"submissions": [bad], "drain": None} if nested else None
    monkeypatch.setattr(repo, "setup", lambda cfg, params: {"submissions": held, "drain": None})
    assert cli.main(argv) == (1 if nested else 0)
    keys = [f["key"] for f in _envelope(capsys.readouterr().out)["findings"]]
    assert keys == (["land.not_exact"] if nested else [])


def test_a_command_that_drained_nothing_succeeds(argv: list[str], monkeypatch: Patch, capsys: Capture) -> None:
    monkeypatch.setattr(repo, "setup", lambda cfg, params: {"drain": None, "exact": 3})
    assert cli.main(argv) == 0
    assert _envelope(capsys.readouterr().out)["ok"] is True


def test_a_failed_exit_without_findings_names_itself(capsys: Capture) -> None:
    cli.human.attach(sys.stderr, False)
    with cli.effort.command("submit", []):  # the envelope names the command that ran
        assert cli._emit("submit", 1, None, []) == 1
    document = _envelope(capsys.readouterr().out)
    assert document["ok"] is False
    assert [f["key"] for f in document["findings"]] == ["internal.no-cause"]
    assert "submit exited 1" in document["findings"][0]["reason"]


def test_missing_host_refuses_config_missing(tmp_path: Path, monkeypatch: Patch, capsys: Capture) -> None:
    monkeypatch.delenv("UNBAKE_HOST", raising=False)
    assert cli.main(["setup", "--project", str(tmp_path)]) == 1
    document = _envelope(capsys.readouterr().out)
    assert document["findings"][0]["key"] == "config.missing"


def test_global_host_and_project(argv: list[str], monkeypatch: Patch, capsys: Capture) -> None:
    monkeypatch.setattr(repo, "setup", lambda cfg, params: {"workers": cfg.host.workers})
    assert cli.main([*argv[1:], argv[0]]) == 0
    assert _envelope(capsys.readouterr().out)["result"]["workers"] > 0


def test_command_host_overrides_global(argv: list[str], monkeypatch: Patch, capsys: Capture) -> None:
    monkeypatch.setattr(repo, "setup", lambda cfg, params: {})
    assert cli.main(["--host", "/not/configured", *argv]) == 0
    assert _envelope(capsys.readouterr().out)["ok"]


def test_a_relative_project_is_resolved_once_before_anything_reads_it(
    tmp_path: Path, monkeypatch: Patch, capsys: Capture
) -> None:
    seen: list[Path] = []

    def load(root: Path, host: Path) -> None:
        seen.append(root)
        raise Refusal(Finding("config.missing", "stop here"))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "load", load)
    monkeypatch.setattr(cli.human, "finish", lambda *a: None)
    cli._run("check", {"host": Path("host.toml"), "project": Path("proj/../proj")}, [])
    assert seen == [tmp_path.resolve() / "proj"]


def test_score_short_prints_the_verdict_and_diff_without_the_stage_tree_or_proofs(
    monkeypatch: Patch, capsys: Capture
) -> None:
    diff = [f"-{i:08x} lw $t0, 0x{i:X}($a0)" for i in range(20)]
    result = {
        "unit": "src/f.c",
        "members": ["f"],
        "version": "a",
        "versions": ["a"],
        "exact": False,
        "score": 0.5,
        "words": {"built": 20, "target": 20, "matched": 10},
        "shift": {
            "first_difference": {"index": 3, "built": "x", "target": "y"},
            "aligned": {"exact": 10, "register_blind": 12},
        },
        "source": [],
        "diff": diff,
        "proofs": [{"missing": ["m" * 500] * 30}],
    }
    monkeypatch.setattr(cli, "_handler", lambda name: lambda cfg, params: result)
    monkeypatch.setattr(cli.config, "load", lambda root, host: object())
    monkeypatch.setattr(cli.effort, "bind", lambda cfg: None)
    for flag, small in (([], False), (["--short"], True)):
        cli.main(["score", "f.c", "--host", "host.toml", *flag])
        out = capsys.readouterr().out
        document = _envelope(out)
        assert (len(out) < 2048) == small
        assert document["result"]["diff"] == diff and (document["result"]["proofs"] == []) == small


def test_the_command_set_is_the_C9_set() -> None:
    assert set(COMMANDS) == {
        "blobs",
        "compilers",
        "layout",
        "names",
        "init",
        "setup",
        "sync",
        "check",
        "audit",
        "cheat",
        "renameproof",
        "buildproof",
        "compare",
        "score",
        "probe",
        "order",
        "deadcode",
        "submit",
        "land",
        "report",
        "map",
        "symbols",
        "verify",
        "types",
    }
    assert [p["name"] for p in COMMANDS["sync"]["params"]] == ["check", "settle", "makefile", "place"]


def test_sync_check_reaches_the_handler(argv: list[str], monkeypatch: Patch, capsys: Capture) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(repo, "sync", lambda cfg, params: seen.append(params) or {"written": [], "stand_ins": {}})
    assert cli.main(["sync", "--check", *argv[1:]]) == 0
    assert seen[0]["check"] is True and _envelope(capsys.readouterr().out)["ok"]


def test_a_refused_sync_check_names_the_files(argv: list[str], monkeypatch: Patch, capsys: Capture) -> None:
    def refuse(cfg: Any, params: Any) -> None:
        raise Refusal(Finding("sync.changed", "2 files differ", missing=("Makefile", "game.ld")))

    monkeypatch.setattr(repo, "sync", refuse)
    assert cli.main(["sync", "--check", *argv[1:]]) == 1
    assert _envelope(capsys.readouterr().out)["findings"][0]["missing"] == ["Makefile", "game.ld"]


def test_short_score_keeps_missing(monkeypatch: Patch, capsys: Capture) -> None:
    result = {
        "unit": "src/f.c",
        "exact": False,
        "proofs": [{"missing": [f"m{i}" for i in range(12)]}, {"missing": ["m0", "z"]}],
    }
    monkeypatch.setattr(cli, "_handler", lambda name: lambda cfg, params: result)
    monkeypatch.setattr(cli.config, "load", lambda root, host: object())
    monkeypatch.setattr(cli.effort, "bind", lambda cfg: None)
    cli.main(["score", "f.c", "--host", "host.toml", "--short"])
    document = _envelope(capsys.readouterr().out)
    assert (
        document["result"]["proofs"] == []
        and document["result"]["missing"] == sorted({*(f"m{i}" for i in range(12)), "z"})[:8]
    )


def test_audit_runs_without_host(tmp_path: Path, monkeypatch: Patch, capsys: Capture) -> None:
    monkeypatch.delenv("UNBAKE_HOST", raising=False)
    seen: list[dict[str, Any]] = []
    loaded: list[Any] = []

    def audit(params: dict[str, Any]) -> dict[str, Any]:
        seen.append(params)
        return {"against": params["against"], "paths": [], "counts_before": {}, "counts_after": {}}

    monkeypatch.setattr(cli, "_handler", lambda name: audit)
    monkeypatch.setattr(config, "load", lambda *a: loaded.append(a))
    assert cli.main(["audit", "--against", "origin/main", "--project", str(tmp_path)]) == 0
    assert _envelope(capsys.readouterr().out)["ok"] and not loaded
    assert seen[0]["project"] == tmp_path.resolve() and seen[0]["against"] == "origin/main"


@pytest.mark.parametrize("outcome", ["ok", "refused", "crash"])
def test_every_command_ends_with_exactly_one_finish(argv: list[str], monkeypatch: Patch, outcome: str) -> None:
    def setup(cfg: Any, params: Any) -> dict[str, Any]:
        if outcome == "refused":
            raise Refusal(Finding("config.missing", "no value"))
        if outcome == "crash":
            raise RuntimeError("boom")
        return {}

    ended: list[Any] = []
    monkeypatch.setattr(repo, "setup", setup)
    monkeypatch.setattr(store, "finish_command", ended.append)
    cli.main(argv)
    assert len(ended) == 1


@pytest.mark.parametrize(
    "global_value, command_value, ambient, expected",
    [
        ("global.toml", None, "environment.toml", "global.toml"),
        ("global.toml", "command.toml", "environment.toml", "command.toml"),
        (None, "command.toml", "environment.toml", "command.toml"),
        (None, None, "environment.toml", "environment.toml"),
        ("global.toml", None, None, "global.toml"),
        (None, None, None, None),
    ],
)
def test_host_selection_uses_click_parameter_sources(monkeypatch, global_value, command_value, ambient, expected):
    if ambient is None:
        monkeypatch.delenv("UNBAKE_HOST", raising=False)
    else:
        monkeypatch.setenv("UNBAKE_HOST", ambient)
    seen = {}
    monkeypatch.setattr(cli, "_run", lambda name, params, argv: seen.update(params) or 0)
    args = (["--host", global_value] if global_value else []) + ["report"]
    args += ["--host", command_value] if command_value else []
    assert cli.main(args) == 0
    assert seen["host"] == (Path(expected) if expected else None)


@pytest.mark.parametrize("command_value", [None, "command-project"])
def test_command_project_still_overrides_global_project(monkeypatch, command_value):
    seen = {}
    monkeypatch.setattr(cli, "_run", lambda name, params, argv: seen.update(params) or 0)
    args = ["--project", "global-project", "report"]
    args += ["--project", command_value] if command_value else []
    assert cli.main(args) == 0
    assert seen["project"] == Path(command_value or "global-project")


def test_version_prints_the_release(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"unbake {unbake.__version__}"


def test_every_result_carries_the_cache_counts_of_its_command_even_when_short(
    monkeypatch: Patch, capsys: Capture
) -> None:
    """The stage tree is what `--short` drops; the counts must not live only there."""

    def handler(cfg: Any, params: dict[str, Any]) -> dict[str, Any]:
        cli.effort.count("score", True)
        with cli.effort.stage("inner"):
            cli.effort.count("build", False)
            cli.effort.count("build", False)
        return {"unit": "src/f.c", "exact": False, "proofs": []}

    monkeypatch.setattr(cli, "_handler", lambda name: handler)
    monkeypatch.setattr(cli.config, "load", lambda root, host: object())
    monkeypatch.setattr(cli.effort, "bind", lambda cfg: None)
    for flag in ([], ["--short"]):
        cli.main(["score", "f.c", "--host", "host.toml", *flag])
        document = _envelope(capsys.readouterr().out)
        assert document["cache"] == {
            "hits": 1,
            "misses": 2,
            "kinds": {"build": {"hits": 0, "misses": 2}, "score": {"hits": 1, "misses": 0}},
        }


def test_a_refused_command_still_reports_its_cache_counts(monkeypatch: Patch, capsys: Capture) -> None:
    def handler(cfg: Any, params: dict[str, Any]) -> dict[str, Any]:
        cli.effort.count("extract", True)
        raise Refusal(Finding("setup.splat", "no"))

    monkeypatch.setattr(cli, "_handler", lambda name: handler)
    monkeypatch.setattr(cli.config, "load", lambda root, host: object())
    monkeypatch.setattr(cli.effort, "bind", lambda cfg: None)
    cli.main(["setup", "--host", "host.toml"])
    document = _envelope(capsys.readouterr().out)
    assert not document["ok"] and document["cache"]["hits"] == 1 and document["cache"]["misses"] == 0


def test_a_call_with_no_host_falls_back_to_the_user_config_or_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.delenv("UNBAKE_HOST", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.default_host() is None
    assert "~/.config/unbake/unbake.toml" in config.NO_HOST
    (tmp_path / ".config/unbake").mkdir(parents=True)
    (tmp_path / ".config/unbake/unbake.toml").write_text("")
    assert config.default_host() == tmp_path / ".config/unbake/unbake.toml"
