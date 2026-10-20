"""Command line entry: builds the click group from commands.toml and runs one command under effort."""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Sequence
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any

import click
from click.core import ParameterSource

from unbake import __version__, config, effort, human, store
from unbake.contracts import Finding, Json, Refusal


def _handler(name: str) -> Any:
    spec = config.load_resource("commands.toml")["command"][name]
    module, function = spec["handler"].split(".")
    return getattr(import_module(f"unbake.{module}"), function)


def _refused(result: Json | None) -> list[Finding]:
    """The findings of every entry a landing drain refused and of every bulk row refused at admission, all at once:
    a command whose work was refused did not succeed, and one run names everything to fix."""
    result = result or {}
    drain = (result["drain"] or {}) if "drain" in result else result  # a command that drained nothing holds None
    rows = result.get("submissions") or ()
    if isinstance(rows, dict):  # a command that submitted one bulk group holds the submit result itself
        rows = rows.get("submissions") or ()
    bodies = [*drain.get("refused", ()), *(row for row in rows if row.get("refused"))]
    return [
        Finding(**{**f, "versions": tuple(f["versions"]), "missing": tuple(f["missing"])})
        for body in bodies
        for f in body["findings"]
    ]


def _noted(result: Json | None) -> list[Finding]:
    """The non-blocking findings a command's result carries under `notes`."""
    return [
        Finding(**{**f, "versions": tuple(f.get("versions", ())), "missing": tuple(f.get("missing", ()))})
        for f in (result or {}).get("notes", ())
    ]


def _failed(findings: Sequence[Finding]) -> int:
    """1 when work was refused; a non-blocking finding is reported, never a failure."""
    return int(any(f.blocking for f in findings))


_SHORT = (
    "unit",
    "members",
    "version",
    "versions",
    "exact",
    "score",
    "words",
    "shift",
    "source",
    "diff",
    "alloc",
    "dumps",
    "notes",
    "sched",
    "variants",
)


def _emit(name: str, code: int, result: Json | None, findings: Sequence[Finding], short: bool = False) -> int:
    if code != 0 and not findings:  # a failure always names its cause
        findings = [Finding("internal.no-cause", f"{name} exited {code} without any finding naming why")]
    ordered = sorted(findings, key=lambda f: not f.blocking)
    stages = effort.tree()
    document = {
        "ok": code == 0,
        "invocation": effort.invocation(),
        "result": result,
        "findings": [asdict(f) for f in ordered],
        "stages": stages,
        "cache": effort.cache_total(stages),  # kept by --short too: the stage tree it drops is not where it is read
    }
    if short and result is not None:  # the verdict and the diff only: no stage tree or proofs
        missing = sorted({m for p in result.get("proofs", ()) for m in p["missing"]} | {*result.get("missing", ())})[:8]
        document = {
            **document,
            "stages": {},
            "result": {**{k: result[k] for k in _SHORT if k in result}, "proofs": [], "missing": missing},
        }
    sys.stdout.write(json.dumps(document, default=str) + "\n")
    human.finish(name, result, ordered)
    return code


def _run(name: str, params: dict[str, Any], argv: Sequence[str]) -> int:
    result: Json | None = None
    findings: list[Finding] = []
    code = 0
    cfg = None
    try:
        with effort.command(name, argv):
            host = params.pop("host")
            handler = _handler(name)
            if not config.load_resource("commands.toml")["command"][name].get("configured", True):
                root = Path(params.pop("project", None) or Path.cwd()).resolve()
                result = handler({**params, "host": host, "project": root})
            else:
                if host is None:
                    raise Refusal(Finding("config.missing", "no host config: pass --host or set UNBAKE_HOST"))
                root = Path(params.pop("project") or Path.cwd()).resolve()  # splat runs elsewhere: one absolute root
                cfg = config.load(root, host)
                effort.bind(cfg)
                result = handler(cfg, params)
                findings = [*_refused(result), *_noted(result)]
                code = _failed(findings)
    except Refusal as refusal:
        code, findings = 1, list(refusal.findings)
    except Exception as exc:
        sys.stderr.write(traceback.format_exc())
        code, findings = 3, [Finding("internal.error", repr(exc))]
    if cfg is not None:  # success, refusal or crash alike: the command's touched keys become one manifest
        store.finish_command(cfg)
    return _emit(name, code, result, findings, bool(params.get("short")))


def _param(p: Json) -> click.Parameter:
    kind = {"path": click.Path(path_type=Path), "int": int}.get(p["type"], str)
    flag = f"--{p['name']}"
    if p["kind"] == "argument":
        return click.Argument([p["name"]], required=p["required"], nargs=-1 if p["multiple"] else 1, type=kind)
    if p["kind"] == "option":
        return click.Option([flag], required=p["required"], multiple=p["multiple"], type=kind, help=p["help"])
    return click.Option([flag], is_flag=True, help=p["help"])


def _group(outcome: dict[str, int], argv: Sequence[str]) -> click.Group:
    path = click.Path(path_type=Path)
    group = click.Group(
        "unbake",
        params=[click.Option(["--host"], type=path, envvar="UNBAKE_HOST"), click.Option(["--project"], type=path)],
    )
    click.version_option(__version__, prog_name="unbake", message="%(prog)s %(version)s")(group)
    for name, spec in config.load_resource("commands.toml")["command"].items():
        params = [_param(p) for p in spec["params"]]
        params.append(click.Option(["--host"], type=path, required=False, envvar="UNBAKE_HOST", help="host config"))
        if name != "init":
            params.append(click.Option(["--project"], type=path, required=False, help="project root, default cwd"))

        def callback(_name: str = name, **kwargs: Any) -> None:
            context = click.get_current_context()
            parent = context.parent
            for key in ("host", "project"):
                if (
                    key in kwargs
                    and parent
                    and (
                        kwargs[key] is None
                        or (
                            context.get_parameter_source(key) != ParameterSource.COMMANDLINE
                            and parent.get_parameter_source(key) == ParameterSource.COMMANDLINE
                        )
                    )
                ):
                    kwargs[key] = parent.params.get(key)
            outcome["code"] = _run(_name, kwargs, argv)

        group.add_command(click.Command(name, params=params, callback=callback, help=spec["help"]))
    return group


def main(argv: Sequence[str] | None = None) -> int:
    human.attach(sys.stderr, sys.stderr.isatty())
    args = list(sys.argv[1:] if argv is None else argv)
    outcome: dict[str, int] = {"code": 0}
    try:
        _group(outcome, args).main(args=args, prog_name="unbake", standalone_mode=False)
    except click.exceptions.Exit as exit_:
        return int(exit_.exit_code)
    except click.ClickException as exc:
        sys.stderr.write(f"usage: {exc.format_message()}\n")
        return 2
    return outcome["code"]
