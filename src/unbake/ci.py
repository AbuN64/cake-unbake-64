"""The generated CI of a game repository: one forge-neutral entry point (ci/run) and the thin wrapper files that call
it. The wrappers are data rows of repo.toml; this module names no forge. Everything is a pure function of the packaged
rules and the release, so a fresh sync writes the bytes that are committed."""

from __future__ import annotations

from string import Template

import unbake
from unbake import config as configuration

RUN = "ci/run"


def wheel(version: str | None = None) -> str:
    v = unbake.__version__ if version is None else version
    return str(configuration.load_resource("repo.toml")["wheel"]).format(version=v)


def _render(name: str, **values: object) -> bytes:
    return Template(configuration.template(name)).substitute(values).encode()


def files() -> dict[str, bytes]:
    """ci/run and one wrapper per forge row of repo.toml, by project path."""
    rules = configuration.load_resource("repo.toml")
    return {
        RUN: _render("run.in", wheel=wheel()),
        **{row["path"]: _render(row["template"], **row["values"]) for row in rules["forge"]},
    }
