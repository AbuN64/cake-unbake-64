"""The generated CI of a game repository: one forge-neutral entry point (ci/run) and the thin wrapper files that call
it. The wrappers are data rows of repo.toml; this module names no forge. The CI runs only the repository's own make
(the game repository never installs or calls unbake); the source rules run where unbake operates on the tree from
outside. Everything is a pure function of the packaged rules, so a fresh sync writes the bytes that are committed."""

from __future__ import annotations

from string import Template

from unbake import config as configuration

RUN = "ci/run"
AUDIT = "ci/abucache-audit"


def _render(name: str, **values: object) -> bytes:
    return Template(configuration.template(name)).substitute(values).encode()


def files() -> dict[str, bytes]:
    """ci/run, the AbuCache64 ownership audit it runs, and one wrapper per forge row of repo.toml, by project path."""
    rules = configuration.load_resource("repo.toml")
    return {
        RUN: _render("run.in"),
        AUDIT: _render("abucache-audit.in"),
        **{row["path"]: _render(row["template"], **row["values"]) for row in rules["forge"]},
    }
