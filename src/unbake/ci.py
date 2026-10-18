"""The generated CI of a game repository: one forge-neutral entry point (ci/run) and the thin wrapper files that call
it. The wrappers are data rows of repo.toml; this module names no forge. Everything is a pure function of the packaged
rules and the release, so a fresh sync writes the bytes that are committed."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from string import Template

import unbake
from unbake import config as configuration
from unbake import store
from unbake.contracts import Finding, Json, Refusal

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
        RUN: _render("run.in", wheel=wheel(), splat=" ".join(f'"{r}"' for r in rules["splat_pip"])),
        **{row["path"]: _render(row["template"], **row["values"]) for row in rules["forge"]},
    }


_REQUIRED = ("git", "make", "splat")


def _tool(name: str) -> str | None:
    """A host tool by its command name: on PATH, else beside the running python (a venv not activated)."""
    command = f"mips-linux-gnu-{name.removeprefix('mips_')}" if name.startswith("mips_") else name
    found = shutil.which(command) or shutil.which(command, path=str(Path(sys.executable).parent))
    return str(Path(found).absolute()) if found else None


def host(params: Json) -> Json:
    """Write a host config for a CI runner, which has none: cores and workers "auto", the cache and toolchains beside
    the file, each tool found on PATH. A tool the schema leaves optional and the runner lacks is left out; git, make
    and splat are required, so a runner without one refuses here by name."""
    path = Path(params["write"]).absolute()
    values = dict(configuration.load_resource("repo.toml")["ci_host"])
    found = {
        name: _tool(name)
        for name in (
            "git",
            "make",
            "cpp",
            "mips_as",
            "mips_ld",
            "mips_objcopy",
            "mips_objdump",
            "n64link",
            "splat",
            "armips",
        )
    }
    missing = [
        Finding("native.missing_tool", f"{name} is not installed", path=name) for name in _REQUIRED if not found[name]
    ]
    if missing:
        raise Refusal(*missing)
    text = Template(configuration.template("host.toml")).substitute(
        cores='"auto"',
        workers='"auto"',
        cache_dir=path.parent / "cache",
        toolchain_root=path.parent / "toolchains",
        **{name: found[name] or "" for name in found},
        **values,
    )
    text = re.sub(r'^\w+ = ""\n', "", text, flags=re.M)  # an absent optional tool is not configured
    store.write(path, text.encode())
    return {"path": str(path), "tools": sorted(name for name, tool in found.items() if tool)}
