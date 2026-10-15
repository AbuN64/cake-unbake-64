"""config: packaged resources, templates, host auto resolution, project metadata and refusals."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fixture
import psutil
import pytest

from unbake import config as configuration
from unbake.contracts import Refusal


def _refusal(call: Any, *args: Any) -> Refusal:
    with pytest.raises(Refusal) as caught:
        call(*args)
    return caught.value


def _keys(refusal: Refusal) -> set[str]:
    return {f.key for f in refusal.findings}


@pytest.mark.parametrize("name", list(configuration.SCHEMA_OF))
def test_load_resource_every_data_file(name: str) -> None:
    document = configuration.load_resource(name)
    assert document


def test_schema_of_has_no_features() -> None:
    assert "features.toml" not in configuration.SCHEMA_OF
    assert len(configuration.SCHEMA_OF) == 9
    assert _keys(_refusal(configuration.load_resource, "features.toml")) == {"config.resource"}


def test_template_known_and_missing() -> None:
    assert "${units}" in configuration.template("Makefile.in")
    assert configuration.template("sdk/" + next(p for p in _sdk_names()))
    refusal = _refusal(configuration.template, "nope")
    assert _keys(refusal) == {"config.resource"}
    assert refusal.findings[0].path == "templates/nope"


def _sdk_names() -> list[str]:
    from importlib import resources

    return [p.name for p in (resources.files("unbake.resources") / "templates" / "sdk").iterdir() if p.is_file()]


def test_sentence_known() -> None:
    assert configuration.sentence("config.schema")


@pytest.fixture(autouse=True)
def _idle_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "getloadavg", lambda: (0.0, 0.0, 0.0))


def test_host_auto_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(12)))
    monkeypatch.setattr(configuration, "_cgroup_memory_max", lambda: "max")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=64 << 30))
    loaded = configuration.load_host(fixture.host(tmp_path, **{"resources.cores": "auto", "resources.workers": "auto"}))
    assert (loaded.cores, loaded.workers) == (12, 9)
    assert loaded.origins["resources.cores"].note == "auto: os.sched_getaffinity -> 12"
    assert loaded.origins["resources.workers"].note == "auto: max(1, floor(12 cores * 0.75) - load 0) -> 9"


def _auto_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpus: int, cgroup: str, total: int, **extra: Any
) -> Any:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(cpus)))
    monkeypatch.setattr(configuration, "_cgroup_memory_max", lambda: cgroup)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=total))
    overrides = {
        "resources.cores": "auto",
        "resources.workers": "auto",
        "resources.memory_parent_bytes": 2 << 30,
        "resources.memory_worker_bytes": 1 << 30,
        **extra,
    }
    return configuration.load_host(fixture.host(tmp_path, **overrides))


def test_host_auto_workers_memory_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _auto_host(tmp_path, monkeypatch, 16, str(12 << 30), 256 << 30)
    assert (loaded.cores, loaded.workers) == (16, 10)
    assert loaded.origins["resources.workers"].note == "auto: min(12 by capacity, 10 by memory) -> 10"
    assert loaded.origins["resources.cores"].note == "auto: os.sched_getaffinity -> 16"


def test_host_auto_workers_cpu_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _auto_host(tmp_path, monkeypatch, 64, "max", 128 << 30)
    assert (loaded.cores, loaded.workers) == (64, 48)
    assert loaded.origins["resources.workers"].note == "auto: max(1, floor(64 cores * 0.75) - load 0) -> 48"


def test_host_numeric_workers_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _auto_host(tmp_path, monkeypatch, 16, str(4 << 30), 4 << 30, **{"resources.workers": 8})
    assert loaded.workers == 8
    assert not loaded.origins["resources.workers"].note


def test_host_explicit_keeps_ints(tmp_path: Path) -> None:
    loaded = configuration.load_host(fixture.host(tmp_path))
    assert (loaded.cores, loaded.workers) == (2, 2)
    assert not loaded.origins["resources.cores"].note


def test_a_host_without_move_wait_s_still_loads_and_waits_600_seconds(tmp_path: Path) -> None:
    path = fixture.host(tmp_path)
    path.write_text(path.read_text().replace("move_wait_s = 5\n", "", 1))
    assert "move_wait_s" not in path.read_text()
    assert configuration.load_host(path).move_wait_s == 600.0
    assert configuration.load_host(fixture.host(tmp_path)).move_wait_s == 5.0


def test_cache_keep_commits_defaults_to_twenty_and_reads_the_host_value(tmp_path: Path) -> None:
    assert configuration.load_host(fixture.host(tmp_path)).cache_keep_commits == 20
    other = tmp_path / "other"
    other.mkdir()
    loaded = configuration.load_host(fixture.host(other, **{"cache.keep_commits": 7}))
    assert loaded.cache_keep_commits == 7


def test_host_workers_exceed_cores(tmp_path: Path) -> None:
    refusal = _refusal(configuration.load_host, fixture.host(tmp_path, **{"resources.workers": 3}))
    assert _keys(refusal) == {"config.file"}
    assert refusal.findings[0].path == "host:resources.workers"


def test_host_schema_1_refused(tmp_path: Path) -> None:
    path = fixture.host(tmp_path)
    path.write_text(path.read_text().replace("schema = 2", "schema = 1", 1))
    assert "config.schema" in _keys(_refusal(configuration.load_host, path))


def test_a_document_of_another_schema_version_is_one_finding_with_an_action() -> None:
    with pytest.raises(Refusal) as caught:
        configuration.validate("layout", {"schema": 1, "group": [{"name": "g", "split": 1}]}, "layout.toml")
    (finding,) = caught.value.findings
    assert (finding.key, finding.path) == ("config.schema", "layout.toml:schema") and "schema 5" in finding.action


def test_host_unknown_key_refused(tmp_path: Path) -> None:
    path = fixture.host(tmp_path)
    path.write_text(path.read_text() + "\n[bogus]\nx = 1\n")
    refusal = _refusal(configuration.load_host, path)
    assert _keys(refusal) == {"config.schema"}


@pytest.mark.parametrize(("extra", "path"), [("[search]\nx = 1\n", "host:search"), ("[cycle]\nx = 1\n", "host:cycle")])
def test_host_old_keys_are_refused_by_the_schema(tmp_path: Path, extra: str, path: str) -> None:
    host = fixture.host(tmp_path)
    host.write_text(host.read_text() + "\n" + extra)
    refusal = _refusal(configuration.load_host, host)
    assert _keys(refusal) == {"config.schema"}
    assert refusal.findings[0].path.startswith("host:")


def test_host_relative_tool_refused(tmp_path: Path) -> None:
    refusal = _refusal(configuration.load_host, fixture.host(tmp_path, **{"tools.git": "git"}))
    assert _keys(refusal) == {"config.file"}
    assert refusal.findings[0].path == "host:tools.git"


def test_host_missing_file_refused(tmp_path: Path) -> None:
    assert _keys(_refusal(configuration.load_host, tmp_path / "absent.toml")) == {"config.missing"}


def test_project_title_and_meta(tmp_path: Path, toolchains: dict[str, Any]) -> None:
    root = fixture.project(
        tmp_path,
        functions={"func_80000400": {v: (fixture.ROM_BASE, b"\x03\xe0\x00\x08\x00\x00\x00\x00") for v in "ab"}},
        region={"a": "North America"},
    )
    project = configuration.load_project(root)
    assert project.title == "Fixture"
    assert project.version_files["a"].meta == {"region": "North America"}
    assert project.version_files["b"].meta == {}


def test_project_old_key_is_refused_by_the_schema(tmp_path: Path, toolchains: dict[str, Any]) -> None:
    config = fixture.config(tmp_path).project.root / "config.toml"
    config.write_text(config.read_text() + "\n[compilers]\nx = 1\n")
    refusal = _refusal(configuration.load_project, config.parent)
    assert _keys(refusal) == {"config.schema"}


def test_project_unknown_key(tmp_path: Path, toolchains: dict[str, Any]) -> None:
    config = fixture.config(tmp_path).project.root / "config.toml"
    config.write_text(config.read_text() + "\n[bogus]\nx = 1\n")
    assert _keys(_refusal(configuration.load_project, config.parent)) == {"config.schema"}


def test_project_sha1_mismatch(tmp_path: Path, toolchains: dict[str, Any]) -> None:
    config = fixture.config(tmp_path).project.root / "config.toml"
    text = config.read_text()
    sha = next(line for line in text.splitlines() if line.startswith("baserom_sha1")).split('"')[1]
    config.write_text(text.replace(sha, "0" * 40))
    refusal = _refusal(configuration.load_project, config.parent)
    assert _keys(refusal) == {"config.file"}
    assert refusal.findings[0].path.endswith(".baserom_sha1")


def test_load_combines_digests(cfg: Any) -> None:
    assert cfg.digest
    assert cfg.digest != cfg.project.digest


@pytest.mark.parametrize("tool", ["cpp", "mips_as", "mips_ld", "mips_objcopy", "mips_objdump", "n64link", "armips"])
def test_optional_host_tool_absent_and_bad_path(tmp_path: Path, tool: str) -> None:
    path = fixture.host(tmp_path)
    path.write_text("\n".join(line for line in path.read_text().splitlines() if not line.startswith(tool + " = ")))
    assert tool not in configuration.load_host(path).tools
    invalid = fixture.host(tmp_path, **{f"tools.{tool}": str(tmp_path / "absent")})
    refusal = _refusal(configuration.load_host, invalid)
    assert refusal.findings[0].key == "config.file"
    assert refusal.findings[0].path == f"host:tools.{tool}"


def test_no_layout_cap_anywhere_in_the_resources() -> None:
    assert "layout_cap" not in configuration.template("config.toml")
    assert "layout_cap" not in json.dumps(configuration._schema("project"))
    assert "cap" not in configuration._schema("layout")["properties"]


@pytest.mark.parametrize(
    "cpus,cap,load,workers",
    [(12, None, 0.0, 9), (12, None, 22.5, 1), (12, None, 3.9, 6), (12, 0.5, 2.0, 4), (4, 0.1, 0.0, 1)],
)
def test_host_auto_workers_size_from_capacity_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpus: int, cap: float | None, load: float, workers: int
) -> None:
    monkeypatch.setattr(os, "getloadavg", lambda: (load, load, load))
    extra = {"resources.cap": cap} if cap else {}
    loaded = _auto_host(tmp_path, monkeypatch, cpus, "max", 128 << 30, **extra)
    assert (loaded.cores, loaded.workers) == (cpus, workers)
