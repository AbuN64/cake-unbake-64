"""config: packaged resources, templates, host auto resolution, project metadata and refusals."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fixture
import psutil
import pytest

from unbake import config as configuration
from unbake import contracts
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
    assert len(configuration.SCHEMA_OF) == 8
    assert _keys(_refusal(configuration.load_resource, "features.toml")) == {"config.resource"}


def test_template_known_and_missing() -> None:
    assert "${compile_recipes}" in configuration.template("Makefile.in")
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
    assert (loaded.cores, loaded.workers) == (12, 12)
    assert loaded.origins["resources.cores"].note == "auto: os.sched_getaffinity -> 12"
    assert loaded.origins["resources.workers"].note == "auto: min(affinity 12, cores 12) -> 12"


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
    assert loaded.origins["resources.workers"].note == "auto: min(16 by capacity, 10 by memory) -> 10"
    assert loaded.origins["resources.cores"].note == "auto: os.sched_getaffinity -> 16"


def test_host_auto_workers_cpu_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _auto_host(tmp_path, monkeypatch, 64, "max", 128 << 30)
    assert (loaded.cores, loaded.workers) == (64, 64)
    assert loaded.origins["resources.workers"].note == "auto: min(affinity 64, cores 64) -> 64"


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


def test_workers_honour_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "getloadavg", lambda: (0.4, 0.4, 0.4))
    explicit = _auto_host(tmp_path, monkeypatch, 12, "max", 128 << 30, **{"resources.workers": 12})
    assert explicit.workers == 12
    auto = _auto_host(tmp_path, monkeypatch, 12, "max", 128 << 30)
    assert auto.workers == 12
    capped = _auto_host(tmp_path, monkeypatch, 12, "max", 128 << 30, **{"resources.cap": 0.75})
    assert capped.workers == 9


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
        configuration.validate("project", {"schema": 3, "name": "g"}, "config.toml")
    (finding,) = caught.value.findings
    assert (finding.key, finding.path) == ("config.schema", "config.toml:schema") and "schema 4" in finding.action


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


@pytest.mark.parametrize("tool", ["cpp", "mips_as", "mips_ld", "mips_objcopy", "mips_objdump", "n64link", "armips"])
def test_optional_host_tool_absent_and_bad_path(tmp_path: Path, tool: str) -> None:
    path = fixture.host(tmp_path)
    path.write_text("\n".join(line for line in path.read_text().splitlines() if not line.startswith(tool + " = ")))
    assert tool not in configuration.load_host(path).tools
    invalid = fixture.host(tmp_path, **{f"tools.{tool}": str(tmp_path / "absent")})
    refusal = _refusal(configuration.load_host, invalid)
    assert refusal.findings[0].key == "config.file"
    assert refusal.findings[0].path == f"host:tools.{tool}"


@pytest.mark.parametrize(
    "cpus,cap,load,workers",
    [
        (12, None, 0.0, 12),
        (12, None, 22.5, 12),
        (12, None, 3.9, 12),
        (12, 0.5, 2.0, 6),
        (12, 0.75, 0.4, 9),
        (4, 0.1, 0.0, 1),
    ],
)
def test_host_auto_workers_keep_capacity_under_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpus: int, cap: float | None, load: float, workers: int
) -> None:
    monkeypatch.setattr(os, "getloadavg", lambda: (load, load, load))
    extra = {"resources.cap": cap} if cap else {}
    loaded = _auto_host(tmp_path, monkeypatch, cpus, "max", 128 << 30, **extra)
    assert (loaded.cores, loaded.workers) == (cpus, workers)


@pytest.mark.parametrize(
    ("path", "source", "header", "stem"),
    [
        ("src/a.c", True, False, "src/a"),
        ("src/AI/ENEMIES/TURRETAI.CPP", True, False, "src/AI/ENEMIES/TURRETAI"),
        ("src/X.C", True, False, "src/X"),
        ("include/AI/ENEMIES.H", False, True, "include/AI/ENEMIES.H"),
        ("asm/a.s", False, False, "asm/a.s"),
        ("src/sdk/a.s", True, False, "src/sdk/a"),
    ],
)
def test_sources_and_headers_are_known_by_suffix_in_any_case(path: str, source: bool, header: bool, stem: str) -> None:
    found = (configuration.is_source(path), configuration.is_header(path), configuration.source_stem(path))
    assert found == (source, header, stem) and configuration.is_text(path) == (source or header)


# ---------------------------------------------------------------- project config, schema 4
HASH_A, HASH_B = "a" * 40, "b" * 40


def _rom(code: bytes = b"NRWE", revision: int = 0) -> bytes:
    return bytes(0x3B) + code + bytes([revision]) + bytes(0x40)


SIZE = len(_rom())
RW = f"""schema = 4
name = "RageWars"
title = "Turok: Rage Wars"
toolchain = "gcc-2.8.1-sn64"
names_from = "us-rev1"
libraries = ["src/libgcc/", "src/sdk/libgultra/"]
cppflags = ["-DF3DEX_GBI_2"]
readme_map = "us-rev1"
readme_map_link = "https://example.test/map/"

[version.us]
sha1 = "{HASH_A}"
size = {SIZE}
cartridge_id = "NUS-NRWE-0"
region = "North America"
description = "First release."
segments = [
  {{ name = "main", rom = 0x1000, vram = 0x80200400 }},
  {{ name = "assets", rom = 0x1B5A50, kind = "bin" }},
]

[version.us-rev1]
sha1 = "{HASH_B}"
size = {SIZE}

[file."src/IO/80200500.CPP"]
toolchain = "gcc-2.7.2-kmc"
cflags = ["-O1"]

[file."src/IO/"]
omit = ["-O2"]
"""
BT = f"""schema = 4
name = "BattleTanx"
toolchain = "gcc-2.7.2-kmc"

[version.us]
sha1 = "{HASH_A}"
size = {SIZE}
"""
TRACKED = ["src/IO/80200500.CPP", "src/IO/80200600.CPP", "src/main.c"]


def _project(tmp_path: Path, text: str, tracked: list[str] = TRACKED, roms: dict[str, bytes] | None = None) -> Any:
    (tmp_path / "config.toml").write_text(text)
    store = roms if roms is not None else {"us": _rom(), "us-rev1": _rom(b"NRWE", 1)}
    for name, raw in store.items():
        (tmp_path / "roms").mkdir(exist_ok=True)
        (tmp_path / "roms" / f"baserom.{name}.z64").write_bytes(raw)

    def reader(path: str) -> bytes:
        if path == "config.toml":
            return text.encode()
        name = path.removeprefix("roms/baserom.").removesuffix(".z64")
        if name not in store:
            raise FileNotFoundError(path)
        return store[name]

    return configuration.load_project(tmp_path, reader, lambda: tracked)


def _with_hashes(text: str, roms: dict[str, bytes]) -> tuple[str, dict[str, bytes]]:
    import hashlib

    for name, placeholder in (("us", HASH_A), ("us-rev1", HASH_B)):
        if name in roms:
            text = text.replace(placeholder, hashlib.sha1(roms[name]).hexdigest())
    return text, roms


def _load(tmp_path: Path, text: str, tracked: list[str] = TRACKED) -> Any:
    roms = {"us": _rom(), "us-rev1": _rom(b"NRWE", 1)}
    text, roms = _with_hashes(text, roms)
    return _project(tmp_path, text, tracked, roms)


def test_rw_and_bt_examples_validate(tmp_path: Path) -> None:
    rw = _load(tmp_path, RW)
    assert rw.versions == ("us", "us-rev1") and rw.libraries == ("src/libgcc/", "src/sdk/libgultra/")
    assert rw.version["us"].segments[1] == contracts.Segment("assets", 0x1B5A50, None, "bin")
    other = tmp_path / "bt"
    other.mkdir()
    bt = _load(other, BT, [])
    assert (bt.name, bt.title, bt.names_from, bt.versions) == ("BattleTanx", "BattleTanx", "us", ("us",))


@pytest.mark.parametrize(
    "extra",
    [
        'id = "x"',
        'versions = ["us"]',
        'baserom = "x"',
        'split = "x"',
        'symbols = "x"',
        'macros = ["X"]',
        "resident = {}",
        "version_differences = []",
        "evidence = {}",
        "layout = {}",
        "naming = {}",
        "text_mask = 1",
        "gnu_asflags = []",
        'library_dirs = ["src/x/"]',
        "[project]\nname = 'x'",
        "[build]\ncflags = []",
        f'[version.zz]\nsha1 = "{HASH_A}"\nbaserom = "x"',
        f'[version.zz]\nsha1 = "{HASH_A}"\nmacros = []',
    ],
)
def test_every_removed_key_is_refused(tmp_path: Path, extra: str) -> None:
    head, _, rest = BT.partition("\n[version.us]")
    text = BT + "\n" + extra + "\n" if extra.startswith("[") else head + "\n" + extra + "\n\n[version.us]" + rest
    refusal = _refusal(_load, tmp_path, text)
    assert _keys(refusal) == {"config.schema"}


def test_the_old_schema_is_one_finding(tmp_path: Path) -> None:
    refusal = _refusal(_load, tmp_path, BT.replace("schema = 4", "schema = 3"))
    assert [f.path for f in refusal.findings] == ["project:schema"]


def test_defaults_are_derived(tmp_path: Path) -> None:
    project = _load(tmp_path, BT, [])
    assert project.toolchain == "gcc-2.7.2-kmc" and project.title == "BattleTanx"
    assert (project.cppflags, project.cflags, project.asflags, project.libraries) == ((), (), (), ())
    assert project.readme_map is None and project.readme_map_link is None
    facts = project.version["us"]
    assert facts.segments == (contracts.Segment("main", 0x1000, None, "code"),)
    assert facts.meta == {} and configuration.rom_meta(project, "us") == {
        "cartridge_id": "NUS-NRWE-0",
        "region": "North America",
    }
    assert contracts.macro("us-rev1") == "VERSION_US_REV1"


def test_names_from_defaults_to_the_first_sorted_version(tmp_path: Path) -> None:
    text = RW.replace('names_from = "us-rev1"\n', "")
    assert _load(tmp_path, text).names_from == "us"


def test_header_gives_cartridge_and_region(tmp_path: Path) -> None:
    project = _load(tmp_path, RW)
    assert configuration.rom_meta(project, "us-rev1") == {"cartridge_id": "NUS-NRWE-1", "region": "North America"}
    assert configuration.rom_meta(project, "us")["description"] == "First release."
    roms = {"us": _rom(b"NPDP", 2)}
    other = tmp_path / "pal"
    other.mkdir()
    text, roms = _with_hashes(BT, roms)
    pal = _project(other, text, [], roms)
    assert configuration.rom_meta(pal, "us") == {"cartridge_id": "NUS-NPDP-2", "region": "Europe"}


def test_file_precedence(tmp_path: Path) -> None:
    project = _load(tmp_path, RW)
    exact = contracts.file_facts(project, "src/IO/80200500.CPP")
    assert (exact.toolchain, exact.cflags, exact.omit) == ("gcc-2.7.2-kmc", ("-O1",), ())
    prefix = contracts.file_facts(project, "src/IO/80200600.CPP")
    assert (prefix.toolchain, prefix.cflags, prefix.omit) == (None, (), ("-O2",))
    assert contracts.file_facts(project, "src/main.c") == contracts.FileFacts()


def test_a_longer_directory_prefix_wins(tmp_path: Path) -> None:
    text = RW + '\n[file."src/"]\ncflags = ["-g"]\n'
    project = _load(tmp_path, text)
    assert contracts.file_facts(project, "src/IO/80200600.CPP").omit == ("-O2",)
    assert contracts.file_facts(project, "src/main.c").cflags == ("-g",)


def test_an_unknown_file_key_is_refused(tmp_path: Path) -> None:
    refusal = _refusal(_load, tmp_path, RW, ["src/main.c"])
    assert _keys(refusal) == {"config.file_unknown"}
    assert {f.path for f in refusal.findings} == {"project:file.src/IO/80200500.CPP", "project:file.src/IO/"}


def test_sha1_mismatch_refuses_where_the_rom_bytes_are_used(tmp_path: Path) -> None:
    project = _project(tmp_path, BT, [], {"us": _rom()})  # loading reads no ROM
    refusal = _refusal(configuration.rom, project, "us")
    assert _keys(refusal) == {"config.file"} and refusal.findings[0].path == "project:version.us.sha1"


def test_a_rom_of_another_size_refuses_before_it_is_hashed(tmp_path: Path) -> None:
    project = _project(tmp_path, BT, [], {"us": _rom()})
    (tmp_path / "roms/baserom.us.z64").write_bytes(_rom() + b"\0")
    refusal = _refusal(configuration.rom, project, "us")
    assert refusal.findings[0].path == "project:version.us.size"


def test_a_version_without_a_size_is_refused(tmp_path: Path) -> None:
    refusal = _refusal(_project, tmp_path, BT.replace(f"size = {SIZE}\n", ""), [])
    assert _keys(refusal) == {"config.schema"}


def test_a_missing_rom_loads_and_refuses_by_name_where_its_bytes_are_used(tmp_path: Path) -> None:
    project = _project(tmp_path, BT, [], {})
    refusal = _refusal(configuration.rom, project, "us")
    assert "roms/baserom.us.z64 does not exist" in refusal.findings[0].reason


def test_load_resource_carries_platform_and_family_rows() -> None:
    rom = configuration.load_resource("rom.toml")
    assert rom["asflags"] == ["-march=vr4300", "-mabi=32", "-EB", "--no-pad-sections", "-mips3"]
    assert rom["text_mask"] == 0x1FFFFFFF
    rows = configuration.toolchains()
    assert {id: r["link_sections"] for id, r in rows.items()} == {
        "gcc-2.8.1-sn64": "per-section",  # its C units' rodata sits in data segments of its own (M-RW)
        "gcc-2.7.2-kmc": "per-object",  # a function's rodata follows its code (M-BT)
        "ido-5.3": "per-section",
        "ido-7.1": "per-section",
    }
    assert rows["gcc-2.7.2-kmc"]["cppflags"] == [
        "-P",
        "-undef",
        "-nostdinc",
        "-D_LANGUAGE_C",
        "-D__GNUC__=2",
        "-D__UNBAKE_STDARG_GCC",
    ]
    assert {r["include_asm"] for id, r in rows.items() if id.startswith("ido")} == {"processor"}
    assert {(r["stand_in_as"], r["data_split"]) for r in rows.values()} == {
        ("{as}", "subsegment")
    }  # splat writes per-symbol data only from subsegments


@pytest.mark.parametrize("affinity,cores,workers", [(12, 4, 4), (4, 12, 4)])
def test_host_auto_workers_respect_both_affinity_and_configured_cores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, affinity: int, cores: int, workers: int
) -> None:
    loaded = _auto_host(tmp_path, monkeypatch, affinity, "max", 128 << 30, **{"resources.cores": cores})
    assert loaded.workers == workers
