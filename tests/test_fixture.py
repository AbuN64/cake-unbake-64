"""The fixture builds a schema-4 project a config can describe, a fake cpp that runs, and an ELF pyelftools reads."""

from __future__ import annotations

import io
import subprocess
import tomllib
from pathlib import Path

import fixture
from elftools.elf.elffile import ELFFile

from unbake import symbols


def test_project_has_two_versions_two_segments_three_modules(tmp_path: Path) -> None:
    root = fixture.project(tmp_path)
    document = tomllib.loads((root / "config.toml").read_text())
    assert document["schema"] == 4 and sorted(document["version"]) == ["a", "b"]
    assert [s["name"] for s in document["version"]["a"]["segments"]] == ["main", "ovl"]
    assert sorted(p.relative_to(root).as_posix() for p in (root / "src").rglob("*.c")) == [
        "src/main/c_func.c",
        "src/main/stand_in.c",
        "src/ovl/fuzzy.c",
    ]
    assert (root / ".git").is_dir() and not (root / "layout.toml").exists()
    table = symbols.parse((root / "versions/b/symbol_addrs.txt").read_text())
    assert table["fuzzy"] == (0x80100010, 0x3010, 0x10, "func")


def test_config_describes_the_project(tmp_path: Path, toolchains: dict) -> None:
    config = fixture.config(tmp_path)
    assert config.project.versions == ("a", "b") and config.host.tools["cpp"].is_file()
    assert config.project.version["a"].segments[1].rom == 0x3000


def test_fake_cpp_keeps_the_live_branch_and_expands_stand_ins(tmp_path: Path) -> None:
    config = fixture.config(tmp_path)
    source = (config.project.root / "src/ovl/fuzzy.c").read_bytes()
    run = lambda *flags: subprocess.run(  # noqa: E731
        [str(config.host.tools["cpp"]), *flags, "-"], input=source, capture_output=True, check=True
    ).stdout.decode()
    plain, fuzzy = run(), run("-DNON_MATCHING")
    assert "nonmatchings/ovl/fuzzy.s" in plain and "return a + 1" not in plain
    assert "return a + 1" in fuzzy and fixture.cpp_runs(tmp_path) == 2


def test_elf_reports_undefined(tmp_path: Path) -> None:
    elf = ELFFile(io.BytesIO(fixture.elf_with_symbols(["f"], ["g", "h"])))
    names = {s.name: s["st_shndx"] for s in elf.get_section_by_name(".symtab").iter_symbols() if s.name}
    assert [n for n, shndx in names.items() if shndx == "SHN_UNDEF"] == ["g", "h"]
    assert names["f"] != "SHN_UNDEF"
