"""The fixture builds a project that unbake.config.load accepts and an ELF pyelftools reads."""

from __future__ import annotations

import io
from pathlib import Path

import fixture
from elftools.elf.elffile import ELFFile

from unbake import config as configuration


def test_project_loads(tmp_path: Path, toolchains: dict) -> None:
    code = bytes.fromhex("03e0000800000000")
    functions = {"f": {"a": (0x1000, code), "b": (0x1100, code)}}
    root = fixture.project(tmp_path, functions=functions, data={"d": {"a": (0x1200, b"\1\2\3\4")}})
    loaded = configuration.load(root, fixture.host(tmp_path))
    assert loaded.project.versions == ("a", "b")
    assert (root / "layout.toml").is_file() and (root / ".git").is_dir()


def test_elf_reports_undefined(tmp_path: Path) -> None:
    elf = ELFFile(io.BytesIO(fixture.elf_with_symbols(["f"], ["g", "h"])))
    names = {s.name: s["st_shndx"] for s in elf.get_section_by_name(".symtab").iter_symbols() if s.name}
    assert [n for n, shndx in names.items() if shndx == "SHN_UNDEF"] == ["g", "h"]
    assert names["f"] != "SHN_UNDEF"
