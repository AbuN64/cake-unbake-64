"""A fact `textpiece<n>_<function>`: a function the ROM holds in several runs (a Psy-Q LNK stream puts a file's strings
and statics between its code chunks) is one `.text.<function>` in the object, cut where each run ends and laid at the
ROM address of each run."""

import subprocess
import sys
from pathlib import Path

import pytest
from test_build import both, make_config, make_tree

from unbake import adapters, build, tree
from unbake.contracts import Refusal

# A real slice shape: three words, a branch over the cut, three more words (a `bne` whose target lies in the next run).
SOURCE = """	.section .text.F,"ax",@progbits
F:
	.set noreorder
	addiu	$29,$29,-8
	bne	$4,$0,.Lend
	nop
	addiu	$2,$0,1
	addiu	$2,$2,2
.Lend:
	jr	$31
	addiu	$29,$29,8
"""
BINUTILS = Path("/usr/bin/mips-linux-gnu-as").exists() and Path("/usr/bin/mips-linux-gnu-ld").exists()


def test_a_textpiece_name_is_read_as_its_run_number_and_function():
    assert tree.textpiece("textpiece2_PAKInterface_Open") == (2, "PAKInterface_Open")
    assert tree.textpiece("textpiece0_f") is None
    assert tree.textpiece("literals_f") is None
    assert tree.textpiece("textpieces_f") is None


def _held():
    held = make_tree(both("src/a.c", "f_1", "textpiece1_f_1", "textpiece2_f_1"), both("src/b.c", "d_5"))
    table = {
        "f_1": (0x80200010, 0x1010, 0x20, "func"),
        "textpiece1_f_1": (0x80200030, 0x1100, 0x30, "func"),
        "textpiece2_f_1": (0x80200060, 0x1200, 0x10, "func"),
        "d_5": (0x80200100, 0x1050, 0x10, "data"),
    }
    held.symbols["us"] = held.symbols["eu-1"] = table
    for m in held.modules:
        m.sections["us"]["textpiece1_f_1"] = m.sections["us"]["textpiece2_f_1"] = ".text"
    return held


def test_the_cuts_chain_the_function_through_its_runs_in_order():
    assert build._text_cuts(_held(), "us") == {"src/a.c": "f_1:32:textpiece1_f_1 textpiece1_f_1:48:textpiece2_f_1"}


def test_a_run_without_a_size_or_a_missing_number_is_refused():
    held = _held()
    held.symbols["us"]["textpiece1_f_1"] = (0x80200030, 0x1100, 0, "func")
    with pytest.raises(Refusal, match="no size"):
        build._text_cuts(held, "us")
    held = _held()
    held.modules[0].owns["us"] = ("f_1", "textpiece2_f_1")
    with pytest.raises(Refusal, match=r"not 1\.\.1"):
        build._text_cuts(held, "us")


def test_the_file_is_built_apart_cut_by_the_makefile_and_each_run_laid_at_its_rom_offset(tmp_path):
    config, held = make_config(tmp_path), _held()
    assert build._apart_paths(config, held, "us") == {"src/a.c"}
    rows = build.text_sizes(config, held, "us").decode()
    assert "build/us/src/a.c.o: FSPLIT := fsplit" in rows
    assert "build/us/src/a.c.o: TEXTCUT := f_1:32:textpiece1_f_1 textpiece1_f_1:48:textpiece2_f_1" in rows
    assert "TEXTSIZE" not in rows.split("src/a.c.o: TEXTCUT")[0].splitlines()[-1]  # code in three runs: no trim
    recipe = build._recipe("gcc-2.8.1-sn64").splitlines()[-1]
    assert "$(if $(TEXTCUT),'&&' $(PASS)textcut_pass.py $@ $(TEXTCUT))" in recipe
    body = [
        ln.strip()
        for ln in build.linker(config, held).decode().splitlines()
        if ln.strip().startswith(("src/", ". = ")) and ".bss" not in ln
    ]
    assert body[:7] == [
        ". = MAX(., 0x10);",
        "src/a.c.o(.text.f_1);",
        ". = MAX(., 0x50);",
        "src/b.c.o(.data);",
        ". = MAX(., 0x100);",
        "src/a.c.o(.text.textpiece1_f_1);",
        ". = MAX(., 0x200);",
    ]


def test_a_textpiece_row_of_a_function_no_file_owns_is_a_finding(tmp_path):
    held = make_tree(both("src/a.c", "f_1"))
    held.symbols["us"]["textpiece1_f_9"] = (0x80200010, 0x1100, 0x10, "func")
    owned = {"src/a.c": {"us": ["f_1"]}}
    sections = {"src/a.c": {"us": {"f_1": ".text"}}}
    findings = tree._textpieces(held.symbols, owned, sections)
    assert [f.unit for f in findings] == ["textpiece1_f_9"]
    held.symbols["us"]["textpiece1_f_1"] = (0x80200010, 0x1100, 0x10, "func")
    del held.symbols["us"]["textpiece1_f_9"]
    assert tree._textpieces(held.symbols, owned, sections) == []
    assert owned["src/a.c"]["us"] == ["f_1", "textpiece1_f_1"]
    assert sections["src/a.c"]["us"]["textpiece1_f_1"] == ".text"


@pytest.mark.skipif(not BINUTILS, reason="needs the MIPS binutils")
def test_the_pass_cuts_one_text_section_in_two_and_the_link_lays_the_runs_apart(tmp_path):
    source, obj = tmp_path / "s.s", tmp_path / "s.o"
    source.write_text(SOURCE)
    subprocess.run(["mips-linux-gnu-as", "-march=vr4300", "-EB", "-mips3", str(source), "-o", str(obj)], check=True)
    subprocess.run([sys.executable, "-I", adapters.make_tool("unreloc_pass.py"), str(obj)], check=True)
    run = lambda *a: subprocess.run([sys.executable, "-I", adapters.make_tool("textcut_pass.py"), str(obj), *a])  # noqa: E731
    assert run("F:12:G").returncode == 0
    sections = subprocess.run(["mips-linux-gnu-readelf", "-SW", str(obj)], capture_output=True, text=True).stdout
    assert ".text.F" in sections and ".text.G" in sections
    script = tmp_path / "l.ld"
    script.write_text(
        "SECTIONS { . = 0x80000000; .main : SUBALIGN(1) { s.o(.text.F); . = MAX(., 0x20); s.o(.text.G); } }\n"
    )
    linked = tmp_path / "s.elf"
    link = ["mips-linux-gnu-ld", "-T", str(script), str(obj), "-o", str(linked), "-e", "0"]
    subprocess.run(link, cwd=tmp_path, check=True)
    flat = tmp_path / "s.bin"
    subprocess.run(["mips-linux-gnu-objcopy", "-O", "binary", str(linked), str(flat)], check=True)
    bytes_ = flat.read_bytes()
    words = [int.from_bytes(bytes_[i : i + 4], "big") for i in range(0, len(bytes_), 4)]
    assert words[:3] == [0x27BDFFF8, 0x14800003, 0]  # the first run: the branch offset was fixed before the cut
    assert words[8:12] == [0x24020001, 0x24420002, 0x03E00008, 0x27BD0008]  # the rest, 0x20 after the start
    symbols = subprocess.run(["mips-linux-gnu-nm", str(linked)], capture_output=True, text=True).stdout
    assert " F" in symbols


@pytest.mark.skipif(not BINUTILS, reason="needs the MIPS binutils")
def test_the_pass_refuses_a_cut_that_does_not_lie_inside_the_section(tmp_path):
    source, obj = tmp_path / "s.s", tmp_path / "s.o"
    source.write_text(SOURCE)
    subprocess.run(["mips-linux-gnu-as", "-march=vr4300", "-EB", "-mips3", str(source), "-o", str(obj)], check=True)
    subprocess.run([sys.executable, "-I", adapters.make_tool("unreloc_pass.py"), str(obj)], check=True)
    for cut in ("F:0:G", "F:99:G", "H:4:G"):
        argv = [sys.executable, "-I", adapters.make_tool("textcut_pass.py"), str(obj), cut]
        done = subprocess.run(argv, capture_output=True)
        assert done.returncode != 0
