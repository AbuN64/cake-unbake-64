"""A file row's `unrelocated`: the unit is held as the ROM holds a relocatable object, before its link."""

import subprocess
import sys
from pathlib import Path

import pytest
import test_build

from unbake import adapters, build
from unbake.contracts import FileFacts

# A real slice: the 172-byte CountActive-shaped leaf the US ROM holds at 0xFD5F0 + 0x1D34 (a Psy-Q LNK code chunk).
# SN gcc 2.8.1 -O0 wrote this assembly from `g_hold->slots[i].active`; the ROM keeps its jumps and its address
# halves zero (`j 0`, `lui v0,0`) and its delay slots hold nops.
SLICE = """gcc2_compiled.:
	.text
	.align	2
	.globl	CountActive
	.type	 CountActive,@function
	.ent	CountActive
CountActive:
	.frame	$fp,16,$31		# vars= 8, regs= 1/0, args= 0, extra= 0
	.mask	0x40000000,-8
	.fmask	0x00000000,0
	subu	$sp,$sp,16
	sw	$fp,8($sp)
	move	$fp,$sp
	sw	$0,4($fp)
	sw	$0,0($fp)
.L2:
	lw	$2,0($fp)
	slt	$3,$2,4
	bne	$3,$0,.L5
	j	.L3
.L5:
	lw	$2,g_hold
	lw	$3,0($fp)
	move	$4,$3
	sll	$5,$4,3
	addu	$5,$5,$3
	sll	$4,$5,3
	addu	$4,$4,$3
	sll	$3,$4,2
	addu	$4,$4,$3
	sll	$3,$4,3
	addu	$2,$2,$3
	lw	$3,88($2)
	beq	$3,$0,.L4
	lw	$2,4($fp)
	addu	$3,$2,1
	sw	$3,4($fp)
.L6:
.L4:
	lw	$2,0($fp)
	addu	$3,$2,1
	sw	$3,0($fp)
	j	.L2
.L3:
	lw	$3,4($fp)
	move	$2,$3
	j	.L1
.L1:
	move	$sp,$fp
	lw	$fp,8($sp)
	addu	$sp,$sp,16
	j	$31
	.end	CountActive
.Lfe1:
	.size	 CountActive,.Lfe1-CountActive
"""
ROM = bytes.fromhex(
    "27bdfff0afbe000803a0f021afc00004afc000008fc2000028430004146000030000000008000000000000003c0200008c4200008fc3000000602021000428c000a32821000520c0008320210004188000832021000418c0004310218c43005810600004000000008fc2000424430001afc300048fc2000024430001afc3000008000000000000008fc3000400601021080000000000000003c0e8218fbe000827bd001003e0000800000000"
)

AS = ("mips-linux-gnu-as", "-march=vr4300", "-mabi=32", "-EB", "-mips3")


def _asn64_moves(text: bytes) -> bytes:
    """The build's assembler front writes `move` as addu where plain gas writes or (ROM 0x03A0F021, gas 0x03A0F025)."""
    words = [int.from_bytes(text[i : i + 4], "big") for i in range(0, len(text), 4)]
    fixed = [w & ~0x3F | 0x21 if w >> 26 == 0 and w & 0x3F == 0x25 and w >> 16 & 31 == 0 else w for w in words]
    return b"".join(w.to_bytes(4, "big") for w in fixed)


def _object(tmp_path: Path, *flags: str) -> Path:
    source, obj = tmp_path / "s.s", tmp_path / "s.o"
    source.write_text(SLICE)
    subprocess.run([*AS, *flags, str(source), "-o", str(obj)], check=True)
    return obj


def _text(obj: Path) -> bytes:
    out = obj.with_suffix(".bin")
    subprocess.run(["mips-linux-gnu-objcopy", "-O", "binary", "-j", ".text", str(obj), str(out)], check=True)
    return out.read_bytes()


def _unrelocate(obj: Path) -> None:
    subprocess.run([sys.executable, "-I", "-c", adapters.unreloc_script(), str(obj)], check=True)


@pytest.mark.skipif(not Path("/usr/bin/mips-linux-gnu-as").exists(), reason="needs the MIPS binutils")
def test_the_assembled_slice_is_the_rom_bytes_once_unrelocated(tmp_path):
    obj = _object(tmp_path, "-O0")
    assert _asn64_moves(_text(obj))[: len(ROM)] != ROM  # the jumps carry their offsets until the pass
    _unrelocate(obj)
    assert _asn64_moves(_text(obj))[: len(ROM)] == ROM
    relocations = subprocess.run(["mips-linux-gnu-readelf", "-r", str(obj)], capture_output=True, text=True).stdout
    assert "R_MIPS" not in relocations


def test_a_file_row_unrelocated_sets_the_makefile_variable_and_runs_the_pass(tmp_path):
    config = test_build.make_config(tmp_path, {"src/a.c": FileFacts(unrelocated=True)})
    makefile = build.makefile(config).decode()
    assert "/src/a.c.o): UNRELOC := 1" in makefile
    recipe = build._recipe("gcc-2.8.1-sn64").splitlines()[-1]
    assert "$(if $(UNRELOC),'&&' @py UNRELOC_PASS $@)" in recipe
    assert "define UNRELOC_PASS\n" in makefile and "export UNRELOC_PASS\n" in makefile
