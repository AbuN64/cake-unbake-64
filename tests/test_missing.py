"""missing: the names the assembly uses that no symbol fact defines, classified by provenance and written."""

from __future__ import annotations

from pathlib import Path

import fixture
import pytest

from unbake import missing, pool, symbols


@pytest.fixture(autouse=True)
def _inline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pool, "_INLINE_AT", 10**6)


def row(word: str, op: str, at: int) -> str:
    return f"/* {at:X} {0x80000400 + at:08X} {word} */  {op}\n"


CODE = "".join(
    [
        row("3C048010", "lui        $a0, %hi(D_auto_80000500)", 0),  # a: data in main (rom 0x1100)
        row("8C840010", "lw         $a0, %lo(D_auto_80000500)($a0)", 4),
        row("0C040004", "jal        func_auto_80000410", 8),  # a: code in main
        row("3C04002B", "lui        $a0, %hi(D_auto_002A81B0)", 12),  # b: a ROM offset constant
        row("3C04A404", "lui        $a0, %hi(D_auto_A4040010)", 16),  # a: hardware register
        row("3C04B000", "lui        $a0, %hi(D_auto_80900000)", 20),  # d: past the end of RAM
        row("3C048040", "lui        $a0, %hi(D_auto_80700010)", 20),  # a: bss, in no segment
        row("0C1C0000", "jal        func_auto_80700000", 20),  # d: code in no segment
        row("3C048000", "lui        $a0, %hi(D_auto_80000400)", 24),  # c: caller names it
    ]
)


def project(tmp: Path) -> symbols.Config:
    table = {v: "caller = 0x80000400; // type:func rom:0x1000 size:0x20\n" for v in ("a", "b")}
    extra = {f"asm/{v}/nonmatchings/main/caller.s": CODE for v in ("a", "b")}
    extra["src/main/uses.c"] = "extern int D_auto_002A81B0;\n"  # the C spells the splat name
    return fixture.config(tmp, symbols=table, extra=extra)


def test_each_undefined_name_gets_the_class_of_its_provenance(tmp_path: Path) -> None:
    config = project(tmp_path)
    found = {r["name"]: r for r in missing.report(config, ["a"])["versions"]["a"]["rows"]}
    assert found["D_auto_80000500"]["class"] == "a" and found["D_auto_80000500"]["kind"] == "data"
    assert found["D_auto_80000500"]["rom"] == 0x1100 and found["D_auto_80000500"]["proposed"] == "D_80000500_a"
    assert found["func_auto_80000410"]["kind"] == "func" and found["func_auto_80000410"]["rom"] == 0x1010
    assert found["D_auto_002A81B0"]["class"] == "b" and found["D_auto_002A81B0"]["proposed"] == "D_002A81B0_a"
    assert found["D_auto_A4040010"]["class"] == "a" and found["D_auto_A4040010"]["proposed"] == "D_A4040010"
    assert found["D_auto_80900000"]["class"] == "d" and found["D_auto_80900000"]["at"][0].endswith("caller.s:6")
    assert found["D_auto_80700010"]["class"] == "a" and found["D_auto_80700010"]["rom"] is None
    assert found["func_auto_80700000"]["class"] == "d"
    assert found["D_auto_80000400"]["class"] == "c" and found["D_auto_80000400"]["symbol"] == "caller"


def test_write_adds_the_placeable_facts_once_and_leaves_the_rest(tmp_path: Path) -> None:
    config = project(tmp_path)
    written = missing.write(config, missing.report(config, ["a"]))
    text = (config.project.root / "versions/a/symbol_addrs.txt").read_text()
    assert "func_80000410_a = 0x80000410; // type:func rom:0x1010\n" in text
    assert "D_80000500_a = 0x80000500; // rom:0x1100 size:0x4\n" in text
    assert "D_002A81B0_a = 0x002A81B0;" in text and "A4040010" not in text
    assert "D_80700010_a = 0x80700010; // size:0x1\n" not in text and "D_80700010_a = 0x80700010;" in text
    assert "80900000" not in text and len(written["a"]) == 4
    assert missing.write(config, missing.report(config, ["a"])) == {"a": []}  # the names are defined now
    aliases = (config.project.root / "versions/a/aliases.txt").read_text()
    assert aliases == "D_auto_002A81B0 = 0x002A81B0; //\n"  # kept: one spelling of the C links in every version
