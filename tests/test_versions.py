"""versions: rows, placements, ROM bytes, symbols, undefined names."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import fixture
import pytest

from unbake import config as configuration
from unbake import versions
from unbake.contracts import Config, Placement, Refusal

CODE = bytes.fromhex("03e0000800000000")


@pytest.fixture
def world(tmp_path: Path, toolchains: dict) -> tuple[Config, dict, list[str]]:
    """Two versions: f in both, g only in b, data d only in a."""
    functions = {"f": {"a": (0x1000, CODE), "b": (0x1000, CODE)}, "g": {"b": (0x1100, CODE)}}
    root = fixture.project(tmp_path, functions=functions, data={"d": {"a": (0x1200, b"\1\2\3\4")}})
    cfg = configuration.load(root, fixture.host(tmp_path))
    reads: list[str] = []

    def reader(path: str) -> bytes:
        reads.append(path)
        return (root / path).read_bytes()

    return cfg, {"reader": reader}, reads


def _version(world, vid: str):
    cfg, ctx, _ = world
    return versions.read(cfg, ctx["reader"])[vid]


def test_rows_and_placements(world) -> None:
    version, reader = _version(world, "a"), world[1]["reader"]
    got = {name: (state, p) for name, state, p in versions.rows(version, reader)}
    assert list(got) == ["f", "d"]
    state, placement = got["f"]
    assert state == "asm"
    assert placement == Placement("a", ".text", 0x1000, 0x1200, fixture.VRAM_BASE, 0)
    state, placement = got["d"]
    assert (state, placement.section) == ("data", ".data")
    assert (placement.rom_start, placement.vram) == (0x1200, fixture.VRAM_BASE + 0x200)
    assert placement.rom_end == len(version.rom.read_bytes())


def test_holders_follow_yaml(world) -> None:
    reader = world[1]["reader"]
    a = {n for n, _, _ in versions.rows(_version(world, "a"), reader)}
    b = {n: p for n, _, p in versions.rows(_version(world, "b"), reader)}
    assert "g" not in a and "g" in b
    assert {p.version for p in b.values()} == {"b"}
    assert b["g"].rom_start == 0x1100


def test_config_toml_never_read(world) -> None:
    cfg, ctx, reads = world
    versions.read(cfg, ctx["reader"])
    assert "config.toml" not in reads
    assert set(reads) == {
        "versions/a/Game.yaml",
        "versions/b/Game.yaml",
        "symbols.toml",
        "versions/a/symbol_addrs.txt",
        "versions/b/symbol_addrs.txt",
    }  # the key reads them, once


def test_read_facts(world) -> None:
    version = _version(world, "a")
    assert version.rom_sha256 and version.split == "versions/a/Game.yaml"
    assert version.symbols["f"] == fixture.VRAM_BASE
    assert version.segments[0][:2] == ("main", 0x1000) and version.segments[0][3] == fixture.VRAM_BASE


def _yaml_version(tmp_path: Path, segment: dict, symbols: str = "") -> tuple:
    rom = tmp_path / "rom.z64"
    rom.write_bytes(bytes(0x2000))
    files = {"s.yaml": json.dumps({"segments": [segment]}).encode(), "sym.txt": symbols.encode()}
    from unbake.contracts import Version

    seg = versions._segment_rows(files["s.yaml"])
    return Version("x", rom, "0", "s.yaml", "sym.txt", {}, seg), files["s.yaml"].__class__ and (lambda p: files[p])


def test_bss_has_no_rom_extent(tmp_path: Path, toolchains: dict) -> None:
    segment = {
        "name": "m",
        "type": "code",
        "start": 0x1000,
        "vram": 0x80000000,
        "subsegments": [[0x1000, "asm", "t"], [0x1100, "bss", "b"], [0x2000]],
    }
    segment["subsegments"][1] = {"start": 0x1100, "type": "bss", "name": "b", "vram": 0x80001000, "size": 0x40}
    version, reader = _yaml_version(tmp_path, segment)
    got = {n: p for n, _, p in versions.rows(version, reader)}
    assert got["b"].section == ".bss"
    assert got["b"].rom_start == got["b"].rom_end == 0x1100
    assert got["b"].size == 0x40 and got["b"].vram == 0x80001000


@pytest.mark.parametrize(
    ("kind", "section"),
    [
        ("asm", ".text"),
        ("c", ".text"),
        ("hasm", ".text"),
        ("rodata", ".rodata"),
        (".rodata", ".rodata"),
        ("data", ".data"),
        (".data", ".data"),
        ("bin", ".data"),
    ],
)
def test_section_mapping(tmp_path: Path, toolchains: dict, kind: str, section: str) -> None:
    segment = {
        "name": "m",
        "type": "code",
        "start": 0x1000,
        "vram": 0x80000000,
        "subsegments": [[0x1000, kind, "n"], [0x1100]],
    }
    version, reader = _yaml_version(tmp_path, segment)
    assert versions.rows(version, reader)[0][2].section == section


@pytest.mark.parametrize("sub", [[0x1000, "asm"], [0x1000, "weird", "n"]])
def test_rows_refuse_unnamed_or_unknown(tmp_path: Path, toolchains: dict, sub: list) -> None:
    segment = {"name": "m", "type": "code", "start": 0x1000, "vram": 0, "subsegments": [sub, [0x1100]]}
    version, reader = _yaml_version(tmp_path, segment)
    with pytest.raises(Refusal) as error:
        versions.rows(version, reader)
    assert error.value.findings[0].key == "version.split"


def test_rom_bytes(world) -> None:
    version = _version(world, "a")
    assert versions.rom_bytes(version, 0x1000, 0x1008) == CODE
    size = len(version.rom.read_bytes())
    for start, end in ((0, size + 1), (-1, 4), (8, 4)):
        with pytest.raises(Refusal) as error:
            versions.rom_bytes(version, start, end)
        assert error.value.findings[0].key == "version.rom"


def test_asm_path(world) -> None:
    cfg = world[0]
    target = cfg.project.root / "asm" / "a" / "sub" / "f.s"
    target.parent.mkdir(parents=True)
    target.write_text("")
    assert versions.asm_path(cfg, "a", "f") == target
    with pytest.raises(Refusal) as error:
        versions.asm_path(cfg, "a", "missing")
    assert error.value.findings[0].key == "version.split"
    assert error.value.findings[0].action == "run unbake setup"


def test_undefined_ignores_declarations(tmp_path: Path) -> None:
    declared = tmp_path / "declared.o"
    declared.write_bytes(fixture.elf_with_symbols(["f"], []))
    used = tmp_path / "used.o"
    used.write_bytes(fixture.elf_with_symbols(["f"], ["D_1", "A_0"]))
    assert versions.undefined(declared) == ()
    assert versions.undefined(used) == ("A_0", "D_1")


def test_resolve_names_missing(world) -> None:
    de = _version(world, "a")
    found = versions.resolve(de, ["D_800C6504_eu", "f"], "src/u.c")
    assert len(found) == 1
    finding = found[0]
    assert (finding.key, finding.missing, finding.versions) == ("symbols.unknown", ("D_800C6504_eu",), ("a",))
    assert (finding.path, finding.unit) == ("symbols.toml", "src/u.c")
    assert versions.resolve(de, ["f"], "src/u.c") == ()


def test_top_level_end_marker_is_not_a_member(tmp_path: Path) -> None:
    from unbake.contracts import Version

    document = {
        "segments": [
            {
                "name": "main",
                "type": "code",
                "start": 0x1000,
                "vram": 0x80000000,
                "subsegments": [[0x1000, "asm", "f"]],
            },
            [0x1008],
        ]
    }
    data = json.dumps(document).encode()
    version = Version("a", tmp_path / "rom", "sha", "s.yaml", "sym.txt", {}, versions._segment_rows(data))
    (row,) = versions.rows(version, lambda path: data)
    assert row[0] == "f" and row[2].rom_end == 0x1008


def test_generated_symbols_use_emitted_facts_not_encoded_names(world):
    cfg, ctx, _ = world
    root = cfg.project.root
    split = root / cfg.project.version_files["a"].split
    import yaml

    document = yaml.safe_load(split.read_text())
    document.setdefault("options", {})["undefined_funcs_auto_path"] = "build/a/undefined_funcs_auto.txt"
    split.write_text(yaml.safe_dump(document))
    auto = root / "build/a/undefined_funcs_auto.txt"
    auto.parent.mkdir(parents=True, exist_ok=True)
    auto.write_text("func_80000488 = 0x80000488;\n")
    asm = root / "asm/a/f.s"
    asm.parent.mkdir(parents=True, exist_ok=True)
    asm.write_text("glabel f\n/* 1000 80000400 03E00008 */ jr $ra\n.LDEADBEEF:\n/* 1004 80000404 00000000 */ nop\n")
    version = versions.read(cfg, ctx["reader"])["a"]
    assert version.symbols["func_80000488"] == 0x80000488
    assert version.symbols[".LDEADBEEF"] == 0x80000404
    assert not versions.resolve(version, [".LDEADBEEF", "func_80000488"], "u")
    asm.write_text(asm.read_text().replace("80000400", "80000420"))
    # Configured identities remain authoritative over incidental generated defaults.
    assert versions.read(cfg, ctx["reader"])["a"].symbols["f"] == fixture.VRAM_BASE


def test_generated_undeclared_label_conflicts_refuse(world):
    cfg, ctx, _ = world
    root = cfg.project.root
    for name, address in [("one", "80000400"), ("two", "80000404")]:
        asm = root / f"asm/a/{name}.s"
        asm.parent.mkdir(parents=True, exist_ok=True)
        asm.write_text(f".Lshared:\n/* 1000 {address} 00000000 */ nop\n")
    with pytest.raises(Refusal, match=r"symbols\.conflict"):
        versions.read(cfg, ctx["reader"])


def test_disassembler_context_unknown_targets_are_facts(world):
    cfg, ctx, _ = world
    dump = cfg.project.root / ".unbake/symbols/a/spim_context_unksegment.csv"
    dump.parent.mkdir(parents=True)
    dump.write_text(
        "category,address,getName\n"
        "symbol,0x81234560,.LDEADBEEF\n"
        "symbol,0x80000100,D_99999999\n"
        "new_pointer_in_data,0x80000300\n"
    )
    symbols = versions.read(cfg, ctx["reader"])["a"].symbols
    assert symbols[".LDEADBEEF"] == 0x81234560 and symbols["D_99999999"] == 0x80000100
    assert not any(name.startswith(("D_ub_", "func_ub_", ".Lub_")) for name in symbols)  # no namespace aliases


def test_jump_targets_decode_from_instruction_words_even_with_disassembler_context(world):
    cfg, ctx, _ = world
    root = cfg.project.root
    context = root / ".unbake/symbols/a/spim_context_x.csv"
    context.parent.mkdir(parents=True)
    context.write_text("category,address,getName\n")
    asm = root / "asm/a/j.s"
    asm.parent.mkdir(parents=True, exist_ok=True)
    asm.write_text(
        "glabel j\n/* 1000 80000400 08003C00 */  j          func_8000F000\n"
        "/* 1004 80000404 0C000000 */   jal        func_ext\n"
    )
    symbols = versions.read(cfg, ctx["reader"])["a"].symbols
    assert symbols["func_8000F000"] == 0x8000F000
    assert symbols["func_ext"] == 0x80000000


def test_hi_lo_pairs_decode_their_address_from_both_instruction_words() -> None:
    text = (
        b"/* 1074 80200474 3C04002A */  lui        $a0, %hi(D_002A71C0)\n"
        b"/* 1078 80200478 248471C0 */  addiu      $a0, $a0, %lo(D_002A71C0)\n"
        b"/* 107C 8020047C 3C058010 */  lui        $a1, %hi(D_80108000)\n"
        b"/* 1080 80200480 8CA48000 */  lw         $a0, %lo(D_80108000)($a1)\n"
    )
    assert versions._asm_symbols(text)[1] == {"D_002A71C0": 0x002A71C0, "D_80108000": 0x800F8000}


def test_rows_refuse_one_name_on_two_rows(tmp_path: Path, toolchains: dict) -> None:
    segment = {
        "name": "m",
        "type": "code",
        "start": 0x1000,
        "vram": 0x80000000,
        "subsegments": [[0x1000, "rodata", "r"], [0x1010, "rodata", "r"], [0x1020]],
    }
    version, reader = _yaml_version(tmp_path, segment)
    with pytest.raises(Refusal) as refused:
        versions.rows(version, reader)
    assert refused.value.findings[0].key == "version.split" and refused.value.findings[0].unit == "r"
    assert "0x1000, 0x1010" in refused.value.findings[0].reason


def test_asm_symbols_name_the_code_the_files_show() -> None:
    text = (
        b"glabel func_a\n/* 0 80200000 03E00008 */  jr         $ra\n"
        b"dlabel D_table\n/* 10 80200010 00000000 */ .word 0\n"
        b"/* 4 80200004 0C0A0000 */  jal        func_b\n"
        b"/* 8 80200008 1440FFFF */  bnez       $v0, .Lauto_80200000 /* handwritten */\n"
    )
    code = versions._asm_symbols(text)[2]
    assert "func_a" in code and "func_b" in code and ".Lauto_80200000" in code and "D_table" not in code


def _elf(symbols):
    """A big-endian ELF32 file with a null section, a symbol table and its string table, nothing else."""
    import struct

    strings, offsets = b"\0", []
    for name, *_ in symbols:
        offsets.append(len(strings))
        strings += name.encode() + b"\0"
    table = bytes(16) + b"".join(
        struct.pack(">IIIBBH", o, 0, 0, info, 0, index) for o, (_, info, index) in zip(offsets, symbols, strict=True)
    )
    body = 52
    shoff = body + len(table) + len(strings)
    fields = struct.pack(">HHIIIIIHHHHHH", 1, 8, 1, 0, 0, shoff, 0, 52, 0, 0, 40, 3, 0)
    header = b"\x7fELF\x01\x02\x01" + bytes(9) + fields
    sections = bytes(40) + struct.pack(">10I", 0, 2, 0, 0, body, len(table), 2, 0, 4, 16)
    sections += struct.pack(">10I", 0, 3, 0, 0, body + len(table), len(strings), 0, 0, 1, 0)
    return header + table + strings + sections


def test_undefined_reads_the_global_and_weak_names_an_object_leaves_open(tmp_path):
    path = tmp_path / "unit.o"
    path.write_bytes(_elf([("needs", 0x10, 0), ("weak_need", 0x20, 0), ("local_open", 0x00, 0), ("defined", 0x10, 1)]))
    assert versions.undefined(path) == ("needs", "weak_need")
    path.write_bytes(b"\x7fELF\x02\x01" + bytes(60))
    with pytest.raises(Refusal) as caught:
        versions.undefined(path)
    assert caught.value.findings[0].reason == "the object is not a 32-bit big-endian ELF file"


@pytest.mark.parametrize(
    "word,expected",
    [
        (0x03E00008, True),
        (0x0320F809, True),
        (0x08000000, True),
        (0x0C000000, True),
        (0x1000FFFF, True),
        (0x54200001, True),
        (0x04010000, True),
        (0x04110000, True),
        (0x04040000, False),
        (0x45000001, True),
        (0x40800000, False),
        (0x00000000, False),
        (0x3C02A450, False),
        (0x23BDFFB8, False),
        (0x0000000C, False),
    ],
)
def test_delay_slot_predicate(word: int, expected: bool) -> None:
    assert versions.delay_slot(word) is expected


def test_one_command_lists_each_versions_files_once_until_they_are_rewritten(world, monkeypatch) -> None:
    from unbake import effort

    cfg, ctx, _ = world
    listed = []
    real = versions.fact_files
    monkeypatch.setattr(versions, "fact_files", lambda config, vid: listed.append(vid) or real(config, vid))
    with effort.command("check", []):
        first = versions.read(cfg, ctx["reader"])
        versions.read(cfg, ctx["reader"], ["a"])
        assert sorted(listed) == ["a", "b"]  # the second read built nothing
        effort.forget("generated", "versions.read")
        assert versions.read(cfg, ctx["reader"]).keys() == first.keys()
    assert sorted(listed) == ["a", "a", "b", "b"]


def test_reused_versions_keep_their_facts_and_read_only_their_split(world, monkeypatch) -> None:
    from unbake import effort

    cfg, ctx, _ = world
    with effort.command("check", []):
        before = versions.read(cfg, ctx["reader"])
    monkeypatch.setattr(versions, "fact_files", lambda *a: pytest.fail("facts were read again"))
    with effort.command("check", []):
        after = versions.read(cfg, ctx["reader"], ["a"], before)
    assert after["a"].symbols == before["a"].symbols and after["a"].code == before["a"].code


def test_a_split_file_is_parsed_once_per_command(world, monkeypatch) -> None:
    from unbake import effort

    ctx = world[1]
    parses = []
    real = versions.yaml.load
    monkeypatch.setattr(versions.yaml, "load", lambda *a, **k: parses.append(1) or real(*a, **k))
    version = _version(world, "a")
    with effort.command("check", []):
        versions.rows(version, ctx["reader"])
        before = len(parses)
        versions.rows(version, ctx["reader"])
    assert len(parses) == before


def test_a_warm_capture_stats_no_asm_file_and_lists_only_the_directory_that_moved(world, monkeypatch) -> None:
    from unbake import layout

    cfg = world[0]
    base = cfg.project.root / "asm/a/nonmatchings"
    for d in range(3):
        (base / f"seg{d}").mkdir(parents=True)
        for n in range(20):
            (base / f"seg{d}/f{n}.s").write_text(f"glabel f{d}_{n}\n")
    cold = layout.capture(cfg)
    stats, scans = [], []
    real_stat, real_scandir = os.stat, os.scandir
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: stats.append(str(p)) or real_stat(p, *a, **k))
    monkeypatch.setattr(os, "scandir", lambda p: scans.append(str(p)) or real_scandir(p))
    assert layout.capture(cfg).digest == cold.digest
    assert not [p for p in stats if p.endswith(".s")] and scans == []
    (base / "seg1/new.s").write_text("glabel added\n")  # a file appears: only its directory is listed again
    listed = versions.fact_files(cfg, "a")[1]
    assert len(listed) == 61 and "asm/a/nonmatchings/seg1/new.s" in listed
    assert [Path(p).name for p in scans] == ["seg1"]
    assert versions.fact_inputs(cfg, "a")[2] != versions.fact_inputs(cfg, "b")[2]


def test_rows_with_a_cache_are_built_once_per_distinct_split_text(monkeypatch) -> None:
    built = []
    kept: dict[str, bytes] = {}
    monkeypatch.setattr(versions, "_rows", lambda version, text, cache: built.append(text) or [])
    monkeypatch.setattr(
        versions,
        "pickled",
        lambda cache, kind, key, produce: kept[key] if key in kept else kept.setdefault(key, produce()),
    )
    version = SimpleNamespace(id="v", segments=(), split="s")
    for _ in range(2):
        versions.effort._memo.clear()
        versions.rows(version, lambda p: b"same", object())
    versions.effort._memo.clear()
    versions.rows(version, lambda p: b"other", object())
    assert built == [b"same", b"other"]
