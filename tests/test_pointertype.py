"""The address-table typer: all words resolve, one word does not, the same rule on two versions."""

import struct
from pathlib import Path
from types import SimpleNamespace

from unbake import config
from unbake import pointertype as pt

SYMBOLS = """g_earlierMenuA_80158C08 = 0x80158C08; //
g_earlierPoolB_45B380 = 0x0045B380; // rom:0x15437C size:0x34
g_earlierMenuC_800E2AC8 = 0x800E2AC8; // rom:0x1 size:0x4
"""


def words(*w: int) -> bytes:
    return b"".join(struct.pack(">I", x) for x in w)


def known():
    return pt.symbols(SYMBOLS)


def test_symbols_key_on_the_low_24_bits_so_a_pool_symbol_without_the_segment_bits_resolves():
    table = pt.read(words(0x80158C08, 0, 0x8045B380), "eu", 0, 12, known())
    assert isinstance(table, pt.Table)
    assert table.targets == ("g_earlierMenuA_80158C08", None, "g_earlierPoolB_45B380")


def test_every_word_resolves_and_the_definition_is_a_typed_pointer_array_of_symbols():
    table = pt.read(words(0x80158C08, 0x800E2AC8, 0x80158C08), "eu", 0, 12, known())
    assert isinstance(table, pt.Table)
    lines = pt.render(table, "g_earlierAddrTable_461750")
    assert lines == [
        "extern char g_earlierMenuA_80158C08[];",
        "extern char g_earlierMenuC_800E2AC8[];",
        "void *g_earlierAddrTable_461750[3] = "
        "{ g_earlierMenuA_80158C08, g_earlierMenuC_800E2AC8, g_earlierMenuA_80158C08 };",
    ]
    assert "0x8015" not in "".join(lines)  # no raw word


def test_one_unresolved_word_refuses_the_span_naming_the_span_and_the_word():
    rom = b"\0" * 8 + words(0x80158C08, 0x801547B0, 0x80158C08)
    got = pt.read(rom, "eu", 8, 20, known())
    assert got == {
        "version": "eu",
        "rom": 8,
        "end": 20,
        "word_rom": 12,
        "word": "0x801547B0",
        "why": "no symbol of the older build's layout has this address",
    }


def test_a_word_that_is_no_address_refuses():
    for word in (0xFFFFFFFF, 0x00250FF0, 0x80800000):
        got = pt.read(words(0x80158C08, word), "eu", 0, 8, known())
        assert isinstance(got, dict) and got["word_rom"] == 4 and got["word"] == f"0x{word:08X}"


def test_an_edge_that_splits_a_word_is_widened_to_the_word():
    rom = words(0x80158C08, 0x800E2AC8)
    table = pt.read(rom, "eu", 1, 7, known())  # the neighbours own byte 0 and byte 7
    assert isinstance(table, pt.Table) and (table.rom, table.end) == (0, 8)
    assert isinstance(pt.read(rom, "eu", 1, 12, known()), dict)  # past the ROM: refused


def _project(tmp_path: Path, monkeypatch, roms: dict[str, bytes]):
    for v in roms:
        path = tmp_path / "versions" / v / "earlier_build_symbols.txt"
        path.parent.mkdir(parents=True)
        path.write_text(SYMBOLS)
    monkeypatch.setattr(config, "rom", lambda project, v: SimpleNamespace(read_bytes=lambda: roms[v]))
    return SimpleNamespace(project=SimpleNamespace(root=tmp_path))


def test_one_rule_closes_both_versions_with_one_guarded_source(tmp_path, monkeypatch):
    table = words(0x80158C08, 0, 0x800E2AC8)
    cfg = _project(tmp_path, monkeypatch, {"eu": b"\0" * 16 + table, "eu-x": b"\0" * 32 + table})
    out = pt.command(cfg, {"spec": ["eu:0x10:0x1c:0x307000", "eu-x:0x20:0x2c:0x312000"]})
    assert out["unfit"] == [] and out["bytes"] == 24
    assert out["written"] == ["EARLIER_ADDRTABLE_307010.CPP", "EARLIER_ADDRTABLE_312020.CPP"]
    text = (tmp_path / "src/DATA/EARLIER_BUILD/EARLIER_ADDRTABLE_307010.CPP").read_text()
    assert "#if defined(VERSION_EU)\n" in text and "void *g_earlierAddrTable_307010[3]" in text
    rows = (tmp_path / "versions/eu/earlier_build_symbols.txt").read_text().splitlines()
    assert rows[-1] == "g_earlierAddrTable_307010 = 0x00307010; // rom:0x10 size:0xC"


def test_same_vram_in_two_versions_shares_one_source_and_a_refused_span_writes_nothing(tmp_path, monkeypatch):
    table = words(0x80158C08, 0x800E2AC8)
    bad = words(0x80158C08, 0x801547B0)
    cfg = _project(tmp_path, monkeypatch, {"eu": b"\0" * 4 + table, "eu-x": b"\0" * 8 + table + bad})
    out = pt.command(cfg, {"spec": ["eu:4:12:0x307000", "eu-x:8:16:0x306FFC", "eu-x:16:24:0x306FFC"]})
    assert out["written"] == ["EARLIER_ADDRTABLE_307004.CPP"]
    text = (tmp_path / "src/DATA/EARLIER_BUILD/EARLIER_ADDRTABLE_307004.CPP").read_text()
    assert text.count("#if defined(VERSION_EU) || defined(VERSION_EU_X)\n") == 1
    assert len(out["unfit"]) == 1 and out["unfit"][0]["word_rom"] == 20 and out["unfit"][0]["word"] == "0x801547B0"


def test_bodies_that_differ_by_version_share_one_source_guarded_apart(tmp_path, monkeypatch):
    eu, eux = words(0x80158C08, 0x800E2AC8), words(0x80158C08, 0x8045B380)
    cfg = _project(tmp_path, monkeypatch, {"eu": b"\0" * 4 + eu, "eu-x": b"\0" * 4 + eux})
    out = pt.command(cfg, {"spec": ["eu:4:12:0x307000", "eu-x:4:12:0x307000"]})
    assert out["unfit"] == [] and out["written"] == ["EARLIER_ADDRTABLE_307004.CPP"]
    text = (tmp_path / "src/DATA/EARLIER_BUILD/EARLIER_ADDRTABLE_307004.CPP").read_text()
    assert text.count("#if defined(VERSION_EU)\n") == 1 and text.count("#elif defined(VERSION_EU_X)\n") == 1
    assert text.count("#endif") == 1 and text.count("g_earlierAddrTable_307004[2]") == 2
