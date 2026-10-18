"""Version facts: the symbol_addrs.txt parser, segment lookup, asm paths and the cached scoring symbol script."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from unbake import store, versions
from unbake.contracts import Refusal, Segment

pytestmark = pytest.mark.usefixtures("plain_roms")

TEXT = (
    b"// a comment\n"
    b"func_one = 0x80200400; // type:func rom:0x1000 size:0x40\n"
    b"gTable = 0x80210000; // type:data rom:0x2010\n"
    b"gLabel = 0x80200410;\n"
    b"bare=0x80200420 ;\n"
)


def make_config(tmp_path: Path, text: bytes = TEXT) -> SimpleNamespace:
    (tmp_path / "versions/us").mkdir(parents=True, exist_ok=True)
    (tmp_path / "versions/us/symbol_addrs.txt").write_bytes(text)
    (tmp_path / "roms").mkdir(exist_ok=True)
    (tmp_path / "roms/baserom.us.z64").write_bytes(bytes(0x4000))
    cuts = (Segment("main", 0x1000, 0x80200400, "code"), Segment("ovl", 0x2000, 0x80210000, "code"))
    project = SimpleNamespace(
        root=tmp_path, versions=("us",), version={"us": SimpleNamespace(size=0x4000, segments=cuts)}
    )
    return SimpleNamespace(project=project)


def test_parser_reads_splat_attributes():
    rows = {r.name: r for r in versions.parse_symbol_addrs(TEXT)}
    assert rows["func_one"] == versions.Row("func_one", 0x80200400, "func", 0x1000, 0x40, 2)
    assert (rows["gTable"].kind, rows["gTable"].rom, rows["gTable"].size) == ("data", 0x2010, None)
    assert (rows["gLabel"].kind, rows["gLabel"].rom) == ("", None)
    assert rows["bare"].vram == 0x80200420


def test_two_addresses_for_one_name_are_refused():
    with pytest.raises(Refusal) as error:
        versions.parse_symbol_addrs(b"a = 0x1;\na = 0x2;\n")
    assert error.value.findings[0].key == "symbols.conflict"


def test_segments_end_where_the_next_begins(tmp_path):
    assert versions.segments(make_config(tmp_path), "us") == (
        ("main", 0x1000, 0x2000, 0x80200400),
        ("ovl", 0x2000, 0x4000, 0x80210000),
    )


def test_asm_path_is_by_segment_not_by_file(tmp_path):
    config = make_config(tmp_path)
    assert versions.asm_path(config, "us", "func_one") == tmp_path / "asm/us/nonmatchings/main/func_one.s"
    assert versions.asm_path(config, "us", "gTable") == tmp_path / "asm/us/nonmatchings/ovl/gTable.s"
    assert versions.asm_path(config, "us", "gLabel") == tmp_path / "asm/us/nonmatchings/main/gLabel.s"  # by vram


def test_unknown_name_has_no_asm_path(tmp_path):
    with pytest.raises(Refusal) as error:
        versions.asm_path(make_config(tmp_path), "us", "nope")
    assert "symbol_addrs.txt" in error.value.findings[0].action


def test_read_gives_the_fact_file_as_the_symbol_table(tmp_path):
    config = make_config(tmp_path)
    got = versions.read(config, lambda p: (tmp_path / p).read_bytes())["us"]
    assert got.symbols["func_one"] == 0x80200400
    assert got.code == frozenset({"func_one"})
    assert (got.split, got.symbols_file) == ("versions/us/splat.yaml", "versions/us/symbol_addrs.txt")


def test_symbols_ld_is_cached_by_the_fact_file_sha(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    kept: dict[tuple[str, str], bytes] = {}
    produced = []

    def cached(_, kind, key, produce):
        if (kind, key) not in kept:
            produced.append(kind)
            kept[(kind, key)] = produce()
        return kept[(kind, key)]

    monkeypatch.setattr(store, "cached", cached)
    first = versions.symbols_ld(config, "us")
    assert versions.symbols_ld(config, "us") == first
    assert produced == ["generated"]
    text = first.decode()
    assert "PROVIDE(func_one = 0x00200400);" in text  # code is linked at its masked address
    assert "PROVIDE(gTable = 0x80210000);" in text  # data keeps its address
    (tmp_path / "versions/us/symbol_addrs.txt").write_bytes(TEXT + b"extra = 0x80200430; // type:func\n")
    assert versions.symbols_ld(config, "us") != first
    assert produced == ["generated", "generated"]
    assert not list(tmp_path.rglob("symbols.ld"))  # the script lives in the cache only
