"""symbols: the project table, its generated files, rename and joins."""

from __future__ import annotations

import pytest

from unbake import symbols
from unbake.contracts import Refusal

VERSIONS = ("a", "b")
TABLE = b"""schema = 1

[symbol.foo]
kind = "function"
a = 0x80000400
b = 0x80000410

[symbol.bar]
kind = "data"
a = 0x80000400
"""


def test_parse_dump_round_trip_and_declared() -> None:
    table = symbols.parse(TABLE, VERSIONS)
    assert table == {
        "foo": {"kind": "function", "a": 0x80000400, "b": 0x80000410},
        "bar": {"kind": "data", "a": 0x80000400},
    }
    assert symbols.parse(symbols.dump(table), VERSIONS) == table
    assert symbols.declared(table, "b") == {"foo": 0x80000410}


@pytest.mark.parametrize("row", ['kind = "function"', 'kind = "function"\nc = 0x1', "a = 0x1", 'kind = "x"\na = 0x1'])
def test_parse_refuses_rows_without_a_known_version_address(row: str) -> None:
    with pytest.raises(Refusal) as error:
        symbols.parse(f"schema = 1\n\n[symbol.n]\n{row}\n".encode(), VERSIONS)
    assert error.value.findings[0].key in {"symbols.table", "config.schema"}


def test_render_marks_functions_and_shared_addresses() -> None:
    table = symbols.parse(TABLE, VERSIONS)
    assert symbols.render(table, "a") == (
        b"bar = 0x80000400; // allow_duplicated:true\nfoo = 0x80000400; // type:func allow_duplicated:true\n"
    )
    assert symbols.render(table, "b") == b"foo = 0x80000410; // type:func\n"


def test_files_name_the_table_and_every_version_file() -> None:
    table = symbols.parse(TABLE, VERSIONS)
    files = {v: f"versions/{v}/symbol_addrs.txt" for v in VERSIONS}
    assert set(symbols.files(table, files)) == {"symbols.toml", *files.values()}


def test_load_refuses_a_missing_table_by_name() -> None:
    def reader(path: str) -> bytes:
        raise FileNotFoundError(path)

    with pytest.raises(Refusal) as error:
        symbols.load(reader, VERSIONS)
    assert (error.value.findings[0].key, error.value.findings[0].path) == ("config.missing", "symbols.toml")


def test_join_keeps_the_name_c_uses_and_refuses_disagreement() -> None:
    table = {"x": {"kind": "data", "a": 1}, "y": {"kind": "function", "b": 2}}
    assert symbols.join(table, "x", "y", {"y"}, "a") == "y"
    assert table == {"y": {"kind": "function", "a": 1, "b": 2}}
    both = {"x": {"kind": "data", "a": 1}, "y": {"kind": "data", "a": 2}}
    assert symbols.join(both, "x", "y", set(), "a") is None
    used = {"x": {"kind": "data", "a": 1}, "y": {"kind": "data", "b": 2}}
    assert symbols.join(used, "x", "y", {"x", "y"}, "a") is None
    assert symbols.join(used, "x", "y", set(), "a") == "x"  # the source version's name survives


def test_fix_rows_adds_and_corrects_the_address_a_versions_code_uses():
    table = {"old": {"kind": "data", "a": 0x80000100}}
    changed = symbols.fix_rows(
        table, {("old", "a"): (0x80000104, False), ("old", "b"): (0x80000200, False), ("call", "a"): (0x80000400, True)}
    )
    assert changed == 3
    assert table == {
        "old": {"kind": "data", "a": 0x80000104, "b": 0x80000200},
        "call": {"kind": "function", "a": 0x80000400},
    }


def test_rename_to_a_name_that_places_itself_in_other_versions_joins_the_identities() -> None:
    table = {"a": {"kind": "data", "de": 0x100}, "b": {"kind": "data", "eu": 0x200}, "c": {"kind": "data", "de": 0x300}}
    symbols.rename(table, "b", "a")
    assert table == {"a": {"kind": "data", "de": 0x100, "eu": 0x200}, "c": {"kind": "data", "de": 0x300}}
    with pytest.raises(Refusal):
        symbols.rename(table, "c", "a")


def test_cross_version_correspondence_cannot_overwrite_existing_regional_address(monkeypatch):
    from types import SimpleNamespace

    table = {"canonical": {"kind": "data", "a": 0x100, "b": 0x100}, "regional": {"kind": "data", "b": 0x200}}
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
        versions={"b": SimpleNamespace(symbols={"regional": 0x200})},
    )
    monkeypatch.setattr(symbols, "referenced", lambda *_: {"canonical"})
    assert symbols.join_pairs(snapshot, table, [("canonical", "a", "regional", "b")]) == 0
    assert table["canonical"]["b"] == 0x100
    del table["regional"]
    assert symbols.join_pairs(snapshot, table, [("canonical", "a", "regional", "b")]) == 0
    assert table["canonical"]["b"] == 0x100
    snapshot.versions["a"] = SimpleNamespace(symbols={"other": 0x300})
    assert symbols.join_pairs(snapshot, table, [("other", "a", "canonical", "b")]) == 0
    assert table["canonical"]["a"] == 0x100
