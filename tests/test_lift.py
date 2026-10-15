"""Address words of extracted data become named symbols only where every holder's ROM agrees."""

from types import SimpleNamespace

import pytest

from unbake import lift
from unbake.contracts import Member, Placement


def _scene(monkeypatch, other_word_b):
    data = Member(
        "data/unresolved/80100000",
        "data",
        "data",
        "g",
        (Placement("a", ".data", 0, 12, 0x80100000), Placement("b", ".data", 100, 112, 0x80100000)),
    )
    target = Member(
        "table",
        "data",
        "data",
        "g",
        (Placement("a", ".data", 50, 60, 0x80200000), Placement("b", ".data", 150, 160, 0x80210000)),
    )
    snapshot = SimpleNamespace(
        layout=SimpleNamespace(members={data.name: data, target.name: target}),
        config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
        digest="snap",
        peek=lambda path: b"extern int declared_elsewhere;",
        versions={"a": SimpleNamespace(id="a"), "b": SimpleNamespace(id="b")},
    )
    words = {"a": [1, 0x80200000, 0x80008000], "b": [1, other_word_b, 0x80008000]}

    def rom(version, start, end):
        base = 0 if version.id == "a" else 100
        return words[version.id][(start - base) // 4].to_bytes(4, "big")

    monkeypatch.setattr(lift.versions, "rom_bytes", rom)
    monkeypatch.setattr("unbake.headers.catalog", lambda snapshot: {"declared_elsewhere": ("include/h.h", 1, "")})
    return snapshot, data


TEXT = "unsigned int D_auto_80100000[] = {\n    0x00000001, 0x80200000, 0x80008000,\n};\n"


def test_a_word_every_holder_agrees_on_is_named(monkeypatch):
    snapshot, data = _scene(monkeypatch, 0x80210000)
    out, declared = lift.lift(snapshot, data, TEXT)
    assert "void *D_auto_80100000[]" in out
    assert "table," in out
    assert "(void *)0x00000001" in out and "(void *)0x80008000" in out
    assert declared == {"table": "extern char table[];"}


def test_a_word_that_differs_from_the_holders_address_stays_a_literal(monkeypatch):
    snapshot, data = _scene(monkeypatch, 0x80200000)  # version b still holds a's address: not a pointer there
    assert lift.lift(snapshot, data, TEXT) == (TEXT, {})


@pytest.mark.parametrize("text", ["int x;\n", "unsigned int a[] = {1, 2};\n"])
def test_text_without_address_words_is_unchanged(monkeypatch, text):
    snapshot, data = _scene(monkeypatch, 0)
    assert lift.lift(snapshot, data, text) == (text, {})


def test_unresolved_rows_are_spelled_d_auto():
    assert lift._symbol("data/unresolved/80126768") == "D_auto_80126768"


def test_the_word_position_counts_every_element_not_only_hex_ones(monkeypatch):
    snapshot, data = _scene(monkeypatch, 0x80210000)
    text = "unsigned int D_auto_80100000[] = {\n    1, 0x80200000, 0x80008000,\n};\n"
    out, _ = lift.lift(snapshot, data, text)
    assert "(void *)1," in out and "(void *)1, table," in out
    mixed = (
        "unsigned int D_auto_80100000[] = {\n    SOMETHING, 0x80200000,\n};\n"  # the address is element 1, ROM word 1
    )
    assert "SOMETHING, table" in lift.lift(snapshot, data, mixed)[0]


def test_a_name_a_header_declares_gets_no_extern(monkeypatch):
    snapshot, data = _scene(monkeypatch, 0x80210000)
    monkeypatch.setattr(lift, "_declared", lambda s: {"table"})
    assert lift.lift(snapshot, data, TEXT)[1] == {}


def test_the_lifted_text_is_no_address_int_or_address_literal_debt(monkeypatch):
    from unbake.families import typedebt

    snapshot, data = _scene(monkeypatch, 0x80210000)
    out, _ = lift.lift(snapshot, data, TEXT)
    assert typedebt.address_int(out) == {}


def test_a_name_only_the_catalog_declares_suppresses_the_extern(monkeypatch):
    snapshot, data = _scene(monkeypatch, 0x80210000)
    monkeypatch.setattr("unbake.headers.catalog", lambda s: {"table": ("include/h.h", 1, "")})
    assert lift.lift(snapshot, data, TEXT)[1] == {}


def test_an_interior_address_of_another_member_stays_a_literal(monkeypatch):
    snapshot, data = _scene(monkeypatch, 0x80210004)
    text = TEXT.replace("0x80200000", "0x80200004")
    assert lift.lift(snapshot, data, text) == (text, {})


def test_a_pointer_into_the_array_itself_is_an_element_address(monkeypatch):
    snapshot, data = _scene(monkeypatch, 0x80100004)
    snapshot.versions["a"] = snapshot.versions["a"]
    words = {"a": [1, 0x80100004, 0x80008000], "b": [1, 0x80100004, 0x80008000]}
    monkeypatch.setattr(
        lift.versions,
        "rom_bytes",
        lambda v, s, e: words[v.id][(s - (0 if v.id == "a" else 100)) // 4].to_bytes(4, "big"),
    )
    text = TEXT.replace("0x80200000", "0x80100004")
    out, declared = lift.lift(snapshot, data, text)
    assert "&D_auto_80100000[0x1]" in out and declared == {}
