"""Zero pads a version carries: the text rule and the `layout pads` action."""

from __future__ import annotations

from pathlib import Path

import pytest

from unbake import folders, pads, symbols
from unbake.contracts import Config, Refusal

VERSIONS = ["a", "b"]
KNOWN = {"a": {"pad_a", "tbl_a", "tbl2_a"}, "b": {"pad_b", "tbl_b"}}


def guarded(version: str, body: str) -> str:
    return f"#if defined(VERSION_{version})\n{body}\n#endif\n"


def run(text: str, known: dict[str, set[str]] = KNOWN) -> pads.Pads:
    return pads.pads(text, VERSIONS, known)


def test_a_zero_pad_with_a_rom_fact_in_every_version_it_names_is_removed() -> None:
    text = (
        "u32 head = 1;\n\n" + guarded("a", "u32 pad_a[1] = { 0 };") + "\n" + guarded("b", "u32 pad_b[2] = { 0x0, 0 };")
    )
    got = run(text)
    assert [r["names"] for r in got.removed] == [["pad_a"], ["pad_b"]]
    assert got.text == "u32 head = 1;\n" and not got.held


def test_data_that_is_not_all_zero_stays() -> None:
    text = guarded("a", "u32 tbl_a[2] = { 0, 1 };")
    assert run(text).text == text and not run(text).removed


def test_a_pad_without_a_symbol_fact_is_held_with_the_reason() -> None:
    text = guarded("a", "u32 pad_x[1] = { 0 };")
    got = run(text)
    assert got.text == text and got.held[0]["why"] == "no data fact in a"


def test_a_pad_the_rest_of_the_file_names_is_held() -> None:
    text = guarded("a", "u32 pad_a[1] = { 0 };") + "u32 *p = pad_a;\n"
    assert run(text).held[0]["why"] == "named by other text of the file"


def test_a_name_in_a_comment_or_string_is_not_a_use() -> None:
    text = guarded("a", "u32 pad_a[1] = { 0 };") + '/* pad_a */ char *s = "pad_a";\n'
    assert [r["names"] for r in run(text).removed] == [["pad_a"]]


def test_a_pad_between_two_data_definitions_of_its_version_is_held() -> None:
    text = "u32 one = 1;\n" + guarded("a", "u32 pad_a[1] = { 0 };") + "u32 two = 2;\n"
    assert run(text).held[0]["why"] == "between other data in a"
    other = "u32 one = 1;\n" + guarded("a", "u32 pad_a[1] = { 0 };") + guarded("b", "u32 two = 2;")
    assert [r["names"] for r in run(other).removed] == [["pad_a"]]  # version b's data does not touch a's pad


def test_a_name_lies_in_data_when_the_facts_around_it_in_its_mapping_are_not_code() -> None:
    table = {
        "f": (0x80, 0x10, 8, "func"),
        "after_code": (0x88, 0x18, 4, "data"),
        "pad": (0x8C, 0x1C, 4, "data"),
        "next": (0x90, 0x20, 4, "data"),
        "tail_pad": (0x94, 0x24, 4, "data"),
        "code": (0x98, 0x28, 8, "func"),
        "other_map": (0x200, 0x100, 4, "data"),
        "first": (0x300, 0x4, 4, "data"),
        "no_rom": (0x400, None, None, "data"),
    }
    assert pads.inside(table) == {"pad", "next"}  # tail_pad is followed by code: the text would move up


def test_text_without_zero_blocks_is_returned_as_is() -> None:
    assert run("s32 x = 1;\n").text == "s32 x = 1;\n"


@pytest.mark.parametrize(
    ("body", "want"),
    [("u32 a[1] = { 0 };", True), ("u32 a = 0x00u;", True), ("u32 a = 1;", False), ("u32 a;", False), ("", False)],
)
def test_zeros_reads_every_initializer(body: str, want: bool) -> None:
    assert pads.zeros(body) is want


def _project(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = cfg.project.root
    (root / "src").mkdir(exist_ok=True)
    (root / "src/tbl.c").write_text("u32 head = 1;\n" + guarded("a", "u32 pad_a[1] = { 0 };"))
    (root / "src/plain.c").write_text("s32 x = 1;\n")
    monkeypatch.setattr(folders, "_tracked", lambda config: ["src/plain.c", "src/tbl.c"])
    rows = {
        "a": {"head": (0x100, 0x100, 4, "data"), "pad_a": (0x104, 0x104, 4, "data"), "tail": (0x108, 0x108, 4, "data")},
        "b": {},
    }
    monkeypatch.setattr(symbols, "load", lambda config, version: rows[version])
    return root


def test_layout_pads_rewrites_the_file_and_is_idempotent(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(cfg, monkeypatch)
    result = folders.pads(cfg, {})
    assert (result["blocks_removed"], result["written"]) == (1, ["src/tbl.c"])
    assert (root / "src/tbl.c").read_text() == "u32 head = 1;\n"
    again = folders.pads(cfg, {})
    assert again["blocks_removed"] == 0 and again["written"] == []


def test_layout_pads_can_be_limited_to_one_file_and_refuses_an_untracked_one(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(cfg, monkeypatch)
    assert folders.pads(cfg, {"file": "src/plain.c"})["written"] == []
    assert "pad_a" in (root / "src/tbl.c").read_text()
    with pytest.raises(Refusal):
        folders.pads(cfg, {"file": "src/none.c"})
    assert folders.run(cfg, {"action": "pads", "file": "src/plain.c"})["action"] == "pads"
