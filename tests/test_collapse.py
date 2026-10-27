"""Version branches that differ only by the names they define: the text rule and the `layout collapse` action."""

from __future__ import annotations

from pathlib import Path

import pytest

from unbake import collapse, folders, symbols
from unbake.contracts import Config, Refusal

VERSIONS = ["a", "b", "c"]
FACTS = {
    "a": {"D_A1": 0x100, "D_A2": 0x200, "other": 0x900},
    "b": {"D_B1": 0x110, "D_B2": 0x210},
    "c": {"D_C1": 0x120, "D_C2": 0x220},
}


def table(version: str, name: str, body: str = "{ 1, 2, 3 }") -> str:
    return f"#if defined(VERSION_{version})\ns32 {name}[] = {body};\n#endif\n"


def run(text: str, versions: list[str] = VERSIONS, facts: dict[str, dict[str, int]] = FACTS) -> collapse.Collapse:
    return collapse.collapse(text, versions, facts, "a")


@pytest.mark.parametrize(
    ("kind", "test", "want"),
    [
        ("if", "defined(VERSION_A)", {"a"}),
        ("if", "defined VERSION_B", {"b"}),
        ("if", "defined(VERSION_A) || defined(VERSION_C)", {"a", "c"}),
        ("if", "!defined(VERSION_A) && !defined(VERSION_C)", {"b"}),
        ("ifdef", "VERSION_C", {"c"}),
        ("ifndef", "VERSION_C", {"a", "b"}),
        ("if", "defined(VERSION_NOPE)", None),
        ("if", "defined(VERSION_A) + 1", None),
        ("if", "GAME", None),
        ("if", "defined(VERSION_A) &&", None),
    ],
)
def test_a_guard_is_the_set_of_versions_it_is_true_for(kind: str, test: str, want: set[str] | None) -> None:
    got = collapse.guard_versions(test, kind, VERSIONS)
    assert (set(got) if got is not None else None) == want


def test_version_tables_equal_once_the_names_are_set_aside_become_one_definition() -> None:
    result = run(table("A", "D_A1") + "\n" + table("B", "D_B1") + "\n" + table("C", "D_C1"))
    assert result.removed == 2 and not result.held
    assert result.text.count("s32 ") == 1 and "#if" not in result.text  # every version: no guard left
    assert "D_A1" in result.text  # the table of names_from stays as the name
    assert result.aliases == {"b": [("D_A1", "D_B1")], "c": [("D_A1", "D_C1")]}


def test_a_group_of_some_versions_keeps_a_guard_over_their_union() -> None:
    result = run(table("B", "D_B1") + "\n" + table("C", "D_C1") + "\ns32 plain = 4;\n")
    assert result.removed == 1
    assert "#if defined(VERSION_B) || defined(VERSION_C)" in result.text and "D_B1" in result.text
    assert result.text.endswith("s32 plain = 4;\n")
    assert result.aliases == {"c": [("D_B1", "D_C1")]}


def test_comments_and_braces_inside_strings_do_not_make_blocks_differ() -> None:
    a = '#if defined(VERSION_A)\ns32 D_A1[] = { 1 }; /* D_A1 at 80000100 */\nchar *s = "x;{";\n#endif\n'
    b = '#if defined(VERSION_B)\ns32 D_B1[] = { 1 }; // D_B1 at 80000110\nchar *s = "x;{";\n#endif\n'
    result = run(a + b, facts={"a": {"D_A1": 1, "s": 5}, "b": {"D_B1": 2, "s": 5}})
    assert result.removed == 1 and result.aliases == {"b": [("D_A1", "D_B1")]}  # `s` is the same name in both


def test_real_regional_content_stays_per_version() -> None:
    result = run(table("A", "D_A1", '{ "Start" }') + table("B", "D_B1", '{ "Anfang" }'))
    assert result.removed == 0 and not result.groups and result.text.count("#if") == 2


@pytest.mark.parametrize(
    "block",
    [
        "#if defined(VERSION_{v})\nvoid D_{v}1(void) {}\n#endif\n",  # code
        "#if defined(VERSION_{v})\nstatic s32 D_{v}1[] = { 1 };\n#endif\n",  # static
        "#if defined(VERSION_{v})\nextern s32 D_{v}1[];\n#endif\n",  # no initializer
        "#if defined(VERSION_{v})\ns32 D_{v}1[] = { 1 };\n#else\ns32 D_{v}1[] = { 2 };\n#endif\n",  # an #else
        "#if defined(VERSION_{v})\n#if X\ns32 D_{v}1[] = { 1 };\n#endif\n#endif\n",  # nested
        "#if GAME\ns32 D_{v}1[] = { 1 };\n#endif\n",  # not a version test
    ],
)
def test_only_plain_data_blocks_are_read(block: str) -> None:
    text = block.replace("{v}", "A") + block.replace("{v}", "B")
    assert collapse.blocks(text.split("\n"), VERSIONS) == []


def test_blocks_inside_braces_are_left_alone() -> None:
    text = "void f(void) {\n" + table("A", "D_A1") + table("B", "D_B1") + "}\n"
    assert run(text).text == text


def test_versions_that_overlap_are_held() -> None:
    result = run(
        table("A", "D_A1") + "#if defined(VERSION_A) || defined(VERSION_B)\ns32 D_B1[] = { 1, 2, 3 };\n#endif\n"
    )
    assert result.removed == 0 and [h["why"] for h in result.held] == ["versions overlap"]


def test_a_name_without_a_fact_is_held() -> None:
    result = run(table("A", "D_A1") + table("B", "D_NEW"))
    assert result.removed == 0 and result.held[0]["why"] == "D_NEW has no fact in b"


def test_a_chosen_name_that_is_another_address_in_a_version_is_held() -> None:
    result = run(table("A", "D_A1") + table("B", "D_B1"), facts={**FACTS, "b": {"D_B1": 0x110, "D_A1": 0x300}})
    assert result.removed == 0 and result.held[0]["why"] == "D_A1 is another address in b"


def test_other_data_of_a_version_between_the_blocks_holds_the_group() -> None:
    held = run(table("A", "D_A1") + "s32 other = 1;\n" + table("B", "D_B1")).held
    assert held[0]["why"] == "other data of a version stands between the blocks"
    # a block of a version the group does not serve does not stand in the way
    assert run(table("A", "D_A1") + table("C", "D_C1", "{ 9 }") + table("B", "D_B1")).removed == 1


def test_a_block_of_several_names_collapses_name_by_name_and_each_group_is_found_in_turn() -> None:
    def pair(v: str, one: str, two: str, body: str) -> str:
        return f"#if defined(VERSION_{v})\ns32 {one}[] = {{ 1 }};\ns32 {two}[] = {{ {body}, {one} }};\n#endif\n"

    result = run(pair("A", "D_A1", "D_A2", "2") + pair("B", "D_B1", "D_B2", "2"))
    assert result.removed == 1 and result.aliases == {"b": [("D_A1", "D_B1"), ("D_A2", "D_B2")]}
    two = run(table("A", "D_A1") + table("B", "D_B1") + table("A", "D_A2", "{ 5 }") + table("B", "D_B2", "{ 5 }"))
    assert two.removed == 2 and len(two.groups) == 2


def test_a_name_inside_a_string_literal_is_text_not_a_name() -> None:
    a = table("A", "D_A1", '{ "D_A1" }')
    b = table("B", "D_B1", '{ "D_B1" }')
    assert run(a + b).removed == 0


# ---------------------------------------------------------------- layout collapse
def _project(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = cfg.project.root
    (root / "src").mkdir(exist_ok=True)
    (root / "src/tbl.c").write_text(table("A", "D_A1") + "\n" + table("B", "D_B1"))
    (root / "src/plain.c").write_text("s32 x = 1;\n")
    monkeypatch.setattr(folders, "_tracked", lambda config: ["src/plain.c", "src/tbl.c"])
    rows = {"a": {"D_A1": (0x100, None, None, "data")}, "b": {"D_B1": (0x110, None, None, "data")}}
    monkeypatch.setattr(symbols, "load", lambda config, version: rows[version])
    return root


def test_layout_collapse_rewrites_the_file_and_gives_the_other_version_a_second_name(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(cfg, monkeypatch)
    result = folders.collapse(cfg, {})
    assert (result["groups"], result["blocks_removed"], result["written"]) == (1, 1, ["src/tbl.c"])
    text = (root / "src/tbl.c").read_text()
    assert text.count("s32 ") == 1 and "#if" not in text
    assert (root / symbols.aliases_path("b")).read_text() == "D_A1 = 0x00000110;\n"
    assert not (root / symbols.aliases_path("a")).exists()
    again = folders.collapse(cfg, {})
    assert again["groups"] == 0 and again["written"] == []  # nothing left to merge


def test_layout_collapse_can_be_limited_to_one_file_and_refuses_an_untracked_one(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(cfg, monkeypatch)
    assert folders.collapse(cfg, {"file": "src/plain.c"})["written"] == []
    assert (root / "src/tbl.c").read_text().count("#if") == 2
    with pytest.raises(Refusal):
        folders.collapse(cfg, {"file": "src/none.c"})
