"""Which merges must stay separate files: the pure rule over plain link-order facts."""

from __future__ import annotations

from typing import Any

import pytest

from unbake import mergecheck


def entry(unit: str, cls: str, size: int, out: str = ".text") -> dict[str, Any]:
    return {"o": out, "k": "entry", "u": unit, "cls": cls, "n": size}


def align(n: int, out: str = ".rodata") -> dict[str, Any]:
    return {"o": out, "k": "align", "a": n}


def run(rows, seq, statics=None, identity=lambda unit: "cc"):
    return {v["module"]: v for v in mergecheck.verdicts(rows, {"us": seq}, statics or {}, identity)}


def test_adjacent_units_in_map_order_merge() -> None:
    rows = [("a", "M"), ("b", "M"), ("c", "N")]
    seq = [entry("a", "text", 8), entry("b", "text", 8), entry("c", "text", 8)]
    assert run(rows, seq) == {"M": {"module": "M", "units": 2, "verdict": "merge", "hard": [], "soft": []}}


def test_another_modules_bytes_between_two_units_keep_them_separate() -> None:
    rows = [("a", "M"), ("x", "N"), ("b", "M")]
    seq = [entry("a", "text", 8), entry("x", "text", 4), entry("b", "text", 8)]
    out = run(rows, seq)["M"]
    assert out["verdict"] == "separate" and out["hard"] == ["gap: us text a .. b has x between"]


def test_an_empty_entry_between_does_not_break_adjacency() -> None:
    rows = [("a", "M"), ("x", "N"), ("b", "M")]
    seq = [entry("a", "text", 8), entry("x", "text", 0), entry("b", "text", 8)]
    assert run(rows, seq)["M"]["verdict"] == "merge"


def test_units_linked_against_the_map_order_or_with_other_flags_stay_separate() -> None:
    rows = [("a", "M"), ("b", "M")]
    seq = [entry("b", "text", 8), entry("a", "text", 8)]
    assert run(rows, seq)["M"]["hard"] == ["order: us text b is linked before a"]
    out = run(rows, [entry("a", "text", 8), entry("b", "text", 8)], identity=lambda unit: unit)["M"]
    assert out["verdict"] == "separate" and out["hard"][0].startswith("flags:")


def test_a_class_in_two_output_sections_stays_separate() -> None:
    rows = [("a", "M"), ("b", "M")]
    seq = [entry("a", "text", 8, ".text"), entry("b", "text", 8, ".boot")]
    assert run(rows, seq)["M"]["hard"] == ["sections: us text sits in several output sections"]


def test_padding_statics_and_literals_are_soft_findings() -> None:
    rows = [("a", "M"), ("b", "M")]
    seq = [entry("a", "rodata", 4, ".rodata"), align(8), entry("b", "rodata", 4, ".rodata")]
    out = run(rows, seq, {"a": {"helper"}, "b": {"helper", "other"}})["M"]
    assert out["verdict"] == "merge"
    assert out["soft"] == [
        "align: us rodata ALIGN(8) before b",
        "literals: us rodata in 2 units",
        "statics: helper is local to a and b",
    ]


def test_single_unit_modules_have_no_verdict() -> None:
    assert run([("a", "M"), ("b", "N")], [entry("a", "text", 8)]) == {}


LINK = """\
SECTIONS {
 .text 0x80000400 : AT(0x400) SUBALIGN(1)
 {
  src/a.o(.text);
  . = ALIGN(8);
  src/b.o(.text .data);
 }
 .rodata : AT(0x10)
 {
  src/a.o(.rodata .rdata);
 }
}
"""


def test_the_link_script_rows_keep_their_order_output_section_and_classes() -> None:
    rows = mergecheck.parse_link(LINK)
    assert [(r["k"], r.get("u"), r["o"]) for r in rows] == [
        ("entry", "src/a", ".text"),
        ("align", None, ".text"),
        ("entry", "src/b", ".text"),
        ("entry", "src/a", ".rodata"),
    ]
    assert rows[2]["cls"] == ("data", "text") and rows[3]["cls"] == ("rodata",)


def test_a_module_map_needs_unit_and_module_columns() -> None:
    assert mergecheck.read_modules("module\tunit\tx\nM\ta\t1\n# c\n\nM\tb\t2\n") == [("a", "M"), ("b", "M")]
    with pytest.raises(ValueError):
        mergecheck.read_modules("a\tb\n1\t2\n")


def piece(size: int, align: int) -> list[tuple[int, int]]:
    return [(size, align)]


def with_parts(row: dict[str, Any], **parts: list[tuple[int, int]]) -> dict[str, Any]:
    return {**row, "parts": parts}


def test_pieces_that_pack_longer_in_one_object_are_a_hard_lma_finding() -> None:
    rows = [("a", "M"), ("b", "M")]
    seq = [
        with_parts(entry("a", "rodata", 2, ".rodata"), rodata=piece(1, 1), data=piece(1, 1)),
        with_parts(entry("b", "rodata", 4, ".rodata"), rodata=piece(4, 4)),
    ]
    out = run(rows, seq)["M"]
    assert out["verdict"] == "separate"
    assert out["hard"] == ["lma: us .rodata merged 9 B against linked 8 B: overlaps the next section"]


def test_an_align_directive_the_merged_object_lacks_is_a_hard_lma_gap() -> None:
    rows = [("a", "M"), ("b", "M")]
    seq = [
        with_parts(entry("a", "rodata", 4, ".rodata"), rodata=piece(4, 4)),
        align(8),
        with_parts(entry("b", "rodata", 4, ".rodata"), rodata=piece(4, 4)),
    ]
    gap = "lma: us .rodata merged 8 B against linked 12 B: leaves a gap before the next section"
    assert run(rows, seq)["M"]["hard"] == [gap]


def test_pieces_that_pack_the_same_merge() -> None:
    rows = [("a", "M"), ("b", "M")]
    seq = [
        with_parts(entry("a", "rodata", 8, ".rodata"), rodata=piece(8, 8)),
        with_parts(entry("b", "rodata", 8, ".rodata"), rodata=piece(8, 8)),
    ]
    assert run(rows, seq)["M"]["verdict"] == "merge"


def test_spans_count_align_directives_and_other_modules_between_the_first_and_last_unit() -> None:
    seq = [
        with_parts(entry("a", "rodata", 4, ".rodata"), rodata=piece(4, 4)),
        align(8),
        with_parts(entry("x", "rodata", 4, ".rodata"), rodata=piece(4, 4)),
        with_parts(entry("b", "rodata", 4, ".rodata"), rodata=piece(4, 4)),
    ]
    assert mergecheck.spans(seq, {"a", "b"}) == [(".rodata", 16, 8)]


def test_units_are_ordered_by_where_their_code_is_linked_not_their_first_data() -> None:
    seq = [entry("a", "rodata", 8, ".rodata"), entry("b", "text", 8), entry("a", "text", 8), entry("c", "data", 8)]
    assert mergecheck.code_order(["a", "b", "c"], {"us": seq}) == ["b", "a", "c"]


def test_merged_units_leave_the_module_map_as_one_row_of_the_merged_file() -> None:
    text = "unit\tmodule\tkind\nA/a.CPP\tA/M.CPP\tcode\nB/b.CPP\tB/N.CPP\tcode\nA/a_2.CPP\tA/M.CPP\tdata\n#c\n"
    got = mergecheck.merged_modules(text, ["A/a.CPP", "A/a_2.CPP"], "A/M/M.CPP")
    assert got == "unit\tmodule\tkind\nA/M/M.CPP\tA/M.CPP\tcode\nB/b.CPP\tB/N.CPP\tcode\n#c\n"
    assert mergecheck.merged_modules(got, ["A/M/M.CPP"], "A/M/M.CPP") == got


def test_a_merged_file_sits_in_the_module_folder_under_the_module_name() -> None:
    assert mergecheck.merged_path("src/game/AI/NAV/PICKUP.CPP") == "src/game/AI/NAV/PICKUP/PICKUP.CPP"
