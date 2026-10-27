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
