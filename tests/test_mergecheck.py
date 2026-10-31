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


def test_units_linked_in_another_order_than_the_map_merge_class_wise_but_other_flags_stay_separate() -> None:
    rows = [("a", "M"), ("b", "M")]
    seq = [entry("b", "text", 8), entry("a", "text", 8)]
    assert run(rows, seq)["M"]["hard"] == []
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


def test_unplaced_audio_microcode_cannot_be_activated_by_merging_into_linked_data():
    base = "src/game/ENGINE/AUDIO/AUDIOMGR/AUDIOMGR_INIT/"
    units = [base + f"AUDIOMGR_INIT_{n}.CPP" for n in (1, 2, 3)]
    rows = [(u, "audio") for u in units]
    seqs = {v: [entry(units[1 if v == "us-rev1" else 0], "data", 8)] for v in ("us", "us-rev1", "de", "eu", "eu-x")}
    result = mergecheck.verdicts(rows, seqs, {}, lambda _: "cc")[0]
    assert result["hard"] == [f"unplaced: {units[2]} has no linker input in any version"]
    assert result["verdict"] == "separate"
    assert mergecheck.verdicts(rows[:2], seqs, {}, lambda _: "cc")[0]["hard"] == []


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


def verdict_of(rows, seqs, module="M"):
    return {v["module"]: v for v in mergecheck.verdicts(rows, seqs, {}, lambda unit: "cc")}[module]


def test_a_class_linked_in_another_unit_order_in_another_version_is_a_soft_order_finding() -> None:
    rows = [("a", "M"), ("b", "M")]
    seqs = {"us": [entry("a", "text", 8), entry("b", "text", 8)], "eu": [entry("b", "text", 8), entry("a", "text", 8)]}
    out = verdict_of(rows, seqs)
    assert out["verdict"] == "merge"
    assert out["hard"] == []
    assert out["soft"] == ["order: eu text links b elsewhere than the other versions"]


def test_each_class_may_be_linked_in_its_own_unit_order() -> None:
    rows = [("a", "M"), ("b", "M")]
    seq = [entry("a", "text", 8), entry("b", "text", 8), entry("b", "data", 4, ".data"), entry("a", "data", 4, ".data")]
    assert run(rows, seq)["M"]["hard"] == []
    assert mergecheck.class_orders(["a", "b"], {"us": seq}) == {"text": {"us": ["a", "b"]}, "data": {"us": ["b", "a"]}}


def test_a_version_that_splits_a_class_over_sections_still_reports_its_order() -> None:
    rows = [("a", "M"), ("b", "M")]
    seqs = {
        "us": [entry("a", "text", 8), entry("b", "text", 8)],
        "eu": [entry("b", "text", 8), entry("a", "text", 8, ".other")],
    }
    out = verdict_of(rows, seqs)
    assert out["hard"] == ["sections: eu text sits in several output sections"]
    assert out["soft"] == ["order: eu text links b elsewhere than the other versions"]


def test_a_unit_only_a_later_version_places_is_ordered_in_its_own_sequence_not_by_an_index_of_another() -> None:
    others = [entry(f"k{i}", "text", 8) for i in range(3)]
    seqs = {
        "us": [entry("a", "text", 8), *others, entry("b", "text", 8)],
        "eu": [entry("a", "text", 8), entry("b", "text", 8), entry("n", "text", 8)],
    }
    assert mergecheck.code_order(["a", "b", "n"], seqs) == ["a", "b", "n"]
    assert verdict_of([("a", "M"), ("b", "M"), ("n", "M")], seqs)["soft"] == []  # no order finding either


def test_a_unit_that_one_version_links_two_places_earlier_is_the_only_moved_unit() -> None:
    # the real rows: ACHIEVE_CREATERESULTSSCREEN.CPP is unit 10 in one version and 8 in another
    names = [f"u{i:02d}" for i in range(12)]
    shifted = [*names[:8], names[10], *names[8:10], names[11]]
    seqs = {
        "de": [entry(u, "rodata", 4, ".rodata") for u in names],
        "eu": [entry(u, "rodata", 4, ".rodata") for u in shifted],
    }
    out = verdict_of([(u, "M") for u in names], seqs)
    assert out["hard"] == []
    assert [r for r in out["soft"] if r.startswith("order")] == [
        "order: eu rodata links u10 elsewhere than the other versions"
    ]


def test_an_align_directive_the_merged_object_lacks_is_a_soft_fill() -> None:
    rows = [("a", "M"), ("b", "M")]
    seq = [
        with_parts(entry("a", "rodata", 4, ".rodata"), rodata=piece(4, 4)),
        align(8),
        with_parts(entry("b", "rodata", 4, ".rodata"), rodata=piece(4, 4)),
    ]
    out = run(rows, seq)["M"]
    assert out["hard"] == []
    assert "fill: us .rodata merged 8 B against linked 12 B: the link fill of the 4 B is regenerated" in out["soft"]


def test_pieces_that_pack_the_same_merge() -> None:
    rows = [("a", "M"), ("b", "M")]
    seq = [
        with_parts(entry("a", "rodata", 8, ".rodata"), rodata=piece(8, 8)),
        with_parts(entry("b", "rodata", 8, ".rodata"), rodata=piece(8, 8)),
    ]
    assert run(rows, seq)["M"]["verdict"] == "merge"


def test_spans_count_align_directives_but_not_the_bytes_of_other_modules_between_units() -> None:
    seq = [
        with_parts(entry("a", "rodata", 4, ".rodata"), rodata=piece(4, 4)),
        align(8),
        with_parts(entry("x", "rodata", 4, ".rodata"), rodata=piece(4, 4)),
        with_parts(entry("b", "rodata", 4, ".rodata"), rodata=piece(4, 4)),
    ]
    assert mergecheck.spans(seq, {"a", "b"}) == [(".rodata", 12, 8, 0)]


def test_units_are_ordered_by_where_their_code_is_linked_not_their_first_data() -> None:
    seq = [entry("a", "rodata", 8, ".rodata"), entry("b", "text", 8), entry("a", "text", 8), entry("c", "data", 8)]
    assert mergecheck.code_order(["a", "b", "c"], {"us": seq}) == ["b", "a", "c"]


def test_merged_units_leave_the_module_map_as_one_row_of_the_merged_file() -> None:
    text = "unit\tmodule\tkind\nA/a.CPP\tA/M.CPP\tcode\nB/b.CPP\tB/N.CPP\tcode\nA/a_2.CPP\tA/M.CPP\tdata\n#c\n"
    got = mergecheck.merged_modules(text, {"A/a.CPP": "A/M/M.CPP", "A/a_2.CPP": "A/M/M.CPP"})
    assert got == "unit\tmodule\tkind\nA/M/M.CPP\tA/M.CPP\tcode\nB/b.CPP\tB/N.CPP\tcode\n#c\n"
    assert mergecheck.merged_modules(got, {"A/M/M.CPP": "A/M/M.CPP"}) == got


def test_a_merged_file_sits_in_the_module_folder_under_the_module_name() -> None:
    assert mergecheck.merged_path("src/game/AI/NAV/PICKUP.CPP") == "src/game/AI/NAV/PICKUP/PICKUP.CPP"


def test_object_facts_read_the_elf_once_and_keep_sizes_alignments_and_static_names(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    class Section(dict):
        def __init__(self, name, size, alignment):
            super().__init__(sh_size=size, sh_addralign=alignment)
            self.name = name

    class Symbol(dict):
        def __init__(self, name, bind, kind):
            super().__init__(st_info={"bind": bind, "type": kind}, st_shndx=1)
            self.name = name

    class Table:
        def iter_symbols(self):
            return iter(
                [
                    Symbol("helper", "STB_LOCAL", "STT_FUNC"),
                    Symbol("state", "STB_LOCAL", "STT_OBJECT"),
                    Symbol("exported", "STB_GLOBAL", "STT_FUNC"),
                    Symbol("label", "STB_LOCAL", "STT_NOTYPE"),
                ]
            )

    sections = [Section(".text", 24, 4), Section(".data", 8, 0), Section(".rdata", 4, 8), Section(".bss", 0, 4)]
    elf = SimpleNamespace(iter_sections=lambda: iter(sections), get_section_by_name=lambda _: Table())
    read = Mock(return_value=elf)
    monkeypatch.setattr(mergecheck, "ELFFile", read)
    monkeypatch.setattr(mergecheck, "SymbolTableSection", Table)
    path = tmp_path / "object.o"
    path.write_bytes(b"fixture")
    assert mergecheck.object_facts(path) == (
        {".text": 24, ".data": 8, ".rdata": 4, ".bss": 0},
        {"helper", "state"},
        {".text": [(24, 4)], ".data": [(8, 1)], ".rdata": [(4, 8)]},
        {"helper", "state", "exported", "label"},
    )
    read.assert_called_once()


def test_fact_scan_reuses_one_read_for_multiple_sections_of_same_object(tmp_path, monkeypatch):
    from unittest.mock import Mock

    (tmp_path / "us.ld").write_text(LINK)
    for name in ("a", "b"):
        path = tmp_path / f"src/{name}.o"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture")
    read = Mock(return_value=({".text": 8, ".data": 4, ".rodata": 4}, {"local"}, {".rodata": [(4, 4)]}, set()))
    monkeypatch.setattr(mergecheck, "object_facts", read)
    seqs, statics = mergecheck.facts_of(tmp_path, ["us"], "{version}.ld", "{unit}.o")
    assert read.call_count == 2
    assert statics == {"src/a": {"local"}, "src/b": {"local"}}
    assert [e["n"] for e in seqs["us"] if e["k"] == "entry"] == [8, 12, 4]


def test_original_linker_minimum_offsets_keep_internal_holes_in_real_data_slice():
    # Six original US-rev1 table fragments: a 32-byte hole after the second piece is
    # a linker constraint, not source padding that disappears safely in one object.
    offsets = [0x923C, 0x9248, 0x9274, 0x9280, 0x929C, 0x92A8]
    sizes = [12, 12, 12, 28, 12, 12]
    text = ".data_800C 0x800C6AB0 : AT(0xC76B0) SUBALIGN(1)\n{\n"
    text += "\n".join(f". = MAX(., 0x{o:X});\nu{i}.o(.data);" for i, o in enumerate(offsets))
    seq = mergecheck.parse_link(text + "\n}")
    for row in seq:
        if row["k"] == "entry":
            size = sizes[int(row["u"][1:])]
            row.update(n=size, cls="data", parts={"data": [(size, 4)]})
    rows = [(f"u{i}", "tables") for i in range(6)]
    assert mergecheck.spans(seq, {u for u, _ in rows}) == [(".data_800C", 120, 88, 0x923C)]
    out = run(rows, seq)["tables"]
    assert out["hard"] == []
    assert (
        "fill: us .data_800C merged 88 B against linked 120 B: the link fill of the 32 B is regenerated" in out["soft"]
    )


def test_minimum_does_not_move_back_or_count_prefix_space_as_module_bytes():
    seq = [
        {"o": ".data", "k": "minimum", "offset": 100},
        with_parts(entry("a", "data", 8, ".data"), data=piece(8, 1)),
        {"o": ".data", "k": "minimum", "offset": 104},
        with_parts(entry("b", "data", 8, ".data"), data=piece(8, 1)),
    ]
    assert mergecheck.spans(seq, {"a", "b"}) == [(".data", 16, 16, 100)]


def test_named_sections_are_counted_once_and_conditional_align_uses_actual_definitions(tmp_path, monkeypatch):
    from unittest.mock import Mock

    (tmp_path / "us.ld").write_text(
        ".rodata : SUBALIGN(1)\n{\na.o(.rodata.first);\n"
        ". = DEFINED(yes) ? ALIGN(8) : .;\na.o(.rodata.second);\n"
        ". = DEFINED(no) ? ALIGN(8) : .;\nb.o(.rodata);\n}\n"
    )
    for unit in ("a", "b"):
        (tmp_path / f"{unit}.o").write_bytes(b"fixture")
    read = Mock(
        side_effect=[
            (
                {".rodata.first": 3, ".rodata.second": 4},
                set(),
                {".rodata.first": [(3, 4)], ".rodata.second": [(4, 4)]},
                {"yes"},
            ),
            ({".rodata": 4}, set(), {".rodata": [(4, 4)]}, set()),
        ]
    )
    monkeypatch.setattr(mergecheck, "object_facts", read)
    seqs, _ = mergecheck.facts_of(tmp_path, ["us"], "{version}.ld", "{unit}.o")
    seq = seqs["us"]
    assert [r["n"] for r in seq if r["k"] == "entry"] == [3, 4, 4]
    assert [r["symbol"] for r in seq if "symbol" in r] == ["yes"]
    assert mergecheck.spans(seq, {"a", "b"}) == [(".rodata", 16, 11, 0)]
    assert read.call_count == 2


def test_missing_selected_object_is_not_an_empty_safe_member(tmp_path):
    (tmp_path / "us.ld").write_text(".text :\n{\na.o(.text);\nb.o(.text);\n}\n")
    seqs, statics = mergecheck.facts_of(tmp_path, ["us"], "{version}.ld", "{unit}.o")
    result = mergecheck.verdicts([("a", "M"), ("b", "M")], seqs, statics, lambda _: "cc")[0]
    assert result["verdict"] == "separate"
    assert result["hard"] == ["object: us a has no built object", "object: us b has no built object"]


def _linked(order: list[str], cls: str) -> list[dict[str, Any]]:
    return [{"o": ".game_data", "k": "entry", "u": u, "cls": cls, "n": 4} for u in order]


def _filled(size: int, unit: str, at: int) -> list[str]:
    names = [f"u{i}" for i in range(size - 1)]
    names.insert(at, unit)
    return names


@pytest.mark.parametrize(
    ("cls", "unit", "base_at", "moved_at", "size"),
    [
        ("rodata", "ACHIEVE_CREATERESULTSSCREEN.CPP", 10, 8, 40),  # PAKINTERFACESCREEN eu and eu-x
        ("text", "NODEPAIR.CPP", 63, 54, 151),  # PAKINTERFACESCREEN eu-x
        ("text", "FRONTEND_SELECTNEXTJOINMODEL.CPP", 0, 3, 30),  # src/MENU.CPP us
    ],
)
def test_order_real_rows_one_unit_sits_elsewhere_in_one_version(
    cls: str, unit: str, base_at: int, moved_at: int, size: int
) -> None:
    base = _filled(size, unit, base_at)
    other = [u for u in base if u != unit]
    other.insert(moved_at, unit)
    found = mergecheck.verdicts(
        [(u, "M") for u in base], {"de": _linked(base, cls), "eu": _linked(other, cls)}, {}, lambda u: "cc"
    )
    assert found[0]["hard"] == []
    assert [r for r in found[0]["soft"] if r.startswith("order")] == [
        f"order: eu {cls} links {unit} elsewhere than the other versions"
    ]


def test_order_real_rows_us_rev1_swaps_the_two_common_units_of_volume() -> None:
    # src/ENGINE/VOLUME.CPP us-rev1 rodata: .game_data_E1720 links the two common units the other way round
    base = ["VOLUME_A.CPP", "VOLUME_B.CPP"]
    found = mergecheck.verdicts(
        [(u, "M") for u in base],
        {"de": _linked(base, "rodata"), "us-rev1": _linked(base[::-1], "rodata")},
        {},
        lambda u: "cc",
    )
    assert found[0]["hard"] == []
    assert len([r for r in found[0]["soft"] if r.startswith("order")]) == 1


def test_order_real_rows_a_unit_only_the_later_version_places_follows_its_neighbour_in_that_version() -> None:
    # PAKINTERFACESCREEN us-rev1 rodata holds 55 units, de 43: the units only us-rev1 places are not ordered by de
    de = [f"u{i}" for i in range(43)]
    us = [*de[:27], "extra1", "extra2", *de[27:]]
    assert mergecheck.code_order([*us], {"de": _linked(de, "rodata"), "us-rev1": _linked(us, "rodata")}, "rodata") == us
