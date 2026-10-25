"""alloc: gcc 2.x allocation priorities, the flip arithmetic and the per-register report from hand-made dumps."""

from collections import Counter

import pytest

from unbake import alloc, symptoms

GREG = (
    ";; Function f\n\n;; 4 regs to allocate: 81 93 88 116 (2)\n;; Register dispositions:\n"
    "81 in 20  93 in 21  88 in 23  116 in 30  \n;; Hard regs used:  20 21 23 30\n"
    ";; Function g\n;; 1 regs to allocate: 5\n"
)
LREG = (
    ";; Function f\n"
    "Register 81 used 48 times across 882 insns; dies in 0 places.\n"
    "Register 93 used 39 times across 778 insns; dies in 0 places.\n"
    "Register 88 used 16 times across 404 insns; dies in 5 places.\n"
    "Register 116 used 51 times across 1756 insns; dies in 2 places.\n"
)
ASM = (
    '.stabs "f:F(0,1)",36,0,1,f\n.stabs "fmt:P(0,2)",64,0,1,20\n.stabs "dprec:r(0,1)",64,0,2,23\n'
    '.stabs "ulval:r(0,4)",64,0,3,21\n.stabs "g:F(0,1)",36,0,9,g\n.stabs "other:r(0,1)",64,0,9,20\n'
)


def test_priority_is_floored_log_of_the_references_times_them_over_the_live_length() -> None:
    assert alloc.priority(48, 882) == pytest.approx(0.27211, abs=1e-5)  # fmt: 5 * 48 / 882
    assert alloc.priority(39, 778) == pytest.approx(0.25064, abs=1e-5)  # ulval: 5 * 39 / 778
    assert alloc.priority(15, 404) < alloc.priority(16, 404)  # 8 occurrences in a loop weigh 16: log2 steps from 3 to 4
    assert alloc.priority(1, 5) == 0.0 and alloc.priority(8, 0) == 0.0


def test_flip_names_the_references_or_the_shorter_life_that_overtake() -> None:
    beat = alloc.priority(48, 882)
    assert alloc.flip(39, 778, beat) == {"more_refs": 4, "shorter_by": 62}
    assert alloc.priority(43, 778) > beat >= alloc.priority(42, 778) and alloc.priority(39, 716) > beat
    assert alloc.flip(39, 778, beat)["shorter_by"] == 778 - 716
    assert alloc.flip(1, 10, 0.5) == {"more_refs": 3, "shorter_by": None}  # one reference never gets there by length


def test_report_names_who_won_each_register_the_target_gives_to_another() -> None:
    votes = {"$s5": Counter({"$s4": 9, "$s5": 1}), "$s7": Counter({"$s7": 4}), "$fp": Counter({"$s7": 3})}
    rows = alloc.report("f", GREG, LREG, ASM, votes)
    by = {row["register"]: row for row in rows}
    assert set(by) == {"$s4", "$s7"}  # $s7 keeps its own votes; the stabs of function g never name f's registers
    s4 = by["$s4"]
    assert (s4["candidate"]["name"], s4["candidate"]["refs"], s4["candidate"]["length"]) == ("fmt", 48, 882)
    assert (s4["target"]["name"], s4["target"]["pseudo"]) == ("ulval", 93)
    assert s4["flip"] == {"more_refs": 4, "shorter_by": 62}
    s7 = by["$s7"]
    assert (s7["candidate"]["name"], s7["target"]["name"]) == ("dprec", "pseudo 116")
    assert s7["candidate"]["priority"] == pytest.approx(0.1584, abs=1e-4)
    assert s7["candidate"]["priority"] > s7["target"]["priority"]
    assert s7["flip"] == alloc.flip(51, 1756, alloc.priority(16, 404))


def test_report_names_a_minority_swap_and_a_register_the_candidate_leaves_free() -> None:
    # 153 aligned sites agree on $s4; one site of the target has $s6, a register no pseudo of f holds (was empty)
    rows = alloc.report("f", GREG, LREG, ASM, {"$s4": Counter({"$s4": 153, "$s6": 1})})
    assert len(rows) == 1 and rows[0]["votes"] == 1 and rows[0]["register"] == "$s6"
    assert (rows[0]["candidate"]["name"], rows[0]["candidate"]["pseudo"]) == ("free", None)
    assert (rows[0]["target"]["name"], rows[0]["flip"]) == ("fmt", None)
    votes = {"$s0": Counter({"$s0": 40, "$s1": 2}), "$s1": Counter({"$s0": 2, "$s1": 30})}
    assert alloc.report("f", GREG, LREG, ASM, votes) == []  # neither register is held by a pseudo of f


def test_report_names_pseudos_local_alloc_placed_and_flips_only_within_one_allocator() -> None:
    # pseudos 140 and 141 live in one block each: local-alloc placed them (f0 = 32, f2 = 34), global never ordered them
    greg = GREG.replace("116 in 30  \n", "116 in 30  140 in 32  \n")
    lreg = LREG + (
        "Register 140 used 6 times across 9 insns in block 2; set 1 time.\n"
        "Register 141 used 3 times across 4 insns in block 3; set 1 time.\n"
        ";; Register 141 in 34.\n"
    )
    (row,) = alloc.report("f", greg, lreg, ASM, {"$f0": Counter({"$f2": 3})})
    assert (row["register"], row["candidate"]["pseudo"], row["candidate"]["allocator"]) == ("$f2", 141, "local")
    assert (row["target"]["pseudo"], row["target"]["allocator"]) == (140, "local")
    assert row["flip"]["cause"] == "conflict"  # 140 outranks 141 already: the prediction names what lowers 141
    (mixed,) = alloc.report("f", greg, lreg, ASM, {"$f0": Counter({"$s4": 2})})  # global winner vs local pseudo
    assert (mixed["candidate"]["allocator"], mixed["target"]["allocator"], mixed["flip"]["cause"]) == (
        "global",
        "local",
        "allocator",
    )


LREG_RTL = (
    ";; Function f\n"
    '(note 1 0 2 ("f.c") 10)\n'
    "(insn 4 2 6 (set (reg/v:SI 81) (const_int 0)) -1 (nil))\n"
    "(note 5 4 6 0 NOTE_INSN_LOOP_BEG)\n"
    '(note 6 5 7 ("f.c") 12)\n'
    "(insn 7 6 8 (set (reg/v:SI 81) (plus:SI (reg/v:SI 81) (reg:SI 93))) -1 (expr_list:REG_DEAD (reg:SI 93) (nil)))\n"
    "(note 8 7 9 0 NOTE_INSN_LOOP_END)\n"
    '(note 9 8 10 ("f.c") 14)\n'
    "(insn 10 9 11 (set (reg:SI 2) (reg/v:SI 81)) -1 (nil))\n"
    ";; Function g\n(insn 3 2 4 (set (reg/v:SI 81) (const_int 1)) -1 (nil))\n"
)


def test_lines_weigh_each_reference_by_its_loop_depth_per_source_line() -> None:
    found = alloc.lines(LREG_RTL, "f", {81, 93})
    assert found[81] == [
        {"line": 10, "depth": 1, "refs": 1, "weighted": 1},
        {"line": 12, "depth": 2, "refs": 2, "weighted": 4},  # in the loop: each reference counts twice
        {"line": 14, "depth": 1, "refs": 1, "weighted": 1},
    ]
    assert found[93] == [{"line": 12, "depth": 2, "refs": 1, "weighted": 2}]  # the REG_DEAD note is no reference


def test_report_rows_carry_each_pseudos_references_by_line() -> None:
    votes = {"$s4": Counter({"$s5": 3})}
    (row,) = alloc.report("f", GREG, LREG + LREG_RTL.split("\n", 1)[1], ASM, votes)
    assert row["candidate"]["lines"] or row["target"]["lines"]
    assert all("weighted" in r for r in row["candidate"]["lines"] + row["target"]["lines"])


def test_every_register_row_proposes_one_c_edit() -> None:
    rows = alloc.report("f", GREG, LREG, ASM, {"$s5": Counter({"$s4": 9}), "$s4": Counter({"$s6": 2})})
    assert rows and all(row["edit"] for row in rows)
    free = next(row for row in rows if row["candidate"]["pseudo"] is None)
    assert "never uses $s6" in free["edit"] and "live across a call" in free["edit"]
    beaten = next(row for row in rows if row["flip"])
    assert "more time(s)" in beaten["edit"]


def _pseudo(reg: str, name: str, priority: float, lines: list[int]) -> dict:
    return {
        "register": reg,
        "pseudo": 7,
        "name": name,
        "priority": priority,
        "refs": 3,
        "length": 4,
        "allocator": "local",
        "lines": [{"line": n, "depth": 1, "refs": 1, "weighted": 1} for n in lines],
    }


def _row(have: str, want: str, held: dict, wanted_name: dict, votes: int) -> dict:
    return {"register": want, "candidate": held, "target": wanted_name, "votes": votes, "flip": None}


def test_rotations_name_three_temps_that_turn_through_three_registers_and_rank_the_precedent_first() -> None:
    index = _pseudo("$v1", "index", 2.0, [10, 12])
    base = _pseudo("$a0", "base", 1.0, [11])
    total = _pseudo("$v0", "total", 1.5, [13])
    rows = [  # the target keeps index in $a0, base in $v0, total in $v1
        _row("$v1", "$a0", base, index, 6),
        _row("$a0", "$v0", total, base, 6),
        _row("$v0", "$v1", index, total, 4),
    ]
    (turn,) = alloc.rotations(rows, lambda line: f"f.c:{line}")
    assert turn["kind"] == "rotation of 3" and sorted(turn["registers"]) == ["$a0", "$v0", "$v1"]
    assert sorted(turn["temps"]) == ["base", "index", "total"]
    assert turn["priority_order"] == ["index 2.0", "total 1.5", "base 1.0"]
    assert turn["needs"] and "outranks" in turn["needs"][0]
    top = turn["rewrites"][0]
    assert top["kind"] == "fold-temp" and top["precedent"] == "func_8024ADD0_de" and top["gain"] == 0.0299
    assert turn["rewrites"][-1]["kind"] == "reorder" and turn["rewrites"][-1]["score"] < top["score"]


def test_rotations_see_a_swap_of_float_temps_and_prefer_the_precedent_of_the_same_size() -> None:
    a, b = _pseudo("$f0", "pseudo 88", 1.0, [5]), _pseudo("$f1", "pseudo 89", 1.0, [6])
    rows = [_row("$f1", "$f0", a, b, 3), _row("$f0", "$f1", b, a, 4)]
    (turn,) = alloc.rotations(rows)
    assert turn["kind"] == "swap" and turn["rewrites"][0]["precedent"] == "func_8024ADD0_de"


def test_rotations_ignore_a_register_the_target_does_not_return_to_the_chain() -> None:
    a, b = _pseudo("$v0", "a", 1.0, [1]), _pseudo("$v1", "b", 1.0, [2])
    assert alloc.rotations([_row("$v1", "$v0", a, b, 3)]) == []


TGREG = (
    ";; Function f\n\n;; 3 regs to allocate: 85 81 90\n;; 81 conflicts: 81 85 90 2\n;; 85 conflicts: 81 85 90\n"
    ";; 90 conflicts: 81 85 90\n\n;; Register dispositions:\n"
    "81 in 20  85 in 19  90 in 21  \n;; Hard regs used: 19 20 21\n"
)
TLREG = (
    ";; Function f\n"
    '(note 1 0 2 ("f.c") 10)\n'
    "(insn 4 2 6 (set (reg/v:SI 90) (const_int 0)) -1 (nil))\n"
    '(note 5 4 6 ("f.c") 11)\n'
    "(insn 7 6 8 (set (reg:SI 85) (mult:SI (reg/v:SI 90) (const_int 1180))) -1 (nil))\n"
    '(note 8 7 9 ("f.c") 12)\n'
    '(insn 9 8 10 (set (reg/v:SI 81) (symbol_ref:SI ("D_1"))) -1 (nil))\n'
    "Register 85 used 9 times across 105 insns; dies in 0 places.\n"
    "Register 81 used 34 times across 678 insns; dies in 0 places.\n"
    "Register 90 used 8 times across 108 insns; dies in 0 places.\n"
)
SOURCE = {10: "n = 0;", 11: "giv = n * 1180;", 12: "res = &D_1;"}


def test_table_names_each_pseudo_and_explains_who_took_the_register_the_target_uses() -> None:
    votes = {85: Counter({"$s4": 5}), 81: Counter({"$s5": 7}), 90: Counter({"$s3": 4})}
    rows = {r["pseudo"]: r for r in alloc.table("f", TGREG, TLREG, votes, lambda n: f"f.c:{n}  `{SOURCE[n]}`")}
    n, giv, res = rows[90], rows[85], rows[81]
    assert [r["pseudo"] for r in alloc.table("f", TGREG, TLREG, votes)] == [85, 81, 90]  # the allocation order
    assert (n["name"], n["refs"], n["length"], n["priority"], n["rank"]) == ("n", 8, 108, 0.2222, 2)
    assert (giv["priority"], res["priority"]) == (0.2571, 0.2507)  # the three priorities of func_8040FB94_de
    assert (n["register"], n["target_register"], n["mismatch"]) == ("$s5", "$s3", True)
    assert [(t["name"], t["earlier"]) for t in n["took"]] == [
        ("tmp@11", True)
    ]  # the giv holds $s3 and was placed first
    assert n["delta"]["more_refs"] == 2 and n["delta"]["above"] == 0.2571  # 8 refs need 10 to pass the giv
    assert "taken first by tmp@11" in n["why"] and n["rom_gives"] == ["$s5->81"]
    assert res["took"][0]["pseudo"] == 90 and res["took"][0]["earlier"] is False  # n sits in $s5, placed after res
    assert not any(r["mismatch"] for r in alloc.table("f", TGREG, TLREG, {85: Counter({"$s3": 5})}))


def test_table_flags_one_site_that_differs_and_leaves_an_unvoted_pseudo_unknown() -> None:
    votes = {90: Counter({"$s5": 9, "$s6": 1})}
    rows = {r["pseudo"]: r for r in alloc.table("f", TGREG, TLREG, votes)}
    assert (rows[90]["target_register"], rows[90]["mismatch"], rows[90]["support"]) == ("$s5/$s6", True, "$s5x9 $s6x1")
    assert "not blocked" in rows[90]["why"]  # no pseudo of f holds $s6
    assert (rows[81]["target_register"], rows[81]["mismatch"]) == (None, False)


def test_pseudo_votes_pin_a_register_to_the_pseudo_whose_instruction_names_it() -> None:
    lreg = (
        ";; Function f\n(insn 6 4 8 (set (reg/v:SF 81) (reg:SF 5 a1)) 192 {m} (nil))\n"
        "(insn 9 8 10 (set (reg:SI 82) (reg:SI 4 a0)) 1 (nil))\n"
    )
    final = (
        ";; Function f\n(insn 9 4 6 (set (reg:SI 3 v1) (reg:SI 4 a0)) 1 (nil))\n"
        "(insn 6 9 8 (set (reg/v:SF 36 $f4) (reg:SF 5 a1)) 192 {m} (nil)\n"
        "    (expr_list:REG_DEAD (reg:SF 5 a1) (nil)))\n"
    )
    named = [["$v1", "$a0"], ["$a1", "$f4"]]  # the built code in final order
    pairs = {0: [("$v1", "$v1"), ("$a0", "$a0")], 1: [("$a1", "$a1"), ("$f4", "$f5")]}
    votes = alloc.pseudo_votes("f", lreg, final, {81: 36, 82: 3}, named, pairs)
    assert votes == {81: Counter({"$f5": 1}), 82: Counter({"$v1": 1})}


def test_operand_pairs_give_the_target_register_at_each_aligned_site() -> None:
    def mtc1(fs: int) -> bytes:  # mtc1 $a1, $fN
        return (0x44850000 | fs << 11).to_bytes(4, "big")

    named, pairs = symptoms.operand_pairs(mtc1(4), mtc1(5))
    assert named == [["$a1", "$f4"]] and pairs == {0: [("$a1", "$a1"), ("$f4", "$f5")]}


def test_a_register_shared_by_variables_is_unknown_not_the_first_one():
    assert alloc._label(["fmt"], 3) == "fmt"
    assert alloc._label(["fmt", "fmt"], 3) == "fmt"
    assert alloc._label(None, 3) == "pseudo 3"
    named = alloc._label(["previous", "table"], 7)
    assert named.startswith("unknown pseudo 7") and "previous/table" in named
