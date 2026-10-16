"""alloc: gcc 2.x allocation priorities, the flip arithmetic and the per-register report from hand-made dumps."""

from collections import Counter

import pytest

from unbake import alloc

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
    assert row["flip"] is None  # the target's 140 outranks 141 already: nothing to lift
    (mixed,) = alloc.report("f", greg, lreg, ASM, {"$f0": Counter({"$s4": 2})})  # global winner vs local pseudo
    assert (mixed["candidate"]["allocator"], mixed["target"]["allocator"], mixed["flip"]) == ("global", "local", None)
