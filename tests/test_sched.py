"""sched: the scheduler dumps of gcc 2.x read into per-insn priority, dependencies, pick and the move's reason."""

from unbake import sched

DUMP = """;; Function f

;;\t -- basic block number 0 from 5 to 9 --
;; ready list initially:
;; 9

;; insn[   5]: priority =    1, ref_count =    2
;; insn[   7]: priority =    5, ref_count =    1
;; insn[   8]: priority =    5, ref_count =    1
;; insn[   9]: priority =    1, ref_count =    1
;; ready list at T-1: 9 (1), now 9
;; ready list at T-2: 8 (5) 7 (5), now 8 7
;; insn 8 has a greater potential hazard, now 8 7
;; ready list at T-3: 7 (5) 6 (1), now 7 6
;; ready list at T-4: 5 (1), now 5

(note 1 0 5 ("f.c") 10)

(insn 5 1 7 (set (reg:SI 2 v0)
        (const_int 1)) 159 {movsi_internal2} (nil)
    (nil))

(note 6 5 7 ("f.c") 11)

(insn 7 5 8 (set (mem:SI (plus:SI (reg:SI 2 v0)
                (const_int 44)))
        (const_int 0)) 159 {movsi_internal2} (insn_list 5 (insn_list:REG_DEP_ANTI 3 (nil)))
    (nil))

(note 9 7 8 ("f.c") 12)

(insn 8 7 9 (set (mem:SI (plus:SI (reg:SI 2 v0)
                (const_int 48)))
        (reg:SI 22 s6)) 159 {movsi_internal2} (insn_list:REG_DEP_OUTPUT 7 (nil))
    (nil))
"""


def test_memory_keys_of_asm_lines_and_rtl_agree() -> None:
    assert sched.memory_key("sw $zero, 0x2C($v0)") == ("s", 0, 44, 2)
    assert sched.memory_key("lw $s6, 0x30($v0)") == ("l", 22, 48, 2)
    assert sched.memory_key("addiu $v0, $v0, 8") is None
    dump = sched.Pass(DUMP, "f")
    assert dump.key == {7: ("s", 0, 44, 2), 8: ("s", 22, 48, 2)}


def test_a_pass_reads_priorities_input_positions_lines_and_dependencies() -> None:
    dump = sched.Pass(DUMP, "f")
    assert dump.priority[7] == (5, 1) and dump.luid[7] == 1 and dump.luid[9] == 3
    assert dump.order == [5, 7, 8] and dump.line == {5: 10, 7: 11, 8: 12}
    assert dump.deps[7] == [(5, "true"), (3, "anti")] and dump.deps[8] == [(7, "output")]


def test_the_pick_names_who_went_before_and_after_with_their_priorities() -> None:
    pick = sched.Pass(DUMP, "f").picked(7)
    assert pick == {
        "at": "T-3",
        "before": [],
        "after": [{"insn": 6, "line": None, "priority": 1, "luid": None}],
        "priority": 5,
    }
    tie = sched.Pass(DUMP, "f").picked(7)
    assert tie is not None and sched.Pass(DUMP, "f").picked(99) is None


def test_moved_pairs_the_lines_the_diff_shows_on_both_sides() -> None:
    rows = ["@@ -1 +1 @@", "+800BAEA0: sw $zero, 0x2C($v0)", "-800BAEAC: sw $zero, 0x2C($v0)", "-800BAEB0: nop"]
    assert sched.moved(rows) == [("sw $zero, 0x2C($v0)", "800BAEA0", "800BAEAC")]


def test_report_rows_name_the_insn_the_line_and_what_decided_it() -> None:
    listing = ["800BAE98: addiu $v0, $v0, 8", "800BAE9C: sw $zero, 0x2C($v0)", "800BAEA0: sw $s6, 0x30($v0)"]
    diff = ["+800BAE9C: sw $zero, 0x2C($v0)", "-800BAE98: sw $zero, 0x2C($v0)"]
    rows = sched.report("f", {"sched2": DUMP}, listing, diff)
    assert [(r["insn"], r["line"], r["words"]) for r in rows] == [(7, 11, -1)]
    assert rows[0]["sched2"]["priority"] == 5 and "sched" not in rows[0]
    assert rows[0]["hint"].startswith(
        "to land 1 words earlier it must be picked after insn 6 has priority 1 against its 5"
    )
    assert sched.report("f", {}, listing, diff) == []


def test_the_hint_tells_a_priority_tie_from_a_priority_gap() -> None:
    tie = {"before": [{"insn": 8, "line": 12, "priority": 5, "luid": 2}], "after": [], "priority": 5}
    assert "ties it at priority 5 and is ordered by input position (luid 2 vs 1)" in sched.hint(1, tie, 1)
    gap = {"before": [], "after": [{"insn": 3, "line": 4, "priority": 3, "luid": 0}], "priority": 5}
    assert "line 4 has priority 3 against its 5" in sched.hint(-1, gap, 1)
    assert "fixed by its dependencies" in sched.hint(-1, {"before": [], "after": [], "priority": 5}, 1)


def test_block_end_groups_read_as_pinned_not_as_a_number() -> None:
    pin = {"before": [{"insn": 8, "line": 12, "priority": 0x7F000001, "luid": 2}], "after": [], "priority": 0x7F000001}
    assert "ties it at priority pinned to the block end" in sched.hint(1, pin, 1)


def test_displacement_is_how_far_a_pass_carried_the_insn_from_its_input_place() -> None:
    from types import SimpleNamespace

    dump = SimpleNamespace(order=[3, 1, 2], luid={1: 0, 2: 1, 3: 2})
    assert [sched.displacement(dump, u) for u in (1, 2, 3, 9)] == [1, 1, -2, 0]


def test_a_diverging_insn_no_pass_moved_names_the_input_order() -> None:
    listing = ["800BAE98: addiu $v0, $v0, 8", "800BAE9C: sw $zero, 0x2C($v0)", "800BAEA0: sw $s6, 0x30($v0)"]
    diff = ["+800BAE9C: sw $zero, 0x2C($v0)", "-800BAE98: sw $zero, 0x2C($v0)"]
    row = sched.report("f", {"sched2": DUMP}, listing, diff)[0]
    assert row["first_pass"] == "input" and row["sched2"]["displaced"] == 0
