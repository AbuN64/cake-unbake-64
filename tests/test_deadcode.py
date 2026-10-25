"""unbake deadcode: which source lines lose every insn to a dead-code pass, from hand-made RTL dumps."""

import pytest

from unbake import deadcode
from unbake.contracts import Refusal

FILE = '("a.c")'


def note(uid: int, line: int) -> str:
    return f"(note {uid} 0 0 {FILE} {line})\n"


def insn(uid: int, body: str = "(set (reg:SI 80) (const_int 4))") -> str:
    return f"(insn {uid} 0 0 {body} -1 (nil)\n    (nil))\n"


def dump(name: str, *parts: str) -> str:
    return f";; Function {name}\n" + "".join(parts)


def chain(**kept: str) -> dict[str, str]:
    """The same function in every pass; `kept` overrides a pass."""
    base = dump(
        "F",
        note(1, 10),
        insn(2),
        note(3, 11),
        insn(4, "(set (mem:SI (reg:SI 4)) (reg:SI 81))"),
        note(5, 12),
        insn(6, "(set (reg:SI 82) (reg:SI 83))"),
        note(7, 13),
        insn(8),
        insn(9),
    )
    return {p: kept.get(p, base) for p in deadcode.PASSES}


def test_a_line_whose_every_insn_cse_or_flow_deleted_is_listed():
    after = dump(
        "F", note(1, 10), insn(2), note(3, 11), note(5, 12), insn(6, "(set (reg:SI 82) (reg:SI 83))"), insn(8), insn(9)
    )
    got = deadcode.dead_lines(chain(flow=after, combine=after, sched=after, greg=after), "F")
    assert got == [{"line": 11, "insns": 1, "passes": ["flow"]}]


def test_register_copies_and_lines_with_a_surviving_insn_are_not_dead_code():
    # line 12 is only a register copy that vanished (copy propagation); line 13 lost insn 9 but insn 8 stays
    kept = dump(
        "F", note(1, 10), insn(2), note(3, 11), insn(4, "(set (mem:SI (reg:SI 4)) (reg:SI 81))"), note(7, 13), insn(8)
    )
    got = deadcode.dead_lines(
        chain(cse=kept, loop=kept, cse2=kept, flow=kept, combine=kept, sched=kept, greg=kept), "F"
    )
    assert got == []
    assert deadcode.dead_lines({"rtl": chain()["rtl"]}, "F") == []  # no later dump: nothing to compare


def test_a_line_the_combiner_merged_away_still_has_an_effect():
    merged = dump(
        "F",
        note(1, 10),
        insn(2),
        note(3, 11),
        insn(4, "(set (mem:SI (reg:SI 4)) (reg:SI 81))"),
        note(5, 12),
        insn(6, "(set (reg:SI 82) (reg:SI 83))"),
    )
    got = deadcode.dead_lines(chain(combine=merged, sched=merged, greg=merged), "F")
    assert got == []


def test_insns_gives_each_insn_its_nearest_line_and_marks_register_copies():
    got = deadcode.insns(chain()["rtl"].split("\n", 1)[1])
    assert got[2] == (10, False) and got[4] == (11, False) and got[6] == (12, True) and got[9] == (13, False)
    assert deadcode.alive(chain()["rtl"]) == {2, 4, 6, 8, 9}
    two = dump("A", insn(1)) + dump("B", insn(2))
    assert set(deadcode.functions(two)) == {"A", "B"}


def test_a_missing_source_is_refused(tmp_path):
    config = type("C", (), {"project": type("P", (), {"root": tmp_path, "names_from": "v"})()})()
    with pytest.raises(Refusal):
        deadcode.deadcode_command(config, {"file": "none.c"})
