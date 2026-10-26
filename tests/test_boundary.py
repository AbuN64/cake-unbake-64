"""boundary: rows a split put in the wrong place, found by an exact function of another version, and moved."""

from __future__ import annotations

import struct

from unbake import boundary

JR = boundary.JR_RA
NOP = 0
PRO = 0x27BDFFE8  # addiu sp, sp, -0x18
EPI = 0x27BD0018  # addiu sp, sp, 0x18
BODY = [0x8FA40010, 0x00851021, 0xAFA20014]


def image(words: list[int]) -> bytes:
    return b"".join(struct.pack(">I", w) for w in words)


def rows_of(table: dict) -> tuple[list, dict]:
    rows = boundary._rows(table)
    return rows, boundary._ends(rows)


def test_a_function_cut_in_two_is_one_row_again():
    # prev (jr ra; nop) | A: prologue + body (no return) | B: epilogue + jr ra; nop | next
    words = [JR, NOP, PRO, *BODY, EPI, JR, NOP, PRO]
    table = {
        "prev": (0x80000000, 0x0, None, "func"),
        "A": (0x80000008, 0x8, None, "func"),
        "B": (0x80000018, 0x18, None, "func"),
        "next": (0x80000024, 0x24, None, "func"),
    }
    rows, ends = rows_of(table)
    move, why = boundary.plan_move("de", "A", "eu:A", 0x8, 0x1C, rows, ends, image(words))
    assert move is not None and why == ""
    assert move.keep == "A" and move.removed == ["B"] and move.trimmed["A"] == (0x80000008, 0x8, 0x1C)


def test_a_prologue_word_alone_on_its_row_joins_the_body():
    words = [JR, NOP, PRO, *BODY, EPI, JR, NOP, PRO]
    table = {
        "prev": (0x80000000, 0x0, None, "func"),
        "Head": (0x80000008, 0x8, None, "func"),  # only `addiu sp, sp, -0x18`
        "func_8000000C": (0x8000000C, 0xC, None, "func"),
        "next": (0x80000024, 0x24, None, "func"),
    }
    rows, ends = rows_of(table)
    move, _ = boundary.plan_move("eu", "func_8000000C", "us-rev1:Head", 0x8, 0x1C, rows, ends, image(words))
    assert move.keep == "Head" and move.removed == ["func_8000000C"] and move.trimmed["Head"][2] == 0x1C


def test_the_next_functions_first_word_left_on_the_row_moves_to_the_next_row():
    words = [PRO, *BODY, EPI, JR, NOP, PRO, *BODY, EPI, JR, NOP]
    table = {
        "A": (0x80000000, 0x0, None, "func"),
        "B": (0x80000020, 0x20, None, "func"),
        "end": (0x80000040, 0x40, None, "func"),
    }
    rows, ends = rows_of(table)
    move, _ = boundary.plan_move("de", "A", "eu:A", 0x0, 0x1C, rows, ends, image([*words, 0]))
    assert move.trimmed["B"] == (0x8000001C, 0x1C, None) and move.trimmed["A"] == (0x80000000, 0x0, 0x1C)


def test_code_before_the_match_that_does_not_end_a_function_refuses():
    words = [0x00851021, 0x00851021, PRO, *BODY, EPI, JR, NOP]
    table = {"A": (0x80000000, 0x0, None, "func"), "end": (0x80000024, 0x24, None, "func")}
    rows, ends = rows_of(table)
    move, why = boundary.plan_move("de", "A", "eu:A", 0x8, 0x1C, rows, ends, image([*words, 0]))
    assert move is None and "not the end of a function" in why


def test_find_takes_the_longest_counterpart_near_the_row():
    fn = [PRO, *BODY, EPI, JR, NOP]
    img = image([JR, NOP, *fn, 0])
    masked = tuple(map(boundary.mask, fn))
    index = {(masked[0], masked[1]): [("eu:A", masked), ("de:A", masked[:4] + masked[4:5])]}
    assert boundary.find(("A", 0xC, 0x24), img, index, "de") == (0x8, 0x1C, "eu:A")


def test_rows_are_rewritten_with_their_new_addresses_and_removed():
    text = (
        "A = 0x80000008; // type:func rom:0x8\n"
        "B = 0x80000018; // type:func rom:0x18\n"
        "C = 0x80000024; // type:func rom:0x24\n"
    )
    move = boundary.Move("de", "A", "eu:A", 0x8, 0x1C, "A", ["B"], {"A": (0x80000008, 0x8, 0x1C)})
    assert boundary.apply_rows(text, move) == (
        "A = 0x80000008; // type:func rom:0x8 size:0x1C\nC = 0x80000024; // type:func rom:0x24\n"
    )


def test_describe_names_the_shapes_of_cut_rows():
    assert boundary.describe([EPI, JR, NOP]) == "tail-fragment"
    assert boundary.describe([PRO, *BODY]) == "no-return"
    assert boundary.describe([0x48656C6C, 0x6F000000]) == "ascii"
    assert boundary.describe([PRO, *BODY, EPI, JR, NOP]) == ""
