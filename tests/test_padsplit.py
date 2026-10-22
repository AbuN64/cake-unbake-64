"""padsplit: the symbol rows of a pad word and the function's move to a file of its own."""

from __future__ import annotations

import pytest

from unbake import padsplit
from unbake.contracts import Refusal

SYMBOLS = (
    "func_a = 0x80100000; // type:func rom:0x1000\n"
    "func_f = 0x8010002C; // type:func rom:0x102C size:0x80\n"
    "func_g = 0x80100100;\n"
)

SOURCE = """#include "include_asm.h"
#include "TYPES.H"
extern int D_80000000;
#if defined(VERSION_US)
#include "US.H"
#endif
INCLUDE_ASM("seg", func_a);
#ifdef NON_MATCHING
int func_f(int x) {
    return x + D_80000000;
}
#else
INCLUDE_ASM("seg", func_f);
#endif
INCLUDE_ASM("seg", func_g);
"""


def test_the_pad_row_goes_in_front_and_the_function_row_moves_a_word() -> None:
    out = padsplit.symbol_rows(SYMBOLS, "func_f", 0x102C, 0x8010002C).splitlines()
    assert out[1] == "D_auto_func_f_padding_102C = 0x8010002C; // rom:0x102C"
    assert out[2] == "func_f = 0x80100030; // type:func rom:0x1030 size:0x7C"
    assert out[0].startswith("func_a") and out[3].startswith("func_g")


def test_a_row_that_is_not_where_the_manifest_says_is_refused() -> None:
    with pytest.raises(Refusal):
        padsplit.symbol_rows(SYMBOLS, "func_f", 0x1030, 0x8010002C)
    with pytest.raises(Refusal):
        padsplit.symbol_rows(SYMBOLS, "func_missing", 0, 0)


def test_the_hull_takes_the_non_matching_block_whole_and_leaves_the_other_functions() -> None:
    start, end, wrappers = padsplit.hull(SOURCE, "func_f")
    block = SOURCE[start:end]
    assert block.startswith("#ifdef NON_MATCHING") and block.endswith("#endif") and "func_g" not in block
    assert wrappers == []


def test_split_file_keeps_the_head_and_the_enclosing_guard_and_removes_the_function() -> None:
    kept, new = padsplit.split_file(SOURCE, "func_f")
    assert "func_f" not in kept and 'INCLUDE_ASM("seg", func_a);' in kept and "func_g" in kept
    assert new.startswith('#include "include_asm.h"') and "extern int D_80000000;" in new
    assert "#ifdef NON_MATCHING" in new and 'INCLUDE_ASM("seg", func_f);' in new and "func_a" not in new
    assert new.count("#if") == new.count("#endif")


def test_a_function_inside_a_version_guard_brings_the_guard_along() -> None:
    text = '#include "x.h"\nINCLUDE_ASM("seg", func_a);\n#if defined(VERSION_US)\nINCLUDE_ASM("seg", func_f);\n#endif\n'
    kept, new = padsplit.split_file(text, "func_f")
    assert "func_f" not in kept and "#if defined(VERSION_US)\nINCLUDE_ASM" in new and new.rstrip().endswith("#endif")


def test_pieces_interleaved_with_another_function_are_refused() -> None:
    text = 'INCLUDE_ASM("seg", func_f);\nint func_x(void) { return 1; }\nINCLUDE_ASM("seg", func_f);\n'
    with pytest.raises(Refusal):
        padsplit.hull(text, "func_f")
