"""The text rule of `layout split`: one unit into a file per target, with the shared and the crossing parts kept."""

from __future__ import annotations

from unbake import split

UNIT = """#include "game.h"
typedef int word;

static word counter = 3;
static void helper(void);
int shared_value = 7;

/* the camera half */
int camera_update(void)
{
    helper();
    return counter;
}

static void helper(void)
{
}

#if DEBUG
int player_debug(void) { return 1; }
#endif

int player_tick(void) { return camera_update() + shared_value; }
INCLUDE_DATA("data", player_table);
"""

OWNER = {
    "counter": "cam",
    "helper": "cam",
    "camera_update": "cam",
    "shared_value": "ply",
    "player_debug": "ply",
    "player_tick": "ply",
    "player_table": "ply",
}


def test_each_target_gets_its_symbols_and_the_shared_statements() -> None:
    out = split.split("u.c", UNIT, OWNER)
    assert out["conflicts"] == [] and out["unassigned"] == []
    cam, ply = out["pieces"]["cam"], out["pieces"]["ply"]
    for text in (cam, ply):
        assert '#include "game.h"' in text and "typedef int word;" in text
    assert "int camera_update(void)" in cam and "static void helper(void);" in cam
    assert "player_tick" not in cam and "player_debug" not in cam and "INCLUDE_DATA" not in cam
    assert "camera_update(void)\n{" not in ply and "helper" not in ply
    assert "int player_tick(void)" in ply and "INCLUDE_DATA(" in ply and "int shared_value = 7;" in ply


def test_the_comment_above_a_statement_moves_with_it() -> None:
    out = split.split("u.c", UNIT, OWNER)
    assert "/* the camera half */" in out["pieces"]["cam"]
    assert "/* the camera half */" not in out["pieces"]["ply"]


def test_a_symbol_used_in_another_file_is_declared_there() -> None:
    out = split.split("u.c", UNIT, OWNER)
    assert "int camera_update(void);" in out["pieces"]["ply"]
    assert out["declared"] == {"cam": [], "ply": ["camera_update"]}
    crossing = split.split("u.c", UNIT, {**OWNER, "counter": "ply"})
    assert "extern word counter;" in crossing["pieces"]["cam"]


def test_a_static_symbol_used_in_another_file_loses_static_and_is_declared() -> None:
    owner = {**OWNER, "counter": "ply", "camera_update": "cam"}
    out = split.split("u.c", UNIT, owner)
    assert out["unstatic"] == ["counter"]
    assert "word counter = 3;" in out["pieces"]["ply"] and "static word counter" not in out["pieces"]["ply"]
    assert "extern word counter;" in out["pieces"]["cam"]


def test_an_if_block_left_empty_goes() -> None:
    out = split.split("u.c", UNIT, OWNER)
    assert "#if DEBUG" not in out["pieces"]["cam"] and "#endif" not in out["pieces"]["cam"]
    assert "#if DEBUG" in out["pieces"]["ply"]


def test_a_symbol_no_target_names_goes_to_the_biggest_target_and_is_listed() -> None:
    out = split.split("u.c", UNIT, {k: v for k, v in OWNER.items() if k != "helper"})
    assert out["unassigned"] == ["helper"]
    assert "\nvoid helper(void)\n{" in out["pieces"]["ply"] and "void helper(void);" in out["pieces"]["cam"]
    assert out["unstatic"] == ["helper"]


def test_one_statement_declaring_symbols_of_two_targets_is_a_conflict() -> None:
    out = split.split("u.c", "int a, b;\nint f(void) { return a; }\n", {"a": "x", "b": "y", "f": "x"})
    assert out["conflicts"] and "a, b" in out["conflicts"][0]
