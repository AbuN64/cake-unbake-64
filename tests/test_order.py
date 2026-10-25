"""unbake order: the dependency guard, the move list and the climb, with probe mocked."""

import contextlib

import pytest

from unbake import order
from unbake.contracts import Refusal

SOURCE = """#include "x.h"
void Thing_Run(Thing *t) {
    s32 a;
    s32 b;
    a = t->x;
    b = Other_Get(2);
    t->y = a + 1;
    Other_Set(b);
    if (a) {
        a = 1;
        b = 2;
        t->z = a;
        t->w = b;
    }
}
"""


def test_a_write_and_a_read_of_one_name_conflict_but_unrelated_statements_swap():
    assert order.conflict("a = b + 1;", "b = 2;")  # write after read
    assert order.conflict("a = 1;", "a = 2;")  # write after write
    assert order.conflict("a = 1;", "c = a;")  # read after write
    assert not order.conflict("a = 1;", "c = 2;")
    assert not order.conflict("v[0] = 1;", "v[1] = 2;")  # array cells apart


def test_calls_and_pointer_stores_are_barriers_and_control_flow_conflicts_with_what_it_names():
    assert order.conflict("t->y = 1;", "x = t->y;")
    assert order.conflict("a = Get();", "b = Other();")  # both reach memory
    assert order.conflict("Other_Set(b);", "t->y = 2;")  # a bare call is a barrier against memory
    assert order.conflict("Other_Set(b);", "Other_Get(c);")
    assert order.conflict("if (a) { a = 1; }", "a = 3;")


def test_statements_group_balanced_blocks_and_keep_blank_lines():
    lines = ["a = 1;", "if (a) {", "  b = 2;", "}", "", "c = 3;"]
    assert order.statements(lines) == ["a = 1;", "if (a) {\n  b = 2;\n}", "", "c = 3;"]


def test_moves_never_cross_a_conflict_or_a_declaration_and_stay_in_the_window():
    block = ["s32 a;", "a = 1;", "c = 2;", "d = 3;"]
    made = order.moves(block, 1)
    assert all(m[0] == "s32 a;" for m in made)  # declarations stay put
    assert ["s32 a;", "c = 2;", "a = 1;", "d = 3;"] in made
    assert ["s32 a;", "a = 1;", "d = 3;", "c = 2;"] in made
    blocked = order.moves(["a = 1;", "b = a;"], 3)
    assert blocked == []
    four = ["a = 1;", "x = 7;", "y = 8;", "z = 9;"]
    assert len(order.moves(four, 1)) < len(order.moves(four, 3))


def test_body_and_blocks_find_the_function_and_its_nested_blocks():
    lines = SOURCE.split("\n")
    opened, closed = order.body(lines, "Thing_Run")
    assert lines[opened].endswith("{") and lines[closed] == "}"
    assert order.blocks(lines, opened, closed) == [(8, 13)]
    with pytest.raises(Refusal):
        order.body(lines, "Missing")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """probe is mocked: a source scores by how early `t->w = b;` stands among the statements."""
    seen = []

    def fake_probe(config, params):
        text = params["file"].read_text()
        seen.append(text)
        lines = text.split("\n")
        place = next(i for i, line in enumerate(lines) if "t->w = b;" in line)
        score = 1.0 - place / 100
        return {"score": score, "exact": place == 10, "words": {"built": 1, "target": 1, "matched": 1}}

    monkeypatch.setattr(order.compare, "probe_command", fake_probe)
    monkeypatch.setattr(order.pool, "map", lambda config, name, fn, items, key=None: [fn(i) for i in items])
    monkeypatch.setattr(order.store, "work", lambda config: contextlib.nullcontext(tmp_path))
    src = tmp_path / "cand.c"
    src.write_text(SOURCE)
    return src, seen


def test_the_climb_keeps_the_best_order_writes_out_and_never_edits_the_input(world, tmp_path):
    src, seen = world
    got = order.order_command(object(), {"file": str(src), "function": "Thing_Run", "window": 6, "rounds": 4})
    assert got["improved"] and got["best"]["score"] > got["start"]["score"]
    assert src.read_text() == SOURCE
    best = (tmp_path / "cand.ordered.c").read_text()
    assert (
        got["out"].endswith("cand.ordered.c")
        and best != SOURCE
        and sorted(best.split("\n")) == sorted(SOURCE.split("\n"))
    )
    assert got["probes"] == len(seen) and got["next"].startswith("unbake probe ")


def test_the_budget_stops_the_climb_and_a_flat_score_writes_nothing(world, monkeypatch, tmp_path):
    src, _ = world
    got = order.order_command(object(), {"file": str(src), "function": "Thing_Run", "budget": 3})
    assert got["probes"] <= 3 and got["stopped"] in ("budget", "exact")
    monkeypatch.setattr(order.compare, "probe_command", lambda c, p: {"score": 0.5, "exact": False, "words": {}})
    flat = order.order_command(object(), {"file": str(src), "function": "Thing_Run", "out": str(tmp_path / "o.c")})
    assert not flat["improved"] and flat["out"] == "" and not (tmp_path / "o.c").exists()
    assert flat["stopped"] == "no move helps"


def test_a_missing_file_or_function_is_refused(world, tmp_path):
    src, _ = world
    with pytest.raises(Refusal):
        order.order_command(object(), {"file": str(tmp_path / "none.c"), "function": "X"})
    with pytest.raises(Refusal):
        order.order_command(object(), {"file": str(src)})
