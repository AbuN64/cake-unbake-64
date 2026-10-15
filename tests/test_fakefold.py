"""Fold plan of invented per-function records: one case per class and the refusals that leave a record unplanned."""

from collections import Counter

import pytest

from unbake import fakefold


def lay(size, *fields):
    return {"size": size, "fields": [{"offset": o, "width": w, "type": t, "name": n} for o, w, t, n in fields]}


REAL = lay(16, (0, 4, "s32", "count"), (4, 4, "void *", "next"), (8, 8, "char[8]", "name"))
VIEW = lay(8, (0, 4, "s32", "unk0"), (4, 4, "struct Real *", "unk4"))


@pytest.mark.parametrize(
    ("symbol", "pointer", "name"),
    [
        ("D_800E3518", True, "Data800E3518Ptr"),
        ("D_auto_80140ff8", False, "Data80140FF8"),
        ("g_menuState", False, "DataMenuState"),
        ("s_menu_list", True, "DataMenuListPtr"),
    ],
)
def test_data_name(symbol, pointer, name):
    assert fakefold.data_name(symbol, pointer) == name


@pytest.mark.parametrize(
    ("mine", "score"),
    [
        (VIEW, 2),  # void * takes the struct pointer
        (lay(8, (0, 4, "s32", "a"), (4, 4, "char[4]", "pad4")), 1),  # pads are not counted
        (lay(8, (0, 2, "s16", "a")), -1),  # overlaps count with another shape
        (lay(20, (0, 4, "s32", "a")), -1),  # runs past the end
    ],
)
def test_fits(mine, score):
    assert fakefold.fits(mine, REAL) == score


def test_reaches_kinds():
    texts = {
        "include/a.h": "extern Rec_func_80100000_de D_80100000;\nstruct S {\n    Rec_func_80100000_de *link;\n};\n"
        "typedef struct Rec_func_80100000_de Alias;\n",
        "src/a.c": "void f(Rec_func_80100000_de *arg0) {\n    Rec_func_80100000_de *temp;\n}\n",
    }
    found = fakefold.reaches(texts, ["Rec_func_80100000_de"])["Rec_func_80100000_de"]
    assert found["data"] == Counter({"D_80100000": 1})
    assert found["member"] == Counter({"*link": 1})
    assert found["local"] == Counter({"*arg0": 1, "*temp": 1})


def test_plan_classes():
    layouts = {
        "Real": REAL,
        "Other": lay(4, (0, 4, "f32", "x")),
        "View_func_80100000_de": VIEW,  # a: fits Real only
        "func_80100010_S1": VIEW,  # a: a typedef names it as Real
        "Sym_func_80100020_de": lay(4, (0, 2, "u16", "a")),  # c: one data symbol
        "Ptr_func_80100024_de": lay(4, (0, 2, "u16", "b")),  # c: same Data record through a pointer: a fold
        "Box_func_80100030_de": lay(12, (8, 1, "u8", "flag")),  # b: same layout, reached as member `box`
        "Box_func_80100040_de": lay(12, (8, 1, "u8", "state")),
        "Lone_func_80100050_de": lay(12, (8, 2, "s16", "x")),  # d: locals only
        "func_80100060_S2": lay(4, (0, 2, "u16", "a")),  # alias of an invented record: follows it
    }
    aliases = {"func_80100010_S1": "Real", "func_80100060_S2": "Sym_func_80100020_de"}
    reached = {
        "Sym_func_80100020_de": {"data": Counter({"D_80200000": 1})},
        "Ptr_func_80100024_de": {"data": Counter({"*D_80200000": 1})},
        "Box_func_80100030_de": {"member": Counter({"*box": 2})},
        "Box_func_80100040_de": {"member": Counter({"*box": 1})},
        "Lone_func_80100050_de": {"local": Counter({"*temp": 1})},
    }
    rows, census = fakefold.plan(layouts, aliases, reached, set(layouts), {"View_func_80100000_de": 40})
    got = {r["subject"]: r["target"] for r in rows}
    assert got == {
        "View_func_80100000_de": "Real",
        "func_80100010_S1": "Real",
        "Sym_func_80100020_de": "Data80200000",
        "Ptr_func_80100024_de": "Data80200000Ptr",
        "Box_func_80100030_de": "BoxBox",
        "Box_func_80100040_de": "BoxBox",
        "func_80100060_S2": "Data80200000",
    }
    assert [r["subject"] for r in rows][-1] == "func_80100060_S2"  # aliases after records: their target exists
    by = {c["record"]: c for c in census}
    assert by["View_func_80100000_de"]["class"] == "a" and by["View_func_80100000_de"]["bytes"] == 40
    assert (
        by["Lone_func_80100050_de"]["class"] == "d" and "no real record fits" in by["Lone_func_80100050_de"]["reason"]
    )
    assert by["Box_func_80100030_de"]["class"] == "b" and by["func_80100060_S2"]["class"] == "c"


def test_plan_refuses_ties_and_taken_names():
    layouts = {
        "A": lay(4, (0, 4, "s32", "x")),
        "B": lay(8, (0, 4, "s32", "y")),
        "C": lay(8, (0, 4, "s32", "z")),
        "T_func_80100000_de": lay(4, (0, 4, "s32", "v")),  # A has the same size: picked
        "U_func_80100010_de": lay(2, (0, 1, "u8", "v")),  # nothing fits, member names taken
        "U_func_80100020_de": lay(2, (0, 1, "u8", "w")),
    }
    reached = {n: {"member": Counter({"*u": 1})} for n in ("U_func_80100010_de", "U_func_80100020_de")}
    rows, census = fakefold.plan(layouts, {}, reached, {*layouts, "UU"})
    assert {r["subject"]: r["target"] for r in rows} == {"T_func_80100000_de": "A"}
    reasons = {c["record"]: c["reason"] for c in census}
    assert "no declarator names the group" in reasons["U_func_80100010_de"]
