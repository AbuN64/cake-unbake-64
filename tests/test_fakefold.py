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
    rows, census = fakefold.plan(layouts, {}, reached, {*layouts, "UU", "U"})
    assert {r["subject"]: r["target"] for r in rows} == {"T_func_80100000_de": "A"}
    reasons = {c["record"]: c["reason"] for c in census}
    assert "no declarator names the group" in reasons["U_func_80100010_de"]


@pytest.mark.parametrize(
    ("tops", "role", "size", "pick"),
    [
        (["ALSynth4C", "ALSynth"], "ALSynth", 32, "ALSynth"),  # the role names it
        (["Shared_PakMenu", "MenuRecAt24"], "Menu", 40, "MenuRecAt24"),  # the role starts only one
        (["Big", "Same"], "Item", 8, "Same"),  # no role evidence: the one of the same size
        (["Same", "Twin"], "Item", 8, ""),  # still a tie: no fold
    ],
)
def test_chosen_breaks_ties(tops, role, size, pick):
    real = {
        "ALSynth4C": {"size": 76},
        "ALSynth": {"size": 76},
        "Shared_PakMenu": {"size": 40},
        "MenuRecAt24": {"size": 40},
        "Big": {"size": 16},
        "Same": {"size": 8},
        "Twin": {"size": 8},
    }
    assert fakefold._chosen(tops, role, size, real) == pick


def test_group_named_from_shared_role():
    layouts = {f"Holder_func_8010000{i}_de": lay(8, (4, 4, "s32", f"v{i}")) for i in range(3)}
    rows, census = fakefold.plan(layouts, {}, {}, set(layouts))
    assert {r["target"] for r in rows} == {"Holder"} and len(rows) == 3
    assert all(c["class"] == "b" and "role Holder" in c["reason"] for c in census)


A = "Brain_func_80212D78_de"
B = "Brain_func_80213000_de"


def flowed(texts, layouts, aliases=None):
    from unbake import policy

    found = [fakefold.facts(policy.scopes(t)["code"]) for t in texts]
    return fakefold.flow(found, layouts, aliases or {})


def test_two_functions_passing_one_pointer_join_their_records():
    layouts = {A: lay(8, (0, 4, "s32", "hp")), B: lay(8, (4, 4, "f32", "speed")), "Real": REAL}
    caller = f"void f({A} *self) {{\n    g(0, self);\n}}\n"
    callee = f"void g(s32 n, {B} *brain) {{\n    brain->speed = 1.0f;\n}}\n"
    assert flowed([caller, callee], layouts) == [(A, B)]


@pytest.mark.parametrize(
    "line",
    ["{b} *q = p;", "{a} *q;\n    q = p->link;", "{b} *q = make();"],
    ids=["variable", "member load", "return value"],
)
def test_assignments_join_records(line):
    layouts = {A: lay(8, (0, 4, f"{B} *", "link")), B: lay(4, (0, 4, "s32", "x"))}
    text = f"{A} *make(void);\nvoid f({A} *p) {{\n    " + line.format(a=A, b=B) + "\n}\n"
    assert flowed([text], layouts) == [(A, B)]


def test_a_flow_component_without_a_real_record_gets_one_shared_record_named_from_evidence():
    layouts = {A: lay(8, (0, 4, "s32", "hp")), B: lay(8, (0, 4, "s32", "hp"), (4, 4, "f32", "speed"))}
    reached = {A: {"local": Counter({"*brain": 2})}, B: {"local": Counter({"*brain": 1})}}
    rows, census = fakefold.plan(layouts, {}, reached, set(layouts), edges=[(A, B)])
    assert rows == [{"op": "type", "subject": B, "target": "Brain"}, {"op": "type", "subject": A, "target": "Brain"}]
    assert {r["class"] for r in census} == {"e"} and census[0]["component"] == census[1]["component"]


def test_naming_falls_back_to_the_role_and_the_module_of_the_lowest_function():
    layouts = {A: lay(8, (0, 4, "s32", "hp")), B: lay(8, (4, 4, "f32", "speed"))}
    reached = {A: {"local": Counter({"*arg0": 1})}, B: {"local": Counter({"*temp": 1})}}
    rows, _ = fakefold.plan(layouts, {}, reached, set(layouts), edges=[(A, B)], modules={"func_80212D78": "code_80212"})
    assert {r["target"] for r in rows} == {"BrainCode80212"}


def test_a_component_holding_a_real_record_folds_into_it_and_a_conflict_refuses_it():
    layouts = {A: lay(8, (0, 4, "s32", "count")), B: lay(8, (4, 4, "void *", "next")), "Real": REAL}
    rows, census = fakefold.plan(layouts, {}, {}, set(layouts), edges=[(A, B), (B, "Real")])
    assert {r["subject"]: r["target"] for r in rows} == {A: "Real", B: "Real"}
    clash = {**layouts, B: lay(8, (0, 2, "s16", "half"))}
    rows, census = fakefold.plan(clash, {}, {}, set(clash), edges=[(A, B), (B, "Real")])
    assert B not in {r["subject"] for r in rows} and "refused" in next(r for r in census if r["record"] == B)["reason"]


def test_a_member_conflict_in_a_component_with_no_real_record_refuses_it():
    layouts = {A: lay(8, (0, 4, "s32", "hp")), B: lay(8, (0, 4, "f32", "speed"))}
    reached = {A: {"local": Counter({"*brain": 1})}}
    rows, census = fakefold.plan(layouts, {}, reached, set(layouts), edges=[(A, B)])
    assert rows == [] and all("conflict" in r["reason"] for r in census)


def test_a_fit_tie_is_broken_by_flow():
    two = {"RealA": lay(8, (0, 4, "s32", "x"), (4, 4, "s32", "y")), "RealB": lay(8, (0, 4, "s32", "x"))}
    layouts = {A: lay(4, (0, 4, "s32", "x")), **two}
    _, census = fakefold.plan(layouts, {}, {}, set(layouts))
    assert census[0]["class"] == "d"
    rows, census = fakefold.plan(layouts, {}, {}, set(layouts), edges=[(A, "RealA")])
    assert rows == [{"op": "type", "subject": A, "target": "RealA"}] and census[0]["class"] == "e"


def test_a_source_record_folds_into_the_header_record_its_flow_reaches_and_never_gets_a_new_name():
    layouts = {"Local": lay(4, (0, 4, "s32", "count")), "Real": REAL}
    rows, census = fakefold.plan(layouts, {}, {}, set(layouts), local={"Local"})
    assert rows == [{"op": "type", "subject": "Local", "target": "Real"}] and census[0]["local"]
    lone = {"Local": lay(4, (0, 4, "f32", "speed")), "Other": lay(4, (0, 4, "f32", "speed"))}
    rows, census = fakefold.plan(lone, {}, {}, set(lone), edges=[("Local", "Other")], local={"Local", "Other"})
    assert rows == [] and {r["class"] for r in census} == {"d"}
