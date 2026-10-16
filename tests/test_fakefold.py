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


def test_traced_d_cases_from_the_rw_census():
    # real cases (census 13:00, G d66a9347a): ALFx_func_802B5B94_de (476 B) reached by nothing and fitting nothing;
    # Actor_func_80214624_de (924 B) in a flow component joining 24 real records; an unnamed same-layout group
    lone = "ALFx_func_802B5B94_de"
    actor, other = "Actor_func_80214624_de", "Actor_func_80299999_de"
    g1, g2 = "Pak_func_80300000_de", "Kap_func_80300100_de"  # two roles: the group gets no role name
    layouts = {
        lone: lay(8, (0, 4, "s32", "delay"), (4, 4, "f32", "gain")),
        actor: lay(8, (0, 4, "s32", "count")),
        other: lay(8, (4, 4, "f32", "speed")),
        "Real": REAL,
        "Mover": lay(12, (4, 4, "f32", "speed")),
        g1: lay(6, (0, 2, "s16", "a")),
        g2: lay(6, (0, 2, "s16", "b")),
    }
    modules = {"func_802B5B94": "audio_802B5000", "func_80300000": "pak_80300000"}
    edges = [(actor, "Real"), (actor, other), (other, "Mover")]
    rows, _ = fakefold.plan(layouts, {}, {}, set(layouts), edges=edges, modules=modules)
    got = {r["subject"]: r["target"] for r in rows}
    assert got[lone] == "ALFxAudio802B5000"  # its own component, named by role and module
    assert got[actor] == "Real" and got[other] == "Mover"  # each takes the real record its own pointer reaches
    assert got[g1] == got[g2] == "PakPak80300000"  # an unnamed group is one component named by role and module


def test_components_of_one_role_in_one_module_and_numbered_views_still_get_names():
    a, b, v = "Actor_func_80212D78_de", "Actor_func_80212E00_de", "func_80212F00_de_S3"
    layouts = {a: lay(4, (0, 4, "s32", "hp")), b: lay(4, (0, 4, "f32", "speed")), v: lay(4, (0, 2, "u16", "x"))}
    modules = {"func_80212D78": "code_80212", "func_80212E00": "code_80212", "func_80212F00": "code_80212"}
    rows, _ = fakefold.plan(layouts, {}, {}, set(layouts), modules=modules)
    assert {r["subject"]: r["target"] for r in rows} == {
        a: "ActorCode80212",
        b: "ActorCode8021280212E00",
        v: "RecCode80212",
    }


def test_a_component_takes_the_module_of_its_lowest_function_with_a_known_module():
    # real case: component 169 (census 13:20): Shape_func_8020D364_de_2 has no module, Shape_func_80267D70_de_2 has one
    a, b = "Shape_func_8020D364_de_2", "Shape_func_80267D70_de_2"
    layouts = {a: lay(4, (0, 4, "s32", "x")), b: lay(4, (0, 4, "s32", "x"), (4, 4, "f32", "y"))}
    rows, _ = fakefold.plan(layouts, {}, {}, set(layouts), edges=[(a, b)], modules={"func_80267D70": "code_802661FC"})
    assert {r["target"] for r in rows} == {"ShapeCode802661FC"}


def field(offset, width, kind, name, *sub):
    return {"offset": offset, "width": width, "type": kind, "name": name, "fields": list(sub)}


def player(name, *, extra=True, rider="struct RiderRecAt130V1 *", arm="unk8"):
    """A SharedPlayer-shaped record (RW-AROUND GAPS 17:20): a union of views at 0x0, a self pointer in a view at
    0x1D8 and a pointee that differs between the clones; SharedPlayer alone names turnAccumulator at 0x16C0, where the
    clones hold padding, and one more arm (navigationTrack) at 0x18."""
    vec = "struct { float; float; float; }"
    views0 = field(
        0,
        20,
        f"union {{ struct {{ char [8]; {vec}; }}; }}",
        "views0",
        field(
            0, 20, f"struct {{ char [8]; {vec}; }}", "view8_2", field(0, 8, "char [8]", "pad"), field(8, 12, vec, arm)
        ),
    )
    arms = [field(0, 4, "char *", "track")] + (
        [field(0, 4, "struct NavigationTrack *", "navigationTrack")] if extra else []
    )
    views18 = field(0x18, 4, "union { char *; }", "views18", *arms)
    self = field(0x1D8, 4, f"union {{ struct {name} *; }}", "views1D8", field(0, 4, f"struct {name} *", "self"))
    tail = (
        [field(0x1600, 0xC0, "char [192]", "pad1600"), field(0x16C0, 4, "float", "turnAccumulator")]
        if extra
        else [field(0x1600, 0xD4, "char [212]", "pad1600")]
    )
    return {"size": 0x16E8, "fields": [views0, views18, self, field(0x5D8, 4, rider, "rider"), *tail]}


def test_a_full_copy_of_one_real_record_folds_into_it_before_its_ambiguous_flow():
    clone = "SharedPlayer_func_80209CD8_de"
    layouts = {
        "SharedPlayer": player("SharedPlayer"),
        "Rider": lay(8, (0, 4, "s32", "id")),
        "View": lay(8, (0, 4, "s32", "key")),
        clone: player(clone, extra=False, rider="struct Rider *"),
    }
    assert fakefold.fits(layouts[clone], layouts["SharedPlayer"]) == -1  # the union types differ: class a misses it
    edges = [(clone, "Rider"), (clone, "View")]  # its flow joins 2 real records
    rows, census = fakefold.plan(layouts, {}, {}, set(layouts), edges=edges)
    assert rows == [{"op": "type", "subject": clone, "target": "SharedPlayer"}]
    assert census[0]["class"] == "a" and census[0]["reason"] == "a full copy of SharedPlayer"


def test_a_full_copy_of_several_real_records_stays_unplanned_and_names_them():
    clone = "SharedPlayer_func_80209CD8_de"
    layouts = {
        "SharedPlayer": player("SharedPlayer"),
        "PlayerTwin": player("PlayerTwin"),
        "View": lay(8, (0, 4, "s32", "key")),
        clone: player(clone, extra=False),
    }
    edges = [(clone, "SharedPlayer"), (clone, "View")]
    rows, census = fakefold.plan(layouts, {}, {}, set(layouts), edges=edges)
    assert rows == []
    row = next(c for c in census if c["record"] == clone)
    assert row["class"] == "d" and row["reason"] == "a full copy of 2 real records equally: PlayerTwin, SharedPlayer"


@pytest.mark.parametrize(
    "clone",
    [
        player("SharedPlayer_func_80209CD8_de", extra=False, arm="pos"),  # a view member another name holds
        {**player("SharedPlayer_func_80209CD8_de", extra=False), "size": 0x16F0},  # another size
        {  # a member of its own where SharedPlayer names turnAccumulator
            **player("SharedPlayer_func_80209CD8_de", extra=False),
            "fields": [*player("x", extra=False)["fields"][:-1], field(0x16C0, 4, "s32", "spin")],
        },
    ],
)
def test_a_record_that_is_no_full_copy_is_not_folded_as_one(clone):
    layouts = {"SharedPlayer": player("SharedPlayer"), "SharedPlayer_func_80209CD8_de": clone}
    rows, census = fakefold.plan(layouts, {}, {}, set(layouts))
    assert rows == [] and "full copy" not in census[0]["reason"]


def test_plan_builds_each_real_records_rows_once_and_never_scores_one_too_small(monkeypatch):
    built, scored = Counter(), []
    index, score = fakefold._Index, fakefold._fits

    class Counted(index):
        def __init__(self, layout):
            built[id(layout)] += 1
            super().__init__(layout)

    def spy(named, size, theirs):
        scored.append((size, theirs.size))
        return score(named, size, theirs)

    monkeypatch.setattr(fakefold, "_Index", Counted)
    monkeypatch.setattr(fakefold, "_fits", spy)
    reals = {f"Real{i}": lay(4 * i, (0, 4, "s32", f"r{i}")) for i in range(1, 6)}
    fakes = {f"Rec_func_8010000{i}_de": lay(12, (0, 4, "s32", f"v{i}"), (8, 4, "f32", "w")) for i in range(6)}
    fakefold.plan({**reals, **fakes}, {}, {}, set(reals) | set(fakes))
    assert sorted(built.values()) == [1] * 5 and set(built) == {id(r) for r in reals.values()}
    assert scored and all(mine <= theirs for mine, theirs in scored)  # Real1, Real2 (4, 8 bytes) are never scored
    assert len(scored) == 6 * 3
