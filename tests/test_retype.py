"""Parameter retyping over cut text: what is retyped, what is skipped and why, and the files each change reaches."""

from __future__ import annotations

import pytest

from unbake import retype
from unbake.families import typedebt

TYPE = "typedef struct Actor { int a; int model; } Actor;\n"
HEADER = '#include "common/actor.h"\nextern int fetch(void *arg0);\nextern void touch(void *arg0, int arg1);\n'


def texts(**extra: str) -> dict[str, str]:
    base = {
        "include/common/actor.h": TYPE,
        "include/span/code.h": HEADER,
        "src/fetch.c": '#include "span/code.h"\nint fetch(void *arg0) {\n    return ((Actor *)arg0)->model;\n}\n',
        "src/touch.c": (
            '#include "span/code.h"\nvoid touch(void *arg0, int arg1) {\n'
            "    if (((Actor *)(arg0))->a == arg1) {\n        ((Actor *)(arg0))->model = 1;\n    }\n}\n"
        ),
    }
    return {**base, **{k.replace("__", "/").replace("_c", ".c"): v for k, v in extra.items()}}


def run(files: dict[str, str]) -> dict:
    return retype.plan(files, {"Actor": "include/common/actor.h"}, lambda a, b: a == b or a == "include/span/code.h")


def test_retypes_parameter_casts_and_declarations():
    result = run(texts())
    writes = result["writes"]
    assert result["retyped"] == 2 and result["skipped"] == {}
    assert writes["src/fetch.c"] == '#include "span/code.h"\nint fetch(Actor *arg0) {\n    return arg0->model;\n}\n'
    assert "if (arg0->a == arg1) {\n        arg0->model = 1;" in writes["src/touch.c"]
    assert "extern int fetch(Actor *arg0);" in writes["include/span/code.h"]
    assert "extern void touch(Actor *arg0, int arg1);" in writes["include/span/code.h"]
    assert result["param_cast"] == {"before": 2, "after": 0}


def test_unnamed_and_integer_declarations_keep_their_shape():
    files = texts(src__use_c="extern int fetch(void *);\nextern void touch(int, int arg1);\n")
    writes = run(files)["writes"]
    # use.c cannot see Actor: it names the tag, declared once before its first declaration
    assert (
        writes["src/use.c"]
        == "struct Actor;\nextern int fetch(struct Actor *);\nextern void touch(struct Actor *, int arg1);\n"
    )


UNTAGGED = "typedef struct { int a; int model; } Actor;\n"


def test_header_without_an_untagged_type_includes_its_owner():
    files = {**texts(), "include/common/actor.h": UNTAGGED, "include/span/code.h": "extern int fetch(void *arg0);\n"}
    writes = retype.plan(files, {"Actor": "include/common/actor.h"}, lambda a, b: False)["writes"]
    assert writes["include/span/code.h"].startswith('#include "common/actor.h"\n')


def test_header_cycle_skips_an_untagged_type():
    files = {**texts(), "include/common/actor.h": UNTAGGED}
    result = retype.plan(files, {"Actor": "include/common/actor.h"}, lambda a, b: a == "include/common/actor.h")
    assert result["skipped"] == {"declaring header includes the file that needs the type": 2}
    assert result["writes"] == {}


@pytest.mark.parametrize(
    "owner",
    [
        "typedef struct ParticleEffect ParticleEffect;\nstruct ParticleEffect { int a; int model; };\n",
        "typedef struct ParticleEffect { int a; int model; } ParticleEffect;\n",
    ],
)
def test_header_blind_to_a_tagged_type_declares_the_tag_instead_of_including(owner):
    # real case: code_8028469C.h prototypes func_8028489C_de(void *) while ParticleEffect lives in common/unused.h,
    # which reaches code_8028469C.h; an include there closed a cycle and broke 245 units
    files = {
        "include/common/unused.h": '#include "span/mesh.h"\n' + owner,
        "include/span/mesh.h": "/* mesh */\nextern int other(int arg0);\nextern float effect_x(void *arg0);\n",
        "src/effect.c": (
            '#include "common/unused.h"\nfloat effect_x(void *arg0) {\n    return ((ParticleEffect *)arg0)->a;\n}\n'
        ),
    }
    result = retype.plan(files, {"ParticleEffect": "include/common/unused.h"}, retype._reaches(files))
    header = result["writes"]["include/span/mesh.h"]
    assert result["retyped"] == 1 and result["skipped"] == {}
    assert header == (
        "/* mesh */\nextern int other(int arg0);\nstruct ParticleEffect;\n"
        "extern float effect_x(struct ParticleEffect *arg0);\n"
    )
    assert "float effect_x(ParticleEffect *arg0)" in result["writes"]["src/effect.c"]


def test_a_cast_spelled_with_struct_also_declares_the_tag_and_adds_no_include():
    # real case: func_8028E770_de casts (struct Shared_World *)object; code_8028DF6C.h gained shared/world.h and the
    # sources that took the same include met another header's conflicting extern of D_8011FE88
    files = {
        "include/shared/world.h": "typedef struct Shared_World {\n    int mode;\n} Shared_World;\n",
        "include/span/sched.h": "extern void *slot(void *object, int index);\n",
        "src/slot.c": (
            '#include "shared/world.h"\nvoid *slot(void *object, int index) {\n'
            "    return &((struct Shared_World *)object)->mode + index;\n}\n"
        ),
    }
    declared = {"Shared_World": "include/shared/world.h", "struct Shared_World": "include/shared/world.h"}
    result = retype.plan(files, declared, retype._reaches(files))
    assert result["writes"]["include/span/sched.h"] == (
        "struct Shared_World;\nextern void *slot(struct Shared_World *object, int index);\n"
    )


def test_a_field_or_a_longer_tag_is_no_declaration_of_the_tag():
    # real case: code_8022BA90.h held "struct PlayerRecAt1450V2;" before func_8022BC14_de(struct Player *)
    code = "struct PlayerRecAt1450V2;\nstruct Holder {\n    struct Player *owner;\n};\n"
    assert not retype._declared_tag(code, "Player")
    assert retype._declared_tag(code + "struct Player;\n", "Player")
    assert retype._declared_tag("typedef struct Player Player;\n", "Player")


def test_tag_already_declared_in_the_header_is_not_declared_again():
    files = {
        "include/common/actor.h": TYPE,
        "include/span/code.h": "struct Actor;\nextern int fetch(void *arg0);\nextern int peek(void *arg0);\n",
        "src/fetch.c": '#include "common/actor.h"\nint fetch(void *arg0) {\n    return ((Actor *)arg0)->model;\n}\n',
        "src/peek.c": '#include "common/actor.h"\nint peek(void *arg0) {\n    return ((Actor *)arg0)->a;\n}\n',
    }
    writes = retype.plan(files, {"Actor": "include/common/actor.h"}, retype._reaches(files))["writes"]
    header = writes["include/span/code.h"]
    assert header == "struct Actor;\nextern int fetch(struct Actor *arg0);\nextern int peek(struct Actor *arg0);\n"


def test_planned_includes_never_close_a_cycle_between_two_retypes():
    # two retypes each add an include that is acyclic alone: unused.h -> b.h; b.h needs a.h's Pos, a.h needs Eff
    files = {
        "include/common/unused.h": '#include "span/b.h"\ntypedef struct { int a; } Eff;\n',
        "include/span/a.h": "typedef struct { int x; } Pos;\nextern int eff_a(void *arg0);\n",
        "include/span/b.h": "extern int pos_x(void *arg0);\n",
        "src/eff.c": '#include "common/unused.h"\nint eff_a(void *arg0) {\n    return ((Eff *)arg0)->a;\n}\n',
        "src/pos.c": '#include "span/a.h"\nint pos_x(void *arg0) {\n    return ((Pos *)arg0)->x;\n}\n',
    }
    declared = {"Eff": "include/common/unused.h", "Pos": "include/span/a.h"}
    result = retype.plan(files, declared, retype._reaches(files))
    assert result["retyped"] == 1
    assert result["skipped"] == {"declaring header includes the file that needs the type": 1}
    after = {**files, **result["writes"]}
    reach = retype._reaches(after)
    heads = [p for p in after if p.endswith(".h")]
    assert not any(a != b and reach(a, b) and reach(b, a) for a in heads for b in heads)


def test_other_uses_skip_but_bare_void_arguments_and_zero_compares_stay():
    files = texts(
        src__arith_c="void grow(void *arg0) {\n    arg0 = (char *)arg0 + 4;\n    ((Actor *)arg0)->a = 1;\n}\n",
        src__pass_c=(
            "extern void sink(void *, int);\nvoid pass(void *arg0) {\n    if (arg0 != 0) {\n"
            "        sink(arg0, 1);\n    }\n    ((Actor *)arg0)->a = 1;\n}\n"
        ),
    )
    result = run(files)
    assert result["skipped"] == {"parameter used outside the cast": 1}
    assert "void pass(Actor *arg0)" in result["writes"]["src/pass.c"]
    assert "src/arith.c" not in result["writes"]


def test_two_cast_targets_and_spellings_skip():
    files = texts(
        src__two_c="void two(void *arg0) {\n    ((Actor *)arg0)->a = 1;\n    ((Other *)arg0)->b = 1;\n}\n",
        src__spell_c="void spell(void *arg0) {\n    ((Actor *)arg0)->a = 1;\n    ((const Actor *)arg0)->a = 1;\n}\n",
    )
    assert run(files)["skipped"] == {"several cast targets": 1, "casts spell the type differently": 1}


def test_callers_that_would_need_a_cast_skip_the_parameter():
    files = texts(
        src__integer_c='#include "span/code.h"\nvoid go(int count) {\n    fetch(count);\n}\n',
        src__other_c='#include "span/code.h"\nvoid go2(Other *thing) {\n    touch(thing, 1);\n}\n',
    )
    result = run(files)
    assert result["retyped"] == 0
    assert result["skipped"] == {"caller passes an integer": 1, "caller passes another pointer type": 1}


def test_void_and_matching_callers_stay_retyped():
    files = texts(
        src__ok_c=(
            '#include "span/code.h"\nvoid go(void *raw, Actor *actor) {\n    fetch(raw);\n    fetch(actor);\n'
            "    fetch(0);\n    touch(raw, 1);\n}\n"
        )
    )
    assert run(files)["retyped"] == 2


def test_parameter_casts_counted_by_the_debt_rule():
    assert len(typedebt.param_cast(texts()["src/touch.c"])) == 1
    after = run(texts())["writes"]["src/touch.c"]
    assert typedebt.param_cast(after) == {}


def test_param_cast_row_is_the_function_so_retyping_one_parameter_adds_no_row():
    # real case: func_8020520C_de(void *source, void *dest) held one row per line; retyping source surfaced dest
    before = "void f(void *source, void *dest) {\n    ((A *)dest)->x = ((Actor *)source)->a;\n}\n"
    after = "void f(Actor *source, void *dest) {\n    ((A *)dest)->x = source->a;\n}\n"
    assert typedebt.param_cast(before) == typedebt.param_cast(after) == {1: "f()"}


def test_a_param_cast_row_withholds_its_function_never_its_callers():
    # real case: subject "f" (431057d) withheld every caller naming f, 1,132 B per version on G 68c8ebdd4
    from unbake import policy
    from unbake.contracts import Finding

    code = "void f(void *a) {\n    ((Actor *)a)->x = 1;\n}\nvoid g(void) {\n    f(0);\n}\n"
    ((line, subject),) = typedebt.param_cast(code).items()
    row = Finding("source.param-cast", "a parameter is cast: " + subject, path="src/u.c", line=line)
    assert policy.withheld(code, [row], ["f", "g"]) == {"f": ("source.param-cast",)}


def test_excluded_parameters_stay_and_touches_name_every_written_file():
    result = run(texts())
    assert result["touches"] == {
        ("fetch", 0): ["include/span/code.h", "src/fetch.c"],
        ("touch", 0): ["include/span/code.h", "src/touch.c"],
    }
    kept = run_excluding(texts(), {("fetch", 0)})
    assert kept["retyped"] == 1 and kept["skipped"] == {"its files would carry new debt": 1}
    assert "src/fetch.c" not in kept["writes"] and "fetch(void *arg0)" in kept["writes"]["include/span/code.h"]


def run_excluding(files: dict[str, str], exclude: set) -> dict:
    return retype.plan(
        files, {"Actor": "include/common/actor.h"}, lambda a, b: a == b or a == "include/span/code.h", exclude
    )


def test_an_invented_tag_is_never_declared_into_a_file_that_never_named_it():
    # real case: struct Brain_func_80212D78_eu_x; forward-declared into .c files added 13 fake-type rows
    owner = "typedef struct Brain_func_802_de { int a; } Brain_func_802_de;\n"
    files = {
        "include/common/brain.h": owner,
        "src/use.c": "extern int think(void *arg0);\n",
        "src/think.c": '#include "common/brain.h"\nint think(void *arg0) {\n'
        "    return ((Brain_func_802_de *)arg0)->a;\n}\n",
    }
    result = retype.plan(files, {"Brain_func_802_de": "include/common/brain.h"}, retype._reaches(files))
    assert result["retyped"] == 0 and result["skipped"] == {"declaration would name an invented type": 1}


def test_a_cast_around_the_cast_parameter_keeps_no_parentheses():
    # real case: ((Shared_ControlView *)((Player *)arg0)->profile) became (Shared_ControlView *)(arg0)->profile
    files = texts(
        src__view_c="int view(void *arg0) {\n"
        "    return ((View *)((Actor *)arg0)->model)->a + (s32)((Actor *)arg0)->a;\n}\n"
    )
    body = run(files)["writes"]["src/view.c"]
    assert "return ((View *)arg0->model)->a + (s32)arg0->a;" in body


def test_converge_drops_only_the_retypes_whose_files_carry_a_new_row():
    from unbake.contracts import Finding

    calls = []

    def planner(skip):
        calls.append(set(skip))
        return run_excluding(texts(), skip)

    def judge(writes):  # src/fetch.c gains a row while it is written
        return [Finding("source.fake-type", "x", path="src/fetch.c")] if "src/fetch.c" in writes else []

    result, rounds = retype.converge(planner, judge, set())
    assert calls == [set(), {("fetch", 0)}]
    assert result["retyped"] == 1 and "src/touch.c" in result["writes"]
    assert [r["dropped"] for r in rounds] == [1, 0]


def test_a_refusal_blames_the_retypes_whose_written_file_breaks_a_dependent():
    # real case: retype-params on G b13b9696d refused by 58 land.breaks_dependent rows, each on an unwritten unit,
    # naming the written breaker only in its reason ("src/code_802BAC64.c breaks src/func_802192C0_de_6.c: ...")
    touches = {("filter", 0): ["src/code_802BAC64.c"], ("fetch", 0): ["include/span/code.h", "src/fetch.c"]}
    rows = [
        {
            "path": "src/func_802192C0_de_6.c",
            "unit": "func_8021C9D8_de",
            "reason": "src/code_802BAC64.c breaks src/func_802192C0_de_6.c: version de: compile.error",
        }
    ]
    assert retype.blamed(rows, touches) == {("filter", 0)}
    assert retype.blamed([{"path": "include/span/code.h", "unit": "", "reason": "x"}], touches) == {("fetch", 0)}


def test_a_refusal_naming_only_a_member_blames_the_retypes_of_its_unit_file():
    # real case: rt2 on G b13b9696d ended on "func_8021B468_de does not match the ROM in every holding version"
    # (no path, no unit) in src/func_802192C0_de_4.c, a file a retype wrote
    touches = {("cb", 1): ["src/func_802192C0_de_4.c"], ("fetch", 0): ["src/fetch.c"]}
    rows = [{"path": "", "unit": "", "reason": "func_8021B468_de does not match the ROM in every holding version"}]
    owners = {"func_8021B468_de": "src/func_802192C0_de_4.c"}
    assert retype.blamed(rows, touches, owners.get) == {("cb", 1)}


def test_bisect_keeps_every_half_that_proves_and_halves_refused_ones_down_to_one():
    # real case: rt2 on G b13b9696d dropped 1-3 retypes per 50 s round for 63 rounds; a bisection proves halves
    bad = {("c", 0), ("f", 0)}
    calls = []

    def prove(halves):
        calls.append(len(halves))
        return [not (set(h) & bad) for h in halves]

    keys = [(n, 0) for n in "abcdefgh"]
    kept, levels = retype.bisect(keys, prove)
    assert kept == set(keys) - bad
    assert calls == [2, 4, 4]  # one call per level: every half of a level proved in one dispatch
    assert [lv["left"] for lv in levels] == [8, 4, 2]


def test_compose_writes_any_subset_of_one_plan_as_planning_it_alone_would():
    files = texts()
    whole = run(files)
    for key in whole["touches"]:
        alone = retype.compose(files, whole["parts"], [key])
        other = run_excluding(files, set(whole["touches"]) - {key})
        assert alone == other["writes"]


def test_an_integer_parameter_fed_only_addresses_is_retyped_and_the_callers_lose_their_casts():
    # real case: address-int, `(s32)&p->f` passed to an s32 parameter the callee only casts to one pointer type
    files = {
        "include/common/actor.h": TYPE,
        "include/span/code.h": '#include "common/actor.h"\nextern void put(s32 arg0);\n',
        "src/put.c": '#include "span/code.h"\nvoid put(s32 arg0) {\n    *(f32 *)arg0 = 1.0f;\n}\n',
        "src/call.c": '#include "span/code.h"\nvoid call(Actor *a) {\n    put((s32)&a->model);\n    put(0);\n}\n',
    }
    result = run(files)
    assert result["retyped"] == 1
    assert "void put(f32 *arg0)" in result["writes"]["src/put.c"] and "*arg0 = 1.0f" in result["writes"]["src/put.c"]
    assert "put(&a->model);" in result["writes"]["src/call.c"]
    assert "extern void put(f32 *arg0);" in result["writes"]["include/span/code.h"]


def test_an_integer_parameter_a_caller_feeds_a_plain_integer_stays():
    files = {
        "include/span/code.h": "extern void put(s32 arg0);\n",
        "src/put.c": '#include "span/code.h"\nvoid put(s32 arg0) {\n    *(f32 *)arg0 = 1.0f;\n}\n',
        "src/call.c": '#include "span/code.h"\nvoid call(s32 n) {\n    put(n);\n}\n',
    }
    result = run(files)
    assert result["retyped"] == 0 and result["skipped"] == {"caller passes an integer": 1}


def test_a_respelled_address_int_row_is_the_same_row():
    # real case: rt1 dropped 4 retypes whose `(s32)&((T *)a)->f` became `(s32)&a->f`, read as new debt
    before = "void f(void *a) {\n    g((s32)&((Actor *)a)->x);\n}\n"
    after = "void f(Actor *a) {\n    g((s32)&a->x);\n}\n"
    assert typedebt.address_int(before) == typedebt.address_int(after) == {2: "(s32)&f()"}
    assert typedebt.address_int("s32 table[] = {(s32)&D_80001000};\n") == {1: "(s32)&D_80001000"}


@pytest.mark.parametrize(
    ("site", "edit"),
    [
        ("*(s32 *)((u8 *)p + 4) = 1;", "p->b = 1;"),
        ("g((char *)p + 0x8);", "g((char *)&p->v);"),
        ("g((u8 *)(p) + 0xA);", "g((u8 *)&p->v.y);"),
        ("*(s16 *)((u8 *)p + 4) = 1;", None),  # no s16 member at 4: a type pun is never written
        ("g((u8 *)q + 4);", None),  # q is no typed record pointer
    ],
)
def test_raw_offsets_off_a_typed_record_pointer_become_its_members(site, edit):
    v = {"offset": 8, "width": 4, "type": "Vec", "name": "v", "fields": []}
    v["fields"] = [
        {"offset": 0, "width": 2, "type": "s16", "name": "x"},
        {"offset": 2, "width": 2, "type": "s16", "name": "y"},
    ]
    rec = {
        "size": 12,
        "fields": [
            {"offset": 0, "width": 4, "type": "s32", "name": "a"},
            {"offset": 4, "width": 4, "type": "s32", "name": "b"},
            v,
        ],
    }
    code = "void f(Rec *p, void *q) {\n    " + site + "\n}\n"
    found, _ = retype.raw_offsets(code, {"Rec": rec})
    out = code
    for first, last, new in sorted(found, reverse=True):
        out = out[:first] + new + out[last:]
    assert (out.splitlines()[1].strip() if found else None) == edit
