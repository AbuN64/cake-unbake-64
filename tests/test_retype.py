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
