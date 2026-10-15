"""Parameter retyping over cut text: what is retyped, what is skipped and why, and the files each change reaches."""

from __future__ import annotations

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
    assert "extern int fetch(Actor *);" in writes["src/use.c"]
    assert "extern void touch(Actor *, int arg1);" in writes["src/use.c"]


def test_header_without_the_type_includes_its_owner():
    files = texts()
    files["include/span/code.h"] = "extern int fetch(void *arg0);\n"
    writes = retype.plan(files, {"Actor": "include/common/actor.h"}, lambda a, b: False)["writes"]
    assert writes["include/span/code.h"].startswith('#include "common/actor.h"\n')


def test_header_cycle_skips_the_parameter():
    files = texts()
    result = retype.plan(files, {"Actor": "include/common/actor.h"}, lambda a, b: a == "include/common/actor.h")
    assert result["skipped"] == {"declaring header includes the file that needs the type": 2}
    assert result["writes"] == {}


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
