"""Units into one source file: the pure text rule, with plain strings and no compiler."""

from __future__ import annotations

from unbake import merge


def never(unit: str, symbol: str, size: int, const: bool) -> bool:
    return False


def always(unit: str, symbol: str, size: int, const: bool) -> bool:
    return True


A = """#include "a.h"
#include <b.h>
void f(void);
static void g(void);
extern u32 D_1[4];
extern s32 count;
void f(void) { g(); }
"""
B = """#include "a.h"
#if X
#include "x.h"
#endif
void f(void);
extern u32 D_1[4];
s32 count = 1;
static void g2(int a);
void h(int a) { f(); }
"""


def test_includes_are_hoisted_once_and_conditional_ones_stay_where_they_are() -> None:
    out = merge.merge([("a.c", A), ("b.c", B)], never)
    head = out["text"].split("\n\n")[0]
    assert head == '#include "a.h"\n#include <b.h>' and out["includes"] == 2
    assert out["text"].count("#include") == 3 and '#include "x.h"' in out["text"]


def test_a_prototype_of_a_function_the_file_defines_is_dropped() -> None:
    out = merge.merge([("a.c", A), ("b.c", B)], never)
    assert "void f(void);" not in out["text"] and out["text"].count("void f(void) { g(); }") == 1
    assert "static void g(void);" in out["text"]


def test_a_repeated_extern_is_kept_once_and_an_extern_of_a_definition_is_dropped() -> None:
    out = merge.merge([("a.c", A), ("b.c", B)], never)
    assert out["text"].count("extern u32 D_1[4];") == 1
    assert "extern s32 count;" not in out["text"] and "s32 count = 1;" in out["text"]
    assert out["conflicts"] == [] and out["dropped"] == 4


def test_declarations_that_disagree_are_reported_not_edited() -> None:
    one = "extern u32 D_1[4];\nu32 f(int a);\nvoid use(void) { f(1); }\n"
    two = "extern u16 D_1[4];\nvoid f(int a) { }\n"
    out = merge.merge([("one.c", one), ("two.c", two)], never)
    kinds = sorted((c["kind"], c["name"]) for c in out["conflicts"])
    assert kinds == [("extern", "D_1"), ("proto", "f")]
    assert "extern u32 D_1[4];" in out["text"] and "extern u16 D_1[4];" in out["text"]


def test_an_extern_of_unknown_size_matches_the_definition() -> None:
    out = merge.merge([("a.c", "extern u8 t[];\n"), ("b.c", "u8 t[3] = {1, 2, 3};\n")], never)
    assert out["conflicts"] == [] and "extern u8 t[];" not in out["text"]


def test_two_statics_of_one_name_are_a_conflict() -> None:
    out = merge.merge([("a.c", "static int x = 1;\n"), ("b.c", "static int x = 2;\n")], never)
    assert [(c["kind"], c["name"]) for c in out["conflicts"]] == [("static", "x")]


def test_small_zero_pads_the_caller_accepts_are_dropped_with_their_empty_guard() -> None:
    text = (
        "#if V\nconst u8 pad_1[3] = {0, 0, 0};\n#endif\nconst u8 pad_2[4] = {0x0, 0};\n"
        "const u8 data[3] = {0, 1, 0};\nconst u8 big[16] = {0};\n"
    )
    out = merge.merge([("a.c", text)], always)
    assert out["pads"] == ["pad_1", "pad_2"]
    assert "#if V" not in out["text"] and "data[3]" in out["text"] and "big[16]" in out["text"]
    assert merge.merge([("a.c", text)], never)["pads"] == []


def test_guarded_declarations_are_left_alone() -> None:
    one = "#if X\nvoid f(void);\n#endif\n"
    out = merge.merge([("a.c", one), ("b.c", "void f(void) { }\n")], never)
    assert "void f(void);" in out["text"] and out["conflicts"] == []


def test_comments_and_strings_do_not_confuse_the_declaration_scan() -> None:
    text = '/* void f(void); { */\nconst char *s = "}; void f(void);";\nvoid f(void) { }\n'
    out = merge.merge([("a.c", text)], never)
    assert out["conflicts"] == [] and out["dropped"] == 0


def test_a_struct_tag_keyword_does_not_make_two_spellings_of_one_type_differ() -> None:
    one = "void f(struct View *v);\nextern struct View D_2;\n"
    two = "void f(View *actor) { }\nView D_2;\n"
    out = merge.merge([("a.c", one), ("b.c", two)], never)
    assert out["conflicts"] == [] and out["dropped"] == 2


HEADERS = {
    "a.h": '#ifndef A_H\n#define A_H\n#include "b.h"\nextern u32 D_1[4];\nvoid f(int a);\n#endif\n',
    "b.h": "#ifndef B_H\n#define B_H\nextern s32 count;\n#endif\n",
}


def test_declarations_are_checked_against_the_included_headers_followed_through_their_own_includes() -> None:
    one = '#include "a.h"\nvoid f(int a) { }\n'
    two = "extern u32 D_1[4];\nextern u16 count;\nvoid f(char a);\nvoid use(void) { f(1); }\n"
    out = merge.merge([("one.c", one), ("two.c", two)], never, HEADERS.get)
    kinds = sorted((c["kind"], c["name"]) for c in out["conflicts"])
    assert kinds == [("extern", "count"), ("proto", "f")]
    assert "extern u32 D_1[4];" not in out["text"]


def test_a_definition_against_a_header_prototype_of_another_type_is_a_conflict() -> None:
    out = merge.merge([("one.c", '#include "a.h"\n'), ("two.c", "void f(char a) { }\n")], never, HEADERS.get)
    assert [(c["kind"], c["name"]) for c in out["conflicts"]] == [("proto", "f")]


def test_a_header_whose_declarations_stand_inside_a_second_include_guard_is_still_checked() -> None:
    guard = "#ifndef G_H\n#define G_H\n{}#endif\n"
    inner = "#ifndef G_INNER\n#define G_INNER\nextern void f(struct T *a);\n#endif\n"
    headers = {"g.h": guard.format("struct T;\n" + inner)}
    out = merge.merge([("one.c", '#include "g.h"\n'), ("two.c", "extern void f(void *a);\n")], never, headers.get)
    assert [(c["kind"], c["name"]) for c in out["conflicts"]] == [("proto", "f")]
