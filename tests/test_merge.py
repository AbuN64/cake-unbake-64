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
    assert kinds == [("extern", "count")]
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


def test_a_prototype_after_a_definition_of_another_type_gives_way_and_is_listed() -> None:
    one = "void f(int a) { }\n"
    two = "void f(char *a);\nvoid use(void) { f(0); }\n"
    out = merge.merge([("one.c", one), ("two.c", two)], never)
    assert out["conflicts"] == [] and "void f(char *a);" not in out["text"]
    assert len(out["retyped"]) == 1 and out["retyped"][0].startswith("f:")


def test_a_prototype_before_its_definition_of_another_type_is_still_a_conflict() -> None:
    out = merge.merge([("one.c", "void f(char *a);\n"), ("two.c", "void f(int a) { }\n")], never)
    assert [(c["kind"], c["name"]) for c in out["conflicts"]] == [("proto", "f")] and out["retyped"] == []


def test_typedef_names_and_their_types_read_alike_and_an_empty_parameter_list_matches_any() -> None:
    one = "typedef unsigned int u32;\ntypedef int s32;\ntypedef float f32;\nextern f32 D_1;\nu32 f(s32 a);\nint g();\n"
    two = "extern float D_1;\nunsigned int f(int a) { }\nint g(void *p) { }\n"
    out = merge.merge([("one.c", one), ("two.c", two)], never)
    assert out["conflicts"] == [] and out["dropped"] == 3


def test_typedefs_of_included_headers_count_and_unlike_types_still_conflict() -> None:
    headers = {"t.h": "typedef unsigned short u16;\ntypedef struct Tag { int x; } Named;\n"}
    one = '#include "t.h"\nextern u16 D_1;\nextern Named D_2;\nextern u16 D_3;\n'
    two = "extern unsigned short D_1;\nextern struct Tag D_2;\nextern unsigned char D_3;\n"
    out = merge.merge([("one.c", one), ("two.c", two)], never, headers.get)
    assert [(c["kind"], c["name"]) for c in out["conflicts"]] == [("extern", "D_3")]


def test_names_that_disagree_the_same_way_are_one_finding_listing_them() -> None:
    one = "".join(f"extern float K_{i};\n" for i in range(3))
    two = "".join(f"extern int K_{i};\n" for i in range(3))
    out = merge.merge([("one.c", one), ("two.c", two)], never)
    assert len(out["conflicts"]) == 1 and out["conflicts"][0]["name"] == "K_0..K_2 (3 names)"


def test_names_declared_together_are_each_checked_and_the_statement_is_kept() -> None:
    headers = {"h.h": "extern int a, b[4];\n"}
    two = "extern int a;\nextern short b[4];\nint x, *y;\nextern int x;\n"
    out = merge.merge([("one.c", '#include "h.h"\n'), ("two.c", two)], never, headers.get)
    assert [(c["kind"], c["name"]) for c in out["conflicts"]] == [("extern", "b")]
    assert "extern int a;" not in out["text"]


def test_a_guarded_declaration_of_another_type_than_a_header_declaration_is_a_conflict() -> None:
    headers = {"h.h": "extern char D_1;\n"}
    two = "#if V\nextern char D_1[];\n#endif\n"
    out = merge.merge([("one.c", '#include "h.h"\n'), ("two.c", two)], never, headers.get)
    assert [(c["kind"], c["name"]) for c in out["conflicts"]] == [("extern", "D_1")]
    assert "extern char D_1[];" in out["text"]


ORDER_A = "int a_text(void) { return 1; }\nint a_data = 1;\n"
ORDER_B = "int b_text(void) { return 2; }\nint b_data = 2;\n"
ORDER_C = "int c_text(void) { return 3; }\nint c_data = 3;\n"
ORDER_CLASSES = {u: {f"{u}_text": "text", f"{u}_data": "data"} for u in "abc"}


def merge_orders(data: dict[str, list[str]]) -> str:
    out = merge.merge(
        [("a", ORDER_A), ("b", ORDER_B), ("c", ORDER_C)],
        never,
        orders={"text": {v: ["a", "b", "c"] for v in data}, "data": data},
        classes=ORDER_CLASSES,
        guards={v: f"VERSION_{v.upper()}" for v in data},
    )
    assert out["conflicts"] == []
    return out["text"]


def test_order_a_unit_one_version_links_elsewhere_is_emitted_again_under_that_versions_guard() -> None:
    # the real rows: one unit sits two places away in one version (eu), the other versions link the base order
    text = merge_orders({"de": ["a", "b", "c"], "eu": ["c", "a", "b"], "us": ["a", "b", "c"]})
    assert "#if defined(VERSION_EU)\nint c_data = 3;\n#endif" in text
    assert "#if !(defined(VERSION_EU))\nint c_data = 3;\n#endif" in text
    assert text.index("#if defined(VERSION_EU)") < text.index("a_data")
    assert text.index("#if !(defined(VERSION_EU))") > text.index("b_data")
    assert text.count("c_text") == 1


def test_order_versions_that_agree_emit_no_guard() -> None:
    text = merge_orders({"de": ["c", "b", "a"], "eu": ["c", "b", "a"]})
    assert "#if" not in text
    assert text.index("c_data") < text.index("b_data") < text.index("a_data")


def test_order_versions_that_move_a_unit_to_the_same_place_share_one_guard() -> None:
    text = merge_orders({"de": ["a", "b", "c"], "eu": ["c", "a", "b"], "us": ["c", "a", "b"]})
    assert "#if defined(VERSION_EU) || defined(VERSION_US)\nint c_data = 3;\n#endif" in text
