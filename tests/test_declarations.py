"""The live part of a preprocessed unit: a header declaration the unit never reaches is no part of its object."""

import pytest

from unbake import declarations
from unbake.contracts import digest

SOURCE = "src/unit.c"


def unit(header: str, body: str, path: str = "include/h.h") -> str:
    """Preprocessed text as the preprocessor gives it: a header, then the unit's own lines."""
    return f'# 1 "{path}"\n{header}\n# 1 "/root/{SOURCE}"\n{body}\n'


def live(header: str, body: str = "int f(void) { return 1; }") -> str:
    return declarations.live(unit(header, body), SOURCE)


BASE = "typedef int A;\ntypedef struct B { A x; } B;\ntypedef struct C { int y; } C;"
USES_B = "int f(B *p) { return p->x; }"


def test_an_unused_declaration_changes_nothing_the_unit_reads():
    edited = BASE + "\ntypedef struct D { int z; } D;\nextern int unused_global;\nvoid unused_proto(int);"
    assert live(BASE, USES_B) == live(edited, USES_B)


def test_a_changed_declaration_the_unit_uses_changes_what_it_reads():
    assert live(BASE, USES_B) != live(BASE.replace("A x;", "A x, w;"), USES_B)


def test_a_declaration_reached_through_another_is_used():
    # f names B; B names A; so editing A is an edit of what f reads, while C stays out of it
    assert live(BASE, USES_B) != live(BASE.replace("typedef int A;", "typedef long A;"), USES_B)
    assert live(BASE, USES_B) == live(BASE.replace("int y;", "long y;"), USES_B)


def test_line_markers_comments_and_blank_lines_are_not_read():
    shifted = unit("\n\n" + BASE, USES_B, path="build/views/abc/include/h.h")
    assert declarations.live(unit(BASE, USES_B), SOURCE) == declarations.live(shifted, SOURCE)


def test_enumerators_and_function_pointers_are_found_by_their_names():
    header = "enum Mode { OFF, ON = 2 };\ntypedef void (*Hook)(int);\ntypedef enum Mode Mode;"
    assert live(header, "int f(void) { return ON; }") != live(
        header.replace("ON = 2", "ON = 3"), "int f(void) { return ON; }"
    )
    assert live(header, "Hook h;") != live(header.replace("(int)", "(long)"), "Hook h;")
    assert live(header, "int g;") == live(header.replace("(int)", "(long)"), "int g;")


def test_what_emits_code_or_data_is_always_read():
    header = "int table[2] = { 1, 2 };\nstatic int twice(int v) { return v * 2; }\n#pragma pack(1)\nint zero;"
    for edit in ("{ 1, 3 }", "v * 3", "pack(2)", "int zero2;"):
        old = header
        new = (
            header.replace("{ 1, 2 }", edit)
            if edit.startswith("{")
            else header.replace({"v * 3": "v * 2", "pack(2)": "pack(1)", "int zero2;": "int zero;"}[edit], edit)
        )
        assert live(old) != live(new)


def test_the_units_own_text_is_always_read():
    assert live(BASE, "typedef int Own;") != live(BASE, "typedef long Own;")


@pytest.mark.parametrize("broken", ["struct B { int x;", "int f(void) { }}", "int f(int a;"])
def test_a_text_that_does_not_split_stays_whole(broken):
    text = unit(broken, "int g;")
    assert declarations.live(text, SOURCE) == text


def test_the_digest_follows_the_live_text_only():
    assert digest(live(BASE, USES_B)) == digest(live(BASE + "\ntypedef char Unused;", USES_B))


def test_a_declaration_running_into_the_units_text_keeps_the_units_text():
    assert "int g;" in declarations.live(unit("typedef int A", "int g;"), SOURCE)


def test_dead_keeps_statements_used_by_other_headers():
    texts = {
        "include/a.h": "/* m2c generated */\nstruct Unused;\ntypedef int Shared;\ntypedef int Gone;\n",
        "include/b.h": "extern Shared g;\n",
    }
    out = declarations.dead(texts, {"g"})
    assert out == {"include/a.h": "typedef int Shared;\n"}
    assert declarations.dead(texts, {"g", "Gone", "Unused"}) == {
        "include/a.h": "struct Unused;\ntypedef int Shared;\ntypedef int Gone;\n"
    }


def test_dead_drops_a_group_of_declarations_that_only_name_each_other():
    texts = {"include/a.h": "typedef struct A { struct B *b; } A;\ntypedef struct B { A *a; } B;\ntypedef int Used;\n"}
    assert declarations.dead(texts, {"Used"}) == {"include/a.h": "typedef int Used;\n"}
    assert declarations.dead(texts, {"B"}) == {"include/a.h": texts["include/a.h"].rsplit("\n", 2)[0] + "\n"}
