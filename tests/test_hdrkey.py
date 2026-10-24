"""hdrkey, the compile key of a unit: a declaration the unit never names is not in it."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from unbake import config as configuration

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler, as the generated Makefile does")


def source() -> str:
    text = configuration.template("Makefile.in")
    return text.split("define HDRKEY_C\n", 1)[1].split("\nendef", 1)[0] + "\n"


@pytest.fixture(scope="session")
def hdrkey(tmp_path_factory) -> Path:
    where = tmp_path_factory.mktemp("hdrkey")
    (where / "hdrkey.c").write_text(source())
    subprocess.run(["cc", "-O2", "-Wall", "-Werror", "-o", str(where / "hdrkey"), str(where / "hdrkey.c")], check=True)
    return where / "hdrkey"


@pytest.fixture
def key(hdrkey, tmp_path):
    def run(text: str) -> str:
        unit = tmp_path / "unit.i"
        unit.write_text(text)
        return subprocess.run([str(hdrkey), str(unit)], check=True, capture_output=True, text=True).stdout

    return run


def test_embedded_source_has_no_dollar():
    assert "$" not in source()


def test_unused_extern_variable_is_dropped_and_used_one_kept(key):
    out = key("extern int a;\nextern int b;\nint f(void) { return b; }\n")
    assert "extern int b;" in out and "extern int a;" not in out


def test_unused_prototype_dropped_called_one_kept(key):
    out = key("int p(int);\nint q(void);\nint f(void) { return q(); }\n")
    assert "int q(void);" in out and "int p(int);" not in out


def test_typedef_kept_when_a_kept_declaration_names_it(key):
    out = key("typedef int T1;\ntypedef int T2;\nextern T1 g;\nint f(void) { return g; }\n")
    assert "typedef int T1;" in out and "T2" not in out


def test_reachability_is_transitive(key):
    out = key(
        "struct C { int z; };\nstruct B { struct C c; };\nstruct D { int d; };\ntypedef struct B BT;\n"
        "extern BT g;\nint f(void) { return g.c.z; }\n"
    )
    assert "struct C" in out and "struct B" in out and "struct D" not in out


def test_struct_and_forward_declaration_by_tag(key):
    out = key(
        "struct U { int a; };\nstruct S;\nstruct V { int b; };\nstruct S *p;\nint f(struct V *v) { return v->b; }\n"
    )
    assert "struct V { int b; };" in out and "struct S;" in out and "struct U" not in out


def test_enumerator_use_keeps_its_enum(key):
    out = key("enum E1 { A1, B1 };\nenum E2 { A2, B2 };\nint f(void) { return B2; }\n")
    assert "enum E2" in out and "enum E1" not in out


def test_variable_definitions_are_always_kept(key):
    out = key("int g = 3;\nstatic int s;\nint t;\nint f(void) { return 0; }\n")
    assert "int g = 3;" in out and "static int s;" in out and "int t;" in out


def test_pragma_and_functions_kept_in_order(key):
    out = key("#pragma intrinsic(x)\nint f(void) { return 1; }\nint g(void) { return 2; }\n")
    assert out.index("#pragma") < out.index("int f") < out.index("int g")


def test_function_pointer_array_and_multi_declarator_externs(key):
    out = key(
        "extern void (*fp)(int);\nextern int arr[3];\nextern int a, b[2], fn(int);\nextern int (*tbl[4])(void);\n"
        "int f(void) { return arr[0] + b[1]; }\n"
    )
    assert "arr[3]" in out and "b[2]" in out and "fp" not in out and "tbl" not in out
    assert "int a, b[2], fn(int);" in out  # one declaration, kept whole when any of its names is used


def test_prototype_with_typedef_parameter_does_not_pin_the_typedef(key):
    out = key("typedef int T;\nextern void h(T);\nint f(void) { return 0; }\n")
    assert out.strip() == "int f(void) { return 0; }"


def test_attribute_packed_typedef(key):
    out = key(
        "typedef struct { char a; int b; } __attribute__((packed)) P;\nextern P v;\nint f(void) { return v.b; }\n"
    )
    assert "packed" in out and "extern P v;" in out


def test_every_declaration_of_a_used_name_is_kept(key):
    out = key("extern int x;\nextern int y;\nextern int x;\nint f(void) { return x; }\n")
    assert out.count("extern int x;") == 2 and "extern int y;" not in out


def test_static_prototype_dropped_unless_defined(key):
    out = key("static int u(int);\nstatic int d(int);\nstatic int d(int a) { return a; }\n")
    assert "static int u" not in out and "static int d(int);" in out


def test_changing_a_referenced_declaration_changes_the_key(key):
    a = key("typedef int T;\nextern T g;\nint f(void) { return g; }\n")
    b = key("typedef short T;\nextern T g;\nint f(void) { return g; }\n")
    assert a != b


def test_changing_a_used_struct_member_changes_the_key(key):
    base = "struct S { %s };\nextern struct S s;\nint f(void) { return s.a; }\n"
    assert key(base % "int a;") != key(base % "int a; int b;")


def test_changing_an_unreferenced_declaration_keeps_the_key(key):
    base = "typedef int T;\nextern %s;\nint f(void) { return 0; }\n"
    assert key(base % "int unused_a") == key(base % "long unused_b[4]")


def test_adding_a_declaration_to_an_unused_struct_keeps_the_key(key):
    base = "struct S { int a; %s};\nint f(void) { return 0; }\n"
    assert key(base % "") == key(base % "int b; ")


def test_whitespace_and_blank_lines_are_not_part_of_the_key(key):
    assert key("int f(void)\n{\n\n  return 1 + 2;\n}\n") == key("int f(void) { return 1 + 2; }")


def test_adjacent_tokens_are_not_merged(key):
    assert key("int f(int a, int b) { return a--b; }\n") != key("int f(int a, int b) { return a- -b; }\n")
    assert key("int f(int a) { return a+ +1; }\n") != key("int f(int a) { return a++1; }\n")


def test_string_contents_count(key):
    assert key('char *s(void) { return "a b"; }\n') != key('char *s(void) { return "a  b"; }\n')


def test_function_body_change_changes_the_key(key):
    assert key("int f(void) { return 1; }\n") != key("int f(void) { return 2; }\n")


def test_kr_definition_makes_the_whole_unit_the_key(key):
    text = "extern int unused;\nint f(a) int a; { return a; }\n"
    out = key(text)
    assert out.startswith("RAW\n") and "extern int unused;" in out


def test_unbalanced_unit_makes_the_whole_unit_the_key(key):
    assert key("int f(void) { return 1; \n").startswith("RAW\n")


def test_struct_returning_function_with_struct_parameter(key):
    out = key(
        "struct R { int a; };\nstruct Q { int b; };\n"
        "struct R mk(struct Q *q) { struct R r; r.a = q->b; return r; }\nextern int z;\n"
    )
    assert "struct R" in out and "struct Q" in out and "extern int z" not in out


def test_unused_declaration_inside_function_body_is_a_reference(key):
    out = key("extern int a;\nint f(void) { extern int a; return a; }\n")
    assert "extern int a;\nint f" in out


def test_initialised_extern_is_a_definition_and_kept(key):
    out = key("extern int a = 3;\nint f(void) { return 0; }\n")
    assert "extern int a = 3;" in out
