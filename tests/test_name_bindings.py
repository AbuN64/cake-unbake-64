"""Canonical declaration facts retain every scope, domain and tracked directory view."""

import pytest

from unbake import api, symbol_names


def test_nine_domains_and_shadowed_bindings_are_distinct():
    text = """#define APPLY(value) (value)
typedef struct Record { int member; } Record;
enum Mode { Off, On = 2 };
static int held;
int global;
int Function(int arg0, int meaningful) {
  int var_s0, named;
  { int var_s0; var_s0 = 1; }
  for (int i = 0; i < 2; ++i) { named += i; }
  return meaningful;
}
"""
    rows = api.name_bindings(text)
    assert {b.kind for b in rows} == {
        "function",
        "global",
        "static",
        "type",
        "member",
        "parameter",
        "local",
        "macro",
        "enumerator",
    }
    shadows = [b for b in rows if b.name == "var_s0"]
    assert len(shadows) == 2 and shadows[0].scope != shadows[1].scope
    assert len({b.start for b in rows}) == len(rows)
    assert all(text[b.start : b.start + len(b.name)] == b.name for b in rows)
    assert {b.name for b in rows if b.kind == "enumerator"} == {"Off", "On"}
    assert any(b.name == "meaningful" and b.kind == "parameter" for b in rows)
    assert any(b.name == "named" and b.kind == "local" for b in rows)


def test_generated_guards_and_header_folders_share_the_path_universe():
    texts = {
        "include/MENU_2/EXTERNS.H": (
            "#ifndef MENU_80200000_H\n#define MENU_80200000_H\nextern int D_blob_80200000;\n#endif\n"
        ),
        "src/game/MENU/a.c": "int func_80200000(void) { return 0; }\n",
        "src/sdk/a.s": "glabel assembly_function\n",
    }
    got = symbol_names.census(texts, "a" * 40)
    assert {p["path"] for p in got["paths"]} == set(texts)
    assert "include/MENU_2" in got["folders"]["header"]
    assert "include/MENU_2" not in got["folders"]["source"]
    assert [d["name"] for d in got["declarations"] if d["guard"]] == ["MENU_80200000_H"]
    assert any(d["name"] == "D_blob_80200000" for d in got["declarations"])
    assert got["semantic_acceptance"] is False
    assert symbol_names.census(dict(reversed(list(texts.items()))), "a" * 40) == got


def test_inactive_definitions_and_pointer_parameters_do_not_merge():
    text = "#if FIRST\nint F(int *arg0) { return *arg0; }\n#else\nint F(int *arg0) { return 1; }\n#endif\n"
    rows = api.name_bindings(text)
    assert len([b for b in rows if b.kind == "function" and b.name == "F"]) == 2
    assert len([b for b in rows if b.kind == "parameter" and b.name == "arg0"]) == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("extern int value;", {"value": False}),
        ("const extern int value[];", {"value": False}),
        ("extern int *value;", {"value": False}),
        ("extern int (*value)(int);", {"value": False}),
        ("extern int (*value)(int) = 0;", {"value": True}),
        ("extern int value = 1;", {"value": True}),
        ("extern int value[] = {1, 2};", {"value": True}),
        ('extern char *value = "extern,=";', {"value": True}),
        ("extern int value /* = 1 */;", {"value": False}),
        ("extern int a, b = 2, c;", {"a": False, "b": True, "c": False}),
        ("extern int a = Call(1, 2), b;", {"a": True, "b": False}),
        ("extern int a[sizeof(int[2])], b = 1;", {"a": False, "b": True}),
        ("int value;", {"value": True}),
        ("static int value;", {"value": True}),
        ('extern "C" { int a; extern int b; }', {"a": True, "b": False}),
        ('extern "C" int value;', {"value": False}),
        ("extern struct Record { int member; } value;", {"value": False}),
        ("struct Record { int member; } value;", {"value": True}),
        ("void F(void) { extern int a; int b; }", {"a": False, "b": True}),
        ("extern int F(void); extern int G(void) { return 1; }", {"F": False, "G": True}),
    ],
)
def test_definition_belongs_to_each_declarator(text, expected):
    rows = api.name_bindings(text)
    assert {b.name: b.definition for b in rows if b.name in expected} == expected
    assert all(text[b.start : b.start + len(b.name)] == b.name for b in rows)


def test_real_extern_census_keeps_header_source_and_conditional_identities():
    # Source 482b437b: COMMON/LONE/D_80152789.H and PAK_OPENMUSICPREVIEWMENU.CPP.
    texts = {
        "include/language.h": '#include "TYPES.H"\nextern u8 g_languageIndex;\n',
        "src/preview.cpp": (
            "#if defined(VERSION_EU) || defined(VERSION_EU_X)\n"
            "extern u8 g_languageIndex[];\nextern char D_800EEE20[];\n#endif\n"
        ),
        "src/language.cpp": "#if VERSION_EU\nu8 g_languageIndex = 0;\n#else\nu8 g_languageIndex = 1;\n#endif\n",
    }
    result = symbol_names.census(texts, "482b437b9a553d554de3ddaa68add8bfb867fbdf")
    rows = result["declarations"]
    assert len(rows) == 5
    assert sum(row["definition"] for row in rows) == 2
    assert all(row["definition"] == (row["path"] == "src/language.cpp") for row in rows)
    assert len({(*row["identity"][:2], row["offset"]) for row in rows}) == 5
    assert result["semantic_acceptance"] is False
    assert symbol_names.census(texts, result["revision"]) == result
