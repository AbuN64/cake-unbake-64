"""Canonical declaration facts retain every scope, domain and tracked directory view."""

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
