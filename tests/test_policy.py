"""Policy contract checks without native tools, workers, or project setup."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from unbake import policy, tree
from unbake.contracts import Config, Finding, Project, Snapshot, SourceView

PATH = "src/unit.c"


@pytest.fixture
def snapshot(monkeypatch):
    monkeypatch.setattr(policy.effort, "stage", Mock(side_effect=lambda name: nullcontext()))
    result = Mock(spec=Snapshot)
    result.config = Mock(spec=Config)
    result.config.project = Mock(spec=Project)
    result.config.project.names_from = "a"
    result.config.project.libraries = ()
    result.config.project.digest = "project"
    result.overlays = {}
    result.versions = {}
    monkeypatch.setattr(policy, "_tree", lambda snap: tree.Tree("t", ()))
    monkeypatch.setattr(policy, "_include_headers", lambda snap: [])
    result.peek = lambda path: None
    result.digest = "snap"
    return result


def _source(text="", dependencies=()):
    return SourceView(PATH, "a", "view-key", text, dependencies, ())


@pytest.mark.parametrize(
    "address",
    ["(0xA0000000 | 0x04600010)", "0x04000000", "0x04FFFFFF", "0xA4000010UL", "83886080", "0x1FC00000", "0xBFCFFFFF"],
)
def test_hardware_cast_allowed(snapshot, monkeypatch, address):
    text = f"*(volatile unsigned int *){address} = x;"
    source = _source(text)
    active = Mock(return_value={7: text})
    monkeypatch.setattr(policy._view, "active_lines", active)
    assert policy.evaluate(snapshot, PATH, "\n" * 6 + "IO_WRITE(reg, x);", source, False) == ()
    active.assert_called_once_with(source, PATH)


@pytest.mark.parametrize("with_view", [False, True])
def test_ram_cast_and_local_denied(snapshot, monkeypatch, with_view):
    text = "(volatile int *)0x80100000;\nvolatile s32 x;"
    source = _source(text) if with_view else None
    monkeypatch.setattr(policy._view, "active_lines", Mock(return_value=dict(enumerate(text.splitlines(), 1))))
    findings = policy.evaluate(snapshot, PATH, text, source, False)
    sentence = next(
        r["sentence"] for r in policy.config.load_resource("rules.toml")["rule"] if r["id"] == "volatile-storage"
    )
    assert findings == tuple(
        Finding("source.volatile-storage", f"{sentence}: {line}", path=PATH, line=n)
        for n, line in enumerate(text.splitlines(), 1)
    )


@pytest.mark.parametrize(
    "text",
    [
        "volatile int x;",
        "void f(volatile int x) {}",
        "struct S { volatile int x; };",
        "void f(void) { volatile int x; }",
        "*(volatile int *)0x80100000 = 1;",
        "*(volatile int *)0x03FFFFFF = 1;",
        "*(volatile int *)0x06000000 = 1;",
        "*(volatile int *)address = 1;",
    ],
)
def test_volatile_nonhardware_forms_denied(snapshot, text):
    findings = policy.evaluate(snapshot, PATH, text, None, False)
    assert [(f.key, f.line) for f in findings] == [("source.volatile-storage", 1)]


@pytest.mark.parametrize("with_view", [False, True])
def test_sdk_group_exempt(snapshot, monkeypatch, with_view):
    text = "(volatile int *)0x80100000;\nvolatile s32 x;"
    source = _source(text) if with_view else None
    monkeypatch.setattr(policy._view, "active_lines", Mock(side_effect=AssertionError("SDK needs no volatile scan")))
    assert policy.evaluate(snapshot, PATH, text, source, True) == ()


@pytest.mark.parametrize(
    ("waiver", "waived"),
    [
        ("/* FAKEMATCH: timing */", True),
        ("/* FAKEMATCH: */", False),
        ("/* FAKEMATCH:   */", False),
        ("// FAKEMATCH: timing", False),
    ],
)
@pytest.mark.parametrize("previous_line", [False, True])
def test_fakematch_waives_only_waivable(snapshot, waiver, waived, previous_line):
    code = 'while (ready); asm("nop");'
    text = waiver + "\n" + code if previous_line else code + " " + waiver
    findings = policy.evaluate(snapshot, PATH, text, None, False)
    expected = {"source.inline-asm"} if waived else {"source.inline-asm", "source.empty-loop"}
    assert {f.key for f in findings} == expected
    assert all(f.line == (2 if previous_line else 1) for f in findings)


@pytest.mark.parametrize(
    ("code", "lines"),
    [
        ("do {\n  x++;\n} while (x < 4);", []),
        ("do { if (a) { b(); } } while (c);", []),
        ("if (a) {\n  b();\n}\nwhile (ready);", [4]),
        ("do { b(); } while (c);\nwhile (ready);", [2]),
        ("do {\n x++;\n}\nwhile (x < 3);", []),
        ("while (x < 3);", [1]),
        ("for (;;);", [1]),
        ("while (f(x)) y = g(z);", []),
        ("for (i = f(a); i < g(b); i++) total += i;", []),
        ("while (x)\n  y++;", []),
        ("for (i = 0; i < n; i++) a[i] = 0;", []),
        ("while (f(x));", [1]),
        ("for (i = f(a); i < g(b); i++);", [1]),
        ("while (x) {}", []),
    ],
)
def test_empty_loop_ignores_do_while_tails(snapshot, code, lines):
    findings = policy.evaluate(snapshot, PATH, code, None, False)
    assert [f.line for f in findings if f.key == "source.empty-loop"] == lines


def test_distant_fakematch_does_not_waive(snapshot):
    findings = policy.evaluate(snapshot, PATH, "/* FAKEMATCH: timing */\n\nwhile (ready);", None, False)
    assert [(f.key, f.line) for f in findings] == [("source.empty-loop", 3)]


@pytest.mark.parametrize("writes", [{PATH, "include/shared.h"}, [PATH, "include/shared.h"]])
def test_scope_new_vs_preexisting(writes):
    old = Finding("source.volatile-storage", "old qualifier", path="include/shared.h", line=1)
    moved = replace(old, line=9)
    new = Finding(old.key, "new qualifier", path=PATH, line=4, blocking=False)
    unwritten = Finding(old.key, "other qualifier", path="src/other.c", line=2)
    blocking, debt = policy.scope((old,), (moved, new, unwritten), writes)
    assert blocking == (replace(new, blocking=True),)
    assert debt == (replace(moved, blocking=False), replace(unwritten, blocking=False))
    assert new.blocking is False and moved.blocking is True


def test_scope_key_includes_rule_and_reason_but_not_path():
    old = Finding("source.empty-loop", "same line", path=PATH)
    other_rule = replace(old, key="source.inline-asm")
    other_path = replace(old, path="include/shared.h")
    blocking, debt = policy.scope((old,), (other_rule, other_path), {PATH, other_path.path})
    assert blocking == (other_rule,) and debt == (replace(other_path, blocking=False),)
    assert policy.scope((), (), ()) == ((), ())


@pytest.mark.parametrize(
    "text",
    [
        "/* volatile int x; */",
        "// volatile int x;",
        'char *s = "volatile int x;";',
        "int c = 'volatile';",
        'char *s = "escaped \\" volatile";',
        "/* multiline\nvolatile\n*/",
        'char *s = "/* volatile */ // asm( )";',
    ],
)
def test_comment_and_string_blanking(snapshot, text):
    assert policy.evaluate(snapshot, PATH, text, None, False) == ()


def test_scope_blanking_preserves_lines(snapshot):
    text = '/* volatile\n */\nchar *s = "volatile";\nvolatile int x;\n#define SAFE "volatile"\n// Generated by tool\n'
    findings = policy.evaluate(snapshot, PATH, text, None, False)
    assert {(f.key, f.line) for f in findings} == {
        ("source.volatile-storage", 4),
        ("source.local-define", 5),
        ("source.tool-comment", 6),
    }


def test_regex_deduplicates_match_line_and_truncates_reason(snapshot):
    text = '  asm("one"); asm("two"); ' + "x" * 160
    findings = policy.evaluate(snapshot, PATH, text, None, False)
    assert findings == (Finding("source.inline-asm", "inline assembly is never allowed: asm(", path=PATH, line=1),)


@pytest.mark.parametrize(
    ("text", "denied"),
    [
        ("dl->words.w0 = 0xE7000000;", True),
        ("w0 = (0xE7000000);", True),
        ("Gfx list[] = { {0xE7000000, 0x00000000} };", True),
        ("gDPWord(dl, 0xE7000000, 0x00000000);", True),
        ("unsigned int value = 0xE7000000;", False),
        ("w0 = 0x12000000;", False),
        ("w0 = 0xE700;", False),
        ('char *s = "w0 = 0xE7000000";', False),
    ],
)
def test_raw_gfx_context(snapshot, text, denied):
    findings = policy.evaluate(snapshot, PATH, text, None, False)
    assert [(f.key, f.line) for f in findings] == ([("source.raw-gfx", 1)] if denied else [])


@pytest.mark.parametrize(
    ("text", "denied"),
    [
        ("((func_80239184_S1 *)(arg0))->unk548 = arg1;", True),
        ("x = ( func_80203848_de_S2 * ) p;", True),
        ("func_80239184(arg0, arg1);", False),
        ("void *p = (void *)func_80239184;", False),
    ],
)
def test_struct_view_cast(snapshot, text, denied):
    findings = [f for f in policy.evaluate(snapshot, PATH, text, None, False) if f.key != "source.fake-type"]
    assert [(f.key, f.line) for f in findings] == ([("source.struct-view-cast", 1)] if denied else [])


def test_shared_declarations_active_headers_only(snapshot, monkeypatch):
    text = "extern int shared;\nint call(int x);\ntypedef int Number;\nextern int private;"
    source = _source(dependencies=((PATH, "s"), ("include/shared.h", "h"), ("src/other.c", "o")))
    active = {
        PATH: dict(enumerate(text.splitlines(), 1)),
        "include/shared.h": {1: "extern int shared;", 3: "int call(int x);", 7: "typedef int Number;"},
        "src/other.c": {1: "extern int private;"},
    }
    monkeypatch.setattr(policy._view, "active_lines", Mock(side_effect=lambda view, path: active[path]))
    monkeypatch.setattr(policy.store, "cached_object", lambda cfg, kind, key, produce: produce())
    assert [(f.key, f.line) for f in policy.evaluate(snapshot, PATH, text, source, False)] == [
        ("source.shared-declarations", 1),
        ("source.shared-declarations", 2),
        ("source.shared-declarations", 3),
    ]
    assert policy.evaluate(snapshot, PATH, text, None, False) == ()


def test_directive_rules_never_let_whitespace_cross_a_line():
    import re
    import tomllib

    rules = tomllib.loads((Path(policy.__file__).parent / "resources/data/rules.toml").read_text())["rule"]
    directive = [r["regex"] for r in rules if r.get("regex", "").startswith("^")]
    assert directive and not any(r"^\s" in regex for regex in directive)  # a blank-line run is not rescanned per line
    assert re.search(directive[0], '\n\n\n  #  include "../x.h"\n', re.M)


def test_local_define_skips_headers(snapshot):
    text = "#define gDPSetColor(pkt, c) 1\n"
    assert policy.evaluate(snapshot, "include/gbi.h", text, None, False, only={"local-define"}) == ()
    found = policy.evaluate(snapshot, PATH, text, None, False, only={"local-define"})
    assert [f.line for f in found] == [1]


def test_census_empty_changes_does_no_work(snapshot, monkeypatch):
    monkeypatch.setattr(policy.config, "load_resource", Mock(side_effect=AssertionError("no staged sources")))
    assert policy.census(snapshot, changes={}) == ((), [])


@pytest.mark.parametrize(
    "path,exempt",
    [
        ("src/vendor/routine.c", True),
        ("src/vendor/nested/routine.c", True),
        ("src/vendor_other/routine.c", False),
        ("src/routine.c", False),
    ],
)
def test_library_policy_exemption_is_limited_to_the_configured_folder(snapshot, path, exempt):
    snapshot.config.project.libraries = ("src/vendor",)
    findings = policy.evaluate(snapshot, path, "union U { int i; float f; };\n#define VIEW(x) (x)\n", None, False)
    assert (not findings) is exempt


def test_every_dynamic_source_rule_has_a_renderable_finding_key():
    from unbake import config

    for rule in config.load_resource("rules.toml")["rule"]:
        assert config.sentence("source." + rule["id"])


def test_scope_multiset_allows_move_between_files():
    row = Finding("source.fake-type", "invented: func_80123456_S1", path="src/a.c")
    moved = replace(row, path="src/b.c")
    assert policy.scope((row,), (moved,), {"src/a.c", "src/b.c"})[0] == ()


def test_scope_refuses_added_row_in_new_file():
    row = Finding("source.fake-type", "invented: func_80123456_S1", path="src/a.c")
    added = replace(row, path="src/new.c")
    blocking, _ = policy.scope((row,), (row, added), {"src/new.c"})
    assert blocking == (added,)


def test_scope_subject_survives_line_edit(snapshot):
    text = "struct func_802070D0_S1 { int x; };\n"
    old = policy.evaluate(snapshot, PATH, text, None, False)
    new = policy.evaluate(snapshot, PATH, "// edited\n\n" + text, None, False)
    assert old[0].line != new[0].line and old[0].reason == new[0].reason
    assert policy.scope(old, new, {PATH})[0] == ()


def test_raw_offset_unwaivable(snapshot):
    text = "/* FAKEMATCH: needed */\nint f(char *p) { return *(int *)((u8 *)p + 0x10); }\n"
    assert [f.key for f in policy.evaluate(snapshot, PATH, text, None, False)] == ["source.raw-offset"]


def test_raw_offset_original_parenthesized_loads_and_stores(snapshot):
    text = (
        "/* FAKEMATCH: needed */\ntemp_a1 = (*(void **)((s8 *)(temp_v0) + (4)));\n(*(void **)((s"
        "8 *)(temp_v0) + (4))) = (void *) (temp_a1 + 8);\n(*(s32 *)((s8 *)(temp_a1) + (0))) = 0"
        "xFA000000;\n(*(s32 *)((s8 *)(temp_a1) + (4))) = (s32) (((*(u8 *)((s8 *)(arg0) + (0))) "
        "<< 0x18) | ((*(u8 *)((s8 *)(arg0) + (1))) << 0x10) | ((*(u8 *)((s8 *)(arg0) + (2))) <"
        "< 8) | (*(u8 *)((s8 *)(arg0) + (3))));\n"
    )
    findings = policy.evaluate(snapshot, PATH, text, None, False, only={"raw-offset"})
    assert [(f.key, f.line) for f in findings] == [("source.raw-offset", n) for n in (2, 3, 4, 5)]


@pytest.mark.parametrize(
    ("byte_type", "operand", "offset"),
    [
        ("u8", "p", "0"),
        ("u8", "( p )", "(4)"),
        ("s8", "p", "0x10"),
        ("s8", "( p )", "(4)"),
        ("char", "p", "((0X10))"),
        ("char", "( p )", "( \t (\n 4 ) )"),
        ("unsigned char", "p", "(4)"),
        ("unsigned char", "( p )", "((0X10))"),
    ],
)
def test_raw_offset_numeric_grouping_and_whitespace(snapshot, byte_type, operand, offset):
    text = f"*(int *)(( {byte_type} * ) {operand} + {offset}) = value;"
    findings = policy.evaluate(snapshot, PATH, text, None, False, only={"raw-offset"})
    assert [(f.key, f.line) for f in findings] == [("source.raw-offset", 1)]


@pytest.mark.parametrize(
    "text",
    [
        "value = words[4];",
        "value = bytes[(4)];",
        "value = object->field;",
        "word = (s32 *)words + (4);",
        "record = (Record *)records + ((4));",
        "byte = (u8 *)(words + (4));",
        "byte = (s8 *)(words + 4);",
        "byte = (char *)(words + 0x10);",
        "byte = (s8 *)words;",
        "/* value = *((s8 *)p + (4)); */",
        '// value = *((s8 *)p + (4));\nconst char *text = "(s8 *)p + (4)";',
    ],
)
def test_raw_offset_keeps_typed_access_and_noncode(snapshot, text):
    assert policy.evaluate(snapshot, PATH, text, None, False, only={"raw-offset"}) == ()


def test_text_findings_needs_no_snapshot():
    found = policy.text_findings(PATH, "int f(void) { return (u8 *)p + 4; }\nunion U { int a; };\n", False)
    assert sorted(f.key for f in found) == ["source.raw-offset", "source.union-view"]
    assert policy.text_findings(PATH, "volatile int x;\n", False) == ()


def test_every_type_debt_rule_keeps_a_unit_out_of_the_exact_credit():
    unclean = set(policy.config.load_resource("rules.toml")["unclean"])
    assert {
        "struct-view-cast",
        "union-view",
        "fake-type",
        "param-cast",
        "address-int",
        "raw-offset",
        "pad-view",
    } <= unclean
    assert "address-literal" not in unclean


@pytest.mark.parametrize(
    ("text", "denied"),
    [
        ("gDPSetFillColor(D_80110634++, ((0xFFFCFFFC)));", False),
        ("gsSPSetGeometryMode(0xFFFFFFFF);", False),
        ("gDPWord(dl, 0xE7000000, 0x00000000);", True),
    ],
)
def test_raw_gfx_skips_gbi_macro_arguments(snapshot, text, denied):
    assert bool(policy._raw_gfx(text)) is denied


# ------------------------------------------------------------ N-way version differences

THREE_WAY = """int f(list) {
#if defined(VERSION_DE)
    x = 0x3C6;
#elif defined(VERSION_EU) || defined(VERSION_US) || defined(VERSION_US_REV1)
    x = 0x3CC;
#elif defined(VERSION_EU_X)
    x = 0x3D0;
#endif
}
"""
HOLDERS = ("de", "eu", "eu-x", "us", "us-rev1")
SIDES = [["de"], ["eu", "us", "us-rev1"], ["eu-x"]]


def _bytes_by_branch(v):
    return {"de": b"a", "eu-x": b"c"}.get(v, b"b")


def test_header_declarations_are_read_once_per_distinct_active_text(snapshot, monkeypatch):
    reads, kept = [], {}
    monkeypatch.setattr(policy, "_declarations", lambda code: reads.append(code) or {})
    monkeypatch.setattr(
        policy.store,
        "cached_object",
        lambda cfg, kind, key, produce: kept[key] if key in kept else kept.setdefault(key, produce()),
    )
    for _ in range(3):
        policy._header_names(snapshot, "extern int a;")
    policy._header_names(snapshot, "extern int b;")
    assert len(reads) == 2  # two distinct texts, however many times a score asks


def test_active_lines_group_the_view_once_for_every_header():
    from unbake import view as view_module

    class Lines(tuple):
        scans = 0

        def __iter__(self):
            Lines.scans += 1
            return super().__iter__()

    source = SimpleNamespace(text="a\nb\nc", lines=Lines(((1, "x.h", 1), (2, "y.h", 4), (3, "x.h", 2))))
    assert view_module.active_lines(source, "x.h") == {1: "a", 2: "c"}
    assert view_module.active_lines(source, "y.h") == {4: "b"} and Lines.scans == 1


def test_gbi_template_encodes_the_v1_texture_command_with_an_eight_bit_on_field():
    words = _gbi_words("gSPTexture", 1, [0xFFFF, 0x8000, 0, 1, 1], {"G_TEXTURE": 0xBB})
    assert words == (0xBB000000 | (1 << 8) | 1, 0xFFFF8000)


HEADER = "#if defined(VERSION_A)\n#define ACTIVE(p) ((p)->flag)\n#else\n#define ACTIVE(p) 1\n#endif\n"


def test_the_sdk_unions_are_not_union_views_but_a_game_union_is():
    text = (
        "typedef union Gfx {\n    struct { u32 w0; u32 w1; } words;\n    u64 force_structure_alignment;\n} Gfx;\n"
        "typedef union {\n    long long int force_union_align;\n} Acmd;\n"
        "union Gfx *list;\n"
        "typedef union { int i; float f; } Pun;\n"
    )
    found = [
        f.line for f in policy.text_findings("include/shared/bt_gfx.h", text, False) if f.key == "source.union-view"
    ]
    assert found == [9]


def test_a_finding_in_a_placed_helper_withholds_the_helper_not_its_member():
    text = "int helper(void *p) { return *(int *)((u8 *)p + 4); }\nint head(void) { return helper(0); }\n"
    found = [Finding("source.raw-offset", "raw: x", line=1)]
    assert policy.withheld(text, found, ["head", "helper"]) == {"helper": ("source.raw-offset",)}
    assert policy.withheld(text, found, ["head"]) == {"head": ("source.raw-offset",)}  # inlined: its caller


@pytest.mark.parametrize(
    "text",
    [
        "((void **) arg0)[16 / sizeof(void *)] = 0;",
        "x = ((s32 *) temp_a3)[176 / sizeof(s32)];",
        "c = ((u16 **) arg4)[0x14 / sizeof(u16 *)];",
        "y = ((unsigned char *)material)[6];",
        "t = *((void **) ((u8 *) ((arg0 + (i * 4))) + 3932));",
        "u32 *words = (u32 *)&scratch[0x18];",
    ],
)
def test_raw_offset_indexes_a_primitive_view_by_a_constant(snapshot, text):
    found = policy.evaluate(snapshot, PATH, text, None, False, only={"raw-offset"})
    assert [f.key for f in found] == ["source.raw-offset"]


@pytest.mark.parametrize(
    "text",
    [
        "b = ((u8 *) buf)[i];",
        "b = ((u8 *) buf)[0];",
        "r = ((Record *) p)[2];",
        "f = (float *)&buf[0];",
        "e = &((s32 *)p)[i];",
    ],
)
def test_raw_offset_leaves_variable_first_and_record_indexes(snapshot, text):
    assert policy.evaluate(snapshot, PATH, text, None, False, only={"raw-offset"}) == ()


def _gbi_words(macro: str, branch: int, args: list[int], constants: dict[str, int]) -> tuple[int, int]:
    """The two words a gbi.h template macro (its `branch`-th definition in the file) encodes, by the bit layout."""
    import re as _re
    from importlib import resources

    text = (resources.files("unbake") / "resources/templates/sdk/gbi.h").read_text()
    lines = [m for m in _re.finditer(rf"^#define {macro}\(([^)]*)\)\s*(.*?)$", text, _re.M)]
    params, body = lines[branch][1].replace(" ", "").split(","), lines[branch][2]
    if body.endswith("\\"):
        body = text[lines[branch].end() :].split("\n")[1].strip()
    inner = _re.search(r"_GBI_CMD\(pkt,\s*(.*)\)\s*$", body)
    assert inner, body
    expr = inner.group(1)

    def shiftl(value: int, shift: int, width: int) -> int:
        return (value & (0xFFFFFFFF >> (32 - width))) << shift

    scope = {"_SHIFTL": shiftl, "unsigned": int, **constants, **dict(zip(params[1:], args, strict=False))}
    code = expr.replace("(unsigned int)(v)", "0").replace("(unsigned int)", "int")
    depth, cut = 0, None
    for at, ch in enumerate(code):  # the comma between the two words is the one outside every parenthesis
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            cut = at
    assert cut is not None
    return eval(code[:cut], scope) & 0xFFFFFFFF, eval(code[cut + 1 :], scope) & 0xFFFFFFFF


@pytest.mark.parametrize(
    ("branch", "expected"),
    [(1, 0x04320040), (2, 0x0404103F)],  # F3D, then F3DEX v1: n=4 vertices loaded at v0=2
)
def test_gbi_template_encodes_the_v1_vertex_load_by_the_ucode_bit_layout(branch, expected):
    words = _gbi_words("gSPVertex", branch, [0, 4, 2], {"G_VTX": 4})
    assert words[0] == expected


def _placed_snapshot(snapshot, monkeypatch):
    """One version "us" with a data symbol D_800D7000 (0x20 bytes) in a segment, owned by the module at PATH."""
    version = SimpleNamespace(
        symbols={"D_800D7000": 0x800D7000, "func_80000400": 0x80000400},
        code=frozenset({"func_80000400"}),
        segments=(("main", 0x1000, 0x2000, 0x80000400), ("data", 0x2000, 0x2020, 0x800D7000)),
    )
    snapshot.versions = {"us": version}
    module = tree.Module(PATH, "main", {"us": ("func_80000400", "D_800D7000")}, (), {})
    monkeypatch.setattr(policy, "_tree", lambda snap: tree.Tree("t", (module,)))
    return module


def test_address_literal_hits_a_symbol_range(snapshot, monkeypatch):
    _placed_snapshot(snapshot, monkeypatch)
    found = policy._address_literals(snapshot, PATH, "int a = 0x800D7010;\nint b = 0x800D8000;\nint c = 7;\n")
    assert found == {1: "0x800D7010"}


def test_cross_global_offset_hits(snapshot, monkeypatch):
    _placed_snapshot(snapshot, monkeypatch)
    code = "char *p = (char *)&D_800D7000 + 0x20;\nchar *q = (char *)&D_800D7000 + 0x1C;\n"
    assert list(policy._address_literals(snapshot, PATH, code)) == [1]


def test_pointer_arithmetic_on_a_named_symbol_is_not_an_address_literal(snapshot, monkeypatch):
    _placed_snapshot(snapshot, monkeypatch)
    code = "f(D_800D7000 + 0x1000 / sizeof(u64));\nf(D_800D7000 + 0x100 * 2);\nf(D_800D7000 + 0x100 << 2);\n"
    assert policy._address_literals(snapshot, PATH, code) == {}
    assert list(policy._address_literals(snapshot, PATH, "f(D_800D7000 + 0x1000);\n")) == [1]


def test_a_sum_with_the_value_of_a_scalar_global_is_not_an_address_literal(snapshot, monkeypatch):
    # BT-OPUS-1: `D_801B4AA8 + 0x5A` (an s32 tick) and `D_80145DA0 + 12000000` (a u64) add to values, not addresses
    _placed_snapshot(snapshot, monkeypatch)
    code = "x = D_800D7000 + 0x5A;\n"
    for declared in ("extern s32 D_800D7000;", "extern u64 D_800D7000;", "extern char *D_800D7000;"):
        view = SimpleNamespace(text=f"{declared}\nvoid f(void) {{ {code} }}\n")
        assert policy._address_literals(snapshot, PATH, code, view) == {}
    for declared in ("extern u8 D_800D7000[];", ""):  # an array decays to its address; no declaration is no proof
        view = SimpleNamespace(text=f"{declared}\nvoid f(void) {{ {code} }}\n")
        assert list(policy._address_literals(snapshot, PATH, code, view)) == [1]
    view = SimpleNamespace(text="extern s32 D_800D7000;\n")
    assert list(policy._address_literals(snapshot, PATH, "p = (char *)&D_800D7000 + 0x20;\n", view)) == [1]


def test_without_a_source_view_the_included_headers_declare_the_scalar(snapshot, monkeypatch):
    _placed_snapshot(snapshot, monkeypatch)
    headers = {"include/game.h": b'#include "ticks.h"\n', "include/ticks.h": b"extern s32 D_800D7000;\n"}
    monkeypatch.setattr(snapshot, "peek", headers.get, raising=False)
    code = '#include "game.h"\nvoid f(void) { x = D_800D7000 + 0x5A; }\n'
    assert policy._address_literals(snapshot, PATH, code) == {}


def test_address_literal_skips_float_fragments_and_reads_octal(snapshot, monkeypatch):
    _placed_snapshot(snapshot, monkeypatch)
    code = "float a = 0.000212f;\nfloat b = 1e-05;\nfloat c = 1.5e+3;\nint d = 020000000000;\nint e = 0x800D7010U;\n"
    assert policy._address_literals(snapshot, PATH, code) == {5: "0x800D7010"}
    assert [policy._integer(t) for t in ("010", "09", "0", "0x10", "7")] == [8, None, 0, 16, 7]


def test_a_unit_outside_the_tree_has_no_address_literal(snapshot):
    assert policy._address_literals(snapshot, PATH, "int a = 0x800D7010;\n") == {}


def test_credited_names_the_defined_functions_that_have_an_address(snapshot, monkeypatch):
    _placed_snapshot(snapshot, monkeypatch)
    snapshot.config.project.names_from = "us"
    text = "void func_80000400(void) { helper(); }\nstatic void helper(void) { }\n"
    snapshot.read = lambda path: text.encode()
    assert policy.credited(snapshot, PATH) == ["func_80000400"]  # helper has no address: inlined, never credited


def test_pieces_are_one_function_to_the_next_symbol(snapshot, monkeypatch):
    _placed_snapshot(snapshot, monkeypatch)
    monkeypatch.setattr(policy.effort, "memo", lambda key, build: build())
    assert policy.pieces(snapshot, "D_800D7000", "us") == [("D_800D7000", 0x20)]
    assert policy.withheld_bytes(snapshot, {"D_800D7000"}, "D_800D7000", "us") == 0x20
    assert policy.withheld_bytes(snapshot, {"other"}, "D_800D7000", "us") == 0
    assert policy.pieces(snapshot, "absent", "us") == []


def _headers(snapshot, monkeypatch, files):
    monkeypatch.setattr(policy, "_include_headers", lambda snap: sorted(files))
    monkeypatch.setattr(policy.store, "cached_object", lambda cfg, kind, key, produce: produce())
    snapshot.read = lambda path: files[path].encode()
    monkeypatch.setattr(policy.config, "sentence", lambda key: key)


def test_header_lint_flags_an_extern_declared_twice(snapshot, monkeypatch):
    _headers(snapshot, monkeypatch, {"include/a.h": "extern int gA;\n", "include/b.h": "extern int gA;\n"})
    found = policy.header_lint(snapshot)
    assert [(f.key, f.path, f.unit) for f in found] == [("source.extern-redeclared", "include/b.h", "gA")]


def test_header_lint_flags_one_name_declared_differently(snapshot, monkeypatch):
    _headers(snapshot, monkeypatch, {"include/a.h": "extern int gA;\n", "include/b.h": "extern u8 gA[4];\n"})
    assert [f.key for f in policy.header_lint(snapshot)] == ["source.extern-conflict"]


def test_header_lint_accepts_one_declaration_each_and_reads_prototypes_and_pointers(snapshot, monkeypatch):
    files = {"include/a.h": "extern void f(int);\nextern int (*cb)(void);\n", "include/b.h": "extern int gB[2];\n"}
    _headers(snapshot, monkeypatch, files)
    assert policy.header_lint(snapshot) == ()
    assert set(policy._externs(files["include/a.h"])) == {"f", "cb"}


def test_header_lint_scopes_to_the_given_paths(snapshot, monkeypatch):
    _headers(snapshot, monkeypatch, {"include/a.h": "extern int gA;\n", "include/b.h": "extern int gA;\n"})
    assert policy.header_lint(snapshot, ["include/a.h"]) == ()


def test_header_externs_are_cached_by_the_header_text_in_header_names(snapshot, monkeypatch):
    seen = []
    monkeypatch.setattr(
        policy.store, "cached_object", lambda cfg, kind, key, produce: seen.append((kind, key)) or produce()
    )
    policy._header_externs(snapshot, "extern int gA;\n")
    policy._header_externs(snapshot, "extern int gA;\n")
    assert seen[0] == seen[1] and seen[0][0] == "header.names"


def _keyed(monkeypatch, reach):
    module = tree.Module("src/a.c", "main", {"a": ("f",)}, (), {})
    monkeypatch.setattr(policy._view, "closure", lambda snap, unit, version: reach)
    monkeypatch.setattr(policy, "_unclean_context", lambda snap: "context")
    monkeypatch.setattr(policy, "_tree", lambda snap: tree.Tree("t", (module,)))
    monkeypatch.setattr(policy, "_unit", lambda snap, mod: mod)
    config_ = SimpleNamespace(project=SimpleNamespace(names_from="a", libraries=()))
    return SimpleNamespace(
        read=lambda path: b"text",
        peek=lambda path: b"x",
        config=config_,
        overlays={},
        versions={"a": SimpleNamespace(symbols={})},
    )


def test_the_unclean_key_moves_with_the_job_that_shapes_its_value(monkeypatch):
    seen = []
    monkeypatch.setattr(policy, "code_identity", lambda *fs: seen.append(fs) or "id")
    policy._unclean_key((_keyed(monkeypatch, ()), "src/a.c", ("r",), ()))
    assert policy._unclean_unit in seen[0]  # a cached verdict of another result shape is never read back


def test_a_header_edit_moves_the_unclean_verdict_key(monkeypatch):
    def key(reach):
        return policy._unclean_key((_keyed(monkeypatch, reach), "src/a.c", ("r",), ()))

    base = key((("include/a.h", "pin1"),))
    assert key((("include/a.h", "pin1"),)) == base
    assert key((("include/a.h", "pin2"),)) != base  # the unit's text is the same, the header it reads is not


# ---------------------------------------------------------------- the guardrails that survived the layout database
def _findings(snapshot, text, path=PATH):
    return policy.evaluate(snapshot, path, text, None, False)


def test_no_inline_asm_in_a_function_body_and_a_stand_in_only_as_a_whole_item(snapshot):
    clean = '#include "include_asm.h"\n\nINCLUDE_ASM("main", f);\ns32 g(void) {\n    return 1;\n}\n'
    assert _findings(snapshot, clean) == ()
    body = 's32 g(void) {\n    __asm__("nop");\n    return 1;\n}\n'
    assert [f.key for f in _findings(snapshot, body)] == ["source.inline-asm"]
    inside = 's32 g(void) {\n    INCLUDE_ASM("main", f);\n    return 1;\n}\n'
    (found,) = _findings(snapshot, inside)
    assert (found.key, found.line) == ("source.stand-in-body", 2) and "whole item" in found.reason
    assert policy.evaluate(snapshot, PATH, 's32 g(void) {\n    INCLUDE_ASM("main", f);\n}\n', None, False)[0].blocking


def test_a_stand_in_after_a_closed_function_is_a_whole_item(snapshot):
    text = 's32 g(void) {\n    return 1;\n}\nINCLUDE_RODATA("main", jtbl);\nINCLUDE_DATA("main", blob);\n'
    assert _findings(snapshot, text) == ()


RECORD = "typedef struct %s {\n    s32 a;\n    f32 b;\n    u8 c;\n} %s;\n"


def _records(monkeypatch, snapshot, files):
    sources = {path: text.encode() for path, text in files.items()}
    snapshot.peek = lambda path: sources.get(path)
    monkeypatch.setattr(policy, "_include_headers", lambda snap: sorted(p for p in sources if p.startswith("include/")))
    monkeypatch.setattr(
        policy,
        "_tree",
        lambda snap: tree.Tree("t", tuple(tree.Module(p, "main", {}, (), {}) for p in sources if p.startswith("src/"))),
    )


def test_a_record_that_repeats_another_records_body_is_a_duplicate(monkeypatch, snapshot):
    files = {"include/types.h": RECORD % ("Actor", "Actor"), PATH: RECORD % ("Thing", "Thing")}
    _records(monkeypatch, snapshot, files)
    (found,) = _findings(snapshot, files[PATH])
    assert found.key == "source.duplicate-record" and "Thing repeats Actor (include/types.h)" in found.reason
    assert _findings(snapshot, files["include/types.h"], "include/types.h") == ()  # the canonical one is clean


def test_two_records_of_one_file_with_one_body_are_a_duplicate_too(monkeypatch, snapshot):
    text = RECORD % ("A", "A") + RECORD % ("B", "B")
    _records(monkeypatch, snapshot, {PATH: text})
    assert [f.reason.endswith("B repeats A (src/unit.c)") for f in _findings(snapshot, text)] == [True]


def test_a_record_with_its_own_body_or_too_few_fields_is_not_a_duplicate(monkeypatch, snapshot):
    small = "typedef struct P {\n    s32 a;\n} P;\ntypedef struct Q {\n    s32 a;\n} Q;\n"
    other = RECORD % ("Actor", "Actor") + RECORD.replace("f32 b", "s16 b") % ("Prop", "Prop")
    _records(monkeypatch, snapshot, {PATH: small + other})
    assert _findings(snapshot, small + other) == ()


def test_the_duplicate_record_rule_keeps_a_unit_out_of_the_exact_credit() -> None:
    rules = policy.config.load_resource("rules.toml")
    assert "duplicate-record" in rules["unclean"]


def test_a_snapshot_is_captured_once_however_many_rules_ask(monkeypatch):
    calls = []
    held = tree.Tree("held", ())
    monkeypatch.setattr(
        policy.tree, "capture", lambda config, overlays: calls.append(dict(overlays)) or tree.Tree("o", ())
    )
    base = SimpleNamespace(config="c", overlays={}, files=held, digest="d0")
    candidate = SimpleNamespace(config="c", overlays={"src/x.c": b"1"}, files=None, digest="d1")
    other = SimpleNamespace(config="c", overlays={"src/x.c": b"2"}, files=None, digest="d2")
    with policy.effort.command("test-policy", ()):
        assert policy._tree(base) is held
        first = policy._tree(candidate)
        assert policy._tree(candidate) is first and policy._tree(candidate) is first
        assert policy._tree(other) is not first
    assert calls == [{"src/x.c": b"1"}, {"src/x.c": b"2"}]


def _cart(version, matched):
    return {"version": version, "functions": {"matched": matched, "fuzzy": 0, "total": 9}}


def _audit_repo(tmp_path, monkeypatch):
    import json
    import subprocess

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.c").write_text("int f(void) { return 1; }\nint g(void) { return 2; }\n")
    (tmp_path / "progress.json").write_text(json.dumps({"carts": [_cart("us", 2)]}))
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "base"]):
        subprocess.run([*git, *args], check=True)
    return json


def test_audit_refuses_a_matched_function_drop_that_removes_no_c(tmp_path, monkeypatch):
    json = _audit_repo(tmp_path, monkeypatch)
    (tmp_path / "src/a.c").write_text("/* touched */\nint f(void) { return 1; }\nint g(void) { return 2; }\n")
    (tmp_path / "progress.json").write_text(json.dumps({"carts": [_cart("us", 0)]}))
    with pytest.raises(policy.Refusal) as refused:
        policy.audit_command({"project": str(tmp_path), "against": "HEAD"})
    (finding,) = refused.value.findings
    assert (finding.key, finding.versions) == ("audit.credit_drop", ("us",))


def test_audit_allows_a_drop_the_removed_c_explains(tmp_path, monkeypatch):
    json = _audit_repo(tmp_path, monkeypatch)
    (tmp_path / "src/a.c").write_text("int f(void) { return 1; }\n")
    (tmp_path / "progress.json").write_text(json.dumps({"carts": [_cart("us", 1)]}))
    result = policy.audit_command({"project": str(tmp_path), "against": "HEAD"})
    assert result["paths"] == ["src/a.c"]


_RENAMED = "void *{name}(void *arg0, void *arg1) {{\n    return ((Actor *)arg0)->parent;\n}}\n"


def _audit_rename(tmp_path, monkeypatch, new_name, body):
    """A committed tree placing func_8027D97C_de in two versions, then the working tree naming it new_name."""
    import subprocess

    json = _audit_repo(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text('[version.de]\nsha1 = "x"\n\n[version.us]\nsha1 = "y"\n')

    def place(name):
        for version, vram in (("de", 0x8027D97C), ("us", 0x8027D9AC)):
            (tmp_path / "versions" / version).mkdir(parents=True, exist_ok=True)
            text = f"{name} = 0x{vram:X}; // type:func\nkeep = 0x80000000;\n"
            (tmp_path / "versions" / version / "symbol_addrs.txt").write_text(text)

    place("func_8027D97C_de")
    (tmp_path / "src/p.c").write_text(_RENAMED.format(name="func_8027D97C_de"))
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    for args in (["add", "-A"], ["commit", "-q", "-m", "placed"]):
        subprocess.run([*git, *args], check=True)
    place(new_name)
    (tmp_path / "src/p.c").write_text(body)
    assert json
    return policy.audit_command({"project": str(tmp_path), "against": "HEAD"})


def test_audit_carries_an_old_row_through_a_function_rename(tmp_path, monkeypatch):
    new = "Particle_ParentTransform"
    result = _audit_rename(tmp_path, monkeypatch, new, _RENAMED.format(name=new))
    assert result["paths"] == ["src/p.c"]


def test_audit_refuses_a_new_row_inside_a_renamed_function(tmp_path, monkeypatch):
    new = "Particle_ParentTransform"
    body = _RENAMED.format(name=new).replace("    return", "    gAddr = (s32)&((Actor *)arg1)->x;\n    return")
    with pytest.raises(policy.Refusal) as refused:
        _audit_rename(tmp_path, monkeypatch, new, body)
    assert [f.key for f in refused.value.findings] == ["source.address-int"]
    assert new in refused.value.findings[0].reason


def test_audit_unchanged_names_keep_their_rows(tmp_path, monkeypatch):
    name = "func_8027D97C_de"
    assert _audit_rename(tmp_path, monkeypatch, name, "/* touched */\n" + _RENAMED.format(name=name))["paths"] == [
        "src/p.c"
    ]


def test_audit_a_new_function_with_an_old_row_shape_is_new(tmp_path, monkeypatch):
    name = "func_8027D97C_de"
    body = _RENAMED.format(name=name) + _RENAMED.format(name="func_other")
    with pytest.raises(policy.Refusal) as refused:
        _audit_rename(tmp_path, monkeypatch, name, body)
    assert [f.key for f in refused.value.findings] == ["source.param-cast"]
    assert "func_other()" in refused.value.findings[0].reason


def test_renames_follow_placement_not_spelling():
    one, two = frozenset({("de", 1), ("us", 2)}), frozenset({("de", 5)})
    before = {"a": one, "b": frozenset({("de", 3)}), "c": two}
    after = {"A": one, "b": frozenset({("de", 3)}), "x": two, "y": two}
    # c's placement is held by two names: no single identity, so no rename is read
    assert policy._renames(before, after) == {"a": "A"}


def test_scope_blames_only_the_paths_whose_own_count_rose():
    # the wtmain9234 shape: a header gains two rows while an unchanged includer keeps its own
    row = Finding("source.union-view", "a union gives one storage two types: union", path="include/UNUSED.H")
    unused = [replace(row, line=n) for n in (1, 2)]
    owner_old = [replace(row, path="include/OWNER.H", line=n) for n in range(5)]
    owner_new = [replace(row, path="include/OWNER.H", line=n) for n in range(7)]
    writes = {"include/UNUSED.H", "include/OWNER.H"}
    blocking, debt = policy.scope(unused + owner_old, unused + owner_new, writes)
    assert {f.path for f in blocking} == {"include/OWNER.H"} and len(blocking) == 2
    assert {f.path for f in debt} == {"include/UNUSED.H", "include/OWNER.H"}
    assert policy.scope(unused + owner_old, unused + owner_old, writes)[0] == ()  # nothing rose
    assert policy.scope(unused + owner_old, unused + owner_old[1:], writes)[0] == ()  # a line shift or a drop


def test_scope_a_row_moved_between_paths_is_free_even_beside_a_rise_elsewhere():
    row = Finding("source.fake-type", "invented: func_1", path="src/a.c")
    other = Finding("source.fake-type", "invented: func_2", path="src/c.c")
    moved = replace(row, path="src/b.c")
    added = replace(other, path="src/d.c")
    blocking, _ = policy.scope((row, other), (moved, other, added), {"src/a.c", "src/b.c", "src/c.c", "src/d.c"})
    assert blocking == (added,)


def test_audit_names_the_path_that_added_the_rows_not_an_unchanged_includer(tmp_path, monkeypatch):
    import subprocess

    json = _audit_repo(tmp_path, monkeypatch)
    union = "union { int a; float b; } v%d;\n"
    (tmp_path / "include").mkdir()
    (tmp_path / "include/a.h").write_text(union % 0)
    (tmp_path / "include/z.h").write_text(union % 1)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    for args in (["add", "-A"], ["commit", "-q", "-m", "unions"]):
        subprocess.run([*git, *args], check=True)
    (tmp_path / "include/a.h").write_text(union % 0 + union % 2 + union % 3)
    (tmp_path / "include/z.h").write_text("/* touched */\n" + union % 1)
    with pytest.raises(policy.Refusal) as refused:
        policy.audit_command({"project": str(tmp_path), "against": "HEAD"})
    assert {f.path for f in refused.value.findings} == {"include/a.h"}
    assert all("union-view 1 -> 3" in f.reason for f in refused.value.findings)
    assert json  # the repo helper's module
