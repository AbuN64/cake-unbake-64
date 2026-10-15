"""Policy contract checks without native tools, workers, or project setup."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from unbake import policy
from unbake.contracts import (
    Config,
    Finding,
    Group,
    LayoutMap,
    Member,
    Placement,
    Project,
    Recipe,
    Snapshot,
    SourceView,
    UnitSpec,
)

PATH = "src/unit.c"


@pytest.fixture
def snapshot(monkeypatch):
    monkeypatch.setattr(policy.effort, "stage", Mock(side_effect=lambda name: nullcontext()))
    result = Mock(spec=Snapshot)
    result.config = Mock(spec=Config)
    result.config.project = Mock(spec=Project)
    result.config.project.names_from = "a"
    result.config.project.library_dirs = ()
    result.config.project.version_differences = ()
    result.config.project.digest = "project"
    result.layout = Mock(units={}, members={})
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


def test_directive_scope_and_continuations(snapshot):
    text = (
        '/* #include "../no.h" */\n#include /* allowed comment */ "../bad.h"\n'
        "#if VERSION_A\n#define X \\\nvolatile int x;\n#endif\n"
    )
    assert {(f.key, f.line) for f in policy.evaluate(snapshot, PATH, text, None, False)} == {
        ("source.local-include", 2),
        ("source.file-version-guard", 3),
        ("source.local-define", 4),
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


@pytest.mark.parametrize(("names_from", "expected_version"), [("a", "a"), ("missing", "a")])
def test_census_units_headers_deduplication_and_sdk(snapshot, monkeypatch, names_from, expected_version):
    snapshot.config.project.names_from = names_from
    units = [
        UnitSpec(path, kind, "group", ("member",), "test", {})
        for path, kind in [(PATH, "c"), ("src/second.c", "c"), ("src/asm.s", "asm"), ("src/data.c", "data")]
    ]
    member = Member(
        "member", "function", "c", "group", (Placement("b", ".text", 0, 4, 0), Placement("a", ".text", 0, 4, 0))
    )
    group = Group("group", "main", ("member",), "authored", (), True, "src/group.c", "include/main/group.h")
    snapshot.layout = LayoutMap({"group": group}, {"member": member}, {u.path: u for u in units}, "layout", (), {})
    snapshot.read.side_effect = lambda path: f"text of {path}".encode()
    recipe = Recipe("test", (), (), (), "recipe")
    resolve = Mock(return_value=recipe)
    get = Mock(
        side_effect=lambda snap, unit, version, rec: replace(
            _source(dependencies=((unit.path, "s"), ("include/shared.h", "h"))), unit=unit.path, version=version
        )
    )
    closure = Mock(
        side_effect=lambda snap, unit, version: (
            (unit.path, "s"),
            ("include/shared.h", "h"),
            ("include/shared.h", "h"),
            ("note.txt", "n"),
        )
    )
    evaluate = Mock(
        side_effect=lambda snap, path, text, source, sdk: (
            Finding("source.volatile-storage", f"debt {path}", path=path, line=2),
        )
    )
    gathered = Mock(side_effect=lambda cfg, groups: [[g[1](item) for item in g[2]] for g in groups])
    monkeypatch.setattr(policy.native, "recorded", lambda *a: None)
    monkeypatch.setattr(policy.recipes, "resolve", resolve)
    monkeypatch.setattr(policy._view, "get", get)
    monkeypatch.setattr(policy._view, "closure", closure)
    monkeypatch.setattr(policy._view, "active_lines", lambda view, path: {})
    monkeypatch.setattr(policy, "evaluate", evaluate)
    monkeypatch.setattr(policy.pool, "gather", gathered)
    monkeypatch.setattr(policy.store, "cached", lambda cfg, kind, key, produce: produce())
    monkeypatch.setattr(policy.types, "conflicts", lambda snap: [])
    findings, _ = policy.census(snapshot)
    assert [(f.path, f.line, f.blocking) for f in findings] == [
        (PATH, 2, False),
        ("src/second.c", 2, False),
        ("include/shared.h", 2, False),
        ("src/asm.s", 0, False),  # assembly outside every library directory is debt
    ]
    (call,) = gathered.call_args_list  # units and headers are one dispatch, each resolving its own warm items
    assert [g[0] for g in call.args[1]] == ["policy.census", "policy.census.headers", "policy.census.warnings"]
    assert call.args[1][2][2] == call.args[1][0][2]  # the kept warnings are read by the workers, not one by one here
    assert call.args[1][0][2] == [(snapshot, u) for u in units[:2]]
    assert call.args[1][1][2] == [(snapshot, "include/shared.h", True)]  # read once, not once per unit
    assert all(g[3] is not None for g in call.args[1][:2])
    assert [call.args for call in resolve.call_args_list] == [(snapshot.config, u, {}) for u in units[:2]] * 2
    assert [call.args for call in get.call_args_list] == [(snapshot, u, expected_version, recipe) for u in units[:2]]
    assert [call.args[1] for call in evaluate.call_args_list] == [PATH, "src/second.c", "include/shared.h"]
    assert evaluate.call_args_list[-1].args[3] is None  # a header has no single view
    assert all(call.args[0] is snapshot and call.args[4] is True for call in evaluate.call_args_list)
    assert all(call.args[2] == f"text of {call.args[1]}" for call in evaluate.call_args_list)

    gathered.reset_mock()
    seen = []
    monkeypatch.setattr(policy.types, "conflicts", lambda snap, paths=None: seen.append(paths) or [])
    from unbake import libasm

    planned = Mock(side_effect=AssertionError("a scoped census planned every library function"))
    monkeypatch.setattr(libasm, "plan", planned)
    policy.census(snapshot, only=["src/second.c"])
    (scoped,) = gathered.call_args_list
    assert [item[1].path for item in scoped.args[1][0][2]] == ["src/second.c"]  # one unit, not every unit
    assert [item[1] for item in scoped.args[1][1][2]] == ["include/shared.h"]  # only the headers that unit reads
    assert "src/second.c" in seen[0]


def test_directive_rules_never_let_whitespace_cross_a_line():
    import re
    import tomllib

    rules = tomllib.loads((Path(policy.__file__).parent / "resources/data/rules.toml").read_text())["rule"]
    directive = [r["regex"] for r in rules if r.get("regex", "").startswith("^")]
    assert directive and not any(r"^\s" in regex for regex in directive)  # a blank-line run is not rescanned per line
    assert re.search(directive[0], '\n\n\n  #  include "../x.h"\n', re.M)


def test_census_counts_a_name_declared_with_two_spellings(snapshot, monkeypatch):
    conflict = Finding(
        "types.conflict",
        "gX is declared 2 ways and defined nowhere",
        unit="gX",
        missing=("gX: src/a.c:3 s32 gX;", "gX: include/a.h:9 f32 gX;"),
    )
    snapshot.layout = LayoutMap({}, {}, {}, "layout", (), {})
    monkeypatch.setattr(policy.pool, "gather", lambda cfg, groups: [[], [], []])
    monkeypatch.setattr(policy.types, "conflicts", lambda snap: [conflict])
    assert [(f.key, f.unit, f.blocking) for f in policy.census(snapshot)[0]] == [("types.conflict", "gX", False)]


def test_census_counts_landed_view_conversions_as_debt_by_unit(snapshot, monkeypatch):
    unit = UnitSpec(PATH, "c", "group", ("member",), "test", {})
    snapshot.layout = LayoutMap(
        {},
        {"member": Member("member", "function", "c", "group", (Placement("a", ".text", 0, 4, 0),))},
        {unit.path: unit},
        "layout",
        (),
        {},
    )
    monkeypatch.setattr(policy.recipes, "resolve", lambda *a: None)
    monkeypatch.setattr(policy.native, "recorded", lambda *a: ("a.c:1: makes pointer from integer", "a.c:2: x"))
    monkeypatch.setattr(policy.native, "warnings", Mock(side_effect=AssertionError("the census never compiles")))
    found = policy._census_warnings((snapshot, unit))
    assert [(f.key, f.path, len(f.missing)) for f in found] == [("land.view-conversion", PATH, 2)]


@pytest.mark.parametrize("declaration", [False, True])
@pytest.mark.parametrize("case", ["different", "equal", "missing", "logic", "literal"])
def test_real_symbol_selection_guard(snapshot, monkeypatch, declaration, case):
    from types import SimpleNamespace

    a = Member("new_call", "function", "asm", "g", (Placement("a", ".text", 0, 8, 0x80000000),))
    b = Member("old_call", "function", "asm", "g", (Placement("b", ".text", 0, 12, 0x80000000),))
    snapshot.layout = SimpleNamespace(members={a.name: a, b.name: b}, units={})
    snapshot.config.project.version_macros = {"a": ("-DVERSION_A",), "b": ()}
    snapshot.versions = {
        "a": SimpleNamespace(id="a", symbols={a.name: 0x80000000}),
        "b": SimpleNamespace(id="b", symbols={b.name: 0x80000000}),
    }
    monkeypatch.setattr(policy.versions, "rom_bytes", lambda v, s, e: b"same" if case == "equal" else v.id.encode())
    if case == "missing":
        snapshot.versions["b"].symbols = {}
    left, right = ("void new_call(int);", "void old_call(int);") if declaration else ("new_call(x);", "old_call(x);")
    if case == "logic":
        right = right.replace("int", "float") if declaration else "old_call(x + 1);"
    if case == "literal":
        left, right = 'new_call("a");', 'old_call("b");'
    text = "#if defined(VERSION_A)\n" + left + "\n#else\n" + right + "\n#endif\n"
    findings = policy.evaluate(snapshot, PATH, text, None, False)
    assert [f.key for f in findings] == ([] if case == "different" else ["source.file-version-guard"])


@pytest.mark.parametrize(
    "kind,bytes_differ,expression,guarded",
    [
        ("data", True, "defined(VERSION_A)", True),
        ("data", True, "defined(VERSION_A) || !defined(VERSION_B)", True),
        ("data", False, "defined(VERSION_A)", False),  # the ROMs hold the same bytes: the guard is not needed
        ("data", True, "defined(VERSION_A) && SOMETHING_ELSE", False),  # a condition that is not a version name only
        ("c", True, "defined(VERSION_A)", False),
    ],
)  # code keeps the symbol-selection form
def test_a_data_unit_may_guard_the_versions_whose_bytes_really_differ(
    snapshot, monkeypatch, kind, bytes_differ, expression, guarded
):
    from types import SimpleNamespace

    member = Member(
        "table",
        "data",
        "data",
        "g",
        (Placement("a", ".rodata", 0, 8, 0x80000000), Placement("b", ".rodata", 0, 8, 0x80000000)),
    )
    unit = UnitSpec(PATH, kind, "g", ("table",), "t", {"add": [], "omit": []})
    snapshot.layout = SimpleNamespace(members={"table": member}, units={PATH: unit})
    snapshot.versions = {"a": SimpleNamespace(id="a"), "b": SimpleNamespace(id="b")}
    monkeypatch.setattr(policy.versions, "rom_bytes", lambda v, s, e: v.id.encode() if bytes_differ else b"same")
    credit = {"data": "data", "c": "code"}[kind]
    monkeypatch.setattr(
        policy.config,
        "load_resource",
        lambda name: (
            {"kind": {kind: {"credit": credit}}}
            if name == "units.toml"
            else {
                "rule": [
                    {
                        "id": "file-version-guard",
                        "scope": "directives",
                        "waivable": False,
                        "sentence": "no guard",
                        "regex": r"^[ \t]*#[ \t]*(?:if|ifdef|ifndef|elif)\b[^\n]*\bVERSION_\w+",
                    }
                ]
            }
        ),
    )
    text = f"#if {expression}\nunsigned int table[2] = {{1, 2}};\n#else\nunsigned int table[2] = {{3, 4}};\n#endif\n"
    findings = policy.evaluate(snapshot, PATH, text, None, False)
    assert [f.key for f in findings] == ([] if guarded else ["source.file-version-guard"])


def test_local_define_skips_headers(snapshot):
    text = "#define gDPSetColor(pkt, c) 1\n"
    assert policy.evaluate(snapshot, "include/gbi.h", text, None, False, only={"local-define"}) == ()
    found = policy.evaluate(snapshot, PATH, text, None, False, only={"local-define"})
    assert [f.line for f in found] == [1]


def test_census_runs_the_groups_it_is_given_in_its_own_dispatch(snapshot, monkeypatch):
    """The compile group fills the workers the census leaves idle: one gather, its results beside the findings."""
    snapshot.layout = LayoutMap({}, {}, {}, "layout", (), {})
    extra = ("land.compile", str, [1, 2])
    gathered = Mock(return_value=[[], [], [], ["1", "2"]])
    monkeypatch.setattr(policy.pool, "gather", gathered)
    monkeypatch.setattr(policy.types, "conflicts", lambda snap: [])
    findings, rest = policy.census(snapshot, [extra])
    assert (findings, rest) == ((), [["1", "2"]])
    (call,) = gathered.call_args_list
    assert [g[0] for g in call.args[1]] == [
        "policy.census",
        "policy.census.headers",
        "policy.census.warnings",
        "land.compile",
    ]


@pytest.mark.parametrize(
    ("text", "denied"),
    [
        ("union { s32 unk04; Effect33920 *next; };", True),
        ("union { AxisWave x; State_func_802337D0_de xState; };", True),
        ("union { char data[0]; Vec3 points[0]; f32 weights[0]; };", True),
        ("typedef union Named { int a; float b; } Named;", True),
        ("/* FAKEMATCH: layout */\nunion { int a; float b; };", True),
        ("/* union { int a; float b; }; */", False),
        ("// union { int a; float b; };", False),
        ('char *s = "union { int a; float b; };";', False),
        ("#define WORD union", False),
        ("struct S { int union_count; };", False),
    ],
)
@pytest.mark.parametrize("sdk", [False, True])
def test_union_view_rule(snapshot, text, denied, sdk):
    found = policy.evaluate(snapshot, PATH, text, None, sdk, {"union-view"})
    assert bool(found) == (denied and not sdk)
    if found:
        assert found[0].key == "source.union-view"
        assert found[0].reason.startswith("a union gives one storage two types; declare the one real type: ")
    assert "union-view" in policy.config.load_resource("rules.toml")["unclean"]


@pytest.mark.parametrize(
    ("old", "new", "blocked"),
    [
        ("", "union { int a; float b; };", True),
        ("union { int a; float b; };", "\n\nunion { int a; float b; };", False),
        ("union { int a; float b; };", "", False),
        ("union { int a; float b; };", "union { int a; float b; };\nunion { int c; };", True),
        ("", 'asm("nop");', True),
    ],
)
@pytest.mark.parametrize(
    ("path", "sdks"),
    [
        ("include/shared.h", (False, True)),
        ("include/shared.h", (True, True)),
        (PATH, (False, True)),
        (PATH, (True, True)),
        ("include/new.h", (True, True)),
    ],
)
@pytest.mark.parametrize("kind", ["c", "data"])
def test_census_changes_scope_and_sdk(snapshot, monkeypatch, old, new, blocked, path, sdks, kind):
    units = [UnitSpec(p, kind, str(n), ("member",), "test", {}) for n, p in enumerate((PATH, "src/second.c"))]
    snapshot.layout = LayoutMap(
        {
            str(n): Group(str(n), "main", ("member",), "authored", (), sdk, f"src/{n}.c", f"include/main/{n}.h")
            for n, sdk in enumerate(sdks)
        },
        {"member": Member("member", "function", "c", "0", (Placement("a", ".text", 0, 4, 0),))},
        {u.path: u for u in units},
        "layout",
        (),
        {},
    )
    monkeypatch.setattr(policy._view, "closure", lambda *a: (("include/shared.h", "pin"),))
    for module, name in ((policy._view, "get"), (policy.pool, "gather"), (policy.types, "conflicts")):
        monkeypatch.setattr(module, name, Mock(side_effect=AssertionError("staged check must only scan text")))
    findings, extra = policy.census(snapshot, changes={path: (old, new)})
    sdk = kind == "c" and all(sdks) if path == "include/shared.h" else sdks[0] if path == PATH else False
    assert bool([f for f in findings if f.blocking]) == (blocked and not sdk)
    assert extra == []
    assert all(f.path == path for f in findings)
    if old and old.strip() == new.strip() and not sdk:
        assert findings[0].line == 3 and not findings[0].blocking


@pytest.mark.parametrize("kind", ["c", "data"])
def test_census_changed_unit_requires_a_real_group(snapshot, monkeypatch, kind):
    unit = UnitSpec(PATH, kind, "missing", ("member",), "test", {})
    snapshot.layout = LayoutMap(
        {},
        {"member": Member("member", "data", "data", "", (Placement("a", ".data", 0, 4, 0),))},
        {PATH: unit},
        "layout",
        (),
        {},
    )
    monkeypatch.setattr(policy._view, "closure", lambda *a: ())
    with pytest.raises(KeyError, match="missing"):
        policy.census(snapshot, changes={PATH: ("", "unsigned char padding[4] = {0};")})


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
    snapshot.config.project.library_dirs = ("src/vendor",)
    findings = policy.evaluate(snapshot, path, "union U { int i; float f; };\n#define VIEW(x) (x)\n", None, False)
    assert (not findings) is exempt


@pytest.mark.parametrize("different,exact,accepted", [(False, True, False), (True, False, False), (True, True, True)])
def test_documented_guard_requires_different_rom_bytes_and_all_version_proofs(
    snapshot, monkeypatch, different, exact, accepted
):
    from types import SimpleNamespace

    from unbake import layout, ownership
    from unbake.contracts import Refusal

    unit = UnitSpec(PATH, "c", "g", ("f",), "t", {})
    member = Member("f", "function", "c", "g", tuple(Placement(v, ".text", 0, 4, 0x80000000) for v in ("a", "b")))
    snapshot.layout = SimpleNamespace(units={PATH: unit}, members={"f": member})
    snapshot.versions = {v: SimpleNamespace(id=v) for v in ("a", "b")}
    snapshot.config.project.version_macros = {"a": ("-DVERSION_A",), "b": ()}
    snapshot.config.project.version_differences = ({"function": "f", "sides": [["a"], ["b"]]},)
    monkeypatch.setattr(policy.versions, "rom_bytes", lambda v, *_: v.id.encode() if different else b"same")
    monkeypatch.setattr(layout, "overlay", lambda s, _: s)
    monkeypatch.setattr(policy.recipes, "resolve", lambda *_: None)
    monkeypatch.setattr(policy.native, "stamp", lambda s, u, v, r: (v, exact))
    monkeypatch.setattr(ownership, "derive", lambda s, u: (s, u))
    prove = Mock(side_effect=lambda args: (SimpleNamespace(version=args[2], exact=exact),))
    monkeypatch.setattr(policy.native, "prove_job", prove)
    text = "#ifdef VERSION_A\nvoid f(void) {}\n#else\nvoid f(void) { g(); }\n#endif\n"
    if accepted:
        assert policy._version_guards(snapshot, PATH, text) == {1}
        assert prove.call_count == 2
    else:
        with pytest.raises(Refusal, match=r"version|bytes"):
            policy._version_guards(snapshot, PATH, text)
        if not different:
            prove.assert_not_called()


def test_every_dynamic_source_rule_has_a_renderable_finding_key():
    from unbake import config

    for rule in config.load_resource("rules.toml")["rule"]:
        assert config.sentence("source." + rule["id"])


def _guarded(snapshot, monkeypatch, guard, enabled, macros):
    from types import SimpleNamespace

    from unbake import layout, ownership

    unit = UnitSpec(PATH, "c", "g", ("f",), "t", {})
    member = Member("f", "function", "c", "g", tuple(Placement(v, ".text", 0, 4, 0x80000000) for v in ("a", "b")))
    snapshot.layout = SimpleNamespace(units={PATH: unit}, members={"f": member})
    snapshot.versions = {v: SimpleNamespace(id=v) for v in ("a", "b")}
    snapshot.config.project.version_macros = macros
    disabled = [v for v in ("a", "b") if v not in enabled]
    snapshot.config.project.version_differences = ({"function": "f", "sides": [enabled, disabled]},)
    monkeypatch.setattr(policy.versions, "rom_bytes", lambda v, *_: v.id.encode())
    monkeypatch.setattr(layout, "overlay", lambda s, _: s)
    monkeypatch.setattr(policy.recipes, "resolve", lambda *_: None)
    monkeypatch.setattr(policy.native, "stamp", lambda s, u, v, r: v)
    monkeypatch.setattr(ownership, "derive", lambda s, u: (s, u))
    monkeypatch.setattr(policy.native, "prove_job", lambda args: (SimpleNamespace(version=args[2], exact=True),))


@pytest.mark.parametrize(
    ("guard", "head", "macros", "enabled"),
    [
        (
            "defined(VERSION_EU) || defined(VERSION_EU_X)",
            "#if defined(VERSION_EU) || defined(VERSION_EU_X)",
            ("-DVERSION_EU_X",),
            ["a"],
        ),
        ("VERSION_EU", "#ifndef VERSION_EU", ("-DVERSION_EU",), ["a"]),
        ("!defined(VERSION_EU)", "#if !defined(VERSION_EU)", ("-DVERSION_EU",), ["b"]),
        ("defined(VERSION_EU)", "#ifdef VERSION_EU", ("-DVERSION_EU",), ["a"]),
    ],
)
def test_guard_expression_or(snapshot, monkeypatch, guard, head, macros, enabled):
    _guarded(snapshot, monkeypatch, guard, enabled, {"a": macros, "b": ()})
    text = f"{head}\nvoid f(void) {{}}\n#else\nvoid f(void) {{ g(); }}\n#endif\n"
    assert policy._version_guards(snapshot, PATH, text) == {1}


def test_guard_selecting_no_holder_refuses(snapshot, monkeypatch):
    from unbake.contracts import Refusal

    _guarded(snapshot, monkeypatch, "", ["a"], {"a": ("-DVERSION_EU",), "b": ()})
    with pytest.raises(Refusal, match="whole sides"):
        policy._version_guards(snapshot, PATH, "#ifdef VERSION_ZZ\nint x;\n#endif\n")


@pytest.mark.parametrize(
    ("expr", "macros", "value"),
    [
        ("defined(A)", ("-DA",), True),
        ("defined(A)", (), False),
        ("A && !defined(B)", ("-DA",), True),
        ("(A || B) && B", ("-DA",), False),
        ("!(defined A)", (), True),
    ],
)
def test_truth_evaluates_version_expressions(expr, macros, value):
    assert policy._truth(expr, macros) is value
    with pytest.raises(ValueError, match=r"parenthesis|ends early|trailing"):
        policy._truth("(A", ())


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


def _placed_snapshot(snapshot):
    from types import SimpleNamespace

    data = Member("D_800D7000", "data", "data", "g", (Placement("us", ".data", 0, 0x20, 0x800D7000),))
    unit = UnitSpec(PATH, "data", "g", ("D_800D7000",), "t", {})
    snapshot.layout = SimpleNamespace(members={"D_800D7000": data}, units={PATH: unit})
    snapshot.digest = "snap"
    return snapshot


def test_address_literal_hits_placed_vram(snapshot):
    _placed_snapshot(snapshot)
    found = policy._address_literals(snapshot, PATH, "int a = 0x800D7010;\nint b = 0x800D8000;\nint c = 7;\n")
    assert found == {1: "0x800D7010"}


def test_cross_global_offset_hits(snapshot):
    _placed_snapshot(snapshot)
    code = "char *p = (char *)&D_800D7000 + 0x20;\nchar *q = (char *)&D_800D7000 + 0x1C;\n"
    assert list(policy._address_literals(snapshot, PATH, code)) == [1]


def test_text_findings_needs_no_snapshot():
    found = policy.text_findings(PATH, "int f(void) { return (u8 *)p + 4; }\nunion U { int a; };\n", False)
    assert sorted(f.key for f in found) == ["source.raw-offset", "source.union-view"]
    assert policy.text_findings(PATH, "volatile int x;\n", False) == ()


def test_audit_counts_and_refuses(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from unbake.contracts import Refusal

    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.c").write_text("union A { int a; };\nunion B { int b; };\n")
    (tmp_path / "src/b.c").write_text("int ok;\n")
    old = {"src/a.c": "union A { int a; };\n", "src/b.c": "int ok;\n"}

    def run(name, argv, cwd, **kw):
        if argv[1] == "diff":
            return SimpleNamespace(exit=0, stdout=b"src/a.c\0src/b.c\0")
        return SimpleNamespace(exit=0, stdout=old[argv[2].split(":", 1)[1]].encode())

    monkeypatch.setattr(policy.process, "run", run)
    monkeypatch.setattr(policy.process, "scratch", lambda root: tmp_path)
    monkeypatch.setattr(policy.shutil, "which", lambda name: "/usr/bin/git")
    with pytest.raises(Refusal) as refusal:
        policy.audit_command({"project": tmp_path, "against": "main"})
    assert [f.key for f in refusal.value.findings] == ["source.union-view"]
    (tmp_path / "src/a.c").write_text("union A { int a; };\n")
    # a layout group that declares no sdk
    (tmp_path / "layout.toml").write_text('[[group]]\nname = "g"\n\n[[unit]]\npath = "src/a.c"\ngroup = "g"\n')
    result = policy.audit_command({"project": tmp_path, "against": "main"})
    assert result["paths"] == ["src/a.c", "src/b.c"] and result["counts_after"] == {"source.union-view": 1}


def test_address_literal_ignores_values_below_the_mapped_address_space_even_inside_a_low_member(snapshot):
    from types import SimpleNamespace

    low = Member("D_00000000", "data", "data", "g", (Placement("us", ".data", 0, 0x100, 0),))
    unit = UnitSpec(PATH, "data", "g", ("D_00000000",), "t", {})
    snapshot.layout = SimpleNamespace(members={"D_00000000": low}, units={PATH: unit})
    snapshot.digest = "snap-low"
    code = "int pad[32] = {0x20};\nint n = 8;\nint z = 0;\nint m = 0021219444;\n"
    assert policy._address_literals(snapshot, PATH, code) == {}


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


def test_census_baseline_text_of_a_unit_whose_group_is_gone_counts_for_nothing(snapshot, monkeypatch):
    unit = UnitSpec(PATH, "c", "gone", ("member",), "test", {})
    snapshot.layout = LayoutMap(
        {},
        {"member": Member("member", "data", "data", "", (Placement("a", ".data", 0, 4, 0),))},
        {PATH: unit},
        "layout",
        (),
        {},
    )
    monkeypatch.setattr(policy._view, "closure", lambda *a: ())
    assert policy._changed_findings(snapshot, {PATH: "union U { int a; };\n"}, baseline=True) == []


def test_address_literal_skips_float_fragments_and_reads_octal(snapshot):
    _placed_snapshot(snapshot)
    code = "float a = 0.000212f;\nfloat b = 1e-05;\nfloat c = 1.5e+3;\nint d = 020000000000;\nint e = 0x800D7010U;\n"
    assert policy._address_literals(snapshot, PATH, code) == {5: "0x800D7010"}
    assert [policy._integer(t) for t in ("010", "09", "0", "0x10", "7")] == [8, None, 0, 16, 7]


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


def _three(snapshot, monkeypatch, byte_of, exact=True, rows=()):
    from types import SimpleNamespace

    from unbake import layout, ownership

    unit = UnitSpec(PATH, "c", "g", ("f",), "t", {})
    member = Member("f", "function", "c", "g", tuple(Placement(v, ".text", 0, 4, 0x80000000) for v in HOLDERS))
    snapshot.layout = SimpleNamespace(units={PATH: unit}, members={"f": member})
    snapshot.versions = {v: SimpleNamespace(id=v) for v in HOLDERS}
    snapshot.config.project.version_macros = {v: (f"-DVERSION_{v.upper().replace('-', '_')}",) for v in HOLDERS}
    snapshot.config.project.version_differences = tuple(rows)
    snapshot.peek = lambda path: THREE_WAY.encode()
    monkeypatch.setattr(policy.config, "kinds", lambda: {"c": {"credit": "code"}})
    monkeypatch.setattr(policy.versions, "rom_bytes", lambda v, *_: byte_of(v.id))
    monkeypatch.setattr(layout, "overlay", lambda s, _: s)
    monkeypatch.setattr(policy.recipes, "resolve", lambda *_: None)
    monkeypatch.setattr(policy.native, "stamp", lambda s, u, v, r: v)
    monkeypatch.setattr(ownership, "derive", lambda s, u: (s, u))
    prove = Mock(side_effect=lambda args: (SimpleNamespace(version=args[2], exact=exact),))
    monkeypatch.setattr(policy.native, "prove_job", prove)
    return prove


def _bytes_by_branch(v):
    return {"de": b"a", "eu-x": b"c"}.get(v, b"b")


def test_three_way_guard_derives_one_row_with_a_side_per_branch(snapshot, monkeypatch):
    prove = _three(snapshot, monkeypatch, _bytes_by_branch)
    assert policy._derived_rows(snapshot) == [{"function": "f", "sides": SIDES}]
    assert prove.call_count == len(HOLDERS)


def test_three_way_row_allows_every_branch_line(snapshot, monkeypatch):
    row = {"function": "f", "sides": SIDES}
    _three(snapshot, monkeypatch, _bytes_by_branch, rows=[row])
    assert policy._version_guards(snapshot, PATH, THREE_WAY) == {2, 4, 6}


def test_guard_with_identical_bytes_is_not_derived_and_refused(snapshot, monkeypatch):
    from unbake.contracts import Refusal

    prove = _three(snapshot, monkeypatch, lambda v: b"same", rows=[{"function": "f", "sides": SIDES}])
    assert policy._derived_rows(snapshot) == []
    prove.assert_not_called()
    with pytest.raises(Refusal, match="bytes do not differ"):
        policy._version_guards(snapshot, PATH, THREE_WAY)


def test_two_branches_sharing_bytes_across_sides_are_not_derived(snapshot, monkeypatch):
    _three(snapshot, monkeypatch, lambda v: b"a" if v == "de" else b"b" if v != "eu-x" else b"a")
    assert policy._derived_rows(snapshot) == []


def test_inexact_guard_is_not_derived(snapshot, monkeypatch):
    _three(snapshot, monkeypatch, _bytes_by_branch, exact=False)
    assert policy._derived_rows(snapshot) == []


def test_row_that_does_not_partition_every_holder_is_refused(snapshot, monkeypatch):
    from unbake.contracts import Refusal

    _three(snapshot, monkeypatch, _bytes_by_branch, rows=[{"function": "f", "sides": [["de"], ["eu"]]}])
    with pytest.raises(Refusal, match="partition"):
        policy._version_guards(snapshot, PATH, THREE_WAY)


def test_version_rows_text_replaces_old_blocks_in_place():
    old = (
        'schema = 3\n\n[project]\nid = "x"\n\n[[project.version_differences]]\nfunction = "g"\nguard = "A"\n'
        'enabled = ["a"]\ndisabled = ["b"]\n\n[build]\ncflags = []\n'
    )
    rows = [{"function": "f", "sides": SIDES}]
    new = policy._rows_text(old, rows)
    assert new == (
        'schema = 3\n\n[project]\nid = "x"\n\n[[project.version_differences]]\nfunction = "f"\n'
        'sides = [["de"], ["eu", "us", "us-rev1"], ["eu-x"]]\n\n[build]\ncflags = []\n'
    )
    assert policy._rows_text(new, rows) == new
    assert policy._rows_text(new, []) == 'schema = 3\n\n[project]\nid = "x"\n\n[build]\ncflags = []\n'


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


def test_unclean_is_judged_by_a_keyed_pool_once_per_snapshot(snapshot, monkeypatch, tmp_path):
    snapshot.config.project.root = tmp_path
    snapshot.layout.units = {"src/a.c": UnitSpec("src/a.c", "c", "g", (), "t", {})}
    monkeypatch.setattr(policy.config, "kinds", lambda: {"c": {"decompiled": True}})
    monkeypatch.setattr(policy, "_unclean_context", lambda snap: "context")
    calls = []
    rules = {**policy.config.load_resource("rules.toml"), "unclean": ["address-int"]}
    load = policy.config.load_resource
    monkeypatch.setattr(policy.config, "load_resource", lambda n: rules if n == "rules.toml" else load(n))
    monkeypatch.setattr(
        policy.pool,
        "map",
        lambda cfg, name, fn, items, key=None: (
            calls.append((name, key, items)) or [((("f", ("source.address-int",)),), {})] * len(items)
        ),
    )
    first = policy.unclean(snapshot)
    assert policy.unclean(snapshot) == first
    assert first == {"f"} and len(calls) == 1 and calls[0][:2] == ("policy.unclean", policy._unclean_key)


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


def test_gbi_template_encodes_the_v1_texture_command_with_an_eight_bit_on_field():
    words = _gbi_words("gSPTexture", 1, [0xFFFF, 0x8000, 0, 1, 1], {"G_TEXTURE": 0xBB})
    assert words == (0xBB000000 | (1 << 8) | 1, 0xFFFF8000)


def _keyed(monkeypatch, reach, types=b"types"):
    monkeypatch.setattr(policy._view, "closure", lambda snap, unit, version: reach)
    monkeypatch.setattr(policy, "_census_version", lambda snap, unit: "a")
    monkeypatch.setattr(policy, "_unclean_context", lambda snap: "context")
    unit = UnitSpec("src/a.c", "c", "g", (), "t", {})
    return SimpleNamespace(
        read=lambda path: b"text", peek=lambda path: types, layout=SimpleNamespace(units={unit.path: unit}, members={})
    )


def test_the_unclean_key_moves_with_the_job_that_shapes_its_value(monkeypatch):
    seen = []
    monkeypatch.setattr(policy, "code_identity", lambda *fs: seen.append(fs) or "id")
    policy._unclean_key((_keyed(monkeypatch, ()), "src/a.c", ("r",), ()))
    assert policy._unclean_unit in seen[0]  # a cached verdict of another result shape is never read back


def test_a_header_edit_moves_the_unclean_verdict_key_and_a_type_map_edit_does_not(monkeypatch):
    """A landing that changed one type map row (World) judged all 7,272 units again: no verdict reads the type map."""

    def key(reach, types=b"types"):
        return policy._unclean_key((_keyed(monkeypatch, reach, types), "src/a.c", ("r",), ()))

    base = key((("include/a.h", "pin1"),))
    assert key((("include/a.h", "pin1"),)) == base
    assert key((("include/a.h", "pin2"),)) != base  # the unit's text is the same, the header it reads is not
    assert key((("include/a.h", "pin1"),), b"other types") == base


def _header_scene(snapshot, monkeypatch, differs, source):
    from types import SimpleNamespace

    function = Member(
        "user",
        "function",
        "asm",
        "g",
        (Placement("a", ".text", 0, 8, 0x80000000), Placement("b", ".text", 0, 8, 0x80000000)),
    )
    unit = UnitSpec(PATH, "c", "g", ("user",), "t", {"add": [], "omit": []})
    snapshot.layout = SimpleNamespace(members={"user": function}, units={PATH: unit})
    snapshot.config.project.version_macros = {"a": ("-DVERSION_A",), "b": ()}
    snapshot.versions = {"a": SimpleNamespace(id="a"), "b": SimpleNamespace(id="b")}
    snapshot.peek = lambda path: source.encode()
    reads = []

    def rom(v, s, e):
        reads.append(v.id)
        return v.id.encode() if differs else b"same"

    monkeypatch.setattr(policy.versions, "rom_bytes", rom)
    monkeypatch.setattr(policy.config, "kind", lambda k: {"credit": "code"})
    monkeypatch.setattr(policy.effort, "memo", lambda key, build: build())
    monkeypatch.setattr("unbake.headers.consumers", lambda s, changed: (PATH,))
    return reads


HEADER = "#if defined(VERSION_A)\n#define ACTIVE(p) ((p)->flag)\n#else\n#define ACTIVE(p) 1\n#endif\n"


@pytest.mark.parametrize(
    "differs,source,expected",
    [
        (True, "void f() { if (ACTIVE(x)) {} }", {1}),
        (False, "void f() { if (ACTIVE(x)) {} }", set()),  # the bytes are equal: the guard is not needed
        (True, "void f() { if (OTHER(x)) {} }", set()),  # the unit never mentions a name the guard defines
    ],
)
def test_a_header_guard_stands_when_a_user_of_its_names_differs_per_side(
    snapshot, monkeypatch, differs, source, expected
):
    _header_scene(snapshot, monkeypatch, differs, source)
    assert policy._header_guards(snapshot, "include/h.h", HEADER) == expected
    assert policy._header_guards(snapshot, "src/unit.c", HEADER) == set()


def test_a_keyword_before_a_parenthesis_is_no_name_of_the_chain(snapshot, monkeypatch):
    _header_scene(snapshot, monkeypatch, True, "void f() { if (x) { return (1); } }")
    body = "do { if (p) { return (1); } } while (0)"
    text = f"#if defined(VERSION_A)\n#define GO(p) {body}\n#else\n#define GO(p) 0\n#endif\n"
    assert policy._chain_names(text, 1) == {"GO"}
    assert policy._header_guards(snapshot, "include/h.h", text) == set()


def test_a_header_guard_reads_no_bytes_for_a_unit_that_names_nothing_of_it(snapshot, monkeypatch):
    reads = _header_scene(snapshot, monkeypatch, True, "void f() {}")
    policy._header_guards(snapshot, "include/h.h", HEADER)
    assert reads == []


def test_unclean_rules_given_units_judges_only_those_units(snapshot, monkeypatch, tmp_path):
    from unbake.contracts import UnitSpec

    snapshot.config.project.root = tmp_path
    monkeypatch.setattr(policy, "_unclean_context", lambda snap: "context")
    snapshot.layout.units = {p: UnitSpec(p, "c", "g", (), "t", {}) for p in ("src/a.c", "src/b.c", "src/c.c")}
    monkeypatch.setattr(policy.config, "kinds", lambda: {"c": {"decompiled": True}})
    calls = []
    rules = {**policy.config.load_resource("rules.toml"), "unclean": ["address-int"]}
    load = policy.config.load_resource
    monkeypatch.setattr(policy.config, "load_resource", lambda n: rules if n == "rules.toml" else load(n))
    monkeypatch.setattr(
        policy.pool,
        "map",
        lambda cfg, name, fn, items, key=None: (
            calls.append([i[1] for i in items])
            or [((), {}) if name == "policy.unclean" and fn is policy._unclean_job else ()] * len(items)
        ),
    )
    policy.unclean_rules(snapshot)
    everyone = calls[-1]
    assert len(everyone) >= 2
    policy.unclean_rules(snapshot, {everyone[0]})
    assert calls[-1] == [everyone[0]]


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


def _spanned(starts):
    from types import SimpleNamespace as NS

    place = NS(version="v", section=".text", vram=0x100, size=0x40)
    member = NS(placements=(place,), kind="function")
    return NS(layout=NS(members={"head": member}), digest="d"), starts


def test_withheld_bytes_are_the_pieces_of_the_withheld_functions_inside_a_member(monkeypatch):
    snapshot, starts = _spanned([(0x100, "head"), (0x110, "helper"), (0x120, "clean"), (0x140, "next")])
    monkeypatch.setattr(policy, "_function_starts", lambda s, v: starts)
    assert policy.withheld_bytes(snapshot, {"helper": ()}, "head", "v") == 0x10
    assert policy.withheld_bytes(snapshot, {"head": ()}, "head", "v") == 0x10
    assert policy.withheld_bytes(snapshot, {"head": (), "clean": ()}, "head", "v") == 0x30
    assert policy.withheld_bytes(snapshot, {"next": ()}, "head", "v") == 0
    assert policy.withheld_bytes(snapshot, {}, "head", "v") == 0


def test_a_member_with_no_function_inside_withholds_all_its_bytes(monkeypatch):
    snapshot, starts = _spanned([])
    monkeypatch.setattr(policy, "_function_starts", lambda s, v: starts)
    assert policy.withheld_bytes(snapshot, {"head": ()}, "head", "v") == 0x40


def test_credited_names_members_and_placed_helpers_not_inlined_ones(monkeypatch):
    from types import SimpleNamespace as NS

    text = "static int inl(int x) { return x; }\nint helper(void) { return inl(1); }\nint head(void) { return 0; }\n"
    unit = NS(members=("head",))
    snapshot = NS(
        layout=NS(units={"src/a.c": unit}, members={"head": NS(kind="function")}),
        versions={"v": NS(symbols={"head": 0x100, "helper": 0x110})},
    )
    monkeypatch.setattr(policy, "_census_version", lambda s, u: "v")
    assert policy.credited(snapshot, "src/a.c", text) == ["head", "helper"]


def test_a_finding_in_a_placed_helper_withholds_the_helper_not_its_member():
    text = "int helper(void *p) { return *(int *)((u8 *)p + 4); }\nint head(void) { return helper(0); }\n"
    found = [Finding("source.raw-offset", "raw: x", line=1)]
    assert policy.withheld(text, found, ["head", "helper"]) == {"helper": ("source.raw-offset",)}
    assert policy.withheld(text, found, ["head"]) == {"head": ("source.raw-offset",)}  # inlined: its caller
