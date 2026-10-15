"""Tests for headers: catalog and fold."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pycparser import CParser, c_generator

from unbake import effort, headers, types
from unbake.contracts import Refusal

HEADER = (
    "#ifndef G\n#define G\ntypedef int s32;\nextern s32 gCount;\nvoid known(int a);\n"
    "struct Pair { int a; int b; };\n#endif\n"
)


def snap(tmp_path: Path, files: dict[str, str]):
    root = tmp_path / "proj"
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    group = SimpleNamespace(segment="main", name="grp", header="include/main/grp.h")
    return SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(root=root, toolchain="t")),
        overlays={},
        tree=False,
        layout=SimpleNamespace(groups={"g": group}),
        read=lambda p: (root / p).read_bytes(),
        peek=lambda p: (root / p).read_bytes() if (root / p).exists() else None,
    )


def unit():
    return SimpleNamespace(path="src/a.c", group="g")


def view(text):
    return SimpleNamespace(text=text, version="a")


def _known(monkeypatch, evidence):
    row = {"evidence": evidence, "signature": "int fresh(int b);"}
    monkeypatch.setattr(types, "load", lambda s: {"function": {"fresh": row}})


@pytest.fixture(autouse=True)
def inline(monkeypatch):
    """The catalog fans out to the pool and its cache; these tests run both in process."""
    monkeypatch.setattr(
        headers.pool, "map", lambda config, name, function, items, key=None: [function(i) for i in items]
    )


@pytest.fixture
def no_types(monkeypatch):
    monkeypatch.setattr(types, "load", lambda s: {"function": {}})


def test_catalog_names(tmp_path):
    result = headers.catalog(snap(tmp_path, {"include/g.h": HEADER}))
    assert {"s32", "gCount", "known", "struct Pair"} <= set(result) and "Pair" not in result
    assert result["gCount"][:2] == ("include/g.h", 4)


def _count_scans(monkeypatch):
    calls = []
    real = headers._scanned
    monkeypatch.setattr(headers, "_scanned", lambda s, paths, *a: calls.append(list(paths)) or real(s, paths, *a))
    return calls


def test_catalog_scans_unchanged_headers_once(tmp_path, monkeypatch):
    snapshot = snap(tmp_path, {"include/g.h": HEADER})
    calls = _count_scans(monkeypatch)
    first = headers.catalog(snapshot)
    assert headers.catalog(snapshot) is first
    assert len(calls) == 1


def test_catalog_follows_an_edited_header(tmp_path, monkeypatch):
    snapshot = snap(tmp_path, {"include/g.h": HEADER})
    calls = _count_scans(monkeypatch)
    assert "fresh" not in headers.catalog(snapshot)
    (tmp_path / "proj" / "include" / "g.h").write_text(HEADER + "void fresh(int a);\n")
    effort.forget("pin")  # what store.write does after it writes a project file
    assert headers.catalog(snapshot)["fresh"][0] == "include/g.h"
    assert len(calls) == 2


_BASE_HEADERS = {"include/a.h": "extern int x;\nextern int y;\n", "include/b.h": "extern long y;\nextern int z;\n"}


@pytest.mark.parametrize(
    "writes",
    [
        {"include/b.h": "extern long y;\nextern int z;\nextern int w;\nextern short x;\n"},  # adds; x stays a.h's
        {"include/a.h": "extern int q;\nextern int x;\nextern int y;\n"},  # a first declarer's line moves
        {"include/a.h": "extern int x;\n"},  # y loses its first declarer: b.h's is next
        {"include/0.h": "extern char x;\n", "include/b.h": None},  # a header sorted first; one removed
    ],
)
def test_an_overlays_catalog_reads_only_the_headers_it_changes_and_equals_a_full_build(tmp_path, monkeypatch, writes):
    base = snap(tmp_path, _BASE_HEADERS)
    parent = headers.catalog(base)
    overlays = {p: (t.encode() if t is not None else None) for p, t in writes.items()}
    child = SimpleNamespace(**{**vars(base), "overlays": overlays, "digest": f"child{sorted(writes)}"})
    child.read = lambda p: overlays[p] if p in overlays else base.read(p)
    child.peek = lambda p: overlays[p] if p in overlays else base.peek(p)
    effort.memo(("parent", child.digest), lambda: (base, overlays))
    calls = _count_scans(monkeypatch)
    derived = headers.catalog(child)
    effort.forget("catalog", "parent")
    assert derived == headers.catalog(child) and parent == headers.catalog(base)
    if "include/a.h" not in writes or "y" in writes["include/a.h"]:
        assert sorted(p for paths in calls[:2] for p in paths) == sorted(
            [p for p in writes if p in _BASE_HEADERS] + [p for p, t in writes.items() if t is not None]
        )  # each changed header before and after, never the others


def test_fold_loop_without_writes_builds_the_catalog_once(tmp_path, monkeypatch):
    snapshot = snap(tmp_path, {"include/g.h": HEADER})
    calls = _count_scans(monkeypatch)
    for _ in range(3):
        headers.catalog(snapshot)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "files, agreed, conflicts",
    [
        # one type in a header and two sources: it wins, whichever file spells it
        (
            {
                "include/g.h": "extern float D_1;\n",
                "src/a.c": "extern  float  D_1;\nint f(void) { return 0; }\n",
                "src/b.c": "extern float D_1;\n",
            },
            {"D_1": "float @"},
            {},
        ),
        # a pointer, an array and a prototype keep what follows and precedes the name
        (
            {"src/a.c": "extern char *D_2[];\nextern int p(unsigned short);\n", "src/b.c": "extern char *D_2[];\n"},
            {"D_2": "char * @[]", "p": "int @(unsigned short)"},
            {},
        ),
        # several declarators in one statement
        (
            {"src/a.c": "extern const f32 D_3[], D_4[];\n", "src/b.c": "extern const f32 D_4[];\n"},
            {"D_3": "const f32 @[]", "D_4": "const f32 @[]"},
            {},
        ),
        # two types: refused with every declaration, file and line
        (
            {"src/a.c": "extern u32 D_5;\n", "src/b.c": "\nextern s32 D_5;\n"},
            {},
            {"D_5": ["D_5: src/a.c:1 extern u32 D_5;", "D_5: src/b.c:2 extern s32 D_5;"]},
        ),
        # a function body is not read for declarations, and f32 is float
        (
            {"src/a.c": "int f(void) { if (1) { return 1; } return 0; }\nextern f32 D_8;\nextern float D_8;\n"},
            {"D_8": "f32 @"},
            {},
        ),
        # a name some source defines has its type from that definition
        (
            {"src/a.c": "extern u32 D_6;\n", "src/b.c": "const float D_6 = 1.0f;\n"},
            {},
            {"D_6": ["D_6: src/b.c:1 const float D_6", "D_6: src/a.c:1 extern u32 D_6;"]},
        ),
        # a definition and an extern that spell one type (an array length aside) agree; a local is no definition
        (
            {
                "src/a.c": "extern const f32 D_7[];\n",
                "src/b.c": "const f32 D_7[4] = {1};\nvoid f(void) {\n    s32 D_7;\n}\n",
            },
            {},
            {},
        ),
        # two definitions of one name with other types, and a definition against a struct extern
        (
            {"src/a.c": "const float D_9[4] = {1};\n", "src/b.c": "s32 D_9;\n"},
            {},
            {"D_9": ["D_9: src/a.c:1 const float D_9[4]", "D_9: src/b.c:1 s32 D_9"]},
        ),
        (
            {"src/a.c": "extern Shared D_10;\n", "src/b.c": "const float D_10[] = {1};\n"},
            {},
            {"D_10": ["D_10: src/b.c:1 const float D_10[]", "D_10: src/a.c:1 extern Shared D_10;"]},
        ),
    ],
)
def test_disagreements_give_one_type_per_name_where_the_declarations_agree(tmp_path, files, agreed, conflicts):
    got, findings = headers.disagreements(snap(tmp_path, files))
    assert got == agreed
    assert {f.unit: list(f.missing) for f in findings} == conflicts
    assert all(not f.blocking and f.key == "types.conflict" for f in findings)


def test_disagreements_leave_out_the_names_the_caller_defines(tmp_path):
    files = {"src/a.c": "extern u32 D_7;\n", "src/b.c": "extern s32 D_7;\n"}
    assert headers.disagreements(snap(tmp_path, files), {"D_7"}) == ({}, [])


def test_disagreements_cache_content_paths_and_explicit_definitions(tmp_path, monkeypatch):
    files = {"include/g.h": "extern int g;\n", "src/a.c": "extern float g;\n"}
    s = snap(tmp_path, files)
    cache, jobs = {}, []

    def cached(config, name, function, items, key=None):
        result = []
        for item in items:
            slot = (name, key(item)) if key else None
            if slot not in cache:
                jobs.append(name)
                value = function(item)
                if slot is not None:
                    cache[slot] = value
            else:
                value = cache[slot]
            result.append(value)
        return result

    monkeypatch.setattr(headers.pool, "map", cached)
    first = headers.disagreements(s)
    assert len(first[1]) == 1
    cold = list(jobs)
    s.commit = "unrelated-commit"
    assert headers.disagreements(s) == first and jobs == cold
    assert headers.disagreements(s, {"g"}) == ({}, [])
    assert jobs.count("headers.disagreements") == 2
    assert jobs.count("headers.catalog") == 2  # the changed definition set keeps file scans warm
    s.overlays["src/a.c"] = b"extern int g;\n"
    read = s.read
    s.read = lambda p: s.overlays[p] if p in s.overlays else read(p)
    assert headers.disagreements(s)[0] == {"g": "int @"}
    assert jobs.count("headers.disagreements") == 3
    assert jobs.count("headers.catalog") == 3
    s.overlays["src/a.c"] = None
    assert headers.disagreements(s)[0] == {"g": "int @"}
    assert jobs.count("headers.disagreements") == 4
    s.overlays["include/other.h"] = b"extern int other;\n"
    assert headers.disagreements(s)[0] == {"g": "int @", "other": "int @"}
    assert jobs.count("headers.disagreements") == 5
    assert jobs.count("headers.catalog") == 4


def test_fold_moves_new_prototype_and_removes_known(tmp_path, no_types):
    source = "void known(int a);\nvoid fresh(int b);\nint f(void) { return 0; }\n"
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": source})
    text = '# 1 "src/a.c"\nvoid known(int a);\nvoid fresh(int b);\nint f(void) { return 0; }\n'
    writes, findings = headers.fold(s, unit(), view(text))
    assert findings == ()
    assert "void fresh(int b);" in writes["include/main/grp.h"].decode()
    assert "fresh" not in writes["src/a.c"].decode()
    assert "known(int a);" not in writes["src/a.c"].decode()
    assert '#include "g.h"' in writes["src/a.c"].decode()


def test_fold_moves_a_typedef_whose_name_is_also_a_struct_tag(tmp_path, no_types):
    header = "struct Obj { int a; };\n"
    source = "typedef struct { int b; } Obj;\nint f(Obj *o) { return o->b; }\n"
    s = snap(tmp_path, {"include/g.h": header, "src/a.c": source})
    text = '# 1 "src/a.c"\n' + source
    writes, findings = headers.fold(s, unit(), view(text))
    assert "typedef struct { int b; } Obj;" in writes["include/main/grp.h"].decode()  # moved, not dropped as known
    assert "include/g.h" not in writes and not findings


@pytest.mark.parametrize(
    "candidate, conflicts",
    [
        ("extern s32 gCount;", False),
        ("extern int gCount;", False),
        ("extern s32 gCount[];", True),
        ("extern float gCount;", True),
    ],
)
def test_fold_refuses_a_declaration_of_a_known_name_with_another_type(tmp_path, no_types, candidate, conflicts):
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": candidate + "\n"})
    text = '# 1 "include/g.h"\ntypedef int s32;\n# 1 "src/a.c"\n' + candidate + "\n"
    _, findings = headers.fold(s, unit(), view(text))
    assert [f.key for f in findings] == (["headers.conflict"] if conflicts else [])
    if conflicts:
        assert "gCount" in findings[0].reason and "include/g.h:4" in findings[0].reason


def test_fold_includes_in_the_group_header_what_its_moved_declarations_name(tmp_path, no_types):
    source = "typedef struct { s32 a; } Rec;\nint f(Rec *r) { return r->a; }\n"
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": source})
    writes, findings = headers.fold(s, unit(), view('# 1 "include/g.h"\ntypedef int s32;\n# 1 "src/a.c"\n' + source))
    header = writes["include/main/grp.h"].decode()
    assert not findings
    assert "typedef struct { s32 a; } Rec;" in header
    assert header.index('#include "g.h"') < header.index("typedef struct")
    assert '#include "main/grp.h"' in writes["src/a.c"].decode()


@pytest.mark.parametrize("reached", [False, True])
def test_a_landed_fold_moves_no_declaration_whose_types_the_group_header_does_not_reach(tmp_path, no_types, reached):
    owner = '#include "sel.h"\n' if reached else "extern int gOther;\n"
    files = {"include/sel.h": "typedef struct Sel { int a; } Sel;\n", "include/main/grp.h": owner}
    source = "extern void pick(Sel *s);\nint g(void) { return 0; }\n"
    s = snap(tmp_path, {**files, "src/a.c": source})
    text = '# 1 "include/sel.h"\ntypedef struct Sel { int a; } Sel;\n# 1 "src/a.c"\n' + source
    writes, findings = headers.fold(s, unit(), view(text), strict=False)
    group = (writes.get("include/main/grp.h") or b"").decode()
    assert findings == () and ("pick" in group) == reached  # the header would bring sel.h to every includer
    assert ("extern void pick(Sel *s);" in writes.get("src/a.c", source.encode()).decode()) != reached


@pytest.mark.parametrize(
    ("other", "moves"),
    [
        ("int g(void) {\n    int pick(void *, int);\n    return pick(0, 1);\n}\n", False),  # a block-scope prototype
        ("int pick(void *, int);\nint g(void) { return pick(0, 1); }\n", False),  # one at file scope, no extern
        ("int g(void) {\n    pick(0, 1);\n    return 0;\n}\n", True),  # a call declares nothing
    ],
)
def test_a_landed_fold_moves_no_declaration_another_source_prototypes_otherwise(tmp_path, no_types, other, moves):
    source = "extern void pick(void *p, int n);\nint f(void) { return 0; }\n"
    s = snap(tmp_path, {"include/main/grp.h": "", "src/a.c": source, "src/b.c": other})
    writes, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\n' + source), strict=False)
    assert findings == () and ("pick" in (writes.get("include/main/grp.h") or b"").decode()) == moves


@pytest.mark.parametrize("evidence", ["authored", "landed"])
def test_fold_conflict_with_landed_signature(tmp_path, monkeypatch, evidence):
    _known(monkeypatch, evidence)
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": "void fresh(int b);\n"})
    writes, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\nvoid fresh(int b);\n'))
    assert [f.key for f in findings] == ["headers.conflict"]
    assert "candidate declares" in findings[0].reason and "type map has int fresh(int b);" in findings[0].reason
    assert findings[0].path == "src/a.c"
    assert "include/main/grp.h" not in writes


def test_a_landed_fold_keeps_its_declaration_where_the_type_map_spells_the_function_otherwise(tmp_path, monkeypatch):
    _known(monkeypatch, "landed")  # move folds landed sources: what builds today is no finding and stays put
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": "void fresh(int b);\n"})
    writes, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\nvoid fresh(int b);\n'), strict=False)
    assert findings == ()
    assert "void fresh(int b);" in writes.get("src/a.c", b"void fresh(int b);").decode()
    assert "fresh" not in (writes.get("include/main/grp.h") or b"").decode()


@pytest.mark.parametrize(
    "source, header",
    [
        ("extern float gCount;\n", '# 1 "include/g.h"\ntypedef int s32;\nextern s32 gCount;\n'),  # beside its header
        ("struct Pair { int c; };\nint f(struct Pair *p) { return p->c; }\n", ""),  # a record its header defines
        ("float known(int a) { return a; }\n", ""),  # a definition its header prototypes otherwise
    ],
)
def test_a_landed_fold_judges_no_declaration_the_source_builds_with_today(tmp_path, no_types, source, header):
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": source})
    text = header + '# 1 "src/a.c"\n' + source
    _, strict = headers.fold(s, unit(), view(text))
    writes, landed = headers.fold(s, unit(), view(text), strict=False)
    assert [f.key for f in strict] == ["headers.conflict"] and landed == ()
    assert '#include "g.h"' not in writes.get("src/a.c", b"").decode()  # the header it contradicts stays out


@pytest.mark.parametrize("through", [False, True])
def test_a_landed_fold_includes_no_header_that_contradicts_a_block_scope_declaration(tmp_path, no_types, through):
    declares = "extern int gOther;\nextern char *gPtr;\n"  # gPtr: char * here, void * in the function
    files = {"include/g.h": declares if not through else '#include "deep.h"\nextern int gOther;\n'}
    files["include/deep.h"] = "extern char *gPtr;\n"
    source = "extern int gOther;\nint f(void) {\n    extern void *gPtr;\n    return gOther + (gPtr != 0);\n}\n"
    s = snap(tmp_path, {**files, "src/a.c": source})
    writes, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\n' + source), strict=False)
    folded = writes.get("src/a.c", source.encode()).decode()
    assert findings == () and '#include "g.h"' not in folded and "extern int gOther;" in folded
    plain = source.replace("void *gPtr", "char *gPtr")  # one that agrees takes the header as before
    writes, _ = headers.fold(
        snap(tmp_path, {**files, "src/a.c": plain}), unit(), view('# 1 "src/a.c"\n' + plain), strict=False
    )
    assert '#include "g.h"' in writes["src/a.c"].decode()


def test_a_landed_fold_includes_no_header_that_reaches_another_type_of_a_file_scope_declaration(tmp_path, no_types):
    files = {"include/g.h": '#include "deep.h"\nextern int gOther;\n', "include/deep.h": "extern char *gPtr;\n"}
    source = "extern int gOther;\nextern void *gPtr;\nint f(void) { return gOther + (gPtr != 0); }\n"
    s = snap(tmp_path, {**files, "src/a.c": source})
    writes, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\n' + source), strict=False)
    folded = writes.get("src/a.c", source.encode()).decode()
    assert findings == () and '#include "g.h"' not in folded and "extern void *gPtr;" in folded


@pytest.mark.parametrize(
    ("deep", "defines", "agrees"),
    [
        ("extern int func_B(char *);\n", "int func_B(void *p) { return p != 0; }\n", "char *p"),  # a function
        ("extern char *gPtr;\n", "void *gPtr;\n", "char *gPtr"),  # an object it defines
        ("extern unsigned char gBuf[];\n", "unsigned char gBuf[4] = {0};\n", None),  # only its size: compatible
    ],
)
def test_a_landed_fold_includes_no_header_that_reaches_another_type_of_a_definition(
    tmp_path, no_types, deep, defines, agrees
):
    files = {"include/g.h": '#include "deep.h"\nextern int gOther;\n', "include/deep.h": deep}
    source = "extern int gOther;\n" + defines + "int f(void) { return gOther; }\n"
    writes, findings = headers.fold(
        snap(tmp_path, {**files, "src/a.c": source}), unit(), view('# 1 "src/a.c"\n' + source), strict=False
    )
    folded = writes.get("src/a.c", source.encode()).decode()
    assert findings == () and ('#include "g.h"' in folded) == (agrees is None)
    if agrees:  # the definition the header spells the same way takes the header as before
        plain = source.replace("void *p", agrees).replace("void *gPtr", agrees)
        writes, _ = headers.fold(
            snap(tmp_path, {**files, "src/a.c": plain}), unit(), view('# 1 "src/a.c"\n' + plain), strict=False
        )
        assert '#include "g.h"' in writes["src/a.c"].decode()


@pytest.mark.parametrize(
    "deep",
    [
        "typedef struct Owner Owner;\n",  # a typedef the source defines too: a redefinition
        "struct Owner { int b; };\n",  # a record it defines
        "#define gWord (gOther + 1)\n",  # a macro of a name it spells: `int gWord;` would read `int (gOther + 1);`
    ],
)
def test_a_landed_fold_includes_no_header_that_brings_a_type_or_macro_the_source_defines_itself(
    tmp_path, no_types, deep
):
    files = {"include/g.h": '#include "deep.h"\nextern int gOther;\n', "include/deep.h": deep}
    source = (
        "typedef struct Owner { int a; } Owner;\nextern int gOther;\nint gWord;\nint f(Owner *o) { return o->a; }\n"
    )
    s = snap(tmp_path, {**files, "src/a.c": source})
    writes, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\n' + source), strict=False)
    assert findings == () and '#include "g.h"' not in writes.get("src/a.c", source.encode()).decode()
    today = '# 1 "include/deep.h"\n# 1 "src/a.c"\n'  # a header it includes today already builds beside it
    assert headers._contradicts(s, "include/g.h", headers._Mine({}, {"Owner", "struct Owner"}, {"gWord"}, set()))
    assert not headers._contradicts(
        s, "include/g.h", headers._Mine({}, {"Owner"}, {"gWord"}, headers._viewed(s, today))
    )


def test_a_landed_fold_keeps_a_declaration_a_header_only_spells_compatibly(tmp_path, no_types):
    source = "int scale();\nint g(void) { return scale(1.5f); }\n"  # K&R: the float goes as a double
    s = snap(tmp_path, {"include/g.h": "int scale(float x);\n", "src/a.c": source})
    text = '# 1 "include/g.h"\nint scale(float x);\n# 1 "src/a.c"\n' + source
    writes, findings = headers.fold(s, unit(), view(text), strict=False)
    assert findings == () and "int scale();" in writes.get("src/a.c", source.encode()).decode()
    strict, _ = headers.fold(s, unit(), view(text))
    assert "int scale();" not in strict["src/a.c"].decode()  # a fresh draft takes the prototype


def test_an_added_include_goes_after_the_includes_before_the_first_statement_not_a_later_one():
    text = '#include "a.h"\nextern int x;\nint f(void) { return x; }\n#include "late.h"\nint g(void) { return 0; }\n'
    out = headers.included(text, ["b.h"])
    assert out.index('#include "b.h"') < out.index("int f(void)")  # what f uses is declared before it
    assert headers.included("#ifndef G\n#define G\n#endif\n", ["b.h"]).startswith(
        '#ifndef G\n#define G\n#include "b.h"'
    )


def test_a_unit_whose_body_holds_a_gnu_statement_expression_parses(tmp_path, no_types):
    source = 'int f(int *p) {\n    return ({ int *c = p; char *s = "})"; *c; }) + 1;\n}\n'  # an expanded va_arg
    nodes = headers.parse('# 1 "src/a.c"\n' + source, "src/a.c").ext
    assert [n.decl.name for n in nodes] == ["f"] and nodes[0].coord.line == 1


def test_a_function_pointer_declaration_names_its_pointer_and_spells_its_parameters():
    rows = headers._declared("extern void (*hook)(s32 offset);\nextern void (*other)(void);\n")
    assert [name for name, _, _ in rows] == ["hook", "other"]  # never the return type `void`
    spelled = {
        t: headers._spelling(t, "hook") for t in ("void (*hook)(s32 a);", "void (*hook)(int);", "void (*hook)();")
    }
    assert spelled == {
        "void (*hook)(s32 a);": "void(*@)(int)",
        "void (*hook)(int);": "void(*@)(int)",
        "void (*hook)();": "void(*@)()",
    }


def test_a_landed_fold_never_moves_a_function_pointer_a_header_types_otherwise(tmp_path, no_types):
    header = "extern void (*hook)(void);\n"  # what another unit includes
    source = "extern void (*hook)(int offset);\nvoid f(void) { hook(1); }\n"
    s = snap(tmp_path, {"include/g.h": header, "src/a.c": source})
    writes, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\n' + source), strict=False)
    assert findings == () and "hook" not in (writes.get("include/main/grp.h") or b"").decode()
    assert '#include "g.h"' not in writes.get("src/a.c", b"").decode()


def test_fold_spacing_and_parameter_names_are_no_conflict_with_a_landed_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(
        types,
        "load",
        lambda snap: {"function": {"fresh": {"signature": "int fresh(int* a, s32 b)", "evidence": "landed"}}},
    )
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": "int fresh(int *x, s32 y);\n"})
    _, findings = headers.fold(s, unit(), view('typedef int s32;\n# 1 "src/a.c"\nint fresh(int *x, s32 y);\n'))
    assert findings == ()


def test_fold_no_conflict_when_inferred_or_same(tmp_path, monkeypatch):
    _known(monkeypatch, "inferred")
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": "void fresh(int b);\n"})
    _, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\nvoid fresh(int b);\n'))
    assert findings == ()


@pytest.mark.parametrize(
    "definition, conflicts",
    [
        ("int f(void) { return 0; }", False),
        ("s32 f(void) { return 0; }", False),
        ("void f(void) { }", True),
        ("int *f(void) { return 0; }", True),
    ],
)
def test_fold_refuses_a_definition_whose_header_prototype_returns_another_type(
    tmp_path, no_types, monkeypatch, definition, conflicts
):
    monkeypatch.setattr(headers, "catalog", lambda snap: {"f": ("include/g.h", 7, "extern int f(void);")})
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": definition + "\n"})
    text = '# 1 "include/g.h"\ntypedef int s32;\n# 1 "src/a.c"\n' + definition + "\n"
    _, findings = headers.fold(s, unit(), view(text))
    assert [f.key for f in findings] == (["headers.conflict"] if conflicts else [])
    if conflicts:
        assert "include/g.h:7" in findings[0].reason and "declares 'int'" in findings[0].reason


def test_a_declaration_that_disagrees_with_the_landed_definition_is_a_conflict(tmp_path):
    files = {
        "src/f.c": "s32 f(s32 arg0, s32 *arg1) {\n    return 0;\n}\n",
        "src/g.c": "extern u32 f(u32 devAddr, u32 *data);\nvoid g(void) { f(0, 0); }\n",
        "src/h.c": "void h(void) { f(1, 0); }\n",
        "include/ok.h": "extern s32 f2(s32, s32 *);\n",
        "src/f2.c": "s32 f2(s32 a, s32 *b) { return 0; }\n",
    }
    agreed, findings = headers.disagreements(snap(tmp_path, files))
    assert [f.unit for f in findings] == ["f"] and "f" not in agreed
    found = findings[0]
    assert found.key == "types.conflict" and not found.blocking
    assert found.symptoms["definition"] == {"path": "src/f.c", "line": 1, "signature": "s32 f(s32 arg0, s32 *arg1)"}
    assert found.symptoms["consumers"] == 1
    assert list(found.missing) == [
        "f: src/f.c:1 s32 f(s32 arg0, s32 *arg1)",
        "f: src/g.c:1 extern u32 f(u32 devAddr, u32 *data);",
    ]


def _graph(files: dict[str, bytes], reads: list[str], digest: str):
    def peek(path):
        reads.append(path)
        return files.get(path)

    units = {p: SimpleNamespace(path=p) for p in files if p.endswith(".c")}
    return SimpleNamespace(peek=peek, digest=digest, overlays={}, layout=SimpleNamespace(units=units))


def test_consumers_read_nothing_without_a_header_edit(monkeypatch):
    reads: list[str] = []
    monkeypatch.setattr(headers, "sources", lambda _: [])
    assert headers.consumers(_graph({"src/a.c": b""}, reads, "no-edit"), []) == () and reads == []


def test_consumers_follow_includes_through_headers_and_name_includers_of_a_deleted_header(monkeypatch):
    files = {
        "src/a.c": b'#include "mid.h"\n',
        "include/mid.h": b'#include "top.h"\n',
        "include/top.h": b"",
        "src/b.c": b'#include "other.h"\n',
        "include/other.h": b"",
        "src/c.c": b'#include "gone.h"\n',
    }
    monkeypatch.setattr(headers, "sources", lambda _: ["include/mid.h", "include/top.h", "include/other.h"])
    snapshot = _graph(files, [], "reach")
    assert headers.consumers(snapshot, ["include/top.h"]) == ("src/a.c",)
    assert headers.consumers(snapshot, ["include/gone.h"]) == ("src/c.c",)


def test_consumers_read_each_file_once_however_many_units_include_the_header(monkeypatch):
    units = {f"src/u{i}.c": b'#include "shared.h"\n' for i in range(300)}
    files = {**units, "include/shared.h": b""}
    reads: list[str] = []
    monkeypatch.setattr(headers, "sources", lambda _: ["include/shared.h"])
    snapshot = _graph(files, reads, "many-units")
    assert sorted(headers.consumers(snapshot, ["include/shared.h"])) == sorted(units)
    assert len(reads) == len(files) and len(set(reads)) == len(files)  # one read per file, not per unit
    headers.consumers(snapshot, ["include/shared.h"])
    assert len(reads) == len(files)  # the graph is one fact of the snapshot: asking again reads nothing


def test_reaching_keeps_only_the_units_whose_text_names_a_changed_declaration_or_a_user_of_one(tmp_path, monkeypatch):
    before = snap(tmp_path, {"include/a.h": "int f(void);\n#define CALL_F f()\ntypedef int Kept;\n"})
    after = SimpleNamespace(
        peek=lambda p: {"include/a.h": b"int f(int x);\n#define CALL_F f()\ntypedef int Kept;\n", **units}.get(p),
        read=lambda p: {"include/a.h": b"int f(int x);\n#define CALL_F f()\ntypedef int Kept;\n"}[p],
        config=before.config,
    )
    units = {
        "src/f.c": b'#include "a.h"\nint g(void) { return f(1); }\n',
        "src/macro.c": b'#include "a.h"\nint h(void) { return CALL_F; }\n',
        "src/kept.c": b'#include "a.h"\nKept k;\n',
        "src/private.c": b'#include "a.h"\n#include "local.inc"\n',
    }
    monkeypatch.setattr(headers, "sources", lambda _: ["include/a.h"])
    parsed: list[int] = []
    real = headers.pool.map
    monkeypatch.setattr(
        headers.pool, "map", lambda c, n, f, items, k=None: parsed.append(len(items)) or real(c, n, f, items, k)
    )
    found = headers.reaching(before, after, ["include/a.h"], sorted(units))
    assert found == ("src/f.c", "src/macro.c", "src/private.c")  # a file outside include/ is never trusted unread
    assert parsed == [1]  # every header parsed in one pool batch
    same = SimpleNamespace(peek=before.peek, read=before.read, config=before.config)
    assert headers.reaching(before, same, ["include/a.h"], sorted(units)) == ()


def test_reaching_keeps_the_units_naming_a_declaration_of_a_header_a_changed_one_newly_includes(tmp_path, monkeypatch):
    before = snap(tmp_path, {"include/a.h": "int f(void);\n", "include/b.h": "extern int gTable[];\n"})
    grown = b'#include "b.h"\nint f(void);\n'  # the same declarations: only what it brings in is new
    after = SimpleNamespace(
        peek=lambda p: grown if p == "include/a.h" else units.get(p, before.peek(p)),
        read=lambda p: grown if p == "include/a.h" else before.read(p),
        config=before.config,
    )
    units = {
        "src/clash.c": b'#include "a.h"\nchar *gTable[2];\n',  # newly meets extern int gTable[]: it must compile
        "src/calls.c": b'#include "a.h"\nint g(void) { return f(); }\n',
    }
    monkeypatch.setattr(headers, "sources", lambda _: ["include/a.h", "include/b.h"])
    assert headers.reaching(before, after, ["include/a.h"], sorted(units)) == ("src/clash.c",)


def test_fold_keeps_a_declaration_in_the_source_when_its_type_header_includes_the_group_header(tmp_path, no_types):
    aggregate = '#include "main/grp.h"\ntypedef struct Late Late;\n'
    source = "typedef struct Late Late;\nvoid take(Late *);\nvoid fresh(int b);\nint f(void) { return 0; }\n"
    s = snap(tmp_path, {"include/g.h": HEADER, "include/all.h": aggregate, "include/main/grp.h": "", "src/a.c": source})
    text = '# 1 "include/all.h"\ntypedef struct Late Late;\n# 1 "src/a.c"\n' + source
    writes, findings = headers.fold(s, unit(), view(text))
    group, code = writes["include/main/grp.h"].decode(), writes["src/a.c"].decode()
    assert not findings
    assert "void fresh(int b);" in group and "take" not in group and "all.h" not in group  # no include cycle
    assert "void take(Late *);" in code and '#include "all.h"' in code


def test_fold_keeps_a_declaration_in_the_source_when_another_landed_source_spells_it_otherwise(tmp_path, no_types):
    source = "void same(int *);\nvoid other(char *);\nextern int shared;\nint f(void) { return 0; }\n"
    landed = "extern void same(int *p);\nextern void other(int *);\nextern long shared;\n"
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": source, "src/b.c": landed})
    writes, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\n' + source))
    group, code = writes["include/main/grp.h"].decode(), writes["src/a.c"].decode()
    assert not findings
    assert "void same(int *);" in group and "other" not in group and "shared" not in group
    assert "void other(char *);" in code and "extern int shared;" in code and "same" not in code


def test_fold_takes_no_second_header_that_contradicts_the_first(tmp_path, no_types):
    first, second = (
        "extern void one(int *);\nextern void both(int *);\n",
        "extern void two(int *);\nextern void both(void *);\n",
    )
    source = "extern void two(int *);\nextern void one(int *);\nextern void both(int *);\nint f(void) { return 0; }\n"
    s = snap(tmp_path, {"include/a.h": first, "include/b.h": second, "src/a.c": source})
    writes, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\n' + source))
    code = writes["src/a.c"].decode()
    assert not findings
    assert '#include "b.h"' in code and '#include "a.h"' not in code
    assert "extern void one(int *);" in code and "two" not in code and "both" not in code  # b.h declares both


@pytest.mark.parametrize("candidate, conflicts", [("extern const s32 gCount;", True), ("extern s32 gCount;", False)])
def test_fold_treats_const_as_a_difference_from_the_header(tmp_path, no_types, candidate, conflicts):
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": candidate + "\n"})
    _, findings = headers.fold(s, unit(), view('typedef int s32;\n# 1 "src/a.c"\n' + candidate + "\n"))
    assert [f.key for f in findings] == (["headers.conflict"] if conflicts else [])


@pytest.mark.parametrize(
    "members, conflicts", [("int a; int b;", False), ("int a; int pad4; int b;", False), ("int a; int b; int c;", True)]
)
def test_fold_refuses_a_struct_whose_header_defines_other_members(tmp_path, no_types, members, conflicts):
    source = "struct Box { " + members + " };\nint f(void) { return 0; }\n"
    header = "struct Box { int a; int b; };\n"
    s = snap(tmp_path, {"include/g.h": header, "src/a.c": source})
    _, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\n' + source))
    assert [f.key for f in findings] == (["headers.conflict"] if conflicts else [])


def _overlay(s, writes):
    for name, data in writes.items():
        (s.config.project.root / name).write_bytes(data)
    return s


def test_reconcile_keeps_one_declaration_in_the_owner_and_includes_it(tmp_path, no_types):
    s = snap(
        tmp_path,
        {"include/owner.h": "extern int D_1;\n", "include/w.h": '#include "x.h"\nextern int D_1;\nextern int D_2;\n'},
    )
    out, findings = headers.reconcile(s, s, {"include/w.h": s.read("include/w.h")})
    assert findings == [] and b"D_1" not in out["include/w.h"] and b"D_2" in out["include/w.h"]
    assert b'#include "owner.h"' in out["include/w.h"]


def test_reconcile_drops_the_copy_that_disagrees_with_the_type_map(tmp_path, monkeypatch):
    monkeypatch.setattr(
        types, "load", lambda s: {"function": {}, "global": {"D_1": {"declaration": "extern s32 D_1;"}}}
    )
    s = snap(tmp_path, {"include/a.h": "extern char D_1;\n", "include/b.h": "extern s32 D_1;\n"})
    out, findings = headers.reconcile(s, s, {p: s.read(p) for p in ("include/a.h", "include/b.h")})
    assert findings == [] and b"D_1" not in out["include/a.h"] and b"D_1" in out["include/b.h"]


def test_reconcile_judges_a_header_by_the_signature_an_earlier_entry_of_the_drain_landed(tmp_path, monkeypatch):
    monkeypatch.setattr(types, "load", lambda s: {"function": {"f": {"signature": "int f(void)"}}, "global": {}})
    s = snap(
        tmp_path,
        {"include/a.h": "int f(void);\n", "include/b.h": "s32 f(s32 a);\n", "src/f.c": "s32 f(s32 a) { return a; }\n"},
    )
    out, findings = headers.reconcile(s, s, {p: s.read(p) for p in ("include/a.h", "include/b.h")})
    assert findings == [] and b"f(" in out["include/b.h"] and b"f(" not in out["include/a.h"]


def test_reconcile_refuses_a_written_header_that_conflicts_with_an_unwritable_one(tmp_path, no_types):
    s = snap(tmp_path, {"include/own.h": '#include "w.h"\nextern int D_1;\n', "include/w.h": "extern char D_1;\n"})
    _, findings = headers.reconcile(s, s, {"include/w.h": s.read("include/w.h")})
    assert [f.key for f in findings] == ["headers.conflict"]
    assert "include/w.h:1" in findings[0].reason and "include/own.h:2" in findings[0].reason


def test_a_multi_line_macro_is_no_declaration_and_takes_no_include():
    text = (
        "#ifndef A_H\n#define A_H\n#define aSet(pkt, f) { Acmd *_a = (Acmd *)(pkt); \\\n"
        "    _a->w0 = _SHIFTL(f,16,8); _a->w1 = 0; }\n#endif\n"
    )
    assert headers._declared(text) == []
    assert headers.included(text, ["b.h"]).endswith('_a->w1 = 0; }\n#include "b.h"\n#endif\n')


@pytest.mark.parametrize(
    "candidate,project,compatible",
    [
        ("void * @(void *)", "int @(struct Record *)", True),
        ("void @(Unknown *)", "void @(struct Record *)", True),
        ("float @[]", "float @", False),
        ("float @", "float @[]", False),
        ("float @[2]", "float @[4]", False),
        ("float @[]", "float @[4]", True),
        ("float @", "int @", False),
        ("Unknown @", "int @", False),
        ("Unknown @", "Other @", False),
        ("void @(Unknown)", "void @(int)", False),
        ("void @(int, int)", "void @(int)", False),
    ],
)
def test_project_adoption_requires_known_matching_abi(candidate, project, compatible):
    assert headers._compatible(candidate, project) is compatible


@pytest.mark.parametrize(
    "fields,compatible", [("int a; int b;", True), ("int a;", False), ("float a; float b;", True), ("double a;", False)]
)
def test_by_value_typedefs_are_resolved_before_project_adoption(fields, compatible):
    ast = headers.parse("typedef struct { int a; int b; } Old; typedef struct { " + fields + " } New;", "types.h")
    typedefs = {n.name: n.type for n in ast.ext}
    assert headers._compatible("void @(Old)", "void @(New)", typedefs) is compatible


def test_loose_pointer_declaration_is_removed_in_favor_of_header(tmp_path, no_types):
    source = "extern void use(void *p);\nvoid f(void) { use(0); }\n"
    s = snap(tmp_path, {"include/g.h": "void use(struct Record *p);\n", "src/a.c": source})
    writes, findings = headers.fold(s, unit(), view('# 1 "src/a.c"\n' + source))
    assert not findings
    assert "extern void use" not in writes["src/a.c"].decode()
    assert '#include "g.h"' in writes["src/a.c"].decode()


def test_project_abi_resolves_forward_typedefs_through_visible_record_bodies():
    ast = headers.parse(
        "typedef struct A A; typedef struct B B; struct A { float x; int y; }; struct B { int words[2]; };", "types.h"
    )
    aliases = {n.name: n.type for n in ast.ext if isinstance(n, headers.c_ast.Typedef)}
    assert not headers._compatible("void @(A)", "void @(B)", aliases)
    assert headers._compatible("void @(A)", "void @(B)", aliases, headers._records(ast))


@pytest.mark.parametrize("spelling", ["signed int", "signed", "long int", "signed long int"])
def test_record_abi_recognizes_standard_integer_spellings(spelling):
    ast = headers.parse(
        "typedef " + spelling + " Word; struct A { Word a[10]; }; "
        "typedef struct A A; struct B { int words[10]; }; typedef struct B B;",
        "types.h",
    )
    aliases = {n.name: n.type for n in ast.ext if isinstance(n, headers.c_ast.Typedef)}
    assert headers._compatible("void @(A)", "void @(B)", aliases, headers._records(ast))


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("void @(void)", "void @()", True),
        ("void @()", "void @(void)", True),
        ("int @(void)", "void @()", False),
        ("void @(int)", "void @()", False),
        ("void @(float)", "void @()", False),
    ],
)
def test_unspecified_prototype_adoption_only_proves_no_arguments(left, right, expected):
    assert headers._compatible(left, right) is expected


def test_fold_writes_the_declaration_to_the_module_header_path_and_guards_it_by_that_path(tmp_path, no_types):
    s = snap(tmp_path, {"include/g.h": HEADER, "src/a.c": "void fresh(int b);\nint f(void) { return 0; }\n"})
    s.layout.groups["g"].header = "include/x/own.h"
    writes, _ = headers.fold(s, unit(), view('# 1 "src/a.c"\nvoid fresh(int b);\nint f(void) { return 0; }\n'))
    assert "#ifndef X_OWN_H" in writes["include/x/own.h"].decode()
    assert '#include "x/own.h"' in writes["src/a.c"].decode()
    assert "include/main/grp.h" not in writes


def test_rewrite_includes_resolves_under_include_and_beside_the_includer():
    text = '#include "a/old.h"\n#include <stdio.h>\n#include "same.h"\n'
    moves = {"include/a/old.h": "include/b/new.h", "src/x/same.h": "src/x/other.h"}
    assert headers.rewrite_includes(text, moves, "src/x/u.c") == (
        '#include "b/new.h"\n#include <stdio.h>\n#include "other.h"\n'
    )
    assert headers.rewrite_includes('#include "a/old.h"\nint x;\n', {"include/a/old.h": ""}, "src/u.c") == "int x;\n"


def test_merged_keeps_each_identical_statement_once_and_refuses_two_definitions():
    one = '#ifndef A\n#define A\n#include "t.h"\ntypedef int s32;\nvoid f(s32 a);\n#endif\n'
    two = '#ifndef B\n#define B\n#include "t.h"\ntypedef int s32;\nvoid g(s32 a);\n#endif\n'
    merged = headers._merged("include/m/h.h", [one, two])
    assert merged.count("typedef int s32;") == 1 and merged.count('#include "t.h"') == 1
    assert merged.startswith("#ifndef M_H_H\n#define M_H_H\n") and "void g(s32 a);" in merged
    clash = ["struct P { int a; };\n", "struct P { int b; };\n"]
    with pytest.raises(Refusal) as caught:
        headers._merged("include/m/h.h", clash)
    assert caught.value.findings[0].key == "move.declaration"


def test_merged_orders_definitions_before_their_uses_and_refuses_a_real_cycle():
    first = "typedef struct { Vec3 at; } Actor;\nvoid f(Actor *a);\n"
    second = "typedef struct { float x; } Vec3;\n#define N 4\nextern Vec3 table[N];\n"
    merged = headers._merged("include/m/h.h", [first, second])
    assert merged.index("} Vec3;") < merged.index("} Actor;")
    assert merged.index("#define N 4") < merged.index("extern Vec3 table")
    same = headers._merged("include/m/h.h", [second, first])
    assert same.index("} Vec3;") < same.index("} Actor;")
    pointers = headers._merged("include/m/h.h", ["struct A { struct B *b; };\n", "struct B { struct A *a; };\n"])
    assert pointers.index("struct A {") < pointers.index("struct B {")
    cycle = ["struct A { struct B b; };\n", "struct B { struct A a; };\n"]
    with pytest.raises(Refusal) as caught:
        headers._merged("include/m/h.h", cycle)
    assert caught.value.findings[0].key == "move.cycle"
    assert "struct A" in caught.value.findings[0].reason and "struct B" in caught.value.findings[0].reason


def test_owned_splits_the_statements_that_declare_the_names():
    text = "typedef int s32;\nvoid f(s32 a);\nextern s32 gCount;\n"
    taken, rest = headers._owned(text, {"f", "gCount"})
    assert taken == "void f(s32 a);\nextern s32 gCount;" and rest == "typedef int s32;\n"
    assert headers._empty(headers.wrapped("include/a.h", "")) and not headers._empty(
        headers.wrapped("include/a.h", "int x;")
    )


def proto_snap(tmp_path, files):
    snapshot = snap(tmp_path, files)
    snapshot.layout.units = {"src/def.c": SimpleNamespace(path="src/def.c", group="g")}
    return snapshot


PROTO_FILES = {
    "include/g.h": "typedef int s32;\n",
    "src/def.c": '#include "g.h"\nint f(s32 a) { return a; }\n',
    "src/use.c": '#include "g.h"\nextern s32 f(s32 *);\nvoid h(void) { f(0); }\n',
    "src/same.c": '#include "g.h"\nextern int f(s32 b);\n',
}


def test_prototypes_give_the_definition_signature_one_home(tmp_path, monkeypatch):
    monkeypatch.setattr(types, "load", lambda s: {"function": {}, "global": {}})
    items, writes, _ = headers.prototypes(proto_snap(tmp_path, PROTO_FILES))
    assert items == {"f": (True, frozenset({"src/use.c", "src/same.c", "include/main/grp.h"}))}
    assert "extern" not in writes["src/use.c"].decode().split("\n", 2)[1]  # the guess is gone, its include came
    assert '#include "main/grp.h"' in writes["src/use.c"].decode()
    assert "extern int f(s32 a);" in writes["include/main/grp.h"].decode()
    assert "src/def.c" not in writes


def test_prototypes_repeat_of_the_header_spelling_is_not_typed(tmp_path, monkeypatch):
    monkeypatch.setattr(types, "load", lambda s: {"function": {}, "global": {}})
    files = {**PROTO_FILES, "include/g.h": "typedef int s32;\nextern int f(s32 a);\n"}
    del files["src/use.c"]
    items, writes, _ = headers.prototypes(proto_snap(tmp_path, files))
    assert items["f"][0] is False
    assert "extern" not in writes["src/same.c"].decode()


def test_prototypes_leave_a_type_no_header_can_name_and_a_name_without_authority(tmp_path, monkeypatch):
    monkeypatch.setattr(types, "load", lambda s: {"function": {}, "global": {}})
    files = {
        "src/def.c": "struct Mine { int a; };\nint f(struct Mine *p) { return 0; }\n",
        "src/use.c": "extern int f(void *);\nextern int g(int);\n",
        "src/other.c": "extern long g(int);\n",
    }
    items, writes, (found,) = headers.prototypes(proto_snap(tmp_path, files))
    assert (items, writes) == ({}, {}) and found.key == "headers.prototype_conflict" and not found.blocking
    assert (
        "'extern int g(int);' in src/use.c" in found.reason and "'extern long g(int);' in src/other.c" in found.reason
    )


def test_prototypes_hoist_the_one_spelling_local_externs_agree_on_into_the_header_of_the_unit_holding_it(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(types, "load", lambda s: {"function": {}, "global": {}})
    files = {"src/a.c": "extern void k(int *);\n", "src/b.c": "extern void k(int *p);\nextern void k();\n"}
    snapshot = snap(tmp_path, files)
    snapshot.layout.units = {"asm/k.s": SimpleNamespace(path="asm/k.s", group="g", members=("k",))}
    items, writes, found = headers.prototypes(snapshot)
    assert found == [] and items["k"][0] is True  # the K&R spelling says nothing: the typed one is the prototype
    assert writes["include/main/grp.h"].decode().count("extern void k(int *") == 1
    assert all("extern" not in writes[p].decode() and "main/grp.h" in writes[p].decode() for p in files)


def test_prototypes_split_a_statement_that_declares_two_names(tmp_path, monkeypatch):
    monkeypatch.setattr(types, "load", lambda s: {"function": {}, "global": {}})
    files = {"src/def.c": "int f(int a) { return a; }\n", "src/use.c": "extern void f(char), k(int);\n"}
    _, writes, _ = headers.prototypes(proto_snap(tmp_path, files))
    assert "extern void k(int);" in writes["src/use.c"].decode() and "f(" not in writes["src/use.c"].decode()


def test_skip_leaves_a_name_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(types, "load", lambda s: {"function": {}, "global": {}})
    assert headers.prototypes(proto_snap(tmp_path, PROTO_FILES), {"f"}) == ({}, {}, [])


TWIN = '#ifndef UNBAKE_A_H\n#define UNBAKE_A_H\n#include "x.h"\nint a(void);\n#endif\n'
FOLDED = "/* twin */\n#ifndef UNBAKE_B_H\n#define UNBAKE_B_H\nint b(void);\n#ifdef Z\nint z;\n#endif\n#endif /* B */\n"


def test_a_folded_header_keeps_one_balanced_guard():
    merged = headers._merged("include/shared/a.h", [TWIN, FOLDED])
    assert merged.count("#ifndef") == 1 and merged.startswith("#ifndef SHARED_A_H")
    openers = merged.count("#if")
    assert openers == merged.count("#endif") == 2  # the guard and the Z block
    assert "UNBAKE_B_H" not in merged and "UNBAKE_A_H" not in merged


def test_folding_again_is_stable():
    merged = headers._merged("include/shared/a.h", [TWIN, FOLDED])
    assert headers._merged("include/shared/a.h", [merged]) == merged
    assert headers._merged("include/shared/a.h", [merged, TWIN]) == merged


def test_a_text_whose_last_endif_closes_something_else_is_left_whole():
    text = "#ifdef A\nint a;\n#endif\n#ifdef B\nint b;\n#endif\n"
    assert headers._unguarded(text) == text


def test_scanning_the_same_text_twice_pools_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        headers.pool,
        "map",
        lambda config, name, function, items, key=None: calls.append(len(items)) or [function(i) for i in items],
    )
    headers._SCANS.clear()
    snapshot = snap(tmp_path, {"include/g.h": HEADER})
    headers._scanned(snapshot, ["include/g.h"])
    headers._scanned(snapshot, ["include/g.h"])
    assert calls == [1]


@pytest.mark.parametrize(
    ("left", "right", "compatible"),
    [
        ("signed char @", "unsigned char @", False),
        ("short *@", "unsigned short *@", False),
        ("int @", "signed int @", True),
    ],
)
def test_adoption_keeps_the_signedness_of_an_object(left, right, compatible):
    assert headers._compatible(left, right) is compatible


def _serial(monkeypatch) -> None:
    monkeypatch.setattr(headers.pool, "map", lambda _c, _n, fn, items, _k: [fn(i) for i in items])


def _tree(files: dict[str, bytes], digest: str):
    snapshot = _graph(files, [], digest)
    snapshot.config = None
    snapshot.read = files.__getitem__
    snapshot.files = files
    return snapshot


def test_a_header_naming_a_dropped_declaration_is_dangling_though_no_unit_uses_it(monkeypatch):
    _serial(monkeypatch)
    types_h = b"typedef int T1;\ntypedef int T2;\n"
    users = {
        "src/a.c": b'#include "span.h"\n',
        "include/span.h": b'#include "types.h"\nstruct S { T1 unk0; };\n',
        "include/other.h": b'#include "types.h"\nstruct O { T2 unk0; };\n',
    }
    before = _tree({**users, "include/types.h": types_h}, "dangle-before")
    after = _tree({**users, "include/types.h": b"typedef int T2;\n"}, "dangle-after")
    monkeypatch.setattr(headers, "sources", lambda s: [p for p in s.files if p.endswith(".h")])
    assert headers.dangling(before, after, ["include/types.h"]) == ("include/span.h",)
    assert headers.consumers(after, list(headers.dangling(before, after, ["include/types.h"]))) == ("src/a.c",)
    assert headers.dangling(before, before, ["include/types.h"]) == ()
    moved = _tree(
        {**users, "include/types.h": b"typedef int T2;\n", "include/elsewhere.h": b"typedef int T1;\n"}, "dangle-moved"
    )
    assert headers.dangling(before, moved, ["include/types.h"]) == ()  # declared in another header: nothing dropped


def test_wrapped_guard_names_the_path_below_include_and_no_tool():
    text = headers.wrapped("include/game/main/actor.h", "int a;")
    assert text.startswith("#ifndef GAME_MAIN_ACTOR_H\n#define GAME_MAIN_ACTOR_H\n") and "UNBAKE" not in text


def test_an_old_tool_guard_is_stripped_so_a_rewrite_converts_it():
    old = "#ifndef UNBAKE_X_H\n#define UNBAKE_X_H\nint x;\n#endif\n"
    assert headers._unguarded(old).strip() == "int x;"
    assert "UNBAKE" not in headers._merged("include/x.h", [old]) and "#ifndef X_H" in headers._merged(
        "include/x.h", [old]
    )


def test_include_order_puts_the_own_header_first_then_sorts_and_dedupes():
    lines = ['#include "z.h"', '#include "own.h"', '#include "a.h"', '#include "z.h"']
    assert headers.include_order(lines, "own.h") == ['#include "own.h"', '#include "a.h"', '#include "z.h"']
    text = '#include "z.h"\n#include "own.h"\n\nint x;\n'
    assert headers.ordered(text, "own.h") == '#include "own.h"\n#include "z.h"\n\nint x;\n'
    scattered = '#include "z.h"\nint y;\n#include "a.h"\n'
    assert headers.ordered(scattered, "") == scattered


def test_a_many_to_one_rename_keeps_each_typedef_once_per_file():
    text = "typedef struct S A;\ntypedef struct S A;\n\nint x;\ntypedef struct S A;\ntypedef struct T B;\n"
    assert headers._dedupe(text) == "typedef struct S A;\n\nint x;\ntypedef struct T B;\n"
    assert headers._dedupe(text, {"typedef struct S A;"}) == "\nint x;\ntypedef struct T B;\n"
    assert headers._dedupe("typedef struct S { int a; } A;\ntypedef struct S { int a; } A;\n").count("typedef") == 2


def test_inherited_typedefs_come_from_quoted_includes_through_others():
    files = {"include/a.h": '#include "b.h"\ntypedef int A;\n', "include/b.h": "typedef int B;\n"}
    found = headers._inherited(files.get, "src/u.c", '#include "a.h"\ntypedef int B;\n')
    assert found == {"typedef int A;", "typedef int B;"}


def test_respell_dedupes_the_typedefs_a_many_to_one_rename_leaves_in_a_file():
    texts = {
        "src/u.c": b'#include "h.h"\ntypedef struct S Old1;\ntypedef struct S Old2;\ntypedef struct S Own;\nint f;\n',
        "include/h.h": b"typedef struct S Shared;\n",
    }
    snapshot = SimpleNamespace(peek=texts.get, layout=SimpleNamespace(units={}), config=SimpleNamespace())
    with patch.object(
        headers, "sources", lambda snap, place="include", suffix=".h": [p for p in texts if p.endswith(suffix)]
    ):
        written = headers.respell(snapshot, {}, {}, [], {"Old1": "Shared", "Old2": "Shared"})
    assert written["src/u.c"].decode() == '#include "h.h"\ntypedef struct S Own;\nint f;\n'


def test_a_typedef_repeated_in_another_conditional_branch_stays():
    branches = "#if V\ntypedef struct S A;\n#else\ntypedef struct S A;\n#endif\n"
    assert headers._dedupe(branches) == branches
    assert headers._dedupe("typedef struct S A;\ntypedef struct S A;\n") == "typedef struct S A;\n"


@pytest.mark.parametrize(("local", "conflicts"), [("int f();", False), ("int f(long q);", True)])
def test_fold_reads_a_knr_declaration_as_the_prototype_a_header_gives_and_keeps_real_conflicts(
    tmp_path, no_types, local, conflicts
):
    prototype = "extern void f(int *p, unsigned char c);"
    s = snap(tmp_path, {"include/g.h": prototype + "\n", "src/a.c": local + "\n"})
    text = f'# 1 "include/g.h"\n{prototype}\n# 1 "src/a.c"\n{local}\n'
    _, findings = headers.fold(s, unit(), view(text))
    assert [f.key for f in findings] == (["headers.conflict"] if conflicts else [])


def test_relocate_follows_a_chain_of_header_moves_to_its_last_header():
    texts = {f"include/{n}.h": f"#ifndef {n.upper()}\n#define {n.upper()}\nint {n}(void);\n#endif\n" for n in "abcd"}
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    events = [("file", "include/a.h", "include/b.h", frozenset()), ("file", "include/b.h", "include/c.h", frozenset())]
    events.append(("file", "include/c.h", "include/d.h", frozenset()))
    _, moved, _ = headers.relocate(snapshot, events)
    assert moved == {f"include/{n}.h": "include/d.h" for n in "abc"}


def test_a_guard_does_not_swallow_code_through_a_later_comment():
    text = "/* license */\n#ifndef G\n#define G\n#if X\nint a;\n#endif\nint b;\n/* tail */\n#endif /* G */\n"
    assert "#ifndef G" not in headers._unguarded(text) and headers._unguarded(text).count("#endif") == 1


def test_merged_keeps_outside_includes_in_place_drops_its_own_and_stays_balanced():
    one = '/* c */\n#ifndef G\n#define G\n#if X\nint a;\n#endif\n#include "a.h"\nint b;\n/* tail */\n#endif /* G */\n'
    two = '#ifndef H\n#define H\n#include "types.h"\n#include "shared/c.h"\nint c;\n#endif\n'
    merged = headers._merged("include/types.h", [one, two])
    lines = merged.splitlines()
    assert lines[:2] == ["#ifndef TYPES_H", "#define TYPES_H"]
    assert merged.index("int a;") < merged.index('#include "a.h"') < merged.index("int b;")
    assert merged.index("int b;") < merged.index('#include "shared/c.h"') < merged.index("int c;")
    assert '#include "types.h"' not in merged and "#ifndef G" not in merged and "/* G */" not in merged
    depth = [1 if re.match(r"\s*#\s*if", s) else -1 for s in lines if re.match(r"\s*#\s*(if|endif)", s)]
    assert sum(depth) == 0


def _merge_world():
    texts = {
        "include/a.h": "#ifndef A_H\n#define A_H\ntypedef int s32;\n#endif\n",
        "include/b.h": '#ifndef B_H\n#define B_H\n#include "a.h"\nextern s32 gB;\n#endif\n',
        "include/c.h": '#ifndef C_H\n#define C_H\n#include "d.h"\nint c;\n#endif\n',
        "include/d.h": '#ifndef D_H\n#define D_H\n#include "a.h"\nint d;\n#endif\n',
    }
    return SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None), texts


def test_a_merged_header_defines_a_type_before_the_merged_file_that_uses_it():
    snapshot, _ = _merge_world()
    out, moved, _ = headers.relocate(snapshot, [("file", "include/b.h", "include/a.h", frozenset())])
    merged = out["include/a.h"]
    assert merged.index("typedef int s32;") < merged.index("extern s32 gB;")
    assert '#include "a.h"' not in merged and moved == {"include/b.h": "include/a.h"}


def test_an_outside_include_that_reaches_back_goes_after_the_body_and_others_stay_in_place():
    snapshot, texts = _merge_world()
    texts["include/c.h"] = '#ifndef C_H\n#define C_H\n#include "d.h"\n#include "e.h"\nint c;\n#endif\n'
    texts["include/e.h"] = "#ifndef E_H\n#define E_H\nint e;\n#endif\n"
    out, _, _ = headers.relocate(snapshot, [("file", "include/c.h", "include/a.h", frozenset())])
    merged = out["include/a.h"]
    assert merged.index('#include "e.h"') < merged.index("int c;") < merged.index('#include "d.h"')
    assert merged.index("typedef int s32;") < merged.index('#include "d.h"') < merged.index("#endif")
    assert merged.count('#include "d.h"') == 1


def test_a_file_move_spells_the_includes_of_the_merged_header_from_include():
    texts = {
        "include/shared/a.h": '#ifndef A\n#define A\n#include "b.h"\nint a;\n#endif\n',
        "include/shared/b.h": "int b;\n",
        "include/types.h": "int t;\n",
    }
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    out, _, _ = headers.relocate(snapshot, [("file", "include/shared/a.h", "include/game/x.h", frozenset())])
    assert '#include "shared/b.h"' in out["include/game/x.h"] and '#include "b.h"' not in out["include/game/x.h"]


def test_many_merges_walk_each_outside_include_once():
    count = 60
    texts = {
        "include/target.h": "#ifndef T_H\n#define T_H\nint t;\n#endif\n",
        "include/common.h": '#ifndef C_H\n#define C_H\n#include "deep.h"\nint c;\n#endif\n',
        "include/deep.h": "#ifndef D_H\n#define D_H\nint d;\n#endif\n",
    }
    for i in range(count):
        texts[f"include/m{i}.h"] = f'#ifndef M{i}_H\n#define M{i}_H\n#include "common.h"\nint m{i};\n#endif\n'
    reads = []

    def peek(path):
        reads.append(path)
        return texts[path].encode() if path in texts else None

    events = [("file", f"include/m{i}.h", "include/target.h", frozenset()) for i in range(count)]
    out, moved, _ = headers.relocate(SimpleNamespace(peek=peek), events)
    assert len(moved) == count and out["include/target.h"].count('#include "common.h"') == 1
    assert reads.count("include/common.h") <= 3  # one walk, not one per merge


def test_a_parse_error_names_the_first_undeclared_type_of_its_run_and_the_includes_that_reached_it() -> None:
    effort.forget("declarations", "words")
    text = (
        '# 1 "src/a.c"\n# 1 "include/a.h" 1\ntypedef int (*H)(void *);\nstruct P { H h; };\n'
        '# 1 "include/b.h" 1\nextern float f(Actor *, int);\n# 3 "include/a.h" 2\n'
    )
    with pytest.raises(headers.Refusal) as error:
        headers.parse(text, "src/a.c")
    (found,) = error.value.findings  # H is a function pointer typedef: declared, the parameter type Actor is not
    assert "view line 6, from include/b.h:1: type Actor is not declared before its use" in found.reason
    assert found.reason.endswith("included through src/a.c > include/a.h)")
    with pytest.raises(headers.Refusal) as error:  # a record spelled on one line: its field type
        headers.parse('# 1 "include/d.h"\nstruct M { unsigned char pad[4]; s32 value; };\n', "src/d.c")
    assert "type s32 is not declared before its use" in error.value.findings[0].reason


def test_a_parse_error_names_the_view_line_and_the_undeclared_type() -> None:
    text = '# 1 "include/a.h"\nstruct Actor {\n    Vec3 pos;\n};\n# 5 "include/b.h"\ntypedef struct Vec3 Vec3;\n'
    with pytest.raises(headers.Refusal) as error:
        headers.parse(text, "src/a.c")
    (found,) = error.value.findings
    assert found.key == "headers.parse" and found.line == 2
    assert "view line 3, from include/a.h:2" in found.reason
    assert "type Vec3 is not declared before its use" in found.reason


def test_split_atoms_keeps_a_definition_whose_signature_names_a_struct():
    text = "struct R { int x; };\ns32 f(struct R *r) { return r->x; }\nvoid g(void) {}\n"
    _, blocks = headers.split_atoms(text, ["f", "g"])
    assert blocks["f"].endswith("s32 f(struct R *r) { return r->x; }") and blocks["g"] == "void g(void) {}"
    assert any(b == "struct R { int x; };" for k, b in blocks.items() if k.startswith("#"))


def _expanded(texts, path, guards=None):
    """The text a TU including `path` reads: quoted includes inlined once per guard, directives dropped."""
    guards = set() if guards is None else guards
    out = []
    for line in texts[path].splitlines():
        if (inc := re.match(r'\s*#\s*include\s*"([^"]+)"', line)) and f"include/{inc[1]}" not in guards:
            guards.add(f"include/{inc[1]}")
            out.append(_expanded(texts, f"include/{inc[1]}", guards))
        elif not line.lstrip().startswith("#"):
            out.append(line)
    return "\n".join(out)


def test_a_back_included_header_gives_the_types_the_merged_body_uses_to_the_merged_header():
    texts = {
        "include/a.h": "#ifndef A_H\n#define A_H\ntypedef int s32;\n#endif\n",
        "include/a2.h": '#ifndef A2_H\n#define A2_H\n#include "b.h"\nextern T g;\n#endif\n',
        "include/b.h": '#ifndef B_H\n#define B_H\n#include "a.h"\ntypedef struct T T;\nstruct T { int x; };\n'
        "extern T h;\n#endif\n",
    }
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    out, _, _ = headers.relocate(snapshot, [("file", "include/a2.h", "include/a.h", frozenset())])
    merged = out["include/a.h"]
    assert merged.index("typedef struct T T;") < merged.index("extern T g;") and "struct T { int x; };" in merged
    assert merged.index("extern T g;") < merged.index('#include "b.h"')
    assert "struct T" not in out["include/b.h"] and '#include "a.h"' in out["include/b.h"]
    assert "extern T h;" in out["include/b.h"]
    world = {**texts, **out, "include/tu.c": '#include "a.h"\n'}
    headers.parse(_expanded(world, "include/tu.c"), "tu.c")  # T is defined before every use


def test_a_back_included_type_that_needs_a_macro_of_its_header_refuses_as_a_cycle():
    texts = {
        "include/a.h": "#ifndef A_H\n#define A_H\nint a;\n#endif\n",
        "include/a2.h": '#ifndef A2_H\n#define A2_H\n#include "b.h"\nextern T g;\n#endif\n',
        "include/b.h": '#ifndef B_H\n#define B_H\n#include "a.h"\n#define N 4\n'
        "typedef struct { int x[N]; } T;\n#endif\n",
    }
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    with pytest.raises(Refusal) as caught:
        headers.relocate(snapshot, [("file", "include/a2.h", "include/a.h", frozenset())])
    assert caught.value.findings[0].key == "move.cycle" and "N" in caught.value.findings[0].reason


def test_headers_moved_into_a_new_header_whose_merge_refuses_keep_their_text_and_no_include_points_at_it():
    texts = {
        "include/a.h": "#ifndef A_H\n#define A_H\nextern int gShared;\nextern int gA;\n#endif\n",
        "include/b.h": "#ifndef B_H\n#define B_H\nextern long gShared;\nextern int gB;\n#endif\n",
        "include/c.h": "#ifndef C_H\n#define C_H\nextern int gC;\n#endif\n",
    }
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    events = [("file", f"include/{h}.h", "include/game/common.h", frozenset()) for h in "ab"]
    events.append(("file", "include/c.h", "include/game/other.h", frozenset()))  # a merge that is made
    failed: list = []
    out, moved, _ = headers.relocate(snapshot, events, failed)
    assert [(n, [f.key for f in fs]) for _, n, fs in failed] == [("include/game/common.h", ["move.declaration"])]
    assert out.get("include/game/common.h") is None  # never made
    assert moved == {"include/c.h": "include/game/other.h"}  # no include is respelled to the header never made
    assert out["include/a.h"] == texts["include/a.h"] and out["include/b.h"] == texts["include/b.h"]
    assert "gC" in out["include/game/other.h"] and out["include/c.h"] is None


def test_a_back_included_header_gives_the_merged_body_the_types_it_uses_whatever_its_last_type_is():
    outside = "struct Rec;\ntypedef struct Rec Rec;\nextern int gKeep;\ntypedef struct Other { int z; } Other;\n"
    body_words = {"extern", "f32", "use", "Rec", "arg0"}  # the body declares `extern f32 use(Rec *arg0);`
    pulled, rest = headers._pulled("include/sdk/lib.h", body_words, "include/out.h", outside)
    assert pulled == ["struct Rec;", "typedef struct Rec Rec;"]  # not the last type's family (Other)
    assert "Rec" not in rest and "Other" in rest and "gKeep" in rest


def test_reordering_a_merged_body_never_moves_a_statement_above_the_includes_before_it():
    body = (
        '#include "types.h"\nextern Rec *useRec(void);\n\n'  # waits for Rec, defined further down
        "extern s32 plain(s32 a);\n\ntypedef struct Rec { s32 x; } Rec;\n"
    )
    out = headers._define_before_use("include/game/m.h", body)
    assert out.index('#include "types.h"') < out.index("plain") < out.index("typedef struct Rec")
    assert out.index("typedef struct Rec") < out.index("useRec")


def test_merged_keeps_one_copy_of_a_type_spelled_two_ways_and_refuses_two_shapes():
    one = "typedef struct { unsigned int w0; unsigned int w1; } Awords;\n"
    two = "typedef struct {\n    unsigned  int w0; /* first */\n    unsigned int w1 ;\n} Awords;\n"
    assert headers._merged("include/m/h.h", [one, two]).count("Awords;") == 1
    three = "typedef struct { unsigned int w0; } Awords;\n"
    with pytest.raises(Refusal) as caught:
        headers._merged("include/m/h.h", [one, three])
    found = caught.value.findings[0]
    assert found.key == "move.declaration" and "w1" in found.reason and found.reason.count("} Awords;") == 2


def test_merged_externs_that_disagree_keep_the_type_map_declaration_or_refuse():
    texts = ["typedef int s32;\nextern s32 X;\n", "typedef int s32;\nextern s32 X[5];\n"]
    merged = headers._merged("include/m/h.h", texts, authority=lambda name: "extern s32 X[5];")
    assert "extern s32 X[5];" in merged and "extern s32 X;" not in merged
    swapped = headers._merged("include/m/h.h", texts, authority=lambda name: "extern int X[5];")
    assert "X[5];" in swapped and "extern s32 X;" not in swapped
    with pytest.raises(Refusal) as caught:
        headers._merged("include/m/h.h", texts)
    assert caught.value.findings[0].key == "move.declaration" and "X" in caught.value.findings[0].reason


def test_probe_compiles_each_header_alone_per_version_and_fails_on_one_error(monkeypatch, tmp_path):
    from unbake import adapters, layout, process, recipes

    unit = SimpleNamespace(path="src/u.c", kind="c", options={})
    texts = {"include/good.h": b"int g;\n", "include/bad.h": b"Vec3 v;\n"}
    config = SimpleNamespace(project=SimpleNamespace(versions=("a", "b"), library_dirs=(), root=tmp_path))
    snapshot = SimpleNamespace(config=config, layout=SimpleNamespace(units={"src/u.c": unit}))
    runs = []

    def run(name, argv, cwd, tmp=None):
        text = (cwd / "probe.i").read_text()
        runs.append(argv)
        bad = "Vec3" in text
        return SimpleNamespace(
            exit=int(bad), signal=None, stderr=b"probe.i:1: parse error before `v'\n" if bad else b""
        )

    monkeypatch.setattr(layout, "capture", lambda config: snapshot)
    monkeypatch.setattr(layout, "overlay", lambda s, writes: (texts.update(writes), s)[1])
    monkeypatch.setattr(headers, "sources", lambda s, *a: sorted(p for p in texts if p.endswith(".h")))
    monkeypatch.setattr(headers, "replace", lambda u, path: SimpleNamespace(**{**vars(u), "path": path}))
    monkeypatch.setattr(recipes, "resolve", lambda *a: SimpleNamespace(toolchain="gcc"))
    monkeypatch.setattr(
        headers._view,
        "get",
        lambda s, u, v, r, lines: SimpleNamespace(
            text=texts[f"include/{texts[u.path].decode().split(chr(34))[1]}"].decode()
        ),
    )
    monkeypatch.setattr(adapters, "chain_steps", lambda *a: [SimpleNamespace(argv=("cc1", "{in}", "-o", "{out}"))])
    monkeypatch.setattr(process, "run", run)
    monkeypatch.setattr(process, "scratch", lambda root: tmp_path)
    monkeypatch.setattr(headers.pool, "map", lambda config, name, fn, items, key=None: [fn(i) for i in items])
    result = headers.probe_command(config, {"probe": True, "versions": "all"})
    rows = {(r["header"], r["version"]): r["errors"] for r in result["headers"]}
    assert rows == {
        ("include/bad.h", "a"): 1,
        ("include/bad.h", "b"): 1,
        ("include/good.h", "a"): 0,
        ("include/good.h", "b"): 0,
    }
    assert result["failing"] == 2 and len(runs) == 4 and runs[0][-1] == "/dev/null"
    assert [r["findings"][0]["key"] for r in result["refused"]] == ["headers.probe"] * 2


def test_many_events_into_one_target_merge_it_once(monkeypatch):
    count = 40
    texts = {"include/target.h": "#ifndef T_H\n#define T_H\ntypedef int s32;\n#endif\n"}
    for i in range(count):
        texts[f"include/m{i}.h"] = f"#ifndef M{i}_H\n#define M{i}_H\nextern s32 m{i};\n#endif\n"
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    calls = {"merge": 0, "order": 0}
    merge, order = headers._merge, headers._define_before_use
    monkeypatch.setattr(headers, "_merge", lambda *a: (calls.__setitem__("merge", calls["merge"] + 1), merge(*a))[1])
    monkeypatch.setattr(
        headers, "_define_before_use", lambda *a: (calls.__setitem__("order", calls["order"] + 1), order(*a))[1]
    )
    events = [("file", f"include/m{i}.h", "include/target.h", frozenset()) for i in range(count)]
    out, moved, _ = headers.relocate(snapshot, events)
    assert calls == {"merge": 1, "order": 1}
    assert all(f"extern s32 m{i};" in out["include/target.h"] for i in range(count)) and len(moved) == count


def test_a_brace_that_closes_nothing_is_a_parse_refusal():
    with pytest.raises(Refusal) as caught:
        headers.parse("int f(void) { return 0; } }\n", "u.c")
    assert caught.value.findings[0].key == "headers.parse" and caught.value.findings[0].reason


def test_fold_shapes_every_comparison_in_one_parse(tmp_path, no_types):
    names = [f"gV{i}" for i in range(6)]
    header = "typedef int s32;\n" + "".join(f"extern s32 {n};\n" for n in names)
    source = "".join(f"extern int {n};\n" for n in names) + "int f(void) { return 0; }\n"
    s = snap(tmp_path, {"include/g.h": header, "src/a.c": source})
    text = '# 1 "include/g.h"\n' + header + '# 1 "src/a.c"\n' + source
    before = headers._Shapes.parses
    _, findings = headers.fold(s, unit(), view(text))
    assert not findings
    assert headers._Shapes.parses - before == 1  # the typedef prefix is parsed once, not twice per candidate


def test_primed_shapes_equal_the_shapes_of_one_parse_each():
    ast = headers.parse("typedef struct { int a; int b; } Pair; typedef unsigned short u16;", "types.h")
    typedefs = {n.name: n.type for n in ast.ext}
    pairs = [("void @(Pair)", "void @(Opaque)"), ("u16 @[4]", "struct R { int x; } @"), ("int @", "u16 @")]
    primed, alone = headers._Shapes(typedefs, None), headers._Shapes(typedefs, None)
    primed.prime(pairs)
    for left, right in pairs:
        names = frozenset(headers._opaque(left + "," + right, typedefs))
        for t in (left, right):
            assert primed.known[t, names] == alone.shape(t, names)


def test_folds_of_one_snapshot_scan_the_landed_sources_once(tmp_path, monkeypatch, no_types):
    files = {f"src/{c}.c": f"extern void {c}_fn(int *);\nint {c}(void) {{ return 0; }}\n" for c in "abcd"}
    s = snap(tmp_path, {"include/g.h": HEADER, **files})
    calls = _count_scans(monkeypatch)
    for c in "abcd":
        u = SimpleNamespace(path=f"src/{c}.c", group="g")
        headers.fold(s, u, view(f'# 1 "src/{c}.c"\n' + files[f"src/{c}.c"]))
    scanned = [p for paths in calls for p in paths if p.endswith(".c")]
    # one index of spellings and one of landed heads, not one per fold
    assert sorted(scanned) == sorted([*files, *files])
    assert types.authority(s) is types.authority(s)


def _unit_text(own: str) -> str:
    shared = '# 1 "include/common.h" 1\ntypedef int s32;\ntypedef struct Pair { s32 a; } Pair;\nextern Pair gPair;\n'
    return shared + '# 3 "src/a.c" 2\n' + own


def test_a_header_text_is_parsed_once_for_every_unit_that_includes_it():
    effort.forget("declarations", "words")
    before = headers._Index.parses
    asts = [headers.parse(_unit_text(f"s32 f{i}(Pair *p) {{ return p->a; }}\n"), "src/a.c") for i in range(4)]
    assert headers._Index.parses - before == 1 + 4  # the shared header once, each unit's own text once
    whole = [CParser().parse(headers.normal(_unit_text("s32 f0(Pair *p) { return p->a; }\n")), filename="src/a.c")]
    generator = c_generator.CGenerator()
    assert generator.visit(asts[0]) == generator.visit(whole[0])
    assert [n.coord.file for n in asts[0].ext] == [n.coord.file for n in whole[0].ext]
    assert [n.coord.line for n in asts[0].ext] == [n.coord.line for n in whole[0].ext]


def test_a_run_parses_again_when_a_name_it_uses_changes_from_type_to_object():
    effort.forget("declarations", "words")
    tail = '# 9 "src/a.c" 2\nint g(void) { T * x; return 0; }\n'
    typed = headers.parse('# 1 "include/t.h" 1\ntypedef int T;\n' + tail, "src/a.c")
    plain = headers.parse('# 1 "include/t.h" 1\nint T;\n' + tail, "src/a.c")
    generator = c_generator.CGenerator()
    assert "T *x;" in generator.visit(typed) and "T * x;" in generator.visit(plain)


def test_a_run_that_ends_inside_a_statement_is_parsed_with_the_runs_after_it():
    effort.forget("declarations", "words")
    text = '# 1 "src/a.c"\nint f(void) {\n# 1 "include/inner.h" 1\nint x;\n# 3 "src/a.c" 2\nreturn x; }\n'
    before = headers._Index.parses
    ast = headers.parse(text, "src/a.c")
    assert [type(n).__name__ for n in ast.ext] == ["FuncDef"]
    assert headers._Index.parses - before == 3  # the run, then joined with each next run until it parses


def test_a_unit_that_does_not_parse_refuses_from_its_failing_run_alone():
    effort.forget("declarations", "words")
    head = '# 1 "include/big.h" 1\n' + "".join(f"extern int gBig{i};\n" for i in range(200))
    text = '# 1 "src/a.c"\n' + head + '# 2 "src/a.c" 2\nint f(void) { return 1 +; }\n'
    before = headers._Index.chars
    with pytest.raises(Refusal) as refused:
        headers.parse(text, "src/a.c")
    with pytest.raises(headers.ParseError) as whole:
        CParser().parse(headers.normal(text), filename="src/a.c")
    assert refused.value.findings == headers.parse_refusal(whole.value, "src/a.c", text).findings  # the same refusal
    normalised = headers.normal(text)
    assert headers._Index.chars - before == len(normalised)  # each run once, never the whole unit again


def test_a_same_file_line_marker_inside_a_body_stays_in_its_run(monkeypatch):
    effort.forget("declarations", "words")
    text = '# 1 "src/a.c"\nint f(void) {\n    int x = 1;\n# 9 "src/a.c"\n    return x;\n}\n'
    before = headers._Index.parses
    assert len(headers._regions(headers.normal(text), "src/a.c")) == 1
    assert headers._Index.parses - before == 1
    assert [type(n).__name__ for n in headers.parse(text, "src/a.c").ext] == ["FuncDef"]


def test_a_header_run_is_one_index_entry_whatever_plain_names_came_before_it():
    effort.forget("declarations", "words")
    shared = '# 1 "include/b.h" 1\nextern int gFoo;\ntypedef int s32;\nextern s32 gBar;\n'
    before = headers._Index.parses
    for first in ("int gFoo;\nint other;\n", "int unrelated;\n"):  # gFoo declared plain before b.h, or never
        text = f'# 1 "src/a.c"\n# 1 "include/a.h" 1\n{first}' + shared + '# 3 "src/a.c" 2\nint f(void);\n'
        headers.parse(text, "src/a.c")
    assert headers._Index.parses - before == 1 + 2 + 1 + 1  # the unit's head, each a.h text, b.h ONCE, its own text


def test_views_made_in_other_overlay_directories_share_one_index_entry_per_header_text():
    effort.forget("declarations", "words")
    before = headers._Index.parses
    for key in ("0123456789abcdef", "fedcba9876543210"):  # each view's overlay directory is named by its own key
        header = f'# 1 "build/views/{key}/include/main/grp.h" 1\ntypedef int s32;\nextern s32 gOne;\n'
        ast = headers.parse(f'# 1 "src/a.c"\n{header}# 2 "src/a.c" 2\ns32 f(void);\n', "src/a.c")
        assert headers._path(snap_root(), ast.ext[0].coord) == "include/main/grp.h"
    assert headers._Index.parses - before == 1 + 1 + 1  # the unit's head, grp.h ONCE, its own text


def snap_root():
    return SimpleNamespace(config=SimpleNamespace(project=SimpleNamespace(root=Path("/nowhere"))))


_CASES = {  # typedef chains, records, K&R against a prototype, and conflicts
    "chain": ("typedef int s32;\ntypedef s32 Word;\ntypedef Word Cell;\nextern Cell gCell;\n", "extern s32 gCell;\n"),
    "record": (
        "typedef struct Pair { int a; short b; } Pair;\nvoid usePair(Pair *p);\n",
        "void usePair(struct Pair *p);\n",
    ),
    "k&r": ("typedef int s32;\ns32 kr(s32 a, char *b);\n", "int kr();\nvoid moved(s32 x);\n"),
    "conflict": (
        "typedef unsigned short u16;\nextern u16 gWide;\nint clash(int a);\n",
        "extern int gWide;\nlong clash(int a);\n",
    ),
}


def _case(tmp_path, name, filler):
    head, own = _CASES[name]
    head += "".join(f"extern int gFill{i};\nvoid fill{i}(int a);\n" for i in range(filler))
    source = '#include "g.h"\n' + own + "int f(void) { return 0; }\n"
    s = snap(tmp_path / f"{name}{filler}", {"include/g.h": head, "src/a.c": source})
    text = '# 1 "src/a.c"\n# 1 "include/g.h" 1\n' + head + '# 2 "src/a.c" 2\n' + own + "int f(void) { return 0; }\n"
    return s, text


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_fold_over_the_index_writes_and_finds_what_a_whole_unit_parse_does(tmp_path, no_types, monkeypatch, name):
    s, text = _case(tmp_path, name, 30)
    effort.forget("declarations", "words")
    indexed = headers.fold(s, unit(), view(text))

    def whole(text, path, config=None):  # the old path: one parse of the whole unit, cut by the file of its nodes
        runs: list[list] = []
        for node in CParser().parse(text, filename=path).ext:
            if not runs or runs[-1][0].coord.file != node.coord.file:
                runs.append([])
            runs[-1].append(node)
        return [headers._entry(nodes, {}, headers._words(text)) for nodes in runs]

    monkeypatch.setattr(headers, "_regions", whole)
    monkeypatch.setattr(
        headers._Shapes, "_body", lambda self, texts: "\n".join(self._typedef(n)[0] for n in self.typedefs)
    )
    assert headers.fold(s, unit(), view(text)) == indexed  # the same writes and findings, byte for byte
    assert indexed[1] if name == "conflict" else not indexed[1]


def test_a_folds_parse_work_is_its_own_text_and_the_named_prefix_flat_as_headers_grow(tmp_path, no_types):
    work = []
    for filler in (10, 99):  # the header grows ninefold (a later wave), the unit's own text keeps its size
        s, text = _case(tmp_path, "record", filler)
        headers.fold(s, unit(), view(text))  # the index holds the header text from here on
        own = text.replace("int f(void)", f"int g{filler}(void)")  # a unit text the index has not seen
        before = headers._Index.chars
        headers.fold(s, unit(), view(own))
        work.append(headers._Index.chars - before)
    assert (
        work[0] == work[1] and work[0] < len(text) // 4
    )  # the unit's text and the typedefs it names, never the header


def test_changed_names_the_declarations_a_write_changes_and_none_when_its_includes_change():
    head = b'#include "a.h"\nint f1(void);\n'
    assert headers.changed(head, head + b"void fresh(int a);\n", "include/g.h") == {"fresh"}
    assert headers.changed(head, head.replace(b"int f1", b"long f1"), "include/g.h") == {"f1"}
    assert headers.changed(head, head + b'#include "b.h"\n', "include/g.h") is None  # what it makes visible changed
    assert headers.changed(head, head, "include/g.h") == frozenset()


def test_a_fold_consults_the_names_it_spells_and_looks_up_never_the_rest_of_its_headers(tmp_path, no_types):
    s, text = _case(tmp_path, "record", 30)
    headers.consulted()
    headers.fold(s, unit(), view(text))
    names, paths = headers.consulted()
    assert {"usePair", "Pair", "struct Pair", "f"} <= names and "src/a.c" in paths
    assert not names & {f"fill{i}" for i in range(30)}  # the header's other declarations were never read
    assert headers.consulted() is None  # taken once: a later caller without a fold of its own learns nothing


def test_a_shared_header_takes_no_header_that_includes_another_and_a_pending_one_is_read_without_it():
    texts = {
        "include/types.h": "#ifndef TYPES_H\n#define TYPES_H\ntypedef int s32;\n#endif\n",
        "include/rec.h": '#ifndef REC_H\n#define REC_H\n#include "types.h"\ntypedef struct R { s32 x; } R;\n#endif\n',
        "include/wide.h": '#ifndef WIDE_H\n#define WIDE_H\n#include "p.h"\nextern s32 gWide;\n#endif\n',
        "include/p.h": '#ifndef P_H\n#define P_H\n#include "types.h"\nextern s32 gP;\n#endif\n',
        "include/m.h": "#ifndef M_H\n#define M_H\nint q;\nint m;\n#endif\n",
    }
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    events = [
        ("file", "include/p.h", "include/m.h", frozenset()),
        ("file", "include/wide.h", "include/types.h", frozenset()),  # includes p.h: not a base header's text
        ("file", "include/rec.h", "include/types.h", frozenset()),
        ("names", "include/m.h", "include/n.h", frozenset({"q"})),  # reads m.h: merged while types.h gathers
    ]
    failed: list = []
    out, moved, _ = headers.relocate(snapshot, events, failed)
    assert [(o, n, [f.key for f in fs]) for o, n, fs in failed] == [
        ("include/wide.h", "include/types.h", ["move.cycle"])
    ]
    assert "typedef struct R" in out["include/types.h"] and "gWide" not in out["include/types.h"]
    assert out["include/wide.h"] == texts["include/wide.h"] and "include/wide.h" not in moved
    merged = out["include/m.h"]  # types.h stays above the body that uses it: it reaches no merged file
    assert merged.index('#include "types.h"') < merged.index("extern s32 gP;")


def test_a_shared_header_takes_no_names_part_that_carries_an_include_of_another_header():
    texts = {
        "include/types.h": "#ifndef TYPES_H\n#define TYPES_H\ntypedef int s32;\n#endif\n",
        "include/p.h": '#ifndef P_H\n#define P_H\n#include "types.h"\ntypedef struct P { s32 v; } P;\n#endif\n',
        "include/m.h": '#ifndef M_H\n#define M_H\n#include "p.h"\nextern P gQ;\nextern P gM;\n#endif\n',
    }
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    failed: list = []
    out, moved, stay = headers.relocate(
        snapshot, [("names", "include/m.h", "include/types.h", frozenset({"gQ"}))], failed
    )
    assert [(o, n, [f.key for f in fs]) for o, n, fs in failed] == [("include/m.h", "include/types.h", ["move.cycle"])]
    assert out["include/m.h"] == texts["include/m.h"] and "p.h" not in out["include/types.h"]
    assert stay == [] and moved == {}


def test_a_walk_reads_a_header_a_shared_header_will_give_back_as_its_own_text():
    texts = {
        "include/types.h": "#ifndef TYPES_H\n#define TYPES_H\ntypedef int s32;\n#endif\n",
        "include/u.h": '#ifndef U_H\n#define U_H\n#include "a.h"\nextern s32 gU;\n#endif\n',
        "include/o.h": '#ifndef O_H\n#define O_H\n#include "u.h"\nextern s32 gO;\n#endif\n',
        "include/a.h": '#ifndef A_H\n#define A_H\n#include "types.h"\n#include "o.h"\nint a;\nint q;\n#endif\n',
        "include/g.h": "#ifndef G_H\n#define G_H\nint g;\n#endif\n",
    }
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    events = [
        ("file", "include/u.h", "include/types.h", frozenset()),  # includes a.h: types.h gives it back
        ("file", "include/a.h", "include/g.h", frozenset()),
        ("names", "include/g.h", "include/n.h", frozenset({"q"})),  # reads g.h: merged while types.h gathers
    ]
    failed: list = []
    out, _, _ = headers.relocate(snapshot, events, failed)
    assert [o for o, _, _ in failed] == ["include/u.h"] and out["include/u.h"] == texts["include/u.h"]
    merged = out["include/g.h"]  # o.h reaches a.h through u.h: it includes g.h back, so it goes after the body
    assert merged.index("int a;") < merged.index('#include "o.h"')


def test_a_back_include_reached_only_through_another_header_gives_back_the_part_it_reaches():
    texts = {
        "include/m.h": "#ifndef M_H\n#define M_H\nint m;\n#endif\n",
        "include/f.h": "#ifndef F_H\n#define F_H\ntypedef int F;\n#endif\n",
        "include/x.h": '#ifndef X_H\n#define X_H\n#include "f.h"\ntypedef F X;\n#endif\n',
        "include/o.h": '#ifndef O_H\n#define O_H\n#include "x.h"\nextern X gO;\n#endif\n',
        "include/g.h": '#ifndef G_H\n#define G_H\n#include "o.h"\nint g;\n#endif\n',
    }
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    events = [("file", f"include/{h}.h", "include/m.h", frozenset()) for h in "fg"]
    failed: list = []
    out, moved, _ = headers.relocate(snapshot, events, failed)
    # merged, x.h entered first would include m.h before its typedef, and m.h's back-include o.h then needs X
    assert [(o, [f.key for f in fs]) for o, _, fs in failed] == [("include/f.h", ["move.cycle"])]
    assert moved == {"include/g.h": "include/m.h"} and out["include/f.h"] == texts["include/f.h"]


def test_a_merge_that_would_make_an_include_cycle_no_order_survives_gives_back_the_part_closing_it():
    texts = {
        "include/lib.h": "#ifndef LIB_H\n#define LIB_H\nint l;\n#endif\n",
        "include/p.h": "#ifndef P_H\n#define P_H\ntypedef int s32;\n#endif\n",
        "include/o.h": '#ifndef O_H\n#define O_H\n#include "p.h"\ntypedef struct O { s32 v; } O;\n#endif\n',
        "include/x.h": '#ifndef X_H\n#define X_H\n#include "o.h"\nextern O gX;\n#endif\n',
        "include/a.h": '#ifndef A_H\n#define A_H\n#include "o.h"\n#include "x.h"\nint a;\n#endif\n',
    }
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    events = [("file", f"include/{h}.h", "include/lib.h", frozenset()) for h in "pa"]
    failed: list = []
    out, moved, _ = headers.relocate(snapshot, events, failed)
    assert [(o, [f.key for f in fs]) for o, _, fs in failed] == [("include/p.h", ["move.cycle"])]
    assert moved == {"include/a.h": "include/lib.h"} and out["include/p.h"] == texts["include/p.h"]
    world = {**texts, **{p: t for p, t in out.items() if t is not None}}
    for first in ("o.h", "x.h", "lib.h"):  # every entry order parses: o.h first no longer reaches lib.h
        headers.parse(_expanded({**world, "include/tu.c": f'#include "{first}"\n'}, "include/tu.c"), "tu.c")


def test_statements_count_the_brackets_of_the_first_branch_of_a_conditional_only():
    text = "int f(int e) {\n#if A\n    if (e == 1) {\n#else\n    if (e == 2) {\n#endif\n        e++;\n    }\n"
    text += "    return e;\n}\nint g;\n"
    found = [" ".join(s.split()) for _, _, s in headers.statements(text, bodies=True)]
    assert len(found) == 2 and found[0].startswith("int f(int e) {") and found[1] == "int g;"


def test_a_back_included_header_that_moved_gives_its_types_from_where_it_went_and_its_own_includes_go_above():
    texts = {
        "include/m.h": "#ifndef M_H\n#define M_H\nint m;\n#endif\n",
        "include/p.h": '#ifndef P_H\n#define P_H\n#include "o.h"\nextern T gp;\nextern U gu;\n#endif\n',
        "include/o.h": '#ifndef O_H\n#define O_H\n#include "s.h"\n#include "x.h"\n'
        "typedef struct T { int a; } T;\n#endif\n",
        "include/s.h": "#ifndef S_H\n#define S_H\ntypedef struct U { int b; } U;\n#endif\n",
        "include/x.h": '#ifndef X_H\n#define X_H\n#include "p.h"\nint x;\n#endif\n',
    }
    snapshot = SimpleNamespace(peek=lambda p: texts[p].encode() if p in texts else None)
    events = [("file", "include/o.h", "include/q.h", frozenset()), ("file", "include/p.h", "include/m.h", frozenset())]
    out, moved, _ = headers.relocate(snapshot, events, [])
    merged = out["include/m.h"]
    assert merged.index("typedef struct T") < merged.index("extern T gp;")  # pulled from q.h, where o.h went
    assert merged.index('#include "s.h"') < merged.index("extern U gu;")  # o.h's include that reaches no merged file
    assert "struct T" not in out["include/q.h"] and moved["include/o.h"] == "include/q.h"
