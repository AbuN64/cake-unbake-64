"""Recovery: nominal origins, all declarations, mechanical edits and cached work counts."""

from __future__ import annotations

import gc
import re
import weakref
from copy import deepcopy
from dataclasses import replace

import fixture
import pytest

from unbake import compare, effort, headers, pool, recovery, store, types
from unbake.contracts import LayoutMap, Member, Placement, Recipe, Refusal, Snapshot, SourceView, UnitSpec, digest
from unbake.families import structs
from unbake.symbols import by_address

HEADER = "struct Foo { int value; };\nstruct Bar { int other; };\ntypedef struct func_ABC_S1 { int x; } func_ABC_S1;\n"


def evidence(raw, path="src/a.c", header=HEADER):
    source = headers._blank(raw)
    source = re.sub(r"^\s*#include[^\n]*", "", source, flags=re.M)
    text = '# 1 "include/shared.h"\n' + header + '# 1 "' + path + '"\n' + source
    return recovery.source_evidence(text, path, raw=raw.encode(), dependencies=(("include/shared.h", digest(header)),))


def derived(raw, path="src/a.c", header=HEADER):
    jobs = recovery.families([evidence(raw, path, header)])
    assert len(jobs) == 1
    return recovery.derive(jobs[0])


def test_existing_real_type_rewrites_only_cast_and_field():
    raw = '#include "shared.h"\nint f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }\n'
    result = derived(raw)
    assert not result["conflicts"]
    assert result["record"] == "struct Foo"
    writes = recovery.rewrite(result, {"src/a.c": raw.encode()})
    assert writes["src/a.c"].decode() == '#include "shared.h"\nint f(struct Foo *p) { return ((p))->value; }\n'


def test_types_expand_typedefs_with_o32_offsets_and_preserve_signedness():
    ast = headers.parse("typedef int s32; struct X { char b; s32 v; void *p; unsigned short u; };", "h")
    record = types.record_layouts(ast)["struct X"]
    assert record["size"] == 16 and record["alignment"] == 4
    assert [(f["offset"], f["width"]) for f in record["fields"]] == [(0, 1), (4, 4), (8, 4), (12, 2)]
    assert record["fields"][1]["type"] == "int"
    assert record["fields"][3]["type"] == "unsigned short"


def test_unknown_layout_and_bitfields_keep_exact_conflicts():
    ast = headers.parse("struct X { unsigned int b:2; }; struct Y { char a[]; };", "h")
    rows = types.record_layouts(ast)
    assert "bitfield" in rows["struct X"]["conflict"]
    assert "incomplete array" in rows["struct Y"]["conflict"]
    ast = headers.parse("struct X { int a; }; void f(void) { struct X { int b; }; }", "h")
    assert "multiple scoped" in types.record_layouts(ast)["struct X"]["conflict"]


def test_anonymous_typedef_record_is_supported():
    header = "typedef struct { int value; } Foo; typedef struct { int x; } func_ABC_S1;\n"
    result = derived("int f(Foo *p) { return ((func_ABC_S1 *)p)->x; }\n", header=header)
    assert not result["conflicts"] and result["record"] == "Foo"


def test_similar_layouts_never_supply_identity():
    result = derived("int f(void *p) { return ((func_ABC_S1 *)p)->x; }\n")
    assert result["record"].startswith("struct Recovered_") and result["record_changes"][0]["open"]


def test_multiple_callers_do_not_leak_backwards():
    header = HEADER.replace("struct Bar { int other; }", "struct Bar { short a; short b; }") + "int generic(void *);\n"
    rows = [
        evidence("int a(struct Foo *p) { generic(p); return ((func_ABC_S1 *)p)->x; }\n", "src/a.c", header),
        evidence("int b(struct Bar *p) { generic(p); return ((func_ABC_S1 *)p)->x; }\n", "src/b.c", header),
        evidence("int generic(void *p) { return ((func_ABC_S1 *)p)->x; }\n", "src/g.c", header),
    ]
    results = {r["identity"]: recovery.derive(r) for r in recovery.families(rows)}
    assert not results["record:struct Foo"]["conflicts"]
    assert {c["reason"] for c in results["record:struct Bar"]["conflicts"]} <= {
        "no unique shared field with identical offset, width and C type"
    }
    unknown = next(r for k, r in results.items() if k.startswith("conflict:"))
    assert any(c.get("anchors") == ["struct Bar", "struct Foo"] for c in unknown["conflicts"])


def _two_callers(left, right, extra=()):
    """`struct L` and `struct R` objects both reach one generic `void *` parameter that is cast."""
    header = left + right + "typedef struct func_ABC_S1 { int x; } func_ABC_S1;\nint generic(void *);\n"
    rows = [
        evidence("int a(struct L *p) { generic(p); return 0; }\n", "src/a.c", header),
        evidence("int b(struct R *p) { generic(p); return 0; }\n", "src/b.c", header),
        evidence("int generic(void *p) { return ((func_ABC_S1 *)p)->x; }\n", "src/g.c", header),
    ]
    return rows + [evidence(raw, path, header) for path, raw in extra]


# Cracker cases from the RageWars census: draft records of one object under different names.
AGREE = [
    # same offsets and widths, different field names: the layout with more fields represents both
    ("struct L { int a; int p[3]; float f; float g; };\n", "struct R { int b; int q[3]; float f; };\n", "struct L"),
    ("struct L { int a; };\n", "struct R { int b; int c; };\n", "struct R"),
    # a pointer member is one access whatever it points to (AiController vs Brain_func_80209308_de)
    (
        "struct P { int k; };\nstruct L { struct P *p; char pad[8]; };\n",
        "struct R { struct P *q; int c; };\n",
        "struct L",
    ),
    # equal arrays of one element type coincide (Matrix vs Matrix_func_80213CF8_de)
    ("struct L { float m[16]; };\n", "struct R { float m[4][4]; };\n", "struct L"),
    # a nested aggregate covers several scalars of the other layout
    (
        "struct V { int x, y, z; };\nstruct L { struct V v; int t; };\n",
        "struct R { int x, y, z; int u; };\n",
        "struct R",
    ),
]


@pytest.mark.parametrize(("left", "right", "merged"), AGREE)
def test_records_that_agree_on_offset_width_and_type_are_one_identity_at_their_origin(left, right, merged):
    jobs = recovery.families(_two_callers(left, right))
    assert [j["identity"] for j in jobs] == ["record:" + merged]
    reasons = {c["reason"] for c in recovery.derive(jobs[0])["conflicts"]}
    assert not {r for r in reasons if "different real record identity" in r or "multiple real" in r}


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("struct L { int a; };\n", "struct R { short a; short b; };\n"),  # straddle (Controller vs Item57BD4)
        ("struct L { int a; };\n", "struct R { float a; };\n"),  # one offset and width, two scalar types
        ("struct L { char pad[4]; };\n", "struct R { int a; };\n"),  # no coinciding scalar
    ],
)
def test_records_that_disagree_stay_a_conflict(left, right):
    (job,) = recovery.families(_two_callers(left, right))
    assert job["identity"].startswith("conflict:") and not job["identities"]
    assert any("multiple real" in c["reason"] for c in recovery.derive(job)["conflicts"])


def test_a_sparse_record_never_bridges_records_that_disagree():
    def fields(*rows):
        return {"fields": [{"offset": o, "width": w, "type": t} for o, w, t in rows]}

    layouts = {
        "struct H": fields((0, 4, "signed int"), (4, 8, "char [8]")),
        "struct A": fields((0, 4, "signed int"), (4, 4, "float")),
        "struct B": fields((0, 4, "signed int"), (4, 4, "signed int")),
    }
    merged = structs._identities({"x": {"struct H", "struct A"}, "y": {"struct H", "struct B"}}, layouts)
    assert merged == {"struct H": "struct A"}


def test_merge_resolves_only_the_meeting_origin_and_keeps_other_families_apart():
    own = ("src/c.c", "int c(struct R *q) { return ((func_ABC_S1 *)q)->x; }\n")
    jobs = {
        j["identity"]: j
        for j in recovery.families(_two_callers("struct L { int a; int c; };\n", "struct R { int b; };\n", [own]))
    }
    assert set(jobs) == {"record:struct L", "record:struct R"}
    assert jobs["record:struct L"]["identities"] == {"struct R": "struct L"}
    assert [c["path"] for c in jobs["record:struct R"]["casts"]] == ["src/c.c"]
    result = recovery.derive(jobs["record:struct L"])
    assert result["record"] == "struct L" and not result["conflicts"]
    # the absorbed record's own declaration is not a conflict: it is the same object
    assert [d["record"] for d in result["declarations"] if d["kind"] == "param"] == ["struct R", None, None]


def test_concrete_parameter_contract_migrates_named_and_unnamed_declarations():
    header = HEADER + "void consume(struct Foo *);\nint f(void *);\n"
    raw = '#include "shared.h"\nint f(void *p) { consume(p); return ((func_ABC_S1 *)p)->x; }\n'
    result = derived(raw, header=header)
    assert not result["conflicts"]
    writes = recovery.rewrite(result, {"src/a.c": raw.encode(), "include/shared.h": header.encode()})
    assert b"int f(struct Foo *p)" in writes["src/a.c"]
    assert b"int f(struct Foo *)" in writes["include/shared.h"]
    assert b"consume(struct Foo *)" in writes["include/shared.h"]


def test_same_named_static_functions_do_not_share_parameter_constraints():
    rows = [
        evidence(
            "static int consume(struct Foo *p) { return 0; }\n"
            "int a(void *p) { consume(p); return ((func_ABC_S1 *)p)->x; }\n",
            "src/a.c",
        ),
        evidence("static int consume(void *p) { return ((func_ABC_S1 *)p)->x; }\n", "src/b.c"),
    ]
    jobs = recovery.families(rows)
    assert {j["identity"] for j in jobs} == {"record:struct Foo", "origin:param:src/b.c:consume:0"}
    assert not recovery.derive(next(j for j in jobs if j["identity"] == "record:struct Foo"))["conflicts"]
    other = recovery.derive(next(j for j in jobs if j["identity"].startswith("origin:")))
    assert other["record"].startswith("struct Recovered_") and other["record_changes"][0]["open"]
    assert {d["owner"] for row in rows for d in row["declarations"] if d["kind"] == "param"} >= {"consume"}


def test_same_named_static_globals_and_returns_are_scoped_to_the_translation_unit():
    rows = [
        evidence(
            "static struct Foo *object;\nstatic struct Foo *get(void) { return object; }\n"
            "int a(void) { return ((func_ABC_S1 *)get())->x; }\n",
            "src/a.c",
        ),
        evidence(
            "static struct Bar *object;\nstatic struct Bar *get(void) { return object; }\n"
            "int b(void) { return ((func_ABC_S1 *)get())->x; }\n",
            "src/b.c",
        ),
    ]
    keys = {d["key"] for row in rows for d in row["declarations"]}
    assert {"global:src/a.c:object", "global:src/b.c:object", "return:src/a.c:get", "return:src/b.c:get"} <= keys
    jobs = recovery.families(rows)
    assert {j["identity"] for j in jobs} == {"record:struct Foo", "record:struct Bar"}
    assert all(not recovery.derive(j)["conflicts"] for j in jobs)
    rows = [
        evidence("static struct Foo *object;\nint a(void) { return ((func_ABC_S1 *)object)->x; }\n", "src/a.c"),
        evidence("static void *object;\nint b(void) { return ((func_ABC_S1 *)object)->x; }\n", "src/b.c"),
    ]
    jobs = recovery.families(rows)
    assert {j["identity"] for j in jobs} == {"record:struct Foo", "origin:global:src/b.c:object"}


def test_data_only_definition_is_retained_and_blocks_byte_global_synthesis():
    header = HEADER + "extern unsigned char blob[8];\n"
    rows = [
        evidence("int a(void) { return ((func_ABC_S1 *)blob)->x; }\n", "src/a.c", header),
        evidence("unsigned char blob[8] = {0};\n", "src/data.c", header),
    ]
    assert any(d["defined"] and d["name"] == "blob" for d in rows[1]["declarations"])
    (job,) = recovery.families(rows)
    assert "src/data.c" in job["sources"]
    result = recovery.derive(job)
    assert not result["mapping"]
    assert any("C definition" in c["reason"] for c in result["conflicts"])


def test_shadowed_names_have_distinct_origins():
    raw = (
        "int f(struct Foo *p) { int a = ((func_ABC_S1 *)p)->x;\n"
        " { struct Bar *p = 0; a += ((func_ABC_S1 *)p)->x; } return a; }\n"
    )
    rows = recovery.families([evidence(raw)])
    assert {r["identity"] for r in rows} == {"record:struct Foo", "record:struct Bar"}
    assert all(not recovery.derive(r)["conflicts"] for r in rows)


def test_reassigned_pointer_with_distinct_object_types_is_a_conflict():
    header = HEADER.replace("struct Bar { int other; }", "struct Bar { short other; short more; }")
    header += "extern struct Foo a; extern struct Bar b;\n"
    result = derived("int f(void) { void *p = &a; p = &b; return ((func_ABC_S1 *)p)->x; }\n", header=header)
    assert not result["mapping"]
    assert any(c.get("anchors") == ["struct Bar", "struct Foo"] for c in result["conflicts"])


def test_signedness_and_overlapping_existing_fields_are_not_guessed():
    header = "struct Foo { unsigned int value; }; typedef struct func_ABC_S1 { int x; } func_ABC_S1;\n"
    result = derived("int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }\n", header=header)
    conflict = next(c for c in result["conflicts"] if "offset, width" in c["reason"])
    assert conflict["existing"][0]["type"] == "unsigned int"
    assert conflict["type"] == "int"


def test_extending_implicit_padding_preserves_all_original_offsets_and_size():
    header = "struct Foo { char first; int last; };\ntypedef struct func_ABC_S1 { char first; char x; } func_ABC_S1;\n"
    raw = "int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }\n"
    result = derived(raw, header=header)
    assert not result["conflicts"] and result["record_changes"][0]["kind"] == "extend"
    writes = recovery.rewrite(result, {"src/a.c": raw.encode(), "include/shared.h": header.encode()})
    ast = headers.parse(writes["include/shared.h"].decode(), "include/shared.h")
    record = types.record_layouts(ast)["struct Foo"]
    assert record["size"] == 8 and record["alignment"] == 4
    assert {f["name"]: f["offset"] for f in record["fields"]}["last"] == 4
    assert {f["name"]: f["offset"] for f in record["fields"]}["x"] == 1


def test_unused_declared_padding_can_be_split_but_observed_padding_is_protected():
    header = (
        "struct Foo { int first; char pad4[8]; int last; };\n"
        "typedef struct func_ABC_S1 { char pad0[4]; int x; } func_ABC_S1;\n"
    )
    raw = "int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }\n"
    result = derived(raw, header=header)
    assert not result["conflicts"]
    writes = recovery.rewrite(result, {"src/a.c": raw.encode(), "include/shared.h": header.encode()})
    record = types.record_layouts(headers.parse(writes["include/shared.h"].decode(), "h"))["struct Foo"]
    assert record["size"] == 16
    assert next(f for f in record["fields"] if f["name"] == "last")["offset"] == 12
    result = derived(raw.replace("return ", "p->pad4[0] = 0; return "), header=header)
    assert any("padding is observed" in c["reason"] for c in result["conflicts"])


def test_named_byte_buffer_is_not_padding():
    header = (
        "struct Foo { int first; char buffer[8]; int last; };\n"
        "typedef struct func_ABC_S1 { char pad0[4]; int x; } func_ABC_S1;\n"
    )
    result = derived("int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }\n", header=header)
    assert result["conflicts"]


def test_sized_byte_global_builds_one_real_struct_and_changes_all_declarations():
    header = "extern unsigned char blob[8];\ntypedef struct func_ABC_S1 { int x; } func_ABC_S1;\n"
    raws = {
        f"src/f{i}.c": '#include "shared.h"\nint f' + str(i) + "(void) { return ((func_ABC_S1 *)blob)->x; }\n"
        for i in range(10)
    }
    rows = [evidence(raw, p, header) for p, raw in raws.items()]
    jobs = recovery.families(rows)
    assert len(jobs) == 1
    result = recovery.derive(jobs[0])
    assert not result["conflicts"] and len(result["files"]) == 10
    contents = {p: raw.encode() for p, raw in raws.items()} | {"include/shared.h": header.encode()}
    writes = recovery.rewrite(result, contents)
    assert len(writes) == 11
    assert all(b"func_ABC_S1 *" not in writes[p] and b"&blob" in writes[p] for p in raws)
    assert writes["include/shared.h"].count((result["record"] + " {").encode()) == 1
    record = types.record_layouts(headers.parse(writes["include/shared.h"].decode(), "h"))[result["record"]]
    assert record["size"] == 8 and record["alignment"] == 4


def test_sized_global_with_nonview_uses_is_reported():
    header = "extern unsigned char blob[8];\ntypedef struct func_ABC_S1 { int x; } func_ABC_S1;\n"
    result = derived("int f(void) { blob[0] = 1; return ((func_ABC_S1 *)blob)->x; }\n", header=header)
    assert any("uses outside" in c["reason"] for c in result["conflicts"])


def test_direct_typed_return_has_interface_evidence_and_is_evaluated_once():
    header = HEADER + "struct Foo *get(void);\n"
    raw = "int f(void) { return ((func_ABC_S1 *)get())->x; }\n"
    result = derived(raw, header=header)
    assert not result["conflicts"]
    text = recovery.rewrite(result, {"src/a.c": raw.encode()})["src/a.c"].decode()
    assert text.count("get()") == 1 and "->value" in text


def test_opaque_return_and_pointer_arithmetic_do_not_establish_identity():
    header = HEADER + "void *get(void);\n"
    for expression in ("get()", "(p + 4)"):
        result = derived("int f(void *p) { return ((func_ABC_S1 *)" + expression + ")->x; }\n", header=header)
        assert result["record"].startswith("struct Recovered_") and result["record_changes"][0]["open"]


def test_standalone_cast_is_retained_as_a_conflict():
    raw = "void f(void *p) { func_ABC_S1 *q = (func_ABC_S1 *)p; }\n"
    rows = evidence(raw)
    assert len(rows["casts"]) == 1
    assert "without direct field access" in rows["casts"][0]["conflict"]


def test_comments_and_strings_supply_no_cast_evidence():
    raw = 'void f(void) { /* ((func_ABC_S1 *)p)->x */ const char *s = "(func_ABC_S1 *)p"; }\n'
    assert not evidence(raw)["casts"]


def test_stale_coordinates_refuse_without_partial_writes():
    raw = "int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }\n"
    result = derived(raw)
    with pytest.raises(ValueError, match="coordinate"):
        recovery.rewrite(result, {"src/a.c": raw.replace(")->x", ")->y").encode()})


def test_assembly_width_conflict_is_reported():
    job = recovery.families([evidence("int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }\n")])[0]
    job["assembly"] = {"v": {"f": {"args": [(0, 0, 2)]}}}
    job["holders"] = {"f": ["v"]}
    result = recovery.derive(job)
    assert any(c.get("asm_widths") == [2] for c in result["conflicts"])


def _scan_counted(item):
    path, raw = item
    return evidence(raw, path)


def test_rerun_computes_no_jobs_and_one_changed_file_only_rederives_its_family(cfg, monkeypatch):
    # Native execution is mocked inline, but use the actual store and pool cache.
    monkeypatch.setattr(pool, "_in_worker", True)
    records = []
    monkeypatch.setattr(effort, "record_pool", lambda name, items, **kw: records.append((name, kw["jobs"])))
    inputs = [
        ("src/a.c", "int a(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }\n"),
        ("src/b.c", "int b(struct Bar *p) { return ((func_ABC_S1 *)p)->x; }\n"),
    ]

    snapshot = Snapshot(cfg, "head", LayoutMap({}, {}, {}, "d", (), {}), {}, {}, "s")
    authored = {"function": {}, "struct": {}}

    def run():
        rows = pool.map(cfg, "recovery.source", _scan_counted, inputs, digest)
        jobs = pool.map(cfg, "recovery.graph", recovery.families, [rows], recovery._derive_key)[0]
        items = [(snapshot, job, authored, {}, {}, [], {}, recovery._derive_key(job)) for job in jobs]
        return pool.map(cfg, "recovery.family", recovery._family_job, items, recovery._keyed)

    first = run()
    assert sum(n for name, n in records if name == "recovery.family") == 2
    records.clear()
    assert run() == first
    assert sum(n for _, n in records) == 0
    records.clear()
    inputs[0] = (inputs[0][0], inputs[0][1].replace("return ", "/* source changed */ return "))
    changed = run()
    assert sum(n for name, n in records if name == "recovery.source") == 1
    assert sum(n for name, n in records if name == "recovery.family") == 1
    before_bar = next(r for r in first if r["record"] == "struct Bar")
    assert next(r for r in changed if r["record"] == "struct Bar") == before_bar


def test_analysis_uses_real_config_and_only_writes_candidates_outside_project(cfg, tmp_path, monkeypatch):
    out = tmp_path / "out"
    store.write(cfg.project.root / "include/shared.h", HEADER.encode())
    store.write(cfg.project.root / "types.toml", b"schema = 1\n[function]\n[global]\n[struct]\n")
    files = {p: p.read_bytes() for p in (cfg.project.root / "include").rglob("*.h")}
    snapshot = Snapshot(cfg, "head", LayoutMap({}, {}, {}, "d", (), {}), {}, {}, "s")
    captured = []
    monkeypatch.setattr(recovery.layout, "capture", lambda config: captured.append(config) or snapshot)
    monkeypatch.setattr(recovery.report, "progress", lambda snapshot: {"versions": {}})
    monkeypatch.setattr(recovery.infer, "scan", lambda snapshot: {})
    monkeypatch.setattr(pool, "_in_worker", True)
    result = recovery.run(cfg, {"out": out, "family": (), "exclude": (), "submit": False})
    assert result["census"]["families"] == 0
    assert captured == [cfg]
    assert all(p.read_bytes() == content for p, content in files.items())
    assert not (out / ".analysis").exists()
    with pytest.raises(Refusal, match="outside"):
        recovery.run(cfg, {"out": cfg.project.root / "out"})


def test_ordinary_compare_admission_keeps_the_existing_measure_path(cfg, tmp_path, monkeypatch):
    unit = UnitSpec("src/a.c", "c", "g", ("a",), "gcc-test", {})
    member = Member("a", "function", "c", "g", (Placement("a", ".text", 0, 8, 0x80000400),))
    snapshot = Snapshot(cfg, "head", LayoutMap({}, {"a": member}, {unit.path: unit}, "d", (), {}), {}, {}, "s")
    primary = tmp_path / "a.c"
    primary.write_bytes(b"void a(void) {}")
    monkeypatch.setattr(compare, "bind", lambda s, f, name, notes=None: (unit, s))
    monkeypatch.setattr(compare.ownership, "derive", lambda s, u, **kw: (s, u))
    monkeypatch.setattr(compare.recipes, "resolve", lambda *a: Recipe("gcc-test", (), (), (), "r"))
    old_calls = []
    proof = fixture.proof(unit.path, "a", "a", True)
    monkeypatch.setattr(compare, "measure", lambda *a: old_calls.append(a) or (proof,))
    monkeypatch.setattr(compare, "measure_units", lambda *a: pytest.fail("ordinary admission used recovery scoring"))
    monkeypatch.setattr(compare.land, "submit", lambda *a: None)
    params = {"function": None, "note": None, "withhold": False}
    done = compare._admit(cfg, snapshot, [primary], params, {}, None, "submit")
    assert old_calls == [(snapshot, unit, {}, None)]
    assert done["proofs"] == (proof,)


def test_prove_extras_reports_a_nonexact_extra_and_never_queues_it(cfg, tmp_path, monkeypatch):
    units = [UnitSpec(f"src/{name}.c", "c", "g", (name,), "gcc-test", {}) for name in ("a", "b")]
    members = {n: Member(n, "function", "c", "g", (Placement("a", ".text", 0, 8, 0x80000400),)) for n in ("a", "b")}
    snapshot = Snapshot(cfg, "head", LayoutMap({}, members, {u.path: u for u in units}, "d", (), {}), {}, {}, "s")
    primary = tmp_path / "a.c"
    primary.write_bytes(b"void a(void) {}")
    monkeypatch.setattr(compare, "_extra", lambda *a: ("src/b.c", b"void b(void) {}"))
    monkeypatch.setattr(compare.layout, "overlay", lambda s, writes: s)
    monkeypatch.setattr(compare, "bind", lambda s, f, name, notes=None: (units[0], s))
    monkeypatch.setattr(compare.ownership, "derive", lambda s, u, **kw: (s, u))
    monkeypatch.setattr(compare.recipes, "resolve", lambda *a: Recipe("gcc-test", (), (), (), "r"))
    monkeypatch.setattr(compare.headers, "consumers", lambda *a: ())
    monkeypatch.setattr(compare, "measure", lambda *a: pytest.fail("prove_extras measured only the primary"))
    proofs = (fixture.proof("src/a.c", "a", "a", True), fixture.proof("src/b.c", "b", "a", False, ("bytes differ",)))
    monkeypatch.setattr(compare, "measure_units", lambda *a: proofs)
    monkeypatch.setattr(compare.layout, "unit_of", lambda s, m: units[0] if m == "a" else units[1])
    monkeypatch.setattr(compare, "_baseline", lambda *a: (fixture.proof("src/b.c", "b", "a", True),))  # exact at HEAD
    params = {"function": None, "note": None, "withhold": False, "prove_extras": True}
    done = compare._admit(cfg, snapshot, [primary, tmp_path / "b.c"], params, {}, None, "submit")
    assert done["submission"] is None
    assert len(done["found"]) == 1 and done["found"][0].unit == "b"
    assert done["found"][0].missing == ("bytes differ",)
    # A one-source recovery must not silently turn a nonexact result into a fuzzy submission.
    monkeypatch.setattr(
        compare, "measure_units", lambda *a: (fixture.proof("src/a.c", "a", "a", False, ("bytes differ",)),)
    )
    monkeypatch.setattr(compare.land, "submit", lambda *a: pytest.fail("nonexact recovery was queued"))
    done = compare._admit(cfg, snapshot, [primary], params, {}, None, "submit")
    assert done["submission"] is None and done["found"][0].unit == "a"


def test_coordinated_scoring_reuses_one_native_pool_map_for_all_units(cfg, monkeypatch):
    units = [UnitSpec(f"src/{name}.c", "c", "g", (name,), "gcc-test", {}) for name in ("a", "b")]
    members = {
        name: Member(name, "function", "c", "g", tuple(Placement(v, ".text", 0, 8, 0x80000400) for v in ("a", "b")))
        for name in ("a", "b")
    }
    snapshot = Snapshot(cfg, "head", LayoutMap({}, members, {u.path: u for u in units}, "d", (), {}), {}, {}, "s")
    monkeypatch.setattr(compare.recipes, "resolve", lambda *a: Recipe("gcc-test", (), (), (), "r"))
    monkeypatch.setattr(compare.ownership, "derive_pairs", lambda pairs, **kw: list(pairs))
    calls = []

    def mapped(config, name, function, jobs, key=None):
        calls.append((name, len(jobs), function))
        if name == "compare.stamps":  # every unit and version reads its own inputs: four distinct stamps
            return [f"{j[1].path}:{j[2]}" for j in jobs]
        return [(fixture.proof(j[1].path, j[1].members[0], j[2], True),) for j in jobs]

    monkeypatch.setattr(compare.pool, "map", mapped)
    proofs = compare.measure_units(snapshot, units, {})
    assert len(proofs) == 4
    assert calls == [("compare.stamps", 4, compare._key), ("compare.measure", 4, compare.native.prove_args)]


ATTRIBUTED = (
    "struct Plain { int a; short b; };\n"
    "struct Aligned { short type; char pad[14]; } __attribute__((aligned(4)));\n"
    "typedef struct Lone { unsigned char a; } __attribute__((aligned(8))) Lone;\n"
    "typedef struct Env { short a; int b; } __attribute__((packed)) Env;\n"
)


def test_aligned_and_packed_records_are_laid_out():
    raw = '#include "shared.h"\nstruct Plain p;\nstruct Aligned a;\nLone l;\nEnv e;\n'
    row = evidence(raw, header=ATTRIBUTED)
    assert row["records"]["struct Plain"] == {
        "size": 8,
        "alignment": 4,
        "fields": row["records"]["struct Plain"]["fields"],
    }
    aligned = row["records"]["struct Aligned"]
    assert (aligned["size"], aligned["alignment"]) == (16, 4)
    assert [(f["name"], f["offset"]) for f in aligned["fields"]] == [("type", 0), ("pad", 2)]
    assert (row["records"]["struct Lone"]["size"], row["records"]["struct Lone"]["alignment"]) == (8, 8)
    env = row["records"]["struct Env"]
    assert (env["size"], env["alignment"]) == (6, 1) and not row["conflicts"]


def test_a_field_alignment_and_a_pragma_pack_region():
    header = (
        "struct F { char a; int b __attribute__((aligned(8))); };\n"
        "#pragma pack(push, 1)\nstruct P { char a; int b; };\n#pragma pack(pop)\nstruct Q { char a; int b; };\n"
    )
    row = evidence('#include "shared.h"\nstruct F f;\nstruct P p;\nstruct Q q;\n', header=header)
    assert (row["records"]["struct F"]["size"], row["records"]["struct F"]["alignment"]) == (16, 8)
    assert row["records"]["struct F"]["fields"][1]["offset"] == 8
    assert row["records"]["struct Q"]["size"] == 8
    assert (row["records"]["struct P"]["size"], row["records"]["struct P"]["alignment"]) == (5, 1)
    assert not row["conflicts"]


@pytest.mark.parametrize("workers", [6, 12])
def test_recovery_conflicts_and_families_dispatch_once_and_warm_run_has_no_jobs(cfg, monkeypatch, workers):
    cfg = replace(cfg, host=replace(cfg.host, workers=workers))
    monkeypatch.setattr(pool, "_in_worker", True)
    records, keys = [], []
    monkeypatch.setattr(effort, "record_pool", lambda name, items, **kw: records.append((name, kw["jobs"])))
    authored = {"function": {}, "struct": {}}
    paths = ["include/shared.h", "src/a.c"]
    texts = [
        "".join(f"extern int global_{n};\n" for n in range(100)),
        "".join(f"extern float global_{n};\n" for n in range(100)),
    ]
    scans = [headers._file_job((t.encode(), p.endswith(".c"))) for p, t in zip(paths, texts, strict=True)]
    source = UnitSpec("src/a.c", "c", "g", (), "gcc-test", {})
    snapshot = Snapshot(cfg, "head", LayoutMap({}, {}, {source.path: source}, "d", (), {}), {}, {}, "s")
    monkeypatch.setattr(headers, "sources", lambda *a: paths[:1])
    monkeypatch.setattr(headers, "_scanned", lambda *a: scans)
    reference = headers._disagreement_rows((paths, scans, ()))[1]
    base = recovery.families([evidence("int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }")])[0]
    families = [{**deepcopy(base), "id": str(n)} for n in range(workers * 2)]
    items = [(snapshot, job, authored, {}, {}, [], {}, recovery._derive_key(job)) for job in families]

    def key(item):
        keys.append(item[-1])
        return recovery._keyed(item)

    expected = []
    for job in deepcopy(families):
        recovery.enrich(snapshot, job, authored, {}, {}, [], {})
        expected.append(recovery.derive(job))
    for iteration in range(2):
        _, found = recovery._contexts(snapshot, authored)
        assert sorted(found, key=lambda f: f.unit) == sorted(reference, key=lambda f: f.unit)
        assert pool.map(cfg, "recovery.family", recovery._family_job, items, key) == expected
        assert len(keys) == len(items)
        if iteration == 0:
            assert sum(n for stage, n in records if stage == "recovery.conflicts") >= workers
            assert sum(n for stage, n in records if stage == "recovery.family") == workers * 2
        else:
            assert sum(n for _, n in records) == 0
        records.clear()
        keys.clear()


@pytest.mark.parametrize("workers", [6, 12])
def test_source_stages_parse_distinct_texts_once_and_warm_run_has_no_jobs(cfg, monkeypatch, workers):
    cfg = replace(cfg, host=replace(cfg.host, workers=workers))
    units = [UnitSpec(f"src/f{n}.c", "c", "g", (), "gcc-test", {}) for n in range(workers * 2)]
    snapshot = Snapshot(cfg, "head", LayoutMap({}, {}, {u.path: u for u in units}, "d", (), {}), {}, {}, "s")
    header = HEADER + "typedef int Index; enum { Limit = 3 };\n"
    store.write(cfg.project.root / "include/shared.h", header.encode())
    for n, unit in enumerate(units):
        store.write(
            cfg.project.root / unit.path,
            f"int f{n}(struct Foo *p) {{ Index i = Limit; {{ int Index = 2; i += Index; }} "
            f"return ((func_ABC_S1 *)p)->x + i; }}\n".encode(),
        )
    versions = tuple(cfg.project.version_macros)
    recipe = Recipe("gcc-test", (), (), (), "r")
    monkeypatch.setattr(recovery.compare, "holders", lambda *a: versions)
    monkeypatch.setattr(recovery.recipes, "resolve", lambda *a: recipe)

    def seen(snapshot, unit, version, recipe, **kwargs):
        raw = snapshot.read(unit.path).decode()
        h = snapshot.read("include/shared.h").decode()
        if unit == units[0] and version == versions[-1]:
            h = h.replace("int x;", "short x;")
        text = (
            f'# 1 "{unit.path}"\n# 1 "include/{unit.path}.h"\n'
            f'extern int private_{units.index(unit)};\n# 1 "include/shared.h"\n'
            + h
            + f'# 2 "{unit.path}"\n# 1 "include/extra.h"\nextern int extra;\n'
            f'# 1 "{unit.path}"\n' + raw
        )
        deps = ((unit.path, digest(raw)), ("include/shared.h", digest(h)))
        return SourceView(unit.path, version, digest(text), text, deps, ())

    monkeypatch.setattr(structs.view, "get", seen)
    expected = [
        recovery.source_evidence(
            v.text, u.path, (str(cfg.project.root), str(cfg.project.root)), snapshot.read(u.path), list(v.dependencies)
        )
        for version in versions
        for u in units
        for v in [seen(snapshot, u, version, recipe)]
    ]
    expected_families = recovery.families(expected)
    parsed, indexed, records = [], [], []
    real_index = structs._index

    def index(ast):
        indexed.append([n.coord.file for n in ast.ext if n.coord])
        return real_index(ast)

    monkeypatch.setattr(structs, "_index", index)
    real_parse = structs._parse

    def parse(text, path, scope=()):
        parsed.append((text, path))
        return real_parse(text, path, scope)

    monkeypatch.setattr(structs, "_parse", parse)
    monkeypatch.setattr(pool, "_in_worker", True)
    monkeypatch.setattr(effort, "record_pool", lambda name, items, **kw: records.append((name, kw["jobs"])))

    def run():
        return recovery._sources(snapshot, str(cfg.project.root), {"c": {"decompiled": True}})

    def closure(snap, unit, version):
        return ((unit.path, digest(snap.read(unit.path))), ("include/shared.h", digest(snap.read("include/shared.h"))))

    monkeypatch.setattr(structs.view, "closure", closure)
    first = run()
    assert first == expected
    assert recovery.families(first) == expected_families
    assert len(parsed) == 2 * len(units) + 4
    assert sum(path == "" for _, path in parsed) == len(units) + 3
    assert len(indexed) == len(parsed)
    assert sum("include/shared.h" in paths for paths in indexed) == 2
    assert sum("include/extra.h" in paths for paths in indexed) == 1
    assert records == [
        ("recovery.views", len(units)),
        ("recovery.prefix", len(units) + 1),
        ("recovery.source", len(units) + 1),
    ]
    parsed.clear()
    indexed.clear()
    records.clear()
    assert run() == first
    assert not parsed and not indexed
    assert sum(jobs for _, jobs in records) == 0
    records.clear()
    store.write(cfg.project.root / units[-1].path, snapshot.read(units[-1].path).replace(b"+ i", b"+ i + 1"))
    changed = run()
    assert len(parsed) == len(indexed) == 1
    assert indexed[0] == [units[-1].path]
    assert records == [("recovery.views", 1), ("recovery.prefix", 0), ("recovery.source", 1)]
    assert changed[0] == first[0]
    assert changed[len(units) - 1] != first[len(units) - 1]


@pytest.mark.parametrize(
    "header,body",
    [
        (
            HEADER + "#pragma pack(push, 1)\n",
            "struct Packed { char c; int x; };\n#pragma pack(pop)\n"
            "int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }",
        ),
        (
            HEADER + "typedef int I;\n",
            "int f(struct Foo *p) { typedef short I; I n = 1; return ((func_ABC_S1 *)p)->x + n; }",
        ),
        (
            HEADER + "static int helper(struct Foo *p) { return p->value; }\n",
            "int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x + helper(p); }",
        ),
        (HEADER + "struct Fragment {\n", "int x; }; int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }"),
        (
            HEADER + "struct __attribute__((aligned(8))) Aligned { int x; };\n",
            "int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }",
        ),
        (HEADER, "int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x + ; }"),
        (HEADER + "extern void old(arg0);\n", "int f(struct Foo *p) { old(p); return ((func_ABC_S1 *)p)->x; }"),
        (HEADER, "int f(p) struct Foo *p; { return ((func_ABC_S1 *)p)->x; }"),
        (
            HEADER + 'typedef struct { int a; } Anonymous;\n# 1 "include/second.h"\nextern Anonymous *global;\n',
            "int f(void) { return ((func_ABC_S1 *)global)->x; }",
        ),
        (
            HEADER + '# 1 "include/second.h"\nstruct Foo { short value; };\n',
            "int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }",
        ),
        (
            HEADER + 'typedef int Size;\n# 1 "include/second.h"\nstruct Box { Size x; };\n',
            "typedef short Size; int f(struct Box *p) { return ((func_ABC_S1 *)p)->x; }",
        ),
        (
            HEADER + "typedef struct { int value; } Alias;\n",
            "typedef int Alias; int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }",
        ),
    ],
)
def test_shared_prefix_preserves_evidence_and_parse_conflicts(cfg, monkeypatch, header, body):
    unit = UnitSpec("src/a.c", "c", "g", (), "gcc-test", {})
    snapshot = Snapshot(cfg, "head", LayoutMap({}, {}, {unit.path: unit}, "d", (), {}), {}, {}, "s")
    raw = body.encode()
    store.write(cfg.project.root / unit.path, raw)
    text = '# 1 "include/shared.h"\n' + header + '# 1 "src/a.c"\n' + body
    version = cfg.project.names_from
    seen = SourceView(unit.path, version, digest(text), text, (), ())
    monkeypatch.setattr(structs.view, "get", lambda *a, **kw: seen)
    job = (snapshot, unit, (version,), Recipe("gcc-test", (), (), (), "r"), str(cfg.project.root))
    key, prefix = structs._prepare_job(job)[0]
    structs._prefix_job((job, prefix))
    found = structs._scan_job((job, key))
    try:
        expected = recovery.source_evidence(text, unit.path, (str(cfg.project.root), str(cfg.project.root)), raw, [])
    except Refusal as error:
        assert found["conflicts"] == [{"reason": f.reason, "path": f.path, "line": f.line} for f in error.findings]
        assert found["casts"][0]["conflict"] == "; ".join(f.reason for f in error.findings)
    else:
        assert found == expected
        assert recovery.families([found]) == recovery.families([expected])
    cache = store.content(cfg)._inner._cache
    cache.delete("recovery.input:" + key)
    cache.delete("recovery.prefix.text:" + prefix)
    cache.delete("recovery.prefix:" + prefix)
    for part in list(cache.iterkeys()):
        if part.startswith("recovery.fragment:"):
            cache.delete(part)
    effort._memo.clear()
    assert structs._scan_job((job, key)) == found


@pytest.mark.parametrize(
    "declaration,expected_types",
    [
        ("extern void old(first, second);", ["", ""]),
        ("void old(first, second) struct Foo *second; int first; {}", ["int", "struct Foo *"]),
        ("void old(first, second) struct Foo *second; {}", ["int", "struct Foo *"]),
    ],
)
def test_identifier_list_parameters_preserve_positions(declaration, expected_types):
    row = evidence(declaration + "\nint f(struct Foo *p) { old(p, p); return ((func_ABC_S1 *)p)->x; }")
    if declaration.startswith("extern"):
        assert not any(flow[1].startswith("param:old:") for flow in row["flows"])
    else:
        params = [d for d in row["declarations"] if d["owner"] == "old"]
        assert [(d["name"], d["index"], d["type"]) for d in params] == [
            ("first", 0, expected_types[0]),
            ("second", 1, expected_types[1]),
        ]
        assert ("param:f:0", "param:old:1", "argument") in row["flows"]
        assert not any(flow[1] == "param:old:0" for flow in row["flows"])
    assert not row["conflicts"]


@pytest.mark.parametrize("stage", ["view", "prefix", "parse", "index", "evidence", "layout"])
def test_source_failure_is_a_unit_conflict_and_other_units_finish(cfg, monkeypatch, stage):
    units = [UnitSpec(f"src/{name}.c", "c", "g", (), "gcc-test", {}) for name in ("bad", "good")]
    snapshot = Snapshot(cfg, "head", LayoutMap({}, {}, {u.path: u for u in units}, "d", (), {}), {}, {}, "s")
    for unit in units:
        store.write(cfg.project.root / unit.path, b"int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x; }")
    recipe = Recipe("gcc-test", (), (), (), "r")
    monkeypatch.setattr(pool, "_in_worker", True)
    monkeypatch.setattr(recovery.compare, "holders", lambda *a: (cfg.project.names_from,))
    monkeypatch.setattr(recovery.recipes, "resolve", lambda *a: recipe)
    monkeypatch.setattr(structs.view, "closure", lambda snap, unit, version: ((unit.path, "pin"),))

    def seen(snapshot, unit, version, recipe, **kwargs):
        if stage == "view" and unit == units[0]:
            raise ValueError("broken evidence")
        text = f'# 1 "include/{unit.path}.h"\n' + HEADER + f'# 1 "{unit.path}"\n' + snapshot.read(unit.path).decode()
        return SourceView(unit.path, version, digest(text), text, (), ())

    monkeypatch.setattr(structs.view, "get", seen)
    owner, name = (
        (types, "record_layouts")
        if stage == "layout"
        else (
            structs,
            {"prefix": "_fragment", "parse": "_parse", "index": "_index", "evidence": "_evidence", "view": "_parse"}[
                stage
            ],
        )
    )
    original = getattr(owner, name)

    def fail(*args, **kwargs):
        if stage == "prefix":
            bad = "src/bad.c.h" in args[1]
        elif stage == "parse":
            bad = args[1] == units[0].path
        elif stage in ("index", "evidence", "layout"):
            bad = any(n.coord and n.coord.file == units[0].path for n in args[0].ext)
        else:
            bad = False
        if bad:
            raise AttributeError("broken evidence")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, fail)
    rows = recovery._sources(snapshot, str(cfg.project.root), {"c": {"decompiled": True}})
    assert len(rows) == 2 and rows[1]["casts"] and not rows[1]["conflicts"]
    assert rows[0]["conflicts"][0]["path"] == units[0].path
    assert rows[0]["conflicts"][0]["reason"] == (
        "ValueError: broken evidence" if stage == "view" else "AttributeError: broken evidence"
    )
    assert rows[0]["casts"][0]["conflict"] == rows[0]["conflicts"][0]["reason"]
    assert recovery.families(rows)


@pytest.mark.parametrize("units", [12, 96])
def test_source_jobs_release_parsed_trees_and_prefix_scopes(cfg, monkeypatch, units):
    from pycparser import c_ast

    parsed, indexed = [], []
    real_parse, real_index = structs._parse, structs._index

    def parse(*args, **kwargs):
        ast, scope = real_parse(*args, **kwargs)
        parsed.extend(weakref.ref(node) for node in ast.ext)
        return ast, scope

    def index(ast):
        indexed.append([node.coord.file for node in ast.ext])
        return real_index(ast)

    monkeypatch.setattr(structs, "_parse", parse)
    monkeypatch.setattr(structs, "_index", index)
    header = HEADER + "".join(f"typedef int Type{i};\n" for i in range(128))
    for n in range(units):
        unit = UnitSpec(f"src/f{n}.c", "c", "g", (), "gcc-test", {})
        snapshot = Snapshot(cfg, "head", LayoutMap({}, {}, {unit.path: unit}, "d", (), {}), {}, {}, "s")
        raw = f"int f{n}(struct Foo *p) {{ return ((func_ABC_S1 *)p)->x; }}".encode()
        store.write(cfg.project.root / unit.path, raw)
        text = '# 1 "include/shared.h"\n' + header + f'# 1 "{unit.path}"\n' + raw.decode()
        version = cfg.project.names_from
        seen = SourceView(unit.path, version, digest(text), text, (), ())
        monkeypatch.setattr(structs.view, "get", lambda *args, seen=seen, **kwargs: seen)
        job = (snapshot, unit, (version,), Recipe("gcc-test", (), (), (), "r"), str(cfg.project.root))
        key, prefix = structs._prepare_job(job)[0]
        structs._prefix_job((job, prefix))
        row = structs._scan_job((job, key))
        assert row["casts"] and not row["conflicts"]
        gc.collect()
        assert not any(ref() is not None for ref in parsed)
        assert not any(isinstance(k, tuple) and k[0].startswith("recovery.") for k in effort._memo)

        def plain(value):
            assert not isinstance(value, c_ast.Node)
            if isinstance(value, dict):
                for child in value.values():
                    plain(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    plain(child)

        plain(row)
    assert len(indexed) == units + 1
    assert sum("include/shared.h" in paths for paths in indexed) == 1


@pytest.mark.parametrize(
    ("spelling", "width"),
    [
        ("char", 1),
        ("s32", 4),
        ("f32", 4),
        ("int", 4),
        ("void *", 4),
        ("long", 4),
        ("long long", 8),
        ("Triple", 12),
        ("struct Pair", 8),
    ],
)
def test_sizeof_extents_use_project_abi(spelling, width):
    header = (
        "typedef int s32; typedef float f32; typedef struct { float x, y, z; } Triple; "
        "struct Pair { char c; void *p; }; "
    )
    ast = headers.parse(header + "struct X { char bytes[sizeof(" + spelling + ")]; int last; };", "h")
    row = types.record_layouts(ast)["struct X"]
    assert row["fields"][0]["width"] == width
    assert row["fields"][1]["offset"] == (width + 3) // 4 * 4


@pytest.mark.parametrize(
    ("bound", "offset"), [("sizeof(int) - 4", 4), ("sizeof(int) / 2", 8), ("010 - sizeof(void *)", 8)]
)
def test_sizeof_padding_including_zero_length_arrays(bound, offset):
    ast = headers.parse("struct X { int first; char pad[" + bound + "]; int last; };", "h")
    row = types.record_layouts(ast)["struct X"]
    assert row["fields"][-1]["offset"] == offset
    assert row["size"] == offset + 4


@pytest.mark.parametrize(
    ("bound", "reason"),
    [
        ("sizeof(void)", "unknown storage type"),
        ("sizeof(struct Missing)", "incomplete record"),
        ("n", "nonconstant array extent"),
        ("1 - sizeof(int)", "negative array extent"),
        ("sizeof(int) / 0", "division"),
    ],
)
def test_sizeof_does_not_invent_unknown_or_invalid_extents(bound, reason):
    ast = headers.parse("struct X { char bytes[" + bound + "]; };", "h")
    assert reason in types.record_layouts(ast)["struct X"]["conflict"]


@pytest.mark.parametrize(
    ("params", "setup", "operand"),
    [
        ("void *p", "", "((struct Holder *)p)->child"),
        ("struct Holder *p", "void *q = (void *)p->child;", "q"),
        ("struct Holder *p", "void *q; q = p->child;", "q"),
        ("struct Foo *p", "void *q;", "(q = (char *)p)"),
        ("struct Foo **p", "", "*p"),
        ("struct Foo **p", "", "p[1]"),
        ("struct Foo *p", "", "p + 1"),
    ],
)
def test_scoped_origins_preserve_field_loads_casts_assignments_and_indirection(params, setup, operand):
    header = HEADER + "struct Holder { struct Foo *child; };\n"
    raw = "int f(" + params + ") { " + setup + " return ((func_ABC_S1 *)(" + operand + "))->x; }\n"
    row = evidence(raw, header=header)
    assert all(c["origin"] and "conflict" not in c for c in row["casts"])
    result = recovery.derive(recovery.families([row])[0])
    assert not result["conflicts"]
    assert result["record"] == "struct Foo"
    writes = recovery.rewrite(result, {"src/a.c": raw.encode(), "include/shared.h": header.encode()})
    assert b"->value" in writes["src/a.c"]


@pytest.mark.parametrize("operand", ["get()", "(char *)p + 4", "*pp"])
def test_untyped_origins_are_traced_without_manufacturing_an_extent(operand):
    header = HEADER + "void *get(void);\n"
    raw = "int f(struct Foo *p, void **pp) { return ((func_ABC_S1 *)(" + operand + "))->x; }\n"
    row = evidence(raw, header=header)
    assert row["casts"][0]["origin"]
    result = recovery.derive(recovery.families([row])[0])
    assert result["record_changes"][0]["open"] and not [c for c in result["conflicts"] if "extent" in c["reason"]]


@pytest.mark.parametrize("allocator", ["malloc(sizeof(int))", "calloc(1, sizeof(int))"])
def test_allocation_extent_builds_real_record_and_preserves_pointer_storage(allocator):
    header = HEADER + "void *malloc(unsigned int); void *calloc(unsigned int, unsigned int);\n"
    raw = "int f(void) { void *p = " + allocator + "; return ((func_ABC_S1 *)p)->x; }\n"
    result = derived(raw, header=header)
    assert not result["conflicts"]
    assert result["record_changes"][0]["layout"]["size"] == 4
    writes = recovery.rewrite(result, {"src/a.c": raw.encode(), "include/shared.h": header.encode()})
    assert (result["record"] + " *p").encode() in writes["src/a.c"]
    assert b"&p" not in writes["src/a.c"]
    assert allocator.encode() in writes["src/a.c"]


@pytest.mark.parametrize("body", ["void *p = malloc(4); p = malloc(8);", "void *p = malloc(4); p = get();"])
def test_all_reaching_allocations_must_prove_one_extent(body):
    header = HEADER + "void *malloc(unsigned int); void *get(void);\n"
    result = derived("int f(int n) { " + body + " return ((func_ABC_S1 *)p)->x; }\n", header=header)
    assert any(c.get("reason") == "new record has no proven complete object extent" for c in result["conflicts"])


@pytest.mark.parametrize(
    ("allocation", "size", "reason"),
    [
        ("", 8, "copy exceeds proven complete object extent"),
        ("p = malloc(4);", 8, "copy exceeds proven complete object extent"),
    ],
)
def test_memcpy_sizes_are_bounds_and_detect_overruns(allocation, size, reason):
    header = HEADER + "void *malloc(unsigned int); void *memcpy(void *, void *, unsigned int);\n"
    raw = (
        "int f(void *p, void *q) { " + allocation + " memcpy(p, q, " + str(size) + "); "
        "return ((func_ABC_S1 *)p)->x; }\n"
    )
    result = derived(raw, header=header)
    assert any(c.get("reason") == reason for c in result["conflicts"])


@pytest.mark.parametrize(("width", "conflicted"), [(4, False), (2, True)])
def test_aggregate_assembly_width_uses_leaf_not_container(width, conflicted):
    header = (
        "struct Triple { int a, b, c; }; struct Foo { struct Triple value; }; "
        "typedef struct func_ABC_S1 { struct Triple x; } func_ABC_S1;\n"
    )
    (job,) = recovery.families([evidence("int f(struct Foo *p) { return ((func_ABC_S1 *)p)->x.a; }\n", header=header)])
    job["assembly"] = {"v": {"f": {"args": [(0, 0, width)]}}}
    result = recovery.derive(job)
    assert bool(result["conflicts"]) is conflicted


@pytest.mark.parametrize(("sizes", "conflicted"), [((4, 4), False), ((4, 8), True)])
def test_symbol_storage_extent_requires_versions_to_agree(cfg, sizes, conflicted):
    header = HEADER + "extern unsigned char blob[];\n"
    raw = "int f(void) { return ((func_ABC_S1 *)blob)->x; }\n"
    (job,) = recovery.families([evidence(raw, header=header)])
    member = Member(
        "blob",
        "data",
        "data",
        "g",
        tuple(Placement(v, ".data", 0, size, 0x80000400) for v, size in zip(("a", "b"), sizes, strict=True)),
    )
    snapshot = Snapshot(cfg, "head", LayoutMap({}, {"blob": member}, {}, "d", (), {}), {}, {}, "s")
    structs.enrich(snapshot, job, {"function": {}, "struct": {}}, {}, {}, [], {"blob": {}})
    result = recovery.derive(job)
    assert bool(result["conflicts"]) is conflicted
    if not conflicted:
        writes = recovery.rewrite(result, {"src/a.c": raw.encode(), "include/shared.h": header.encode()})
        assert b"&blob" in writes["src/a.c"]
        assert b"char blob[]" not in writes["include/shared.h"]


@pytest.mark.parametrize("confirmed", [False, True])
def test_byte_stride_needs_independent_assembly_evidence(cfg, confirmed):
    raw = "int f(char *p, int i) { return ((func_ABC_S1 *)(p + i * 4))->x; }\n"
    (job,) = recovery.families([evidence(raw)])
    snapshot = Snapshot(cfg, "head", LayoutMap({}, {}, {}, "d", (), {}), {}, {}, "s")
    usages = {"v": {"f": {"steps": [("a0", 4)] if confirmed else []}}}
    structs.enrich(snapshot, job, {"function": {}, "struct": {}}, {}, usages, [], {})
    result = recovery.derive(job)
    assert bool(result["conflicts"]) is not confirmed
    if confirmed:
        writes = recovery.rewrite(result, {"src/a.c": raw.encode(), "include/shared.h": HEADER.encode()})
        assert b"p + (i * 4)" in writes["src/a.c"]
        assert b"char *p" in writes["src/a.c"]


OPEN = (
    "typedef struct func_ABC_S1 { char pad0[8]; int f08; char pad0C[32]; float f2C; } func_ABC_S1;\n"
    "struct Foo { int value; };\n"
)


def test_open_extent_from_unobserved_param_root():
    raw = "float f(void *arg0) { ((func_ABC_S1 *)arg0)->f08 = 1; return ((func_ABC_S1 *)arg0)->f2C; }\n"
    result = derived(raw, header=OPEN)
    assert not result["conflicts"]
    (change,) = result["record_changes"]
    assert change["kind"] == "new" and change["open"] and change["layout"]["size"] == 0x30
    assert [f["offset"] for f in change["layout"]["fields"]] == [8, 0x2C]


def test_observed_disagreeing_extents_stay_conflict():
    header = OPEN + "extern unsigned char blob[8];\n"
    job = recovery.families([evidence("int f(void) { return ((func_ABC_S1 *)blob)->f08; }\n", header=header)])[0]
    job["extents"].append({"origin": "global:blob", "size": 16, "kind": "symbol"})
    result = recovery.derive(job)
    assert any("no proven complete object extent" in c["reason"] for c in result["conflicts"])


def test_uncorroborated_stride_is_conflict():
    raw = "int f(char *arg0, int i) { return ((func_ABC_S1 *)(arg0 + i * 12))->f08; }\n"
    result = derived(raw, header=OPEN)
    reasons = [c for c in result["conflicts"] if "no proven complete object extent" in c["reason"]]
    assert reasons and reasons[0]["detail"] == "observed extent is unproven"


def test_global_placement_size_without_brackets():
    member = Member("D_8014D080", "data", "data", "", (Placement("us", ".data", 0, 0x30, 0x8014D080),))
    snapshot = Snapshot(None, "h", LayoutMap({}, {"D_8014D080": member}, {}, "d", (), {}), {}, {}, "s")
    header = OPEN + "extern struct Foo D_8014D080;\n"
    row = evidence("int f(void) { return ((func_ABC_S1 *)&D_8014D080)->f08; }\n", header=header)
    job = recovery.families([row])[0]
    structs.enrich(snapshot, job, {"function": {}, "struct": {}}, {}, {}, [], {})
    assert {(e["kind"], e["size"]) for e in job["extents"]} == {("symbol", 0x30)}


def test_extension_past_end_of_open_record():
    header = "struct Foo { int a; };\n"
    use = {"offset": 8, "width": 4, "type": "int", "alignment": 4, "declaration": "int f08;", "field": "f08"}
    record = {
        "size": 4,
        "alignment": 4,
        "fields": [{"name": "a", "offset": 0, "width": 4, "alignment": 4}],
        "open": True,
    }
    owners = {"struct Foo": [{"path": "include/shared.h"}]}
    job = {"record_owners": owners, "accesses": [], "casts": [], "extents": []}
    field = structs._extension(job, "struct Foo", record, use, [])
    assert field["name"] == "f08" and record["size"] == 12
    closed = {**record, "open": False, "size": 4}
    assert structs._extension({**job, "extents": [{"kind": "declaration"}]}, "struct Foo", closed, use, []) is None
    assert header


def test_param_cast_family_retypes_param():
    header = "struct Real { int a; int b; };\n"
    raw = "int f(void *arg0) { return ((struct Real *)arg0)->b; }\n"
    jobs = recovery.families([evidence(raw, header=header)])
    result = recovery.derive(jobs[0])
    assert not result["conflicts"] and result["record"] == "struct Real"
    assert len(result["mapping"]) == 1 and len(result["declarations"]) == 1
    writes = recovery.rewrite(result, {"src/a.c": raw.encode()})
    assert writes["src/a.c"].decode() == "int f(struct Real *arg0) { return ((arg0))->b; }\n"


def test_prepare_key_ignores_unrelated_config(cfg):
    snapshot = Snapshot(cfg, "head", LayoutMap({}, {}, {}, "d", (), {}), {}, {}, "s")
    unit = UnitSpec("src/a.c", "c", "g", (), "t", {})
    recipe = Recipe("t", (), ("-O2",), (), "d")
    base = (snapshot, unit, [], recipe, "/root")
    other = replace(cfg, host=replace(cfg.host, workers=cfg.host.workers + 1), digest="other")
    moved = (replace(snapshot, config=other, tree=True, overlays={"x": b"1"}), unit, [], recipe, "/root")
    assert structs._prepare_key(base) == structs._prepare_key(moved)
    assert structs._prepare_key(base) != structs._prepare_key((snapshot, unit, [], recipe, "/other"))


def test_analysis_cached_whole(cfg, monkeypatch):
    snapshot = Snapshot(cfg, "head", LayoutMap({}, {}, {}, "d", (), {}), {}, {}, "s")
    calls = []
    monkeypatch.setattr(recovery, "_derive_all", lambda *a: calls.append(a) or ([], [], []))
    first = recovery._analysis(snapshot, "/root", {})
    assert recovery._analysis(snapshot, "/root", {}) == first and len(calls) == 1


def test_scan_batch_decodes_shared_fragments_once(monkeypatch):
    decoded, merged = [], []
    index = lambda part: ([], {}, {}, set(), {}, {}, {})  # noqa: E731
    monkeypatch.setattr(structs, "_decode", lambda job, key, part: decoded.append(part) or index(part))
    monkeypatch.setattr(structs, "_merge", lambda rows: merged.append(len(rows)) or index(None))
    monkeypatch.setattr(structs, "_HEAD", [None, {}, None])
    for key, parts in (("p1", ("a", "b")), ("p1", ("a", "b")), ("p2", ("a", "b", "c")), ("p3", ("a", "c"))):
        structs._head(("job",), key, parts)
    assert decoded == ["a", "b", "c"] and merged == [2, 3, 2]


def _fields(*rows):
    return {"fields": [{"name": n, "offset": o, "width": w, "type": t} for n, o, w, t in rows]}


def _graph_table(rows, reads=(), objects=None):
    spec = structs.graph((rows, objects or {}, reads))[1]
    return structs.identity_table(spec, [structs.classes_job(i) for i in structs.components(spec)], objects or {})


def _table(rows):
    return _graph_table(rows)


def _object(symbols, pairs=(), widths=None):
    order = sorted(symbols.values())
    accesses = {a: {"width": w} for a, w in (widths or {}).items()}
    return {
        "symbols": symbols,
        "order": order,
        "addresses": sorted(accesses),
        "accesses": accesses,
        "pairs": list(pairs),
    }


def test_identity_meeting_origin_morph_pair():
    left = "struct Morph { float a; float b; };\n"
    right = "struct Morph_func_802A2B58_de { float a; float b; };\n"
    rows = _two_callers(left, right)
    rows[0] = evidence(
        "int a(struct Morph *p) { generic(p); return 0; }\n",
        "src/a.c",
        rows[0]["declarations"] and left + right + HEADER + "int generic(void *);\n",
    )
    rows[1] = evidence(
        "int b(struct Morph_func_802A2B58_de *p) { generic(p); return 0; }\n",
        "src/b.c",
        left + right + HEADER + "int generic(void *);\n",
    )
    (row,) = _table(rows)
    assert row["canonical"] == "struct Morph" and len(row["members"]) == 2
    assert [f["name"] for f in row["fields"]] == ["a", "b"] and not row["conflicts"]


def test_identity_pass_through_shared_animinfo():
    header = (
        "struct Shared_AnimInfo { int a; float f; };\nstruct Info_func_80296F7C_S1 { int a; float f; };\n"
        "int forward(void *);\nint use(void *);\n"
    )
    rows = [
        evidence("int a(struct Shared_AnimInfo *p) { forward(p); return 0; }\n", "src/a.c", header),
        evidence("int b(struct Info_func_80296F7C_S1 *p) { forward(p); return 0; }\n", "src/b.c", header),
        evidence("int forward(void *p) { return use(p); }\n", "src/f.c", header),
        evidence("int use(void *p) { return ((func_ABC_S1 *)p)->x; }\n", "src/u.c", header + HEADER),
    ]
    (row,) = _table(rows)
    assert row["canonical"] == "struct Shared_AnimInfo"


def test_identity_screen_entries_cover_window():
    layouts = {
        "Screen42": _fields(("entries", 0, 8, "int [2]"), ("id", 8, 4, "int")),
        "Screen_func_8043B0CC_de": _fields(("window", 0, 4, "int"), ("slots", 4, 4, "int"), ("id", 8, 4, "int")),
    }
    (members,), lost = structs._classes({"x": set(layouts)}, layouts)
    assert set(members) == set(layouts) and not lost


def test_identity_entity_prefix():
    names = ["Actor_func_8024A7A0_de", "Object_func_8028B274_de", "Object_func_8028C934_de"]
    layouts = {n: _fields(("a", 8, 4, "int"), ("b", 0x18, 4, "float"), ("c", 0x50, 4, "int")) for n in names}
    layouts[names[2]] = _fields(
        ("a", 8, 4, "int"), ("b", 0x18, 4, "float"), ("c", 0x50, 4, "int"), ("d", 0x6C, 4, "int")
    )
    (members,), _ = structs._classes({"x": set(names)}, layouts)
    assert set(members) == set(names)


def test_base_register_joins_menu_globals():
    table = _object(
        {"D_80145040": 0x80145040, "D_80145088": 0x80145088, "D_801462C8": 0x801462C8},
        [("D_80145040", "D_801462C8"), ("D_80145088", "D_801462C8")],
    )
    (alias,) = structs._alias_anchors({"us": table})
    assert alias["members"] == {"D_80145040": 0, "D_80145088": 0x48, "D_801462C8": 0x1288}
    layouts = {
        "Game": _fields(("a", 0, 4, "int")),
        "Menu": _fields(("b", 0, 4, "int")),
        "Match": _fields(("c", 0, 4, "int")),
    }
    places = {"alias:D_80145040": {"Game": 0, "Menu": 0x48, "Match": 0x1288}}
    (members,), lost = structs._classes({"alias:D_80145040": set(layouts)}, layouts, places)
    assert members == {"Game": 0, "Menu": 0x48, "Match": 0x1288} and not lost


def test_base_register_negative_offset_runtime_state():
    table = _object({"D_80145088": 0x80145088, "D_801462E5": 0x801462E5}, [("D_80145088", "D_801462E5")])
    (alias,) = structs._alias_anchors({"us": table})
    assert alias["members"] == {"D_80145088": 0, "D_801462E5": 0x125D}


def test_separate_symbols_stay_separate():
    table = _object({"D_8011FE88": 0x8011FE88, "D_8011BDD4": 0x8011FE90, "D_8011BE14": 0x8011FEC8})
    assert structs._alias_anchors({"us": table}) == []
    assert structs._covers("D_8011FE88", 0x50, {"us": table}, {"D_8011FE88"}, {"us": by_address(table["symbols"])}) == [
        "D_8011BDD4",
        "D_8011BE14",
    ]


def test_helper_void_param_not_anchor():
    header = "".join(f"struct R{n} {{ int a; }};\n" for n in range(12)) + "int fwd(void *);\n"
    rows = [evidence(f"int a{n}(struct R{n} *p) {{ fwd(p); return 0; }}\n", f"src/a{n}.c", header) for n in range(12)]
    rows.append(evidence("int fwd(void *p) { return 0; }\n", "src/f.c", header))
    assert _table(rows) == []


def test_width_conflict_asm_decides():
    layouts = {
        "ActorMotion": _fields(("a", 0, 4, "int"), ("w", 0xAC, 4, "int")),
        "BoxContactRay": _fields(("a", 0, 4, "int"), ("w", 0xAC, 2, "short")),
    }
    classes, lost = structs._classes({"alias:x": set(layouts)}, layouts, {"alias:x": {}}, {"alias:x": {0xAC: 4}})
    assert classes == [] and [(c["record"], c["with"]) for c in lost] == [("BoxContactRay", "ActorMotion")]


def test_canonical_prefers_shared_named():
    owners = {"SharedPlayer": [{"path": "include/player.h"}], "Shared_ActorView": [{"path": "include/shared/actor.h"}]}
    members = {"SharedPlayer": 0, "Shared_ActorView": 0}
    canonical = structs._canonical(members, owners, {"SharedPlayer": 9})
    layouts = {
        "SharedPlayer": _fields(("a", 0, 4, "int")),
        "Shared_ActorView": _fields(("a", 0, 4, "int"), ("b", 4, 4, "int")),
    }
    assert canonical == "Shared_ActorView"
    assert [f["name"] for f in structs._flat(canonical, members, layouts, {})] == ["a", "b"]


def test_duplicate_record_rule_named_clone():
    clones = {"SteeringParticle": "Shared_Particle (include/shared/p.h)"}
    assert structs.typedebt.duplicate_records("struct SteeringParticle { int a; };\n", clones) == {
        1: "SteeringParticle duplicates Shared_Particle (include/shared/p.h)"
    }
    assert structs.typedebt.duplicate_records("struct Shared_Particle { int a; };\n", clones) == {}


def test_role_prefixed_function_names_are_views_not_identities():
    assert structs._is_view("struct Actor_func_8024A7A0_de") and not structs._is_view("struct Shared_Actor")


def test_gate_morph_void_pointer_vs_float_is_one_class_with_a_marked_conflict():
    common = (("count", 0, 4, "int"), ("blends", 4, 4, "int *"), ("first", 8, 4, "int *"))
    layouts = {
        "Morph": _fields(*common, ("frame", 0x10, 4, "void *")),
        "Morph_func_802A2B58_de": _fields(*common, ("scaleS", 0x10, 4, "float")),
    }
    (members,), _ = structs._classes({"x": set(layouts)}, layouts)
    flat = structs._flat("Morph_func_802A2B58_de", members, layouts, {})
    rows = {m: structs._map(layouts[m], 0, flat) for m in layouts}
    assert [r["conflict"] for r in rows["Morph"] if r["offset"] == 0x10] == ["type differs: void * vs float"]
    assert "conflict" not in rows["Morph_func_802A2B58_de"][0] and rows["Morph"][0]["canonical"] == "count"


def test_gate_assembly_reads_anchor_a_forwarder_no_c_body_reads():
    header = "struct Actor_func_8024A7A0_de { int a; float b; };\n"
    header += "struct Object_func_8028B274_de { int a; float b; };\nint fwd(void *);\n"
    rows = [
        evidence("int a(struct Actor_func_8024A7A0_de *p) { fwd(p); return 0; }\n", "src/a.c", header),
        evidence("int b(struct Object_func_8028B274_de *p) { fwd(p); return 0; }\n", "src/b.c", header),
        evidence("int fwd(void *p) { return 0; }\n", "src/f.c", header),
    ]
    assert _graph_table(rows) == []
    (row,) = _graph_table(rows, {"param:fwd:0"})
    assert len(row["members"]) == 2


def test_flat_layout_never_holds_a_union_typed_field():
    layouts = {"A": _fields(("a", 0, 4, "int"), ("u", 4, 4, "union Choice"))}
    assert [f["name"] for f in structs._flat("A", {"A": 0}, layouts, {})] == ["a"]


def test_anonymous_members_flatten_to_parent_offsets():
    ast = headers.parse(
        "struct M { int a; struct { short b; short c; }; union { float f; int i; }; int d; };\n"
        "struct Bad { int a : 3; };\n",
        "h",
    )
    rows = types.record_layouts(ast)
    got = {f["name"]: f["offset"] for f in rows["struct M"]["fields"]}
    assert got == {"a": 0, "b": 4, "c": 6, "f": 8, "i": 8, "d": 12} and rows["struct M"]["size"] == 16
    assert [f["name"] for f in rows["struct M"]["fields"] if f.get("arm")] == ["f", "i"]
    assert "bitfield" in rows["struct Bad"]["conflict"]
    arms = {"fields": [{"name": "x", "offset": 8, "width": 4, "type": "float"}]}
    assert structs._agree(rows["struct M"], arms) and structs._agree(rows["struct M"], _fields(("p", 8, 4, "int")))


def test_strongest_agreement_merges_first_not_alphabetical_order():
    layouts = {
        "Actor": _fields(("a", 0, 4, "int"), ("b", 8, 4, "float")),
        "Big": _fields(("a", 0, 4, "int"), ("c", 12, 4, "int")),
        "Object": _fields(("b", 8, 4, "float")),
    }
    classes, _ = structs._classes({"param:f:0": {"Actor", "Big", "Object"}}, layouts)
    assert classes == [{"Actor": 0, "Object": 0}]


def test_components_split_unrelated_groups():
    spec = {
        "groups": {"x": {"A", "B"}, "y": {"B", "C"}, "z": {"D", "E"}},
        "layouts": {n: _fields(("a", 0, 4, "int")) for n in "ABCDE"},
        "places": {},
        "asm": {},
    }
    parts = structs.components(spec)
    assert sorted(sorted(set().union(*p[0].values())) for p in parts) == [["A", "B", "C"], ["D", "E"]]


def test_split_moves_agreeing_part_when_both_remainders_stay_classes():
    layouts = {
        "A": _fields(("a", 0, 4, "int"), ("b", 4, 4, "int")),
        "B": _fields(("a", 0, 4, "int"), ("b", 4, 4, "int"), ("c", 8, 4, "int")),
        "C": _fields(("a", 0, 4, "int"), ("b", 4, 4, "int"), ("d", 12, 4, "int")),
        "X": _fields(("a", 0, 4, "int"), ("e", 8, 4, "float")),
        "Y": _fields(("e", 8, 4, "float"), ("c", 12, 4, "int")),
    }
    classes, _ = structs._classes({"g": {"A", "B", "C", "X", "Y"}}, layouts)
    assert all(len(c) > 1 for c in classes) and {n for c in classes for n in c} >= {"A", "B", "C"}


def test_hub_group_alone_gives_no_identity_but_two_hubs_do():
    layouts = {
        "Actor": _fields(("a", 0, 4, "int"), ("b", 8, 4, "float"), ("c", 12, 4, "int")),
        "Object": _fields(("a", 0, 4, "int"), ("b", 8, 4, "float")),
        "Panel": _fields(("b", 8, 4, "float"), ("c", 12, 4, "int")),
        **{f"N{i}": _fields((f"n{i}", 64 + 4 * i, 4, "int")) for i in range(10)},
    }
    filler = {f"N{i}" for i in range(10)}
    groups = {
        "param:h1:0": {"Actor", "Object"} | filler,
        "param:h2:0": {"Actor", "Object"} | filler - {"N0"},
        "param:h3:0": {"Actor", "Panel"} | filler,
    }
    classes, _ = structs._classes(groups, layouts)
    assert classes == [{"Actor": 0, "Object": 0}]
