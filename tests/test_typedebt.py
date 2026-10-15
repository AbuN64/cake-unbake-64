"""Type-debt counters: one positive and one negative case per rule, and the staged ratchet."""

from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock

import pytest

from unbake import policy
from unbake.contracts import Config, Finding, Project, Snapshot
from unbake.families import typedebt

PATH = "src/unit.c"
CASES = {
    "fake-type": ("struct func_80123456_us_S1 { int x; };\n", "struct Real { int x; };\n"),
    "pad-view": ("struct V { int a; char pad4[28]; };\n", "struct W { int a; int b; int c; int d; char pad[4]; };\n"),
    "placeholder-record": (
        "typedef struct Rec28f0fe70 { s32 unk0; s32 unk4; } Rec28f0fe70;\nint f(Rec28f0fe70 *p);\n",
        "typedef struct Real { u8 unk0; s32 count; } Real;\nint f(Real *p);\n",
    ),
    "param-cast": (
        "int f(void *p) {\n  return ((struct Foo *)p)->x;\n}\n",
        "int f(void *p) {\n  return *(u8 *)p;\n}\nint g(struct Foo *p) {\n  return ((struct Bar *)p)->x;\n}\n",
    ),
    "address-int": (
        "void f(void) {\n  g((s32)&D_80100000, 0);\n}\n",
        "void f(void) {\n  g(&D_80100000, sizeof(u32) & 3, (u32)&((T *)0)->field, a && b, (x) & 4);\n}\n",
    ),
}


@pytest.fixture
def snapshot(monkeypatch):
    monkeypatch.setattr(policy.effort, "stage", Mock(side_effect=lambda name: nullcontext()))
    result = Mock(spec=Snapshot)
    result.config = Mock(spec=Config)
    result.config.project = Mock(spec=Project)
    result.config.project.library_dirs = ()
    result.config.project.version_differences = ()
    result.layout = Mock(units={}, members={})
    return result


def _debt(snapshot, text):
    return [(f.key, f.line, f.blocking) for f in policy.evaluate(snapshot, PATH, text, None, False)]


@pytest.mark.parametrize("rule", CASES)
def test_each_rule_counts_its_case_and_not_the_clean_one(snapshot, rule):
    bad, good = CASES[rule]
    assert _debt(snapshot, bad)[0][:1] == (f"source.{rule}",) and len(_debt(snapshot, bad)) == 1
    assert _debt(snapshot, good) == []


def test_a_repeated_name_counts_once_and_blocks(snapshot):
    text = CASES["fake-type"][0] + "int f(struct func_80123456_us_S1 *p);\n"
    assert _debt(snapshot, text) == [("source.fake-type", 1, True)]


def test_pad_view_needs_half_the_bytes_in_pads_and_three_names(snapshot):
    assert _debt(snapshot, "struct A { int a; int b; int c; int d; char pad[16]; };\n") == []  # four names
    assert _debt(snapshot, "struct B { int a; int b; char pad[8]; };\n")[0][0] == "source.pad-view"  # exactly half


def test_offset_named_arrays_with_hex_sizes_are_padding(snapshot):
    rec = "typedef struct Rec28f0fe70 { u8 unk0[0x1C]; u8 unk1C; } Rec28f0fe70;\n"
    assert _debt(snapshot, rec)[0][0] == "source.pad-view"
    assert _debt(snapshot, "struct F { u8 field_0[0x10]; s32 a; };\n")[0][0] == "source.pad-view"
    assert _debt(snapshot, "struct N { u8 name[0x1C]; u8 kind; };\n") == []  # a real name is not padding


def test_a_placeholder_record_is_unclean_in_source_but_not_a_header(snapshot):
    text = "struct R { s32 unk0; s32 field_4; };\nint f(struct R *p);\n"
    assert _debt(snapshot, text)[0][0] == "source.placeholder-record"
    assert policy.evaluate(snapshot, "include/r.h", text, None, False) == ()
    assert _debt(snapshot, "struct R { s32 unk0; s32 count; };\nint f(struct R *p);\n") == []
    assert _debt(snapshot, "struct R { s32 unk0; s32 unk4; };\n") == []  # defined, never used


def test_the_ratchet_refuses_an_increase_and_accepts_equal_or_lower():
    old = [Finding("source.fake-type", "x", path=PATH, line=n) for n in (1, 2)]
    one = Finding("source.fake-type", "x", path=PATH, line=9)
    assert policy.scope(old, old, {PATH})[0] == ()
    assert policy.scope(old, old[:1], {PATH})[0] == ()
    blocking, debt = policy.scope(old, (*old, one), {PATH})
    assert blocking == (replace(one, blocking=True),) and len(debt) == 2
    assert policy.scope(old, (*old, one), {"src/other.c"})[0] == ()  # only written files are judged


def test_counterless_debt_blocks_new_row(snapshot):
    snapshot.peek = Mock(return_value=b"")
    found = policy.evaluate(snapshot, PATH, CASES["fake-type"][0], None, False)
    assert policy.fresh(snapshot, PATH, found, False) == found
    snapshot.peek = Mock(return_value=CASES["fake-type"][0].encode())
    assert policy.fresh(snapshot, PATH, found, False) == ()


def test_a_record_named_after_a_function_symbol_is_fake_and_the_function_itself_is_not(snapshot):
    for named in ("Actor_func_802A274C_de", "View_func_8023B3F8_de"):
        assert _debt(snapshot, f"typedef struct {{ int x; }} {named};\n")[0][0] == "source.fake-type"
    assert _debt(snapshot, "struct func_802A274C_de { int x; };\n")[0][0] == "source.fake-type"
    assert _debt(snapshot, "int f(void) {\n  return func_802A274C_de(1);\n}\n") == []
    assert (
        _debt(snapshot, "extern u8 D_auto_func_802BB3D0_de_padding[];\nu8 data_func_802A274C_de_800C5CD0[4];\n") == []
    )
    assert _debt(snapshot, "void f(Track_func_802A274C_de *t) {}\n")[0][0] == "source.fake-type"


def test_view_types_are_unions_and_pad_views_by_tag_and_typedef_name_and_uses_skip_definitions():
    from unbake.families import typedebt

    header = (
        "typedef union { int a; float b; } Word;\nstruct Pads { int a; char pad4[28]; };\n"
        "typedef struct Real { int a; int b; int c; int d; } Alias;\n"
    )
    views = typedebt.view_types(header)
    assert views == {"Word", "Pads"}
    uses = typedebt.view_uses("void f(Word *w);\nvoid g(struct Pads *p);\nstruct Pads {\n};\n", views)
    assert uses == {1: "Word", 2: "Pads"}


def test_a_unit_naming_a_header_view_record_is_measured_not_unclean(snapshot):
    unit = Mock(path=PATH)
    snapshot.read = Mock(return_value=b"typedef union { int a; float b; } Word;\n")
    found = policy._view_uses(snapshot, unit, "void f(Word *w) {}\n", ["include/w.h"])
    assert [(f.key, f.line, f.path) for f in found] == [("source.uses-view-type", 1, PATH)]
    assert "uses-view-type" not in policy.config.load_resource("rules.toml")["unclean"]
    assert policy._view_uses(snapshot, unit, "void f(int *w) {}\n", ["include/w.h"]) == ()


OWNER = (
    "typedef struct Data80135834Ptr {\n    u8 unk_0[0x8];\n    u16 unk8;\n    u8 unk_A[0x16];\n    s32 unk20;\n"
    "} Data80135834Ptr;\n"
)


@pytest.mark.parametrize(
    ("declared", "found"),
    [
        ("extern Data80135834Ptr *D_auto_80135834;\n", False),  # the symbol's own record: its pads are evidence
        ("", True),  # no symbol declared as it: a view like any other
    ],
)
def test_pad_view_spares_a_data_symbols_owner_local_record_only(declared, found):
    code = OWNER + declared
    assert bool(typedebt.pad_view(code)) is found
    assert ("Data80135834Ptr" in typedebt.view_types(code)) is found
    assert bool(typedebt.pad_view(code.replace("Data80135834Ptr", "Shape"))) is True
