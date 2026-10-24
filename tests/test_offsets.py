"""Offset-named fields: the layout must put each where its name says."""

import subprocess
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from unbake import offsets, policy, tree
from unbake.contracts import Config, Project, Snapshot

RIGHT = """typedef struct Voice {
    s32 a;
    char pad4[0x4];
    s32 f_8;
    u8 mode;
    char pad_0D[0x3];
    s32 zero10;
} Voice;
"""
OFF_BY_FOUR = RIGHT.replace("pad_0D[0x3]", "pad_0D[0x7]")


def _sites(text, headers=None):
    code = policy.scopes(text)
    return offsets.sites(offsets.registry(headers or {}), code["code"], code["directives"])


def test_a_correct_record_has_no_site():
    assert _sites(RIGHT) == []


def test_a_record_whose_padding_is_off_by_four_names_each_later_field():
    found = _sites(OFF_BY_FOUR)
    assert [(s.line, s.record, s.field, s.named, s.actual) for s in found] == [(7, "Voice", "zero10", 0x10, 0x14)]
    assert found[0].subject == "Voice.zero10 is named for offset 0x10 but sits at 0x14"


def test_a_field_before_the_misplaced_one_is_not_blamed():
    assert {s.field for s in _sites(OFF_BY_FOUR)} == {"zero10"}


@pytest.mark.parametrize(
    ("name", "offset"),
    [
        ("f_A4", 0xA4),
        ("padB0", 0xB0),
        ("unk_10", 0x10),
        ("unk_0x001C", 0x1C),
        ("field_1C", 0x1C),
        ("flag20", 0x20),
        ("zeroAC", 0xAC),
        ("unknown_C", 0xC),
        ("off8", 0x8),
    ],
)
def test_names_that_state_an_offset(name, offset):
    assert offsets.named_offset(name) == offset


@pytest.mark.parametrize("name", ["padded", "flagged", "x_10", "state4", "count", "pad", "unk", "offset"])
def test_names_that_state_no_offset(name):
    assert offsets.named_offset(name) is None


def test_a_padding_array_may_be_named_for_its_start_the_field_before_it_or_where_it_ends():
    text = "struct S { s32 a; char pad0[0xC]; s32 f_10; u8 b; char pad_14[0x3]; s32 c; char pad_18[0x4]; };"
    assert _sites(text) == []


def test_a_counted_padding_array_is_no_offset():
    assert _sites("struct S { s32 a; char pad1[0x10]; s32 b; char pad2[0x10]; };") == []


def test_a_union_and_a_record_with_a_preprocessor_line_are_not_judged():
    assert _sites("union U { s32 a; s32 f_8; };") == []
    assert _sites("struct S {\n s32 a;\n#if X\n s32 b;\n#endif\n s32 f_8;\n};") == []


def test_a_name_out_of_order_with_the_named_fields_before_it_is_not_judged():
    assert _sites("struct S { s32 f_8; s32 f_4; };") == []


def test_a_local_record_reads_the_types_of_the_headers():
    headers = {"include/a.H": "typedef struct Pair { s32 x; s32 y; } Pair;"}
    text = "typedef struct Q { Pair p; s32 f_8; s32 f_C; s32 f_10; } Q;"
    assert [(s.field, s.actual) for s in _sites(text, headers)] == []
    text = "typedef struct Q { Pair p; s32 f_8; s32 f_10; s32 f_14; } Q;"
    assert [(s.field, s.actual) for s in _sites(text, headers)] == [("f_10", 0xC)]


def test_a_local_record_overrides_the_header_record_of_its_name():
    headers = {"include/a.H": "typedef struct Pair { s32 x; s32 y; s32 z; } Pair;"}
    text = "typedef struct Pair { s32 x; s32 f_4; } Pair;"
    assert _sites(text, headers) == []


def test_a_type_that_cannot_be_sized_ends_the_judging_there():
    assert _sites("struct S { s32 a; Unknown u; s32 f_40; };") == []


@pytest.fixture
def snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(policy.effort, "stage", Mock(side_effect=lambda name: nullcontext()))
    result = Mock(spec=Snapshot)
    result.config = Mock(spec=Config)
    result.config.project = Mock(spec=Project)
    result.config.project.root = tmp_path
    result.config.project.names_from = "a"
    result.config.project.libraries = ()
    result.config.project.digest = "project"
    result.overlays = {}
    result.versions = {}
    monkeypatch.setattr(policy, "_tree", lambda snap: tree.Tree("t", ()))
    monkeypatch.setattr(policy, "_include_headers", lambda snap: ["include/v.h"])
    result.peek = lambda path: b"typedef struct Head { s32 a; } Head;"
    result.digest = "snap"
    return result


def test_the_census_lists_each_site_with_its_file_and_line(snapshot):
    found = policy.evaluate(snapshot, "src/v.c", OFF_BY_FOUR, None, False)
    assert [(f.key, f.path, f.line) for f in found] == [("source.offset-name", "src/v.c", 7)]
    assert "Voice.zero10 is named for offset 0x10 but sits at 0x14" in found[0].reason


def _repo(tmp_path, monkeypatch, text):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    (tmp_path / "src").mkdir()
    (tmp_path / "src/v.c").write_text(text)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "base"]):
        subprocess.run([*git, *args], check=True)


def test_audit_refuses_a_record_the_tree_adds_off_by_four(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, RIGHT)
    (tmp_path / "src/v.c").write_text(OFF_BY_FOUR)
    with pytest.raises(policy.Refusal) as refused:
        policy.audit_command({"project": str(tmp_path), "against": "HEAD"})
    (finding,) = refused.value.findings
    assert finding.key == "source.offset-name"
    assert (finding.path, finding.line) == ("src/v.c", 7)


def test_audit_allows_a_correct_record_and_a_site_the_base_already_held(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, OFF_BY_FOUR)
    (tmp_path / "src/v.c").write_text("/* touched */\n" + OFF_BY_FOUR)
    assert policy.audit_command({"project": str(tmp_path), "against": "HEAD"})["counts_after"] == {
        "source.offset-name": 1
    }
    (tmp_path / "src/v.c").write_text(RIGHT)
    assert policy.audit_command({"project": str(tmp_path), "against": "HEAD"})["counts_after"] == {}


def test_audit_judges_a_source_by_the_header_the_change_edits(tmp_path, monkeypatch):
    (tmp_path / "include").mkdir()
    (tmp_path / "include/p.h").write_text("typedef struct Pair { s32 x; s32 y; } Pair;\n")
    _repo(tmp_path, monkeypatch, "typedef struct W { Pair p; s32 f_8; s32 f_C; } W;\n")
    (tmp_path / "include/p.h").write_text("typedef struct Pair { s32 x; s32 y; s32 z; } Pair;\n")
    with pytest.raises(policy.Refusal) as refused:
        policy.audit_command({"project": str(tmp_path), "against": "HEAD"})
    assert refused.value.findings[0].path == "src/v.c"
