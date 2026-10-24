"""Host overrides must not hide inside matching C or behind a file-scope macro."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from unbake import land, naming_baseline, non_matching
from unbake.contracts import Refusal

PATH = "src/input.c"
DIRECT = "void Read(void) {\n#ifdef NON_MATCHING\nwait(0);\n#else\nwait(1);\n#endif\n}\n"
MACRO = "#ifdef NON_MATCHING\n#define WAIT 0\n#else\n#define WAIT 1\n#endif\nvoid Read(void) { wait(WAIT); }\n"
FALLBACK = '#ifdef NON_MATCHING\nvoid Read(void) { wait(0); }\n#else\nINCLUDE_ASM("code", Read);\n#endif\n'


@pytest.mark.parametrize("text", [DIRECT, MACRO, MACRO.replace("wait(WAIT)", "wait(ALIAS)") + "#define ALIAS WAIT\n"])
def test_override_inside_c_and_file_scope_macro_are_refused(text):
    (finding,) = non_matching.findings(PATH, text, ("us", "eu"))
    assert finding.key == non_matching.KEY and finding.unit == "Read" and finding.blocking
    assert finding.versions == ("us", "eu")
    assert finding.line == (2 if text == DIRECT else 1)
    assert "shim" in finding.action


@pytest.mark.parametrize(
    "guard",
    ["#ifdef NON_MATCHING", "#if defined(NON_MATCHING)", "#if NON_MATCHING", "#if 1 && \\\n defined(NON_MATCHING)"],
)
def test_guard_spellings_cannot_bypass_check(guard):
    assert non_matching.findings(PATH, DIRECT.replace("#ifdef NON_MATCHING", guard))


def test_inverted_guard_and_elif_are_one_conditional_group():
    inverted = DIRECT.replace("#ifdef NON_MATCHING", "#ifndef NON_MATCHING")
    assert len(non_matching.findings(PATH, inverted)) == 1
    text = DIRECT.replace("#ifdef NON_MATCHING", "#if 0\nwait(9);\n#elif defined(NON_MATCHING)")
    assert len(non_matching.scan(PATH, text)) == 1
    assert len(non_matching.findings(PATH, text)) == 1


def test_full_function_assembly_fallback_is_allowed():
    assert non_matching.findings(PATH, FALLBACK) == ()
    (row,) = non_matching.scan(PATH, FALLBACK, ("us",))
    assert row["class"] == "A"
    assert row["functions"]["Read"]["A"] == ["us"]


def test_version_alias_and_assembly_in_another_module_are_fallbacks():
    text = "#ifdef NON_MATCHING\n#define Read ReadUS\nvoid Read(void) {}\n#endif\n"
    (row,) = non_matching.scan(PATH, text, ("us",), {"us": {"ReadUS"}})
    assert row["class"] == "A" and row["functions"]["ReadUS"]["A"] == ["us"]


def test_nested_override_inside_assembly_fallback_is_not_class_b():
    text = "#ifdef NON_MATCHING\n" + DIRECT + '\n#else\nINCLUDE_ASM("code", Read);\n#endif\n'
    rows = non_matching.scan(PATH, text)
    assert len(rows) == 2 and all(row["class"] == "A" for row in rows)
    assert non_matching.findings(PATH, text) == ()


def test_versions_distinguish_matching_c_from_fallback():
    text = (
        "#if defined(VERSION_US) || defined(NON_MATCHING)\n" + DIRECT + '\n#else\nINCLUDE_ASM("code", Read);\n#endif\n'
    )
    rows = non_matching.scan(PATH, text, ("us", "eu"))
    assert all(row["functions"]["Read"]["B"] == ["us"] for row in rows)
    assert all(row["functions"]["Read"]["A"] == ["eu"] for row in rows)


def test_comments_strings_dead_branches_identical_code_and_unused_macros_are_other():
    text = """/* #ifdef NON_MATCHING */
const char *name = "#ifdef NON_MATCHING";
#if 0 && defined(NON_MATCHING)
void Read(void) { wait(0); }
#endif
#ifdef NON_MATCHING
#define UNUSED 0
#endif
void Other(void) {
#ifdef NON_MATCHING
wait(1);
#else
wait(1);
#endif
}
"""
    assert len(non_matching.scan(PATH, text)) == 3
    assert non_matching.findings(PATH, text) == ()


def test_string_changes_are_not_masked_out_of_comparison():
    text = DIRECT.replace("wait(0)", 'wait("host")').replace("wait(1)", 'wait("rom")')
    assert non_matching.findings(PATH, text)


def _baseline(root, text=MACRO):
    document = {"schema": 1, "revision": "a" * 40, "files": naming_baseline.by_file(non_matching.findings(PATH, text))}
    target = root / non_matching.PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document))
    return document


def test_baseline_allows_only_reviewed_content_occurrences_and_paths(tmp_path):
    _baseline(tmp_path)
    found, _ = non_matching.check(tmp_path, {PATH: "\n\n" + MACRO})
    assert len(found) == 1 and not found[0].blocking
    for files in ({PATH: MACRO.replace("WAIT 0", "WAIT 2")}, {"src/new.c": MACRO}, {PATH: MACRO + MACRO}):
        found, _ = non_matching.check(tmp_path, files)
        assert any(f.blocking for f in found)


def test_missing_and_malformed_baselines_do_not_exempt_debt(tmp_path):
    found, baseline = non_matching.check(tmp_path, {PATH: DIRECT})
    assert baseline is None and found[0].blocking
    target = tmp_path / non_matching.PATH
    target.parent.mkdir()
    for content in ("{", '{"schema": 1, "files": {}}'):
        target.write_text(content)
        with pytest.raises(Refusal):
            non_matching.check(tmp_path, {PATH: DIRECT})


def _world(tmp_path, monkeypatch, text=MACRO):
    root = tmp_path
    (root / PATH).parent.mkdir(parents=True)
    (root / PATH).write_text(text)
    config = SimpleNamespace(project=SimpleNamespace(root=root, versions=("us",)))
    held = SimpleNamespace(modules=[SimpleNamespace(path=PATH, owns={"us": ("Read",)}, stand_ins={})])
    monkeypatch.setattr(land.tree, "capture", lambda c: held)
    monkeypatch.setattr(land.api, "snapshot", lambda c: SimpleNamespace(commit="revision"))
    monkeypatch.setattr(land.policy, "evaluate", lambda *a: ())
    monkeypatch.setattr(land.report, "record", Mock())
    bodies = Mock(return_value=())
    monkeypatch.setattr(land, "_bodies", bodies)
    return config, bodies


def test_public_check_refuses_override_before_any_scoring_or_build(tmp_path, monkeypatch):
    config, bodies = _world(tmp_path, monkeypatch)
    with pytest.raises(Refusal) as caught:
        land.check_command(config, {"paths": [PATH]})
    assert caught.value.findings[0].key == non_matching.KEY
    bodies.assert_not_called()


def test_full_check_shrinks_debt_and_reintroduction_is_refused(tmp_path, monkeypatch):
    config, _ = _world(tmp_path, monkeypatch, "void Read(void) { wait(1); }\n")
    _baseline(tmp_path)
    land.check_command(config, {})
    assert non_matching.read(tmp_path)["files"] == {}
    (tmp_path / PATH).write_text(MACRO)
    with pytest.raises(Refusal):
        land.check_command(config, {})


def test_partial_and_failed_check_never_rewrite_baseline(tmp_path, monkeypatch):
    config, _ = _world(tmp_path, monkeypatch, "void Read(void) {}\n")
    original = _baseline(tmp_path)
    land.check_command(config, {"paths": [PATH]})
    assert non_matching.read(tmp_path) == original
    (tmp_path / PATH).write_text(MACRO.replace("WAIT 0", "WAIT 2"))
    with pytest.raises(Refusal):
        land.check_command(config, {})
    assert non_matching.read(tmp_path) == original


def test_current_baseline_debt_is_reported_and_strict_still_refuses(tmp_path, monkeypatch):
    config, _ = _world(tmp_path, monkeypatch)
    _baseline(tmp_path)
    result = land.check_command(config, {})
    assert result["counts"]
    with pytest.raises(Refusal) as caught:
        land.check_command(config, {"strict": True})
    assert caught.value.findings[0].key == "check.debt"


def test_included_original_c_is_an_override_but_assembly_is_a_fallback():
    text = '#ifdef NON_MATCHING\nvoid Read(void) {}\n#else\n#include "original.inc.c"\n#endif\n'
    (row,) = non_matching.scan(PATH, text, ("us",), matching={"us": {"Read"}})
    assert row["class"] == "B"
    (row,) = non_matching.scan(PATH, text, ("us",), assembled={"us": {"Read"}})
    assert row["class"] == "A"


def test_public_check_reads_included_c_ownership(tmp_path, monkeypatch):
    text = '#ifdef NON_MATCHING\nvoid Read(void) {}\n#else\n#include "original.inc.c"\n#endif\n'
    config, bodies = _world(tmp_path, monkeypatch, text)
    with pytest.raises(Refusal) as caught:
        land.check_command(config, {"paths": [PATH]})
    assert caught.value.findings[0].key == non_matching.KEY
    bodies.assert_not_called()
