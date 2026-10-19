"""foreign: a NON_MATCHING body that is not the body of its function."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import fixture
import pytest

from unbake import foreign, land, symbols, tree
from unbake.contracts import Refusal

TEXT = """#include "x.h"
#ifdef NON_MATCHING
void other(void) {
}

void f(void) {
    g();
}
#else
INCLUDE_ASM("seg", f);
#endif
"""


@pytest.fixture(autouse=True)
def _compiler_made(monkeypatch):
    """Every target is compiler output unless a test says otherwise (the stub config has no assembly to read)."""
    monkeypatch.setattr(symbols, "handwritten", lambda config, version, name: False)


def _world(tmp_path, **module):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.c").write_text(TEXT)
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    row = {"path": "src/a.c", "fuzzy": ("f", "other"), "unnamed": {}}
    return config, [SimpleNamespace(**{**row, **module})]


def test_the_line_of_a_body_is_where_its_definition_opens_inside_the_block():
    assert foreign._line(TEXT, "f") == 6 and foreign._line(TEXT, "other") == 3


def test_a_name_with_no_body_in_a_block_has_no_line():
    assert foreign._line(TEXT, "missing") == 0


def test_the_blocks_are_those_fuzzy_only_switches():
    assert tree.non_matching_blocks(TEXT.splitlines(keepends=True)) == [(1, 8, 10)]


def test_a_body_scoring_under_the_floor_in_every_version_is_a_finding_with_file_line_and_score(tmp_path):
    config, modules = _world(tmp_path)
    scores = {"f": {"us": 0.02, "eu": 0.03}, "other": {"us": 0.8}}
    (finding,) = foreign.findings(config, modules, scores)
    assert (finding.key, finding.path, finding.line, finding.unit, finding.versions) == (
        "source.non_matching_foreign_body",
        "src/a.c",
        6,
        "f",
        ("eu",),
    )
    assert "3%" in finding.reason and finding.blocking


def test_a_body_that_scores_well_in_one_version_is_its_functions(tmp_path):
    config, modules = _world(tmp_path)
    assert foreign.findings(config, modules, {"f": {"us": 0.01, "eu": 0.6}}) == ()


def test_a_body_that_does_not_build_scores_nothing_and_is_not_called_foreign(tmp_path):
    config, modules = _world(tmp_path)
    assert foreign.findings(config, modules, {"f": {"us": 0.0}}) == ()


def test_an_unmeasured_body_is_no_finding(tmp_path):
    config, modules = _world(tmp_path)
    assert foreign.findings(config, modules, {}) == ()


def test_a_body_defining_a_function_no_symbol_names_is_a_finding_naming_the_versions(tmp_path):
    config, modules = _world(tmp_path, fuzzy=(), unnamed={"us": ("f",), "eu": ("f",)})
    (finding,) = foreign.findings(config, modules, {})
    assert (finding.unit, finding.line, finding.versions) == ("f", 6, ("us", "eu"))
    assert "not a symbol of us, eu" in finding.reason


def test_check_judges_the_bodies_of_the_given_paths_by_their_cached_scores(tmp_path, monkeypatch):
    config, modules = _world(tmp_path)
    monkeypatch.setattr(land.tree, "capture", lambda c: SimpleNamespace(digest="d", modules=modules))
    monkeypatch.setattr(land.report, "facts", lambda c: {})
    monkeypatch.setattr(land.rescore, "rows", lambda c, held, facts: {"f": {"us": 0.02}})
    (finding,) = land._bodies(config, ["src/a.c"])
    assert finding.unit == "f"
    assert land._bodies(config, ["src/other.c"]) == ()


def test_check_without_a_rom_names_it_and_does_not_refuse(tmp_path, monkeypatch):
    config, modules = _world(tmp_path)
    monkeypatch.setattr(land.tree, "capture", lambda c: SimpleNamespace(digest="d", modules=modules))
    monkeypatch.setattr(land.report, "facts", lambda c: {})

    def missing(c, held, facts):
        raise Refusal(foreign.Finding("config.file", "roms/baserom.us.z64 does not exist"))

    monkeypatch.setattr(land.rescore, "rows", missing)
    (finding,) = land._bodies(config, ["src/a.c"])
    assert finding.key == "config.file" and not finding.blocking


def test_a_body_whose_target_the_project_marks_handwritten_is_not_judged_by_its_score(tmp_path, monkeypatch):
    config, modules = _world(tmp_path)
    monkeypatch.setattr(symbols, "handwritten", lambda c, version, name: version == "us")
    (finding,) = foreign.findings(config, modules, {"f": {"us": 0.02, "eu": 0.03}})
    assert finding.versions == ("eu",)  # the handwritten version drops out, the compiler-made one is still judged
    assert foreign.findings(config, modules, {"f": {"us": 0.02}}) == ()
    assert foreign.findings(config, modules, {"f": {"us": 0.0}}) == ()


def test_the_project_marks_a_function_handwritten_by_the_comment_splat_writes_above_its_assembly(tmp_path, monkeypatch):
    monkeypatch.undo()  # the real reader
    config = fixture.config(tmp_path)
    asm = Path(config.project.root) / "asm/a/nonmatchings/ovl/fuzzy.s"
    asm.parent.mkdir(parents=True, exist_ok=True)
    asm.write_text("glabel fuzzy\n")
    assert not symbols.handwritten(config, "a", "fuzzy")
    asm.write_text('.include "macro.inc"\n\n/* Handwritten function */\nglabel fuzzy\n')
    assert symbols.handwritten(config, "a", "fuzzy")
    assert not symbols.handwritten(config, "b", "fuzzy")  # the other version has no assembly extracted
