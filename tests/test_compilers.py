"""compilers: object fingerprints, catalog shapes, the smallest directory rows, dead rows and the command surface."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from unbake import cli, compilers
from unbake import config as configuration
from unbake.contracts import Config, FileFacts, Refusal

FILLED = """\
00000000 <filled>:
   0:	27bdffe8 	addiu	$29,$29,-24
   4:	afbf0014 	sw	$31,20($29)
   8:	0c000000 	jal	8 <filled+0x8>
			8: R_MIPS_26	callee
   c:	00000000 	sll	$0,$0,0
  10:	8fbf0014 	lw	$31,20($29)
  14:	03e00008 	jr	$31
  18:	27bd0018 	addiu	$29,$29,24
"""
UNFILLED = """\
00000000 <unfilled>:
   0:	27bdffe8 	addiu	$29,$29,-24
   2:	afbf0014 	sw	$31,20($29)
   4:	3c020000 	lui	$2,0x0
			4: R_MIPS_HI16	D_80000000
   8:	8c430000 	lw	$3,0($2)
			8: R_MIPS_LO16	D_80000000
   c:	27bd0018 	addiu	$29,$29,24
  10:	03e00008 	jr	$31
  14:	00000000 	sll	$0,$0,0
"""
MACRO = """\
00000000 <macro>:
   0:	3c010000 	lui	$1,0x0
			0: R_MIPS_HI16	D_80000000
   4:	8c220000 	lw	$2,0($1)
			4: R_MIPS_LO16	D_80000000
   8:	03e00008 	jr	$31
   c:	46006306 	mov.s	$f12,$f1
"""


def test_parse_splits_functions_and_keeps_relocations() -> None:
    functions = compilers.parse(FILLED + UNFILLED)
    assert list(functions) == ["filled", "unfilled"]
    assert functions["filled"][2][2] == "R_MIPS_26" and functions["filled"][2][3] == "jal T"
    assert functions["unfilled"][2][2] == "R_MIPS_HI16"


def test_a_branch_inside_the_function_is_named_by_offset_a_call_by_nothing() -> None:
    text = "00000000 <f>:\n   0:\t10400003 \tbeqz\t$2,10 <f+0x10>\n   4:\t0c000000 \tjal\t0 <other>\n"
    ins = compilers.parse(text)["f"]
    assert ins[0][3] == "beqz $2,L+0x10" and ins[1][3] == "jal T"


def test_filled_and_unfilled_epilogues_are_told_apart() -> None:
    filled = compilers.features(compilers.parse(FILLED)["filled"])
    unfilled = compilers.features(compilers.parse(UNFILLED)["unfilled"])
    assert (filled["epi_filled"], filled["epi_unfilled"]) == (1, 0)
    assert (unfilled["epi_filled"], unfilled["epi_unfilled"]) == (0, 1)
    assert compilers.shape(filled) == "F-" and compilers.shape(unfilled) == "US"


def test_address_formation_and_odd_fprs_and_the_save_area() -> None:
    macro = compilers.features(compilers.parse(MACRO)["macro"])
    assert macro["at_hilo"] == 1 and macro["fp_odd_s"] == 1 and compilers.shape(macro) == "-M+f64"
    filled = compilers.features(compilers.parse(FILLED)["filled"])
    assert filled["save_top"] == 1 and filled["save_low"] == 0 and filled["pro_sched"] == 0


def test_shapes_use_the_catalog_vocabulary_and_allows_checks_them() -> None:
    seen = compilers.shapes(compilers.features(compilers.parse(UNFILLED)["unfilled"]))
    assert seen["epilogue"] == {"unfilled"} and seen["addresses"] == {"split"} and seen["save_area"] == {"top"}
    rows = configuration.toolchains()
    allowing = sorted(t for t, r in rows.items() if "fingerprint" in r and compilers.allows(r["fingerprint"], seen))
    assert allowing and all(rows[t]["fingerprint"]["epilogue"] == "unfilled" for t in allowing)
    odd = {"fp": {"fp64"}}
    assert not compilers.allows({"epilogue": "filled", "addresses": "macro", "save_area": "top", "fp": ["fp32"]}, odd)


def test_every_catalog_row_declares_its_shapes_and_probe_flags_it_supports() -> None:
    for name, row in configuration.toolchains().items():
        assert "fingerprint" in row, name
        assert {"epilogue", "addresses", "save_area", "fp"} <= row["fingerprint"].keys()
        for probe in row.get("probe_flags", ()):
            assert set(probe) <= set(row["supported_options"]), name


def test_smallest_rows_make_a_directory_row_when_it_saves_rows() -> None:
    exact = {
        "src/IO/a.c": frozenset({"B"}),
        "src/IO/b.c": frozenset({"B"}),
        "src/IO/c.c": frozenset({"A", "B"}),
        "src/IO/d.c": frozenset({"A"}),
        "src/IO/f.c": frozenset({"B"}),
        "src/G/e.c": frozenset({"A"}),
    }
    assert compilers.smallest_rows(exact, "A", {}) == [("src/IO/", "B"), ("src/IO/d.c", "A")]


def test_smallest_rows_keep_the_inherited_tag_when_a_directory_row_saves_nothing() -> None:
    exact = {"src/IO/a.c": frozenset({"B"}), "src/IO/d.c": frozenset({"A"}), "src/G/e.c": frozenset({"A", "B"})}
    assert compilers.smallest_rows(exact, "A", {}) == [("src/IO/a.c", "B")]


def test_smallest_rows_prefer_a_units_current_tag_for_an_exception() -> None:
    exact = {"src/a.c": frozenset({"B", "C"}), "src/b.c": frozenset({"A"})}
    assert compilers.smallest_rows(exact, "A", {"src/a.c": "C"}) == [("src/a.c", "C")]
    assert compilers.smallest_rows(exact, "A", {}) == [("src/a.c", "B")]


def test_smallest_rows_leave_a_default_project_with_no_rows() -> None:
    assert compilers.smallest_rows({"src/a.c": frozenset({"A"})}, "A", {}) == []
    assert compilers.smallest_rows({}, "A", {}) == []


def _project(**rows: FileFacts) -> Config:
    project = SimpleNamespace(toolchain="gcc-test", files=rows, libraries=())
    return SimpleNamespace(project=project)  # type: ignore[return-value]


def test_a_file_row_equal_to_its_directory_row_is_dead() -> None:
    cfg = _project(
        **{"src/IO/": FileFacts("gcc-x"), "src/IO/a.c": FileFacts("gcc-x"), "src/IO/b.c": FileFacts("gcc-y")}
    )
    assert compilers.dead_rows(cfg) == ["src/IO/a.c"]


def test_a_row_that_gives_the_project_default_is_dead_and_flags_make_it_live() -> None:
    cfg = _project(
        **{"src/a.c": FileFacts("gcc-test"), "src/b.c": FileFacts(None, ("-mfp32",)), "src/c.c": FileFacts()}
    )
    assert compilers.dead_rows(cfg) == ["src/a.c", "src/c.c"]


def test_the_directory_below_decides_not_the_one_above() -> None:
    cfg = _project(
        **{"src/": FileFacts("gcc-x"), "src/IO/": FileFacts("gcc-test"), "src/IO/a.c": FileFacts("gcc-test")}
    )
    assert compilers.dead_rows(cfg) == ["src/IO/a.c"]


def test_proposal_reports_rows_dead_rows_disagreements_and_drafts(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    def cand(*lines: int | None) -> dict[str, Any]:
        return {"fns": {f"f{i}": n for i, n in enumerate(lines)}}

    census = [
        {"k": "cand", "label": "gcc-test", "toolchain": "gcc-test", "add": [], "omit": []},
        {"k": "cand", "label": "gcc-test -x", "toolchain": "gcc-test", "add": ["-x"], "omit": []},
        {
            "k": "unit",
            "version": "a",
            "unit": "src/a.c",
            "nfn": 1,
            "cands": {"gcc-test": cand(0), "gcc-test -x": cand(4)},
            "drafts": {"fns": ["f9"], "cands": {"gcc-test": {"fns": {"f9": 3}}, "gcc-test -x": {"fns": {"f9": 0}}}},
        },
        {
            "k": "unit",
            "version": "a",
            "unit": "src/b.c",
            "nfn": 1,
            "cands": {"gcc-test": cand(2), "gcc-test -x": cand(0)},
        },
    ]
    shapes = {"epilogue": [], "addresses": [], "save_area": [], "fp": []}
    prints = [
        {"k": "unit", "unit": u, "functions": 1, "classes": {"F-": 1}, "shapes": shapes} for u in ("src/a.c", "src/b.c")
    ]
    monkeypatch.setattr(compilers, "_current_label", lambda config, unit: "gcc-test")
    result = compilers.proposal(cfg, "a", prints, census)
    assert [(r["path"], r["tag"]) for r in result["rows"]] == [("src/b.c", "gcc-test -x")]
    assert result["judged"] == 2 and result["exact_current"] == 1 and result["unresolved"] == []
    assert result["rows"][0]["add"] == ["-x"] and result["rows"][0]["kind"] == "file"
    assert result["drafts"] == [{"unit": "src/a.c", "fn": "f9", "exact_under": ["gcc-test -x"], "exact_now": False}]


def test_the_result_validates_against_its_schema() -> None:
    document = {
        "action": "propose",
        "version": "a",
        "judged": 1,
        "rows": [{"path": "src/", "kind": "dir", "tag": "t", "toolchain": "t", "add": [], "omit": [], "covers": 1}],
        "dead_rows": [],
        "notes": [],
    }
    configuration.validate("result.compilers", json.loads(json.dumps(document)), "compilers")
    with pytest.raises(Refusal):
        configuration.validate("result.compilers", {"action": "nothing", "version": "a"}, "compilers")


def test_the_command_reaches_its_handler_and_an_unknown_action_is_refused(cfg: Config) -> None:
    assert configuration.load_resource("commands.toml")["command"]["compilers"]["handler"] == "compilers.run"
    with pytest.raises(Refusal) as refusal:
        compilers.run(cfg, {"action": "bogus"})
    assert refusal.value.findings[0].key == "compilers.input"
    assert cli.main(["compilers", "--help"]) in (0, None)


def test_fingerprint_without_a_build_is_refused(cfg: Config) -> None:
    with pytest.raises(Refusal) as refusal:
        compilers.fingerprint(cfg, {"version": "a"})
    assert refusal.value.findings[0].key == "compilers.input"


def test_propose_without_a_census_is_refused(cfg: Config) -> None:
    with pytest.raises(Refusal) as refusal:
        compilers.propose(cfg, {"version": "a"})
    assert refusal.value.findings[0].key == "compilers.input"
