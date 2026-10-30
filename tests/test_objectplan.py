"""Judging targets by the object the linker built: slices of the real refused modules and the rules on plain atoms."""

from __future__ import annotations

from pathlib import Path

import pytest

from unbake import folders, objectplan
from unbake.objectplan import Atom

DATA = Path(__file__).parent / "data" / "objectplan"
FILES = DATA / "FILES.tsv"


@pytest.fixture(scope="module")
def verdicts() -> dict[tuple[str, str], dict[str, str]]:
    rows = objectplan.judge(objectplan.read_dir(FILES))
    return {(r["version"], r["target"].rsplit("/", 1)[-1]): r for r in rows}


@pytest.mark.parametrize("version", ["us", "us-rev1"])
def test_setn64_of_six_objects_merges_as_one_run(verdicts, version: str) -> None:
    row = verdicts[(version, "SETN64.CPP")]
    assert row["verdict"] == "merge"
    assert "objects 119,120,121,122,123,124" in row["reason"] or version == "us-rev1"


@pytest.mark.parametrize("version", ["de", "eu"])
def test_setn64_with_another_files_object_between_settles(verdicts, version: str) -> None:
    row = verdicts[(version, "SETN64.CPP")]
    if version == "eu":
        assert row["verdict"] == "settle"
        assert ".game_code run [113, 114, 115] keeps the target, run [117] is not adjacent" in row["reason"]
    else:
        assert row["verdict"] == "merge"


def test_playerai_merges_in_every_version_that_has_it(verdicts) -> None:
    assert verdicts[("de", "PLAYERAI.CPP")]["verdict"] == "absent"
    assert {verdicts[(v, "PLAYERAI.CPP")]["verdict"] for v in ("eu", "us")} == {"merge"}


def test_menu_merges_with_its_unnamed_bytes_and_fill_measured(verdicts) -> None:
    for version in ("de", "eu"):
        row = verdicts[(version, "MENU.CPP")]
        assert row["verdict"] == "merge"
        assert "B pad absorbed" in row["reason"] and "B unnamed inside objects" in row["reason"]
    assert verdicts[("us", "MENU.CPP")]["verdict"] == "absent"


def test_every_row_of_the_slice_is_covered_once(verdicts) -> None:
    assert len(verdicts) == 3 * 5


def atom(address: int, size: int, kind: str = "data", obj: int = 1, linkage: str = "extern", target: str = "T") -> Atom:
    return Atom(address, size, kind, linkage, obj, target, f"s{address:X}")


def test_hole_between_two_objects_over_one_alignment_refuses() -> None:
    findings, *_ = objectplan.lay_out([atom(0x100, 16, obj=1), atom(0x130, 8, obj=2)], ".game_data")
    assert findings == ["gap: 32 B of no atom between objects 1 and 2 before s130 at 00000130"]


def test_hole_between_two_objects_under_one_alignment_is_fill() -> None:
    findings, _, fill, unnamed = objectplan.lay_out([atom(0x101, 1, obj=1), atom(0x108, 8, obj=2)], ".game_data")
    assert (findings, fill, unnamed) == ([], 6, 0)


def test_hole_inside_one_object_is_its_unnamed_bytes() -> None:
    findings, _, fill, unnamed = objectplan.lay_out([atom(0x100, 16), atom(0x150, 8)], ".game_data")
    assert (findings, fill, unnamed) == ([], 0, 64)


def test_pad_atoms_are_absorbed_but_a_pad_of_real_bytes_is_not() -> None:
    pad = atom(0x110, 8, "pad", linkage="drop")
    assert (
        objectplan.lay_out([atom(0x100, 16, "function", 1), pad, atom(0x118, 4, "function", 2)], ".game_code")[1] == 8
    )
    real = atom(0x110, 32, "pad", linkage="extern")
    assert objectplan.lay_out([atom(0x100, 16, "function"), real], ".game_code")[0]


def test_function_gap_and_overlap_refuse() -> None:
    assert objectplan.lay_out([atom(0x100, 16, "function"), atom(0x114, 4, "function")], ".game_code")[0]
    assert objectplan.lay_out([atom(0x100, 16, "function"), atom(0x108, 4, "function")], ".game_code")[0]


def test_function_typed_literal_pool_in_a_data_segment_is_data() -> None:
    pool = atom(0x800E2BEF, 1356, "function")
    assert not objectplan.is_code(pool, ".game_data") and objectplan.is_code(pool, ".game_code")
    assert objectplan.lay_out([pool], ".game_data")[0] == []


def test_object_code_spanning_two_code_segments_refuses() -> None:
    files = "target\t" + "\t".join(f"obj_{v}" for v in objectplan.VERSIONS) + "\nT\t-\t-\t-\t1\t-\n"
    objects = (
        "version\tobject\ttarget\tsegment\tstart\tend\n"
        "us\t1\tT\t.game_code\t100\t110\nus\t1\tT\t.ovl_frontendMenu\t200\t210\n"
    )
    head = "version\tsymbol\tkind\tsegment\taddress\tsize\tobject\ttarget\tlinkage\n"
    assign = (
        head
        + "us\ta\tfunction\t.game_code\t100\t16\t1\tT\textern\n"
        + "us\tb\tfunction\t.ovl_frontendMenu\t200\t16\t1\tT\textern\n"
    )
    row = next(r for r in objectplan.judge(objectplan.read_plan(files, objects, assign)) if r["verdict"] != "absent")
    assert (
        row["verdict"] == "refuse"
        and "split: the code of one object spans .game_code and .ovl_frontendMenu" in row["reason"]
    )


def test_plan_disagreeing_with_objects_refuses() -> None:
    files = "target\t" + "\t".join(f"obj_{v}" for v in objectplan.VERSIONS) + "\nT\t-\t-\t-\t1,2\t-\n"
    objects = "version\tobject\ttarget\tsegment\tstart\tend\nus\t1\tT\t.game_code\t100\t110\n"
    head = "version\tsymbol\tkind\tsegment\taddress\tsize\tobject\ttarget\tlinkage\n"
    plan = objectplan.read_plan(files, objects, head + "us\ta\tfunction\t.game_code\t100\t16\t1\tT\textern\n")
    row = next(r for r in objectplan.judge(plan) if r["verdict"] != "absent")
    assert row["verdict"] == "refuse" and row["reason"].startswith("plan: FILES.tsv names objects [1, 2]")


def test_missing_header_column_is_refused() -> None:
    with pytest.raises(ValueError, match=r"FILES\.tsv needs a header row"):
        objectplan.read_plan("x\n", "", "")


def test_mergecheck_action_writes_the_tsv(tmp_path: Path) -> None:
    out = tmp_path / "mergecheck.tsv"
    result = folders.mergecheck(None, {"plan": str(FILES), "out": str(out)})  # type: ignore[arg-type]
    assert result["targets"] == 15 and result["refused"] == [] and result["verdicts"]["settle"] == 2
    lines = out.read_text().splitlines()
    assert lines[0] == "version\ttarget\tverdict\treason" and len(lines) == 16


def test_mergecheck_action_refuses_a_plan_without_its_siblings(tmp_path: Path) -> None:
    (tmp_path / "FILES.tsv").write_text(FILES.read_text())
    with pytest.raises(folders.Refusal):
        folders.mergecheck(None, {"plan": str(tmp_path / "FILES.tsv")})  # type: ignore[arg-type]
