import json
from pathlib import Path

import pytest

from unbake import lanecheck
from unbake.contracts import Refusal

DIFF = "diff --git a/src/A/a.c b/src/A/a.c\n--- a/src/A/a.c\n+++ b/src/A/a.c\n@@ -1 +1 @@\n-x\n+y\n"
OTHER = DIFF.replace("src/A/a.c", "src/B/b.c")
SHARED = DIFF.replace("src/A/a.c", "include/g.h")
RENAMED = "diff --git a/src/A/a.c b/src/B/b.c\nsimilarity index 90%\nrename from src/A/a.c\nrename to src/B/b.c\n"


@pytest.fixture
def names(tmp_path: Path) -> Path:
    folder = tmp_path / "files"
    folder.mkdir()
    for lane, held in {"P1": ["src/A/a.c"], "P2": ["src/B/b.c"], "GLOBALS": ["include/g.h"]}.items():
        (folder / f"{lane}.json").write_text(json.dumps({"lane": lane, "files": held}))
    return tmp_path


def diff(tmp_path: Path, text: str, name: str = "in.diff") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_hand_input_inside_its_lane_passes(names: Path, tmp_path: Path) -> None:
    got = lanecheck.check(names, "P1", [diff(tmp_path, DIFF)])
    assert got == {"lane": "P1", "hand": True, "files": ["src/A/a.c"]}


def test_a_file_of_another_lane_is_refused_naming_the_owner(names: Path, tmp_path: Path) -> None:
    with pytest.raises(Refusal) as refused:
        lanecheck.check(names, "P1", [diff(tmp_path, DIFF + OTHER)])
    (finding,) = refused.value.findings
    assert finding.key == "lane.outside" and finding.path == "src/B/b.c"
    assert "owned by P2, not by P1" in finding.reason


def test_a_globals_file_and_a_rename_across_lanes_are_refused(names: Path, tmp_path: Path) -> None:
    with pytest.raises(Refusal, match="include/g.h is owned by GLOBALS.*tool batches only"):
        lanecheck.check(names, "P1", [diff(tmp_path, SHARED)])
    with pytest.raises(Refusal, match="src/B/b.c is owned by P2"):
        lanecheck.check(names, "P1", [diff(tmp_path, RENAMED)])


def test_a_lane_without_a_file_set_is_a_tool_batch_and_a_file_in_no_set_is_nobodys(names: Path, tmp_path: Path) -> None:
    assert lanecheck.check(names, "NAMES", [diff(tmp_path, OTHER + SHARED)])["hand"] is False
    free = DIFF.replace("src/A/a.c", "Makefile")
    assert lanecheck.check(names, "P1", [diff(tmp_path, free)])["files"] == ["Makefile"]


def test_the_command_reads_params(names: Path, tmp_path: Path) -> None:
    params = {"names": str(names), "lane": "P2", "diff": [str(diff(tmp_path, OTHER))]}
    assert lanecheck.lane_command(params)["hand"] is True
