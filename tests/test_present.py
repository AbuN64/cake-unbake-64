"""The files a site reads live: golden progress.json, memory map and symbol lines of a small fake project."""

import json
import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_report import _compiler_row, snapshot_factory  # noqa: F401 (fixtures)

from unbake import present, report
from unbake.contracts import Member, Placement, UnitSpec

GOLDEN = Path(__file__).parent / "golden" / "present"
SOURCES = {
    "src/a.c": b"void f_exact(void);\n\nvoid\nf_exact(void)\n{\n    f_exact();\n}\n",
    "src/fuzzy/f_draft.c": b"int f_draft(int a) {\n    return a;\n}\n",
}


def _function(name, rom, size, versions=("us", "eu")):
    return Member(
        name, "function", "asm", "g", tuple(Placement(v, ".text", rom, rom + size, 0x80000000) for v in versions)
    )


@pytest.fixture
def project(snapshot_factory, monkeypatch):  # noqa: F811
    monkeypatch.setattr(present.process, "git", Mock(side_effect=OSError("no repository")))
    members = [
        _function("f_exact", 0x1000, 0x40),
        _function("f_draft", 0x1040, 0x30),
        _function("f_asm", 0x1070, 0x4020),  # crosses the 16 KiB row edge at 0x4000
        _function("f_late", 0x20000, 0x100, ("us",)),  # rows between hold no code and are left out
        Member("d_table", "data", "data", "g", (Placement("us", ".data", 0x5090, 0x50A0, 0x80010000),)),
    ]
    fuzzy = {"f_draft": {"path": "src/fuzzy/f_draft.c", "scores": {"us": 0.5, "eu": 0.25}, "key": {}}}
    unit = UnitSpec("src/a.c", "c", "g", ("f_exact",), "test", {})
    snapshot = snapshot_factory(members, units=[unit], fuzzy=fuzzy, versions=("us", "eu"), overlays=SOURCES)
    files = dict(snapshot.config.project.version_files)
    files["us"] = replace(files["us"], meta={"cartridge_id": "NUS-NXXE-0"})
    segments = {"us": (("main", 0x1000, 0x5100, 0), ("ovl", 0x20000, 0x20100, 0)), "eu": (("main", 0x1000, 0x5100, 0),)}
    return replace(
        snapshot,
        config=replace(snapshot.config, project=replace(snapshot.config.project, version_files=files)),
        versions={v: replace(row, segments=segments[v]) for v, row in snapshot.versions.items()},
    )


def _golden(name, data, update=False):
    path = GOLDEN / name
    if update:  # flip once to regenerate after a deliberate change, then read the diff
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert data == path.read_bytes()


def test_files_match_the_golden_files(project):
    made = present.files(project, report.progress(project))
    assert set(made) == set(present.paths(("us", "eu")))
    for path, data in made.items():
        _golden(path.replace("/", "_"), data)


def test_progress_counts_partition_the_code_bytes(project):
    document = json.loads(present.progress(project, report.progress(project), updated="2026-10-15T00:00:00Z"))
    us, eu = document["carts"]
    assert (document["schema"], document["head"], document["updated"]) == (1, "b" * 40, "2026-10-15T00:00:00Z")
    assert (us["id"], us["version"], eu["id"]) == ("NUS-NXXE-0", "us", "eu")  # the version id without a cartridge
    assert us["functions"] == {"matched": 1, "total": 4} and us["toolchain"] == "test"
    assert us["bytes"] == {"matched": 0x40, "fuzzy": 0x30, "asm": 0x4020 + 0x100, "total": 0x40 + 0x30 + 0x4020 + 0x100}
    assert us["text_pct"] == report.progress(project)["versions"]["us"]["code_percent"]


def test_progress_keeps_its_bytes_while_the_numbers_hold(project):
    """head and updated name the tree a file was made from: a fresh setup of the same numbers never rewrites it."""
    current = report.progress(project)
    old = present.progress(project, current, updated="2020-01-01T00:00:00Z").replace(b"b" * 40, b"c" * 40)
    same = replace(project, overlays={**project.overlays, present.PROGRESS: old})
    assert present.progress(same, current) == old
    moved = {**current, "versions": {**current["versions"], "us": {**current["versions"]["us"], "code_matched": 1}}}
    assert json.loads(present.progress(same, moved))["head"] == "b" * 40


@pytest.mark.parametrize(
    "text, line",
    [
        ("int f(void);\nint\nf(void)\n{\n}\n", 2),  # the prototype is skipped, the return type line starts it
        ("static void f(int a,\n    int b) {\n}\n", 1),
        ("f(void) {\n}\n", 1),
        ("void g(void) {\n    f(1);\n}\n", 0),  # a call is no definition
        ("#define f(x) x\n", 0),
        ("\n\nglabel f\n", 3),
        ("x:\nf:\n", 2),
    ],
)
def test_the_line_a_function_starts_on(text, line):
    assert present._line(text, "f") == line


@pytest.mark.parametrize(
    "score, exact, step",
    [(0.0, False, 0), (0.01, False, 1), (0.5, False, 5), (0.99, False, 9), (1.0, False, 9), (0.0, True, 10)],
)
def test_step(score, exact, step):
    assert present.step(score, exact) == step


def test_the_map_splits_a_function_at_a_row_edge_and_titles_both_pieces(project):
    svg = present.memory_map(project, "us").decode()
    assert svg.count("<title>f_asm · 0% · 16416 B</title>") == 2
    assert svg.count("<g ") == 3  # 0x0000, 0x4000 and 0x20000: the rows between hold no code
    assert svg.count('class="h"') == 2  # one hairline per segment start
    assert "d_table" not in svg


def test_symbols_name_tracked_sources_only_one_per_line(project):
    data = present.symbols(project, "us")
    document = json.loads(data)
    functions = document["functions"]
    assert document["schema"] == 2 and set(functions) == {"f_draft", "f_exact"}  # f_asm, f_late: asm/ is untracked
    assert {k: functions[k]["path"] for k in functions} == {"f_draft": "src/fuzzy/f_draft.c", "f_exact": "src/a.c"}
    assert functions["f_exact"]["line"] == 3 and functions["f_exact"]["percent"] == 100.0
    words = functions["f_exact"]["words"]
    assert words[0] == words[1] > 0 and set(functions["f_draft"]) == {"path", "line", "percent", "words"}
    assert (
        0 <= functions["f_draft"]["percent"] < 100
        and functions["f_draft"]["words"][0] < functions["f_draft"]["words"][1]
    )
    assert data.count(b"\n") == 4


def test_report_json_has_one_unit_per_line():
    document = {
        "categories": [{"id": "a"}],
        "measures": {"m": 1},
        "units": [{"name": "u1"}, {"name": "u2"}],
        "version": 2,
    }
    data = report.dump_objdiff(document)
    assert json.loads(data) == document
    assert data.splitlines()[-3:] == [b'{"name":"u1"},', b'{"name":"u2"}', b"]}"]


def test_recomp_starter_names_the_title_only():
    assert json.loads(present.recomp("A Title")) == {
        "$schema": "https://recomp.fyi/schema/v1.json",
        "game": "A Title",
        "system": "N64",
        "type": "decomp",
    }


def test_the_overview_merges_adjacent_functions_of_one_status_and_leaves_out_empty_rows(project):
    svg = present.overview(project, "us").decode()
    # exact f_exact, fuzzy high f_draft, none f_asm and f_late (rows 0 and 2)
    assert svg.count("M") == 4  # one block each: f_exact, f_draft, f_asm and f_late
    assert svg.count('<path class="') == 3 and svg.count("<rect") == 4  # three tones in use, four legend blocks
    assert svg.count(">0x") == 2  # rows 0x000000 and 0x020000 only
    assert ">0 %<" in svg and ">100 %<" in svg and "fuzzy<" not in svg and "one row per" not in svg
    assert "<title>" not in svg and 'viewBox="0 0 720 ' in svg


@pytest.mark.parametrize("size, rows", [(1, 1), (0x10000, 1), (0x10001, 2), (0x30000, 3)])
def test_the_overview_row_count_fits_the_span(snapshot_factory, size, rows):  # noqa: F811
    snapshot = snapshot_factory([_function("only", 0x100000, size, ("us",))], versions=("us",))
    svg = present.overview(snapshot, "us").decode()
    assert svg.count(">0x") == rows


def test_the_readme_image_is_the_overview_not_the_detailed_map():
    assert present.overview_path("us") == "docs/overview-us.svg" and present.map_path("us") == "docs/memory-map-us.svg"


def test_the_overview_merges_touching_functions_of_one_status_into_one_block(snapshot_factory):  # noqa: F811
    members = [_function(n, 0x1000 + i * 0x40, 0x40, ("us",)) for i, n in enumerate("abc")]
    snapshot = snapshot_factory(members, versions=("us",))
    assert present.overview(snapshot, "us").decode().count("M") == 1


def test_every_tone_the_overview_uses_has_a_light_and_a_dark_rule(project):
    svg = present.overview(project, "us").decode()
    light, dark = svg.split("<style>")[1].split("</style>")[0].split("@media (prefers-color-scheme:dark){")
    used = set(re.findall(r'class="(q\d+)"', svg))
    assert used == {"q0", "q3", "q7", "q10"}  # the four tones, in the map and in the legend
    for tone in used:
        assert f".{tone}{{fill:" in light and f".{tone}{{fill:" in dark
    assert dark.count(".q0{fill:#4a463d}") == 1  # the dark none is not the light exact
