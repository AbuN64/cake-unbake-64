"""present: only the README image is written, drawn from the report rows."""

from types import SimpleNamespace

from unbake import present, report


def _row(name, rom, code, exact, score, version="us", kind="function"):
    return report.Credit(
        name,
        report.Symbol(kind, "src/a.c", "c", rom),
        None,
        "main",
        version,
        code,
        0,
        code if exact else 0,
        exact,
        score,
    )


ROWS = [
    _row("f_exact", 0x1000, 0x40, True, 0.0),
    _row("f_high", 0x1040, 0x30, False, 0.75),
    _row("f_low", 0x1070, 0x4020, False, 0.25),
    _row("f_none", 0x20000, 0x100, False, 0.0),
    _row("f_other", 0x1000, 0x40, False, 0.0, version="eu"),
    _row("tbl", 0x5090, 0, False, 0.0, kind="data"),
]


def test_overview_is_the_rom_of_one_version_in_four_tones():
    svg = present.overview(ROWS, "us").decode()
    assert svg.startswith("<svg ") and svg.endswith("</svg>\n")
    for cls in ("q10", "q7", "q3", "q0"):
        assert f'<path class="{cls}"' in svg
    assert "0x000000" in svg and "0x020000" in svg and "0x010000" not in svg  # rows without code are left out


def test_overview_is_deterministic_and_reads_one_version():
    assert present.overview(ROWS, "us") == present.overview(list(reversed(ROWS)), "us")
    assert present.overview(ROWS, "us") != present.overview(ROWS, "eu")


def test_files_lists_only_the_overview_of_the_readme_map(monkeypatch):
    monkeypatch.setattr(report, "credits", lambda config, tree: ROWS)
    config = SimpleNamespace(project=SimpleNamespace(readme_map="us"))
    made = present.files(config, object())
    assert list(made) == ["docs/overview-us.svg"] == [present.overview_path("us")]
    assert made["docs/overview-us.svg"] == present.overview(ROWS, "us")
    assert present.files(SimpleNamespace(project=SimpleNamespace(readme_map=None)), object()) == {}


def test_nothing_else_is_written():
    for gone in ("memory_map", "symbols", "progress", "recomp", "map_path", "symbols_path", "paths", "PROGRESS"):
        assert not hasattr(present, gone)
