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
    monkeypatch.setattr(report, "layers", lambda config, tree, version: {})
    config = SimpleNamespace(project=SimpleNamespace(readme_map="us"))
    made = present.files(config, object())
    assert list(made) == ["docs/overview-us.svg"] == [present.overview_path("us")]
    assert made["docs/overview-us.svg"] == present.overview(ROWS, "us")
    assert present.files(SimpleNamespace(project=SimpleNamespace(readme_map=None)), object()) == {}


def test_nothing_else_is_written():
    for gone in ("memory_map", "symbols", "progress", "recomp", "map_path", "symbols_path", "paths", "PROGRESS"):
        assert not hasattr(present, gone)


def test_overview_draws_platform_spans_and_labels_their_layer():
    svg = present.overview([], "us", {"platform": [(0, 4096), (0x20000, 0x20100)]}).decode()
    assert '<path class="p1"' in svg and ">platform</text>" in svg
    assert "0x000000" in svg and "0x020000" in svg


def test_overview_legend_is_one_strip_in_the_site_palette():
    layers = {"decoded": [(0, 256)], "identified": [(256, 512)], "platform": [(0x10000, 0x10100)]}
    svg = present.overview([], "us", layers).decode()
    for yellow in ("#ecdcaa", "#b98b1d", "#e3c27a", "#b87a14", "#7a6232", "#d9a441"):
        assert yellow not in svg
    assert ".b1{fill:#bfdcd6}.b2{fill:#3f8f84}" in svg and ".b1{fill:#2f4f4a}.b2{fill:#6fb8ac}" in svg
    assert "prefers-color-scheme:dark" in svg
    swatches = [m for m in svg.split("<rect")[1:]]
    assert len({m.split('y="')[1].split('"')[0] for m in swatches}) == 1  # every swatch on the one row
    for name in ("identified", "decoded", "platform", "0 %", "100 %"):
        assert f">{name}</text>" in svg
    plain = present.overview([], "us", {}).decode()
    assert "identified" not in plain and "platform" not in plain


def test_overview_collapses_runs_of_uniform_rows_into_one_labeled_band():
    row = present.OVERVIEW_ROW
    layers = {"decoded": [(5 * row, 9 * row)], "identified": [(10 * row, 11 * row)]}
    mixed = SimpleNamespace(version="us", member=SimpleNamespace(rom=0), code=64, data=0, score=1.0, exact=True)
    svg = present.overview([mixed], "us", layers).decode()
    assert ">0x050000\u20130x080000</text>" in svg  # four uniform rows, one band
    assert "0x060000</text>" not in svg and "0x000000</text>" in svg and "0x0A0000</text>" in svg  # one row stays a row
    assert 'viewBox="0 0 720 ' + str(3 * present.OVERVIEW_PITCH + 6 + 11) + '"' in svg


def test_overview_bands_rows_whose_only_breaks_are_seams_between_blobs_of_one_class():
    row = present.OVERVIEW_ROW
    seam = present.OVERVIEW_SEAM
    layers = {"decoded": [(0, row // 2), (row // 2 + seam, 3 * row), (3 * row + 4096, 4 * row)]}
    svg = present.overview([], "us", layers).decode()
    assert ">0x000000\u20130x020000</text>" in svg  # the seam does not split the band
    assert "0x030000</text>" in svg  # a 4 KiB hole is a real gap: that row stays mixed


def _luminance(color: str) -> float:
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_blob_tones_keep_their_label_text_readable_in_both_modes():
    # label ink on each blob fill (WCAG AA 4.5): dark ink on teal in light mode, light ink on identified in dark mode
    assert _contrast("#bfdcd6", "#1b1a17") >= 4.5 and _contrast("#3f8f84", "#1b1a17") >= 4.5
    assert _contrast("#2f4f4a", "#f0ead8") >= 4.5 and _contrast("#6fb8ac", "#1b1a17") >= 4.5


def test_a_decoded_band_label_is_not_the_colour_of_its_dark_mode_fill():
    svg = present.overview([], "us", {"decoded": [(0, 2 * present.OVERVIEW_ROW)]}).decode()
    assert ">0x000000" in svg and ".bt.b2{fill:#1b1a17}" in svg.split("prefers-color-scheme:dark")[-1]
