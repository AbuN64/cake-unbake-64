"""Earlier-build windows: expansion of a bin segment, the separately loaded symbol file, the splat and link output
around a window, blob-row retirement and the partition that counts window bytes as source."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from unbake import blobs, build, store, symbols, versions, windows
from unbake.contracts import Refusal
from unbake.tree import Tree

pytestmark = pytest.mark.usefixtures("plain_roms")

DELTA = 0x80000000
EU_DELTA = 0x7FFFD400
ROWS = [
    {"name": "main", "rom": 0x1000, "vram": 0x80200400},
    {"name": "dbg", "rom": 0x2000, "kind": "bin", "earlier_build": [{"rom": 0x2400, "end": 0x2480, "delta": DELTA}]},
    {"name": "tail", "rom": 0x3000, "kind": "bin"},
]
EARLIER = b"gOld = 0x80002400; // rom:0x2400 size:0x40\ngNext = 0x80002440; // rom:0x2440 size:0x40\n"


def make(tmp_path: Path, rows=ROWS, earlier: bytes | None = EARLIER) -> SimpleNamespace:
    (tmp_path / "versions/us").mkdir(parents=True, exist_ok=True)
    (tmp_path / "versions/us/symbol_addrs.txt").write_bytes(b"func_a = 0x80200400; // type:func rom:0x1000 size:0x40\n")
    if earlier is not None:
        (tmp_path / windows.symbols_path("us")).write_bytes(earlier)
    segments, problems = windows.expand("us", rows, 0x4000)
    assert not problems
    version = SimpleNamespace(size=0x4000, segments=segments)
    project = SimpleNamespace(
        root=tmp_path,
        name="Game",
        title="Game",
        toolchain="gcc-2.8.1-sn64",
        versions=("us",),
        version={"us": version},
        files={},
        cppflags=(),
        cflags=(),
        asflags=(),
    )
    return SimpleNamespace(project=project, host=SimpleNamespace(workers=2))


def test_vram_is_given_or_a_delta_from_the_rom():
    assert windows.vram_of({"rom": 0x2400, "end": 0x2480, "vram": 0x80123400}) == 0x80123400
    assert windows.vram_of({"rom": 0x15C380, "end": 0x15C3E0, "delta": EU_DELTA}) == 0x80159780


def test_a_bin_segment_splits_into_bin_window_and_bin():
    segments, problems = windows.expand("us", ROWS, 0x4000)
    assert problems == []
    assert [(s.name, s.rom, s.kind, s.vram, s.address_space, s.window_of) for s in segments] == [
        ("main", 0x1000, "code", 0x80200400, "main", None),
        ("dbg", 0x2000, "bin", None, "main", None),
        ("dbg_earlier_2400", 0x2400, "code", 0x80002400, "earlier_build", "dbg"),
        ("dbg", 0x2480, "bin", None, "main", None),
        ("tail", 0x3000, "bin", None, "main", None),
    ]
    assert windows.spans(segments, 0x4000) == [(0x2400, 0x2480)]


def test_windows_at_the_edges_leave_no_empty_bin():
    rows = [
        {
            "name": "dbg",
            "rom": 0x2000,
            "kind": "bin",
            "earlier_build": [{"rom": 0x2000, "end": 0x3000, "vram": 0x80100000}],
        }
    ]
    segments, _ = windows.expand("us", rows, 0x3000)
    assert [(s.name, s.kind) for s in segments] == [("dbg_earlier_2000", "code")]


@pytest.mark.parametrize(
    "window, reason",
    [
        ({"rom": 0x2400, "end": 0x2480}, "exactly one of vram and delta"),
        ({"rom": 0x2400, "end": 0x2480, "vram": 1, "delta": 1}, "exactly one of vram and delta"),
        ({"rom": 0x2480, "end": 0x2400, "vram": 0x80002400}, "outside the segment"),
        ({"rom": 0x2F00, "end": 0x3100, "vram": 0x80002400}, "outside the segment"),
        ({"rom": 0x2401, "end": 0x2480, "vram": 0x80002400}, "word aligned"),
        ({"rom": 0x2400, "end": 0x2480, "vram": 0x80002402}, "word aligned"),
        ({"rom": 0x2400, "end": 0x2480, "vram": 0x80002400, "name": "tail"}, "is taken"),
    ],
)
def test_a_malformed_window_names_its_problem(window, reason):
    rows = [{**ROWS[1], "earlier_build": [window]}, ROWS[2]]
    _, problems = windows.expand("us", rows, 0x4000)
    assert len(problems) == 1 and reason in problems[0]


def test_overlapping_windows_are_refused():
    held = [{"rom": 0x2400, "end": 0x2480, "vram": 0x80002400}, {"rom": 0x2440, "end": 0x24C0, "vram": 0x80001000}]
    _, problems = windows.expand("us", [{**ROWS[1], "earlier_build": held}], 0x4000)
    assert len(problems) == 1 and "overlaps the window before it" in problems[0]


def test_a_window_on_a_code_segment_is_refused():
    _, problems = windows.expand(
        "us", [{**ROWS[0], "earlier_build": [{"rom": 0x1100, "end": 0x1200, "vram": 4}]}], 0x4000
    )
    assert "belongs to a bin segment" in problems[0]


def test_earlier_symbols_load_apart_and_join_the_table(tmp_path):
    config = make(tmp_path)
    table = symbols.load(config, "us")
    assert table["func_a"][0] == 0x80200400 and table["gOld"] == (0x80002400, 0x2400, 0x40, "data")
    rows = {r.name: r for r in versions.symbol_addrs(config, "us")}
    assert set(rows) == {"func_a", "gOld", "gNext"}
    assert versions.symbol_addrs(make(tmp_path / "x", ROWS[:1], None), "us")[0].name == "func_a"


def test_a_name_at_two_addresses_across_the_files_is_refused():
    with pytest.raises(Refusal) as error:
        symbols.merged("gOld = 0x80300000;\n", "", (), earlier="gOld = 0x80002400;\n")
    assert error.value.findings[0].key == "symbols.conflict"
    assert symbols.merged("gOld = 0x80300000;\n", "", (), earlier="gOld = 0x80300000;\n")["gOld"][0] == 0x80300000


def test_a_declared_window_needs_its_symbol_file(tmp_path):
    config = make(tmp_path / "a", earlier=None)
    for call in (lambda: symbols.load(config, "us"), lambda: versions.symbol_addrs(config, "us")):
        with pytest.raises(Refusal) as error:
            call()
        assert error.value.findings[0].key == "symbols.earlier"


def test_the_link_provides_earlier_symbols_at_the_older_addresses(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "cached", lambda _c, _k, _key, produce: produce())
    script = versions.symbols_ld(make(tmp_path), "us").decode()
    assert "PROVIDE(gOld = 0x80002400);" in script and "PROVIDE(func_a = 0x00200400);" in script


def test_splat_reads_both_symbol_files_and_cuts_the_bin_around_the_window(tmp_path):
    config = make(tmp_path)
    yaml = build.splat_yaml(config, "us", Tree("d", (), {"us": symbols.load(config, "us")})).decode()
    assert (
        '  symbol_addrs_path:\n    - "versions/us/symbol_addrs.txt"\n    - "versions/us/earlier_build_symbols.txt"'
        in yaml
    )
    assert "  - name: dbg\n    type: bin\n    start: 0x2000" in yaml
    assert "  - name: dbg_earlier_2400\n    type: code\n    start: 0x2400\n    vram: 0x80002400" in yaml
    assert (
        'exclusive_ram_id: "earlier_build"' in yaml and "  - name: dbg_2480\n    type: bin\n    start: 0x2480" in yaml
    )


def test_the_link_places_the_window_at_its_vram_inside_the_bin(tmp_path):
    script = build.linker(make(tmp_path), Tree("d", (), {})).decode()
    assert ".dbg_earlier_2400 0x80002400 (INFO) : AT(0x2400)" in script
    assert ".dbg 0x00002000 : AT(0x2000)" in script and ".dbg_2480 0x00002480 : AT(0x2480)" in script
    assert 'ASSERT(SIZEOF(.dbg_earlier_2400) <= 0x80, "ROM extent: dbg_earlier_2400")' in script


DOC = {
    "schema": 1,
    "blob": {
        "copy": {
            "type": "leftover",
            "format": "copy",
            "state": "decoded",
            "rom": {"us": [[0x2380, 0x2500]], "eu": [[0x2380, 0x2500]]},
        },
        "inside": {"type": "leftover", "format": "copy", "state": "decoded", "rom": {"us": [[0x2400, 0x2480]]}},
        "child": {"type": "model", "state": "decoded", "parent": "outer", "decoded": [0, 4]},
        "outer": {"type": "model", "state": "decoded", "rom": {"us": [[0x5000, 0x5100]]}},
    },
}


def test_blob_rows_lose_the_bytes_a_window_types():
    after = blobs.retire(DOC, {"us": [(0x2400, 0x2480)]})
    assert after["blob"]["copy"]["rom"] == {"us": [[0x2380, 0x2400], [0x2480, 0x2500]], "eu": [[0x2380, 0x2500]]}
    assert "inside" not in after["blob"] and after["blob"]["child"] == DOC["blob"]["child"]
    assert DOC["blob"]["inside"]["rom"] == {"us": [[0x2400, 0x2480]]}  # the input is not changed


def test_a_retired_row_with_children_refuses():
    with pytest.raises(Refusal) as error:
        blobs.retire(DOC, {"us": [(0x5000, 0x5100)]})
    assert error.value.findings[0].key == "blobs.window"


def test_retired_is_the_file_that_says_what_load_sees(tmp_path, monkeypatch):
    config = make(tmp_path)
    document = {"doc": DOC}
    monkeypatch.setattr(blobs, "_read", lambda c: b"x")
    monkeypatch.setattr(blobs, "parse", lambda c, data: document["doc"])
    text = blobs.retired(config)
    assert text is not None and b"inside" not in text and b"0x2380, 0x2400" in text
    document["doc"] = blobs.retire(DOC, {"us": [(0x2400, 0x2480)]})
    assert blobs.retired(config) is None


def test_load_retires_the_rows_under_a_window(tmp_path):
    config = make(tmp_path)
    config.project.root = tmp_path
    (tmp_path / "config").mkdir()
    (tmp_path / "config/blobs.toml").write_bytes(blobs.emit(DOC))
    config.project.version["eu"] = config.project.version["us"]
    document = blobs.load(config)
    assert "inside" not in document["blob"] and document["blob"]["copy"]["rom"]["us"] == [
        [0x2380, 0x2400],
        [0x2480, 0x2500],
    ]


def test_window_bytes_count_as_source_and_a_row_under_them_is_not_counted_twice():
    spans = [(0x2400, 0x2480)]
    partition = blobs.sweep(0x4000, [(0x1000, 0x1040)], spans, [], [(0x2380, 0x2500)], [])
    assert (partition["data"], partition["code"], partition["overlap"]) == (0x80, 0x40, 0)
    assert partition["blobs_decoded"] == 0x180 - 0x80 and partition["shadowed"] == 0x80
    retired = blobs.retire(DOC, {"us": spans})
    decoded, _ = blobs.ranges(retired, "us")
    after = blobs.sweep(0x4000, [(0x1000, 0x1040)], spans, [], decoded, [])
    assert after["shadowed"] == 0 and after["data"] == 0x80 and after["unknown"] == partition["unknown"]
