"""Object edges: the 16-byte .text rule per version on mocked ROMs, the cross-version merge and its census."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

from unbake import edges

JR, NOP, OP = 0x03E00008, 0, 0x27BDFFE8


def _rom(tmp_path, name, functions, size=0x2000):
    """A ROM holding `functions` [(name, rom_start, words)]; each maps to vram 0x80000000 + rom."""
    rom = bytearray(size)
    for _, start, words in functions:
        rom[start : start + 4 * len(words)] = struct.pack(f">{len(words)}I", *words)
    path = tmp_path / f"{name}.z64"
    path.write_bytes(bytes(rom))
    return path


def _rows(functions, section=".text", seg=0):
    return tuple((n, section, s, s + 4 * len(w), 0x80000000 + s, seg) for n, s, w in functions)


@pytest.mark.parametrize(
    ("gap", "vram", "same", "want"),
    [
        (4, 0x80001010, True, "pad"),
        (0, 0x8000101C, True, "unal"),
        (0, 0x80001020, True, "al0"),
        (16, 0x80001020, True, "gap"),
        (4, 0x8000101C, True, "gap"),  # padding before an unaligned start: no object rule explains it
        (0, 0x80001020, False, "seg"),
    ],
)
def test_classify_is_the_16_byte_rule(gap, vram, same, want):
    assert edges.classify(gap, vram, same) == want


def test_a_version_job_classifies_each_pair_from_the_rom_words(tmp_path):
    functions = [
        ("a", 0x1000, [OP, JR, NOP, NOP]),  # a padding word after the delay slot: the next object starts at 0x1010
        ("b", 0x1010, [OP, JR, NOP]),  # ends 0x101C
        ("c", 0x101C, [OP, OP, JR, NOP, OP]),  # unaligned and adjacent: same object; ends 0x1030
        ("d", 0x1030, [OP, JR, OP, OP]),  # aligned, no padding: open; ends 0x1040
        ("e", 0x1040, [OP, JR, NOP]),  # ends 0x104C, then 4 unmapped zero bytes
        ("f", 0x1050, [OP, JR, NOP]),
    ]
    rom = _rom(tmp_path, "v", functions)
    pairs, data, first = edges._version_job(("sha", str(rom), _rows(functions)))
    assert pairs == (("a", "b", "pad"), ("b", "c", "unal"), ("c", "d", "al0"), ("d", "e", "al0"), ("e", "f", "pad"))
    assert (data, first) == ((), "a")


def test_a_segment_change_and_a_long_gap_are_cuts(tmp_path):
    rom = _rom(tmp_path, "v", [("a", 0x1000, [OP, JR, NOP]), ("b", 0x1100, [OP, JR, NOP])])
    rows = (*_rows([("a", 0x1000, [OP, JR, NOP])]), *_rows([("b", 0x1100, [OP, JR, NOP])], seg=1))
    assert edges._version_job(("sha", str(rom), rows))[0] == (("a", "b", "seg"),)
    rows = _rows([("a", 0x1000, [OP, JR, NOP]), ("b", 0x1100, [OP, JR, NOP])])
    assert edges._version_job(("sha", str(rom), rows))[0] == (("a", "b", "gap"),)


def test_data_runs_end_at_other_sections(tmp_path):
    rom = _rom(tmp_path, "v", [])
    rows = (
        ("x", ".data", 0x1000, 0x1004, 0x80001000, 0),
        ("y", ".data", 0x1010, 0x1014, 0x80001010, 0),
        ("z", ".data", 0x1014, 0x1018, 0x80001014, 0),
        ("fn", ".text", 0x1018, 0x1020, 0x80001018, 0),
        ("w", ".data", 0x1020, 0x1024, 0x80001020, 0),
    )
    _, data, _ = edges._version_job(("sha", str(rom), rows))
    assert data == (("x", "y", "pad"), ("y", "z", "unal"))


def _edges(**pairs):
    return edges.Edges(pairs, {}, {v: rows[0][0] for v, rows in pairs.items()})


def test_any_version_proves_a_cut_or_a_join_and_both_contradict():
    e = _edges(
        a=(("f", "g", "pad"), ("g", "h", "al0"), ("h", "i", "unal")),
        b=(("f", "g", "al0"), ("g", "h", "unal"), ("h", "i", "pad")),
    )
    assert e.verdicts == {("f", "g"): "cut", ("g", "h"): "join", ("h", "i"): "contradiction"}
    census = e.census("b")
    assert (census["cut"], census["cut_other"], census["join"], census["join_other"]) == (0, 1, 1, 0)
    assert census["contradiction"] == 1 and census["objects_min"] == 2
    assert e.contradictions() == [{"left": "h", "right": "i", "classes": {"a": "unal", "b": "pad"}}]


def test_a_member_gets_its_class_per_version_and_its_reference_verdict():
    e = _edges(a=(("f", "g", "al0"), ("g", "h", "unal")), b=(("f", "g", "pad"),))
    assert e.member("g", "a") == ({"a": "al0", "b": "pad"}, "cut")
    assert e.member("f", "a") == ({"a": "seg", "b": "seg"}, "cut")  # a version's first function starts a segment
    assert e.member("h", "a") == ({"a": "unal"}, "join")


def test_census_estimates_hidden_objects_among_open_pairs():
    e = _edges(a=tuple((f"f{i}", f"f{i + 1}", "pad" if i < 6 else "al0") for i in range(9)))
    census = e.census("a")
    assert (census["cut"], census["open"], census["objects_min"], census["objects_estimate"]) == (6, 3, 7, 9)


def test_data_census_counts_module_changes_per_class():
    e = edges.Edges({}, {"a": (("x", "y", "unal"), ("y", "z", "al0"))}, {})
    assert e.data_census({"x": "m", "y": "m", "z": "n"}) == {
        "a": {"al0": {"module_cut": 1, "n": 1}, "unal": {"module_cut": 0, "n": 1}}
    }


def _snapshot(tmp_path, functions):
    rom = _rom(tmp_path, "a", functions)
    version = SimpleNamespace(
        rom=rom,
        rom_sha256="sha-a",
        segments=(("main", 0x1000, 0x2000, 0x80001000),),
        symbols={n: 0x80000000 + s for n, s, _ in functions},
        code=frozenset(n for n, _, _ in functions),
    )
    return SimpleNamespace(digest="d1", config=SimpleNamespace(), versions={"a": version})


def test_rows_come_from_the_symbols_of_a_version_and_end_where_the_next_starts(tmp_path):
    snapshot = _snapshot(tmp_path, [("a", 0x1000, [OP, JR, NOP, NOP]), ("b", 0x1010, [OP, JR, NOP])])
    snapshot.versions["a"].symbols["D_x"] = 0x80001100
    assert edges._rows(snapshot)["a"] == [
        ("a", ".text", 0x1000, 0x1010, 0x80001000, 0),
        ("b", ".text", 0x1010, 0x1100, 0x80001010, 0),
        ("D_x", ".data", 0x1100, 0x2000, 0x80001100, 0),
    ]


def test_of_maps_each_version_once_per_snapshot_keyed_by_rom_sha_and_rows(tmp_path, monkeypatch):
    calls = []

    def fake_map(config, name, function, items, key):
        calls.append([key(i) for i in items])
        return [function(i) for i in items]

    monkeypatch.setattr(edges.pool, "map", fake_map)
    snapshot = _snapshot(tmp_path, [("a", 0x1000, [OP, JR, NOP, NOP]), ("b", 0x1010, [OP, JR, NOP])])
    first = edges.of(snapshot)
    assert first.pairs == {"a": (("a", "b", "pad"),)} and edges.of(snapshot) is first and len(calls) == 1
    moved = _snapshot(
        tmp_path / ".." / tmp_path.name, [("a", 0x1000, [OP, JR, NOP, NOP]), ("b", 0x1010, [OP, JR, NOP])]
    )
    moved.digest = "d2"
    edges.of(moved)
    assert calls[0] == calls[1]  # the key names the ROM by its sha, never its path: a copied tree hits
