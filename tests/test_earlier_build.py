"""The earlier-build reader: a slice of the real eu stale bytes, synthetic functions (`Index` is the seam)."""

import struct
from pathlib import Path

import pytest

from unbake import earlier_build as eb

RAW = (Path(__file__).parent / "data" / "eu_stale_slice.bin").read_bytes()
# the five eu spans (ROM 0x14920c, 0x164abd, 0x16b2f8, 0x16eecd, 0x172ced) in order, 3,457 bytes: each is put at a
# multiple of 0x1000 plus its offset in a word, as the ROM holds it
ORIGIN = [(0x14920C, 196), (0x164ABD, 1051), (0x16B2F8, 64), (0x16EECD, 1055), (0x172CED, 1091)]
SPANS = []
_rom = bytearray(0x6000)
_at = 0
for _i, (_rom_at, _size) in enumerate(ORIGIN):
    _start = 0x1000 * _i + _rom_at % 4
    _rom[_start : _start + _size] = RAW[_at : _at + _size]
    SPANS.append((_start, _start + _size))
    _at += _size
SLICE = bytes(_rom)
RA = 0x03E00008


def fn(*body):
    """A function with a stack prologue: addiu sp,sp,-24; the body; jr ra; addiu sp,sp,24 in the delay slot."""
    return (0x27BDFFE8, *body, RA, 0x27BD0018)


def body(n, seed=1):
    return tuple(0x8C820000 | 4 * ((seed + i) % 0x1FFF) for i in range(n))  # lw v0, off(a0)


def compiled(name, words, version="eu", rom=0x1000):
    return eb.Compiled(version, name, 0x80000000 + rom, rom, tuple(words), f"src/{name}.c", "build/x.elf", "0" * 64)


def rom_of(*pieces, size=0x400):
    out = bytearray(size)
    for at, words in pieces:
        out[at : at + 4 * len(words)] = struct.pack(f">{len(words)}I", *words)
    return bytes(out)


def test_real_eu_slice_holds_three_code_pieces_and_address_tables():
    pieces = [eb.disassemble(SLICE, a, b) for a, b in SPANS]
    assert [[len(p["words"]) for p in ps] for ps in pieces] == [[], [29], [], [29], [29]]
    assert all(p["head"] and p["tail"] for ps in pieces for p in ps)
    assert eb.shape(SLICE, *SPANS[2]) == "address table"  # 0x16b2f8: sixteen 0x8015xxxx jump table words
    assert eb.shape(SLICE, SPANS[1][0], SPANS[1][0] + 200) == "address table"  # the head of the 0x164abd span
    assert eb.shape(SLICE, *SPANS[0]) == "unread"


def test_real_eu_pieces_have_no_function_in_the_tree_so_they_stay_unknown():
    index = eb.Index([compiled("A", fn(*body(40)))], ["eu"])
    closed, refused = eb.read(SLICE, "eu", *SPANS[3], index)
    assert closed == []
    assert [r["words"] for r in refused] == [29]
    assert refused[0]["near"] == []


def test_a_stale_function_equal_to_a_compiled_one_closes_with_an_empty_difference_list():
    words = fn(*body(30))
    rom = rom_of((0x100, words))
    index = eb.Index([compiled("Track_A", (0, 0, *words, 0))], ["eu"])
    closed, refused = eb.read(rom, "eu", 0, len(rom), index)
    assert refused == []
    (row,) = closed
    assert row["stream"] == [0x100, 0x100 + 4 * len(words)]
    assert (row["function"], row["from_version"], row["word"]) == ("Track_A", "eu", 2)
    assert row["receipt"]["differences"] == []
    assert row["receipt"]["rebuilt"]["sha256"] == row["receipt"]["stream_sha256"]


def test_a_differing_word_splits_the_piece_each_equal_side_closes_and_the_word_between_stays_unknown():
    words = list(fn(*body(60)))
    stale = list(words)
    stale[30] ^= 0x4  # one word an older compile gave differently
    rom = rom_of((0x100, stale))
    index = eb.Index([compiled("F", words)], ["eu"])
    closed, refused = eb.read(rom, "eu", 0, len(rom), index)
    assert [c["stream"] for c in closed] == [[0x100, 0x100 + 4 * 30], [0x100 + 4 * 31, 0x100 + 4 * len(words)]]
    assert [c["word"] for c in closed] == [0, 31]
    rows = eb.spans(closed, [(0, len(rom))])
    assert eb.blobs_subtract([(0x100, 0x100 + 4 * len(words))], [tuple(r["region"]) for r in rows]) == [
        (0x100 + 4 * 30, 0x100 + 4 * 31)
    ]
    assert eb.differences(stale, words)[0]["word"] == 30
    assert refused == []


def test_the_same_function_in_another_version_is_a_candidate_ordered_by_version_then_offset():
    words = fn(*body(30))
    index = eb.Index(
        [compiled("B", words, "us", 0x3000), compiled("A", words, "de", 0x2000), compiled("C", words, "de", 0x1000)],
        ["de", "eu", "us"],
    )
    found = index.candidates(words, "eu")
    assert [(c.version, c.name) for c, _ in found] == [("de", "C"), ("de", "A"), ("us", "B")]


def test_a_function_at_the_pieces_own_place_is_not_its_source():
    words = fn(*body(30))
    index = eb.Index([compiled("Self", words, "eu", 0x100)], ["eu"])
    assert index.candidates(words, "eu", 0x104) == []


def test_zero_pad_is_never_code():
    index = eb.Index([compiled("Z", (0,) * 64)], ["eu"])
    assert eb.stretch(index, (0,) * 64, 0, "eu", 0) is None
    assert eb.shape(bytes(64), 0, 64) == "pad"


def test_prove_checks_the_region_against_the_function_word_for_word():
    words = fn(*body(30))
    rom = rom_of((0x100, words))
    index = eb.Index([compiled("Track_A", words)], ["eu"])
    stream = [0x100, 0x100 + 4 * len(words)]
    receipt = eb.prove(rom, 0x100, stream[1], stream, "eu", "Track_A", 0, index)
    assert receipt["differences"] == [] and receipt["region"] == stream
    with pytest.raises(eb.EarlierBuildError, match="not inside its stream"):
        eb.prove(rom, 0x100, stream[1] + 4, stream, "eu", "Track_A", 0, index)
    other = eb.Index([compiled("Track_A", (*words[:-3], 0x24020001, *words[-2:]))], ["eu"])
    with pytest.raises(eb.EarlierBuildError, match="differently from the ROM: word 30"):
        eb.prove(rom, 0x100, stream[1], stream, "eu", "Track_A", 0, other)
    with pytest.raises(eb.EarlierBuildError, match="no compiled function"):
        eb.prove(rom, 0x100, stream[1], stream, "eu", "Nope", 0, index)


def test_skeleton_lists_blocks_edges_calls_and_opcode_classes():
    words = (0x27BDFFE8, 0x10400002, 0, 0x0C000400, 0, RA, 0x27BD0018)
    skel = eb.skeleton(words)
    assert skel["edges"] == [[1, 4]] and skel["calls"] == [3] and skel["blocks"] == 3
    assert eb.calls(words) == [{"word": 3, "target": 0x80001000}]


def test_references_pair_a_lui_with_the_access_that_completes_the_address():
    lui = 0x3C040000 | 0x8015
    lw = 0x8C820000 | 0x47B0  # lw v0, 0x47b0(a0)
    (ref,) = [r for r in eb.references((lui, 0, lw)) if r["address"] == 0x801547B0]
    assert ref["size"] == 4 and ref["word"] == 2


def test_blobs_prove_reads_an_earlier_build_row_through_the_compiled_index(monkeypatch):
    from types import SimpleNamespace

    from unbake import blob_formats

    words = fn(*body(30))
    rom = rom_of((0x100, words))
    monkeypatch.setattr(blob_formats, "_compiled", lambda config: eb.Index([compiled("Track_A", words)], ["eu"]))
    stream = [0x100, 0x100 + 4 * len(words)]
    row = {
        "reader": "earlier-build",
        "region": stream,
        "stream": stream,
        "from_version": "eu",
        "function": "Track_A",
        "word": 0,
    }
    done = blob_formats.prove(SimpleNamespace(), row, rom, {})
    assert done["function"]["name"] == "Track_A" and done["differences"] == []
    with pytest.raises(ValueError, match="earlier-build: eu has no compiled function Nope"):
        blob_formats.prove(SimpleNamespace(), {**row, "function": "Nope"}, rom, {})
