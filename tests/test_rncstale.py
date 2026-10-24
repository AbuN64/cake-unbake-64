"""Damaged stale RNC2 streams: read to an end marker where the header says, the packed CRC failing."""

import pytest
from test_rnc2 import Writer

from unbake import blob_stale, rncstale


def build(literals=1500):
    """A stream of `literals` distinct-ish bytes then a match: (rom bytes of the stream, offset of a literal in it)."""
    data = bytes((i * 7 + i // 5) % 251 for i in range(literals))
    w = Writer()
    w.literals(data)
    w.match(6, 3)
    out = data + (data[-3:] * 2)
    return w.stream(out), 18 + 40


def damaged(stream, at, value=0x5A):
    raw = bytearray(stream)
    raw[at] = value
    return bytes(raw)


def rom_of(stream, at=0x100, size=0x2000):
    rom = bytearray(b"\xff" * size)
    rom[at : at + len(stream)] = stream
    return bytes(rom), at


def test_a_whole_stream_is_the_business_of_rnc_not_of_this_reader():
    stream, _ = build()
    rom, at = rom_of(stream)
    with pytest.raises(rncstale.StreamError, match="packed CRC passes"):
        rncstale.check(rom, at)


def test_a_literal_damaged_stream_reads_to_its_end_marker_and_is_measured():
    stream, _ = build()
    rom, at = rom_of(damaged(stream, 0x28))
    got = rncstale.check(rom, at)
    assert got["end_marker_at"] == got["packed"] and got["output"] == got["unpacked"]
    assert got["packed_crc_read"] != got["packed_crc"] and got["tokens"]["literal"] == 1500
    assert rncstale.find(rom, at + 0x100, at + 0x200) == [(at, got["stored"])]
    assert rncstale.find(rom, 0x1000, 0x1100) == []
    rows = rncstale.spans(rncstale.find(rom, at, at + 0x100), [(at + 0x20, at + 0x80), (0x1800, 0x1900)])
    assert rows == [{"region": [at + 0x20, at + 0x80], "stream": [at, at + got["stored"]]}]
    receipt = rncstale.prove(rom, at + 0x20, at + 0x80, rows[0]["stream"])
    assert receipt["stream"] == rows[0]["stream"] and receipt["region"] == [at + 0x20, at + 0x80]


def test_bits_that_derail_or_miss_the_end_are_no_stream():
    stream, _ = build()
    rom, at = rom_of(stream)
    for cut, size in ((at + len(stream) - 4, 4), (at + len(stream) - 2, 2)):
        wrecked = bytearray(rom)
        wrecked[cut : cut + size] = b"\xff" * size
        with pytest.raises(rncstale.StreamError):
            rncstale.check(bytes(wrecked), at)
    again = bytearray(rom)  # a damaged control byte re-reads literals as tokens and still reaches the marker
    again[at + 18 + 9] = 0xFF
    assert rncstale.check(bytes(again), at)["end_marker_at"] == len(stream) - 18
    short, _ = build(literals=200)
    rom, at = rom_of(damaged(short, 0x28))
    with pytest.raises(rncstale.StreamError, match="fewer than"):
        rncstale.check(rom, at)


def test_a_region_outside_its_stream_or_a_wrong_stream_range_proves_nothing():
    stream, _ = build()
    rom, at = rom_of(damaged(stream, 0x28))
    stored = len(stream)
    with pytest.raises(rncstale.StreamError, match="not inside"):
        rncstale.prove(rom, at - 4, at + 8, [at, at + stored])
    with pytest.raises(rncstale.StreamError, match="stored size"):
        rncstale.prove(rom, at, at + 8, [at, at + stored - 2])


def test_scan_rows_are_the_unknown_parts_of_damaged_streams():
    stream, _ = build()
    rom, at = rom_of(damaged(stream, 0x28))
    rows = blob_stale.rnc_streams(rom, [(at + 6, at + 0x40), (at + 0x80, at + 0x100)])
    assert [r["region"] for r in rows] == [[at + 6, at + 0x40], [at + 0x80, at + 0x100]]
    assert {r["reader"] for r in rows} == {"rnc-stale"} and {tuple(r["stream"]) for r in rows} == {
        (at, at + len(stream))
    }
