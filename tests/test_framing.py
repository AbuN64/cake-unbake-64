"""Block headers a builder wrote between blocks: frames, whole streams through them and copies that carry them."""

import random

import pytest
from test_rnc2enc import text

from unbake import blob_formats, blob_stale, framed_copy, framing, rnc2enc, rnc_whole

HEADER = b"\x02\x00\x40"
BLOCK = 0x1000


def framed(raw, points=8, anchor=0x20, noise=1):
    """A ROM whose image is cut every BLOCK bytes by HEADER, with `raw` written through it from the anchor + 3."""
    rng = random.Random(noise)
    period = BLOCK + len(HEADER)
    size = anchor + period * points
    rom = bytearray(rng.randbytes(size))
    for k in range(points):
        rom[anchor + k * period : anchor + k * period + 3] = HEADER
    at, left = anchor + len(HEADER), bytes(raw)
    while left:
        if (at - anchor) % period == 0:
            at += len(HEADER)
        room = period - (at - anchor) % period
        rom[at : at + min(room, len(left))] = left[:room]
        at, left = at + min(room, len(left)), left[room:]
    return bytes(rom)


def frame():
    return framing.Frame(0x20, HEADER, BLOCK)


def test_a_frame_is_held_as_far_as_the_header_stands_and_refused_when_short():
    rom = framed(b"", points=8)
    assert framing.held(rom, frame()) == 8 and framing.accepted(rom, frame()) == 8
    broken = bytearray(rom)
    broken[0x20 + 3 * (BLOCK + 3)] ^= 0xFF
    assert framing.held(bytes(broken), frame()) == 3
    with pytest.raises(framing.FrameError, match="fewer than"):
        framing.accepted(bytes(broken), frame())


def test_a_frame_row_must_name_a_block_and_a_short_header():
    assert framing.parse(frame().row()) == frame()
    for row in ({"anchor": 0, "header": "", "block": BLOCK}, {"anchor": 0, "header": "02", "block": 8}, {"anchor": 0}):
        with pytest.raises(framing.FrameError):
            framing.parse(row)


def test_discover_finds_the_chain_and_strip_and_raw_end_agree():
    rom = framed(b"")
    found = framing.discover(rom, [HEADER, b"\x00\x00"])
    assert [f for f in found if f.header == HEADER] == [frame()]
    count = framing.held(rom, frame())
    low, high = 0x20 + 3 + 100, 0x20 + 3 + BLOCK + 100
    cut = framing.strip(rom, low, high, frame(), count)
    assert cut == rom[low : 0x20 + BLOCK + 3] + rom[0x20 + BLOCK + 6 : high]
    assert framing.raw_end(frame(), count, low, high - low - 3) == high
    with pytest.raises(framing.FrameError, match="cut"):
        framing.strip(rom, 0x20 + BLOCK + 4, high, frame(), count)


def test_a_stream_crossing_a_header_is_whole_only_through_the_frame():
    raw = rnc2enc.stream(text(9000), 2)
    rom = framed(b"\xff" * (BLOCK - 40) + raw)
    at = 0x20 + 3 + BLOCK - 40
    found = rnc_whole.find(rom, [frame()], at, at + 100)
    assert [(a, f) for a, _, f in found] == [(at, frame())]
    receipt = rnc_whole.prove(rom, at + 18, at + 30, at, frame().row())
    assert receipt["stream"][0] == at and receipt["headers"] == [] and receipt["unpacked"] == 9000
    crossing = rnc_whole.prove(rom, at + 18, found[0][1], at, frame().row())
    assert crossing["headers"] == [0x20 + 3 + BLOCK]
    with pytest.raises(rnc_whole.StreamError):
        rnc_whole.prove(rom, at + 18, found[0][1], at)


def test_a_flipped_byte_or_a_region_outside_the_stream_is_refused():
    raw = rnc2enc.stream(text(9000), 2)
    rom = bytearray(framed(b"\xff" * (BLOCK - 40) + raw))
    at = 0x20 + 3 + BLOCK - 40
    row = frame().row()
    with pytest.raises(rnc_whole.StreamError, match="not inside"):
        rnc_whole.prove(bytes(rom), at - 8, at + 40, at, row)
    rom[at + 50] ^= 0x01
    with pytest.raises(rnc_whole.StreamError):
        rnc_whole.prove(bytes(rom), at + 18, at + 60, at, row)
    assert rnc_whole.find(bytes(rom), [frame()], at, at + 100) == []


def test_a_frame_with_no_point_inside_the_stream_is_named_none():
    raw = rnc2enc.stream(text(900), 2)
    rom = framed(raw)
    with pytest.raises(rnc_whole.StreamError, match="no lattice point"):
        rnc_whole.prove(rom, 0x23 + 18, 0x23 + 40, 0x23, frame().row())
    assert rnc_whole.prove(rom, 0x23 + 18, 0x23 + 40, 0x23)["unpacked"] == 900


def test_a_copy_that_crosses_a_header_has_the_source_on_both_sides():
    rng = random.Random(9)
    source = rng.randbytes(0x400)
    rom = bytearray(framed(b""))
    point = 0x20 + BLOCK + 3
    rom[point - 100 : point] = source[100:200]
    rom[point + 3 : point + 103] = source[200:300]
    rom = bytes(rom)
    assert framed_copy.find(rom, source, point, point + 3) == 200
    receipt = framed_copy.prove(rom, source, point, point + 3, 200, frame().row())
    assert receipt["before"][1] == 200 and receipt["after"][0] == 200 and receipt["after"][1] >= 300
    for bad_source in (199, 201):
        with pytest.raises(framed_copy.CopyError, match="fewer than"):
            framed_copy.prove(rom, source, point, point + 3, bad_source, frame().row())
    with pytest.raises(framed_copy.CopyError, match="lattice point"):
        framed_copy.prove(rom, source, point + 1, point + 4, 200, frame().row())
    assert framed_copy.find(rom, rng.randbytes(0x400), point, point + 3) is None


def test_scan_rows_and_the_readers_are_named():
    raw = rnc2enc.stream(text(9000), 2)
    rom = framed(b"\xff" * (BLOCK - 40) + raw)
    at = 0x20 + 3 + BLOCK - 40
    held = blob_stale.frames(rom, [(0x20, 0x23), (at + 18, at + 60)])
    assert held == [frame()]
    rows = blob_stale.whole_streams(rom, held, [(at + 18, at + 60)])
    assert [(r["reader"], r["stream"]) for r in rows] == [("rnc-whole", at)] and rows[0]["frame"] == frame().row()
    assert {"rnc-whole", "framed-copy"} <= blob_formats.READERS


def test_the_row_readers_prove_through_blob_formats(tmp_path):
    """The dispatch passes the row's stream, frame and source to the readers and names refusals by reader."""
    raw = rnc2enc.stream(text(9000), 2)
    rom = framed(b"\xff" * (BLOCK - 40) + raw)
    at = 0x20 + 3 + BLOCK - 40
    row = {"version": "v", "reader": "rnc-whole", "region": [at + 18, at + 30], "stream": at, "frame": frame().row()}
    assert blob_formats.prove(None, row, rom, {})["unpacked"] == 9000
    with pytest.raises(ValueError, match="rnc-whole"):
        blob_formats.prove(None, {**row, "stream": at + 1}, rom, {})


def test_framed_copy_dispatch_reads_the_source_rom_of_the_named_version(monkeypatch):
    from types import SimpleNamespace

    source = random.Random(9).randbytes(0x400)
    rom = bytearray(framed(b""))
    point = 0x20 + BLOCK + 3
    rom[point - 100 : point] = source[100:200]
    rom[point + 3 : point + 103] = source[200:300]
    config = SimpleNamespace(project=SimpleNamespace(version={"a": 1, "b": 2}))
    monkeypatch.setattr(blob_formats, "_baserom", lambda _config, version: source if version == "b" else b"")
    row = {
        "version": "a",
        "reader": "framed-copy",
        "region": [point, point + 3],
        "source": 200,
        "source_version": "b",
        "frame": frame().row(),
    }
    assert blob_formats.prove(config, row, bytes(rom), {})["source_version"] == "b"
    with pytest.raises(ValueError, match="framed-copy"):
        blob_formats.prove(config, {**row, "source_version": "zz"}, bytes(rom), {})


def test_find_reads_every_stream_not_only_the_first_of_a_scan():
    first, second = rnc2enc.stream(text(5000), 2), rnc2enc.stream(text(7000), 2)
    rom = framed(b"\xff" * 8 + first + b"\xff" * (BLOCK - 8 - len(first) - 100) + second, points=10)
    found = rnc_whole.find(rom, [frame()], 0, len(rom))
    assert len(found) == 2 and found[0][0] == 0x23 + 8 and found[0][1] == 0x23 + 8 + len(first)
    assert rnc_whole.prove(rom, found[1][0] + 18, found[1][0] + 30, found[1][0], found[1][2].row())["unpacked"] == 7000
