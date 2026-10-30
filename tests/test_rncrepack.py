"""Stale RNC2 streams the packer reproduces: whole streams are rnc's, damaged ones are proven as far as they re-pack."""

from pathlib import Path

import pytest
from test_rnc2enc import text

from unbake import blob_formats, blob_stale, rnc2enc, rncrepack


def placed(raw, at=0x100, size=0x4000):
    rom = bytearray(b"\xff" * size)
    rom[at : at + len(raw)] = raw
    return bytes(rom), at


def literal_byte(data, nth):
    """Offset in the stream (header included) of the data byte of the nth literal token."""
    tokens = rnc2enc.parse(data)
    picks = [k for k, t in enumerate(tokens) if t[0] == "L"]
    return 18 + len(rnc2enc.emit(data, tokens[: picks[nth] + 1])[0]) - 1


def test_a_whole_stream_belongs_to_rnc():
    rom, at = placed(rnc2enc.stream(text(6000), 2))
    with pytest.raises(rncrepack.StreamError, match="packed CRC passes"):
        rncrepack.measure(rom, at)


def test_a_literal_byte_replaced_later_still_packs_back_to_the_same_bytes():
    data = text(6000)
    raw = bytearray(rnc2enc.stream(data, 2))
    spot = literal_byte(data, -1)  # a late literal: no later token looks at it
    raw[spot] ^= 0x55
    rom, at = placed(bytes(raw))
    got = rncrepack.measure(rom, at)
    assert got["credit"] == got["packed"] and got["runs"] == [[0, got["packed"]]]
    assert got["packed_crc_read"] != got["packed_crc"] and got["decoded"] == 6000
    assert got["settings"]["window"] == 4096 and len(got["decoded_sha256"]) == 64


def test_a_damaged_control_byte_ends_the_run_and_the_proof_stays_off_the_damage():
    data = text(9000)
    raw = bytearray(rnc2enc.stream(data, 2))
    cut = 18 + len(raw) // 2
    raw[cut : cut + 3] = b"\xff\xff\xff"
    rom, at = placed(bytes(raw))
    got = rncrepack.measure(rom, at)
    first = got["runs"][0]
    assert first[0] == 0 and 200 < first[1] <= cut - 18 and got["credit"] == sum(b - a for a, b in got["runs"])
    assert all(b <= cut - 18 or a >= cut - 18 + 3 for a, b in got["runs"])
    rows = rncrepack.spans(rncrepack.find(rom, at, at + len(raw)), [(at + 18, at + 18 + 400), (at + cut, at + cut + 3)])
    assert [r["region"] for r in rows] == [[at + 18, at + 18 + 400]]
    ok = rncrepack.prove(rom, at + 18, at + 18 + 400, rows[0]["stream"])
    assert ok["region"] == [at + 18, at + 18 + 400] and ok["rebuilt_sha256"] == ok["region_sha256"]
    with pytest.raises(rncrepack.StreamError, match="not inside"):
        rncrepack.prove(rom, at + 18, at + cut + 100, rows[0]["stream"])
    with pytest.raises(rncrepack.StreamError, match="not inside"):
        rncrepack.prove(rom, at + 4, at + 40, rows[0]["stream"])  # the header is not proven here


REAL = Path(__file__).parent / "data" / "de_stale_1d3917.bin"  # the first 4900 bytes of the German stream at 0x1d3917
REAL_RUNS = [[0, 4251], [4271, 4602], [4654, 4674]]  # packed-byte offsets of its runs before the cut


def test_the_real_german_stream_is_read_past_its_damage_window_by_window():
    rom, at = placed(REAL.read_bytes())
    got = rncrepack.measure(rom, at)
    assert got["runs"][:3] == REAL_RUNS and got["packed_crc_read"] != got["packed_crc"]
    rows = rncrepack.spans(rncrepack.find(rom, at, at + len(REAL.read_bytes())), [(at + 18 + 4271, at + 18 + 4602)])
    assert [r["region"] for r in rows] == [[at + 18 + 4271, at + 18 + 4602]]
    ok = rncrepack.prove(rom, *rows[0]["region"], rows[0]["stream"])  # a run after the first damage
    assert ok["rebuilt_sha256"] == ok["region_sha256"] and ok["decoded"] > 4900
    for low, high in ([4251, 4271], [4200, 4300], [4602, 4654]):  # the damage, and regions straddling it
        with pytest.raises(rncrepack.StreamError, match="not inside"):
            rncrepack.prove(rom, at + 18 + low, at + 18 + high, rows[0]["stream"])


def test_every_run_rebuilds_to_the_rom_bytes_exactly_and_a_flipped_early_bit_leaves_nothing_to_claim():
    rom, at = placed(REAL.read_bytes())
    stream = [at, at + 7355]
    for low, high in rncrepack.measure(rom, at)["runs"]:
        ok = rncrepack.prove(rom, at + 18 + low, at + 18 + high, stream)
        assert ok["rebuilt_sha256"] == ok["region_sha256"]
    spoiled = bytearray(rom)
    spoiled[at + 18 + 2] ^= 0x80  # the third packed byte: the tokens that follow change
    with pytest.raises(rncrepack.StreamError, match="no run"):
        rncrepack.measure(bytes(spoiled), at)


def test_noise_and_short_proofs_are_refused():
    import random

    noise = random.Random(5).randbytes(0x4000)
    head = b"RNC\x02" + (5000).to_bytes(4, "big") + (4000).to_bytes(4, "big") + b"\x12\x34\x56\x78\x02\x01"
    rom = bytearray(noise)
    rom[0x100 : 0x100 + len(head)] = head
    with pytest.raises(rncrepack.StreamError):
        rncrepack.measure(bytes(rom), 0x100)
    assert rncrepack.find(bytes(rom), 0x100, 0x2000) == []


def test_scan_rows_and_the_reader_name_the_reproduced_bytes(tmp_path):
    data = text(9000)
    raw = bytearray(rnc2enc.stream(data, 2))
    raw[18 + len(raw) // 2 : 18 + len(raw) // 2 + 3] = b"\xff\xff\xff"
    rom, at = placed(bytes(raw))
    rows = blob_stale.rnc_streams(rom, [(at + 18, at + 18 + 200)])
    assert [r["reader"] for r in rows] == ["rnc-repack"] and rows[0]["region"] == [at + 18, at + 18 + 200]
    assert "rnc-repack" in blob_formats.READERS and "rnc-stale" not in blob_formats.READERS
