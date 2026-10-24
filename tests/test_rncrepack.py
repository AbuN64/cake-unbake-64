"""Stale RNC2 streams the packer reproduces: whole streams are rnc's, damaged ones are proven as far as they re-pack."""

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
    assert got["credit"] == got["packed"] and got["first_difference"] == got["packed"]
    assert got["packed_crc_read"] != got["packed_crc"] and got["decoded"] == 6000
    assert got["settings"]["window"] == 4096 and len(got["decoded_sha256"]) == 64


def test_a_damaged_control_byte_ends_the_proof_at_the_first_byte_the_packer_would_not_write():
    data = text(9000)
    raw = bytearray(rnc2enc.stream(data, 2))
    cut = 18 + len(raw) // 2
    raw[cut : cut + 3] = b"\xff\xff\xff"
    rom, at = placed(bytes(raw))
    got = rncrepack.measure(rom, at)
    assert 200 < got["credit"] <= cut and got["first_difference"] <= cut
    rows = rncrepack.spans(
        rncrepack.find(rom, at, at + len(raw)), [(at + 18, at + 18 + 400), (at + cut + 200, at + cut + 300)]
    )
    assert [r["region"] for r in rows] == [[at + 18, at + 18 + 400]]
    ok = rncrepack.prove(rom, at + 18, at + 18 + 400, rows[0]["stream"])
    assert ok["region"] == [at + 18, at + 18 + 400] and ok["credit"] == got["credit"]
    with pytest.raises(rncrepack.StreamError, match="not inside"):
        rncrepack.prove(rom, at + 18, at + cut + 100, rows[0]["stream"])
    with pytest.raises(rncrepack.StreamError, match="not inside"):
        rncrepack.prove(rom, at + 4, at + 40, rows[0]["stream"])  # the header is not proven here


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
