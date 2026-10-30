"""The pad reader on a slice of the real us ROM at 0x15fca0: a 343 byte zero run, then 32 bytes of an older link."""

from types import SimpleNamespace as N

import pytest

from unbake import blob_pad, blobs

RUN = 0x15FCA9  # the zero run starts where the sn-debug blob ends and stops at the 0x100 boundary below
STOP = 0x15FE00
END = 0x15FE15  # the gap's end: the next accounted blob
HEAD = bytes.fromhex("00f414000077020000")  # 0x15fca0..0x15fca9, the sn-debug blob's last bytes
TAIL = bytes.fromhex("031823000310400043102100a2102300021080020210218c4300048c820004c4")  # 0x15fe00..0x15fe20


def image(size=0x160000, **patch):
    rom = bytearray(size)
    rom[0x15FCA0:0x15FCA9] = HEAD
    rom[STOP : STOP + len(TAIL)] = TAIL
    for at, value in patch.items():
        rom[int(at[1:], 16)] = value
    return bytes(rom)


def test_the_leading_zero_run_of_a_gap_up_to_an_aligned_nonzero_neighbour_is_pad():
    pad, refused = blob_pad.runs(image(), [(RUN, END)])
    assert pad == [(RUN, STOP)] and refused == []  # the zeros at 0x15fe03.. inside the tail are no pad


def test_a_run_over_the_limit_is_refused_and_listed():
    pad, refused = blob_pad.runs(image(), [(RUN, END)], limit=300)
    assert pad == []
    assert refused == [{"rom": [RUN, STOP], "reason": "zero run of 343 bytes is over the limit 300"}]


def test_a_run_that_ends_off_the_boundary_at_a_nonzero_byte_is_refused():
    pad, refused = blob_pad.runs(image(x15fdfd=0x7F), [(RUN, END)])
    assert pad == [] and refused[0]["rom"] == [RUN, 0x15FDFD]
    assert "off the 4-byte boundary" in refused[0]["reason"]


def test_a_gap_that_is_one_zero_run_is_pad_whatever_its_alignment_and_a_gap_without_zeros_is_not():
    assert blob_pad.runs(image(), [(0x15FCB1, 0x15FCB6), (STOP, STOP + 8)]) == ([(0x15FCB1, 0x15FCB6)], [])


def test_a_short_leading_run_inside_a_longer_gap_is_no_pad_and_is_not_listed():
    assert blob_pad.runs(image(), [(0x15FE03, END)]) == ([], [])  # one zero byte of the tail words


def test_prove_rebuilds_the_zero_bytes_and_refuses_what_differs_or_exceeds_the_limit():
    proof = blob_pad.prove(image(), RUN, STOP, 512)
    assert proof["bytes"] == 343 and proof["limit"] == 512
    with pytest.raises(ValueError, match="differ from the ROM"):
        blob_pad.prove(image(), RUN, 0x15FE04, 512)
    with pytest.raises(ValueError, match="over the limit 100"):
        blob_pad.prove(image(), RUN, STOP, 100)
    with pytest.raises(ValueError, match="integer limit"):
        blob_pad.prove(image(), RUN, STOP, None)


def row(*spans, **extra):
    return {"type": "fill", "format": "pad", "state": "decoded", "limit": 512, "rom": {"us": list(spans)}, **extra}


def test_a_pad_row_needs_its_limit_and_no_range_may_exceed_it():
    assert blobs._pad_findings("p", row((RUN, STOP)), "f") == []
    assert "over its limit" in blobs._pad_findings("p", row((RUN, STOP), limit=100), "f")[0].reason
    assert "needs type fill" in blobs._pad_findings("p", {**row((RUN, STOP)), "type": "text"}, "f")[0].reason


def test_verify_rebuilds_every_pad_range_against_the_rom(tmp_path):
    (tmp_path / "roms").mkdir()
    (tmp_path / "roms" / "baserom.us.z64").write_bytes(image())
    config = N(project=N(root=str(tmp_path)))
    good = {"blob": {"p": row((RUN, STOP))}}
    assert blobs._pad_bytes_findings(config, good, "f") == []
    bad = {"blob": {"p": row((RUN, 0x15FE04))}}
    assert "is not zero" in blobs._pad_bytes_findings(config, bad, "f")[0].reason
    config = N(project=N(root=str(tmp_path / "none")))
    assert "no ROM for us" in blobs._pad_bytes_findings(config, good, "f")[0].reason


def test_the_limit_is_emitted_and_read_back():
    text = blobs.emit({"schema": 1, "blob": {"p": row((RUN, STOP))}}).decode()
    assert "limit = 512" in text and 'rom."us" = [[0x15fca9, 0x15fe00]]' in text
