"""The SN object-file page reader, with synthetic sectors."""

import struct
from types import SimpleNamespace as N

import pytest

from unbake import blob_formats, blob_stale, snpage

CODE = [0x46012801, 0x3C010000, 0x28820003, 0x27BDFFC0, 0x24020001, 0xAFBF003C, 0x00A21023, 0x00021080, 0x00A02021]
LATER = [*CODE[:4], 0x24030001, *CODE[5:]]
LEAD = b"\x03\x18\x23"


def pack(words):
    return struct.pack(f">{len(words)}I", *words)


def records(count):
    """Line number records (tag 0x34: u16 and u8): whole object records of four bytes."""
    return b"".join(b"\x34" + struct.pack("<HB", i, 1) for i in range(count))


def sector(lead=LEAD, front=CODE, count=12, cut=True, length=600):
    """One sector: stale code, whole records, a code record the sector end cuts (or records to the end)."""
    out = lead + pack(front) + records(count)
    if cut:
        out += b"\x02" + struct.pack("<H", length)
        out += (pack(CODE) * 40)[: snpage.SECTOR - len(out)]
    else:
        out += records(snpage.SECTOR)[: snpage.SECTOR - len(out)]
    return out


def rom(page, at=0x400, size=0x1000):
    out = bytearray(size)
    out[at : at + snpage.SECTOR] = page
    return bytes(out)


def test_a_page_is_stale_code_whole_records_and_a_cut_code_record():
    page = sector()
    got = snpage.prove(rom(page), 0x400, 0x600, [0x400, 0x600])
    assert got["front"]["lead"] == LEAD.hex()
    assert [w["word"] for w in got["front"]["words"]] == [f"{w:08x}" for w in CODE]
    assert got["front"]["words"][0]["asm"].split()[0] == "sub.s"
    assert got["front"]["range"] == [0x400, 0x400 + 3 + 4 * len(CODE)]
    assert got["records"]["count"] == 12
    assert got["records"]["items"][0] == {"at": 0x400 + 3 + 36, "end": 0x400 + 3 + 40, "tag": 0x34, "fields": [0, 1]}
    assert (got["cut"]["tag"], got["cut"]["length"]) == (2, 600)
    assert all(w["word"] for w in got["cut"]["words"])
    assert "compared" not in got


def test_a_page_without_stale_code_or_without_a_cut_record_is_a_page():
    only = records(snpage.SECTOR // 4)
    got = snpage.prove(rom(only), 0x400, 0x600, [0x400, 0x600])
    assert (got["front"], got["cut"], got["records"]["count"]) == (None, None, snpage.SECTOR // 4)
    got = snpage.prove(rom(sector(lead=b"", cut=False)), 0x400, 0x600, [0x400, 0x600])
    assert got["cut"] is None and got["front"]["lead"] == ""


def test_the_region_credits_part_of_the_page_and_the_page_is_one_aligned_sector():
    data = rom(sector())
    assert snpage.prove(data, 0x400, 0x415, [0x400, 0x600])["region"] == [0x400, 0x415]
    for bad in ([0x400, 0x5FF], [0x401, 0x601], [0x400, 0x800], [0x800, 0xA00]):
        with pytest.raises(snpage.PageError):
            snpage.prove(data, 0x400, 0x415, bad)
    for start, end in ((0x3FF, 0x415), (0x400, 0x601)):
        with pytest.raises(snpage.PageError, match="not inside"):
            snpage.prove(data, start, end, [0x400, 0x600])


def test_the_front_must_be_enough_canonical_code():
    with pytest.raises(snpage.PageError, match="fewer than"):
        snpage.prove(rom(sector(front=CODE[:5], count=snpage.MIN_RECORDS)), 0x400, 0x600, [0x400, 0x600])
    data = sector(front=[*CODE[:4], 0x7FFFFFFF, *CODE[5:]], count=snpage.MIN_RECORDS)
    with pytest.raises(snpage.PageError, match="not a canonical instruction"):
        snpage.prove(rom(data), 0x400, 0x600, [0x400, 0x600])
    with pytest.raises(snpage.PageError, match="repeat"):
        snpage.prove(rom(sector(front=[0x00431021] * 9, count=snpage.MIN_RECORDS)), 0x400, 0x600, [0x400, 0x600])


def test_the_records_must_chain_to_the_end_of_the_page():
    with pytest.raises(snpage.PageError, match="no chain"):
        snpage.prove(rom(sector(count=snpage.MIN_RECORDS - 1)), 0x400, 0x600, [0x400, 0x600])
    with pytest.raises(snpage.PageError, match="no chain"):
        snpage.prove(rom(b"\xff" * snpage.SECTOR), 0x400, 0x600, [0x400, 0x600])
    with pytest.raises(snpage.PageError, match="no chain"):
        snpage.prove(rom(bytes(snpage.SECTOR)), 0x400, 0x600, [0x400, 0x600])


def test_the_code_offsets_of_the_records_must_not_go_back():
    backwards = b"".join(b"\x34" + struct.pack("<HB", 100 - i, 1) for i in range(12))
    page = LEAD + pack(CODE) + backwards
    page += b"\x02" + struct.pack("<H", 600) + (pack(CODE) * 40)[: snpage.SECTOR - len(page) - 3]
    with pytest.raises(snpage.PageError, match="no chain"):
        snpage.prove(rom(page), 0x400, 0x600, [0x400, 0x600])
    patch = b"\x0a\x60" + struct.pack("<H", 2) + b"\x00" + struct.pack("<I", 7)  # a patch: kind, offset, constant
    restart = b"\x06\x01\x00" + b"\x02" + struct.pack("<H", 8) + pack(CODE[:2]) + patch
    page = LEAD + pack(CODE) + records(10) + restart
    page += b"\x02" + struct.pack("<H", 600) + (pack(CODE) * 40)[: snpage.SECTOR - len(page) - 3]
    # the offsets restart after the section switch
    assert snpage.prove(rom(page), 0x400, 0x600, [0x400, 0x600])["records"]["count"] == 10 + 3


def test_most_records_must_be_line_records():
    patches = b"".join(b"\x0a\x60" + struct.pack("<H", 2 * i) + b"\x00" + struct.pack("<I", 7) for i in range(12))
    page = LEAD + pack(CODE) + patches
    page += b"\x02" + struct.pack("<H", 600) + (pack(CODE) * 40)[: snpage.SECTOR - len(page) - 3]
    with pytest.raises(snpage.PageError, match="no chain"):
        snpage.prove(rom(page), 0x400, 0x600, [0x400, 0x600])


def test_the_cut_record_must_be_a_code_record_the_end_really_cuts_with_code_in_it():
    for length in (100, 602, 4):  # whole inside the page, or not whole words
        with pytest.raises(snpage.PageError, match="no chain"):
            snpage.prove(rom(sector(length=length)), 0x400, 0x600, [0x400, 0x600])
    cut = len(LEAD) + 4 * len(CODE) + 4 * 12
    data = bytearray(sector())
    data[cut + 3 : cut + 7] = pack([0x7FFFFFFF])  # a payload word that is no instruction
    with pytest.raises(snpage.PageError, match="no chain"):
        snpage.prove(rom(bytes(data)), 0x400, 0x600, [0x400, 0x600])
    data = bytearray(sector())
    data[cut] = 0x3E  # a tag no object file has
    with pytest.raises(snpage.PageError, match="no chain"):
        snpage.prove(rom(bytes(data)), 0x400, 0x600, [0x400, 0x600])


def test_the_front_is_compared_with_the_other_versions_code():
    later = bytes(0x40) + pack(LATER) + bytes(0x40)
    got = snpage.prove(rom(sector()), 0x400, 0x600, [0x400, 0x600], {"b": later, "c": bytes(0x100)})
    assert got["compared"] == [{"version": "b", "rom": 0x40, "equal": 8, "of": 9, "first_difference": 4}]


def test_find_lists_the_proven_pages_that_overlap_a_span():
    data = rom(sector(), at=0x400, size=0x1000)
    assert snpage.find(data, 0x3F0, 0x410) == [0x400]
    assert snpage.find(data, 0x600, 0x800) == []
    assert snpage.find(data, 0x1F0, 0xFFF) == [0x400]
    assert snpage.find(rom(b"\xaa" * snpage.SECTOR), 0, 0x1000) == []


def test_scan_rows_credit_the_spans_inside_a_page():
    data = rom(sector())
    rows = blob_stale.pages(data, [(0x400, 0x415), (0x439, 0x600), (0x700, 0x800)])
    assert rows == [
        {"version": "", "reader": "sn-page", "region": [0x400, 0x415]},
        {"version": "", "reader": "sn-page", "region": [0x439, 0x600]},
    ]


def test_the_blobs_prove_reader_takes_the_row_and_compares_with_the_other_versions(tmp_path):
    (tmp_path / "roms").mkdir()
    (tmp_path / "roms" / "baserom.b.z64").write_bytes(bytes(0x40) + pack(LATER))
    config = N(project=N(root=tmp_path, version={"v": None, "b": None}))
    data = rom(sector())
    spec = dict(id="fixture", version="v", region=[0x400, 0x415], reader="sn-page")
    got = blob_formats.prove(config, spec, data, {"blob": {}})
    assert got["records"]["count"] == 12 and got["compared"][0]["version"] == "b"
    with pytest.raises(ValueError, match="sn-page"):  # a region in a sector that is no page
        blob_formats.prove(config, dict(spec, region=[0x600, 0x680]), data, {"blob": {}})
