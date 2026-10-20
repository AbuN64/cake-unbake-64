"""Owning-format boundaries, with synthetic bytes and mocked decompression."""

import struct
from types import SimpleNamespace as N

import pytest

from unbake import blob_archive, blob_formats, blob_proof, blobs, snsym


def words(*values):
    return struct.pack(f">{len(values)}I", *values)


def cfg(rom_size=512):
    segments = (
        N(name="reserved", rom=0, vram=None, kind="bin"),
        N(name="program", rom=8, vram=0x80000000, kind="code"),
    )
    return N(project=N(version={"v": N(segments=segments, size=rom_size)}))


def row(reader, region, **more):
    return dict(id="fixture", version="v", region=region, reader=reader, evidence="synthetic owning layout", **more)


def prove(spec, rom, doc=None):
    return blob_proof.prove_one(cfg(len(rom)), spec, rom, doc or {"blob": {}}, [])


def archive_fixture(base=4, padding=b"\0" * 6):
    # First image: 8-byte pixels and 34-byte palette; second starts at relative 48.
    image = bytes(8) + struct.pack(">6H2I", 4, 1, 1, 17, 0, 0, 0, 8)
    image += bytes(8) + struct.pack(">6H2I", 16, 1, 1, 0, 0, 0, 48, 0)
    sections = [image, b"", b"", b"", words(0, 1), b"\0", b"", b"", bytes(42) + padding + bytes(8), b""]
    header, at = [], 84
    for body in sections:
        header += [at, len(body)]
        at += len(body)
    return bytes(base) + words(*header, 0xFFFFFFFF) + b"".join(sections), base + header[16] + 42


def test_archive_padding_uses_section_offsets_at_unaligned_rom_base():
    rom, start = archive_fixture()
    assert (start + 6) % 8 != 0
    result = prove(row("archive-fill", [start, start + 6], base=4), rom)
    assert result["decoded"]["fill"] == [[start, start + 6]]
    with pytest.raises(ValueError, match="not archive"):
        prove(row("archive-fill", [start - 1, start + 6], base=4), rom)
    with pytest.raises(ValueError, match="gap"):
        blob_archive.archive(archive_fixture(padding=b"x" * 6)[0], 4)


def test_stale_header_is_decoded_without_claiming_a_live_copy():
    rom, _ = archive_fixture()
    result = prove(row("archive-header", [4, 80], base=4), rom)
    assert result["decoded"]["sentinel"] == 0xFFFFFFFF
    bad = bytearray(rom)
    bad[15] ^= 1
    with pytest.raises(ValueError, match="valid archive"):
        prove(row("archive-header", [4, 80], base=4), bytes(bad))
    assert blob_archive.header_at(b"\0" * 83, 0) is None


def test_nonzero_slack_requires_both_directory_and_complete_stream(monkeypatch):
    rom = words(1, 16, 48, 0) + b"packed!!" + b"builder-tail" * 2
    monkeypatch.setattr(blob_formats.rnc, "header", lambda _: N(stored=8, unpacked=4))
    monkeypatch.setattr(blob_formats.rnc, "unpack", lambda _: b"body")
    spec = row("rnc-slack", [24, 48], container=[0, 48], entry_path=[0])
    result = prove(spec, rom)
    assert result["decoded"]["slack"] == [24, 48]
    with pytest.raises(ValueError, match="not after the stream"):
        prove(dict(spec, region=[23, 48]), rom)
    with pytest.raises(ValueError, match="index outside"):
        prove(dict(spec, entry_path=[1]), rom)
    with pytest.raises(ValueError, match="ordering"):
        prove(spec, words(1, 12, 48, 0) + rom[16:])
    monkeypatch.setattr(blob_formats.rnc, "unpack", lambda _: (_ for _ in ()).throw(ValueError("bad CRC")))
    with pytest.raises(ValueError, match="bad CRC"):
        prove(spec, rom)


def test_string_leaf_checks_slots_and_has_no_uninterpreted_tail(monkeypatch):
    body = words(16, 1) + b"NULL" + bytes(12)
    assert blob_formats.string_leaf(body)["names"] == ["NULL"]
    used_slot = body[:13] + bytes(range(1, 12))
    assert blob_formats.string_leaf(used_slot)["slot_slack_hex"] == [bytes(range(1, 12)).hex()]
    for wrong in (body[:-1], body + b"\0", words(16, 1) + b"x" * 16):
        with pytest.raises(ValueError):
            blob_formats.string_leaf(wrong)
    rom = words(1, 16, 48, 0) + bytes(32)
    monkeypatch.setattr(blob_formats.rnc, "header", lambda _: N(stored=24, unpacked=len(body)))
    monkeypatch.setattr(blob_formats.rnc, "unpack", lambda _: body)
    spec = row("string-leaf", [16, 40], container=[0, 48], entry_path=[0])
    assert prove(spec, rom)["decoded"]["decoded_size"] == 24
    with pytest.raises(ValueError, match="complete packed stream"):
        prove(dict(spec, region=[16, 48]), rom)


def test_explicit_segment_fill_needs_the_entire_declared_bin_extent():
    rom = bytes(8) + b"instructions"
    spec = row("segment-fill", [0, 8], segment="reserved", value=0)
    assert prove(spec, rom)["decoded"]["rom"] == [0, 8]
    for wrong in (dict(spec, region=[0, 7]), dict(spec, segment="program"), dict(spec, value=255)):
        with pytest.raises(ValueError):
            prove(wrong, rom)
    with pytest.raises(ValueError, match="needs project evidence"):
        prove(dict(spec, evidence=""), rom)


def definition(name):
    return struct.pack("<IBHHI", 0, 0x94, 8, 5, 0) + bytes([len(name)]) + name


def test_cut_debug_head_requires_a_real_record_and_following_record_witness():
    first, following = definition(b"paramSamples"), definition(b"next")
    source = first + following
    start = len(source) + 64
    rom = source + b"\xff" * 64 + b"amSamples" + following
    result = snsym.stream_receipt(rom, start, len(rom))
    assert result["head"]["bytes"] == 9
    assert result["head"]["source"] == len(first) - 9
    assert snsym.stream_start(rom, start + 9, start) == start
    # Identical text alone, with no valid containing record, proves nothing.
    bad = b"\xff" * (len(first) - 9) + rom[len(first) - 9 :]
    assert snsym.stream_start(bad, start + 9, start) == start + 9
    with pytest.raises(snsym.SymError, match="no record"):
        snsym.cut_head(bad, start, 9)


def test_copy_needs_decoded_source_and_byte_equality():
    rom = b"abcdefabcdef"
    doc = {"blob": {"source": {"state": "decoded", "rom": {"v": [[0, 6]]}}}}
    spec = row("copy", [6, 12], source=0, source_blob="source")
    assert prove(spec, rom, doc)["decoded"]["source"] == [0, 6]
    with pytest.raises(ValueError, match="differs"):
        prove(spec, rom[:-1] + b"x", doc)
    doc["blob"]["source"]["state"] = "observed"
    with pytest.raises(ValueError, match="decoded owner"):
        prove(spec, rom, doc)


def test_retired_unclassified_observation_can_be_removed_without_fake_credit():
    document = {"blob": {"old": {"type": "unclassified", "state": "observed", "rom": {"v": [[8, 12]]}}}}
    packet = {"blob": {}, "supersedes": ["old"]}
    assert blobs.supersede_findings(document, packet, {"old"}) == []
    before = blobs.ranges(document, "v")
    assert before == blobs.ranges({"blob": {}}, "v") == ([], [])
    document["blob"]["old"]["state"] = "decoded"
    assert blobs.supersede_findings(document, packet, {"old"})
