"""Stale bytes proven against what the project accounts for, with synthetic ROMs."""

import random
import struct

from unbake import blob_stale, snsym

NOISE = random.Random(7).randbytes(4096)


def rom(*pieces, size=0x1000):
    out = bytearray(size)
    for at, data in pieces:
        out[at : at + len(data)] = data
    return bytes(out)


def test_find_lists_exact_copies_of_accounted_bytes_in_any_version():
    roms = {"a": rom((0x100, NOISE[:300])), "b": rom((0x400, NOISE[:300]), (0x800, NOISE[100:200]))}
    spans = {"a": [(0x100, 0x22C)], "b": [(0, 0x10)]}
    windows = blob_stale.index(roms, spans)
    found = blob_stale.find(roms, spans, windows, "b", [(0x400, 0x800)], ["a", "b"])
    assert found == [
        {"version": "b", "region": [0x400, 0x52C], "reader": "copy", "source": 0x100, "source_version": "a"}
    ]


def test_a_copy_stops_where_the_source_stops_being_accounted_and_where_the_target_span_ends():
    roms = {"a": rom((0x100, NOISE[:300])), "b": rom((0x400, NOISE[:300]))}
    spans = {"a": [(0x100, 0x180)], "b": []}
    found = blob_stale.find(roms, spans, blob_stale.index(roms, spans), "b", [(0x400, 0x500)], ["a", "b"])
    assert [(f["region"], f["source"]) for f in found] == [([0x400, 0x480], 0x100)]  # 0x80 accounted bytes, no more
    found = blob_stale.find(roms, spans, blob_stale.index(roms, spans), "b", [(0x410, 0x44F)], ["a", "b"])
    assert found == []  # 63 bytes left in the target span: under a copy


def test_a_run_of_few_values_is_no_copy():
    roms = {"a": rom((0x100, bytes([1, 2]) * 100)), "b": rom((0x400, bytes([1, 2]) * 100))}
    spans = {"a": [(0x100, 0x1C8)], "b": []}
    assert blob_stale.find(roms, spans, blob_stale.index(roms, spans), "b", [(0x400, 0x4C8)]) == []


def test_the_nearer_source_wins_a_tie():
    roms = {"a": rom((0x100, NOISE[:200])), "b": rom((0x100, NOISE[:200]), (0x600, NOISE[:200]))}
    spans = {"a": [(0x100, 0x1C8)], "b": [(0x100, 0x1C8)]}
    (found,) = blob_stale.find(roms, spans, blob_stale.index(roms, spans), "b", [(0x600, 0x700)], ["a", "b"])
    assert found["source_version"] == "b"


def test_objects_are_found_by_their_magic_and_their_whole_records():
    body = b"".join(b"\x4c" + struct.pack("<HII", 2, i, 5) for i in range(60))
    data = rom((0x100, snsym.OBJ_MAGIC + body), (0x800, snsym.OBJ_MAGIC + body[:60]), size=0x1000)
    found = blob_stale.objects(data, [(0x80, 0x1000)])
    assert found == [{"version": "", "region": [0x100, 0x100 + 4 + len(body)], "reader": "sn-debug", "dialect": "obj"}]
