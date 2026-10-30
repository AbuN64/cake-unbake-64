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
    found = blob_stale.find(roms, spans, blob_stale.index(roms, spans), "b", [(0x410, 0x42F)], ["a", "b"])
    assert found == []  # 31 bytes left in the target span: under a copy
    found = blob_stale.find(roms, spans, blob_stale.index(roms, spans), "b", [(0x410, 0x430)], ["a", "b"])
    assert [(f["region"], f["source"]) for f in found] == [([0x410, 0x430], 0x110)]  # 32 bytes are a copy


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


def test_short_terminated_text_can_copy_from_accounted_bytes_at_any_byte_phase():
    text = b"pThis != NULL\0"
    roms = {"a": rom((0x103, text)), "b": rom((0x407, text))}
    spans = {"a": [(0x103, 0x103 + len(text))], "b": []}
    sources = blob_stale.text_index(roms, spans)
    assert blob_stale.text_copies(roms["b"], sources, "b", [(0x400, 0x500)]) == [
        {"version": "b", "region": [0x407, 0x407 + len(text)], "reader": "copy", "source": 0x103, "source_version": "a"}
    ]


def test_text_needs_a_whole_accounted_source_and_a_whole_terminated_target():
    text = b"pThis != NULL\0"
    roms = {"a": rom((0x100, text)), "b": rom((0x400, text), (0x500, b"different text\0"))}
    short = blob_stale.text_index(roms, {"a": [(0x100, 0x100 + len(text) - 1)], "b": []})
    assert blob_stale.text_copies(roms["b"], short, "b", [(0x400, 0x600)]) == []
    sources = blob_stale.text_index(roms, {"a": [(0x100, 0x100 + len(text))], "b": []})
    assert blob_stale.text_copies(roms["b"], sources, "b", [(0x400, 0x400 + len(text) - 1)]) == []
    assert blob_stale.text_copies(roms["b"], sources, "b", [(0x500, 0x600)]) == []


def test_tiny_and_uniform_text_cannot_supply_copy_evidence():
    data = rom((0x100, b"pThis\0aaaaaaaaaaaa\0"))
    assert blob_stale.text_index({"a": data}, {"a": [(0x100, 0x200)]}) == {}


def test_text_source_choice_is_deterministic_prefers_same_rom_and_never_self():
    text = b"pThis != NULL\0"
    data = rom((0x100, text), (0x300, text))
    sources = {text: [("z", 0x200), ("b", 0x300), ("a", 0x400), ("b", 0x100)]}
    (found,) = blob_stale.text_copies(data, sources, "b", [(0x100, 0x200)], ["a", "z", "b"])
    assert (found["source_version"], found["source"]) == ("b", 0x300)
    sources[text] = [("z", 0x200), ("b", 0x100), ("a", 0x400)]
    (found,) = blob_stale.text_copies(data, sources, "b", [(0x100, 0x200)], ["a", "z", "b"])
    assert (found["source_version"], found["source"]) == ("a", 0x400)
    assert blob_stale.text_copies(data, {text: [("b", 0x100)]}, "b", [(0x100, 0x200)]) == []


def _definition(name):
    return struct.pack("<IBHHI", 0, 0x94, 8, 5, 0) + bytes([len(name)]) + name


def test_a_cut_stream_head_is_listed_only_when_the_page_bytes_are_unknown_and_a_witness_holds_the_record():
    first, following = _definition(b"paramSamples"), _definition(b"next")
    page = rom((0x1000 - 0x40, b"\xff" * 0x40), (0x1000, b"amSamples" + following), size=0x2000)
    other = rom((0x10, first + following))
    roms = {"a": page, "b": other}
    start = 0x1000 + 9
    found = blob_stale.heads(roms, "a", [(0xFC0, 0x1009)], [start], ["a", "b"])
    assert found == [{"version": "a", "region": [0x1000, 0x1009], "reader": "sn-head"}]
    assert blob_stale.heads(roms, "a", [(0xFC0, 0x1008)], [start], ["a", "b"]) == []  # a head byte is accounted
    assert blob_stale.heads(roms, "a", [(0xFC0, 0x1009)], [start], ["a"]) == []  # no witness anywhere
    assert blob_stale.heads(roms, "a", [(0xFC0, 0x1009)], [0x1000], ["a", "b"]) == []  # a stream at the page start


def test_code_streams_fill_the_unknown_spans_a_stream_covers_and_name_the_stream_whole():
    from test_mipsstream import function, pack

    words = function(size=14) + function(size=20, frame=48)
    data = rom((0x204, pack(words)), size=0x800)
    end = 0x204 + 4 * len(words)
    rows = blob_stale.code_streams(data, [(0x1F0, 0x240), (0x300, 0x400), (end - 8, end + 0x20)])
    assert rows == [
        {"version": "", "reader": "mips-stream", "region": [0x204, 0x240], "stream": [0x204, end]},
        {"version": "", "reader": "mips-stream", "region": [end - 8, end], "stream": [0x204, end]},
    ]
    shifted = rom((0x203, pack(words)), size=0x800)  # the same functions one byte off the word grid are not code
    assert blob_stale.code_streams(shifted, [(0x1F0, 0x240)]) == []
    assert blob_stale.code_streams(rom((0, NOISE)), [(0, 0x1000)]) == []


def test_a_cut_object_head_is_listed_with_its_dialect():
    def obj(name):
        return bytes([0x52]) + struct.pack("<HI", 1, 0) + _definition(name)[5:]

    first, following = obj(b"paramSamples"), obj(b"next")
    page = rom((0x1000 - 0x40, b"\x00" * 0x40), (0x1000, b"amSamples" + following), size=0x2000)
    roms = {"a": page, "b": rom((0x10, first + following))}
    found = blob_stale.heads(roms, "a", [(0xFC0, 0x1009)], [0x1009], ["a", "b"])
    assert found == [{"version": "a", "region": [0x1000, 0x1009], "reader": "sn-head", "dialect": "obj"}]
