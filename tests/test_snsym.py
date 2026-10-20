"""SN debug records: reader, run finder, types and functions on streams built here (no game bytes)."""

import struct

import pytest

from unbake import snsym


def _name(text: str) -> bytes:
    return bytes([len(text)]) + text.encode()


def d94(value: int, cls: int, typ: int, size: int, name: str) -> bytes:
    return struct.pack("<IBHHI", value, 0x94, cls, typ, size) + _name(name)


def d96(value: int, cls: int, typ: int, size: int, dims: tuple[int, ...], tag: str, name: str) -> bytes:
    head = struct.pack("<IBHHIH", value, 0x96, cls, typ, size, len(dims))
    return head + struct.pack(f"<{len(dims)}I", *dims) + _name(tag) + _name(name)


def func(addr: int, name: str, frame: int = 24, line: int = 7) -> bytes:
    return struct.pack("<IBHIHIiI", addr, 0x8C, 29, frame, 31, 0x80000000, -4, line) + _name("A.CPP") + _name(name)


STRUCT = (
    d94(0, 10, 8, 12, "CNode_t")
    + d94(0, 8, 4, 0, "m_nCount")
    + d96(4, 8, 0x18, 0, (), "CNode_t", "m_pNext")
    + d94(64, 18, 14, 3, "m_Flags")
    + d96(12, 102, 0, 12, (), "CNode_t", ".eos")
    + d96(0, 13, 8, 12, (), "CNode_t", "CNode")
)
ENUM = d94(0, 15, 10, 4, "Modes") + d94(0, 16, 11, 0, "MODE_IDLE") + d94(0xFFFFFFFF, 16, 11, 0, "MODE_NONE")
ENUM += d96(4, 102, 0, 4, (), "Modes", ".eos")


def test_record_reads_each_layout() -> None:
    rec = snsym.record(STRUCT, 0)
    assert (rec.cls, rec.type, rec.size, rec.name, rec.end) == (10, 8, 12, "CNode_t", 21)
    arr = d96(8, 8, 0x3F, 1024, (128,), "", "Stack")
    rec = snsym.record(arr, 0)
    assert (rec.dims, rec.tag_name, rec.name, rec.end) == ((128,), "", "Stack", len(arr))
    line = struct.pack("<IBB", 0x80204668, 0x82, 3) + struct.pack("<IBI", 0, 0x88, 360) + _name("B.CPP")
    recs, stop = snsym.records(line)
    assert [r.tag for r in recs] == [0x82, 0x88] and recs[0].extra == (3,) and recs[1].name == "B.CPP"
    assert stop == len(line)


@pytest.mark.parametrize(
    "raw",
    [
        struct.pack("<IBHHI", 0, 0x94, 200, 8, 4) + _name("x"),  # class that is not a COFF storage class
        struct.pack("<IBHHI", 0, 0x94, 8, 4, 0) + b"\x05ab\x01cd",  # name with a control byte
        struct.pack("<IB", 0, 0x77),  # unknown tag
        d94(0, 8, 4, 0, "m_nCount")[:-3],  # cut inside the name
    ],
)
def test_record_refuses_non_records(raw: bytes) -> None:
    with pytest.raises(snsym.SymError):
        snsym.record(raw, 0)


def test_records_stop_at_a_cut_and_find_recovers_a_cut_front() -> None:
    body = (STRUCT + ENUM) * 6
    cut = body[7:] + d94(0, 8, 4, 0, "m_nTail")[:9]  # front cut mid-record, tail cut mid-record
    junk = b"\xff" * 40
    data = junk + cut + junk
    recs, stop = snsym.records(data, 40 + len(STRUCT) - 7, len(data))
    assert stop == 40 + len(cut) - 9 and recs[0].name == "Modes"
    [(start, end)] = snsym.find(data, minimum=10)
    assert end == stop and start <= 40 + 21 - 7
    assert snsym.records(data, start, end)[1] == end


def test_types_build_struct_enum_typedef_and_bit_fields() -> None:
    recs, _ = snsym.records(STRUCT + ENUM + d94(0, 12, 9, 4, "UValue_t") + d94(0, 11, 6, 0, "f"))
    out = snsym.types(recs)
    assert [t["name"] for t in out] == ["CNode_t", "CNode", "Modes", "UValue_t"]
    node, typedef, modes, union = out
    assert node["kind"] == "struct" and node["size"] == 12
    assert node["fields"] == [
        {"name": "m_nCount", "offset": 0, "type": "int", "size": 0},
        {"name": "m_pNext", "offset": 4, "type": "struct CNode_t *", "size": 0},
        {"name": "m_Flags", "offset": 8, "type": "unsigned int", "bit": 0, "bits": 3},
    ]
    assert typedef["kind"] == "typedef" and typedef["type"] == "struct CNode_t"
    assert modes["fields"] == [{"name": "MODE_IDLE", "value": 0}, {"name": "MODE_NONE", "value": -1}]
    assert union["kind"] == "union" and union.get("cut") is True  # still open where the run stops


def test_type_text_derivations() -> None:
    assert snsym.type_text(0x3F, dims=(128,)) == "unsigned long [128]"
    assert snsym.type_text(0x24) == "int ()"
    assert snsym.type_text(0x64) == "int *()"  # function returning a pointer: the nearest derivation is lowest
    assert snsym.type_text(0x94) == "int (*)()"  # pointer to a function
    assert snsym.type_text(0x7C, dims=(4,)) == "unsigned char *[4]"
    assert snsym.type_text(0xDC, dims=(4,)) == "unsigned char (*)[4]"
    assert snsym.type_text(0xD8, "CNode_t", (2,)) == "struct CNode_t (*)[2]"


def test_functions_carry_parameters_and_globals() -> None:
    stream = d94(0x203908, 2, 0x24, 0, "Turret_Setup") + func(0x203908, "Turret_Setup")
    stream += d96(0, 9, 0x18, 0, (), "CNode_t", "pNode") + d94(5, 17, 4, 0, "nMode")
    stream += struct.pack("<IBI", 0x203910, 0x90, 8) + d94(0, 1, 4, 0, "local")
    stream += struct.pack("<IBI", 0x203930, 0x92, 9) + struct.pack("<IBI", 0x203940, 0x8E, 10)
    recs, stop = snsym.records(stream)
    assert stop == len(stream)
    glob, fn = snsym.functions(recs)
    assert glob == {"global": "Turret_Setup", "address": 0x203908, "static": False, "type": "int ()"}
    params = [{"name": "pNode", "type": "struct CNode_t *"}, {"name": "nMode", "type": "int", "reg": 5}]
    want = {"name": "Turret_Setup", "address": 0x203908, "file": "A.CPP", "line": 7, "frame": 24}
    assert fn == dict(want, params=params)


def obj(record: bytes, section: int = 2) -> bytes:
    """A linked-stream definition rewritten as an object-file one: tag first, then a u16 section and the value."""
    value, tag = struct.unpack_from("<IB", record)
    return bytes([tag - 0x42]) + struct.pack("<HI", section, value) + record[5:]


def test_object_dialect_reads_code_patches_functions_and_definitions() -> None:
    stream = struct.pack("<BH", 0x1C, 7) + _name("C:\\SRC\\PAK.CPP") + struct.pack("<BH", 0x06, 2)
    stream += struct.pack("<BH", 0x02, 4) + b"\x27\xbd\xff\xe8"  # four code bytes
    stream += struct.pack("<BBH", 0x0A, 0x52, 0) + struct.pack("<BBHBI", 0x2C, 0x04, 2, 0x00, 0x10)  # sect 2 + 0x10
    stream += struct.pack("<BHIHI", 0x4A, 2, 0x40, 7, 317) + struct.pack("<HIHIi", 30, 56, 31, 0xC0000000, -4)
    stream += _name("PAK_OnCreate") + obj(d96(0, 9, 0x18, 0, (), "CNode_t", "pNode"))
    stream += struct.pack("<BHII", 0x4C, 2, 0x80, 320) + b"".join(obj(STRUCT[a:b]) for a, b in _spans(STRUCT))
    recs, stop = snsym.records(stream, dialect="obj")
    assert stop == len(stream)
    assert [r.tag for r in recs[:5]] == [0x1C, 0x06, 0x02, 0x0A, 0x8C]
    (fn,) = snsym.functions(recs)
    assert fn["file"] == "C:\\SRC\\PAK.CPP" and fn["line"] == 317 and fn["frame"] == 56 and fn["section"] == 2
    assert fn["params"] == [{"name": "pNode", "type": "struct CNode_t *"}]
    assert [t["name"] for t in snsym.types(recs)] == ["CNode_t", "CNode"]
    assert snsym.find(b"\0" * 9 + stream * 4 + b"\0" * 9, minimum=20, dialect="obj") == [(9, 9 + 4 * len(stream))]
    with pytest.raises(snsym.SymError):
        snsym.object_record(struct.pack("<BBH", 0x0A, 0x52, 0) + b"\x21\x00", 0)  # odd operator code


def _spans(stream: bytes) -> list[tuple[int, int]]:
    recs, _ = snsym.records(stream)
    return [(r.at, r.end) for r in recs]
