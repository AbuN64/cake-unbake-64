"""SN Systems debug symbol records (the stream SN's linker emits after a linked image: line numbers, functions and
COFF-style type definitions), read from the public record layout. Every record is a little-endian u32 value and a tag
byte, then the tag's fields (u16/u32 little-endian, names as a length byte and that many characters, no alignment).
Tags: 0x01/0x02 symbol/label (name); 0x80 line + 1; 0x82/0x84/0x86 line + u8 / + u16 / = u32; 0x88 line = u32 and a
file name; 0x8A line end; 0x8C function start (frame register u16, frame size u32, return register u16, mask u32, mask
offset u32, line u32, file, name); 0x9C the same with a float mask u32 and its offset u32 before the line;
0x8E/0x90/0x92 function end / block start / block end (line u32); 0x94 definition (class u16, type u16, size u32,
name); 0x96 definition with dimensions and tag (class, type, size, dimension count u16, that many u32 dimensions, tag
name, name); 0x98 overlay (length u32, id u32); 0x9A overlay switch.
A definition's type is a COFF type word: base in bits 0-3, then 2-bit derivations from bit 4 up, nearest the name first
(1 pointer, 2 function, 3 array). Classes: 8 struct member, 10 struct tag, 11 union member, 12 union tag, 13 typedef,
15 enum tag, 16 enumerator, 18 bit field (value = bit offset, size = bit width), 102 end of struct (value = size)."""

from __future__ import annotations

import hashlib
import struct
from collections import Counter
from dataclasses import dataclass, field

BASES = (
    "null", "void", "char", "short", "int", "long", "float", "double",
    "struct", "union", "enum", "enumerator", "unsigned char", "unsigned short", "unsigned int", "unsigned long",
)  # fmt: skip
_FIXED = {0x80: "", 0x82: "B", 0x84: "H", 0x86: "I", 0x8A: "", 0x8E: "I", 0x90: "I", 0x92: "I", 0x98: "II", 0x9A: ""}
_KINDS = {10: "struct", 12: "union", 15: "enum"}
_MEMBERS = {8, 11, 16, 18}
EOS, TYPEDEF = 102, 13
CLASSES = frozenset(range(19)) | frozenset(range(100, 108))  # COFF storage classes; anything else is not a record


class SymError(ValueError):
    """Bytes that are not a whole SN debug record."""


@dataclass(frozen=True)
class Record:
    at: int
    end: int
    value: int
    tag: int
    cls: int = 0
    type: int = 0
    size: int = 0
    dims: tuple[int, ...] = ()
    tag_name: str = ""
    name: str = ""
    extra: tuple[int, ...] = field(default=())
    section: int = -1  # object dialect: the section the value is relative to (-1: a linked stream's absolute value)
    file: int = -1  # object dialect: the file number of a function start (its 0x1C file record names it)


def _name(data: bytes | memoryview, at: int) -> tuple[str, int]:
    if at >= len(data):
        raise SymError(f"name past the end at {at:#x}")
    n = data[at]
    raw = bytes(data[at + 1 : at + 1 + n])
    if len(raw) != n or any(c < 0x20 or c > 0x7E for c in raw):
        raise SymError(f"name at {at:#x} is not {n} printable characters")
    return raw.decode("ascii"), at + 1 + n


def _check(at: int, cls: int, size: int, dims: tuple[int, ...]) -> None:
    if cls not in CLASSES or size >> 24 or len(dims) > 8 or any(d >> 24 for d in dims):
        raise SymError(f"definition at {at:#x} has class {cls}, size {size:#x}, dimensions {dims}: not a record")


def record(data: bytes | memoryview, at: int) -> Record:
    """The one record starting at `at`; SymError when it is not whole and well formed."""
    try:
        value, tag = struct.unpack_from("<IB", data, at)
        q = at + 5
        if tag in (1, 2):
            name, q = _name(data, q)
            return Record(at, q, value, tag, name=name)
        if tag in _FIXED:
            extra = struct.unpack_from("<" + _FIXED[tag], data, q)
            return Record(at, q + struct.calcsize("<" + _FIXED[tag]), value, tag, extra=extra)
        if tag == 0x88:
            (line,) = struct.unpack_from("<I", data, q)
            name, q = _name(data, q + 4)
            return Record(at, q, value, tag, name=name, extra=(line,))
        if tag == 0x8C:
            extra = struct.unpack_from("<HIHIiI", data, q)
            file, q = _name(data, q + 20)
            name, q = _name(data, q)
            return Record(at, q, value, tag, tag_name=file, name=name, extra=extra)
        if tag == 0x9C:
            extra = struct.unpack_from("<HIHIiIiI", data, q)
            file, q = _name(data, q + 28)
            name, q = _name(data, q)
            return Record(at, q, value, tag, tag_name=file, name=name, extra=extra)
        if tag == 0x94:
            cls, typ, size = struct.unpack_from("<HHI", data, q)
            _check(at, cls, size, ())
            name, q = _name(data, q + 8)
            return Record(at, q, value, tag, cls, typ, size, name=name)
        if tag == 0x96:
            cls, typ, size, count = struct.unpack_from("<HHIH", data, q)
            dims = struct.unpack_from(f"<{count}I", data, q + 10)
            _check(at, cls, size, dims)
            tag_name, q = _name(data, q + 10 + 4 * count)
            name, q = _name(data, q)
            return Record(at, q, value, tag, cls, typ, size, tuple(dims), tag_name, name)
    except struct.error as error:
        raise SymError(f"record at {at:#x} runs past the end") from error
    raise SymError(f"unknown tag {tag:#x} at {at:#x}")


_OBJ_FIXED = {0x06: "H", 0x08: "I", 0x2E: "B", 0x30: "HI", 0x32: "H", 0x34: "HB", 0x36: "HH", 0x38: "HI", 0x3A: "HIH",
              0x3C: "H"}  # fmt: skip
_OBJ_LEAF = {0x00: "I", 0x02: "H", 0x04: "H", 0x0C: "H", 0x16: "H"}  # constant, symbol, section base/start/end


def _expression(data: bytes | memoryview, at: int, depth: int = 0) -> int:
    """The end of one patch expression: a leaf (constant, symbol or section reference) or an even operator code from
    0x20 to 0x5A followed by its two operands."""
    op = data[at]
    if op in _OBJ_LEAF:
        return at + 1 + struct.calcsize("<" + _OBJ_LEAF[op])
    if 0x20 <= op <= 0x5A and op % 2 == 0 and depth < 16:
        return _expression(data, _expression(data, at + 1, depth + 1), depth + 1)
    raise SymError(f"patch expression operator {op:#x} at {at:#x}")


def object_record(data: bytes | memoryview, at: int) -> Record:
    """The one record of an SN/Psy-Q object file (the "LNK" format) starting at `at`: the tag byte comes first and
    definitions carry a u16 section before their u32 value. Tags: 0x02 code (u16 length, bytes), 0x06 section switch,
    0x08 uninitialised size, 0x0A patch (u8 kind, u16 offset, expression), 0x0C exported symbol (u16 number, u16
    section, u32 offset, name), 0x0E imported symbol (u16 number, name), 0x10 section (u16 number, u16 group, u8
    alignment, name), 0x12 local symbol (u16 section, u32 offset, name), 0x14 group (u16, u8, name), 0x1C file (u16
    number, name), 0x2E cpu, 0x30 constant, 0x32-0x3C line numbers, 0x4A function start (u16 section, u32 offset, u16
    file, u32 line, then the 0x8C fields and name), 0x4C/0x4E/0x50 function end / block start / block end (u16 section,
    u32 offset, u32 line), 0x52/0x54 the 0x94/0x96 definitions. Definitions and function records come back with the
    linked stream's tag (0x94, 0x96, 0x8C, 0x8E, 0x90, 0x92) so `types` and `functions` read both dialects."""
    try:
        tag = data[at]
        q = at + 1
        if tag in _OBJ_FIXED:
            fmt = "<" + _OBJ_FIXED[tag]
            return Record(at, q + struct.calcsize(fmt), 0, tag, extra=struct.unpack_from(fmt, data, q))
        if tag == 0x02:
            (n,) = struct.unpack_from("<H", data, q)
            if q + 2 + n > len(data):
                raise SymError(f"code record at {at:#x} runs past the end")
            return Record(at, q + 2 + n, 0, tag, size=n)
        if tag == 0x0A:
            return Record(at, _expression(data, q + 3), 0, tag, extra=struct.unpack_from("<BH", data, q))
        if tag in (0x0C, 0x0E, 0x10, 0x12, 0x14, 0x1C):
            fmt = {0x0C: "<HHI", 0x0E: "<H", 0x10: "<HHB", 0x12: "<HI", 0x14: "<HB", 0x1C: "<H"}[tag]
            extra = struct.unpack_from(fmt, data, q)
            name, q = _name(data, q + struct.calcsize(fmt))
            return Record(at, q, extra[-1] if tag in (0x0C, 0x12) else 0, tag, name=name, extra=extra)
        if tag in (0x4C, 0x4E, 0x50):
            section, value, line = struct.unpack_from("<HII", data, q)
            return Record(at, q + 10, value, tag + 0x42, extra=(line,), section=section)
        if tag == 0x4A:
            section, value, file, line = struct.unpack_from("<HIHI", data, q)
            extra = struct.unpack_from("<HIHIi", data, q + 12)
            name, q = _name(data, q + 28)
            return Record(at, q, value, 0x8C, name=name, extra=(*extra, line), section=section, file=file)
        if tag in (0x52, 0x54):
            section, value = struct.unpack_from("<HI", data, q)
            rec = record(bytes(data[q + 2 : q + 6]) + bytes([tag + 0x42]) + bytes(data[q + 6 : q + 6 + 1024]), 0)
            return Record(at, q + 1 + rec.end, value, rec.tag, rec.cls, rec.type, rec.size, rec.dims, rec.tag_name,
                          rec.name, section=section)  # fmt: skip
    except (struct.error, IndexError) as error:
        raise SymError(f"object record at {at:#x} runs past the end") from error
    raise SymError(f"unknown object tag {tag:#x} at {at:#x}")


def records(
    data: bytes | memoryview, at: int = 0, end: int | None = None, dialect: str = "sym"
) -> tuple[list[Record], int]:
    """The records from `at` while whole ones fit before `end`, and where the run stopped: `end` when the stream
    reaches it exactly, else the start of the first byte run that is not a whole record (a cut or the stream's end)."""
    end = len(data) if end is None else end
    out: list[Record] = []
    while at < end:
        try:
            rec = (record if dialect == "sym" else object_record)(memoryview(data)[:end], at)
        except SymError:
            break
        out.append(rec)
        at = rec.end
    return out, at


def find(
    data: bytes | memoryview, start: int = 0, end: int | None = None, minimum: int = 50, dialect: str = "sym"
) -> list[tuple[int, int]]:
    """The record runs of at least `minimum` whole records in [start, end), as [run start, run stop) spans. A run
    found from a definition tag is extended back to the earliest start whose records land on it (at most 2 KiB back,
    one record never being longer), so a run cut at its front starts at its first whole record."""
    tags, lead = (b"\x94", b"\x96"), 4
    if dialect != "sym":
        tags, lead = (b"\x52", b"\x54"), 0
    raw = bytes(data)
    end = len(raw) if end is None else end
    out: list[tuple[int, int]] = []
    at = start
    while at < end:
        recs, stop = records(raw, at, end, dialect)
        if len(recs) < minimum:
            hits = [x for x in (raw.find(t, at + lead + 1, end + lead) for t in tags) if x >= 0]
            at = min(hits) - lead if hits else end
            continue
        first = at
        for back in range(at - 1, max(out[-1][1] if out else start, at - 0x800) - 1, -1):
            got, land = records(raw, back, at, dialect)
            if land == at and got:
                first = back
        out.append((first, stop))
        at = stop + 1
    return out


def _pname(text: str) -> bytes:
    raw = text.encode("ascii")
    if len(raw) > 255:
        raise SymError(f"name {text!r} is longer than 255 characters")
    return bytes([len(raw)]) + raw


def encode(rec: Record) -> bytes:
    """The bytes of one linked-stream record (`record` inverse): the fields written in the stream's own layout, no
    alignment. The object dialect is not invertible (its records come back normalised) and is refused."""
    head = struct.pack("<IB", rec.value, rec.tag)
    tag = rec.tag
    if rec.section >= 0 or rec.file >= 0:
        raise SymError(f"record at {rec.at:#x} is object dialect: not invertible")
    if tag in (1, 2):
        return head + _pname(rec.name)
    if tag in _FIXED:
        return head + struct.pack("<" + _FIXED[tag], *rec.extra)
    if tag == 0x88:
        return head + struct.pack("<I", *rec.extra) + _pname(rec.name)
    if tag in (0x8C, 0x9C):
        fmt = "<HIHIiI" if tag == 0x8C else "<HIHIiIiI"
        return head + struct.pack(fmt, *rec.extra) + _pname(rec.tag_name) + _pname(rec.name)
    if tag == 0x94:
        return head + struct.pack("<HHI", rec.cls, rec.type, rec.size) + _pname(rec.name)
    if tag == 0x96:
        return (
            head
            + struct.pack(f"<HHIH{len(rec.dims)}I", rec.cls, rec.type, rec.size, len(rec.dims), *rec.dims)
            + _pname(rec.tag_name)
            + _pname(rec.name)
        )
    raise SymError(f"unknown tag {tag:#x} at {rec.at:#x}")


def emit(recs: list[Record]) -> bytes:
    """The stream of `recs` (`records` inverse). `emit(records(data, at, end)[0]) == data[at:stop]` for any stream the
    reader parses: every byte of a record is a field, so the round trip is exact or `encode` raises."""
    return b"".join(encode(rec) for rec in recs)


def cut_tail(data: bytes | memoryview, at: int, end: int) -> dict[str, int | str]:
    """Prove `data[at:end]`, the bytes after the last whole record of a stream, is a strict prefix of one valid record:
    the reader refuses it for running past the end, yet appending filler (zeros for numeric fields, then printable
    text) completes a record whose encoding reproduces every byte of the tail. Returns the tag, how many bytes the
    record still needs and the tail's hex; SymError when the tail is no prefix of any record (a cut before the tag
    byte proves nothing, so it is refused)."""
    tail = bytes(data[at:end])
    if not tail:
        raise SymError(f"no tail at {at:#x}")
    try:
        record(tail, 0)
    except SymError:
        pass
    else:
        raise SymError(f"tail at {at:#x} is a whole record")
    for zeros in range(40):  # zeros finish the numeric fields, printable text finishes the names
        for text in range(300):
            padded = tail + bytes(zeros) + b"A" * text
            if zeros + text == 0:
                continue
            try:
                rec = record(padded, 0)
            except SymError:
                continue
            if rec.end > len(tail) and encode(rec) == padded[: rec.end]:
                return {"at": at, "bytes": len(tail), "tag": rec.tag, "needs": rec.end - len(tail), "hex": tail.hex()}
    raise SymError(f"tail at {at:#x} is not the start of any record")


def cut_head(data: bytes | memoryview, start: int, size: int) -> dict[str, int | str]:
    """Prove `data[start:start+size]`, the first bytes of a stream that a 4 KiB write cut from the previous page, is the
    end of the record that straddles `start`: some earlier record start (at most 2 KiB back) chains whole records
    onto exactly `start + size`, and the last of them begins before `start`. Returns where that record begins."""
    for back in range(start - 1, max(0, start - 0x800) - 1, -1):
        recs, stop = records(data, back, start + size)
        if stop == start + size and len(recs) >= 2 and recs[-1].at < start:
            return {"at": start, "bytes": size, "record": recs[-1].at, "tag": recs[-1].tag}
    # A later section write can overwrite the prefix of the record. Require an intact
    # witness elsewhere: the same suffix followed by a whole record, with a valid
    # containing record whose canonical encoding agrees with its source bytes.
    following, _ = records(data, start + size, min(len(data), start + size + 2048))
    if following:
        witness = bytes(data[start : following[0].end])
        raw = bytes(data)
        at = raw.find(witness)
        while at >= 0:
            if at != start:
                for back in range(at - 1, max(-1, at - 0x800), -1):
                    try:
                        rec = record(data, back)
                    except SymError:
                        continue
                    if rec.end == at + size and encode(rec) == raw[back : rec.end]:
                        return {
                            "at": start,
                            "bytes": size,
                            "record": back,
                            "tag": rec.tag,
                            "source": at,
                            "witness_end": at + len(witness),
                        }
            at = raw.find(witness, at + 1)
    raise SymError(f"head at {start:#x} is no record's end")


def stream_start(data: bytes | memoryview, first: int, floor: int) -> int:
    """Include a cut leading record only when the boundary has a checked suffix witness."""
    if 0 <= floor < first and first - floor <= 255:
        try:
            cut_head(data, floor, first - floor)
        except SymError:
            pass
        else:
            return floor
    return first


def object_receipt(data: bytes | memoryview, start: int, end: int) -> dict[str, object]:
    """The tiling proof of an object-file stream (the "LNK" dialect, whose records come back normalised and cannot be
    re-emitted): `data[start:end]` is cut into whole well-formed records from its first byte to its last, none of them
    running past `end`, and a stream with fewer than 50 records proves nothing. The receipt holds the record count, the
    tag census and the sha256 of the slice. Raises SymError when the records do not reach `end` exactly."""
    recs, stop = records(data, start, end, "obj")
    if len(recs) < 50:
        raise SymError(f"only {len(recs)} object records parse in {start:#x}..{end:#x}")
    if stop != end:
        raise SymError(f"object records stop at {stop:#x}, short of {end:#x}")
    raw = bytes(data[start:end])
    return {
        "start": start, "end": end, "bytes": end - start, "dialect": "obj", "records": len(recs),
        "records_at": start, "records_end": stop, "slice_sha256": hashlib.sha256(raw).hexdigest(),
        "tags": {f"{tag:#04x}": n for tag, n in sorted(Counter(r.tag for r in recs).items())},
    }  # fmt: skip


def stream_receipt(data: bytes | memoryview, start: int, end: int, dialect: str = "sym") -> dict[str, object]:
    """The round-trip proof of the stream in `data[start:end]`: the whole records, the head bytes of a record a 4 KiB
    write cut at the front (`cut_head`), the tail of a record cut at the back (`cut_tail`), and the sha256 of the
    slice against the sha256 of head + `emit(records)` + tail. Raises SymError when the re-emit differs, a head or tail
    is no record prefix/suffix, or no record parses. The head is the offset (0..15) whose records reach furthest."""
    if dialect == "obj":
        return object_receipt(data, start, end)
    best = max(range(16), key=lambda d: records(data, start + d, end)[1] if start + d < end else -1)
    recs, stop = records(data, start + best, end)
    if not recs:
        raise SymError(f"no record parses in {start:#x}..{end:#x}")
    raw = bytes(data[start:end])
    body = emit(recs)
    if body != raw[best : stop - start]:
        raise SymError(f"re-emit of {start:#x}..{end:#x} differs from the stream")
    receipt: dict[str, object] = {
        "start": start, "end": end, "bytes": end - start, "records": len(recs), "records_at": start + best,
        "records_end": stop, "records_bytes": len(body), "records_sha256": hashlib.sha256(body).hexdigest(),
        "slice_sha256": hashlib.sha256(raw).hexdigest(),
        "tags": {f"{tag:#04x}": n for tag, n in sorted(Counter(r.tag for r in recs).items())},
    }  # fmt: skip
    if best:
        receipt["head"] = {**cut_head(data, start, best), "hex": raw[:best].hex()}
    if stop < end:
        receipt["tail"] = cut_tail(data, stop, end)
    again = raw[:best] + body + raw[stop - start :]
    receipt["reemit_sha256"] = hashlib.sha256(again).hexdigest()
    if again != raw:
        raise SymError(f"head + re-emit + tail of {start:#x}..{end:#x} differs from the stream")
    return receipt


def type_text(code: int, tag: str = "", dims: tuple[int, ...] = ()) -> str:
    """A COFF type word as a C abstract declarator: `struct CNode_t *`, `unsigned long [128]`, `int (*)()`."""
    base = BASES[code & 0xF]
    if code & 0xF in (8, 9, 10):
        base = f"{base} {tag}".strip() if base != "enumerator" else base
    decl, left = "", list(dims)
    code >>= 4
    while code:
        kind = code & 3
        if kind == 1:
            decl = "*" + decl
        elif kind == 2:
            decl = (f"({decl})" if decl.startswith("*") else decl) + "()"
        elif kind == 3:
            decl = (f"({decl})" if decl.startswith("*") else decl) + (f"[{left.pop(0)}]" if left else "[]")
        code >>= 2
    return f"{base} {decl}".rstrip()


def types(recs: list[Record]) -> list[dict[str, object]]:
    """The type definitions of a record run, in stream order: struct/union/enum with their members (name, offset,
    type, bits for bit fields, value for enumerators) and typedefs (name, type). A definition still open where the run
    stops is kept with `cut: true`."""
    out: list[dict[str, object]] = []
    open_: dict[str, object] | None = None
    for rec in recs:
        if rec.tag not in (0x94, 0x96):
            continue
        if rec.cls in _KINDS:
            open_ = {"name": rec.name, "kind": _KINDS[rec.cls], "size": rec.size, "fields": [], "at": rec.at}
            out.append(open_)
        elif rec.cls in _MEMBERS and open_ is not None:
            member: dict[str, object] = {"name": rec.name}
            if rec.cls == 16:
                member["value"] = rec.value - (1 << 32) if rec.value >> 31 else rec.value
            else:
                member["offset"] = rec.value // 8 if rec.cls == 18 else rec.value
                member["type"] = type_text(rec.type, rec.tag_name, rec.dims)
                if rec.cls == 18:
                    member["bit"], member["bits"] = rec.value % 8, rec.size
                else:
                    member["size"] = rec.size
            fields = open_["fields"]
            assert isinstance(fields, list)
            fields.append(member)
        elif rec.cls == EOS and open_ is not None:
            open_ = None
        elif rec.cls == TYPEDEF:
            out.append({"name": rec.name, "kind": "typedef", "size": rec.size, "type": type_text(rec.type, rec.tag_name,
                        rec.dims), "at": rec.at})  # fmt: skip
    if open_ is not None:
        open_["cut"] = True
    return out


def functions(recs: list[Record]) -> list[dict[str, object]]:
    """The functions of a record run, in stream order: name, address, source file, first line, frame size and the
    parameters (name, type, `reg` for a register parameter) defined before the function's first block or its end.
    Definitions of classes 2 and 3 (external, static) outside a function come back as `{"global": ...}` rows."""
    out: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    files = {rec.extra[0]: rec.name for rec in recs if rec.tag == 0x1C and rec.section < 0 and rec.extra}
    for rec in recs:
        if rec.tag in (0x8C, 0x9C):
            line = rec.extra[-1]
            file = rec.tag_name if rec.file < 0 else files.get(rec.file, f"file {rec.file}")
            current = {"name": rec.name, "address": rec.value, "file": file, "line": line, "frame": rec.extra[1],
                       "params": []}  # fmt: skip
            if rec.section >= 0:
                current["section"] = rec.section
            out.append(current)
        elif rec.tag in (0x8E, 0x90):
            current = None
        elif rec.tag in (0x94, 0x96) and rec.cls in (9, 17) and current is not None:
            params = current["params"]
            assert isinstance(params, list)
            param: dict[str, object] = {"name": rec.name, "type": type_text(rec.type, rec.tag_name, rec.dims)}
            if rec.cls == 17:
                param["reg"] = rec.value
            params.append(param)
        elif rec.tag in (0x94, 0x96) and rec.cls in (2, 3) and current is None:
            out.append({"global": rec.name, "address": rec.value, "static": rec.cls == 3,
                        "type": type_text(rec.type, rec.tag_name, rec.dims)})  # fmt: skip
    return out
