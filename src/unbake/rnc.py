"""RNC methods 1 and 2 (Rob Northen ProPack) streams: the header, its two CRCs and the decoder, from the public format.
A stream is an 18-byte big-endian header ("RNC", method 1, unpacked size, packed size, unpacked CRC, packed CRC, leeway,
chunk count) and the packed bits. The bits are read low bit first from 16-bit little-endian words, one word of
lookahead kept above the unread bits. Each pack chunk starts with three Huffman tables (literal run, match distance,
match length: a 5-bit symbol count, then a 4-bit code length per symbol; a count of 0 keeps the previous table) and a
16-bit count of sub-chunks. A sub-chunk is a literal run copied straight from the packed bytes, then (except after the
last one) a match: distance + 1 bytes back, length + 2 bytes long. Symbols 0 and 1 are their value; symbol n >= 2 is
1 << (n - 1) plus n - 1 further bits. Both CRCs are CRC-16/ARC (reflected 0x8005, zero start)."""

from __future__ import annotations

from dataclasses import dataclass

HEADER = 18
MAGIC = b"RNC\x01"
MAGICS = (MAGIC, b"RNC\x02")


class RncError(ValueError):
    """A stream that is not a well-formed RNC stream, or whose bytes do not match its header."""


@dataclass(frozen=True)
class Header:
    unpacked: int
    packed: int
    unpacked_crc: int
    packed_crc: int
    leeway: int
    chunks: int
    method: int = 1

    @property
    def stored(self) -> int:
        """The bytes of the stream in the ROM: header plus packed bits (any alignment after it is not the stream's)."""
        return HEADER + self.packed


def _crc_table() -> tuple[int, ...]:
    table = []
    for byte in range(256):
        value = byte
        for _ in range(8):
            value = (value >> 1) ^ 0xA001 if value & 1 else value >> 1
        table.append(value)
    return tuple(table)


_CRC = _crc_table()


def crc16(data: bytes | memoryview) -> int:
    value = 0
    for byte in data:
        value = (value >> 8) ^ _CRC[(value ^ byte) & 0xFF]
    return value


def header(data: bytes | memoryview, at: int = 0) -> Header:
    """The header at `at`; accepts methods 1 and 2 with sizes that fit the data."""
    raw = bytes(data[at : at + HEADER])
    if at < 0 or len(raw) < HEADER or raw[:4] not in MAGICS:
        raise RncError(f"no RNC header at {at:#x}")
    out = Header(
        int.from_bytes(raw[4:8], "big"),
        int.from_bytes(raw[8:12], "big"),
        int.from_bytes(raw[12:14], "big"),
        int.from_bytes(raw[14:16], "big"),
        raw[16],
        raw[17],
        raw[3],
    )
    if not out.unpacked or not out.packed or at + out.stored > len(data):
        raise RncError(f"RNC header at {at:#x} has sizes outside the data")
    return out


class _Bits:
    """The bit reader: `bits` holds `left` unread bits of the current word with the next word above them."""

    def __init__(self, data: bytes) -> None:
        self.data, self.at, self.left = data, 0, 0
        self.bits = self._word()

    def _word(self) -> int:
        data, at = self.data, self.at
        if at >= len(data):
            return 0
        return data[at] | (data[at + 1] << 8 if at + 1 < len(data) else 0)

    def take(self, count: int) -> int:
        value = written = 0
        while count:
            part = min(count, 16)
            got = self.bits & ((1 << part) - 1)
            shift = part
            if shift > self.left:
                if self.at >= len(self.data):
                    raise RncError("RNC1 bits end early")
                shift -= self.left
                self.bits >>= self.left
                self.at += 2
                self.bits |= self._word() << 16
                self.left = 16
            self.bits >>= shift
            self.left -= shift
            value |= got << written
            written += part
            count -= part
        return value

    def skip_literal(self, count: int) -> bytes:
        out = self.data[self.at : self.at + count]
        if len(out) != count:
            raise RncError("RNC1 literal run past the packed bytes")
        self.at += count
        self.bits = (self.bits & ((1 << self.left) - 1)) | (self._word() << self.left)
        return out


def _reverse(value: int, bits: int) -> int:
    out = 0
    for _ in range(bits):
        out = out << 1 | (value & 1)
        value >>= 1
    return out


def _table(reader: _Bits, previous: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """(bits, reversed code, symbol) entries of one Huffman table in canonical order."""
    symbols = reader.take(5)
    if not symbols:
        return previous
    lengths = [reader.take(4) for _ in range(symbols)]
    out: list[tuple[int, int, int]] = []
    code = 0
    for bits in range(1, 16):
        for symbol, length in enumerate(lengths):
            if length != bits:
                continue
            if code >= 1 << bits:
                raise RncError("RNC1 Huffman table is over-full")
            out.append((bits, _reverse(code, bits), symbol))
            code += 1
        code <<= 1
    return out


def _number(reader: _Bits, table: list[tuple[int, int, int]]) -> int:
    for bits, code, symbol in table:
        if reader.bits & ((1 << bits) - 1) == code:
            reader.take(bits)
            if symbol < 2:
                return symbol
            extra = symbol - 1
            return reader.take(extra) | (1 << extra)
    raise RncError("RNC1 bits match no Huffman code")


def unpack(data: bytes | memoryview, at: int = 0, check: bool = True) -> bytes:
    """The decoded bytes of the stream at `at`. With `check`, both CRCs and the unpacked size must match the header."""
    head = header(data, at)
    packed = bytes(data[at + HEADER : at + head.stored])
    if check and crc16(packed) != head.packed_crc:
        raise RncError(f"RNC{head.method} stream at {at:#x}: packed CRC mismatch")
    if head.method == 2:
        from unbake import rnc2

        try:
            decoded = rnc2.unpack(packed, head.unpacked)
        except ValueError as error:
            raise RncError(str(error)) from error
        if check and crc16(decoded) != head.unpacked_crc:
            raise RncError(f"RNC2 stream at {at:#x}: unpacked CRC mismatch")
        return decoded
    reader = _Bits(packed)
    reader.take(2)
    out = bytearray()
    literal: list[tuple[int, int, int]] = []
    distance: list[tuple[int, int, int]] = []
    length: list[tuple[int, int, int]] = []
    while reader.at < len(packed):
        literal = _table(reader, literal)
        distance = _table(reader, distance)
        length = _table(reader, length)
        chunks = reader.take(16)
        if not chunks:
            raise RncError("RNC1 pack chunk with no sub-chunks")
        for chunk in range(chunks):
            run = _number(reader, literal)
            if run:
                if len(out) + run > head.unpacked:
                    raise RncError("RNC1 output past the unpacked size")
                out += reader.skip_literal(run)
            if chunk + 1 == chunks:
                break
            back = _number(reader, distance) + 1
            count = _number(reader, length) + 2
            if back > len(out) or len(out) + count > head.unpacked:
                raise RncError("RNC1 match outside the output")
            start = len(out) - back
            if back >= count:
                out += out[start : start + count]
            else:
                for i in range(count):
                    out.append(out[start + i])
    if len(out) != head.unpacked:
        raise RncError(f"RNC1 stream at {at:#x} decodes to {len(out)} of {head.unpacked} bytes")
    if check and crc16(bytes(out)) != head.unpacked_crc:
        raise RncError(f"RNC1 stream at {at:#x}: unpacked CRC mismatch")
    return bytes(out)


def find(data: bytes | memoryview, start: int = 0, end: int | None = None) -> list[tuple[int, Header]]:
    """Every RNC header in `[start, end)` whose stream lies inside it (CRCs not yet checked)."""
    end = len(data) if end is None else end
    raw = bytes(data[:end])
    out = []
    at = raw.find(b"RNC", start)
    while at != -1:
        try:
            head = header(raw, at)
        except RncError:
            pass
        else:
            out.append((at, head))
        at = raw.find(b"RNC", at + 1)
    return out
