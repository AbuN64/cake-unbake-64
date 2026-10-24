"""Bounded RNC2 token reader. Control bytes are MSB first; raw bytes share their cursor.

Token prefixes: 0 literal; 1000/1010 matches of 4/5; 10010/10011 matches of 6/7;
10110 match of 8; 10111 raw run; 110 match of 2; 1110 match of 3; 1111 long match
or chunk/end marker. Distance high bits have a separate prefix code (0..15).
Reference format: https://dsll32.github.io/nviewer/wiki/compression/rnc.html
"""

from __future__ import annotations

from dataclasses import dataclass

# Prefixes include the leading nonliteral bit. No speculative reads across raw bytes.
_LENGTHS = {"1000": 4, "1010": 5, "10010": 6, "10011": 7, "10110": 8, "10111": 0, "110": 2, "1110": 3, "1111": -1}
_HIGH = {
    "0": 0,
    "110": 1,
    "1000": 2,
    "1001": 3,
    "10101": 4,
    "10111": 5,
    "11101": 6,
    "11111": 7,
    "101000": 8,
    "101001": 9,
    "101100": 10,
    "101101": 11,
    "111000": 12,
    "111001": 13,
    "111100": 14,
    "111101": 15,
}


class _Reader:
    def __init__(self, packed: bytes) -> None:
        self.packed = packed
        self.cursor = self.control = self.remaining = 0

    def raw(self, size: int) -> bytes:
        end = self.cursor + size
        if end > len(self.packed):
            raise ValueError("RNC2 packed bytes end early")
        result = self.packed[self.cursor : end]
        self.cursor = end
        return result

    def bits(self, count: int) -> int:
        result = 0
        for _ in range(count):
            if not self.remaining:
                self.control = self.raw(1)[0]
                self.remaining = 8
            self.remaining -= 1
            result = result * 2 + ((self.control >> self.remaining) & 1)
        return result

    def prefix(self, table: dict[str, int], initial: str = "") -> int:
        code = initial
        while code not in table:
            code += str(self.bits(1))
            if len(code) > 6:
                raise ValueError("RNC2 invalid token prefix")
        return table[code]


def unpack(packed: bytes, size: int) -> bytes:
    """Decode exactly one unprotected stream; enforce termination, bounds and complete consumption."""
    reader = _Reader(packed)
    if reader.bits(2):
        raise ValueError("RNC2 locked or encrypted stream is unsupported")
    output = bytearray()
    while True:
        if reader.bits(1) == 0:
            if len(output) == size:
                raise ValueError("RNC2 literal past unpacked size")
            output += reader.raw(1)
            continue
        length = reader.prefix(_LENGTHS, "1")
        if length == 0:
            count = reader.bits(4) * 4 + 12
            if len(output) + count > size:
                raise ValueError("RNC2 raw run past unpacked size")
            output += reader.raw(count)
            continue
        if length == -1:
            length = reader.raw(1)[0]
            if length == 0:
                if reader.bits(1):
                    continue
                break
            length += 8
        high = 0 if length == 2 else reader.prefix(_HIGH)
        distance = high * 256 + reader.raw(1)[0] + 1
        if distance > len(output) or len(output) + length > size:
            raise ValueError("RNC2 match outside output")
        for _ in range(length):
            output.append(output[-distance])
    if len(output) != size:
        raise ValueError(f"RNC2 stream decodes to {len(output)} of {size} bytes")
    if reader.cursor != len(packed):
        raise ValueError("RNC2 bytes after end marker")
    return bytes(output)


@dataclass(frozen=True)
class Walk:
    """What reading the bits of one stream gave: the output, the packed bytes read, how it stopped and the tokens."""

    output: int
    cursor: int
    ended: bool
    error: str
    tokens: dict[str, int]


def walk(packed: bytes, cap: int) -> Walk:
    """Read `packed` as RNC2 tokens from the first bit, without the checks `rnc2.unpack` ends with: it stops at the end
    marker (`ended`) or at the first token that cannot be (`error`). `cap` bounds the output."""
    reader = _Reader(packed)
    size = 0
    tokens = {"literal": 0, "match": 0, "raw": 0, "chunk": 0}
    try:
        if reader.bits(2):
            raise ValueError("locked or encrypted stream")
        while True:
            if reader.bits(1) == 0:
                if size >= cap:
                    raise ValueError("literal past the output cap")
                reader.raw(1)
                size += 1
                tokens["literal"] += 1
                continue
            length = reader.prefix(_LENGTHS, "1")
            if length == 0:
                count = reader.bits(4) * 4 + 12
                if size + count > cap:
                    raise ValueError("raw run past the output cap")
                reader.raw(count)
                size += count
                tokens["raw"] += 1
                continue
            if length == -1:
                length = reader.raw(1)[0]
                if length == 0:
                    if reader.bits(1):
                        tokens["chunk"] += 1
                        continue
                    return Walk(size, reader.cursor, True, "", tokens)
                length += 8
            high = 0 if length == 2 else reader.prefix(_HIGH)
            distance = high * 256 + reader.raw(1)[0] + 1
            if distance > size or size + length > cap:
                raise ValueError("match outside the output")
            size += length
            tokens["match"] += 1
    except ValueError as error:
        return Walk(size, reader.cursor, False, str(error), tokens)
