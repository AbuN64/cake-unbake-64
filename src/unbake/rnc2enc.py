"""A deterministic RNC method 2 (ProPack) encoder that reproduces the original packer's output byte for byte.

The rules were read off the 32 whole method 2 streams of the German ROM (every packed byte of all 32 is reproduced) and
are the settings a receipt names:
- tokens: a literal (flag bit 0, one byte), a match (length 2 to MAX_LENGTH, distance 1 to WINDOW), a raw run (3 to 15
  groups of 4 literals) and a chunk end marker; the control bits are taken from a control byte opened when the first
  bit is needed, so raw bytes and control bytes interleave the way `rnc2.unpack` reads them;
- a match at a position is the longest one inside WINDOW bytes, the nearest of the longest; length 2 needs a distance of
  at most 256 (it has no distance bits);
- one step of lazy matching: the packer takes a literal when the match one byte on is longer, looking only at sources
  at least 2 bytes back (a run that starts at the next byte is not seen), except that a match farther than FAR loses to
  such a run;
- a run of literals of at least RAW_MIN bytes is written as its (count mod 4) leading literals, raw runs of at most
  RAW_MAX bytes (multiples of 4, at least RAW_MIN) and the rest as literals;
- a chunk holds at most CHUNK output bytes: a marker (more = 1) is written before the first token that starts at or ends
  after the chunk limit; the stream ends with a marker (more = 0).
The header's leeway byte is not derived (it is 2 or 4 in the streams read and no function of the tokens was found)."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from typing import Any

from unbake import rnc

WINDOW = 4096  # farthest source of a match
MAX_LENGTH = 263  # longest match: 8 plus a length byte
CHUNK = 0x3000  # output bytes per chunk
RAW_MIN = 12  # the shortest raw run (and the shortest literal run written as one)
RAW_MAX = 72
FAR = 3072  # a match farther than this loses to a run starting at the next byte
Token = tuple[Any, ...]  # ('L', at) ('M', at, length, distance) ('R', at, count) ('E', at, more)


def _best(data: bytes, i: int, index: dict[bytes, list[int]], skip: int = 0) -> tuple[int, int]:
    """(length, distance) of the longest, then nearest, match at `i` from sources at least `skip + 1` bytes back."""
    n = len(data)
    if i + 2 > n:
        return 0, 0
    length, distance = 0, 0
    limit = min(MAX_LENGTH, n - i)
    for j in reversed(index.get(data[i : i + 2], ())):
        d = i - j
        if d <= skip:
            continue
        if d > WINDOW:
            break
        size = 2
        while size < limit and data[i + size] == data[i + size - d]:
            size += 1
        if size > length and (size > 2 or d <= 256):
            length, distance = size, d
            if size == limit:
                break
    return length, distance


def parse(data: bytes, chunk: int = CHUNK) -> list[Token]:
    """The tokens the packer writes for `data` (see `Token`)."""
    n = len(data)
    index: dict[bytes, list[int]] = {}
    tokens: list[Token] = []
    pending: list[int] = []
    indexed = 0
    limit = chunk

    def feed(upto: int) -> None:
        nonlocal indexed
        while indexed < min(upto, n - 1):
            index.setdefault(data[indexed : indexed + 2], []).append(indexed)
            indexed += 1

    def flush() -> None:
        count = len(pending)
        if count < RAW_MIN:
            tokens.extend(("L", p) for p in pending)
        else:
            lead = count % 4
            tokens.extend(("L", p) for p in pending[:lead])
            at, rest = pending[0] + lead, count - lead
            while rest >= RAW_MIN:
                size = min(RAW_MAX, rest)
                tokens.append(("R", at, size))
                at, rest = at + size, rest - size
            tokens.extend(("L", p) for p in range(at, at + rest))
        pending.clear()

    i = 0
    while i < n:
        if i >= limit:
            flush()
            tokens.append(("E", i, 1))
            limit += chunk
        feed(i)
        length, distance = _best(data, i, index)
        if length:
            feed(i + 1)
            if _best(data, i + 1, index, skip=1)[0] > length:
                length = 0
            elif distance > FAR:
                ahead, back = _best(data, i + 1, index)
                if ahead > length and back == 1:
                    length = 0
        if length:
            flush()
            if i + length > limit:
                tokens.append(("E", i, 1))
                limit += chunk
            tokens.append(("M", i, length, distance))
            i += length
        else:
            pending.append(i)
            i += 1
    flush()
    tokens.append(("E", n, 0))
    return tokens


class _Bits:
    def __init__(self) -> None:
        self.data = bytearray()
        self.slot = self.used = 0
        self.put(0, 2)

    def put(self, value: int, count: int) -> None:
        for shift in range(count - 1, -1, -1):
            if self.used == 0:
                self.slot = len(self.data)
                self.data.append(0)
            self.data[self.slot] |= ((value >> shift) & 1) << (7 - self.used)
            self.used = (self.used + 1) % 8


_LENGTH_BITS = {2: (6, 3), 3: (14, 4), 4: (8, 4), 5: (10, 4), 6: (18, 5), 7: (19, 5), 8: (22, 5)}


def emit(data: bytes, tokens: Sequence[Token]) -> tuple[bytes, list[int]]:
    """The packed bytes of `tokens` over `data` and, per token, how many packed bytes precede it (the control byte its
    first bit opens is counted before it)."""
    out = _Bits()
    before: list[int] = []
    for t in tokens:
        before.append(len(out.data))
        kind = t[0]
        if kind == "L":
            out.put(0, 1)
            out.data.append(data[t[1]])
        elif kind == "R":
            at, count = t[1], t[2]
            out.put(0b10111, 5)
            out.put((count - RAW_MIN) // 4, 4)
            out.data.extend(data[at : at + count])
        elif kind == "E":
            out.put(15, 4)
            out.data.append(0)
            out.put(int(t[2]), 1)
        else:
            length, distance = t[2], t[3]
            if length in _LENGTH_BITS:
                out.put(*_LENGTH_BITS[length])
            else:
                out.put(15, 4)
                out.data.append(length - 8)
            high, low = divmod(distance - 1, 256)
            if length != 2:
                if high == 0:
                    out.put(0, 1)
                elif high == 1:
                    out.put(6, 3)
                elif high < 4:
                    out.put(8 + high - 2, 4)
                else:
                    value = high if high < 8 else high // 2
                    for bit in ((1, 1), ((value >> 1) & 1, 1), (1, 1), (value & 1, 1), (int(high < 8), 1)):
                        out.put(*bit)
                    if high >= 8:
                        out.put(high & 1, 1)
            out.data.append(low)
    return bytes(out.data), before


def pack(data: bytes) -> bytes:
    """The packed bits (no header) the original packer writes for `data`."""
    return emit(data, parse(data))[0]


def stream(data: bytes, leeway: int) -> bytes:
    """A whole stream: the 18-byte header (unpacked and packed sizes, both CRCs, leeway, chunk count) and the bits."""
    tokens = parse(data)
    packed = emit(data, tokens)[0]
    chunks = 1 + sum(1 for t in tokens if t[0] == "E" and t[2])
    head = struct.pack(
        ">4sIIHHBB", b"RNC\x02", len(data), len(packed), rnc.crc16(data), rnc.crc16(packed), leeway, chunks
    )
    return head + packed


def end(token: Token) -> int:
    """The output position after a token."""
    kind, at = token[0], token[1]
    return int(at + (1 if kind == "L" else token[2] if kind in "RM" else 0))
