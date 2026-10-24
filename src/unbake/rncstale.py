"""Stale RNC2 streams that later writes damaged: proves bytes are the packed bits of a Rob Northen method 2 stream by
the structure of the bits, when the stream's own checks (the packed CRC) fail and no oracle copy of its bytes exists.

`rnc` proves a stream whole: both CRCs and the unpacked size agree with the header. An older link left streams in the
ROM that a later write damaged in places, so their CRC fails for good. They are still no noise: the token grammar of
RNC2 is dense, but a stream read from its first token to its end marker is not something chance writes. A stream here
satisfies every one of these, and the receipt records each measure:
- a well-formed method 2 header (`rnc.header`: sizes inside the data) and an unpacked size of at least MIN_UNPACKED;
- the packed bits fail their CRC (a stream that passes is `rnc`'s, not this reader's);
- read token by token from the first bit with no limit but a generous output cap, the bits reach an end marker, never
  an invalid token, a match before the start of the output or a raw run past the cap;
- the end marker lies within SLACK bytes of the header's packed size (the damage moved the parse by a few bytes, never
  by a page), and the output within 1/OUTPUT_SKEW of the header's unpacked size.
A row credits a region inside the stored extent (header and packed bits) of such a stream. The proof says the bytes are
one RNC2 stream, damaged; it does not say what the damaged bytes were."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from unbake import rnc, rnc2
from unbake.contracts import sha256

MIN_UNPACKED = 1024  # the fewest bytes a stream unpacks to (a shorter one reaches its marker by chance too often)
SLACK = 8  # the most bytes between the header's packed size and where the end marker was read
OUTPUT_SKEW = 8  # the unpacked size and the output differ by at most 1/OUTPUT_SKEW of the size
CAP_FACTOR = 2  # the output may grow to this many times the header's unpacked size before the read gives up


class StreamError(ValueError):
    """Bytes that are not the packed bits of a damaged stream."""


def check(rom: bytes | memoryview, at: int) -> dict[str, Any]:
    """The measures of the stream whose header is at `at`; raises StreamError naming the first one that fails."""
    try:
        head = rnc.header(rom, at)
    except rnc.RncError as error:
        raise StreamError(str(error)) from error
    if head.method != 2:
        raise StreamError("only method 2 streams are read here")
    if head.unpacked < MIN_UNPACKED:
        raise StreamError(f"{head.unpacked} unpacked bytes are fewer than the {MIN_UNPACKED} a stream needs")
    packed = bytes(rom[at + rnc.HEADER : at + head.stored])
    window = bytes(rom[at + rnc.HEADER : min(len(rom), at + rnc.HEADER + head.packed + SLACK)])
    crc = rnc.crc16(packed)
    if crc == head.packed_crc:
        raise StreamError("the packed CRC passes: the stream is whole, a rnc reader proves it")
    got = rnc2.walk(window, head.unpacked * CAP_FACTOR)
    if not got.ended:
        raise StreamError(f"the bits reach no end marker ({got.error}) after {got.output} bytes")
    if abs(got.cursor - head.packed) > SLACK:
        raise StreamError(f"the end marker is read at {got.cursor}, the header says {head.packed} packed bytes")
    if abs(got.output - head.unpacked) * OUTPUT_SKEW > head.unpacked:
        raise StreamError(f"the output is {got.output} bytes, the header says {head.unpacked}")
    return {
        "method": head.method,
        "unpacked": head.unpacked,
        "packed": head.packed,
        "stored": head.stored,
        "packed_crc": head.packed_crc,
        "packed_crc_read": crc,
        "unpacked_crc": head.unpacked_crc,
        "end_marker_at": got.cursor,
        "output": got.output,
        "tokens": got.tokens,
    }


def find(rom: bytes | memoryview, low: int, high: int) -> list[tuple[int, int]]:
    """The streams (header offset, stored size) inside rom[low-reach:high+reach] that pass `check` and overlap
    [low, high). A header may sit a few bytes before the span (a directory prefix) or inside it."""
    found = []
    for at, head in rnc.find(rom, max(0, low - 0x10000), min(len(rom), high + 0x10000)):
        if head.method != 2 or at + head.stored <= low or at >= high:
            continue
        try:
            check(rom, at)
        except StreamError:
            continue
        found.append((at, head.stored))
    return found


def spans(found: Iterable[tuple[int, int]], targets: Iterable[tuple[int, int]]) -> list[dict[str, Any]]:
    """The rows (region inside targets, stream [header, end)) that streams give."""
    rows = []
    for low, high in targets:
        for at, stored in found:
            a, b = max(low, at), min(high, at + stored)
            if a < b:
                rows.append({"region": [a, b], "stream": [at, at + stored]})
    return sorted(rows, key=lambda r: r["region"])


def prove(rom: bytes | memoryview, start: int, end: int, stream: list[int]) -> dict[str, Any]:
    """The receipt of rom[start:end] as part of the damaged stream rom[stream[0]:stream[1]]."""
    first, last = stream
    if not 0 <= first < last <= len(rom):
        raise StreamError("a stream is a byte range inside the ROM")
    if not first <= start < end <= last:
        raise StreamError("the region is not inside its stream")
    measures = check(rom, first)
    if last != first + measures["stored"]:
        raise StreamError("the stream range is not the header's stored size")
    return {
        "stream": [first, last],
        "stream_sha256": sha256(bytes(rom[first:last])),
        **measures,
        "region": [start, end],
        "region_sha256": sha256(bytes(rom[start:end])),
    }
