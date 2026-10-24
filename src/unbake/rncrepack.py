"""Stale RNC2 streams proved by re-packing: bytes are packed bits of a method 2 stream when the packer reproduces them.

`rnc` proves a stream whole: both CRCs and the unpacked size agree with the header. An older link left streams in the
ROM that later writes damaged, so their CRC fails for good. This reader proves the part that is still intact, exactly:
the bits are decoded token by token (`rnc2.walk`, as far as they decode), the decoded bytes are packed again with the
deterministic packer `rnc2enc` (which reproduces all 32 whole method 2 streams of the German ROM byte for byte), and the
packed bytes must equal the ROM's. A byte a later write replaced inside a literal decodes to other data and packs back
to the same byte, so it stays equal; a damaged control bit or length changes the tokens, and the first byte where the
packer's output differs ends the proof. A token is credited only when the data its choice looked at (a match is at most
MAX_LENGTH long and one step of lazy matching looks one match further) is decoded as well, unless the stream decodes
whole to the header's unpacked size. Every byte past the proof stays with whatever else accounts for it. The receipt
names the decoded bytes (size, sha256), the packer's settings, the first differing byte and the token it belongs to."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from unbake import rnc, rnc2, rnc2enc
from unbake.contracts import sha256

MARGIN = 2 * rnc2enc.MAX_LENGTH + 16  # decoded bytes a token's choice may have looked at past its end
MIN_CREDIT = 64  # the fewest packed bytes a proof credits
CAP_FACTOR = 2  # the output may grow to this many times the header's unpacked size before the read gives up
SETTINGS = {
    "window": rnc2enc.WINDOW,
    "max_length": rnc2enc.MAX_LENGTH,
    "chunk": rnc2enc.CHUNK,
    "raw_min": rnc2enc.RAW_MIN,
    "raw_max": rnc2enc.RAW_MAX,
    "far": rnc2enc.FAR,
    "lazy": "one step, sources two or more bytes back; a far match loses to a run starting at the next byte",
}


class StreamError(ValueError):
    """Bytes the packer does not reproduce."""


def measure(rom: bytes | memoryview, at: int) -> dict[str, Any]:
    """The proof of the stream whose header is at `at`: how many packed bytes the packer reproduces, and the facts the
    receipt holds. Raises StreamError naming why there is none."""
    try:
        head = rnc.header(rom, at)
    except rnc.RncError as error:
        raise StreamError(str(error)) from error
    if head.method != 2:
        raise StreamError("only method 2 streams are read here")
    packed = bytes(rom[at + rnc.HEADER : at + head.stored])
    crc = rnc.crc16(packed)
    if crc == head.packed_crc:
        raise StreamError("the packed CRC passes: the stream is whole, a rnc reader proves it")
    walked = rnc2.walk(packed, head.unpacked * CAP_FACTOR)
    tokens = rnc2enc.parse(walked.data)
    mine, before = rnc2enc.emit(walked.data, tokens)
    size = min(len(mine), len(packed))
    first = next((i for i in range(size) if mine[i] != packed[i]), size)
    complete = walked.ended and len(walked.data) == head.unpacked
    credit, last = 0, -1
    for k, token in enumerate(tokens):
        stop = before[k + 1] if k + 1 < len(tokens) else len(mine)
        if stop > first or (not complete and rnc2enc.end(token) > len(walked.data) - MARGIN):
            break
        credit, last = stop, k
    if credit < MIN_CREDIT:
        raise StreamError(f"the packer reproduces {credit} packed bytes of it ({walked.error or 'no error'})")
    return {
        "method": head.method,
        "unpacked": head.unpacked,
        "packed": head.packed,
        "stored": head.stored,
        "packed_crc": head.packed_crc,
        "packed_crc_read": crc,
        "unpacked_crc": head.unpacked_crc,
        "decoded": len(walked.data),
        "decoded_sha256": sha256(walked.data),
        "decode_stopped": walked.error or "end marker",
        "tokens": len(tokens),
        "first_difference": first,
        "difference_token": last + 1,
        "credit": credit,
        "settings": SETTINGS,
    }


def find(rom: bytes | memoryview, low: int, high: int) -> list[tuple[int, int, int]]:
    """The proven streams as (header offset, stored size, credited packed bytes) among the method 2 headers whose
    stored extent overlaps [low, high)."""
    found = []
    for at, head in rnc.find(rom, max(0, low - 0x10000), min(len(rom), high + 0x10000)):
        if head.method != 2 or at + head.stored <= low or at >= high:
            continue
        try:
            found.append((at, head.stored, measure(rom, at)["credit"]))
        except StreamError:
            continue
    return found


def spans(found: Iterable[tuple[int, int, int]], targets: Iterable[tuple[int, int]]) -> list[dict[str, Any]]:
    """The rows (region inside targets, stream [header, stored end)) that proven streams give; a region is the packed
    bytes the packer reproduces."""
    rows = []
    for low, high in targets:
        for at, stored, credit in found:
            a, b = max(low, at + rnc.HEADER), min(high, at + rnc.HEADER + credit)
            if a < b:
                rows.append({"region": [a, b], "stream": [at, at + stored]})
    return sorted(rows, key=lambda r: r["region"])


def prove(rom: bytes | memoryview, start: int, end: int, stream: list[int]) -> dict[str, Any]:
    """The receipt of rom[start:end] as packed bits of the stream whose header is at stream[0]."""
    at = stream[0]
    measures = measure(rom, at)
    if not at + rnc.HEADER <= start < end <= at + rnc.HEADER + measures["credit"]:
        raise StreamError("the region is not inside the packed bytes the packer reproduces")
    return {
        "stream": [at, at + measures["stored"]],
        **measures,
        "region": [start, end],
        "region_sha256": sha256(bytes(rom[start:end])),
    }
