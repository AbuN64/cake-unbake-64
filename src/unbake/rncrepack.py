"""Stale RNC2 streams proved by re-packing: bytes are packed bits of a method 2 stream when the packer reproduces them.

`rnc` proves a stream whole: both CRCs and the unpacked size agree with the header. An older link left streams in the
ROM that later writes damaged, so their CRC fails for good. This reader proves the parts that are still intact, exactly:
the bits are decoded token by token (`rnc2.walk`, as far as they decode), the decoded bytes are packed again with the
deterministic packer `rnc2enc` (which reproduces all 32 whole method 2 streams of the German ROM byte for byte), and a
run of tokens the walk read is proven when the packer writes the very same tokens at the same output positions and the
bytes they pack to equal the ROM's. A byte a later write replaced inside a literal decodes to other data and packs back
to the same byte, so it stays equal; a damaged control bit or length changes the tokens, the run ends before it and the
walk goes on reading: after a burst of damage the bits fall back into the packer's tokens and a new run begins, so the
tail of a damaged stream is proven too, window by window. A run begins on a control byte (the first one also holds the
two lock bits) and ends where the control byte it last used is spent, or at the end marker; a token is credited only
when the data its choice looked at (a match is at most MAX_LENGTH long and one step of lazy matching looks one match
further) is decoded as well, unless the stream decodes whole to the header's unpacked size. Every byte outside a run
stays with whatever else accounts for it. The receipt names the decoded bytes (size, sha256), the packer's settings and
the runs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from unbake import rnc, rnc2, rnc2enc
from unbake.contracts import sha256

MARGIN = 2 * rnc2enc.MAX_LENGTH + 16  # decoded bytes a token's choice may have looked at past its end
MIN_RUN = 16  # the fewest packed bytes a run credits
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


def _read(rom: bytes | memoryview, at: int) -> tuple[Any, bytes, Any, list[tuple[int, int, bytes]]]:
    """The header, the packed bytes, the walk and the runs of the stream whose header is at `at`: each run is
    (first, end, rebuilt bytes) in packed-byte offsets, and the rebuilt bytes equal packed[first:end]."""
    try:
        head = rnc.header(rom, at)
    except rnc.RncError as error:
        raise StreamError(str(error)) from error
    if head.method != 2:
        raise StreamError("only method 2 streams are read here")
    packed = bytes(rom[at + rnc.HEADER : at + head.stored])
    if rnc.crc16(packed) == head.packed_crc:
        raise StreamError("the packed CRC passes: the stream is whole, a rnc reader proves it")
    walked = rnc2.walk(packed, head.unpacked * CAP_FACTOR)
    mine = {(t[1], t[0] == "E"): t for t in rnc2enc.parse(walked.data)}
    complete = walked.ended and len(walked.data) == head.unpacked
    reach = len(walked.data) - MARGIN
    same = [mine.get((t[1], t[0] == "E")) == t and (complete or rnc2enc.end(t) <= reach) for t in walked.tokens]
    found, k, count = [], 0, len(same)
    while k < count:
        if not same[k]:
            k += 1
            continue
        stop = k
        while stop < count and same[stop]:
            stop += 1
        first, last = k, stop - 1
        while first <= last and walked.marks[first][1]:
            first += 1
        if not (walked.ended and last == count - 1):  # the end marker's control byte is spent by definition
            while last >= first and walked.marks[last + 1][1]:
                last -= 1
        if first <= last:
            low, high = walked.marks[first][0], walked.marks[last + 1][0]
            rebuilt = rnc2enc.emit(walked.data, walked.tokens[first : last + 1], first == 0)[0]
            if high - low >= MIN_RUN and rebuilt == packed[low:high]:
                found.append((low, high, rebuilt))
        k = stop
    return head, packed, walked, found


def _facts(head: Any, packed: bytes, walked: Any, found: list[tuple[int, int, bytes]]) -> dict[str, Any]:
    if not found:
        why = walked.error or "no error"
        raise StreamError(f"the packer reproduces no run of {MIN_RUN} packed bytes of it ({why})")
    return {
        "method": head.method,
        "unpacked": head.unpacked,
        "packed": head.packed,
        "stored": head.stored,
        "packed_crc": head.packed_crc,
        "packed_crc_read": rnc.crc16(packed),
        "unpacked_crc": head.unpacked_crc,
        "decoded": len(walked.data),
        "decoded_sha256": sha256(walked.data),
        "decode_stopped": walked.error or "end marker",
        "tokens": len(walked.tokens),
        "runs": [[low, high] for low, high, _ in found],
        "credit": sum(high - low for low, high, _ in found),
        "settings": SETTINGS,
    }


def measure(rom: bytes | memoryview, at: int) -> dict[str, Any]:
    """The proof of the stream whose header is at `at`: the packed bytes the packer reproduces, as runs, and the facts
    the receipt holds. Raises StreamError naming why there is none."""
    head, packed, walked, found = _read(rom, at)
    return _facts(head, packed, walked, found)


Found = tuple[int, int, tuple[tuple[int, int], ...]]


def find(rom: bytes | memoryview, low: int, high: int) -> list[Found]:
    """The proven streams as (header offset, stored size, runs) among the method 2 headers whose stored extent
    overlaps [low, high); a run is (first, end) in packed-byte offsets."""
    found = []
    for at, head in rnc.find(rom, max(0, low - 0x10000), min(len(rom), high + 0x10000)):
        if head.method != 2 or at + head.stored <= low or at >= high:
            continue
        try:
            runs = tuple((a, b) for a, b, _ in _read(rom, at)[3])
        except StreamError:
            continue
        if runs:
            found.append((at, head.stored, runs))
    return found


def spans(found: Iterable[Found], targets: Iterable[tuple[int, int]]) -> list[dict[str, Any]]:
    """The rows (region inside targets, stream [header, stored end)) that proven streams give; a region is the packed
    bytes of a run the packer reproduces."""
    rows = []
    for low, high in targets:
        for at, stored, runs in found:
            for first, end in runs:
                a, b = max(low, at + rnc.HEADER + first), min(high, at + rnc.HEADER + end)
                if a < b:
                    rows.append({"region": [a, b], "stream": [at, at + stored]})
    return sorted(rows, key=lambda r: r["region"])


def prove(rom: bytes | memoryview, start: int, end: int, stream: list[int]) -> dict[str, Any]:
    """The receipt of rom[start:end] as packed bits of the stream whose header is at stream[0]: the region lies inside
    one run and the bytes the packer writes for it equal the ROM's."""
    at = stream[0]
    head, packed, walked, found = _read(rom, at)
    measures = _facts(head, packed, walked, found)
    base = at + rnc.HEADER
    run = next(((a, b) for a, b in measures["runs"] if base + a <= start < end <= base + b), None)
    if run is None:
        raise StreamError("the region is not inside one run of packed bytes the packer reproduces")
    whole = next(r for a, b, r in found if (a, b) == tuple(run))
    rebuilt = whole[start - base - run[0] : end - base - run[0]]
    if rebuilt != bytes(rom[start:end]):
        raise StreamError("the packer writes other bytes than the ROM holds there")
    return {
        "stream": [at, at + measures["stored"]],
        **measures,
        "region": [start, end],
        "region_sha256": sha256(bytes(rom[start:end])),
        "rebuilt_sha256": sha256(rebuilt),
    }
