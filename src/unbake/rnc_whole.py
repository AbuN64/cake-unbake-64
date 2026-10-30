"""Stale RNC streams that are whole once a builder's block headers are taken out (`framing`), proven by both CRCs.

`rnc` proves a stream the ROM holds contiguously. A stream an older link left in an image the builder wrote in blocks
has the block header inside it wherever it crosses a block boundary, and a later write that copied part of it from
another version leaves the rest unclaimed. This reader names the stream (`stream`, the offset of its header) and the
frame (none, when the stream crosses no lattice point): the header bytes of the frame inside the stream's extent are
removed, and what is left must be the stream the header describes: its packed CRC, its unpacked size and its unpacked
CRC all pass (`rnc.unpack` with check). The row's region lies inside the raw extent [stream, raw end): the packed bytes
and the block headers between them are the stream's, byte for byte, whatever else once copied part of them. The receipt
names the stream (method, sizes, CRCs, decoded sha256), the frame and the lattice points inside the region."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from unbake import framing, rnc
from unbake.contracts import sha256

LOOK = 64  # stripped bytes read to parse a stream header: the header, and a block header may stand inside it
REACH = 0x100  # raw bytes past a header's stored size that block headers may add
MAX_UNPACKED = 16 << 20  # the largest stream proved (the bound the other RNC readers keep)


class StreamError(ValueError):
    """Bytes that are not a whole stream through the named frame."""


def _header(raw: bytes) -> rnc.Header:
    """The header fields of a stripped stream start; the sizes cannot be checked against data not yet read."""
    if len(raw) < rnc.HEADER or raw[:4] not in rnc.MAGICS:
        raise rnc.RncError("no RNC header")
    out = rnc.Header(
        int.from_bytes(raw[4:8], "big"),
        int.from_bytes(raw[8:12], "big"),
        int.from_bytes(raw[12:14], "big"),
        int.from_bytes(raw[14:16], "big"),
        raw[16],
        raw[17],
        raw[3],
    )
    if not out.unpacked or not out.packed:
        raise rnc.RncError("RNC header names no data")
    return out


def _extent(
    rom: bytes | memoryview, at: int, frame: framing.Frame | None, count: int
) -> tuple[rnc.Header, bytes, int, list[int]]:
    """The header, the stripped stream, the raw end and the lattice points inside [at, raw end)."""
    if at < 0 or at + rnc.HEADER > len(rom):
        raise StreamError("stream outside the ROM")
    try:
        if frame is None:
            look = bytes(rom[at : at + rnc.HEADER])
        else:
            look = framing.strip(rom, at, min(len(rom), at + LOOK + len(frame.header) * 2), frame, count)[: rnc.HEADER]
        head = _header(look)
    except (rnc.RncError, framing.FrameError) as error:
        raise StreamError(str(error)) from error
    if head.unpacked > MAX_UNPACKED:
        raise StreamError("the stream's unpacked size exceeds the proof bound")
    end = at + head.stored if frame is None else framing.raw_end(frame, count, at, head.stored)
    if end > len(rom):
        raise StreamError("stream runs past the ROM")
    inside = [] if frame is None else framing.points(frame, count, at, end)
    try:
        stream = bytes(rom[at:end]) if frame is None else framing.strip(rom, at, end, frame, count)
    except framing.FrameError as error:
        raise StreamError(str(error)) from error
    return head, stream, end, inside


def prove(
    rom: bytes | memoryview, start: int, end: int, stream: int, frame: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The receipt of rom[start:end] as part of the stream whose header is at `stream`, through `frame` (a request
    row of `framing.parse`) or none. Raises StreamError naming what fails."""
    try:
        lattice = None if frame is None else framing.parse(frame)
        count = 0 if lattice is None else framing.accepted(rom, lattice)
    except framing.FrameError as error:
        raise StreamError(str(error)) from error
    head, data, raw, inside = _extent(rom, stream, lattice, count)
    if lattice is not None and not inside:
        raise StreamError("the frame has no lattice point inside the stream: name no frame")
    if not stream <= start < end <= raw:
        raise StreamError(f"region {start:#x}..{end:#x} is not inside the stream {stream:#x}..{raw:#x}")
    try:
        decoded = rnc.unpack(data, 0)
    except rnc.RncError as error:
        raise StreamError(f"not a whole stream: {error}") from error
    out: dict[str, Any] = {
        "stream": [stream, raw],
        "method": head.method,
        "unpacked": head.unpacked,
        "packed": head.packed,
        "stored": head.stored,
        "packed_crc": head.packed_crc,
        "unpacked_crc": head.unpacked_crc,
        "decoded_sha256": sha256(decoded),
        "sha256": sha256(bytes(rom[start:end])),
    }
    if lattice is not None:
        out["frame"] = lattice.row()
        out["headers"] = [p for p in inside if start <= p < end]
    return out


def find(
    rom: bytes | memoryview, frames: Sequence[framing.Frame], low: int, high: int
) -> list[tuple[int, int, framing.Frame | None]]:
    """The whole streams as (header offset, raw end, frame) whose raw extent overlaps [low, high): a stream the ROM
    holds contiguously first, else through the first frame that makes it whole."""
    found = []
    counts = {f: framing.held(rom, f) for f in frames}
    data = bytes(rom)
    for at in sorted(m.start() for magic in rnc.MAGICS for m in re.finditer(re.escape(magic), data)):
        if at >= high or at + rnc.HEADER > len(data) or at < low - MAX_UNPACKED:
            continue
        try:
            head = _header(data[at : at + rnc.HEADER])
        except rnc.RncError:
            continue
        if at + head.stored + REACH <= low:
            continue
        for frame in (None, *frames):
            try:
                _, data, end, inside = _extent(rom, at, frame, counts.get(frame, 0) if frame else 0)
                if frame is not None and (not inside or counts[frame] < framing.MIN_HELD):
                    continue
                rnc.unpack(data, 0)
            except (StreamError, rnc.RncError):
                continue
            if at < high and end > low:
                found.append((at, end, frame))
            break
    return found


def spans(
    found: Iterable[tuple[int, int, framing.Frame | None]], targets: Iterable[tuple[int, int]]
) -> list[dict[str, Any]]:
    """The rows (region inside targets, stream, frame) the whole streams give; each stream credits the target bytes
    inside its raw extent."""
    rows = []
    for low, high in targets:
        for at, end, frame in found:
            a, b = max(low, at), min(high, end)
            if a < b:
                row: dict[str, Any] = {"region": [a, b], "stream": at}
                if frame is not None:
                    row["frame"] = frame.row()
                rows.append(row)
    return sorted(rows, key=lambda r: r["region"])
