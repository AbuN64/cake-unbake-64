"""Block headers a ROM builder wrote between the blocks of its image, and the reads exact once they are out.

A builder that wrote an image in fixed blocks put the same header bytes in front of every block, so the headers sit
on an arithmetic lattice: `anchor + k * (len(header) + block)`. Whatever was written through that image is cut by
them: a compressed stream or a copy of another version's bytes that crosses a block boundary has the header
inside it, and reads as damaged until the header is removed. A `Frame` names the lattice (anchor, header bytes,
block length). It is held as far as the header really stands at every lattice point from the anchor on (`held`); a
frame is accepted only when it holds at least MIN_HELD points, and a read that crosses a lattice point the ROM does
not hold refuses.

`discover` finds the frames a ROM holds for header patterns the caller names; `strip` removes the header bytes inside a
span and `raw_end` gives the ROM offset where a span with a known stripped length ends. The proof of a read through a
frame is the read's own check (both CRCs of a stream, the bytes of a source): a wrong frame cannot pass it."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

MIN_HELD = 6  # the fewest consecutive lattice points a frame holds: chance does not repeat a header this often
MIN_BLOCK = 0x1000  # the shortest block: shorter periods are table rows, not a builder's block size
MAX_PATTERNS = 4096  # a header that occurs more often in a ROM than this is fill, not a header
MAX_HEADER = 8


class FrameError(ValueError):
    """A frame the ROM does not hold."""


@dataclass(frozen=True)
class Frame:
    anchor: int
    header: bytes
    block: int

    @property
    def period(self) -> int:
        return len(self.header) + self.block

    def row(self) -> dict[str, Any]:
        return {"anchor": self.anchor, "header": self.header.hex(), "block": self.block}


def parse(row: Mapping[str, Any]) -> Frame:
    """The frame a request row names: {"anchor": int, "header": hex, "block": int}."""
    try:
        anchor, block, header = row["anchor"], row["block"], bytes.fromhex(row["header"])
    except (KeyError, TypeError, ValueError) as error:
        raise FrameError(f"a frame needs anchor, header (hex) and block: {error}") from error
    if type(anchor) is not int or type(block) is not int or anchor < 0 or block < MIN_BLOCK:
        raise FrameError(f"a frame needs a nonnegative integer anchor and a block of at least {MIN_BLOCK:#x}")
    if not 0 < len(header) <= MAX_HEADER:
        raise FrameError(f"a frame header is 1 to {MAX_HEADER} bytes")
    return Frame(anchor, header, block)


def held(rom: bytes | memoryview, frame: Frame) -> int:
    """How many lattice points from the anchor on carry the header in the ROM."""
    count, at, size = 0, frame.anchor, len(frame.header)
    while at + size <= len(rom) and bytes(rom[at : at + size]) == frame.header:
        count += 1
        at += frame.period
    return count


def accepted(rom: bytes | memoryview, frame: Frame) -> int:
    """The held points of a frame; refuses a frame that holds fewer than MIN_HELD."""
    count = held(rom, frame)
    if count < MIN_HELD:
        raise FrameError(f"the frame holds {count} lattice points from {frame.anchor:#x}, fewer than {MIN_HELD}")
    return count


def points(frame: Frame, count: int, low: int, high: int) -> list[int]:
    """The held lattice points whose header starts in [low, high)."""
    first = max(0, -(-(low - frame.anchor) // frame.period))
    last = min(count - 1, (high - 1 - frame.anchor) // frame.period) if high > frame.anchor else -1
    return [frame.anchor + k * frame.period for k in range(first, last + 1)]


def strip(rom: bytes | memoryview, low: int, high: int, frame: Frame, count: int) -> bytes:
    """rom[low:high] without the header bytes of the held lattice points that start inside it; a header cut by `low`
    or `high` refuses."""
    size = len(frame.header)
    out, at = bytearray(), low
    for point in points(frame, count, low - size + 1, high):
        if point < low or point + size > high:
            raise FrameError(f"the header at {point:#x} is cut by the span {low:#x}..{high:#x}")
        out += rom[at:point]
        at = point + size
    out += rom[at:high]
    return bytes(out)


def raw_end(frame: Frame, count: int, low: int, length: int) -> int:
    """The ROM offset where a read of `length` stripped bytes from `low` ends (headers counted inside it)."""
    size, end = len(frame.header), low + length
    while True:
        more = low + length + size * len(points(frame, count, low, end))
        if more == end:
            return end
        end = more


def discover(rom: bytes | memoryview, patterns: Iterable[bytes]) -> list[Frame]:
    """The frames the ROM holds for header patterns: per pattern, every maximal chain of at least MIN_HELD occurrences
    at one stride (not shorter than MIN_BLOCK), longest first. The stride comes from a pair of occurrences and the
    chain from the anchor on, so occurrences between the chain's points do not break it."""
    data = bytes(rom)
    found: list[Frame] = []
    for pattern in sorted(set(patterns)):
        if not 0 < len(pattern) <= MAX_HEADER:
            continue
        where = [m.start() for m in re.finditer(b"(?=" + re.escape(pattern) + b")", data, re.DOTALL)]
        if len(where) < MIN_HELD or len(where) > MAX_PATTERNS:
            continue
        seen = set(where)
        used: set[int] = set()
        for i, first in enumerate(where):
            if first in used:
                continue
            best: list[int] = []
            for second in where[i + 1 :]:
                period = second - first
                if period < MIN_BLOCK + len(pattern):
                    continue
                chain = [first]
                while chain[-1] + period in seen:
                    chain.append(chain[-1] + period)
                if len(chain) > len(best):
                    best = chain
            if len(best) >= MIN_HELD:
                used.update(best)
                found.append(Frame(first, pattern, best[1] - best[0] - len(pattern)))
    return sorted(found, key=lambda f: (-held(data, f), f.anchor))
