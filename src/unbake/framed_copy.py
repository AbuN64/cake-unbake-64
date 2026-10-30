"""Stale copies a builder cut with its block headers (`framing`): the header is the only difference from the source.

A copy of another version's bytes that crosses a block boundary of the builder's image carries the header between the
two runs it copied. Each run is the source's bytes exactly; the header is the frame's, standing on a held lattice
point, and the source continues right where the first run ended and the second begins. The row names the header's
bytes (`region`), the source offset where the copy would go on (`source`), the version (`source_version`) and the
frame. The proof compares the ROM with the source byte for byte on both sides of the header, as far as they agree
(at least MIN_RUN each side, and no further than the next lattice point), and the receipt lists both runs, so every byte
of the copy has its source and the header is shown to be the only insertion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from unbake import framing
from unbake.contracts import sha256

MIN_RUN = 32  # the fewest source bytes on each side of the header


class CopyError(ValueError):
    """Bytes that are not a source copy with a block header in it."""


def runs(
    rom: bytes | memoryview, other: bytes | memoryview, start: int, end: int, source: int, reach: int
) -> tuple[int, int]:
    """How many bytes before `start` equal the source before `source`, and how many from `end` equal the source from
    `source`, each at most `reach`."""
    before = 0
    while before < min(reach, start, source) and rom[start - 1 - before] == other[source - 1 - before]:
        before += 1
    after = 0
    while (
        after < reach
        and end + after < len(rom)
        and source + after < len(other)
        and rom[end + after] == other[source + after]
    ):
        after += 1
    return before, after


def prove(
    rom: bytes | memoryview, other: bytes | memoryview, start: int, end: int, source: int, frame: Mapping[str, Any]
) -> dict[str, Any]:
    """The receipt of rom[start:end] as the header of `frame` standing between two runs of the source (`other`) that
    continues at `source`. Raises CopyError naming what fails."""
    try:
        lattice = framing.parse(frame)
        count = framing.accepted(rom, lattice)
    except framing.FrameError as error:
        raise CopyError(str(error)) from error
    if type(source) is not int or source < 0 or source > len(other):
        raise CopyError("copy source outside the ROM")
    if end - start != len(lattice.header) or start not in framing.points(lattice, count, start, end):
        raise CopyError("the region is not a header on a held lattice point of the frame")
    reach = lattice.block
    before, after = runs(rom, other, start, end, source, reach)
    if before < MIN_RUN or after < MIN_RUN:
        raise CopyError(f"the source agrees {before} bytes before and {after} after the header, fewer than {MIN_RUN}")
    return {
        "frame": lattice.row(),
        "source": source,
        "before": [source - before, source],
        "after": [source, source + after],
        "sha256": sha256(bytes(rom[start - before : end + after])),
    }


def find(rom: bytes | memoryview, other: bytes | memoryview, start: int, end: int) -> int | None:
    """The source offset where the copy goes on after rom[start:end]: the one place in `other` that holds the
    MIN_RUN bytes before the header and then the MIN_RUN bytes after it; None when there is none or more than one."""
    if start < MIN_RUN or end + MIN_RUN > len(rom):
        return None
    pre, post = bytes(rom[start - MIN_RUN : start]), bytes(rom[end : end + MIN_RUN])
    where = bytes(other).find(pre + post)
    if where < 0 or bytes(other).find(pre + post, where + 1) >= 0:
        return None
    return where + MIN_RUN
