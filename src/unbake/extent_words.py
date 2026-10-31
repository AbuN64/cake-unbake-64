"""Extent words: the offsets and sizes a sectioned container records about its own sections.

Two record shapes, both big-endian words, both laid out by the builder that packed the image:
  wrapper    `1, HEADER, size, 0` then an RNC stream: `size` is the stream's stored length plus the wrapper header,
             rounded up to ALIGN. The stream's own header gives the stored length, so the three words are fixed by it.
  container  `n, offset[0..n-1], size` then n sections: offset[0] is the header's own length (4 * (n + 2)), every
             later offset is the previous offset plus the previous section's extent, and `size` ends the last section.
A section is self-sized when its own bytes fix its extent: a wrapper whose size word matches its stream, or a nested
container whose size word equals the extent. The proof of a claimed word follows the chain: an offset is proven when
every section before it is self-sized and the extents add up to it; `size` is proven the same way, a section
that is not self-sized having to lie in bytes accounted already. The row names the wrapper or the container;
the claimed region must lie inside its header words (a wrapper claim may also take the zero slack before it, shorter
than ALIGN). Nothing but the ROM bytes decides: a word that does not follow from the chain refuses the row."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any

from unbake import blobs, rnc
from unbake.contracts import sha256

ALIGN = 8  # sections start and end on this boundary, relative to their container
HEADER = 0x10  # a wrapper's own length: its four words
MAX_SECTIONS = 64  # the most sections a container header may count
REACH = 256  # the longest unknown span `find` looks into for header words


class ExtentError(ValueError):
    """Words that do not follow from the sections they describe."""


def _word(rom: bytes | memoryview, at: int) -> int:
    if at < 0 or at + 4 > len(rom):
        raise ExtentError(f"word at {at:#x} lies outside the ROM")
    return int.from_bytes(rom[at : at + 4], "big")


def _up(size: int) -> int:
    return -(-size // ALIGN) * ALIGN


def wrapper(rom: bytes | memoryview, at: int, extent: int | None = None) -> dict[str, int]:
    """The wrapper whose header stands at `at` (with `extent`, the section it must fill): its size word must be the
    stream's stored length plus the header, rounded up."""
    words = [_word(rom, at + 4 * i) for i in range(4)]
    if words[0] != 1 or words[1] != HEADER or words[3] != 0:
        raise ExtentError(f"no wrapper header at {at:#x}")
    try:
        stream = rnc.header(rom, at + HEADER)
    except rnc.RncError as error:
        raise ExtentError(f"wrapper at {at:#x}: {error}") from error
    want = _up(HEADER + stream.stored)
    if words[2] != want or (extent is not None and extent != want):
        raise ExtentError(f"wrapper at {at:#x} says {words[2]:#x}, its stream makes {want:#x}")
    return {"at": at, "size": want, "stream": at + HEADER, "stored": stream.stored}


def shape(rom: bytes | memoryview, at: int) -> tuple[int, list[int], int]:
    """(n, offsets, size) of the container header at `at` when its words fit one another; ExtentError otherwise."""
    n = _word(rom, at)
    if not 1 <= n <= MAX_SECTIONS:
        raise ExtentError(f"no container at {at:#x}: {n} sections")
    offsets = [_word(rom, at + 4 + 4 * i) for i in range(n)]
    size = _word(rom, at + 4 + 4 * n)
    if offsets[0] != 4 * (n + 2):
        raise ExtentError(f"container at {at:#x}: first offset {offsets[0]:#x} is not its header length")
    ordered = all(a < b for a, b in itertools.pairwise(offsets)) and offsets[-1] < size
    if not ordered or any(x % ALIGN for x in offsets[1:]) or size % ALIGN or at + size > len(rom):
        raise ExtentError(f"container at {at:#x}: offsets and size do not fit one another")
    return n, offsets, size


def _held(held: list[tuple[int, int]], a: int, b: int) -> bool:
    """rom[a:b] lies inside the accounted ranges `held` (merged and sorted)."""
    return not blobs.subtract([(a, b)], held)


def sections(rom: bytes | memoryview, at: int, held: list[tuple[int, int]]) -> list[dict[str, Any]]:
    """The sections of the container at `at`, laid one after the other from the end of its header, each with the kind
    its own bytes give: `wrapper`, `container`, or `opaque` (extent known only from the neighbouring offset). The
    builder left up to ALIGN - 1 bytes of slack before a section (zeros, or the end of a longer repacked stream that
    is accounted already), so a section stands where the one before it ended plus its slack, and the offsets (extents)
    stay the ones the words state."""
    _, offsets, size = shape(rom, at)
    out = []
    here = at + offsets[0]
    for offset, end in zip(offsets, [*offsets[1:], size], strict=True):
        extent = end - offset
        found: tuple[str, int] | None = None
        for slack in range(ALIGN):
            if slack and rom[here + slack - 1] != 0 and not _held(held, here + slack - 1, here + slack):
                break
            try:
                wrapper(rom, here + slack, extent)
                found = ("wrapper", slack)
                break
            except ExtentError:
                pass
            try:
                if shape(rom, here + slack)[2] == extent:
                    found = ("container", slack)
                    break
            except ExtentError:
                pass
        kind, slack = found or ("opaque", 0)
        out.append({"offset": offset, "extent": extent, "kind": kind, "at": here + slack, "slack": slack})
        here += slack + extent
    return out


def _words(start: int, end: int, base: int, count: int) -> list[int]:
    """The indices of the container header's words (0 the count, 1..n the offsets, n + 1 the size) that rom[start:end]
    touches."""
    return [k for k in range(count) if base + 4 * k < end and start < base + 4 * k + 4]


def prove(
    rom: bytes | memoryview,
    held: Iterable[tuple[int, int]],
    start: int,
    end: int,
    *,
    container: int | None = None,
    wrapper_at: int | None = None,
) -> dict[str, Any]:
    """The receipt of rom[start:end] as header words of the wrapper at `wrapper_at` or of the container at
    `container`; `held` are the ROM ranges already accounted: an opaque section of the container must lie in them."""
    if (container is None) == (wrapper_at is None) or start >= end:
        raise ExtentError("a row names one wrapper or one container and a region")
    if wrapper_at is not None:
        made = wrapper(rom, wrapper_at)
        slack = max(0, wrapper_at - start)
        if slack >= ALIGN or start < 0 or end > wrapper_at + HEADER or end <= wrapper_at or any(rom[start:wrapper_at]):
            raise ExtentError("the region is not a wrapper header with at most zero slack before it")
        return {**made, "slack": slack, "sha256": sha256(bytes(rom[start:end]))}
    assert container is not None
    n, offsets, size = shape(rom, container)
    header = 4 * (n + 2)
    if start < container or end > container + header:
        raise ExtentError("the region is not inside the container's header words")
    merged = blobs.merge([*held, (start, end)])
    parts = sections(rom, container, merged)
    claimed = _words(start, end, container, n + 2)
    for k in claimed:
        if k < 2:
            continue  # the count and the first offset are fixed by the header's own length
        for part in parts[: k - 1] if k <= n else parts:
            if part["kind"] == "opaque" and not _held(merged, part["at"], part["at"] + part["extent"]):
                at = f"{part['at']:#x}"
                raise ExtentError(f"word {k} does not follow: the section at {at} is neither self-sized nor accounted")
    return {
        "container": container,
        "count": n,
        "offsets": offsets,
        "size": size,
        "sections": parts,
        "words": claimed,
        "sha256": sha256(bytes(rom[start:end])),
    }


def find(
    rom: bytes | memoryview, held: list[tuple[int, int]], targets: Iterable[tuple[int, int]]
) -> list[dict[str, Any]]:
    """The rows (region, wrapper or container) the unknown spans of at most REACH bytes give: wrappers first, then
    containers over what a wrapper did not take."""
    rows: list[dict[str, Any]] = []
    for low, high in targets:
        if high - low > REACH:
            continue
        left = [(low, high)]
        for at in range(max(HEADER, low + 1), min(high + HEADER, len(rom)) + 1):
            if rom[at : at + 3] != b"RNC":
                continue
            wrap = at - HEADER
            first = wrap
            while first > low and wrap - first < ALIGN - 1 and rom[first - 1] == 0:
                first -= 1
            region = (max(first, low), min(at, high))
            if region[0] < region[1] and blobs.subtract([region], left) == []:
                try:
                    prove(rom, held, *region, wrapper_at=wrap)
                except ExtentError:
                    continue
                rows.append({"region": list(region), "wrapper": wrap})
                left = blobs.subtract(left, [region])
        for base in range(max(0, low - 4 * (MAX_SECTIONS + 2)), high):
            try:
                reach = base + 4 * (shape(rom, base)[0] + 2)
            except ExtentError:
                continue
            for region in list(left):
                a, b = max(region[0], base), min(region[1], reach)
                if a >= b:
                    continue
                try:
                    prove(rom, held, a, b, container=base)
                except ExtentError:
                    continue
                rows.append({"region": [a, b], "container": base})
                left = blobs.subtract(left, [(a, b)])
    return sorted(rows, key=lambda r: r["region"])
