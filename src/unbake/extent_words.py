"""Extent words: the offsets and sizes a sectioned container records about its own sections.

Two record shapes, both big-endian words, both laid out by the builder that packed the image:
  wrapper    `1, HEADER, size, 0` then an RNC stream: `size` is the stream's stored length plus the wrapper header,
             rounded up to ALIGN. The stream's own header gives the stored length, so the three words are fixed by it.
  container  `n, offset[0..n-1], size` then n sections: offset[0] is the header's own length (4 * (n + 2), zero-padded
             to ALIGN when n is odd), every later offset is the previous offset plus the previous section's extent,
             and `size` ends the last section. A container of no sections is the two words `0, 8`.
A section is self-sized when its own bytes fix its extent: a wrapper whose size word matches its stream, a bare RNC
stream whose stored length rounded up is the extent, a nested container whose size word equals the extent, or ALIGN
zero bytes (the builder's empty section) with a wrapper or container right after them. The proof of a claimed word
follows the chain: an offset is proven when every section before it is self-sized and the extents add up to it;
`size` is proven the same way, a section that is not self-sized having to lie in bytes accounted already. The row
names the wrapper or the container; the claimed region must lie inside what the container's bytes determine: its
header words and their zero pad, the zero slack and zero tail pad around a section, the bytes of an empty section (a
wrapper claim may also take the zero slack before it, shorter than ALIGN). Nothing but the ROM bytes decides: a word
that does not follow from the chain refuses the row."""

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
LOOKBACK = 0x400  # how far before an unknown span `find` looks for the container that holds it


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
    if not 0 <= n <= MAX_SECTIONS:
        raise ExtentError(f"no container at {at:#x}: {n} sections")
    offsets = [_word(rom, at + 4 + 4 * i) for i in range(n)]
    size = _word(rom, at + 4 + 4 * n)
    words = 4 * (n + 2)
    if n == 0:
        if size != ALIGN:
            raise ExtentError(f"container at {at:#x}: no sections but size {size:#x}")
        return n, offsets, size
    if offsets[0] not in (words, _up(words)) or any(rom[at + words : at + offsets[0]]):
        raise ExtentError(f"container at {at:#x}: first offset {offsets[0]:#x} is not its header length")
    ordered = all(a < b for a, b in itertools.pairwise(offsets)) and offsets[-1] < size
    if not ordered or any(x % ALIGN for x in offsets[1:]) or size % ALIGN or at + size > len(rom):
        raise ExtentError(f"container at {at:#x}: offsets and size do not fit one another")
    return n, offsets, size


def _begins(rom: bytes | memoryview, at: int) -> bool:
    """A wrapper or a container of sections starts at `at`."""
    try:
        return shape(rom, at)[0] > 0
    except ExtentError:
        pass
    try:
        wrapper(rom, at)
    except ExtentError:
        return False
    return True


def _held(held: list[tuple[int, int]], a: int, b: int) -> bool:
    """rom[a:b] lies inside the accounted ranges `held` (merged and sorted)."""
    return not blobs.subtract([(a, b)], held)


def _section(rom: bytes | memoryview, at: int, extent: int) -> tuple[str, int] | None:
    """(kind, tail) of the self-sized section of `extent` bytes at `at`; `tail` is where its own bytes end."""
    try:
        return "wrapper", at + HEADER + wrapper(rom, at, extent)["stored"]
    except ExtentError:
        pass
    try:
        if shape(rom, at)[2] == extent:
            return "container", at + extent
    except ExtentError:
        pass
    try:
        if rom[at : at + 3] == b"RNC" and _up(rnc.header(rom, at).stored) == extent:
            return "stream", at + rnc.header(rom, at).stored
    except rnc.RncError:
        pass
    if extent == ALIGN and not any(rom[at : at + ALIGN]) and _begins(rom, at + ALIGN):
        return "zero", at + extent
    return None


def sections(rom: bytes | memoryview, at: int, held: list[tuple[int, int]]) -> list[dict[str, Any]]:
    """The sections of the container at `at`, laid one after the other from the end of its header, each with the kind
    its own bytes give: `wrapper`, `container`, `stream` (a bare RNC stream), `zero` (an empty section), or `opaque`
    (extent known only from the neighbouring offset). The builder left up to ALIGN - 1 bytes of slack before a section
    (zeros, or the end of a longer repacked stream that is accounted already), so a section stands where the one
    before it ended plus its slack, and the offsets (extents) stay the ones the words state. `end` is where the
    section's own bytes stop: the bytes from there to the extent are zero pad."""
    _, offsets, size = shape(rom, at)
    out = []
    here = at + (offsets[0] if offsets else ALIGN)
    for offset, end in zip(offsets, [*offsets[1:], size], strict=True):
        extent = end - offset
        found: tuple[str, int, int] | None = None
        for slack in range(ALIGN):
            if slack and rom[here + slack - 1] != 0 and not _held(held, here + slack - 1, here + slack):
                break
            if (kind := _section(rom, here + slack, extent)) is not None:
                found = (kind[0], slack, kind[1])
                break
        kind_name, slack, own = found or ("opaque", 0, here + extent)
        out.append(
            {"offset": offset, "extent": extent, "kind": kind_name, "at": here + slack, "slack": slack, "end": own}
        )
        here += slack + extent
    return out


def determined(rom: bytes | memoryview, at: int, parts: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """The bytes of the container at `at` that its own words fix: the header, its zero pad, the zero slack before each
    section, the zero tail after the section's own bytes, and the bytes of an empty section. A pad that holds a
    non-zero byte is left out."""
    n, offsets, _ = shape(rom, at)
    pads = [(at + 4 * (n + 2), at + (offsets[0] if offsets else 4 * (n + 2)))]
    fixed = [(at, at + 4 * (n + 2))]
    for part in parts:
        begin = part["at"]
        pads.append((begin - part["slack"], begin))
        if part["kind"] in ("wrapper", "stream"):
            pads.append((part["end"], begin + part["extent"]))
        if part["kind"] == "zero" or (part["kind"] == "container" and part["extent"] == ALIGN):
            fixed.append((begin, begin + part["extent"]))
    return blobs.merge([*fixed, *((a, b) for a, b in pads if a < b and not any(rom[a:b]))])


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
    """The receipt of rom[start:end] as header words (or the pad and empty sections the words fix) of the wrapper at
    `wrapper_at` or of the container at `container`; `held` are the ROM ranges already accounted: an opaque section of
    the container must lie in them."""
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
    if n == 0:
        raise ExtentError("a container without sections is claimed through the container that holds it")
    merged = blobs.merge([*held, (start, end)])
    parts = sections(rom, container, merged)
    if blobs.subtract([(start, end)], determined(rom, container, parts)):
        raise ExtentError("the region is not in the container's header words, pad or empty sections")
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


def _wrappers(
    rom: bytes | memoryview, held: list[tuple[int, int]], low: int, high: int, left: list[tuple[int, int]]
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """The wrapper rows of rom[low:high] and what they leave of `left`."""
    rows: list[dict[str, Any]] = []
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
    return rows, left


def find(
    rom: bytes | memoryview, held: list[tuple[int, int]], targets: Iterable[tuple[int, int]]
) -> list[dict[str, Any]]:
    """The rows (region, wrapper or container) the unknown spans of at most REACH bytes give: wrappers first, then
    containers over what a wrapper did not take."""
    rows: list[dict[str, Any]] = []
    for low, high in targets:
        if high - low > REACH:
            continue
        found, left = _wrappers(rom, held, low, high, [(low, high)])
        rows += found
        for base in range(max(0, low - LOOKBACK), high):
            try:
                if shape(rom, base)[0] == 0:
                    continue
                fixed = determined(rom, base, sections(rom, base, held))
            except ExtentError:
                continue
            for region in blobs.subtract(left, blobs.subtract(left, fixed)):
                try:
                    prove(rom, held, *region, container=base)
                except ExtentError:
                    continue
                rows.append({"region": list(region), "container": base})
                left = blobs.subtract(left, [region])
    return sorted(rows, key=lambda r: r["region"])
