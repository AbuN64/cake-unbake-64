"""Extent directories: the tables a bank keeps of the extents of the blobs it holds, proven from those blobs.

A builder that packed blobs into a bank wrote their boundaries next to them as big-endian words. Two shapes:
  pairs  `start, end` ROM offsets, one pair per blob, the pairs tiling the bank from the end of the table.
  ends   `n, length, end[0..n-1]` at the table's own offset: `length` is the table's byte length (4 * (n + 2)), the
         first child starts where the table ends and `end[k]` is the end of child k, relative to the table's offset.
The extents come from the decoded blobs of the version: a word is proven when it is the start or the end of a decoded
or identified blob (one that no other blob contains), the words ascend, and the shape's own words (the length, the
tiling, the first child at the table's end) hold. Nothing but the ROM bytes and those boundaries decides: one word that
names no boundary refuses the row."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from unbake.contracts import sha256

REACH = 256  # the longest unknown span `find` reads as a directory
SHAPES = ("pairs", "ends")  # the shapes a row may name


class DirectoryError(ValueError):
    """Words that are not the extents of the blobs they name."""


def boundaries(document: Mapping[str, Any], version: str) -> frozenset[int]:
    """The starts and ends of the decoded and identified blobs of one version that no other blob contains."""
    out: set[int] = set()
    for row in document.values():
        if row["state"] in ("decoded", "identified") and "parent" not in row:
            out.update(x for pair in row.get("rom", {}).get(version, ()) for x in pair)
    return frozenset(out)


def _words(rom: bytes | memoryview, start: int, end: int) -> list[int]:
    if start < 0 or end > len(rom) or (end - start) % 4 or start >= end:
        raise DirectoryError("the region is not whole words inside the ROM")
    return [int.from_bytes(rom[at : at + 4], "big") for at in range(start, end, 4)]


def _check(words: list[int], bounds: frozenset[int], base: int) -> None:
    for word in words:
        if base + word not in bounds:
            raise DirectoryError(f"{base + word:#x} is the boundary of no decoded blob")


def prove(rom: bytes | memoryview, bounds: frozenset[int], start: int, end: int, shape: str) -> dict[str, Any]:
    """The receipt of rom[start:end] as an extent directory of `shape`; DirectoryError when a word is not an extent."""
    words = _words(rom, start, end)
    digest = sha256(bytes(rom[start:end]))
    if shape == "pairs":
        if len(words) % 2 or words[0] != end:
            raise DirectoryError("the first pair does not start where the table ends")
        starts, ends = words[0::2], words[1::2]
        if any(a >= b for a, b in zip(starts, ends, strict=True)) or starts[1:] != ends[:-1]:
            raise DirectoryError("the pairs do not tile the bank")
        _check(words, bounds, 0)
        return {"shape": shape, "blobs": [[a, b] for a, b in zip(starts, ends, strict=True)], "sha256": digest}
    if shape != "ends":
        raise DirectoryError(f"no directory shape {shape!r}")
    if len(words) < 3:
        raise DirectoryError("a directory of ends holds a count, a length and one end at least")
    count, length, offsets = words[0], words[1], words[2:]
    if length != end - start or length != 4 * (count + 2) or len(offsets) != count:
        raise DirectoryError("the length word is not the table's own length")
    if any(a >= b for a, b in zip([length, *offsets], offsets, strict=False)):
        raise DirectoryError("the ends do not ascend from the end of the table")
    _check([length, *offsets], bounds, start)
    return {"shape": shape, "table": start, "ends": [start + x for x in offsets], "sha256": digest}


def find(rom: bytes | memoryview, bounds: frozenset[int], targets: Iterable[tuple[int, int]]) -> list[dict[str, Any]]:
    """The rows (region, shape) of the unknown spans of at most REACH bytes that read as an extent directory."""
    rows: list[dict[str, Any]] = []
    for low, high in targets:
        for shape in SHAPES if high - low <= REACH else ():
            try:
                prove(rom, bounds, low, high, shape)
            except DirectoryError:
                continue
            rows.append({"region": [low, high], "shape": shape})
            break
    return rows
