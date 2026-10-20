"""Independent, lossless typed views of resource graphs. Project layouts supply the field facts.

Unlike a native endian conversion, overlapping views do not have to agree on scalar widths. Each view must
re-emit its source bytes, and the union must cover the payload. Zero bytes are not evidence of padding: only
short runs ending at the next typed record's alignment qualify (the same B1 rule as blobs.fill).
Relative pointer null values are explicit.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import pairwise

from unbake.contracts import Json, sha256

_REF = re.compile(r"@(\d+):([A-Za-z_]\w*)\Z")


def uint(data: bytes, at: int, width: int = 4) -> int:
    if at < 0 or width < 1 or at + width > len(data):
        raise ValueError(f"scalar outside extent: {at}+{width}/{len(data)}")
    return int.from_bytes(data[at : at + width], "big")


def aligned_zero(data: bytes, start: int, end: int, alignment: int) -> None:
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    if not 0 <= start <= end <= len(data):
        raise ValueError("padding outside extent")
    if end > start and (end % alignment or end - start >= alignment or any(data[start:end])):
        raise ValueError(f"unproved padding {start:#x}..{end:#x} (alignment {alignment})")


def directory(data: bytes, *, header_only: bool = False, extent: int | None = None) -> Json:
    """count, count+1 relative offsets, and zero alignment; an empty table may have terminal offset zero."""
    count = uint(data, 0)
    head = 4 * (count + 2)
    if head > len(data):
        raise ValueError("directory table outside extent")
    offsets = [uint(data, 4 + 4 * i) for i in range(count + 1)]
    aligned = (head + 7) & ~7
    if count == 0 and offsets == [0] and len(data) == 8:
        return {"count": 0, "offsets": offsets, "header_size": 8, "zero_padding": []}
    if offsets[0] != aligned or any(a > b for a, b in pairwise(offsets)):
        raise ValueError("invalid directory first offset or ordering")
    aligned_zero(data, head, aligned, 8)
    limit = len(data) if extent is None else extent
    if offsets[-1] > limit:
        raise ValueError("directory child outside enclosing extent")
    if header_only:
        if len(data) != aligned or offsets[-1] != limit:
            raise ValueError("directory header or final offset differs from its declared extent")
    else:
        aligned_zero(data, offsets[-1], len(data), 8)
    return {
        "count": count,
        "offsets": offsets,
        "header_size": aligned,
        "zero_padding": [[head, aligned]] if head < aligned else [],
    }


class Graph:
    """Decode the existing fields/leaf/directory/head_then layout vocabulary into separate scalar views."""

    def __init__(self, data: bytes, layout: Json):
        self.data, self.layout = data, layout
        self.types = layout["types"]
        self.coverage = bytearray(len(data))
        self.emitted = bytearray(len(data))
        self.records: list[Json] = []
        self.padding: list[list[int]] = []
        self.seen: set[tuple[str, int, tuple[tuple[int, int], ...]]] = set()
        self.pools: list[tuple[int, int]] = []
        self.nulls = layout["null_values"]
        if not isinstance(self.nulls, list) or any(type(n) is not int or not 0 <= n <= 0xFFFFFFFF for n in self.nulls):
            raise ValueError("null_values must be an explicit list of u32 values")

    def write(self, start: int, raw: bytes) -> None:
        end = start + len(raw)
        if start < 0 or end > len(self.data) or raw != self.data[start:end]:
            raise ValueError("typed view does not re-emit its source")
        self.coverage[start:end] = b"\1" * len(raw)
        self.emitted[start:end] = raw

    def pad(self, start: int, end: int, alignment: int) -> None:
        aligned_zero(self.data, start, end, alignment)
        if start < end:
            self.write(start, bytes(end - start))
            self.padding.append([start, end])

    def tokens(self, name: str) -> list[tuple[int, tuple[int, str] | None]]:
        result = []
        for token in self.types[name]["fields"].split():
            match = _REF.fullmatch(token)
            if match:
                result.append((4, (int(match[1]), match[2])))
            elif re.fullmatch(r"[1248]+", token):
                result.extend((int(width), None) for width in token)
            else:
                raise ValueError(f"unsupported scalar field {token!r}")
        if not result:
            raise ValueError("empty scalar record")
        return result

    def width(self, name: str) -> int:
        return sum(width for width, _ in self.tokens(name))

    def record(self, name: str, start: int, end: int, sections: Sequence[tuple[int, int]]) -> int:
        tokens = self.tokens(name)
        size = sum(w for w, _ in tokens)
        if start < 0 or start + size > end:
            raise ValueError(f"record {name} outside extent at {start:#x}")
        key = (name, start, tuple(sections))
        if key in self.seen:
            return start + size
        self.seen.add(key)
        at, values, refs = start, [], []
        for width, ref in tokens:
            value = uint(self.data, at, width)
            values.append([width, value])
            self.write(at, value.to_bytes(width, "big"))
            if ref and value not in self.nulls:
                section, target = ref
                if section >= len(sections):
                    raise ValueError("reference names no pool section")
                low, high = sections[section]
                refs.append((target, low + value, high))
            at += width
        self.records.append({"type": name, "offset": start, "fields": values})
        for target, low, high in refs:
            self.record(target, low, high, sections)
        return at

    def walk(self, name: str, start: int, end: int, sections: Sequence[tuple[int, int]] = ()) -> None:
        if not 0 <= start <= end <= len(self.data):
            raise ValueError("graph node outside extent")
        body = self.types[name]
        if len(body) != 1:
            raise ValueError("a graph type needs exactly one shape")
        shape, value = next(iter(body.items()))
        if shape == "fields":
            stop = self.record(name, start, end, sections)
            self.pad(stop, end, 8)
        elif shape == "leaf":
            stride, count = uint(self.data, start), uint(self.data, start + 4)
            if stride != self.width(value) or start + 8 + stride * count > end:
                raise ValueError("leaf stride/count outside extent or different from record width")
            self.header(start, [stride, count])
            stop = start + 8
            for _ in range(count):
                stop = self.record(value, stop, stop + stride, sections)
            self.pad(stop, end, 8)
        elif shape == "head_then":
            head, element = value
            stop = self.record(head, start, end, sections)
            width = self.width(element)
            while stop + width <= end:
                stop = self.record(element, stop, stop + width, sections)
            self.pad(stop, end, 8)
        elif shape == "directory":
            table = directory(self.data[start:end])
            self.header(start, [table["count"], *table["offsets"]])
            for low, high in table["zero_padding"]:
                self.pad(start + low, start + high, 8)
            spans = [(start + a, start + b) for a, b in pairwise(table["offsets"])]
            targets = [value["each"]] * len(spans) if isinstance(value, dict) and set(value) == {"each"} else value
            if not isinstance(targets, list) or len(targets) != len(spans):
                raise ValueError("directory shape differs from section count")
            for (low, high), target in zip(spans, targets, strict=True):
                if target is None:
                    self.pools.append((low, high))
                else:
                    self.walk(target, low, high, spans)
            self.pad(start + (table["offsets"][-1] or 8), end, 8)
        else:
            raise ValueError(f"unsupported graph shape {shape}")

    def header(self, start: int, values: list[int]) -> None:
        raw = b"".join(value.to_bytes(4, "big") for value in values)
        self.write(start, raw)
        self.records.append({"type": "u32-header", "offset": start, "fields": [[4, v] for v in values]})

    def finish(self) -> Json:
        self.walk(self.layout["root"], 0, len(self.data))
        for start, end in self.pools:
            at = start
            while at < end:
                if self.coverage[at]:
                    at += 1
                    continue
                stop = at + 1
                while stop < end and not self.coverage[stop]:
                    stop += 1
                # A referenced record boundary is the next item, exactly as for B1 ROM fill.
                self.pad(at, stop, stop & -stop)
                at = stop
        if not all(self.coverage) or self.emitted != self.data:
            missing = [i for i, byte in enumerate(self.coverage) if not byte]
            raise ValueError(f"untyped graph bytes: {len(missing)}, first {missing[:8]}")
        return {
            "bytes": len(self.data),
            "sha256": sha256(self.data),
            "reemit_sha256": sha256(self.emitted),
            "typed_bytes": len(self.data) - sum(b - a for a, b in self.padding),
            "zero_alignment": self.padding,
            "records": self.records,
        }
