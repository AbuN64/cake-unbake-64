"""Earlier-build windows: typed data of an older build placed inside a bin segment at that build's addresses.

A `kind = "bin"` segment may declare `earlier_build = [{ rom, end, vram | delta }]`: the ROM range holds data the
build of an earlier version linked at other addresses (`vram = rom + delta`). Loading expands the segment into the
bin bytes before the window, the window as a code-kind segment of the distinct image `earlier_build` at its vram (so
typed sources, splat pieces and the link treat it as they treat any data segment) and the bin bytes after it. The
pointers of window data resolve against `versions/<v>/earlier_build_symbols.txt`, loaded apart from symbol_addrs.txt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from unbake.contracts import Segment

SPACE = "earlier_build"  # the distinct image every window of a version is linked in
INFIX = "_earlier_"  # between the bin segment name and the ROM offset in the default name of a window
MASK = 0xFFFFFFFF


def is_window(name: str) -> bool:
    """Whether a section is a window of an older build by its default name."""
    return INFIX in name


def symbols_path(version: str) -> str:
    return f"versions/{version}/earlier_build_symbols.txt"


def vram_of(window: Mapping[str, Any]) -> int:
    """The window's address: its `vram`, else `rom + delta` (the older build's address of a ROM offset)."""
    return int(window["vram"]) if "vram" in window else (window["rom"] + window["delta"]) & MASK


def expand(vid: str, rows: Sequence[Mapping[str, Any]], size: int) -> tuple[tuple[Segment, ...], list[str]]:
    """(segments, problems): the declared segment rows with each bin row's windows cut out of it. A row ends where
    the next declared row begins, the last at the ROM size. A malformed or overlapping window is a problem."""
    out: list[Segment] = []
    problems: list[str] = []
    for i, row in enumerate(rows):
        base = Segment(
            row["name"], row["rom"], row.get("vram"), row.get("kind", "code"), row.get("address_space", "main")
        )
        held = row.get("earlier_build", ())
        if not held:
            out.append(base)
            continue
        where = f"segment {base.name}"
        end = rows[i + 1]["rom"] if i + 1 < len(rows) else max(base.rom, size)
        if base.kind != "bin" or base.address_space != "main" or base.vram is not None:
            problems.append(f"{where}: earlier_build belongs to a bin segment with no vram or address_space")
            out.append(base)
            continue
        cursor = base.rom
        names: set[str] = set()
        for w in sorted(held, key=lambda w: w["rom"]):
            label = f"{where} window 0x{w['rom']:X}..0x{w['end']:X}"
            if ("vram" in w) == ("delta" in w):
                problems.append(f"{label}: give exactly one of vram and delta")
                continue
            if not base.rom <= w["rom"] < w["end"] <= end:
                problems.append(f"{label}: outside the segment 0x{base.rom:X}..0x{end:X}")
                continue
            if w["rom"] < cursor:
                problems.append(f"{label}: overlaps the window before it, which ends at 0x{cursor:X}")
                continue
            if w["rom"] % 4 or w["end"] % 4 or vram_of(w) % 4:
                problems.append(f"{label}: rom, end and vram must be word aligned")
                continue
            name = w.get("name", f"{base.name}{INFIX}{w['rom']:X}")
            if name in names or name == base.name or any(r["name"] == name for r in rows):
                problems.append(f"{label}: the name {name} is taken")
            names.add(name)
            if w["rom"] > cursor:
                out.append(Segment(base.name, cursor, None, "bin"))
            out.append(Segment(name, w["rom"], vram_of(w), "code", SPACE, base.name))
            cursor = w["end"]
        if cursor < end:
            out.append(Segment(base.name, cursor, None, "bin"))
    return tuple(out), problems


def declared(segments: Sequence[Segment]) -> tuple[Segment, ...]:
    """The window segments of a version."""
    return tuple(s for s in segments if s.window_of is not None)


def spans(segments: Sequence[Segment], size: int) -> list[tuple[int, int]]:
    """The ROM range of each window of a version: a segment ends where the next begins, the last at the ROM size."""
    ordered = sorted(segments, key=lambda s: s.rom)
    ends = [after.rom for after in ordered[1:]] + [max(ordered[-1].rom, size)] if ordered else []
    return [(s.rom, e) for s, e in zip(ordered, ends, strict=True) if s.window_of is not None]
