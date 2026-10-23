"""Stale ROM bytes the builder left behind, proven against what the project already accounts for: object code that a
known body explains (`objcode`) and exact copies of accounted bytes in the same or another version's ROM.

`accounted` is what a copy may be a copy of: code, data and the decoded or identified blobs of a version. `find`
lists the exact copies (at least MIN_COPY bytes) of those bytes inside unknown spans; `scan` runs both readers over the
unknown bytes of every version and lists the requests `unbake blobs prove` takes."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from unbake import blobs, objcode, snsym
from unbake.contracts import Config, Json

MIN_COPY = 64  # the fewest bytes of one copy
WINDOW = 32  # source bytes hashed at multiples of WINDOW; any copy of MIN_COPY bytes holds one
MIN_DISTINCT = 4  # a window of fewer distinct byte values (fill, a repeated word) names no source
Spans = list[tuple[int, int]]
_ACCOUNTED = ("code", "data", "decoded", "identified")


def accounted(parts: Json) -> Spans:
    """The bytes of one version's partition that a copy may be a copy of."""
    return blobs.merge([tuple(s) for k in _ACCOUNTED for s in parts["spans"][k]])


def source_spans(config: Config, version: str) -> Spans:
    """`accounted` for one version of a project."""
    tree = import_module("unbake.tree").capture(config)
    return accounted(import_module("unbake.report").partitions(config, tree)[version])


def unknown(parts: Json) -> Spans:
    """The bytes of one version's partition that no class holds."""
    held = blobs.merge([tuple(s) for spans in parts["spans"].values() for s in spans])
    return blobs.subtract([(0, parts["rom_size"])], held)


def index(roms: Mapping[str, bytes], spans: Mapping[str, Spans]) -> dict[bytes, list[tuple[str, int]]]:
    """Every WINDOW-aligned window inside accounted spans, by its bytes."""
    found: defaultdict[bytes, list[tuple[str, int]]] = defaultdict(list)
    for version, rom in roms.items():
        for low, high in spans[version]:
            at = (low + WINDOW - 1) // WINDOW * WINDOW
            while at + WINDOW <= high:
                window = rom[at : at + WINDOW]
                if len(set(window)) >= MIN_DISTINCT:
                    found[window].append((version, at))
                at += WINDOW
    return found


def find(
    roms: Mapping[str, bytes],
    spans: Mapping[str, Spans],
    windows: Mapping[bytes, Sequence[tuple[str, int]]],
    version: str,
    targets: Spans,
    order: Sequence[str] = (),
) -> list[Json]:
    """The copies in `targets` of `version`: maximal runs of at least MIN_COPY bytes equal to accounted bytes of one
    version, left to right, the longest source first (then a source in `version`, then the project's version order, then
    the lowest offset)."""
    rank = {v: i for i, v in enumerate(order)}
    rom = roms[version]
    starts = {v: [a for a, _ in s] for v, s in spans.items()}
    found: list[Json] = []
    for low, high in targets:
        at, floor = low, low
        while at + WINDOW <= high:
            best: tuple[tuple[int, bool, int, int], int, int, str, int] | None = None
            for source_version, source in windows.get(rom[at : at + WINDOW], ()):
                other = roms[source_version]
                i = bisect_right(starts[source_version], source) - 1
                s_low, s_high = spans[source_version][i]
                back = 0
                while at - back > floor and source - back > s_low and rom[at - back - 1] == other[source - back - 1]:
                    back += 1
                fwd = WINDOW
                while at + fwd < high and source + fwd < s_high and rom[at + fwd] == other[source + fwd]:
                    fwd += 1
                key = (-(back + fwd), source_version != version, rank.get(source_version, len(rank)), source - back)
                if best is None or key < best[0]:
                    best = (key, back, fwd, source_version, source - back)
            if best is not None and best[1] + best[2] >= MIN_COPY:
                _, back, fwd, source_version, source = best
                found.append(
                    {
                        "version": version,
                        "region": [at - back, at + fwd],
                        "reader": "copy",
                        "source": source,
                        "source_version": source_version,
                    }
                )
                at = floor = at + fwd
            else:
                at += 1
    return found


def objects(rom: bytes, targets: Spans) -> list[Json]:
    """The object files inside `targets` whose records tile from the magic to where they stop (the proof is the
    sn-debug reader of the object dialect)."""
    found: list[Json] = []
    for low, high in targets:
        at = rom.find(snsym.OBJ_MAGIC, low, high)
        while at >= 0:
            records, stop = snsym.records(rom, at + 4, high, "obj")
            if len(records) >= 50:
                found.append({"version": "", "region": [at, stop], "reader": "sn-debug", "dialect": "obj"})
            at = rom.find(snsym.OBJ_MAGIC, max(stop, at + 1), high)
    return found


def scan(config: Config) -> Json:
    """`unbake blobs scan stale`: per version, the object-code regions and the exact copies that fill unknown bytes,
    as the rows `unbake blobs prove` takes (without an id or its evidence), and the bytes left."""
    tree = import_module("unbake.tree").capture(config)
    parts = import_module("unbake.report").partitions(config, tree)
    roms = {v: (Path(config.project.root) / "roms" / f"baserom.{v}.z64").read_bytes() for v in parts}
    spans = {v: accounted(p) for v, p in parts.items()}
    windows = index(roms, spans)
    bodies = objcode.corpus(config, roms=roms)
    out: dict[str, Any] = {}
    for v, p in parts.items():
        targets = unknown(p)
        streams = [{**s, "version": v} for s in objects(roms[v], targets)]
        left_over = blobs.subtract(targets, blobs.merge([tuple(s["region"]) for s in streams]))
        code = bodies.regions(roms[v], v, left_over)
        rest = blobs.subtract(left_over, blobs.merge([tuple(r["rom"]) for r in code]))
        copies = find(roms, spans, windows, v, rest, list(parts))
        left = blobs.subtract(rest, blobs.merge([tuple(c["region"]) for c in copies]))
        out[v] = {
            "unknown": blobs.length(targets),
            "streams": streams,
            "code": [
                {"version": v, "region": r["rom"], "reader": "obj-code", "tiles": r["tiles"], "bodies": r["bodies"]}
                for r in code
            ],
            "copies": copies,
            "left": blobs.length(left),
        }
    return out
