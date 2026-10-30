"""The pad reader: a zero run between aligned objects, recorded as a fill blob up to a stated limit.

The linker pads an object to the alignment of the next one with zero bytes. Such a run is neither code, data nor
content, and no source defines it, so it stays unknown until something proves it. Proof is by rebuilding: the reader
emits `length` zero bytes and the range closes only when they equal the ROM bytes. No threshold or similarity decides.

The pad rule, over one unknown gap (the bytes between two accounted objects that no class holds):
  - the whole gap is zero: the run lies between the two objects;
  - the gap starts with zero bytes and the first nonzero byte begins at a `ALIGN` boundary: the run pads the previous
    object up to the alignment of the object that starts there.
A run longer than `LIMIT` is refused, however it lies: a long zero stretch is as likely to be a zeroed table or a
buffer. Zeros inside a gap, or at its end behind other unknown bytes, are no pad and stay unknown, as does every
gap that holds a nonzero byte beyond the leading run. `scan` lists what it refuses."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from unbake import blobs
from unbake.contracts import Config, Json, sha256

LIMIT = 512  # the longest zero run a pad row may record: the widest alignment pad measured is 343 bytes
ALIGN = 4  # the boundary the first nonzero byte of a gap must start at, for a run that does not fill the gap
MIN_RUN = blobs.RUN_MIN  # the shortest run that pads inside a gap of other bytes
Spans = list[tuple[int, int]]


def runs(rom: bytes, gaps: Spans, limit: int = LIMIT) -> tuple[Spans, list[Json]]:
    """`(pad, refused)`: the zero runs of `gaps` that the pad rule accepts, and the candidates it refuses with why."""
    pad: Spans = []
    refused: list[Json] = []
    for start, end in gaps:
        zeros = len(rom[start:end]) - len(rom[start:end].lstrip(b"\0"))
        stop = start + zeros
        if not zeros or (stop != end and zeros < MIN_RUN):
            continue
        if stop != end and stop % ALIGN:
            refused.append(
                {"rom": [start, stop], "reason": f"zero run ends at {stop:#x}, off the {ALIGN}-byte boundary"}
            )
        elif zeros > limit:
            refused.append({"rom": [start, stop], "reason": f"zero run of {zeros} bytes is over the limit {limit}"})
        else:
            pad.append((start, stop))
    return pad, refused


def prove(rom: bytes, start: int, end: int, limit: object) -> Json:
    """The proof of one pad range: `end - start` zero bytes, rebuilt, equal the ROM bytes, and the range is within
    `limit`. Anything else is refused."""
    if type(limit) is not int or limit < 1:
        raise ValueError("pad needs an integer limit of at least 1")
    if end - start > limit:
        raise ValueError(f"pad of {end - start} bytes is over the limit {limit}")
    rebuilt = bytes(end - start)
    if rom[start:end] != rebuilt:
        raise ValueError("rebuilt zero bytes differ from the ROM bytes")
    return {"limit": limit, "bytes": end - start, "rebuilt_sha256": sha256(rebuilt)}


def scan(config: Config) -> Json:
    """`unbake blobs scan pad`: per version, the pad ranges the rule closes (each proven by `prove`), the candidates it
    refuses, and the unknown bytes left after the pad."""
    tree = import_module("unbake.tree").capture(config)
    parts = import_module("unbake.report").partitions(config, tree)
    stale = import_module("unbake.blob_stale")
    out: dict[str, Any] = {}
    for version, part in parts.items():
        rom = (Path(config.project.root) / "roms" / f"baserom.{version}.z64").read_bytes()
        gaps = stale.unknown(part)
        pad, refused = runs(rom, gaps)
        for low, high in pad:
            prove(rom, low, high, LIMIT)
        out[version] = {
            "limit": LIMIT,
            "pad": [[a, b] for a, b in pad],
            "pad_bytes": blobs.length(pad),
            "refused": refused,
            "unknown_before": blobs.length(gaps),
            "unknown_after": blobs.length(blobs.subtract(gaps, pad)),
        }
    return out
