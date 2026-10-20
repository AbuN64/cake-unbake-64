"""`blobs prove REQUEST RECEIPT`: pinned, read-only byte evidence without rescoring or granting credit.

Requests select ROM extents and existing project layouts. Readers are generic: a typed graph, directory header,
absolute bank pairs, automatic fill, or an observation. An observation includes bounds, catalog/segment owners
and a bounded RNC check; neither zero content nor successful decompression grants a data/decoded claim.
"""

from __future__ import annotations

import json
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

from unbake import blob_decode, blob_formats, blobs, rnc
from unbake.contracts import Config, Finding, Json, Refusal, sha256


def context(config: Config, version: str, region: tuple[int, int], document: Json) -> Json:
    start, end = region
    segments = sorted(config.project.version[version].segments, key=lambda s: s.rom)
    stops = [s.rom for s in segments[1:]] + [config.project.version[version].size]
    segment = [
        {"name": s.name, "rom": [s.rom, stop], "kind": s.kind, "vram": s.vram}
        for s, stop in zip(segments, stops, strict=True)
        if s.rom < end and start < stop
    ]
    owners = [
        {"id": name, "state": row["state"], "type": row["type"], "rom": [low, high]}
        for name, row in sorted(document["blob"].items())
        for low, high in row.get("rom", {}).get(version, ())
        if low < end and start < high
    ]
    return {
        "segments": segment,
        "catalog": owners,
        "loaded_data_candidate": any(s["vram"] is not None and s["kind"] != "bin" for s in segment),
    }


def fill_spans(config: Config, version: str, rom: bytes, document: Json) -> list[tuple[int, int]]:
    """Conservative fill outside loaded segments and catalogued content. Never infer gaps inside code/data."""
    held = []
    segments = sorted(config.project.version[version].segments, key=lambda s: s.rom)
    stops = [s.rom for s in segments[1:]] + [len(rom)]
    for segment, stop in zip(segments, stops, strict=True):
        if segment.kind != "bin":
            held.append((segment.rom, stop))
    decoded, identified = blobs.ranges(document, version)
    gaps = blobs.subtract([(0, len(rom))], blobs.merge(held + decoded + identified))
    return blobs.fill(rom, gaps)


def observe(rom: bytes, start: int, end: int) -> Json:
    data = rom[start:end]
    result: dict[str, Any] = {
        "bytes": len(data),
        "sha256": sha256(data),
        "prefix_hex": data[:64].hex(),
        "all_zero": not any(data),
        "distinct_values": len(set(data)),
    }
    # Only inspect the immediately preceding signature, at most 64 KiB back. A candidate is not a typed proof.
    at = rom.rfind(rnc.MAGIC, max(0, start - 65536), end)
    if at >= 0:
        try:
            header = rnc.header(rom, at)
            if header.unpacked > 16 << 20:
                raise ValueError("RNC decoded size exceeds inspection bound")
            decoded = rnc.unpack(rom, at)
            result["rnc"] = {
                "start": at,
                "end": at + header.stored,
                "decoded_size": len(decoded),
                "decoded_sha256": sha256(decoded),
                "typed": False,
                "covers_region": at <= start and end <= at + header.stored,
            }
        except ValueError as error:
            result["rnc"] = {"start": at, "refusal": str(error)}
    return result


def prove_one(config: Config, row: Json, rom: bytes, document: Json, fills: list[tuple[int, int]]) -> Json:
    start, end = row["region"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(rom):
        raise ValueError("region must be a nonempty interval inside its ROM")
    reader = row["reader"]
    data = rom[start:end]
    result: dict[str, Any] = {
        "id": row["id"],
        "version": row["version"],
        "rom": [start, end],
        "sha256": sha256(data),
        "reader": reader,
        "context": context(config, row["version"], (start, end), document),
    }
    if reader == "observe":
        return {
            **result,
            "status": "observed",
            "observation": observe(rom, start, end),
            "automatic_fill": not blobs.subtract([(start, end)], blobs.merge(fills)),
            "limit": row["limit"],
        }
    if reader == "fill":
        if blobs.subtract([(start, end)], blobs.merge(fills)):
            raise ValueError("region is not fully covered by automatic layout-proven fill")
        return {**result, "status": "decoded", "fill": [[a, b] for a, b in fills if a < end and start < b]}
    if not row.get("evidence"):
        raise ValueError("a typed reader needs project evidence")
    result["evidence"] = row["evidence"]
    if reader in blob_formats.READERS:
        result["decoded"] = blob_formats.prove(config, row, rom, document)
    elif reader == "graph":
        layout_path = Path(row["layout_file"])
        raw = layout_path.read_bytes()
        if sha256(raw) != row["layout_sha256"]:
            raise ValueError("layout file sha256 differs from its request pin")
        layouts = json.loads(raw)["blob_layouts"]
        selected = [layout for layout in layouts if layout.get("blob") == row["layout_id"]]
        if len(selected) != 1:
            raise ValueError("layout id must select exactly one existing layout")
        layout = selected[0]
        codec = row.get("codec", "none")
        if codec == "rnc1":
            header = rnc.header(data)
            if header.unpacked > 16 << 20:
                raise ValueError("RNC decoded size exceeds proof bound")
            blob_decode.aligned_zero(data, header.stored, len(data), row["stored_alignment"])
            body = rnc.unpack(data)
            result["stored_zero_alignment"] = [start + header.stored, end]
        elif codec == "none":
            body = data
        else:
            raise ValueError("unsupported codec")
        result["layout"] = layout
        result["layout_sha256"] = row["layout_sha256"]
        result["decoded"] = blob_decode.Graph(body, layout).finish()
    elif reader == "directory-header":
        result["decoded"] = blob_decode.directory(data, header_only=True, extent=row["extent"])
    elif reader == "bank-pairs":
        if len(data) % 8:
            raise ValueError("partial bank pair")
        words = [blob_decode.uint(data, i) for i in range(0, len(data), 4)]
        banks = list(zip(words[::2], words[1::2], strict=True))
        if words[0] != end or any(not end <= a < b <= len(rom) for a, b in banks):
            raise ValueError("bank pointers outside ROM or first bank not after table")
        if any(b != c for (_, b), (c, _) in pairwise(banks)):
            raise ValueError("bank pairs do not tile the bank extent")
        result["decoded"] = {"fields": [[4, w] for w in words], "banks": banks}
    else:
        raise ValueError(f"unknown proof reader {reader}")
    return {**result, "status": "decoded"}


def run(config: Config, request_path: str, receipt_path: str) -> Json:
    raw = Path(request_path).read_bytes()
    request = json.loads(raw)
    rows = request.get("rows", [])
    if request.get("schema") != 1 or not rows:
        raise Refusal(Finding("blobs.proof", "proof request needs schema 1 and nonempty rows"))
    document = blobs.load(config)
    roms, fills, results = {}, {}, []
    for row in rows:
        wall, cpu = time.monotonic(), time.process_time()
        version = row["version"]
        if version not in config.project.version:
            raise Refusal(Finding("blobs.proof", f"unknown ROM version {version}"))
        if version not in roms:
            rom = (Path(config.project.root) / "roms" / f"baserom.{version}.z64").read_bytes()
            if sha256(rom) != request["rom_sha256"][version]:
                raise Refusal(Finding("blobs.proof", f"ROM sha256 mismatch: {version}"))
            roms[version] = rom
            fills[version] = fill_spans(config, version, rom, document)
        try:
            proof = prove_one(config, row, roms[version], document, fills[version])
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise Refusal(Finding("blobs.proof", f"{row.get('id')}: {error}")) from error
        results.append({**proof, "wall_s": time.monotonic() - wall, "cpu_s": time.process_time() - cpu})
    receipt = {
        "schema": 1,
        "request_sha256": sha256(raw),
        "rom_sha256": request["rom_sha256"],
        "blobs_sha256": blobs.sha(config),
        "rows": results,
    }
    target = Path(receipt_path)
    if target.exists():
        raise Refusal(Finding("blobs.proof", "receipt exists; choose a new evidence path", path=receipt_path))
    encoded = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode()
    target.write_bytes(encoded)
    # Proof is deliberately independent of matching reports. It grants no catalog credit and runs no build.
    return {
        "count": len(document["blob"]),
        "census": blobs.census(document),
        "partition": {},
        "debt": [],
        "proved": {"receipt": str(target), "sha256": sha256(encoded), "rows": len(results)},
    }
