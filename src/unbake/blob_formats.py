"""Proofs at container boundaries: typed records, stale fragments and allocated stream slack."""

from __future__ import annotations

from unbake import blob_archive, blob_decode, blobs, rnc, snsym
from unbake.contracts import Config, Json, sha256

READERS = frozenset({"archive-fill", "archive-header", "rnc-slack", "string-leaf", "sn-debug", "copy", "segment-fill"})


def interval(rom: bytes, span: list[int]) -> tuple[int, int]:
    if len(span) != 2 or any(type(x) is not int for x in span):
        raise ValueError("extent needs two integer offsets")
    a, b = span
    if not 0 <= a < b <= len(rom):
        raise ValueError("extent outside ROM")
    return a, b


def entry(rom: bytes, container: list[int], path: list[int]) -> Json:
    """Follow actual relative directory offsets; requested bounds never establish ownership."""
    a, b = interval(rom, container)
    if not path:
        raise ValueError("directory entry path is empty")
    steps = []
    for index in path:
        table = blob_decode.directory(rom[a:b])
        if type(index) is not int or not 0 <= index < table["count"]:
            raise ValueError("directory entry index outside table")
        low, high = table["offsets"][index : index + 2]
        steps.append({"container": [a, b], "index": index, "offsets": [low, high]})
        a, b = a + low, a + high
        if a == b:
            raise ValueError("empty directory entry")
    return {"rom": [a, b], "directories": steps}


def string_leaf(data: bytes) -> Json:
    """A stride/count array of fixed-width, NUL-terminated ASCII slots; unused slot storage is preserved."""
    stride, count = blob_decode.uint(data, 0), blob_decode.uint(data, 4)
    if stride < 1 or 8 + stride * count != len(data):
        raise ValueError("string leaf stride/count does not tile payload")
    names, tails = [], []
    for at in range(8, len(data), stride):
        slot = data[at : at + stride]
        zero = slot.find(b"\0")
        if zero < 0 or any(x < 32 or x > 126 for x in slot[:zero]):
            raise ValueError("string slot lacks ASCII name and NUL termination")
        names.append(slot[:zero].decode("ascii"))
        tails.append(slot[zero + 1 :])
    again = stride.to_bytes(4, "big") + count.to_bytes(4, "big")
    again += b"".join(name.encode("ascii") + b"\0" + tail for name, tail in zip(names, tails, strict=True))
    if again != data:
        raise ValueError("string table re-emission differs")
    return {
        "stride": stride,
        "count": count,
        "names": names,
        "slot_slack_hex": [tail.hex() for tail in tails],
        "reemit_sha256": sha256(again),
    }


def prove(config: Config, row: Json, rom: bytes, document: Json) -> Json:
    start, end = interval(rom, row["region"])
    reader = row["reader"]
    if reader == "segment-fill":
        # The project declares the entire extracted segment as fill. Unlike automatic
        # gap alignment, its builder-owned extent is explicit and may end off alignment.
        segments = sorted(config.project.version[row["version"]].segments, key=lambda s: s.rom)
        matches = [
            (s, segments[i + 1].rom if i + 1 < len(segments) else len(rom))
            for i, s in enumerate(segments)
            if s.name == row["segment"] and s.rom == start
        ]
        if len(matches) != 1 or matches[0][0].kind != "bin" or matches[0][1] != end:
            raise ValueError("fill must occupy exactly one configured binary segment")
        value = row["value"]
        if type(value) is not int or not 0 <= value <= 255 or rom[start:end] != bytes([value]) * (end - start):
            raise ValueError("declared segment fill differs from its byte value")
        return {"segment": row["segment"], "rom": [start, end], "value": value}
    if reader == "archive-fill":
        archive = blob_archive.archive(rom, row["base"])
        fills = [(a, b) for a, b, _ in archive["fills"]]
        if blobs.subtract([(start, end)], blobs.merge(fills)):
            raise ValueError("region is not archive section-relative padding")
        return {
            "base": archive["base"],
            "end": archive["end"],
            "fill": [[a, b] for a, b in fills if a < end and start < b],
        }
    if reader == "archive-header":
        base = row["base"]
        header = blob_archive.header_at(rom, base)
        if header is None or not base <= start < end <= base + 84:
            raise ValueError("region is not a valid archive directory header")
        again = b"".join(w.to_bytes(4, "big") for w in header)
        if again != rom[base : base + 84]:
            raise ValueError("archive directory re-emission differs")
        return {
            "base": base,
            "sections": list(zip(header[:-1:2], header[1:-1:2], strict=True)),
            "sentinel": header[-1],
            "reemit_sha256": sha256(again),
        }
    if reader == "rnc-slack":
        owner = entry(rom, row["container"], row["entry_path"])
        a, b = owner["rom"]
        packed_header = rnc.header(rom[a:b])
        if packed_header.unpacked > 16 << 20:
            raise ValueError("RNC decoded size exceeds proof bound")
        body = rnc.unpack(rom[a:b])
        packed_end = a + packed_header.stored
        if not packed_end <= start < end <= b:
            raise ValueError("region is not after the stream inside its allocated entry")
        return {
            **owner,
            "stream": [a, packed_end],
            "decoded_sha256": sha256(body),
            "slack": [packed_end, b],
            "slack_sha256": sha256(rom[packed_end:b]),
        }
    if reader == "string-leaf":
        owner = entry(rom, row["container"], row["entry_path"])
        a, b = owner["rom"]
        if start != a:
            raise ValueError("string leaf must begin at its directory entry")
        packed_header = rnc.header(rom[a:b])
        if packed_header.unpacked > 16 << 20 or end != a + packed_header.stored:
            raise ValueError("string leaf region differs from complete packed stream")
        body = rnc.unpack(rom[a:b])
        return {**owner, "decoded_size": len(body), "strings": string_leaf(body)}
    if reader == "sn-debug":
        return snsym.stream_receipt(rom, start, end, row.get("dialect", "sym"))
    if reader == "copy":
        source = row["source"]
        if type(source) is not int or not 0 <= source < source + end - start <= len(rom):
            raise ValueError("copy source outside ROM")
        source_end = source + end - start
        owner = document["blob"][row["source_blob"]]
        if owner["state"] != "decoded" or blobs.subtract(
            [(source, source_end)], blobs.merge(owner.get("rom", {}).get(row["version"], []))
        ):
            raise ValueError("copy source is not covered by its decoded owner")
        if rom[source:source_end] != rom[start:end]:
            raise ValueError("stale copy differs from decoded source")
        return {"source": [source, source_end], "source_blob": row["source_blob"], "sha256": sha256(rom[start:end])}
    raise ValueError(f"unknown container reader {reader}")
