"""Proofs at container boundaries: typed records, stale fragments and allocated stream slack."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from unbake import blob_archive, blob_decode, blob_stale, blobs, mipsstream, objcode, rnc, snsym
from unbake.contracts import Config, Json, sha256

READERS = frozenset(
    {
        "archive-fill",
        "archive-header",
        "rnc-slack",
        "string-leaf",
        "c-string",
        "sn-debug",
        "sn-head",
        "copy",
        "segment-fill",
        "obj-code",
        "mips-stream",
        "rnc-copy",
    }
)
_CORPUS: dict[object, objcode.Corpus] = {}  # the known bodies of a project, once per Config digest
_SESSION: dict[object, object] | None = None  # while a request is proved: what its rows share


@contextmanager
def session() -> Iterator[None]:
    """A request of thousands of rows (copies of accounted bytes of other versions) reads each ROM and captures each
    version's accounted spans once, not once per row."""
    global _SESSION
    previous, _SESSION = _SESSION, {}
    try:
        yield
    finally:
        _SESSION = previous


def _shared(key: object, make: Callable[[], Any]) -> Any:
    if _SESSION is None:
        return make()
    if key not in _SESSION:
        _SESSION[key] = make()
    return _SESSION[key]


def _accounted(config: Config, version: str) -> list[tuple[int, int]]:
    """`blob_stale.source_spans` of a version."""
    spans: list[tuple[int, int]] = _shared(("spans", version), lambda: blob_stale.source_spans(config, version))
    return spans


def _baserom(config: Config, version: str) -> bytes:
    """One version's ROM bytes."""
    path = Path(config.project.root) / "roms" / f"baserom.{version}.z64"
    rom: bytes = _shared(("rom", version), path.read_bytes)
    return rom


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


def _copy(config: Config, row: Json, rom: bytes, document: Json, start: int, end: int) -> Json:
    """`rom[start:end]` is a byte-for-byte copy of `source` in the ROM of `source_version` (default: the row's own),
    and those source bytes are already accounted: covered by the decoded blob `source_blob`, or, with none named, code,
    data or a decoded or identified blob of that version."""
    version = row.get("source_version", row["version"])
    if version not in config.project.version:
        raise ValueError(f"copy source version {version} is unknown")
    other = rom if version == row["version"] else _baserom(config, version)
    source = row["source"]
    if type(source) is not int or not 0 <= source < source + end - start <= len(other):
        raise ValueError("copy source outside ROM")
    source_end = source + end - start
    if version == row["version"] and source < end and start < source_end:
        raise ValueError("copy source overlaps its copy")
    if "source_blob" in row:
        owner = document["blob"][row["source_blob"]]
        if owner["state"] != "decoded" or blobs.subtract(
            [(source, source_end)], blobs.merge(owner.get("rom", {}).get(version, []))
        ):
            raise ValueError("copy source is not covered by its decoded owner")
    elif blobs.subtract([(source, source_end)], _accounted(config, version)):
        raise ValueError("copy source is not code, data or a decoded blob of its version")
    if other[source:source_end] != rom[start:end]:
        raise ValueError("stale copy differs from decoded source")
    out: Json = {"source": [source, source_end], "sha256": sha256(rom[start:end])}
    if "source_blob" in row:
        out = {**out, "source_blob": row["source_blob"]}
    if version != row["version"]:
        out = {**out, "source_version": version}
    return out


def _sn_head(config: Config, row: Json, rom: bytes, start: int, end: int) -> Json:
    """`rom[start:end]` is the end of the debug record that a 4 KiB write cut from the stream starting at `end`: the
    bytes before it are zeros or another write, so the record is proven whole by a witness, in this ROM or another
    version's, that holds the same bytes and then the same following record inside a whole record."""
    size = end - start
    if not 0 < size <= 255:
        raise ValueError("a stream head is 1 to 255 bytes")
    others = {v: _baserom(config, v) for v in sorted(config.project.version) if v != row["version"]}
    try:
        cut = snsym.cut_head(rom, start, size, others)
    except snsym.SymError as error:
        raise ValueError(str(error)) from error
    return {**cut, "hex": rom[start:end].hex(), "sha256": sha256(rom[start:end])}


def _rnc_copy(config: Config, row: Json, rom: bytes, document: Json, start: int, end: int) -> Json:
    """Two complete CRC-checked streams decode equally, and the source has a decoded blob owner."""
    version = row["source_version"]
    if version not in config.project.version:
        raise ValueError("RNC copy source version is unknown")
    other = rom if version == row["version"] else _baserom(config, version)
    source = row["source"]
    if type(source) is not int or source < 0:
        raise ValueError("RNC copy source needs a nonnegative offset")
    target_header, source_header = rnc.header(rom, start), rnc.header(other, source)
    source_end = source + source_header.stored
    if end != start + target_header.stored:
        raise ValueError("RNC copy must be one complete target stream")
    if version == row["version"] and source < end and start < source_end:
        raise ValueError("RNC copy source overlaps target")
    owner = document["blob"][row["source_blob"]]
    if owner["state"] != "decoded" or blobs.subtract(
        [(source, source_end)], blobs.merge(owner.get("rom", {}).get(version, []))
    ):
        raise ValueError("RNC copy source is not covered by its decoded owner")
    if max(target_header.unpacked, source_header.unpacked) > 16 << 20:
        raise ValueError("RNC decoded size exceeds proof bound")
    if target_header.unpacked != source_header.unpacked:
        raise ValueError("RNC copy decoded sizes differ")
    target_body, source_body = rnc.unpack(rom, start), rnc.unpack(other, source)
    if target_body != source_body:
        raise ValueError("RNC copy decoded bytes differ")
    return {
        "source_version": version,
        "source_blob": row["source_blob"],
        "source": [source, source_end],
        "source_sha256": sha256(other[source:source_end]),
        "source_codec": f"rnc{source_header.method}",
        "codec": f"rnc{target_header.method}",
        "decoded_size": len(target_body),
        "decoded_sha256": sha256(target_body),
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
    if reader == "c-string":
        # The request supplies an independently reviewed C initializer, not a printable-run guess.
        text = row["text"]
        if not isinstance(text, str) or not text:
            raise ValueError("C string needs nonempty text")
        if any(not (32 <= ord(char) <= 126 or 160 <= ord(char) <= 255 or char in "\t\n\r") for char in text):
            raise ValueError("C string contains a nontext character")
        encoding = "ascii" if text.isascii() else "latin-1"
        encoded = text.encode(encoding) + b"\0"
        if encoded != rom[start:end]:
            raise ValueError("C string initializer does not exactly re-emit its terminated extent")
        return {"text": text, "encoding": encoding, "terminator": 0, "reencoded_sha256": sha256(encoded)}
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
    if reader == "sn-head":
        return _sn_head(config, row, rom, start, end)
    if reader == "mips-stream":
        try:
            return mipsstream.prove(rom, start, end, row["stream"])
        except (mipsstream.StreamError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"mips-stream: {error}") from error
    if reader == "rnc-copy":
        return _rnc_copy(config, row, rom, document, start, end)
    if reader == "copy":
        return _copy(config, row, rom, document, start, end)
    if reader == "obj-code":
        key = getattr(config, "digest", id(config))
        if key not in _CORPUS:
            _CORPUS.clear()
            _CORPUS[key] = objcode.corpus(config)
        try:
            return _CORPUS[key].prove(rom, row["version"], start, end)
        except objcode.ObjError as error:
            raise ValueError(str(error)) from error
    raise ValueError(f"unknown container reader {reader}")
