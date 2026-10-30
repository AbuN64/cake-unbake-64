"""SN object-file page reader: proves that one 512-byte sector of the debug segment is the last write of an SN/Psy-Q
object file's record stream, left behind by an earlier link, and reconstructs it as typed definitions.

The builder writes the debug stream in sectors, and a sector the later link did not overwrite keeps what an earlier
link put there. Such a sector (`SECTOR` bytes, sector aligned) holds, in this order and nothing else:
- front: the tail of an object file's code record, cut by the sector start. The bytes are the payload of a record
  whose header lies in the sector before, so they are compiled MIPS words on the grid that ends at the first whole
  record, after at most 3 bytes of a word cut at the front. Every word is a canonical instruction (`mipsstream`), at
  least MIN_WORDS words and MIN_WORDS // 2 distinct ones;
- records: at least MIN_RECORDS whole records of the object dialect (`snsym.object_record`), chained from the first
  one to the byte where the sector ends or the last record is cut. At least two thirds of them are line records. The code offsets they carry (a patch's offset, a
  line record's address) never fall inside a run of records, which a stream written in order guarantees and bytes
  read as records by chance do not. The first record is the lowest start with code before it from which the chain
  lands there;
- cut: a code record (tag 0x02, a u16 length that is a whole number of words) whose payload the sector end cut: the
  header lies inside the sector, the length runs past it, and the payload words that are in it are canonical
  instructions on the grid that starts at the payload.
A sector that is no such page is refused with the rule it breaks, and its bytes stay unknown.

The receipt reconstructs the page as typed definitions: the front as its instruction words and their assembly, the
records as their tags and fields, the cut as its code record header and payload words. It also compares the front
with the code of the project's other versions (`others`): where the older link's words equal a later build's, and the
first word where they differ. The comparison is a measurement; the proof is the structure above. A row credits a
region inside the page."""

from __future__ import annotations

import struct
from collections.abc import Iterator, Mapping
from typing import Any

import rabbitizer

from unbake import mipsstream, snsym
from unbake.contracts import sha256

SECTOR = 512  # the write size that cuts a record: the sector the builder wrote, aligned in the ROM
MIN_RECORDS = 8  # the fewest whole records of a page
MIN_WORDS = 8  # the fewest instruction words of a front
SEED = 4  # words of the front that must occur in another version's code to compare against it
CODE = 0x02  # the object-file record that carries code: tag, u16 length, then that many bytes
SECTION = 0x06  # the record that switches section, with the code record: a new run of offsets
LINES = frozenset({0x32, 0x34, 0x36, 0x38, 0x3A, 0x3C})  # line records: their first field is the code offset
PATCH = 0x0A  # a patch record: u8 kind, u16 code offset, then the expression
Json = dict[str, Any]


class PageError(ValueError):
    """A sector that is not the last write of an object file's record stream."""


def _words(rom: bytes | memoryview, first: int, count: int) -> list[int]:
    return list(struct.unpack(f">{count}I", bytes(rom[first : first + 4 * count])))


def _typed(rom: bytes | memoryview, first: int, count: int, what: str) -> list[Json]:
    """`count` words from `first` as typed instructions; each must be canonical code."""
    out = []
    for i, word in enumerate(_words(rom, first, count)):
        if not mipsstream.canonical(word):
            raise PageError(f"{what}: word {word:#010x} at {first + 4 * i:#x} is not a canonical instruction")
        out.append({"at": first + 4 * i, "word": f"{word:08x}", "asm": rabbitizer.Instruction(word).disassemble()})
    return out


def _front(rom: bytes | memoryview, low: int, high: int) -> Json:
    """The tail of a code record's payload in [low, high): the words on the grid that ends at `high`."""
    count = (high - low) // 4
    if count < MIN_WORDS:
        raise PageError(f"front: {high - low} bytes before the first record are fewer than {MIN_WORDS} words of code")
    first = high - 4 * count
    words = _typed(rom, first, count, "front")
    if len({w["word"] for w in words}) < MIN_WORDS // 2:
        raise PageError("front: the words repeat, which is fill and no code")
    return {"range": [low, high], "lead": bytes(rom[low:first]).hex(), "words": words}


def _cut(rom: bytes | memoryview, at: int, high: int) -> Json:
    """The code record whose payload the sector end `high` cut, from its header at `at`."""
    if rom[at] != CODE:
        raise PageError(f"cut: the record at {at:#x} has tag {rom[at]:#x}, not a code record")
    if high - at < 3:
        raise PageError(f"cut: the code record header at {at:#x} is cut")
    (length,) = struct.unpack_from("<H", rom, at + 1)
    if length % 4 or at + 3 + length <= high:
        raise PageError(f"cut: the code record at {at:#x} of {length} bytes is whole or not whole words")
    count = (high - at - 3) // 4
    return {"at": at, "tag": CODE, "length": length, "words": _typed(rom, at + 3, count, "cut")}


def _lined(records: list[snsym.Record]) -> bool:
    """At least two thirds of the records are line records: the debug part of a compile is mostly line numbers, and
    bytes of code read as records come out mostly as patches."""
    return 3 * sum(rec.tag in LINES for rec in records) >= 2 * len(records)


def _ordered(records: list[snsym.Record]) -> bool:
    """No code offset of a patch or line record is lower than the one before it, within a run between code records and
    section switches."""
    last = 0
    for rec in records:
        if rec.tag in (CODE, SECTION):
            last = 0
        elif rec.tag == PATCH or rec.tag in LINES:
            offset = rec.extra[1] if rec.tag == PATCH else rec.extra[0]
            if offset < last:
                return False
            last = offset
    return True


def _chains(rom: bytes | memoryview, low: int, high: int) -> Iterator[tuple[int, list[snsym.Record], Json | None]]:
    """The record starts in [low, high), lowest first, whose chain of whole records lands on `high` or on a cut code
    record."""
    for first in range(low, high):
        records, stop = snsym.records(rom, first, high, "obj")
        if len(records) < MIN_RECORDS or not _lined(records) or not _ordered(records):
            continue
        if stop == high:
            yield first, records, None
            continue
        try:
            yield first, records, _cut(rom, stop, high)
        except PageError:
            continue


def _split(rom: bytes | memoryview, low: int, high: int) -> tuple[int, list[snsym.Record], Json | None, Json | None]:
    """The page's first record, its records, its cut and its front: the lowest chain start with code before it."""
    refusal: PageError | None = None
    for first, records, cut in _chains(rom, low, high):
        try:
            return first, records, cut, _front(rom, low, first) if first > low else None
        except PageError as error:
            refusal = refusal or error
    raise refusal or PageError(f"records: no chain of {MIN_RECORDS} whole object records lands on the end of the page")


def _record(rec: snsym.Record) -> Json:
    out: Json = {"at": rec.at, "end": rec.end, "tag": rec.tag}
    if rec.name:
        out["name"] = rec.name
    if rec.extra:
        out["fields"] = list(rec.extra)
    if rec.size:
        out["size"] = rec.size
    if rec.cls or rec.type:
        out["class"], out["type"] = rec.cls, rec.type
    return out


def compare(front: Json, others: Mapping[str, bytes]) -> list[Json]:
    """Where the words of the front occur in the other versions' ROMs (their first SEED words as a seed), how many
    words are equal from there, and the first word that differs."""
    words = [bytes.fromhex(w["word"]) for w in front["words"]]
    out: list[Json] = []
    for version, rom in sorted(others.items()):
        seed = b"".join(words[:SEED])
        at = rom.find(seed)
        while at >= 0:
            if at % 4 == 0:
                same = [i for i, w in enumerate(words) if rom[at + 4 * i : at + 4 * i + 4] == w]
                differ = next((i for i in range(len(words)) if i not in same), None)
                out.append({"version": version, "rom": at, "equal": len(same), "of": len(words), "first_difference": differ})
            at = rom.find(seed, at + 1)
    return out


def prove(rom: bytes, start: int, end: int, page: list[int], others: Mapping[str, bytes] | None = None) -> Json:
    """The receipt of the region rom[start:end] as part of the page rom[page[0]:page[1]]."""
    low, high = page
    if high - low != SECTOR or low % SECTOR or not 0 <= low < high <= len(rom):
        raise PageError(f"a page is one aligned sector of {SECTOR} bytes inside the ROM")
    if not low <= start < end <= high:
        raise PageError("the region is not inside its page")
    first, records, cut, front = _split(rom, low, high)
    rebuilt = bytes.fromhex(front["lead"]) + b"".join(bytes.fromhex(w["word"]) for w in front["words"]) if front else b""
    if rebuilt != rom[low:first]:
        raise PageError("front: the typed words do not rebuild the page bytes")
    receipt: Json = {
        "page": [low, high],
        "page_sha256": sha256(rom[low:high]),
        "front": front,
        "records": {"range": [first, records[-1].end], "count": len(records), "items": [_record(r) for r in records]},
        "cut": cut,
        "region": [start, end],
        "region_sha256": sha256(rom[start:end]),
    }
    if front and others:
        receipt["compared"] = compare(front, others)
    return receipt


def find(rom: bytes, low: int, high: int) -> list[int]:
    """The page starts (sector aligned) of the sectors that overlap [low, high) and are proven pages."""
    found = []
    for page in range(low // SECTOR * SECTOR, high, SECTOR):
        try:
            prove(rom, max(low, page), min(high, page + SECTOR), [page, page + SECTOR])
        except (PageError, struct.error, snsym.SymError):
            continue
        found.append(page)
    return found
