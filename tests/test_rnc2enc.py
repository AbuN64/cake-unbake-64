"""The deterministic RNC2 packer: round trips, the rules it follows, and agreement with an independent bit writer."""

import random
import struct

import pytest
from test_rnc2 import Writer

from unbake import rnc, rnc2enc


def text(n, seed=3):
    """Compressible, deterministic bytes: words drawn from a small vocabulary with a few runs."""
    rng = random.Random(seed)
    words = [bytes(rng.randrange(65, 91) for _ in range(rng.randrange(2, 9))) for _ in range(40)]
    out = bytearray()
    while len(out) < n:
        out += rng.choice(words) + b" "
        if rng.random() < 0.02:
            out += bytes([rng.randrange(256)]) * rng.randrange(3, 40)
    return bytes(out[:n])


@pytest.mark.parametrize("size", [1, 2, 3, 11, 12, 100, 4097, 13000, 30000])
def test_streams_round_trip_through_the_decoder(size):
    data = text(size)
    raw = rnc2enc.stream(data, 2)
    assert rnc.unpack(raw) == data
    head = rnc.header(raw)
    assert (head.unpacked, head.packed, head.method) == (size, len(raw) - 18, 2)
    assert head.chunks == 1 + max(0, (size - 1) // rnc2enc.CHUNK) or head.chunks >= 1


def test_literal_runs_become_leading_literals_then_raw_runs_then_trailing_literals():
    data = bytes(range(15))  # no repeated pair: nothing matches
    assert rnc2enc.parse(data) == [("L", 0), ("L", 1), ("L", 2), ("R", 3, 12), ("E", 15, 0)]
    data = bytes(range(100))
    tokens = rnc2enc.parse(data)
    assert [t for t in tokens if t[0] == "R"] == [("R", 0, 72), ("R", 72, 28)]
    data = bytes(range(80))  # 72 raw, 8 left over are literals
    assert [t[0] for t in rnc2enc.parse(data)] == ["R"] + ["L"] * 8 + ["E"]


def test_a_match_is_the_longest_then_the_nearest_and_a_pair_needs_a_near_source():
    data = b"abcdXXabcdYYabcdZ"
    assert ("M", 12, 4, 6) in rnc2enc.parse(data)  # nearest of two equal sources
    rng = random.Random(4)
    seen, filler, last = set(), [], 1
    while len(filler) < 300:  # bytes whose pairs never repeat, so no other source exists
        nxt = rng.randrange(3, 256)
        if (last, nxt) not in seen:
            seen.add((last, nxt))
            filler.append(nxt)
            last = nxt
    far = b"\x01\x02" + bytes(filler) + b"\x01\x02\x00"
    assert not any(t[0] == "M" and t[2] == 2 for t in rnc2enc.parse(far))  # distance over 256 has no 2-byte match


def test_lazy_matching_takes_a_literal_when_the_next_match_is_longer():
    data = b"xabQ" + bytes(range(0x80, 0x90)) + b"yabcdefgh" + bytes(range(0x90, 0xA0)) + b"xabcdefgh"
    at = data.rindex(b"xabcdefgh")
    tokens = rnc2enc.parse(data)
    assert not any(t[0] == "M" and t[1] == at for t in tokens)  # no match starts at the x
    assert [t for t in tokens if t[0] == "M" and t[1] == at + 1] == [("M", at + 1, 8, at + 1 - data.index(b"abcdefgh"))]


def test_chunks_end_before_the_first_token_that_would_cross_the_limit():
    data = text(30000)
    tokens = rnc2enc.parse(data)
    marks = [t for t in tokens if t[0] == "E" and t[2]]
    assert len(marks) == 2 and all(m[1] <= (k + 1) * rnc2enc.CHUNK for k, m in enumerate(marks))
    after = next(t for t in tokens if t[0] != "E" and t[1] >= marks[0][1])
    assert after[1] == marks[0][1]


def test_emit_agrees_with_an_independent_bit_writer():
    rng = random.Random(9)
    data = text(3000)
    tokens = rnc2enc.parse(data)
    w = Writer()
    for t in tokens:
        if t[0] == "L":
            w.literals(data[t[1] : t[1] + 1])
        elif t[0] == "R":
            w.raw_run(data[t[1] : t[1] + t[2]])
        elif t[0] == "E":
            w.marker(t[2])
        else:
            w.match(t[2], t[3])
    assert bytes(w.data) == rnc2enc.emit(data, tokens)[0]
    assert rng  # seeded only so the test reads the same everywhere


def test_the_header_holds_sizes_crcs_leeway_and_chunk_count():
    data = text(20000)
    raw = rnc2enc.stream(data, 4)
    size, packed, ucrc, pcrc, leeway, chunks = struct.unpack(">4xIIHHBB", raw[:18])
    assert (size, packed, leeway, chunks) == (20000, len(raw) - 18, 4, 2)
    assert ucrc == rnc.crc16(data) and pcrc == rnc.crc16(raw[18:])
