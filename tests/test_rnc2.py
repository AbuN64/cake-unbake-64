"""Independent token encoder round trips and fixed RNC2 golden bytes, without game assets."""

import struct

import pytest

from unbake import rnc


class Writer:
    def __init__(self):
        self.data = bytearray()
        self.slot = self.used = 0
        self.bits(0, 2)

    def bits(self, value, count):
        for shift in range(count - 1, -1, -1):
            if self.used == 0:
                self.slot = len(self.data)
                self.data.append(0)
            self.data[self.slot] |= ((value >> shift) & 1) << (7 - self.used)
            self.used = (self.used + 1) % 8

    def literals(self, data):
        for byte in data:
            self.bits(0, 1)
            self.data.append(byte)

    def raw_run(self, data):
        self.bits(0b10111, 5)
        self.bits((len(data) - 12) // 4, 4)
        self.data.extend(data)

    def marker(self, more):
        self.bits(15, 4)
        self.data.append(0)
        self.bits(int(more), 1)

    def match(self, length, distance):
        if length == 2:
            self.bits(6, 3)
        elif length == 3:
            self.bits(14, 4)
        elif length <= 5:
            self.bits(8 + (length - 4) * 2, 4)
        elif length <= 8:
            self.bits({6: 18, 7: 19, 8: 22}[length], 5)
        else:
            self.bits(15, 4)
            self.data.append(length - 8)
        high, low = divmod(distance - 1, 256)
        if length != 2:
            if high == 0:
                self.bits(0, 1)
            elif high == 1:
                self.bits(6, 3)
            elif high < 4:
                self.bits(8 + high - 2, 4)
            else:
                value = high if high < 8 else high // 2
                self.bits(1, 1)
                self.bits((value >> 1) & 1, 1)
                self.bits(1, 1)
                self.bits(value & 1, 1)
                self.bits(int(high < 8), 1)
                if high >= 8:
                    self.bits(high & 1, 1)
        self.data.append(low)

    def stream(self, output):
        self.marker(False)
        return stream(bytes(self.data), output)


def stream(packed, output):
    return (
        struct.pack(">4sIIHHBB", b"RNC\x02", len(output), len(packed), rnc.crc16(output), rnc.crc16(packed), 0, 1)
        + packed
    )


def test_literal_golden_and_interleaved_match():
    # Two flags, three literal controls, 1111 end, zero length byte, zero final bit.
    assert rnc.unpack(stream(bytes.fromhex("076162638000"), b"abc")) == b"abc"
    w = Writer()
    w.literals(b"abc")
    w.match(6, 3)
    assert rnc.unpack(w.stream(b"abcabcabc")) == b"abcabcabc"


@pytest.mark.parametrize("length", [2, 3, 4, 5, 6, 7, 8, 9, 10, 263])
@pytest.mark.parametrize(
    "distance", [1, 2, 256, 257, 512, 769, 1025, 1281, 1537, 1793, 2049, 2305, 2561, 2817, 3073, 3329, 3585, 4096]
)
def test_all_match_lengths_distances_and_overlaps(length, distance):
    if length == 2 and distance > 256:
        return  # two-byte matches have no high distance bits
    prefix = bytes(i % 251 for i in range(distance))
    output = bytearray(prefix)
    for _ in range(length):
        output.append(output[-distance])
    w = Writer()
    w.literals(prefix)
    w.match(length, distance)
    assert rnc.unpack(w.stream(bytes(output))) == bytes(output)


@pytest.mark.parametrize("count", range(12, 73, 4))
def test_raw_runs_and_chunk_continuation(count):
    w = Writer()
    data = bytes(range(count))
    w.raw_run(data)
    w.marker(True)
    w.literals(b"end")
    result = w.stream(data + b"end")
    assert rnc.unpack(result) == data + b"end"
    assert [(at, h.method) for at, h in rnc.find(b"pad" + result)] == [(3, 2)]


def test_corruption_bounds_and_termination():
    w = Writer()
    w.literals(b"abc")
    good = w.stream(b"abc")
    for index in (12, 14):
        damaged = bytearray(good)
        damaged[index] ^= 1
        with pytest.raises(rnc.RncError, match="CRC mismatch"):
            rnc.unpack(damaged)
    assert rnc.unpack(damaged, check=False) == b"abc"
    for packed, output, message in [
        (good[18:-1], b"abc", "end early"),
        (good[18:] + b"x", b"abc", "after end"),
        (good[18:], b"abcd", "decodes to"),
        (good[18:], b"ab", "past unpacked"),
        (b"\x80", b"a", "locked or encrypted"),
        (b"\x40", b"a", "locked or encrypted"),
    ]:
        with pytest.raises(rnc.RncError, match=message):
            rnc.unpack(stream(packed, output))
    w = Writer()
    w.match(3, 1)
    with pytest.raises(rnc.RncError, match="match outside"):
        rnc.unpack(w.stream(b"aaa"))
    w = Writer()
    w.raw_run(bytes(12))
    with pytest.raises(rnc.RncError, match="raw run past"):
        rnc.unpack(w.stream(b"a"))
