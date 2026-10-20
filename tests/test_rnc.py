"""RNC1 streams: CRC, header, decoder and finder on streams built here bit by bit (no game bytes)."""

import struct

import pytest

from unbake import rnc


class _Writer:
    """Packs bits low first into 16-bit little-endian words the way the reader takes them; a literal run goes in
    straight after the word being filled, and the next word starts after it."""

    def __init__(self) -> None:
        self.out, self.word, self.used = bytearray(), 0, 16

    def bits(self, value: int, count: int) -> None:
        for i in range(count):
            if self.used == 16:
                self.word, self.used = len(self.out), 0
                self.out += b"\0\0"
            self.out[self.word + self.used // 8] |= (value >> i & 1) << (self.used % 8)
            self.used += 1

    def table(self, lengths: list[int]) -> None:
        self.bits(len(lengths), 5)
        for length in lengths:
            self.bits(length, 4)

    def number(self, code: int, size: int, value: int) -> None:
        """A one-bit-code symbol: `code` for the symbol, then the extra bits of `value` (symbol = bit length)."""
        self.bits(code, size)
        if value >= 2:
            self.bits(value, value.bit_length() - 1)


def _stream(packed: bytes, unpacked: bytes) -> bytes:
    head = struct.pack(">4sIIHHBB", rnc.MAGIC, len(unpacked), len(packed), rnc.crc16(unpacked), rnc.crc16(packed), 0, 1)
    return head + packed


def _abcabc() -> bytes:
    """'abc', a match 3 back of 6 bytes, then the final empty literal run: 'abcabcabc'."""
    w = _Writer()
    w.bits(0, 2)
    w.table([1, 0, 1])  # literal runs: symbol 0 (code 0), symbol 2 (code 1, values 2..3)
    w.table([0, 0, 1])  # distances: symbol 2 (values 2..3)
    w.table([0, 0, 0, 1])  # lengths: symbol 3 (values 4..7)
    w.bits(2, 16)  # two sub-chunks
    w.number(1, 1, 3)
    w.out += b"abc"  # the rest of the word being filled is read first, then the word after the run
    w.number(0, 1, 2)  # distance 2 + 1 = 3
    w.number(0, 1, 4)  # length 4 + 2 = 6
    w.number(0, 1, 0)  # last sub-chunk: empty literal run
    return _stream(bytes(w.out), b"abcabcabc")


def test_crc16_is_arc():
    assert rnc.crc16(b"123456789") == 0xBB3D and rnc.crc16(b"") == 0


def test_a_stream_with_a_literal_run_and_an_overlapping_match_decodes():
    stream = _abcabc()
    head = rnc.header(stream)
    assert (head.unpacked, head.stored) == (9, len(stream))
    assert rnc.unpack(b"pad" + stream, 3) == b"abcabcabc"


def test_crc_and_size_mismatches_refuse():
    stream = bytearray(_abcabc())
    stream[15] ^= 1  # packed CRC
    with pytest.raises(rnc.RncError, match="packed CRC"):
        rnc.unpack(bytes(stream))
    assert rnc.unpack(bytes(stream), check=False) == b"abcabcabc"
    stream[15] ^= 1
    stream[13] ^= 1  # unpacked CRC
    with pytest.raises(rnc.RncError, match="unpacked CRC"):
        rnc.unpack(bytes(stream))
    with pytest.raises(rnc.RncError, match="sizes outside"):
        rnc.header(bytes(stream[:-1]))
    with pytest.raises(rnc.RncError, match="no RNC1 header"):
        rnc.header(b"RNC\x02" + bytes(14))


def test_find_lists_whole_streams_only():
    stream = _abcabc()
    data = bytes(7) + stream + b"RNC\x01" + bytes(4)
    assert [at for at, _ in rnc.find(data)] == [7]
    assert rnc.find(data, 8) == [] and rnc.find(data, 0, 7 + len(stream) - 1) == []
