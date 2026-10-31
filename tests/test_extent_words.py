"""Extent words: wrapper size words and container offset/size words proven from the sections they describe."""

import pytest
from test_rnc2enc import text

from unbake import blob_formats, extent_words, rnc2enc

SALT = 11


def wrapped(seed):
    """A wrapper record: its four words, an RNC stream and the zero padding up to the next 8."""
    stream = rnc2enc.stream(text(300 + seed), 2)
    size = -(-(0x10 + 18 + len(rnc2enc_payload(stream))) // 8) * 8
    body = (1).to_bytes(4, "big") + (0x10).to_bytes(4, "big") + size.to_bytes(4, "big") + bytes(4) + stream
    return body + bytes(size - len(body))


def rnc2enc_payload(stream):
    """The packed bytes of a stream (it is 18 header bytes, the packed data and the trailer)."""
    return stream[18:]


def container(sections, slack=0):
    """`n, offsets, size` then the sections, each followed by `slack` zero bytes the words do not count."""
    n = len(sections)
    offsets = [4 * (n + 2)]
    for part in sections:
        offsets.append(offsets[-1] + len(part))
    words = [n, *offsets]
    head = b"".join(w.to_bytes(4, "big") for w in words)
    return head + b"".join(part + bytes(slack) for part in sections)


def test_a_wrapper_header_follows_from_its_stream():
    rom = bytes(8) + wrapped(1)
    got = extent_words.prove(rom, [], 8, 0x18, wrapper_at=8)
    assert got["at"] == 8 and got["size"] == len(rom) - 8 and got["slack"] == 0


def test_a_wrapper_with_a_wrong_size_word_or_stream_is_refused():
    rom = bytearray(wrapped(2))
    rom[11] += 8
    with pytest.raises(extent_words.ExtentError, match="says"):
        extent_words.prove(bytes(rom), [], 0, 0x10, wrapper_at=0)
    rom = bytearray(wrapped(2))
    rom[0x10] = ord("X")
    with pytest.raises(extent_words.ExtentError, match="wrapper"):
        extent_words.prove(bytes(rom), [], 0, 0x10, wrapper_at=0)


def test_a_wrapper_claim_takes_only_zero_slack_before_its_header():
    rom = b"\xff" + bytes(3) + wrapped(3)
    assert extent_words.prove(rom, [], 1, 0x14, wrapper_at=4)["slack"] == 3
    with pytest.raises(extent_words.ExtentError):
        extent_words.prove(rom, [], 0, 0x14, wrapper_at=4)
    with pytest.raises(extent_words.ExtentError):
        extent_words.prove(rom, [], 4, 0x18, wrapper_at=4)


def test_container_words_follow_from_self_sized_sections():
    parts = [wrapped(4), wrapped(5), wrapped(6), wrapped(7)]
    rom = container(parts)
    head = 4 * (len(parts) + 2)
    got = extent_words.prove(rom, [], 0, head, container=0)
    assert got["words"] == list(range(len(parts) + 2)) and got["size"] == len(rom)
    assert [s["kind"] for s in got["sections"]] == ["wrapper"] * 4


def test_a_word_that_does_not_add_up_is_refused():
    rom = bytearray(container([wrapped(4), wrapped(5)]))
    rom[8 + 3] ^= 8  # offset[1] now disagrees with the first section's extent
    with pytest.raises(extent_words.ExtentError):
        extent_words.prove(bytes(rom), [], 0, 16, container=0)


def test_zero_slack_before_a_section_does_not_move_the_words():
    parts = [wrapped(4), wrapped(5), wrapped(6), wrapped(7)]
    rom = bytearray(container(parts))
    at = 4 * 6 + len(parts[0])  # where the second section stands; three zeros move it and the ones after it
    rom[at:at] = bytes(3)
    got = extent_words.sections(bytes(rom), 0, [])
    assert [s["slack"] for s in got] == [0, 3, 0, 0] and [s["kind"] for s in got] == ["wrapper"] * 4
    assert got[3]["at"] == 4 * 6 + sum(len(p) for p in parts[:3]) + 3
    head = extent_words.prove(bytes(rom), [], 0, 4 * 6, container=0)
    assert head["words"] == list(range(6))


def test_an_opaque_section_blocks_the_words_after_it_unless_it_is_accounted():
    parts = [wrapped(4), bytes(range(1, 41)), wrapped(6), wrapped(7)]
    rom = container(parts)
    head = 4 * (len(parts) + 2)
    with pytest.raises(extent_words.ExtentError, match="neither self-sized nor accounted"):
        extent_words.prove(rom, [], 0, head, container=0)
    middle = (head + len(parts[0]), head + len(parts[0]) + 40)
    got = extent_words.prove(rom, [middle], 0, head, container=0)
    assert [s["kind"] for s in got["sections"]] == ["wrapper", "opaque", "wrapper", "wrapper"]
    # offset[1] needs only the section before it; the later words are what an opaque section blocks
    early = extent_words.prove(rom, [], 0, 12, container=0)
    assert early["words"] == [0, 1, 2]


def test_find_closes_the_short_unknown_spans_and_leaves_the_rest():
    parts = [wrapped(4), wrapped(5)]
    rom = bytes(16) + container(parts)
    head = 16 + 4 * 4
    got = extent_words.find(rom, [], [(16, head), (0, 4)])
    assert got == [{"region": [16, head], "container": 16}]
    wrap = 16 + 4 * 4 + 0
    again = extent_words.find(rom, [], [(wrap, wrap + 0x10)])
    assert again == [{"region": [wrap, wrap + 0x10], "wrapper": wrap}]


def test_the_reader_is_a_blob_format_with_its_own_dispatch():
    assert "extent-words" in blob_formats.READERS
