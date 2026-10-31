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


def padded(sections):
    """A container as the builder wrote it: an odd count pads the header to 8, the sections stand 8-aligned."""
    n = len(sections)
    first = -(-4 * (n + 2) // 8) * 8
    offsets = [first]
    for part in sections:
        offsets.append(offsets[-1] + len(part))
    head = b"".join(w.to_bytes(4, "big") for w in [n, *offsets])
    return head + bytes(first - len(head)) + b"".join(sections)


def bare(seed):
    """A bare RNC stream section, zero padded to the 8 the next offset needs."""
    stream = rnc2enc.stream(text(300 + seed), 2)
    return stream + bytes(-len(stream) % 8)


def test_an_odd_count_pads_its_header_and_the_pad_is_determined():
    parts = [wrapped(8), wrapped(9), wrapped(10)]
    rom = padded(parts)
    head = 8 * 3
    got = extent_words.prove(rom, [], 0, head, container=0)
    assert got["size"] == len(rom) and got["words"] == list(range(5))
    assert (0, head) in extent_words.determined(rom, 0, extent_words.sections(rom, 0, []))


def test_an_empty_container_is_its_two_words():
    rom = (0).to_bytes(4, "big") + (8).to_bytes(4, "big")
    assert extent_words.shape(rom, 0) == (0, [], 8)
    with pytest.raises(extent_words.ExtentError):
        extent_words.prove(rom, [], 0, 8, container=0)  # nothing follows from no sections


def test_bare_streams_and_zero_sections_are_self_sized_sections():
    parts = [bare(11), bytes(8), wrapped(13), bare(14)]
    rom = padded(parts)
    assert [s["kind"] for s in extent_words.sections(rom, 0, [])] == ["stream", "zero", "wrapper", "stream"]
    assert extent_words.prove(rom, [], 0, 4 * 6, container=0)["words"] == list(range(6))
    # the stream's zero pad and the zero section after it are one determined range
    assert (243, 256) in extent_words.determined(rom, 0, extent_words.sections(rom, 0, []))


def test_a_zero_section_is_taken_only_before_a_section_that_begins():
    rom = padded([bytes(8), wrapped(16)])
    assert [s["kind"] for s in extent_words.sections(rom, 0, [])] == ["zero", "wrapper"]
    lone = padded([bytes(8), bytes(range(1, 9))])
    assert extent_words.sections(lone, 0, [])[0]["kind"] == "opaque"
