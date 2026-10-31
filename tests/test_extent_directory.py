"""Extent directories: tables of the extents of the blobs a bank holds, proven from the decoded blobs."""

import pytest

from unbake import blob_formats, extent_directory


def words(values):
    return b"".join(v.to_bytes(4, "big") for v in values)


def blobs(*pairs):
    return {f"b{i}": {"state": "decoded", "rom": {"de": [list(p)]}} for i, p in enumerate(pairs)}


def test_the_boundaries_are_those_of_top_level_decoded_blobs():
    document = blobs((0x20, 0x40))
    document["child"] = {"state": "decoded", "parent": "b0", "rom": {"de": [[0x28, 0x30]]}}
    document["gone"] = {"state": "unknown", "rom": {"de": [[0x50, 0x60]]}}
    document["other"] = {"state": "identified", "rom": {"us": [[0x70, 0x80]]}}
    assert extent_directory.boundaries(document, "de") == {0x20, 0x40}


def test_pairs_tile_the_bank_from_the_end_of_the_table():
    rom = words([0x10, 0x30, 0x30, 0x50]) + bytes(0x40)
    bounds = extent_directory.boundaries(blobs((0x10, 0x30), (0x30, 0x50)), "de")
    got = extent_directory.prove(rom, bounds, 0, 0x10, "pairs")
    assert got["blobs"] == [[0x10, 0x30], [0x30, 0x50]]


def test_pairs_that_leave_a_gap_or_start_elsewhere_are_refused():
    bounds = extent_directory.boundaries(blobs((0x10, 0x30), (0x38, 0x50), (0x20, 0x28)), "de")
    with pytest.raises(extent_directory.DirectoryError, match="tile"):
        extent_directory.prove(words([0x10, 0x30, 0x38, 0x50]), bounds, 0, 0x10, "pairs")
    with pytest.raises(extent_directory.DirectoryError, match="first pair"):
        extent_directory.prove(words([0x20, 0x28, 0x28, 0x30]), bounds, 0, 0x10, "pairs")


def test_a_pair_word_that_names_no_blob_is_refused():
    bounds = extent_directory.boundaries(blobs((0x10, 0x30)), "de")
    with pytest.raises(extent_directory.DirectoryError, match="boundary of no decoded blob"):
        extent_directory.prove(words([0x10, 0x30, 0x30, 0x50]), bounds, 0, 0x10, "pairs")


def test_ends_are_relative_to_the_table_and_the_length_word_is_checked():
    rom = bytes(0x100) + words([2, 16, 0x30, 0x50])
    bounds = extent_directory.boundaries(blobs((0x110, 0x130), (0x130, 0x150)), "de")
    got = extent_directory.prove(rom, bounds, 0x100, 0x110, "ends")
    assert got["table"] == 0x100 and got["ends"] == [0x130, 0x150]
    for bad in ([2, 20, 0x30, 0x50], [3, 16, 0x30, 0x50], [2, 16, 0x50, 0x30]):
        with pytest.raises(extent_directory.DirectoryError):
            extent_directory.prove(bytes(0x100) + words(bad), bounds, 0x100, 0x110, "ends")


def test_an_unknown_shape_or_a_region_off_the_rom_is_refused():
    with pytest.raises(extent_directory.DirectoryError, match="shape"):
        extent_directory.prove(bytes(16), frozenset(), 0, 8, "rings")
    with pytest.raises(extent_directory.DirectoryError, match="whole words"):
        extent_directory.prove(bytes(16), frozenset(), 0, 6, "pairs")


def test_find_reads_only_the_short_spans_that_prove():
    rom = words([0x10, 0x30, 0x30, 0x50]) + bytes(0x400)
    bounds = extent_directory.boundaries(blobs((0x10, 0x30), (0x30, 0x50)), "de")
    got = extent_directory.find(rom, bounds, [(0, 0x10), (0x10, 0x18), (0, 0x400)])
    assert got == [{"region": [0, 16], "shape": "pairs"}]


def test_the_reader_is_a_blob_format():
    assert "extent-directory" in blob_formats.READERS
