"""The rodata pool typer: synthetic pool bytes (strings, floats, a double, small words, refused words)."""

import struct

from unbake import pooltype as pt


def words(*w: int) -> bytes:
    return b"".join(struct.pack(">I", x) for x in w)


def text(s: str) -> bytes:
    raw = s.encode() + b"\0"
    return raw + b"\0" * (-len(raw) % 4)


PATH = "C:/Projects/DM/n64/Source/Camera/gsky.cpp"
POOL = (
    words(0x3F800000, 0x4F000000, 0, 0x18)  # floats, a zero, small integers
    + text("pThis")
    + words(0x40449BA6, 0x9999999A)  # a double on an 8-byte phase
    + words(0x0026FCB4)  # a pointer-looking word beside nothing: refused
    + text(PATH)
    + words(0x3F000000)
    + text("pCamera")
)


def test_itemize_types_by_bits_and_refuses_the_rest():
    items, refused = pt.itemize(POOL, 0, len(POOL), 0)
    assert [i.kind for i in items] == ["float", "float", "float", "uint", "str", "double", "str", "float", "str"]
    assert items[2].value == (0,)  # the zero follows the floats before it
    assert items[5].value == (0x40449BA6, 0x9999999A)
    assert [(r["rom"], r["end"]) for r in refused] == [(0x20, 0x24)]


def test_string_padding_must_be_nul():
    assert pt.itemize(b"abc\0xyz\0", 0, 8)[0][0].size == 4
    items, _ = pt.itemize(b"abcd\0\0\0\x01", 0, 8)
    assert [i.kind for i in items] == ["uint"]  # padding that is not NUL: no string, the first word is refused


def test_units_split_per_file_after_the_last_string_before_a_path():
    items, _ = pt.itemize(POOL, 0, len(POOL), 0)
    units = pt.units(items)
    assert [u.path for u in units] == [PATH, PATH]  # the refused word splits the file into two windows
    assert [i.kind for i in units[0].items] == ["float", "float", "float", "uint", "str", "double"]
    assert units[1].file == f"EARLIER_CAMERA_GSKY_{units[1].rom:X}.CPP"
    assert pt.stem(PATH) == "Gsky"


def test_render_floats_exactly_and_strings_padded():
    floats = pt.objects([pt.Item("float", 0, 4, (0x3F800000,)), pt.Item("float", 4, 4, (0x3FC90FDB,))])
    assert pt.render(floats[0], "x") == ["float x[2] = { 1.0f, 1.57079637f };"]
    assert pt.render(pt.Item("str", 0, 8, 'a"b'), "s") == ['char s[8] = "a\\"b";']
    assert pt.render(pt.Item("uint", 0, 8, (0x18, 5)), "w") == ["unsigned int w[2] = { 0x18, 5 };"]
    assert pt.render(pt.Item("int", 0, 4, (0xFFFFFFFF,)), "n") == ["int n[1] = { -1 };"]
