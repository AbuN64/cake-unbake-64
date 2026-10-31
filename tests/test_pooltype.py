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


def test_a_double_off_its_windows_8_byte_phase_opens_its_own_window():
    pool = words(0x3F800000, 0x4123D70A, 0xA4060000, 0x40000000)
    items, _ = pt.itemize(pool, 0, len(pool), 4)
    assert [i.kind for i in items] == ["float", "double", "float"]
    assert [[i.kind for i in u.items] for u in pt.units(items)] == [["float"], ["double", "float"]]


def test_render_floats_exactly_and_strings_padded():
    floats = pt.objects([pt.Item("float", 0, 4, (0x3F800000,)), pt.Item("float", 4, 4, (0x3FC90FDB,))])
    assert pt.render(floats[0], "x") == ["float x[2] = { 1.0f, 1.57079637f };"]
    assert pt.render(pt.Item("str", 0, 8, 'a"b'), "s") == ['char s[8] = "a\\"b";']
    assert pt.render(pt.Item("uint", 0, 8, (0x18, 5)), "w") == ["unsigned int w[2] = { 0x18, 5 };"]
    assert pt.render(pt.Item("int", 0, 4, (0xFFFFFFFF,)), "n") == ["int n[1] = { -1 };"]


def rom_of(*w: int) -> bytes:
    """A ROM image big enough that 0x22xxxx is an offset inside it."""
    body = words(*w)
    return body + b"\0" * (0x300000 - len(body))


def kinds(*w: int) -> tuple[list[str], list[dict]]:
    items, refused = pt.itemize(rom_of(*w), 0, 4 * len(w), 0)
    return [i.kind for i in items], refused


def test_offset_words_are_typed_in_a_run_or_beside_a_typed_word():
    assert kinds(0x00223260, 0x00223270, 0x18)[0] == ["uint", "uint", "uint"]  # two offsets in a row
    assert kinds(0x3F800000, 0x3F000000, 0x0026FCB4)[0] == ["float", "float", "uint"]  # one beside a float
    assert kinds(0x00223260, 0x084B0E0C, 0x084B0E0D)[0] == []  # a bare offset beside packed bytes
    assert kinds(0x00223261, 0x00223271)[0] == []  # not word aligned: no offset
    assert pt.itemize(words(0x00223260, 0x00223270), 0, 8, 0)[0] == []  # beyond this image's size: no offset


def test_pointer_words_are_typed_inside_a_record_never_alone():
    assert kinds(0x801708C4, 0x0022E004, 0x0, 0x801708D4)[0] == [
        "ptr",
        "uint",
        "uint",
        "ptr",
    ]  # zero joins the integers
    assert kinds(0x801708C4, 0x00000064)[0] == ["ptr", "uint"]  # a pointer beside a small integer
    got, refused = kinds(0x801547B4, 0x801547B0, 0x801547A0)
    assert got == []  # a table of pointers alone is the address-table reader's
    assert refused == [{"rom": 0, "end": 12, "why": "pointer table (typed by the address-table reader)"}]
    assert kinds(0x80000100, 0x0022E004)[0] == []  # below the image: no pointer, and a bare offset is no record


def test_a_pointer_run_is_a_void_pointer_array_of_address_constants():
    items, _ = pt.itemize(rom_of(0x801708C4, 0x0022E004, 0, 0x801708D4), 0, 16, 0)
    unit = pt.Unit("", 0x8000, items)
    lines, symbols = pt.body(unit, 0x80100000)
    assert lines == [
        "void *g_earlierPoolPointers_80100000[1] = { (void *)0x801708C4 };",
        "unsigned int g_earlierPoolWords_80100004[2] = { 0x22E004, 0 };",
        "void *g_earlierPoolPointers_8010000C[1] = { (void *)0x801708D4 };",
    ]
    assert all(s["name"].startswith("g_earlier") and "rom" in s for s in symbols)


def test_words_of_widens_to_words_without_entering_code_data_or_other_windows():
    unknown = [(0x101, 0x103), (0x10A, 0x10D), (0x200, 0x202), (0x300, 0x303)]
    hard = [(0x0, 0x100), (0x204, 0x300)]
    got = pt.words_of(unknown, hard, 0, 0x400)
    assert got == [(0x100, 0x104), (0x108, 0x110), (0x200, 0x204), (0x300, 0x304)]
    assert pt.words_of([(0x101, 0x103)], [(0, 0x101)], 0, 0x400) == []  # no whole word is free
    assert pt.words_of([(0x101, 0x107)], [], 0x100, 0x106) == [(0x100, 0x104)]  # never past the range
    assert pt.words_of([(0x100, 0x104), (0x104, 0x108)], [], 0, 0x400) == [(0x100, 0x108)]  # touching spans merge


def test_latin1_accents_count_as_text_after_an_ascii_letter():
    items, _ = pt.itemize(b"publi\xe9\0\0d\xe9velopp\xe9 par\0\0\0", 0, 24)
    assert [(i.kind, i.value, i.size) for i in items] == [("str", "publi\xe9", 8), ("str", "d\xe9velopp\xe9 par", 16)]
    assert pt.objects(items)[0].value == "publi\xe9"
    assert pt._literal("publi\xe9 \xe9s") == '"publi\\351 \\351s"'


def test_a_high_byte_without_a_letter_before_it_is_no_text():
    for raw in (b"\xe9bc\0", b"ab \xe9c\0", b"abc\xe9\xe9\x01\0"):
        padded = raw.ljust(8, b"\0")
        assert [i.kind for i in pt.itemize(padded, 0, 8)[0]][:1] != ["str"]
