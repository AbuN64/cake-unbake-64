"""linkdiag: the first object a link script places off the address its first name owns."""

from unbake import linkdiag

LINK = """SECTIONS
{
    .span_1000 0x80200400 : AT(0x1000) SUBALIGN(1)
    {
        src/a.c.o(.text);
        src/b.c.o(.text);
        src/c.c.o(.text);
    }
    .span_2000 0x80300000 : AT(0x2000) SUBALIGN(1)
    {
        src/d.c.o(.text);
    }
}
"""
SIZES = {"src/a.c": 0x40, "src/b.c": 0x30, "src/c.c": 0x10, "src/d.c": 0x8}


def _size(obj: str, section: str) -> int | None:
    return SIZES.get(obj)


def test_a_link_where_every_object_starts_at_its_first_name_is_clean() -> None:
    expected = {("src/a.c", ".text"): 0x80200400, ("src/b.c", ".text"): 0x80200440, ("src/c.c", ".text"): 0x80200470}
    assert linkdiag.walk(LINK, _size, expected) is None


def test_the_first_object_after_a_too_long_one_is_named_with_both_addresses() -> None:
    expected = {("src/a.c", ".text"): 0x80200400, ("src/b.c", ".text"): 0x80200438, ("src/c.c", ".text"): 0x80200468}
    assert linkdiag.walk(LINK, _size, expected) == ("src/b.c", 0x80200438, 0x80200440)


def test_a_segment_restarts_the_cursor_and_unowned_objects_are_skipped() -> None:
    expected = {("src/d.c", ".text"): 0x80300000}
    assert linkdiag.walk(LINK, _size, expected) is None
