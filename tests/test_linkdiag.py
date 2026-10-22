"""linkdiag: the first object a link places off the address its first name owns."""

from __future__ import annotations

from unbake import linkdiag

MAP = """Linker script and memory map

.span_1000       0x0000000080200400      0x7f0 load address 0x0000000000001000
 .text          0x0000000080200400       0x40 src/a.c.o
 .text          0x0000000080200440       0x30 src/b.c.o
 .rdata         0x0000000080200470       0x10 src/b.c.o
 .bss           0x0000000080200480       0x20 src/b.c.o
 .text
                0x0000000080200480       0x10 src/some/very/long/path/name/c.c.o
 .text          0x0000000080200490        0x0 src/empty.c.o
"""


def test_parse_reads_nonempty_sections_in_link_order_with_a_long_name_on_its_own_line() -> None:
    rows = linkdiag.parse(MAP)
    assert rows == [
        ("src/a.c.o", ".text", 0x80200400),
        ("src/b.c.o", ".text", 0x80200440),
        ("src/b.c.o", ".rodata", 0x80200470),
        ("src/some/very/long/path/name/c.c.o", ".text", 0x80200480),
    ]


def test_a_link_where_every_object_is_at_its_first_name_is_clean() -> None:
    expected = {("src/a.c.o", ".text"): 0x80200400, ("src/b.c.o", ".text"): 0x80200440}
    assert linkdiag.walk(linkdiag.parse(MAP), expected) is None


def test_the_first_object_of_a_persisting_shift_is_named_with_both_addresses() -> None:
    expected = {
        ("src/a.c.o", ".text"): 0x80200400,
        ("src/b.c.o", ".text"): 0x80200438,
        ("src/b.c.o", ".rodata"): 0x80200468,
    }
    assert linkdiag.walk(linkdiag.parse(MAP), expected) == ("src/b.c.o", 0x80200438, 0x80200440)


def test_an_object_that_starts_with_unnamed_bytes_is_off_alone_and_not_a_shift() -> None:
    expected = {("src/a.c.o", ".text"): 0x80200404, ("src/b.c.o", ".text"): 0x80200440}
    assert linkdiag.walk(linkdiag.parse(MAP), expected) is None
