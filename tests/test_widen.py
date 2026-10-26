"""widen: the conditional rewrite that lets a version build the C the reference matches, and the counterpart reading of
the data symbols that C spells."""

from __future__ import annotations

from unbake import widen

U = widen.Universe(("de", "eu", "us", "us-rev1"), "us-rev1", ("de",))
ALL = widen.Universe(("de", "eu", "us", "us-rev1"), "us-rev1", ("de", "eu", "us"))


def taken(text: str, u: widen.Universe, version: str, nm: bool = False) -> str:
    """The lines version VERSION compiles of TEXT (a toy preprocessor over the conditionals widen writes)."""
    out, stack = [], []
    env = u.env(version, nm)
    for line in text.splitlines():
        word, cond = widen._condition(line) if line.lstrip().startswith("#") else ("", "")
        live = all(s[0] for s in stack)
        if word in ("if", "ifdef", "ifndef"):
            ok = widen.evaluator(cond, u.known)(env)  # type: ignore[misc]
            stack.append([ok, ok])
        elif word == "elif":
            ok = not stack[-1][1] and widen.evaluator(cond, u.known)(env)  # type: ignore[misc]
            stack[-1] = [ok, stack[-1][1] or ok]
        elif word == "else":
            stack[-1] = [not stack[-1][1], True]
        elif word == "endif":
            stack.pop()
        elif live:
            out.append(line)
    return "\n".join(out)


SIMPLE = """#include "x.h"
#if defined(VERSION_US_REV1)
s32 Foo(void) { return D_80100000; }
#else
INCLUDE_ASM("span_1000", Foo);
#endif
INCLUDE_ASM("span_1000", Bar);
"""


def test_the_widened_version_builds_what_the_reference_builds_and_no_other_version_changes():
    after, decided, unjudged = widen.widen_text(SIMPLE, {"Foo"}, U)
    assert not unjudged and decided == {"Foo"}
    assert "#if defined(VERSION_US_REV1) || defined(VERSION_DE)" in after
    for v in U.versions:
        for nm in (False, True):
            want = taken(SIMPLE, U, U.as_ref(v), nm)
            assert taken(after, U, v, nm) == want


def test_widening_to_every_version_leaves_only_the_c():
    after, _, _ = widen.widen_text(SIMPLE, {"Foo"}, ALL)
    assert "#if" not in after and 'INCLUDE_ASM("span_1000", Foo)' not in after
    assert "s32 Foo(void)" in after and 'INCLUDE_ASM("span_1000", Bar)' in after


def test_a_draft_no_version_builds_any_more_goes_with_its_conditional():
    text = (
        "#if defined(NON_MATCHING) && !defined(VERSION_US_REV1)\nvoid Foo(void) { draft(); }\n"
        '#elif !defined(VERSION_US_REV1)\nINCLUDE_ASM("span_1000", Foo);\n#endif\nint keep;\n'
    )
    after, _, _ = widen.widen_text(text, {"Foo"}, ALL)
    assert after == "int keep;\n"


def test_a_condition_naming_the_widened_version_is_written_again_from_its_truth_table():
    text = (
        '#if !defined(VERSION_DE) && !defined(VERSION_EU)\nvoid Foo(void) {}\n#else\nINCLUDE_ASM("s", Foo);\n#endif\n'
    )
    after, _, unjudged = widen.widen_text(text, {"Foo"}, U)
    assert not unjudged and "VERSION_DE" not in after.splitlines()[0]
    for v in U.versions:
        for nm in (False, True):
            assert taken(after, U, v, nm) == taken(text, U, U.as_ref(v), nm)


def test_another_macro_beside_the_widened_version_is_not_judged():
    line = "#if defined(VERSION_DE) && LIBGCC2_EXTERNAL_CLZ\n"
    assert widen.rewrite(line, U) is None
    other = "#if defined(VERSION_US_REV1) && LIBGCC2_EXTERNAL_CLZ\n"  # the reference only: the small edit is sound
    assert widen.rewrite(other, U) == "#if (defined(VERSION_US_REV1) || defined(VERSION_DE)) && LIBGCC2_EXTERNAL_CLZ\n"


def test_a_block_of_several_functions_brings_them_all():
    text = "#if defined(VERSION_US_REV1)\nvoid Foo(void) {}\nvoid Baz(void) {}\n#endif\n"
    _, decided, _ = widen.widen_text(text, {"Foo"}, U)
    assert decided == {"Foo", "Baz"}


def lui(reg: int, high: int) -> int:
    return 0x3C000000 | reg << 16 | high


def lw(rt: int, base: int, low: int) -> int:
    return 0x8C000000 | base << 21 | rt << 16 | (low & 0xFFFF)


def jal(target: int) -> int:
    return 0x0C000000 | (target >> 2) & 0x3FFFFFF


def test_counterpart_reads_the_address_the_other_version_builds_at_the_paired_words():
    ref = [lui(4, 0x8010), lw(4, 4, 0x0010), jal(0x80200000), 0]  # D_80100000 + 0x10 (a field), then a call
    other = [lui(4, 0x800F), lw(4, 4, -0x7FF0 & 0xFFFF), jal(0x801F0000), 0]
    assert widen.counterpart(0x80100000, 0x20, ref, 0x80300000, other, 0x80300000) == {0x800E8000}
    assert widen.counterpart(0x80200000, 4, ref, 0x80300000, other, 0x80300000) == {0x801F0000}


def test_derive_refuses_a_symbol_read_two_ways_and_one_no_word_builds():
    ref = {
        "Foo": (0x80300000, 0, 16, "func"),
        "D_a": (0x80100000, None, 4, "data"),
        "D_b": (0x80110000, None, 4, "data"),
    }
    other = {"Foo": (0x80300000, 0, 16, "func")}
    rimg = b"".join(w.to_bytes(4, "big") for w in [lui(4, 0x8010), lw(4, 4, 0), lui(5, 0x8010), lw(5, 5, 0)])
    oimg = b"".join(w.to_bytes(4, "big") for w in [lui(4, 0x800F), lw(4, 4, 0), lui(5, 0x800E), lw(5, 5, 0)])
    derived, refused = widen.derive({"Foo"}, {"D_a", "D_b", "Foo"}, (ref, rimg), (other, oimg))
    assert derived == {} and any(r.startswith("D_a: read as 0x800E0000, 0x800F0000") for r in refused)
    assert any(r.startswith("D_b: no ported word") for r in refused)
