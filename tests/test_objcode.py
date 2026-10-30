"""The object-code reader with synthetic MIPS words: sites come from the body, tiles equal it except at sites."""

import struct

import pytest

from unbake import objcode

SP, RA, A0, V0, AT, V1 = 29, 31, 4, 2, 1, 3


def lui(rt, imm):
    return 0x3C000000 | rt << 16 | imm


def addiu(rt, rs, imm):
    return 0x24000000 | rs << 21 | rt << 16 | imm & 0xFFFF


def lw(rt, off, rs):
    return 0x8C000000 | rs << 21 | rt << 16 | off & 0xFFFF


def sw(rt, off, rs):
    return 0xAC000000 | rs << 21 | rt << 16 | off & 0xFFFF


def jal(target):
    return 0x0C000000 | target


def beqz(rs, off):
    return 0x10000000 | rs << 21 | off & 0xFFFF


def addu(rd, rs, rt):
    return rs << 21 | rt << 16 | rd << 11 | 0x21


def body(hi=0x8015, lo=0x2208, call=0x100, other=0x80, frame=-32, size=24):
    """A function: a frame, an address pair, a table read by an index add, calls, a branch, filler, the return."""
    words = [addiu(SP, SP, frame), sw(RA, 28, SP), lui(AT, hi), addiu(A0, AT, lo), lui(V1, 0x800E), addu(V1, V1, V0)]
    words += [lw(V0, lo, V1), jal(call), 0, jal(other), 0, beqz(V0, 3), 0]
    words += [addiu(V0, V0, 7 + i) for i in range(size)]
    words += [lw(RA, 28, SP), 0x03E00008, addiu(SP, SP, -frame)]
    return words


def pack(words):
    return struct.pack(f">{len(words)}I", *words)


def corpus(*bodies):
    return objcode.Corpus(
        [objcode.Body(v, 0x1000 * (i + 1), tuple(w), tuple(objcode.sites(w)), (0,)) for i, (v, w) in enumerate(bodies)],
        [v for v, _ in bodies],
    )


def test_sites_follow_the_address_registers_of_the_body():
    words = body()
    flags = objcode.sites(words)
    assert flags[2] == objcode.IMMEDIATE and flags[3] == objcode.IMMEDIATE  # lui/addiu of the pair
    assert flags[4] == objcode.IMMEDIATE and flags[6] == objcode.IMMEDIATE  # the table read through addu
    assert flags[7] == flags[9] == objcode.TARGET
    assert flags[0] == flags[1] == flags[5] == flags[11] == 0  # frame, store through sp, the add, the branch
    constants = [lui(AT, 0x3F80), addiu(A0, AT, 0x1234), lw(V0, 4, AT)]
    assert objcode.sites(constants) == [0, 0, 0]  # a float constant is no address


def test_a_load_ends_the_address_in_its_register():
    words = [lui(AT, 0x8015), lw(AT, 8, AT), lw(V0, 4, AT)]
    assert objcode.sites(words) == [objcode.IMMEDIATE, objcode.IMMEDIATE, 0]


def test_a_forward_branch_hands_its_registers_to_the_target():
    words = [lui(AT, 0x8015), beqz(V0, 2), 0, addiu(AT, 0, 5), lw(V0, 4, AT)]
    assert objcode.sites(words)[4] == objcode.IMMEDIATE
    assert objcode.sites([lui(AT, 0x8015), 0x03E00008, 0, lw(V0, 4, AT)])[3] == 0  # after a jr the label is unreached


def test_agrees_allows_only_the_field_of_a_site():
    assert objcode.agrees(jal(5), jal(9), objcode.TARGET) and not objcode.agrees(jal(5), 0x08000009, objcode.TARGET)
    assert objcode.agrees(lui(AT, 0x8020), lui(AT, 0x8015), objcode.IMMEDIATE)
    assert objcode.agrees(lui(AT, 0), lui(AT, 0x8015), objcode.IMMEDIATE)  # not yet relocated
    assert not objcode.agrees(lui(AT, 0x3F80), lui(AT, 0x8015), objcode.IMMEDIATE)  # a constant in an address place
    assert not objcode.agrees(lui(V0, 0x8020), lui(AT, 0x8015), objcode.IMMEDIATE)  # another register
    assert objcode.agrees(lw(V0, 9, V1), lw(V0, 4, V1), objcode.IMMEDIATE) and not objcode.agrees(
        lw(V0, 9, V1), lw(V0, 4, V1), 0
    )


def test_tiles_equal_a_body_except_at_its_sites():
    known = body()
    window = body(hi=0x8020, lo=0x3000, call=0x222, other=0x333)  # another link of the same function
    tiles = corpus(("v", known)).tiles(pack(window), "v")
    assert [(t.start, t.end, t.at) for t in tiles] == [(0, 4 * len(window), 0)]
    receipt = corpus(("v", known)).prove(pack(window), "v", 0, 4 * len(window))
    (tile,) = receipt["tiles"]
    assert tile["sites"] == {"target": 2, "hi": 1, "lo": 2} and tile["words"] == len(window)
    assert [d[:2] for d in tile["differences"]][:3] == [[2, "hi"], [3, "lo"], [6, "lo"]]
    assert tile["body"] == {"version": "v", "rom": [0x1000, 0x1000 + 4 * len(window)]}


@pytest.mark.parametrize("change", [dict(frame=-40), dict(size=25)])
def test_a_difference_outside_a_site_is_no_tile(change):
    window = body(**change)
    with pytest.raises(objcode.ObjError, match="match no known body"):
        corpus(("v", body())).prove(pack(window), "v", 0, 4 * len(window))


def test_a_run_shorter_than_a_tile_proves_nothing():
    short = body(size=0)[:12]
    assert corpus(("v", body(size=0))).tiles(pack(short), "v") == []


def test_a_window_of_mostly_sites_proves_nothing():
    jumps = [w for i in range(12) for w in (jal(0x100 + i), 0)]
    other = [w for i in range(12) for w in (jal(0x900 + i), 0)]
    assert corpus(("v", jumps)).tiles(pack(other), "v") == []  # no word equal to its body's: nothing but sites


def test_code_sits_only_at_word_aligned_rom_offsets():
    window = body(hi=0x8020)
    cor = corpus(("v", body()))
    shifted = b"\x02\x10\x00" + pack(window) + b"\xff"  # the same words three bytes off the word grid
    assert cor.tiles(shifted, "v") == []
    assert cor.regions(shifted, "v", [(0, len(shifted))]) == []
    aligned = b"\x02\x10\x00\x00" + pack(window) + b"\xff"
    assert cor.regions(aligned, "v", [(0, len(aligned))]) == [
        {"rom": [4, 4 + 4 * len(window)], "tiles": 1, "bodies": ["v"]}
    ]
    # a span that starts off the grid reads on the ROM's grid, not its own
    assert cor.regions(aligned, "v", [(1, len(aligned))]) == [
        {"rom": [4, 4 + 4 * len(window)], "tiles": 1, "bodies": ["v"]}
    ]


def test_a_tie_goes_to_the_body_of_the_version_then_the_version_order():
    cor = corpus(("a", body()), ("b", body()))
    assert cor.bodies[cor.tiles(pack(body()), "b")[0].body].version == "b"
    assert cor.bodies[cor.tiles(pack(body()), "c")[0].body].version == "a"


def test_a_tile_may_cover_several_functions_of_a_body():
    first, second = body(), body(size=30)
    cor = objcode.Corpus(
        [
            objcode.Body(
                "v", 0, tuple(first + second), tuple(objcode.sites(first) + objcode.sites(second)), (0, len(first))
            )
        ],
        ["v"],
    )
    window = body(hi=0x8021) + body(hi=0x8021, size=30)
    (tile,) = cor.prove(pack(window), "v", 0, 4 * len(window))["tiles"]
    assert tile["functions"] == [0, 4 * len(first)]


def test_prove_refuses_a_range_that_is_not_whole_words():
    with pytest.raises(objcode.ObjError):
        corpus(("v", body())).prove(pack(body()), "v", 0, 5)
