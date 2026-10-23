"""The code stream reader with synthetic MIPS words: whole canonical functions prove, nothing else does."""

import random
import struct

import pytest

from unbake import mipsstream

SP, RA, A0, V0, V1, AT, S0, K0 = 29, 31, 4, 2, 3, 1, 16, 26
RETURN = 0x03E00008


def addiu(rt, rs, imm):
    return 0x24000000 | rs << 21 | rt << 16 | imm & 0xFFFF


def lw(rt, off, rs):
    return 0x8C000000 | rs << 21 | rt << 16 | off & 0xFFFF


def sw(rt, off, rs):
    return 0xAC000000 | rs << 21 | rt << 16 | off & 0xFFFF


def lui(rt, imm):
    return 0x3C000000 | rt << 16 | imm


def beqz(rs, off):
    return 0x10000000 | rs << 21 | off & 0xFFFF


def addu(rd, rs, rt):
    return rs << 21 | rt << 16 | rd << 11 | 0x21


def jal(target):
    return 0x0C000000 | target


def function(size=10, frame=32, branch=True):
    """A framed function: prologue, saves, a forward branch over some work, restores and the return."""
    words = [addiu(SP, SP, -frame), sw(RA, 28, SP), sw(S0, 24, SP), addu(S0, A0, 0), lui(AT, 0x800E)]
    if branch:
        words += [beqz(S0, size + 2), 0]
    words += [addiu(V0, V0, i + 1) for i in range(size)]
    words += [jal(0x100), 0, lw(RA, 28, SP), lw(S0, 24, SP), RETURN, addiu(SP, SP, frame)]
    return words


def pack(words):
    return struct.pack(f">{len(words)}I", *words)


def rom_with(words, at, size=0x1000, filler=None):
    out = bytearray(filler if filler is not None else random.Random(1).randbytes(size))
    out[at : at + 4 * len(words)] = pack(words)
    return bytes(out)


def test_canonical_words_of_compiled_code_pass_and_odd_ones_fail():
    assert all(mipsstream.canonical(w) for w in function())
    assert not mipsstream.canonical(0x40085000)  # mfc0: no coprocessor 0 in game code
    assert not mipsstream.canonical(addu(K0, 4, 5))  # $k0 is the kernel's
    assert not mipsstream.canonical(lw(V0, 8, 0))  # a load through $zero is a data table's pointer word
    assert mipsstream.canonical(0xA0000000)  # the null store an assertion plants
    assert not mipsstream.canonical(0x0000_0001 | 5 << 21)  # a shift with a register in the zero field
    assert mipsstream.canonical(0x0C0AAA50)  # jal: its 26-bit target is not register fields


def test_a_branch_or_jump_has_no_branch_or_jump_in_its_delay_slot():
    words = function()
    ok = mipsstream.usable(words)
    assert all(ok)
    broken = [*words[:5], beqz(S0, 3), beqz(S0, 2), *words[6:]]
    assert not mipsstream.usable(broken)[6]


def test_a_zero_run_over_the_limit_is_no_part_of_a_stream():
    words = function()
    ok = mipsstream.usable([*words, *[0] * 20, *words])
    assert not all(ok) and ok[len(words)] and not ok[len(words) + 17]


def test_a_framed_function_is_a_stream_and_every_other_cut_of_it_is_not():
    words = function()
    mipsstream.check(words)
    with pytest.raises(mipsstream.StreamError, match="prologue"):
        mipsstream.check(words[1:])
    with pytest.raises(mipsstream.StreamError, match="return"):
        mipsstream.check(words[:-2])
    with pytest.raises(mipsstream.StreamError, match="canonical"):
        mipsstream.check([*words[:8], 0x40085000, *words[9:]])
    with pytest.raises(mipsstream.StreamError, match="fewer"):
        mipsstream.check(words[:8])


def test_a_branch_that_leaves_the_stream_cuts_it():
    words = function()
    away = [*words[:5], beqz(S0, 400), 0, *words[7:]]
    with pytest.raises(mipsstream.StreamError, match="leaves"):
        mipsstream.check(away)
    assert mipsstream.pieces(away) == []  # cut at the branch: neither side holds a frame and a return of 16 words


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_find_reads_a_stream_at_any_byte_phase_and_trims_the_words_either_side(phase):
    words = function(size=14) + function(size=20, frame=48)
    at = 0x200 + phase
    neighbours = [lw(V0, 8, 0)] * 3  # canonical-looking words of no function (a load through $zero)
    data = rom_with([*neighbours, *words, *neighbours], at - 12, filler=b"\x40" * 0x400)
    (stream,) = mipsstream.find(data, 0, len(data))
    assert stream == (at, at + 4 * len(words))


def test_random_bytes_hold_no_stream():
    noise = random.Random(5).randbytes(1 << 18)
    assert mipsstream.find(noise, 0, len(noise)) == []


def test_prove_lists_the_functions_returns_and_calls_and_pins_the_bytes():
    words = [*function(size=14), 0, 0, *function(size=20, frame=48)]
    data = rom_with(words, 0x200)
    receipt = mipsstream.prove(data, 0x210, 0x2C0, (0x200, 0x200 + 4 * len(words)))
    assert receipt["stream"] == [0x200, 0x200 + 4 * len(words)] and receipt["phase"] == 0
    assert receipt["functions"] == [0x200, 0x200 + 4 * (len(function(size=14)) + 2)]
    assert len(receipt["returns"]) == 2 and receipt["calls"] == 2 and receipt["branches"] == 2
    assert receipt["region"] == [0x210, 0x2C0]


def test_prove_refuses_a_region_outside_its_stream_and_bytes_that_are_no_stream():
    words = function()
    data = rom_with(words, 0x200)
    with pytest.raises(mipsstream.StreamError, match="inside its stream"):
        mipsstream.prove(data, 0x1F0, 0x210, (0x200, 0x200 + 4 * len(words)))
    with pytest.raises(mipsstream.StreamError, match=r"canonical|prologue|return|leaves"):
        mipsstream.prove(data, 0x208, 0x230, (0x204, 0x204 + 4 * len(words)))


def test_spans_clip_a_stream_to_the_unknown_bytes():
    rows = mipsstream.spans([(0x100, 0x200)], [(0x80, 0x120), (0x180, 0x400)])
    assert rows == [
        {"region": [0x100, 0x120], "stream": [0x100, 0x200]},
        {"region": [0x180, 0x200], "stream": [0x100, 0x200]},
    ]
