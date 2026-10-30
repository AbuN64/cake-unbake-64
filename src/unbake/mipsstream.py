"""Code stream reader: proves that ROM bytes are compiled MIPS code by structure alone, with no body to compare to.

`objcode` proves code another known function explains. Stale code of an older link has no such function, so this reader
proves it the way a disassembler's author would: the bytes are a whole number of functions, and nothing in them is
anything but an instruction. A stream is the word range [start, end) of the ROM and every one of these holds:
- it starts at a word-aligned ROM offset (the compiler and the linker place code only on word boundaries, so a span
  that decodes only at a shifted word phase is data read out of step, never code);
- every word is a canonical VR4300 instruction of compiled code: opcode and function in the instruction set, every
  field the encoding fixes at zero is zero, no base register of a load or store is $zero or $k0/$k1/$gp (the one null
  store `sb $zero, 0($zero)` that an assertion plants is the exception) and no kernel or global pointer register is
  read or written;
- the word after a branch or jump (its delay slot) is no branch or jump;
- no run of more than MAX_NOPS zero words (alignment padding, never a data table);
- the first word is a function prologue, `addiu $sp, $sp, -N`, and the stream ends with a return, `jr $ra`, and its
  delay slot, so it is whole functions from a first entry to a last return;
- every conditional branch lands on a word of the stream, so no branch leaves it.
A stream is at least MIN_WORDS words. Random data fails these at the first return word (a 32-bit pattern), the first
prologue (24 fixed bits) and the first off-stream branch. The receipt lists each function entry, return, branch count
and call shape, so a reader can check them against the bytes. A row credits a region inside such a stream."""

from __future__ import annotations

import struct
from collections.abc import Iterable, Sequence
from typing import Any

from unbake.contracts import sha256

RETURN = 0x03E00008  # jr $ra
MIN_WORDS = 16  # the fewest words of one stream (the 64 B a copy needs)
WORD = 4  # code sits only at ROM offsets that are a multiple of this
PAGE = 0x1000  # the write size that cuts a stream: a stream ending on a page boundary may lack its return
MAX_NOPS = 16  # the most zero words in a row inside a stream
NULL_STORES = frozenset({0xA0000000, 0xAC000000})  # sb/sw $zero, 0($zero): the crash an assertion plants
_KERNEL = frozenset({26, 27, 28})  # $k0 $k1 $gp
_SHIFTS = frozenset({0x00, 0x02, 0x03, 0x38, 0x3A, 0x3B, 0x3C, 0x3E, 0x3F})  # rs is zero
_NO_SHAMT = frozenset(
    {0x04, 0x06, 0x07, 0x14, 0x16, 0x17, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27}
    | {0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F}
)  # register arithmetic: shift amount zero
_MULDIV = frozenset({0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F})  # rd and shift amount zero
_FROM_HILO = frozenset({0x10, 0x12})  # rs, rt, shift amount zero
_TO_HILO = frozenset({0x11, 0x13})  # rt, rd, shift amount zero
_PLAIN = frozenset({8, 9, 10, 11, 12, 13, 14, 0x18, 0x19, 4, 5, 0x14, 0x15})  # no field rule beyond registers
_ZERO_RT = frozenset({6, 7, 0x16, 0x17})  # blez bgtz and their likelies: rt is zero
_MEMORY = frozenset(
    {0x1A, 0x1B, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E}
    | {0x30, 0x31, 0x34, 0x35, 0x37, 0x38, 0x39, 0x3C, 0x3D, 0x3F}
)  # loads and stores
_FLOAT_MEMORY = frozenset({0x31, 0x35, 0x39, 0x3D})  # the data register is a float register
_REGIMM = frozenset({0, 1, 2, 3, 0x10, 0x11, 0x12, 0x13})
_FLOAT_OPS = frozenset(set(range(16)) | {0x20, 0x21, 0x24, 0x25} | set(range(0x30, 0x40)))
_BRANCHES = frozenset({4, 5, 6, 7, 0x14, 0x15, 0x16, 0x17})


class StreamError(ValueError):
    """Words that are no whole compiled functions."""


def unpack(data: bytes | memoryview) -> tuple[int, ...]:
    if len(data) % 4:
        raise StreamError("a stream is a whole number of words")
    return struct.unpack(f">{len(data) // 4}I", data)


def canonical(w: int) -> bool:
    """The word is an instruction compiled code can hold: encoding fields, registers and the zero fields all fixed."""
    op, rs, rt, rd, sh, fn = w >> 26, (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31, (w >> 6) & 31, w & 63
    if op == 0:
        if {rs, rt, rd} & _KERNEL and fn not in (0x0C, 0x0D):
            return False
        if fn in _SHIFTS:
            return rs == 0
        if fn in _NO_SHAMT:
            return sh == 0
        if fn == 8:
            return rt == 0 and rd == 0 and sh == 0
        if fn == 9:
            return rt == 0 and sh == 0
        if fn in _FROM_HILO:
            return rs == 0 and rt == 0 and sh == 0
        if fn in _TO_HILO:
            return rt == 0 and rd == 0 and sh == 0
        if fn in _MULDIV:
            return rd == 0 and sh == 0
        return fn in (0x0C, 0x0D)  # syscall, break (an assertion's code is any value)
    if op == 1:
        return rt in _REGIMM and rs not in _KERNEL
    if op == 15:
        return rs == 0 and rt not in _KERNEL
    if op in (2, 3):
        return True  # j, jal: the 26-bit target
    if op in _PLAIN:
        return not ({rs, rt} & _KERNEL)
    if op in _ZERO_RT:
        return rt == 0 and rs not in _KERNEL
    if op in _MEMORY:
        if w in NULL_STORES:
            return True
        return rs != 0 and rs not in _KERNEL and (op in _FLOAT_MEMORY or rt not in _KERNEL)
    if op == 0x11:
        if rs in (0, 1, 2, 4, 5, 6):
            return w & 0x7FF == 0 and rt not in _KERNEL
        if rs == 8:
            return rt in (0, 1, 2, 3)
        if rs in (0x10, 0x11):
            return fn in _FLOAT_OPS and (fn != 4 or rt == 0)
        if rs in (0x14, 0x15):
            return fn in (0x20, 0x21) and rt == 0
    return False


def transfers(w: int) -> bool:
    """A branch or jump: the word after it is a delay slot."""
    op = w >> 26
    return (
        op in (2, 3)
        or op in _BRANCHES
        or (op == 0 and w & 63 in (8, 9))
        or (op == 1 and (w >> 16) & 31 in _REGIMM)
        or (op == 0x11 and (w >> 21) & 31 == 8)
    )


def conditional(w: int) -> bool:
    """A branch with an offset (not a jump through a register or an absolute jump)."""
    return transfers(w) and w >> 26 not in (0, 2, 3)


def offset(w: int) -> int:
    value = w & 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def prologue(w: int) -> bool:
    """`addiu $sp, $sp, -N`: the frame a function opens with."""
    return w >> 16 == 0x27BD and w & 0x8000 != 0 and w & 0xFFFF != 0


def usable(words: Sequence[int]) -> list[bool]:
    """Per word: it is canonical, it is no transfer in a delay slot and it is no part of a zero run over MAX_NOPS."""
    ok = [canonical(w) for w in words]
    zeros = 0
    for i, w in enumerate(words):
        if i and ok[i - 1] and transfers(words[i - 1]) and transfers(w):
            ok[i] = False
        zeros = zeros + 1 if w == 0 else 0
        if zeros > MAX_NOPS:
            ok[i] = False
    return ok


def runs(ok: Sequence[bool]) -> list[tuple[int, int]]:
    """The maximal runs of usable words."""
    out, at = [], -1
    for i, good in enumerate([*ok, False]):
        if good and at < 0:
            at = i
        elif not good and at >= 0:
            out.append((at, i))
            at = -1
    return out


def frame(words: Sequence[int], low: int, high: int) -> tuple[int, int] | None:
    """The first prologue to the last return (with its delay slot) inside words[low:high], or None."""
    first = next((i for i in range(low, high) if prologue(words[i])), None)
    if first is None:
        return None
    last = next((i for i in range(high - 2, first - 1, -1) if words[i] == RETURN), None)
    return None if last is None else (first, last + 2)


def pieces(words: Sequence[int], low: int = 0, high: int | None = None) -> list[tuple[int, int]]:
    """The streams inside words[low:high], which must be a usable run: the frame from a first prologue to a last
    return, cut at every conditional branch that leaves it and tried again on each side."""
    work = [(low, len(words) if high is None else high)]
    out: list[tuple[int, int]] = []
    while work:
        a, b = work.pop()
        if b - a < MIN_WORDS:
            continue
        found = frame(words, a, b)
        if found is None or found[1] - found[0] < MIN_WORDS:
            continue
        h, t = found
        leave = next((i for i in range(h, t) if conditional(words[i]) and not h <= i + 1 + offset(words[i]) < t), None)
        if leave is None:
            out.append((h, t))
        else:
            work += [(a, leave), (leave + 1, b)]
    return sorted(out)


def phases(low: int, shifted: bool = False) -> list[int]:
    """The byte phases of a read that starts at ROM offset `low`: the one that lands on a word-aligned ROM offset, or
    (shifted) the three that do not, which only a false candidate decodes at."""
    aligned = -low % WORD
    return [p for p in range(WORD) if (p != aligned) == shifted]


def find(rom: bytes | memoryview, low: int, high: int, shifted: bool = False) -> list[tuple[int, int]]:
    """Every stream (ROM byte range) inside rom[low:high] that starts at a word-aligned ROM offset; with `shifted`, the
    streams the three other phases give instead: not code, listed only to name what the aligned read refuses."""
    found: list[tuple[int, int]] = []
    for phase in phases(low, shifted):
        count = (high - low - phase) // 4
        if count < MIN_WORDS:
            continue
        words = unpack(bytes(rom[low + phase : low + phase + 4 * count]))
        for a, b in runs(usable(words)):
            whole = pieces(words, a, b)
            found += [(low + phase + 4 * s, low + phase + 4 * e) for s, e in whole]
            first = next((i for i in range(max([a, *(e for _, e in whole)]), b) if prologue(words[i])), None)
            if first is None:
                continue
            for stop in range(b, first + MIN_WORDS - 1, -1):  # the last page boundary in the run that leaves a stream
                if (low + phase + 4 * stop) % PAGE:
                    continue
                try:
                    check(words[first:stop], True)
                except StreamError:
                    continue
                found.append((low + phase + 4 * first, low + phase + 4 * stop))
                break
    return sorted(found)


def shape(words: Sequence[int], at: int) -> dict[str, Any]:
    """What the stream holds, for the receipt: ROM offsets of the entries and returns, branch and call counts."""
    entries = [at + 4 * i for i, w in enumerate(words) if prologue(w) and (i == 0 or _after_return(words, i))]
    returns = [at + 4 * i for i, w in enumerate(words) if w == RETURN]
    branches = [i for i, w in enumerate(words) if conditional(w)]
    calls = [w & 0x3FFFFFF for w in words if w >> 26 == 3]
    return {
        "functions": entries,
        "returns": returns,
        "branches": len(branches),
        "calls": len(calls),
        "jumps": sum(1 for w in words if w >> 26 == 2),
        "call_targets": len(set(calls)),
    }


def _after_return(words: Sequence[int], i: int) -> bool:
    """Word i follows a return and its delay slot, with only zero padding between."""
    k = i - 1
    while k >= 0 and words[k] == 0:
        k -= 1
    return k >= 0 and (words[k] == RETURN or (k >= 1 and words[k - 1] == RETURN))


def check(words: Sequence[int], cut: bool = False) -> None:
    """Raise StreamError naming the first thing that makes `words` no stream. A `cut` stream ends where a page write
    cut it: no return or padding at its end, and a branch may land after the end, never before it starts."""
    if len(words) < MIN_WORDS:
        raise StreamError(f"{len(words)} words are fewer than the {MIN_WORDS} of a stream")
    ok = usable(words)
    if not all(ok):
        i = ok.index(False)
        raise StreamError(f"word {i} ({words[i]:#010x}) is no canonical instruction in compiled code")
    if not prologue(words[0]):
        raise StreamError(f"the first word {words[0]:#010x} is no frame prologue (addiu $sp, $sp, -N)")
    if not cut and words[-2] != RETURN:
        raise StreamError("the stream does not end with a return and its delay slot")
    if cut and words[-1] == 0:
        raise StreamError("a cut stream ends on code, not on padding")
    for i, w in enumerate(words):
        target = i + 1 + offset(w)
        if conditional(w) and not (target >= 0 if cut else 0 <= target < len(words)):
            raise StreamError(f"the branch at word {i} leaves the stream")


def prove(rom: bytes | memoryview, start: int, end: int, stream: Sequence[int]) -> dict[str, Any]:
    """The receipt of the region rom[start:end] as part of the stream rom[stream[0]:stream[1]]."""
    first, last = stream
    if not 0 <= first < last <= len(rom) or (last - first) % 4:
        raise StreamError("a stream is a whole number of words inside the ROM")
    if first % WORD:
        raise StreamError("not code: the stream starts at a shifted word phase, code sits only on word boundaries")
    if not first <= start < end <= last:
        raise StreamError("the region is not inside its stream")
    data = bytes(rom[first:last])
    words = unpack(data)
    cut = last % PAGE == 0 and (len(words) < 2 or words[-2] != RETURN)
    check(words, cut)
    return {
        "cut": cut,
        "stream": [first, last],
        "phase": first % 4,
        "words": len(words),
        "stream_sha256": sha256(data),
        **shape(words, first),
        "region": [start, end],
        "region_sha256": sha256(bytes(rom[start:end])),
    }


def spans(found: Iterable[tuple[int, int]], targets: Iterable[tuple[int, int]]) -> list[dict[str, Any]]:
    """The rows (region inside targets, stream) that streams give."""
    rows = []
    for low, high in targets:
        for s, e in found:
            a, b = max(low, s), min(high, e)
            if a < b:
                rows.append({"region": [a, b], "stream": [s, e]})
    return sorted(rows, key=lambda r: r["region"])
