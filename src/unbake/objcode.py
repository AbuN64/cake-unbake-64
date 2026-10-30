"""Object-code reader: proves that ROM bytes are MIPS code a known body already explains, every difference a
relocation site, never a percentage.

A window of words is cut into tiles. A tile is a run of words of the window that equals a run of one known body (the
functions of any version, read from the project's own symbols) word for word, except at relocation sites of that body:
- a jump target (`j`, `jal`: the 26-bit field);
- the high half of an address (a `lui` whose immediate lies in the RAM address window 0x8000..0x807F in the body), where
  the window holds another address of that window, or 0 (an object not yet relocated);
- the low half of an address (an immediate-bearing instruction, load or store whose base register carries a `lui` of
  that window, possibly through an `addu`/`or` index add): any 16 bits.
Everything else (the opcode, the registers, a frame size, a struct offset, a branch offset, a float constant) must be
equal. The sites come from decoding the body, never from the window, so a window that differs from the body anywhere
else is no tile. A tile holds at least MIN_WORDS words of which MIN_LIVE are not zero and equal the body's. A range
proves when whole tiles cover it exactly; the receipt lists each tile's body and every site where the two differ."""

from __future__ import annotations

import struct
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unbake.contracts import Config, sha256

ADDRESS = range(0x8000, 0x8080)  # the high half of an address in N64 RAM
MIN_WORDS = 16  # the fewest words of one tile (the 64 B a copy needs)
MIN_LIVE = 8  # the fewest words of one tile that are not zero and equal their body's word (no site)
SEED = 6  # consecutive words that agree (relocation fields blanked) before a tile is tried
TARGET, IMMEDIATE = 1, 2  # what a site may change in a body word
_BRANCHES = frozenset({1, 4, 5, 6, 7, 20, 21, 22, 23})
_LOADS = frozenset({0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x30, 0x34, 0x37})  # memory ops that write rt
_NO_WRITE = frozenset({8, 12, 13, 0x11, 0x13, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F})  # R-type, no rd
_BASES = (2, 3)  # coprocessor moves from a register file write rt
_END = frozenset({8})  # jr


class ObjError(ValueError):
    """Words that no known body explains."""


@dataclass(frozen=True)
class Body:
    """Contiguous code of one version: its words from ROM offset `rom`, the sites of each word and the word indexes
    where a function starts."""

    version: str
    rom: int
    words: tuple[int, ...]
    flags: tuple[int, ...]
    starts: tuple[int, ...]


@dataclass(frozen=True)
class Tile:
    start: int  # first byte, relative to the data it was cut from
    end: int
    body: int  # index into Corpus.bodies
    at: int  # the body word that equals the word at `start`


def unpack(data: bytes | memoryview) -> tuple[int, ...]:
    """The big-endian words of `data`, a whole number of words."""
    if len(data) % 4:
        raise ObjError("code is a whole number of words")
    return struct.unpack(f">{len(data) // 4}I", data)


def seed_key(word: int) -> int:
    """A word with its relocation fields blanked, for finding bodies; the proof is `agrees`, never this."""
    op = word >> 26
    if op in (2, 3):
        return op << 26
    return word if op == 0 else word & 0xFFFF0000


def sites(words: Sequence[int]) -> list[int]:
    """Per word of one function body: TARGET for a jump, IMMEDIATE for the immediate of an address half, else 0. A
    register carries an address once a `lui` of the address window wrote it (or an add of such a register); a load, any
    other write or a `jr` clears it, and a forward branch hands its state to its target."""
    flags = [0] * len(words)
    taint = 0  # bit mask of registers that carry an address
    due: defaultdict[int, int] = defaultdict(int)  # branch target -> registers owed to it
    after: defaultdict[int, list[int]] = defaultdict(list)  # delay slot -> branch targets
    ends: set[int] = set()  # delay slots of a jr
    for i, w in enumerate(words):
        op, rs, rt, rd = w >> 26, (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31
        taint |= due.pop(i, 0)
        if op in (2, 3):
            flags[i] = TARGET
        elif op == 15:
            if (w & 0xFFFF) in ADDRESS:
                flags[i], taint = IMMEDIATE, taint | 1 << rt
            else:
                taint &= ~(1 << rt)
        elif op in (8, 9, 13):
            if taint >> rs & 1:
                flags[i], taint = IMMEDIATE, taint | 1 << rt
            else:
                taint &= ~(1 << rt)
        elif op >= 0x20:
            if taint >> rs & 1:
                flags[i] = IMMEDIATE
            if op in _LOADS:
                taint &= ~(1 << rt)
        elif op == 0:
            func = w & 63
            if func in (0x20, 0x21, 0x25):
                taint = taint | 1 << rd if taint >> rs & 1 or taint >> rt & 1 else taint & ~(1 << rd)
            elif func not in _NO_WRITE:
                taint &= ~(1 << rd)
            if func in _END:
                ends.add(i + 1)
        elif op in (10, 11, 12, 14) or (op in (16, 17, 18) and rs in (0, *_BASES)):
            taint &= ~(1 << rt)
        if op in _BRANCHES:
            target = i + 1 + struct.unpack(">h", struct.pack(">H", w & 0xFFFF))[0]
            if target > i + 1:
                after[i + 1].append(target)
        for target in after.pop(i, ()):
            due[target] |= taint
        if i in ends:
            taint = 0
    return flags


def agrees(window: int, body: int, flag: int) -> bool:
    """The window word equals the body word, or differs only in the field its site may change."""
    if window == body:
        return True
    if flag == TARGET:
        return window >> 26 == body >> 26
    if flag == IMMEDIATE:
        return window >> 16 == body >> 16 and (body >> 26 != 15 or (window & 0xFFFF) in ADDRESS or not window & 0xFFFF)
    return False


def differences(window: Sequence[int], body: Sequence[int], flags: Sequence[int]) -> list[list[Any]]:
    """The sites where a tile differs from its body: [word offset, kind, window field, body field]."""
    out: list[list[Any]] = []
    for i, (a, b) in enumerate(zip(window, body, strict=True)):
        if a != b:
            if flags[i] == TARGET:
                out.append([i, "target", a & 0x3FFFFFF, b & 0x3FFFFFF])
            else:
                out.append([i, "hi" if b >> 26 == 15 else "lo", a & 0xFFFF, b & 0xFFFF])
    return out


class Corpus:
    """The known bodies and their index by seed."""

    def __init__(self, bodies: Iterable[Body], order: Sequence[str] = ()) -> None:
        self.bodies = list(bodies)
        self.order = {v: i for i, v in enumerate(order)}
        self.index: defaultdict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
        for n, body in enumerate(self.bodies):
            keys = [seed_key(w) for w in body.words]
            for i in range(len(keys) - SEED + 1):
                self.index[tuple(keys[i : i + SEED])].append((n, i))

    def runs(self, window: Sequence[int]) -> list[tuple[int, int, int, int]]:
        """Every maximal run of `window` that a body explains, as (first word, end word, body, body word at first),
        a run found from each place its seed meets a body."""
        keys = [seed_key(w) for w in window]
        found: list[tuple[int, int, int, int]] = []
        reach: dict[tuple[int, int], int] = {}  # (body, diagonal) -> the window word its last run reached
        for p in range(len(window) - SEED + 1):
            for n, i in self.index.get(tuple(keys[p : p + SEED]), ()):
                if reach.get((n, i - p), -1) > p:
                    continue
                body = self.bodies[n]
                back = 0
                while (
                    p - back > 0
                    and i - back > 0
                    and agrees(window[p - back - 1], body.words[i - back - 1], body.flags[i - back - 1])
                ):
                    back += 1
                fwd = 0
                while (
                    p + fwd < len(window)
                    and i + fwd < len(body.words)
                    and agrees(window[p + fwd], body.words[i + fwd], body.flags[i + fwd])
                ):
                    fwd += 1
                if fwd >= SEED:
                    reach[(n, i - p)] = p + fwd
                    found.append((p - back, p + fwd, n, i - back))
        return found

    def tiles(self, data: bytes | memoryview, version: str, phases: Sequence[int] = (0,)) -> list[Tile]:
        """The tiles of `data` (a ROM of `version`, read as words from each byte phase of `phases`): every run of at
        least MIN_WORDS words (MIN_LIVE of them not zero and equal) that a body explains. A longer run keeps its words;
        a tie goes to a body of `version`, then to the project's version order, then to the lowest ROM offset."""
        runs: list[tuple[int, int, int, int]] = []  # in bytes
        for phase in phases:
            count = (len(data) - phase) // 4
            if count >= SEED:
                for s, e, n, at in self.runs(unpack(bytes(data[phase : phase + 4 * count]))):
                    runs.append((phase + 4 * s, phase + 4 * e, n, at))

        def rank(run: tuple[int, int, int, int]) -> tuple[int, bool, int, int, int, int]:
            body = self.bodies[run[2]]
            order = self.order.get(body.version, len(self.order))
            return (run[0] - run[1], body.version != version, order, body.rom, run[3], run[0])

        taken = bytearray(len(data))
        out: list[Tile] = []
        for start, end, n, at in sorted(runs, key=rank):
            lo = start
            while lo < end:
                if any(taken[lo : lo + 4]):
                    lo += 4
                    continue
                hi = lo
                while hi < end and not any(taken[hi : hi + 4]):
                    hi += 4
                words = self.bodies[n].words
                live = 0
                for k in range(lo, hi, 4):
                    word = int.from_bytes(data[k : k + 4], "big")
                    live += word != 0 and word == words[at + (k - start) // 4]
                if hi - lo >= 4 * MIN_WORDS and live >= MIN_LIVE:
                    out.append(Tile(lo, hi, n, at + (lo - start) // 4))
                    taken[lo:hi] = b"\1" * (hi - lo)
                lo = hi
        return sorted(out, key=lambda t: t.start)

    def receipt(self, data: bytes | memoryview, base: int, tile: Tile) -> dict[str, Any]:
        """The proof of one tile of `data`, which starts at ROM offset `base`: where it is, the body it equals and
        every site where they differ."""
        body = self.bodies[tile.body]
        n = (tile.end - tile.start) // 4
        raw = bytes(data[tile.start : tile.end])
        theirs = body.words[tile.at : tile.at + n]
        diff = differences(unpack(raw), theirs, body.flags[tile.at : tile.at + n])
        return {
            "rom": [base + tile.start, base + tile.end],
            "words": n,
            "body": {"version": body.version, "rom": [body.rom + 4 * tile.at, body.rom + 4 * (tile.at + n)]},
            "functions": [body.rom + 4 * s for s in body.starts if tile.at <= s < tile.at + n],
            "sites": {k: sum(1 for d in diff if d[1] == k) for k in ("target", "hi", "lo")},
            "differences": diff,
            "sha256": sha256(raw),
            "body_sha256": sha256(struct.pack(f">{n}I", *theirs)),
        }

    def prove(self, rom: bytes | memoryview, version: str, start: int, end: int) -> dict[str, Any]:
        """The receipt of `rom[start:end]`: whole tiles from its first word to its last. ObjError names the first
        words no tile holds."""
        if (end - start) % 4 or not 0 <= start < end <= len(rom):
            raise ObjError("object code is a whole number of words inside the ROM")
        data = bytes(rom[start:end])
        tiles = self.tiles(data, version)
        at = 0
        for tile in tiles:
            if tile.start != at:
                break
            at = tile.end
        if at != len(data):
            stop = min((t.start for t in tiles if t.start > at), default=len(data))
            live = sum(1 for k in range(at, stop - 3, 4) if data[k : k + 4] != bytes(4))
            raise ObjError(
                f"{(stop - at) // 4} words ({live} not zero) from {start + at:#x} match no known body in a run of "
                f"{MIN_WORDS} words with {MIN_LIVE} not zero and equal"
            )
        return {
            "start": start,
            "end": end,
            "bytes": end - start,
            "sha256": sha256(data),
            "tiles": [self.receipt(data, start, tile) for tile in tiles],
        }

    def regions(self, rom: bytes | memoryview, version: str, spans: Iterable[tuple[int, int]]) -> list[dict[str, Any]]:
        """The ranges inside `spans` that whole tiles cover, found only at word-aligned ROM offsets (code sits only on
        word boundaries; a tile at a shifted phase is a false match); tiles that touch are one region."""
        out: list[dict[str, Any]] = []
        for low, high in spans:
            if high - low < 4 * MIN_WORDS:
                continue
            run: list[Tile] = []
            for tile in [*self.tiles(rom[low:high], version, (-low % 4,)), None]:
                if run and (tile is None or tile.start != run[-1].end):
                    out.append(
                        {
                            "rom": [low + run[0].start, low + run[-1].end],
                            "tiles": len(run),
                            "bodies": sorted({self.bodies[t.body].version for t in run}),
                        }
                    )
                    run = []
                if tile is not None:
                    run.append(tile)
        return out


def corpus(config: Config, versions: Sequence[str] | None = None, roms: Mapping[str, bytes] | None = None) -> Corpus:
    """The bodies of a project: per version, the functions its symbols place in the ROM, each up to the next function
    (or 8 KiB), joined where they touch."""
    from unbake import symbols

    names = list(versions or config.project.version)
    bodies: list[Body] = []
    for v in names:
        if roms is not None:
            rom = roms.get(v)
        else:
            path = Path(config.project.root) / "roms" / f"baserom.{v}.z64"
            rom = path.read_bytes() if path.is_file() else None
        if rom is None:
            continue
        starts = sorted({s[1] for s in symbols.load(config, v).values() if s[3] == "func" and s[1] is not None})
        extents = []
        for i, a in enumerate(starts):
            b = starts[i + 1] if i + 1 < len(starts) else a + 0x2000
            b = min(b if b - a <= 0x20000 else a + 0x2000, len(rom))
            b -= (b - a) % 4
            if a % 4 == 0 and b > a:
                extents.append((a, b))
        run: list[tuple[int, int]] = []
        for extent in [*extents, None]:
            if run and (extent is None or extent[0] != run[-1][1]):
                words: list[int] = []
                flags: list[int] = []
                firsts: list[int] = []
                for a, b in run:
                    part = unpack(rom[a:b])
                    firsts.append(len(words))
                    words += part
                    flags += sites(part)
                bodies.append(Body(v, run[0][0], tuple(words), tuple(flags), tuple(firsts)))
                run = []
            if extent is not None:
                run.append(extent)
    return Corpus(bodies, names)
