"""Earlier-build reader: proves that stale ROM bytes are code the compiler makes from the current source, byte for byte,
never a percentage.

`objcode` proves code a known body explains except at relocation sites; `mipsstream` proves bytes are whole canonical
functions without saying whose. This reader goes from the second to the first by the compiler:
- disassemble: the usable runs of a span on the mipsstream word grid (word-aligned ROM offsets only) are cut into pieces
  after each return and its delay slot; a piece opens with a prologue or mid-function (`head`) and ends with a return
  or where the span cut it (`tail`);
- skeleton: a piece reduces to its tokens, one per word: the opcode class and, for a branch, its offset; registers,
  immediates, shift amounts and jump targets are free. The control-flow edges, call targets, data references (a `lui`
  and the load, store or `addiu` that completes the address) and the class counts are listed beside them;
- match: every compiled function of any version that holds the piece's tokens in a row is a candidate; the chosen one
  is the first in the order: a function of the version itself, then the project's version order, then the lowest ROM
  offset, and every candidate is listed;
- rebuild: the compiler's own output for the candidate (the function linked from the current source, `Compiled`) is
  compared with the ROM words. The piece closes only when every word is equal; any word that differs is listed with
  the stale and the rebuilt instruction, and nothing closes. No source variant is invented: a difference with no
  variant that explains it leaves the bytes unknown.

A row credits a region inside a piece the compiler reproduced. The receipt holds the matched function, its source, the
rebuilt hash, the data the piece references and the list of differing instructions with the source change that
explains each (none, when the bytes are equal)."""

from __future__ import annotations

import struct
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unbake import mipsstream
from unbake.contracts import Json, sha256

MIN_WORDS = mipsstream.MIN_WORDS  # the fewest words of a piece
WINDOW = 8  # tokens hashed at every position of a compiled function; a piece holds one
ATTEMPTS = 8  # the most candidates rebuilt for one piece
SHOWN = 16  # the most candidates and differing words listed for a refused piece
RAM = (0x80000000, 0x80800000)  # the address window of data references and call targets
_LOADS = {0x20: 1, 0x24: 1, 0x21: 2, 0x25: 2, 0x23: 4, 0x27: 4, 0x31: 4, 0x37: 8, 0x35: 8, 0x22: 4, 0x26: 4}
_STORES = {0x28: 1, 0x29: 2, 0x2B: 4, 0x39: 4, 0x3F: 8, 0x3D: 8, 0x2A: 4, 0x2E: 4}
_ADDRESS_OPS = frozenset({9, 13})  # addiu, ori: an address taken
_BRANCHES = frozenset({4, 5, 6, 7, 0x14, 0x15, 0x16, 0x17})  # beq, bne, blez, bgtz and their likely forms


class EarlierBuildError(ValueError):
    """Bytes the compiler does not reproduce from the matched source."""


@dataclass(frozen=True)
class Compiled:
    """A function of one version as the compiler made it from the current source: its words as linked, the file that
    defines it and the linked image they came from."""

    version: str
    name: str
    vram: int
    rom: int | None
    words: tuple[int, ...]
    source: str
    image: str
    image_sha256: str


def token(w: int) -> int:
    """The skeleton token of a word: its opcode class, plus the offset of a branch."""
    op = w >> 26
    if w == 0:
        return -1
    if op == 0:
        return w & 63
    if op == 1:
        rt = (w >> 16) & 31
        return 0x100 + rt | ((w & 0xFFFF) << 16 if rt in (0, 1, 2, 3, 0x10, 0x11, 0x12, 0x13) else 0)
    if op in (0x10, 0x11):
        rs = (w >> 21) & 31
        if op == 0x11 and rs == 8:  # bc1
            return 0x1000 + ((w >> 16) & 3) | ((w & 0xFFFF) << 16)
        return (0x200 if op == 0x10 else 0x300) + rs * 64 + (w & 63 if rs >= 16 else 0)
    if op in _BRANCHES:
        return 0x400 + op | ((w & 0xFFFF) << 16)
    return 0x500 + op


def tokens(words: Sequence[int]) -> tuple[int, ...]:
    return tuple(token(w) for w in words)


def _kind(w: int) -> str:
    op = w >> 26
    if w == 0:
        return "nop"
    if op == 3 or (op == 0 and w & 63 == 9):
        return "call"
    if op == 2 or (op == 0 and w & 63 == 8) or mipsstream.conditional(w):
        return "branch"
    if op in _LOADS:
        return "load"
    if op in _STORES:
        return "store"
    if op in (0x10, 0x11):
        return "coprocessor"
    return "alu"


def skeleton(words: Sequence[int]) -> Json:
    """The control-flow graph, call shape and opcode classes of a piece: what any recompile of it keeps."""
    edges = [
        [i, i + 1 + mipsstream.offset(w)]
        for i, w in enumerate(words)
        if mipsstream.conditional(w) and 0 <= i + 1 + mipsstream.offset(w) < len(words)
    ]
    leaders = sorted({0, *(t for _, t in edges), *(f + 2 for f, _ in edges if f + 2 < len(words))})
    return {
        "words": len(words),
        "blocks": len(leaders),
        "edges": edges,
        "calls": [i for i, w in enumerate(words) if w >> 26 == 3],
        "classes": dict(sorted(Counter(_kind(w) for w in words).items())),
        "key": sha256(struct.pack(f">{len(words)}q", *tokens(words))),
    }


def calls(words: Sequence[int]) -> list[Json]:
    """Every `jal`: the word, the target and whether it lies in the RAM window."""
    out: list[Json] = []
    for i, w in enumerate(words):
        if w >> 26 == 3:
            target = 0x80000000 | (w & 0x3FFFFFF) << 2
            out.append({"word": i, "target": target})
    return out


def references(words: Sequence[int]) -> list[Json]:
    """The data a piece reads, writes or takes the address of: each `lui` of the RAM window and the instruction that
    completes its address (a load or store through the register, or an `addiu`/`ori` of it), with the size of the
    access. A register holds its `lui` until the piece overwrites it."""
    held: dict[int, tuple[int, int]] = {}
    out: list[Json] = []
    for i, w in enumerate(words):
        op, rs, rt = w >> 26, (w >> 21) & 31, (w >> 16) & 31
        if op == 0x0F:
            hi = w & 0xFFFF
            if RAM[0] >> 16 <= hi < RAM[1] >> 16:
                held[rt] = (hi, i)
            else:
                held.pop(rt, None)
            continue
        if rs in held and (op in _LOADS or op in _STORES or op in _ADDRESS_OPS):
            hi, at = held[rs]
            lo = w & 0xFFFF
            lo = lo - 0x10000 if lo & 0x8000 and op != 13 else lo
            kind = "load" if op in _LOADS else "store" if op in _STORES else "address"
            out.append(
                {
                    "address": (hi << 16) + lo,
                    "size": _LOADS.get(op) or _STORES.get(op),
                    "kind": kind,
                    "lui": at,
                    "word": i,
                    "float": op in (0x31, 0x35, 0x39, 0x3D),
                }
            )
        if op == 0 and w & 63 not in (8, 0x18, 0x19, 0x1A, 0x1B, 0x11, 0x13):
            held.pop((w >> 11) & 31, None)
        elif op in _LOADS or op in _ADDRESS_OPS or op in (8, 10, 11, 12):
            held.pop(rt, None)
    return out


@dataclass(frozen=True)
class Piece:
    """A cut of a usable run: words[first:last] of a stream; `head` and `tail` say which end a cut left open."""

    first: int
    last: int
    head: bool  # opens mid-function: no prologue
    tail: bool  # ends where a cut left it: no return


def pieces(words: Sequence[int], low: int, high: int) -> list[Piece]:
    """The pieces of the usable run words[low:high]: cut after every return and its delay slot (and the zero padding
    after it); a piece with fewer than MIN_WORDS words is dropped."""
    out: list[Piece] = []
    start = low
    i = low
    while i < high:
        if words[i] == mipsstream.RETURN and i + 1 < high:
            end = i + 2
            out.append(Piece(start, end, not mipsstream.prologue(words[start]), False))
            while end < high and words[end] == 0:
                end += 1
            start = i = end
            continue
        i += 1
    if start < high:
        out.append(Piece(start, high, not mipsstream.prologue(words[start]), True))
    return [p for p in out if p.last - p.first >= MIN_WORDS]


def disassemble(rom: bytes | memoryview, low: int, high: int, shifted: bool = False) -> list[Json]:
    """The pieces of rom[low:high] on the word-aligned phase of the mipsstream grid, as ROM ranges; with `shifted`, the
    pieces of the other three phases instead (never code: `not_code` names them). A piece that another phase of the
    same kind also gives is kept once, at the phase with the longer usable run."""
    found: dict[tuple[int, int], Json] = {}
    for phase in mipsstream.phases(low, shifted):
        count = (high - low - phase) // 4
        if count < MIN_WORDS:
            continue
        at = low + phase
        words = mipsstream.unpack(bytes(rom[at : at + 4 * count]))
        for a, b in mipsstream.runs(mipsstream.usable(words)):
            if b - a < MIN_WORDS:
                continue
            for piece in pieces(words, a, b):
                span = (at + 4 * piece.first, at + 4 * piece.last)
                found[span] = {
                    "stream": [span[0], span[1]],
                    "phase": phase,
                    "head": piece.head,
                    "tail": piece.tail,
                    "words": words[piece.first : piece.last],
                }
    rows = sorted(found.values(), key=lambda r: r["stream"])
    kept: list[Json] = []
    for row in rows:  # two phases of one run overlap: the longer wins
        if kept and row["stream"][0] < kept[-1]["stream"][1]:
            if row["stream"][1] - row["stream"][0] > kept[-1]["stream"][1] - kept[-1]["stream"][0]:
                kept[-1] = row
            continue
        kept.append(row)
    return kept


def classes(words: Sequence[int]) -> tuple[int, ...]:
    """The opcode class of each word with no branch offset: the loosest skeleton, used to name the function an
    unreproduced piece is nearest to, never to close it."""
    return tuple(-1 if w == 0 else w >> 26 if w >> 26 else 64 + (w & 63) for w in words)


def _window(tokens_: Sequence[int], at: int) -> int:
    return hash(tuple(tokens_[at : at + WINDOW]))


class Index:
    """The compiled functions of a project, searched by the tokens of a piece; `order` lists its versions."""

    def __init__(self, compiled: Iterable[Compiled], order: Sequence[str] = ()) -> None:
        self.compiled = list(compiled)
        self.order = {v: i for i, v in enumerate(order)}
        self.named = {(c.version, c.name): c for c in self.compiled}
        self._tokens = [tokens(c.words) for c in self.compiled]
        self._windows: dict[int, list[tuple[int, int]]] = {}
        self._exact: dict[int, list[tuple[int, int]]] = {}
        self._classes: dict[int, list[tuple[int, int]]] = {}
        for i, tok in enumerate(self._tokens):
            words = self.compiled[i].words
            klass = classes(words)
            for at in range(len(tok) - WINDOW + 1):
                self._windows.setdefault(_window(tok, at), []).append((i, at))
                if any(words[at : at + WINDOW]):
                    self._exact.setdefault(_window(words, at), []).append((i, at))
                    self._classes.setdefault(_window(klass, at), []).append((i, at))

    def function(self, version: str, name: str) -> Compiled:
        found = self.named.get((version, name))
        if found is None:
            raise EarlierBuildError(f"{version} has no compiled function {name}")
        return found

    def candidates(self, piece: Sequence[int], version: str, rom: int | None = None) -> list[tuple[Compiled, int]]:
        """The compiled functions whose tokens hold the piece's, with the word of the function the piece starts at. The
        order is the choice: the version itself, the project's order, the lowest offset, the name. A function of the
        version that sits where the piece does (`rom` is the piece's offset) is the piece itself, not a copy of it."""
        want = tokens(piece)
        if len(want) < WINDOW:
            return []
        found = []
        for i, at in self._windows.get(_window(want, 0), ()):
            c = self.compiled[i]
            if (
                rom is not None
                and c.version == version
                and c.rom is not None
                and c.rom <= rom < c.rom + 4 * len(c.words)
            ):
                continue
            if self._tokens[i][at : at + len(want)] == want:
                found.append((c, at))
        found.sort(
            key=lambda c: (
                c[0].version != version,
                self.order.get(c[0].version, len(self.order)),
                c[0].rom if c[0].rom is not None else 1 << 32,
                c[0].name,
                c[1],
            )
        )
        return found


def _order(index: Index, version: str, c: Compiled, at: int) -> tuple[bool, int, int, str, int]:
    return (
        c.version != version,
        index.order.get(c.version, len(index.order)),
        c.rom if c.rom is not None else 1 << 32,
        c.name,
        at,
    )


def stretch(index: Index, words: Sequence[int], first: int, version: str, rom: int) -> tuple[Compiled, int, int] | None:
    """The longest run of words[first:] that one compiled function holds word for word, as (function, its word, length),
    or None. `rom` is the offset of words[0]; a function of the version at that place is the piece itself."""
    want = tuple(words[first : first + WINDOW])
    if len(want) < WINDOW or not any(want):
        return None
    best: tuple[tuple[int, bool, int, int, str, int], Compiled, int, int] | None = None
    for i, at in index._exact.get(_window(words, first), ()):
        c = index.compiled[i]
        if c.words[at : at + WINDOW] != want:
            continue
        here = rom + 4 * first
        if c.version == version and c.rom is not None and c.rom <= here < c.rom + 4 * len(c.words):
            continue
        n = WINDOW
        while first + n < len(words) and at + n < len(c.words) and words[first + n] == c.words[at + n]:
            n += 1
        key = (-n, *_order(index, version, c, at))
        if best is None or key < best[0]:
            best = (key, c, at, n)
    return None if best is None else (best[1], best[2], best[3])


def near(index: Index, words: Sequence[int], version: str) -> list[Json]:
    """The functions whose opcode classes hold the first WINDOW words' classes of a piece the compiler does not
    reproduce: the candidates it was compiled from, in the order of the choice. Naming them closes nothing."""
    klass = classes(words)
    if len(klass) < WINDOW:
        return []
    found = []
    for i, at in index._classes.get(_window(klass, 0), ()):
        c = index.compiled[i]
        if classes(c.words[at : at + WINDOW]) == klass[:WINDOW]:
            found.append((c, at))
    found.sort(key=lambda f: _order(index, version, f[0], f[1]))
    return [
        {
            "version": c.version,
            "name": c.name,
            "rom": c.rom,
            "word": at,
            "differing": differences(words, c.words[at : at + len(words)])[:SHOWN]
            if at + len(words) <= len(c.words)
            else None,
        }
        for c, at in found[:SHOWN]
    ]


def differences(stale: Sequence[int], rebuilt: Sequence[int]) -> list[Json]:
    """Every word of the stale piece that the rebuild does not give."""
    from rabbitizer import Instruction

    return [
        {
            "word": i,
            "stale": f"{a:#010x}",
            "rebuilt": f"{b:#010x}",
            "stale_asm": Instruction(a).disassemble(),
            "rebuilt_asm": Instruction(b).disassemble(),
        }
        for i, (a, b) in enumerate(zip(stale, rebuilt, strict=True))
        if a != b
    ]


def _listed(candidates: Sequence[tuple[Compiled, int]]) -> list[Json]:
    return [{"version": c.version, "name": c.name, "rom": c.rom, "word": k} for c, k in candidates[:SHOWN]]


def receipt(
    words: Sequence[int], f: Compiled, at: int, candidates: Sequence[tuple[Compiled, int]], stream: Sequence[int]
) -> Json:
    """The receipt of a piece the compiler reproduced: the skeleton, the match, the rebuild and its list of differing
    instructions (empty: the compiler's words are the stale words)."""
    n = len(words)
    rebuilt = f.words[at : at + n]
    diff = differences(words, rebuilt)
    return {
        "stream": list(stream),
        "words": n,
        "stream_sha256": sha256(struct.pack(f">{n}I", *words)),
        "skeleton": skeleton(words),
        "calls": calls(words),
        "references": references(words),
        "function": {
            "version": f.version,
            "name": f.name,
            "vram": f.vram,
            "rom": f.rom,
            "word": at,
            "words": len(f.words),
        },
        "chosen": 0,
        "candidates": _listed(candidates),
        "rebuilt": {
            "source": f.source,
            "image": f.image,
            "image_sha256": f.image_sha256,
            "sha256": sha256(struct.pack(f">{n}I", *rebuilt)),
        },
        "differences": diff,
        "source_change": "none: the compiler makes every word from the current source" if not diff else "unexplained",
    }


def read(rom: bytes | memoryview, version: str, low: int, high: int, index: Index) -> tuple[list[Json], list[Json]]:
    """(closed pieces, refused pieces) of rom[low:high]: a closed piece is the row {stream, function, word, receipt}
    the compiler reproduced word for word (the whole piece, or each stretch of it that one function holds); a refused
    one is a run of words no function holds, with its skeleton, the functions it is nearest to and the words that
    differ."""
    closed: list[Json] = []
    refused: list[Json] = []
    for piece in disassemble(rom, low, high):
        words, base = piece["words"], piece["stream"][0]
        at, loose = 0, None
        while at < len(words):
            found = stretch(index, words, at, version, base)
            if found is None or found[2] < MIN_WORDS:
                loose = at if loose is None else loose
                at += 1
                continue
            if loose is not None:
                refused.append(_refused(words, loose, at, base, piece, index, version))
                loose = None
            f, k, n = found
            part = words[at : at + n]
            span = [base + 4 * at, base + 4 * (at + n)]
            cands = index.candidates(part, version, span[0])
            closed.append(
                {
                    "stream": span,
                    "function": f.name,
                    "from_version": f.version,
                    "word": k,
                    "receipt": receipt(part, f, k, cands, span),
                }
            )
            at += n
        if loose is not None:
            refused.append(_refused(words, loose, len(words), base, piece, index, version))
    return closed, [r for r in refused if r["words"] >= MIN_WORDS and any(words_of(rom, r))]


def words_of(rom: bytes | memoryview, row: Json) -> tuple[int, ...]:
    return mipsstream.unpack(bytes(rom[row["stream"][0] : row["stream"][1]]))


def shape(rom: bytes | memoryview, low: int, high: int) -> str:
    """What the bytes of a range the reader leaves are, by their words: `pad` (zeros), `address table` (words in the
    RAM window: the jump and address tables of an older link), `code` (a run of canonical instructions no function
    holds) or `unread`."""
    a, b = (low + 3) & ~3, high & ~3
    words = mipsstream.unpack(bytes(rom[a:b])) if b > a else ()
    if not any(words):
        return "pad"
    if sum(RAM[0] <= w < RAM[1] or w == 0 for w in words) * 4 >= len(words) * 3:
        return "address table"
    if all(mipsstream.usable(words)):
        return "code"
    return "unread"


def _kind_of(rom: bytes | memoryview, low: int, high: int, shifted: Sequence[Json]) -> str:
    """`shape`, except that bytes the aligned read leaves unread and only a shifted phase decodes are `not-code`."""
    kind = shape(rom, low, high)
    if kind == "unread" and any(low < n["stream"][1] and n["stream"][0] < high for n in shifted):
        return "not-code"
    return kind


def _refused(words: Sequence[int], a: int, b: int, base: int, piece: Json, index: Index, version: str) -> Json:
    part = words[a:b]
    return {
        "stream": [base + 4 * a, base + 4 * b],
        "words": len(part),
        "head": piece["head"] if a == 0 else True,
        "tail": piece["tail"] if b == len(words) else True,
        "skeleton": skeleton(part),
        "calls": calls(part),
        "references": references(part),
        "near": near(index, part, version),
    }


def prove(
    rom: bytes | memoryview,
    start: int,
    end: int,
    stream: Sequence[int],
    version: str,
    name: str,
    word: int,
    index: Index,
) -> Json:
    """The receipt of the region rom[start:end] inside the stream rom[stream[0]:stream[1]] that the compiler makes as
    words word.. of the function `name` of `version`. Any differing word refuses with the list of them."""
    first, last = stream
    if not 0 <= first < last <= len(rom) or (last - first) % 4:
        raise EarlierBuildError("a stream is a whole number of words inside the ROM")
    if first % mipsstream.WORD:
        raise EarlierBuildError(
            "not code: the stream starts at a shifted word phase, code sits only on word boundaries"
        )
    if not first <= start < end <= last:
        raise EarlierBuildError("the region is not inside its stream")
    words = mipsstream.unpack(bytes(rom[first:last]))
    ok = mipsstream.usable(words)
    if not all(ok):
        raise EarlierBuildError(f"word {ok.index(False)} is no canonical instruction in compiled code")
    f = index.function(version, name)
    if word < 0 or word + len(words) > len(f.words):
        raise EarlierBuildError(f"the stream is longer than {name} from word {word}")
    diff = differences(words, f.words[word : word + len(words)])
    if diff:
        shown = "; ".join(f"word {d['word']} {d['stale_asm']} != {d['rebuilt_asm']}" for d in diff[:SHOWN])
        raise EarlierBuildError(f"{name} makes {len(diff)} words differently from the ROM: {shown}")
    return {
        **receipt(words, f, word, index.candidates(words, version, first), stream),
        "region": [start, end],
        "region_sha256": sha256(bytes(rom[start:end])),
    }


def spans(closed: Iterable[Json], targets: Iterable[tuple[int, int]]) -> list[Json]:
    """The rows (region inside targets, stream, function) that closed pieces give."""
    rows = []
    for low, high in targets:
        for c in closed:
            a, b = max(low, c["stream"][0]), min(high, c["stream"][1])
            if a < b:
                rows.append(
                    {
                        "region": [a, b],
                        "stream": c["stream"],
                        "function": c["function"],
                        "from_version": c["from_version"],
                        "word": c["word"],
                        "receipt": c["receipt"],
                    }
                )
    return sorted(rows, key=lambda r: r["region"])


def compiled(config: Any, tree: Any) -> list[Compiled]:
    """The compiler's output as the build left it: each function's words in `build/<version>/<project>.elf`, the image
    `make` links from the current source. A function with no C body (still behind an `INCLUDE_*` stand-in, or fuzzy)
    is assembled or approximate, never what the compiler makes of source, and is left out."""
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection

    from unbake import symbols

    root = Path(config.project.root)
    skip = {(v, n) for m in tree.modules for v, names in m.stand_ins.items() for n in names}
    skip |= {(v, n) for m in tree.modules for v in m.owns for n in m.fuzzy}
    out: list[Compiled] = []
    for version in config.project.version:
        path = root / "build" / version / f"{config.project.name}.elf"
        if not path.is_file():
            continue
        digest = sha256(path.read_bytes())
        known = symbols.load(config, version)
        with path.open("rb") as handle:
            elf = ELFFile(handle)
            for section in elf.iter_sections():
                if not isinstance(section, SymbolTableSection):
                    continue
                for symbol in section.iter_symbols():
                    index = symbol["st_shndx"]
                    if symbol["st_info"]["type"] != "STT_FUNC" or not isinstance(index, int) or not symbol["st_size"]:
                        continue
                    body = elf.get_section(index)
                    source = tree.owner(version, symbol.name)
                    if body["sh_type"] != "SHT_PROGBITS" or source is None or (version, symbol.name) in skip:
                        continue
                    low = symbol["st_value"] - body["sh_addr"]
                    data = body.data()[low : low + symbol["st_size"]]
                    if len(data) % 4:
                        continue
                    fact = known.get(symbol.name)
                    out.append(
                        Compiled(
                            version,
                            symbol.name,
                            symbol["st_value"],
                            None if fact is None else fact[1],
                            mipsstream.unpack(data),
                            source,
                            path.relative_to(root).as_posix(),
                            digest,
                        )
                    )
    return out


def unowned(config: Any, tree: Any, version: str) -> list[tuple[int, int]]:
    """The ROM extents of the symbols of a version no source file defines: the bytes of data the older link left."""
    from unbake import report

    return [(at, at + size) for at, size, _ in report.unowned(config, tree)[version]]


def blobs_subtract(spans_: Iterable[tuple[int, int]], taken: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    from unbake import blobs

    return blobs.subtract(blobs.merge(list(spans_)), blobs.merge(list(taken)))


NOT_CODE = (
    "shifted word phase: decodes only at a ROM offset that is not word-aligned, and code sits only on word boundaries"
)


def not_code(
    rom: bytes | memoryview, spans_: Iterable[tuple[int, int]], taken: Iterable[tuple[int, int]]
) -> list[Json]:
    """The spans that decode as code only at a shifted word phase, each with its reason: never a candidate, never
    counted as code-shaped. `taken` are the ROM ranges the aligned read already holds (a shifted piece inside them is
    theirs)."""
    own = sorted(taken)
    out: list[Json] = []
    for low, high in spans_:
        for piece in disassemble(rom, low, high, shifted=True):
            a, b = piece["stream"]
            if any(a < e and s < b for s, e in own) or not any(piece["words"]):
                continue
            out.append({"stream": [a, b], "bytes": b - a, "phase": a % mipsstream.WORD, "reason": NOT_CODE})
    return out


def scan(roms: Mapping[str, bytes], targets: Mapping[str, Sequence[tuple[int, int]]], index: Index) -> dict[str, Json]:
    """Per version: the rows the compiler closes inside `targets` (as the rows `unbake blobs prove` takes, the receipt
    in each) and the pieces it does not, with their candidates and the words that differ."""
    out: dict[str, Json] = {}
    for version, spans_ in targets.items():
        closed: list[Json] = []
        refused: list[Json] = []
        for low, high in spans_:
            c, r = read(roms[version], version, low, high, index)
            closed.extend(c)
            refused.extend(r)
        rows = spans(closed, spans_)
        left = blobs_subtract(spans_, [tuple(r["region"]) for r in rows])
        aligned = [tuple(r["stream"]) for r in [*closed, *refused]]
        shifted = not_code(roms[version], left, aligned)
        out[version] = {
            "rows": rows,
            "refused": refused,
            "not_code": shifted,
            "left": [
                {"region": [a, b], "bytes": b - a, "kind": _kind_of(roms[version], a, b, shifted)} for a, b in left
            ],
        }
    return out
