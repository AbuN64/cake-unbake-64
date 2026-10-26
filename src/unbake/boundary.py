"""Boundary: the function rows a split put in the wrong place, found by the code other versions match, and moved.

A row that is not exact (map score below 100) is often not a decompilation job at all: the split started it a word
early (a pad or the last word of the function before), cut it in two (a row with no return and a fragment after it),
or left only its prologue word on the row and the body on the next. Other versions hold the same function exact, at
the length it really has. For each such row this finds where an exact function of another version sits in this
version's ROM, compared up to the fields the link rewrites (the mask of lui/jal immediates and the low half of
addiu/ori/load/store), at a start within MAX_SHIFT bytes of the row: the span it covers is the function's true row.
The rows it overlaps are moved to it: the row that keeps the function (the one of the counterpart's name, else the one
the span overlaps most) starts at the span and has its size; a row inside the span goes (with its INCLUDE_ASM
stand-in); a row the span cuts keeps the part outside it.

A row with no counterpart is only described (tail-fragment: it releases a frame it never made; no-return; bad-end;
ascii): a pad word in front of a function is `submit --pad-split`'s. `submit --boundary` applies the moves to
versions/<v>/symbol_addrs.txt, drops the stand-ins of the rows that go, runs sync and writes the diff (`# MSG:` first
line); a removed row whose name the source spells other than in its stand-in refuses the move. The asm is re-extracted
by `make extract` before a build proves the ROM (the result says so).
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unbake import api, progress
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Refusal

MAX_SHIFT = 64  # bytes a true start may lie before or after the row's
JR_RA = 0x03E00008
_MEM = frozenset(range(32, 47)) | {49, 53, 57, 61}
MSG_MAX = 72
PROLOGUE_MAX = 8  # bytes of the next function's start a row may hold after its own end
MIN_COUNTERPART = 16  # bytes: a shorter exact function (a `jr ra; nop` stub) matches anywhere


def mask(w: int) -> int:
    """A word up to the fields the link writes (and stack offsets): the census's mask (QUICKWIN classify.py)."""
    op = w >> 26
    if op in (2, 3):
        return op << 26
    if op in (9, 13, 15) or op in _MEM:
        return w & 0xFFFF0000
    return w


def describe(words: Sequence[int]) -> str:
    """Why a row with no counterpart looks cut: '' when it reads as one whole function."""
    if not words:
        return "empty"
    data = b"".join(struct.pack(">I", w) for w in words)
    if all(32 <= b < 127 or b in (0, 9, 10) for b in data) and any(32 <= b < 127 for b in data):
        return "ascii"
    first = next((w for w in words if w >> 16 == 0x27BD), None)  # addiu sp, sp, imm
    if first is not None and (first & 0xFFFF) < 0x8000 and (first & 0xFFFF):
        return "tail-fragment"  # releases a frame it never made: the previous function's epilogue
    if JR_RA not in words and not any(w >> 26 == 2 for w in words):
        return "no-return"
    ends = (len(words) >= 2 and (words[-2] == JR_RA or words[-2] >> 26 == 2)) or (
        words[-1] == 0 and JR_RA in words[-4:]
    )
    return "" if ends else "bad-end"


@dataclass
class Row:
    name: str
    vram: int
    rom: int
    size: int | None


@dataclass
class Move:
    version: str
    row: str  # the remaining row it started from
    counterpart: str  # version:name of the exact function the span equals
    start: int  # rom
    size: int
    keep: str
    removed: list[str] = field(default_factory=list)
    trimmed: dict[str, tuple[int, int, int | None]] = field(default_factory=dict)  # name -> (vram, rom, size)

    def shown(self) -> Json:
        return {
            "version": self.version,
            "row": self.row,
            "counterpart": self.counterpart,
            "span": f"0x{self.start:X}+0x{self.size:X}",
            "keep": self.keep,
            "removed": self.removed,
            "moved": {
                n: f"0x{v:08X} rom:0x{r:X}" + (f" size:0x{s:X}" if s else "") for n, (v, r, s) in self.trimmed.items()
            },
        }


def _words(image: bytes, rom: int, size: int) -> tuple[int, ...]:
    size = max(min(size, len(image) - rom), 0) // 4 * 4
    return struct.unpack_from(f">{size // 4}I", image, rom) if size else ()


def _rows(table: Mapping[str, tuple[int, int | None, int | None, str]]) -> list[Row]:
    return sorted(
        (Row(n, r[0], r[1], r[2]) for n, r in table.items() if r[1] is not None),  # a data row ends a function too
        key=lambda r: r.rom,
    )


def _ends(rows: Sequence[Row]) -> dict[str, int]:
    """name -> rom end: its size, else where the next row begins."""
    out = {}
    for k, r in enumerate(rows):
        out[r.name] = r.rom + r.size if r.size else (rows[k + 1].rom if k + 1 < len(rows) else r.rom)
    return out


def _map(root: Path, v: str) -> list[list[Any]]:
    try:
        rows: list[list[Any]] = json.loads((root / progress.map_path(v)).read_text())["fns"]
    except (OSError, ValueError, KeyError):
        return []
    return rows


def find(
    target: tuple[str, int, int],
    image: bytes,
    index: Mapping[tuple[int, int], list[tuple[str, tuple[int, ...]]]],
    version: str = "",
) -> tuple[int, int, str] | None:
    """(rom, size, counterpart) of the longest exact function of another version than VERSION whose masked words
    this IMAGE holds at a start within MAX_SHIFT of the row and covering part of it; None when there is none."""
    _name, rom, end = target
    best: tuple[int, int, str] | None = None
    for p in range(max(rom - MAX_SHIFT, 0), rom + MAX_SHIFT + 1, 4):
        head = _words(image, p, 8)
        if len(head) < 2:
            continue
        for who, masked in index.get((mask(head[0]), mask(head[1])), ()):
            if who.startswith(f"{version}:"):
                continue
            size = 4 * len(masked)
            if (best and size <= best[1]) or not (p < end and rom < p + size):
                continue
            if tuple(map(mask, _words(image, p, size))) == masked:
                best = (p, size, who)
    return best


def _zero(words: Sequence[int]) -> bool:
    return all(w == 0 for w in words)


def ends_function(words: Sequence[int]) -> bool:
    """Whether WORDS (bytes a move leaves to the row before) end a function: a `jr ra` and its delay slot, then only
    zero padding."""
    w = list(words)
    while w and w[-1] == 0 and not (len(w) >= 2 and w[-2] == JR_RA):
        w.pop()
    return len(w) >= 2 and w[-2] == JR_RA


def plan_move(
    v: str, name: str, who: str, start: int, length: int, rows: Sequence[Row], ends: Mapping[str, int], image: bytes
) -> tuple[Move | None, str]:
    """The row moves that give the function the span [START, START + LENGTH), or (None, why) when the bytes the move
    would hand to a neighbour are not the neighbour's (the code differs there: a port, not a boundary)."""
    end = start + length
    over = [r for r in rows if r.rom < end and start < ends[r.name]]
    cname = who.split(":", 1)[1]
    keep = (
        cname
        if cname in {r.name for r in over}
        else max(over, key=lambda r: min(ends[r.name], end) - max(r.rom, start)).name
    )
    move = Move(v, name, who, start, length, keep)
    first, last = over[0], over[-1]
    k = next(r for r in over if r.name == keep)
    at = rows.index(first)
    if start > first.rom:  # the bytes before the span leave its rows: they must end the function before
        lead = _words(image, first.rom, start - first.rom)
        if not (_zero(lead) or ends_function(lead)):
            return None, f"its first 0x{start - first.rom:X} bytes differ from {who}: not the end of a function"
        if first.name != keep:
            move.trimmed[first.name] = (first.vram, first.rom, (start - first.rom) if first.size else None)
        elif at > 0 and rows[at - 1].size:
            prev = rows[at - 1]
            move.trimmed[prev.name] = (prev.vram, prev.rom, start - prev.rom)
        elif at == 0:
            return None, "no row before it to take its first bytes"
    for r in over:
        if r.name != keep and start <= r.rom and ends[r.name] <= end:
            move.removed.append(r.name)
    size = length
    tail_end = ends[last.name]
    if tail_end > end:
        tail = _words(image, end, tail_end - end)
        if _zero(tail):
            size = (tail_end - start) if last.name == keep else length  # padding stays with the function
        elif last.name != keep:
            delta = end - last.rom
            move.trimmed[last.name] = (last.vram + delta, end, (last.size - delta) if last.size else None)
        else:  # the next function's first words sit on this row (a prologue word): the next row starts at the span end
            nxt = rows[rows.index(last) + 1] if rows.index(last) + 1 < len(rows) else None
            if tail_end - end > PROLOGUE_MAX:
                return (
                    None,
                    f"the row goes on 0x{tail_end - end:X} bytes past {who}'s span: other code, not a stray word",
                )
            if nxt is None or describe(_words(image, end, ends[nxt.name] - end)) == "tail-fragment":
                return None, f"0x{tail_end - end:X} bytes after {who}'s span are no function start"
            delta = nxt.rom - end
            move.trimmed[nxt.name] = (nxt.vram - delta, end, (nxt.size + delta) if nxt.size else None)
    if keep == last.name and tail_end > end and _zero(_words(image, end, tail_end - end)) and start == k.rom:
        return move, ""  # only padding differs: the row is right
    move.trimmed[keep] = (k.vram + (start - k.rom), start, size)
    if start == k.rom and size == ends[keep] - k.rom and not move.removed and len(move.trimmed) == 1:
        move.trimmed.clear()  # the row already is the span
    return move, ""


def census(config: Config, versions: Sequence[str], only: str | None = None) -> tuple[list[Move], list[Json]]:
    """(moves, the cut rows no counterpart explains) for the remaining rows of VERSIONS."""
    root = config.project.root
    images = {v: configuration.rom(config.project, v).read_bytes() for v in config.project.versions}
    maps = {v: _map(root, v) for v in config.project.versions}
    index: dict[tuple[int, int], list[tuple[str, tuple[int, ...]]]] = {}
    exact: dict[str, set[tuple[int, ...]]] = {}
    for v, fns in maps.items():
        for name, rom, size, score, *_ in fns:
            if score < 100 or size < MIN_COUNTERPART:
                continue
            masked = tuple(map(mask, _words(images[v], rom, size)))
            if masked in exact.setdefault(v, set()):
                continue
            exact[v].add(masked)
            index.setdefault((masked[0], masked[1]), []).append((f"{v}:{name}", masked))
    moves: list[Move] = []
    loose: list[Json] = []
    for v in versions:
        table = api.symbols(config, v)
        rows = _rows(table)
        ends = _ends(rows)
        by_name = {r.name: r for r in rows}
        done: set[str] = set()
        for name, rom, size, score, *_ in maps[v]:
            if score >= 100 or name in done or (only and name != only) or name not in by_name:
                continue
            row = by_name[name]
            found = find((name, row.rom, ends[name]), images[v], index, v)
            if found is None or (found[0] == row.rom and found[0] + found[1] == ends[name]):
                why = describe(_words(images[v], rom, size))
                if why:
                    loose.append({"version": v, "row": name, "rom": f"0x{rom:X}", "size": size, "looks": why})
                continue
            start, length, who = found
            move, why = plan_move(v, name, who, start, length, rows, ends, images[v])
            if move is None:
                loose.append({"version": v, "row": name, "rom": f"0x{rom:X}", "size": size, "looks": why})
                continue
            if move.trimmed or move.removed:
                done |= {move.keep, *move.removed, *move.trimmed}
                moves.append(move)
    return moves, loose


_ROW = re.compile(r"^(\s*)(\w+)(\s*=\s*)0x([0-9A-Fa-f]+)(\s*;)(.*)$")


def apply_rows(text: str, move: Move) -> str:
    """symbol_addrs text with MOVE's rows removed and moved (vram, rom: and size: of each)."""
    out = []
    for line in text.splitlines(keepends=True):
        found = _ROW.match(line.rstrip("\n"))
        name = found[2] if found else ""
        if name in move.removed:
            continue
        if found and name in move.trimmed:
            vram, rom, size = move.trimmed[name]
            comment = found[6]
            comment = re.sub(r"rom:0x[0-9A-Fa-f]+", f"rom:0x{rom:X}", comment)
            if size is not None:
                if re.search(r"size:0x[0-9A-Fa-f]+", comment):
                    comment = re.sub(r"size:0x[0-9A-Fa-f]+", f"size:0x{size:X}", comment)
                else:
                    comment = comment.rstrip() + f" size:0x{size:X}"
            line = f"{found[1]}{name}{found[3]}0x{vram:08X}{found[5]}{comment}\n"
        out.append(line)
    return "".join(out)


def _stand_ins(config: Config, names: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """({path: text without the INCLUDE_ASM lines of NAMES}, names the source spells otherwise)."""
    from unbake import widen

    root = config.project.root
    files = widen.files_spelling(config, names, ("src", "include"))
    texts: dict[str, str] = {}
    spelled: set[str] = set()
    for path in files:
        text = (root / path).read_text(encoding="utf-8")
        new = text
        for n in names:
            new = re.sub(rf"(?m)^[ \t]*INCLUDE_ASM\(\s*\"[^\"]*\"\s*,\s*{re.escape(n)}\s*\);[ \t]*\n", "", new)
            if re.search(rf"\b{re.escape(n)}\b", new):
                spelled.add(f"{n} ({path})")
        if new != text:
            texts[path] = new
    return texts, sorted(spelled)


def run(config: Config, params: Json) -> Json:
    from unbake import repo, symbols

    versions = [v for v in (params.get("version") or ()) if v] or [
        v for v in config.project.versions if v != config.project.names_from
    ]
    bad = [v for v in versions if v not in config.project.versions]
    if bad:
        raise Refusal(Finding("boundary.request", f"no version {', '.join(bad)}"))
    moves, loose = census(config, versions, params.get("function"))
    shown = {"moves": [m.shown() for m in moves], "cut": loose}
    if params.get("check") or not moves:
        return {**shown, "written": []}
    root = config.project.root
    texts: dict[str, str] = {}
    for v in sorted({m.version for m in moves}):
        path = symbols.path(v)
        text = (root / path).read_text()
        for m in (m for m in moves if m.version == v):
            text = apply_rows(text, m)
        texts[path] = text
    removed = sorted({n for m in moves for n in m.removed})
    sources, spelled = _stand_ins(config, removed)
    if spelled:
        raise Refusal(*[Finding("boundary.spelled", f"a row that goes is spelled by the source: {s}") for s in spelled])
    texts.update(sources)
    old = {p: (root / p).read_bytes() for p in texts}
    synced = False
    try:
        for p, t in texts.items():
            (root / p).write_text(t, encoding="utf-8")
        synced = True
        done = repo.sync(config, {})
    except BaseException:
        for p, before in old.items():
            (root / p).write_bytes(before)
        if synced:
            repo.sync(config, {})
        raise
    written = sorted({*texts, *done.get("written", ())})
    from unbake import widen

    head = f"Move {len(moves)} split row{'s' * (len(moves) > 1)} to the code other versions match"
    out = Path(params["out"]) if params.get("out") else root / ".unbake" / "boundary"
    target = widen.write_diff(
        config, written, head, out / f"boundary-{'-'.join(sorted({m.version for m in moves}))}.diff"
    )
    return {**shown, "written": written, "diff": str(target), "next": "run `make extract`, then build every version"}
