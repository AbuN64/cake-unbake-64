"""Type an older build's address table: a ROM span in, a typed pointer array out, or the span refused naming the word.

An address table is words the older link left that are addresses of the older build (jump tables, handler lists).
The proof rule: a span is typed only when every word is zero (a null pointer) or an address in the RAM window that
is exactly the address of a symbol of the older build's reconstructed layout (`versions/<v>/earlier_build_symbols.txt`,
the symbols OLD-DATA and OLD-RODATA typed). One word that is no address, or an address no symbol has, refuses the whole
span with the span and that word: a table is never half typed and no word is emitted raw. The rule names no version:
a version is a ROM and its older-build symbols, and a table both versions hold has one body and one source.

An edge that splits a word (the neighbour range owns its first or last byte) is widened to the word; the widened bytes
are reported so the neighbour's row is retired with the window."""

from __future__ import annotations

import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass

from unbake import config, windows
from unbake.contracts import Config, Finding, Json, Refusal

RAM = (0x80000000, 0x80800000)  # the address window an older-build pointer lies in
KEY = 0xFFFFFF  # symbols are kept with or without the 0x80 segment bits: both sides compare on the low 24 bits
SYMBOL = re.compile(r"^(\w+)\s*=\s*0x([0-9A-Fa-f]+);\s*(?://\s*(.*))?$")


@dataclass(frozen=True)
class Table:
    """An address table typed: the words it holds as symbol names (None is a null pointer)."""

    rom: int  # word-aligned start
    end: int
    targets: tuple[str | None, ...]


def symbols(text: str) -> dict[int, str]:
    """address (low 24 bits) -> symbol name of an earlier_build_symbols.txt; the first name of an address wins."""
    out: dict[int, str] = {}
    for line in text.splitlines():
        match = SYMBOL.match(line.strip())
        if match:
            out.setdefault(int(match.group(2), 16) & KEY, match.group(1))
    return out


def widen(low: int, high: int) -> tuple[int, int]:
    return low & ~3, (high + 3) & ~3


def refusal(version: str, low: int, high: int, at: int, word: int, why: str) -> Json:
    """The span refused and the word that refused it (`at` is the word's ROM offset)."""
    return {"version": version, "rom": low, "end": high, "word_rom": at, "word": f"0x{word:08X}", "why": why}


def read(rom: bytes, version: str, low: int, high: int, known: Mapping[int, str]) -> Table | Json:
    """The span in rom[low:high] as a Table, or the refusal row of its first word that does not resolve."""
    a, b = widen(low, high)
    if b > len(rom):
        return refusal(version, low, high, len(rom), 0, "span runs past the ROM")
    names: list[str | None] = []
    for at in range(a, b, 4):
        (word,) = struct.unpack(">I", rom[at : at + 4])
        if word == 0:
            names.append(None)
        elif not RAM[0] <= word < RAM[1]:
            return refusal(version, low, high, at, word, "not an address of the older build's RAM window")
        elif (word & KEY) not in known:
            return refusal(version, low, high, at, word, "no symbol of the older build's layout has this address")
        else:
            names.append(known[word & KEY])
    return Table(a, b, tuple(names))


def render(table: Table, name: str) -> list[str]:
    """C lines: an extern per distinct target, then the pointer array (a null word is 0)."""
    used = sorted({t for t in table.targets if t})
    cells = [t or "0" for t in table.targets]
    return [*[f"extern char {t}[];" for t in used], f"void *{name}[{len(cells)}] = {{ {', '.join(cells)} }};"]


def label(vram: int) -> str:
    return f"g_earlierAddrTable_{vram & KEY:06X}"


def command(cfg: Config, params: Json) -> Json:
    """`type-pointers --spec V:LOW:HIGH:DELTA ...`: type each span as a pointer array of older-build symbols and write
    one source per table (not with --check) with its symbol row; report the tables typed, the bytes they close and
    every refused span with the word that refused it."""
    specs = []
    for text in params.get("spec") or ():
        v, low, high, delta = text.split(":")
        specs.append((v, int(low, 0), int(high, 0), int(delta, 0)))
    if not specs:
        raise Refusal(Finding("type-pointers.spec", "give --spec VERSION:LOW:HIGH:DELTA (hex ok) at least once"))
    rows: list[Json] = []
    refused: list[Json] = []
    found: dict[int, dict[tuple[str, ...], list[str]]] = {}  # older vram -> body -> versions
    symbols_of: dict[str, list[Json]] = {}
    known: dict[str, dict[int, str]] = {}
    for v, low, high, delta in specs:
        if v not in known:
            path = cfg.project.root / windows.symbols_path(v)
            known[v] = symbols(path.read_text()) if path.exists() else {}
        rom = config.rom(cfg.project, v).read_bytes()
        got = read(rom, v, low, high, known[v])
        if not isinstance(got, Table):
            refused.append(got)
            continue
        vram = (got.rom + delta) & windows.MASK
        body = tuple(render(got, label(vram)))
        found.setdefault(vram, {}).setdefault(body, []).append(v)
        rows.append(
            {
                "version": v,
                "rom": got.rom,
                "end": got.end,
                "delta": delta,
                "widened": [[got.rom, low], [high, got.end]] if (got.rom, got.end) != (low, high) else [],
            }
        )
        symbols_of.setdefault(v, []).append(
            {"name": label(vram), "vram": vram, "rom": got.rom, "size": got.end - got.rom}
        )
    written = []
    for vram, bodies in sorted(found.items()):
        name = f"EARLIER_ADDRTABLE_{vram & KEY:06X}.CPP"  # one source per address: bodies that differ are guarded apart
        written.append(name)
        if not params.get("check"):
            arms = []
            for i, (body, vs) in enumerate(bodies.items()):
                guard = " || ".join("defined(VERSION_" + v.upper().replace("-", "_") + ")" for v in vs)
                arms.append(("#if " if i == 0 else "#elif ") + guard + "\n" + "\n".join(body) + "\n")
            target = cfg.project.root / (params.get("dest") or "src/DATA/EARLIER_BUILD") / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("/* Older build's address table. */\n" + "".join(arms) + "#endif\n")
    if not params.get("check"):
        for v, syms in symbols_of.items():
            path = cfg.project.root / windows.symbols_path(v)
            have = path.read_text().splitlines() if path.exists() else []
            names = {line.split(" =")[0] for line in have}
            add = [
                f"{s['name']} = 0x{s['vram']:08X}; // rom:0x{s['rom']:X} size:0x{s['size']:X}"
                for s in syms
                if s["name"] not in names
            ]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join([*have, *add]) + "\n")
    return {"tables": rows, "unfit": refused, "written": written, "bytes": sum(r["end"] - r["rom"] for r in rows)}
