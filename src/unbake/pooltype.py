"""Type an older build's rodata pool: a ROM range in, typed C out, every item the rules cannot type refused and listed.

The pool is the read-only data the older link put between its source files: NUL-terminated strings padded with NULs
to a word, runs of floats, doubles on 8-byte boundaries and small integer words. A word is typed only when its bits
say what it is: a printable NUL-terminated run (3+ characters) padded with NULs is a string, a word whose exponent
is a plausible float exponent is a float, a pair on an 8-byte boundary whose low word is no float and no small
integer is a double, a word within -0x8000..0xFFFF is an integer, and zero follows its previous run. Anything else
(pointer-looking words, packed bytes, high-bit text, a string whose padding is not NUL) is refused with its span and
reason, never guessed. A file path string (`C:/.../name.cpp`, the assertion macro's __FILE__) names a source file:
the items after the last string before that path belong to it, because the older compiler wrote each function's
floats before its strings. Each run of accepted items inside one file is a window and one source file."""

from __future__ import annotations

import re
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from unbake import config, windows
from unbake.contracts import Config, Finding, Json, Refusal

FLOAT_LOW, FLOAT_HIGH = 0x60, 0xA0  # biased exponents (2^-31..2^33) a float constant of game code has
FLT_MAX = 0x7F7FFFFF
SMALL = range(-0x8000, 0x100001)
JOIN = 8  # most words of an odd run that join the integer run beside it
PRINTABLE = frozenset(range(0x20, 0x7F)) | {9, 10, 13}
PATH = re.compile(r"[A-Za-z]:[/\\](?:[^/\\]+[/\\])*([^/\\]+)\.(?:cpp|c|h)$", re.IGNORECASE)
ESCAPES = {9: "\\t", 10: "\\n", 13: "\\r", 34: '\\"', 92: "\\\\"}
KINDS = {"str": "Str", "float": "Floats", "double": "Doubles", "int": "Ints", "uint": "Words"}


@dataclass(frozen=True)
class Item:
    kind: str  # str, float, double, int (signed words), uint (unsigned words)
    at: int  # ROM offset
    size: int
    value: str | tuple[int, ...]  # a string's text, else the raw words (a double is one value of two words)


def _float_word(w: int) -> bool:
    return w == FLT_MAX or FLOAT_LOW <= (w >> 23) & 0xFF <= FLOAT_HIGH


def _small(w: int) -> bool:
    return (w - (1 << 32 if w >> 31 else 0)) in SMALL


def _string(data: bytes, at: int, high: int) -> tuple[str, int] | None:
    """(text, padded size) of the string at `at`: 3+ printable characters, a NUL, NULs up to the next word."""
    end = at
    while end < high and data[end] in PRINTABLE:
        end += 1
    if end - at < 3 or end >= high or data[end] != 0:
        return None
    stop = end + 1
    while stop % 4:
        if stop >= high or data[stop]:
            return None
        stop += 1
    return data[at:end].decode("ascii"), stop - at


def _double(hi: int, lo: int) -> bool:
    """A double: a plausible 11-bit exponent in the high word, a low word that is no float, integer or text."""
    return (
        0x380 <= (hi >> 20) & 0x7FF <= 0x47F and lo != 0 and not _float_word(lo) and not _small(lo) and not _texty(lo)
    )


def itemize(data: bytes, low: int, high: int, shift: int = 0) -> tuple[list[Item], list[Json]]:
    """(items, refused) of the pool in data[low:high]; `shift` is added to a ROM offset to get its 8-byte phase."""
    items: list[Item] = []
    refused: list[Json] = []
    at = low + (-low) % 4
    if at > low:
        refused.append({"rom": low, "end": at, "why": "unaligned head"})

    def refuse(start: int, end: int, why: str) -> None:
        if refused and refused[-1]["end"] == start and refused[-1]["why"] == why:
            refused[-1]["end"] = end
        else:
            refused.append({"rom": start, "end": end, "why": why})

    while at + 4 <= high:
        text = _string(data, at, high)
        word = struct.unpack_from(">I", data, at)[0]
        if text:
            items.append(Item("str", at, text[1], text[0]))
            at += text[1]
        elif (at + shift) % 8 == 0 and at + 8 <= high and _double(word, struct.unpack_from(">I", data, at + 4)[0]):
            items.append(Item("double", at, 8, (word, struct.unpack_from(">I", data, at + 4)[0])))
            at += 8
        elif word == 0:
            items.append(Item("zero", at, 4, (0,)))
            at += 4
        elif _float_word(word):
            items.append(Item("float", at, 4, (word,)))
            at += 4
        elif _small(word):
            items.append(Item("int" if word >> 31 else "uint", at, 4, (word,)))
            at += 4
        else:
            items.append(Item("odd", at, 4, (word,)))
            at += 4
    if at < high:
        refuse(at, high, "unaligned tail")
    return _settle(_join(items, refuse)), refused


def _texty(word: int) -> bool:
    return sum(b in PRINTABLE and b > 0x20 for b in word.to_bytes(4, "big")) >= 3


def _join(items: Sequence[Item], refuse: Callable[[int, int, str], None]) -> list[Item]:
    """The items with each short run of odd words that touches an integer word typed as unsigned words (the integer
    type its neighbour gives); the other odd words are refused."""
    kept: list[Item] = []
    i = 0
    while i < len(items):
        if items[i].kind != "odd":
            kept.append(items[i])
            i += 1
            continue
        j = i
        while j < len(items) and items[j].kind == "odd" and items[j].at == items[i].at + 4 * (j - i):
            j += 1
        before = next((k for k in reversed(items[:i]) if k.kind != "zero"), None)
        after = next((k for k in items[j:] if k.kind != "zero"), None)
        touch = (before and before.kind in {"int", "uint"} and before.at + before.size <= items[i].at) or (
            after and after.kind in {"int", "uint"} and after.at >= items[j - 1].at + 4
        )
        near = (items[i - 1].kind in {"int", "uint", "zero"} if i else False) or (
            items[j].kind in {"int", "uint", "zero"} if j < len(items) else False
        )
        if j - i <= JOIN and touch and near and not any(_texty(w.value[0]) for w in items[i:j]):  # type: ignore[index]
            kept.extend(Item("uint", w.at, 4, w.value) for w in items[i:j])
        else:
            for w in items[i:j]:
                refuse(
                    w.at,
                    w.at + 4,
                    "word is no float, integer or string and joins no integer run (pointer, packed bytes or code)",
                )
        i = j
    return kept


def _settle(items: Sequence[Item]) -> list[Item]:
    """The items with each zero word joined to the numeric run before it, else the run after it, else unsigned."""
    out = list(items)
    numeric = {"float", "int", "uint"}
    for i, item in enumerate(out):
        if item.kind == "zero":
            before = out[i - 1] if i and out[i - 1].at + out[i - 1].size == item.at else None
            out[i] = Item(before.kind if before and before.kind in numeric else "zero", item.at, 4, (0,))
    for i in range(len(out) - 1, -1, -1):
        if out[i].kind == "zero":
            after = out[i + 1] if i + 1 < len(out) and out[i + 1].at == out[i].at + 4 else None
            out[i] = Item(after.kind if after and after.kind in numeric else "uint", out[i].at, 4, (0,))
    return out


def _contiguous(items: Sequence[Item]) -> list[list[Item]]:
    runs: list[list[Item]] = []
    for item in items:
        # a double keeps its 8-byte phase only where its window offset is a multiple of 8; elsewhere it opens a window
        phased = item.kind != "double" or not runs or (item.at - runs[-1][0].at) % 8 == 0
        if runs and phased and runs[-1][-1].at + runs[-1][-1].size == item.at:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


def sources(items: Sequence[Item]) -> list[tuple[str, int]]:
    """(path, index of the first item) of each source file the path strings name, in order."""
    starts: list[tuple[str, int]] = []
    last_string = -1
    for i, item in enumerate(items):
        if item.kind != "str":
            continue
        if PATH.search(str(item.value)):
            first = 0 if not starts else last_string + 1
            starts.append((str(item.value), first))
        last_string = i
    return starts


def _float(w: int) -> str:
    x = struct.unpack(">f", struct.pack(">I", w))[0]
    text = f"{x:.9g}"
    text = text if any(c in text for c in ".en") else text + ".0"
    return text + "f"


def _double_text(hi: int, lo: int) -> str:
    x = struct.unpack(">d", struct.pack(">II", hi, lo))[0]
    text = f"{x:.17g}"
    return text if any(c in text for c in ".en") else text + ".0"


def _int(w: int, signed: bool) -> str:
    return str(w - (1 << 32)) if signed and w >> 31 else f"0x{w:X}" if w > 9 else str(w)


def _literal(text: str) -> str:
    return '"' + "".join(ESCAPES.get(ord(c), c) for c in text) + '"'


def objects(items: Sequence[Item]) -> list[Item]:
    """The items with each run of one numeric kind merged into one array item (a string stays one object)."""
    out: list[Item] = []
    for item in items:
        last = out[-1] if out else None
        if last and item.kind == last.kind and item.kind != "str" and last.at + last.size == item.at:
            out[-1] = Item(last.kind, last.at, last.size + item.size, (*last.value, *item.value))  # type: ignore[misc]
        else:
            out.append(item)
    return out


def stem(path: str) -> str:
    match = PATH.search(path)
    return re.sub(r"[^A-Za-z0-9]", "", match.group(1)).capitalize() if match else "Pool"


def render(obj: Item, name: str) -> list[str]:
    if obj.kind == "str":
        return [f"char {name}[{obj.size}] = {_literal(str(obj.value))};"]
    count = obj.size // (8 if obj.kind == "double" else 4)
    values = obj.value
    if obj.kind == "float":
        cells = [_float(w) for w in values]  # type: ignore[union-attr]
        kind = "float"
    elif obj.kind == "double":
        cells, kind = [_double_text(*values[i : i + 2]) for i in range(0, len(values), 2)], "double"  # type: ignore[index]
    else:
        cells, kind = [_int(w, obj.kind == "int") for w in values], "int" if obj.kind == "int" else "unsigned int"  # type: ignore[union-attr]
    return [f"{kind} {name}[{count}] = {{ {', '.join(cells)} }};"]


@dataclass
class Unit:
    """One source file of the pool for one run of accepted items."""

    path: str
    rom: int  # the older build's address of the first item, the file name's tail
    items: list[Item]

    @property
    def file(self) -> str:
        match = PATH.search(self.path)
        parts = [p for p in re.split(r"[/\\]", match.group(0)) if p][-2:-1] if match else []
        names = [re.sub(r"[^A-Za-z0-9]", "", p).upper() for p in (*parts, match.group(1) if match else "POOL")]
        return "EARLIER_" + "_".join(n for n in names if n) + f"_{self.rom:X}.CPP"


def units(items: Sequence[Item]) -> list[Unit]:
    """The pool split per source file, then per run of accepted items (a refused gap ends a window)."""
    starts = sources(items) or [("", 0)]
    if starts[0][1] != 0:
        starts.insert(0, ("", 0))
    out: list[Unit] = []
    for k, (path, first) in enumerate(starts):
        last = starts[k + 1][1] if k + 1 < len(starts) else len(items)
        for run in _contiguous(items[first:last]):
            out.append(Unit(path, run[0].at, list(run)))
    return out


def body(unit: Unit, delta: int) -> tuple[list[str], list[Json]]:
    """(C lines, symbol rows) of one unit; names carry the older build's address, so both versions agree."""
    label = stem(unit.path)
    lines: list[str] = []
    symbols: list[Json] = []
    for obj in objects(unit.items):
        vram = (obj.at + delta) & windows.MASK
        name = f"g_earlier{label}{KINDS[obj.kind]}_{vram:06X}"
        lines.extend(render(obj, name))
        symbols.append({"name": name, "vram": vram, "rom": obj.at, "size": obj.size})
    return lines, symbols


def command(cfg: Config, params: Json) -> Json:
    """`type-pool --spec V:LOW:HIGH:DELTA ...`: write a source per unit and its symbol rows (not with --check) and
    report the windows (`rom`, `end`, `delta` per version) and every refused span."""
    specs = []
    for text in params.get("spec") or ():
        v, low, high, delta = text.split(":")
        specs.append((v, int(low, 0), int(high, 0), int(delta, 0)))
    if not specs:
        raise Refusal(Finding("type-pool.spec", "give --spec VERSION:LOW:HIGH:DELTA (hex ok) at least once"))
    found: dict[tuple[str, int], dict[tuple[str, ...], list[str]]] = {}  # (path, older vram) -> body -> versions
    rows: list[Json] = []
    refused: list[Json] = []
    symbols: dict[str, list[Json]] = {}
    for v, low, high, delta in specs:
        items, gaps = itemize(config.rom(cfg.project, v).read_bytes(), low, high, delta)
        refused += [{"version": v, **g} for g in gaps]
        for unit in units(items):
            lines, syms = body(unit, delta)
            first = unit.items[0].at
            found.setdefault((unit.path, (first + delta) & windows.MASK), {}).setdefault(tuple(lines), []).append(v)
            rows.append(
                {
                    "version": v,
                    "rom": first,
                    "end": unit.items[-1].at + unit.items[-1].size,
                    "delta": delta,
                    "path": unit.path,
                }
            )
            symbols.setdefault(v, []).extend(syms)
    written = []
    for (path, vram), bodies in sorted(found.items()):
        for lines, vs in bodies.items():
            guard = " || ".join("defined(VERSION_" + v.upper().replace("-", "_") + ")" for v in vs)
            note = f"Older build's rodata of {path}." if path else "Older build's rodata pool."
            name = Unit(path, vram, []).file + ("" if len(bodies) == 1 else "." + "_".join(vs))
            written.append(name)
            if not params.get("check"):
                target = cfg.project.root / (params.get("dest") or "src/DATA/EARLIER_BUILD") / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"#if {guard}\n/* {note} */\n" + "\n".join(lines) + "\n#endif\n")
    if not params.get("check"):
        for v, syms in symbols.items():
            path = cfg.project.root / windows.symbols_path(v)
            have = path.read_text().splitlines() if path.exists() else []
            known = {line.split(" =")[0] for line in have}
            add = [
                f"{s['name']} = 0x{s['vram']:08X}; // rom:0x{s['rom']:X} size:0x{s['size']:X}"
                for s in syms
                if s["name"] not in known
            ]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join([*have, *add]) + "\n")
    return {"windows": rows, "unfit": refused, "written": written, "bytes": sum(r["end"] - r["rom"] for r in rows)}
