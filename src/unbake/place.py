"""Place a function's read-only data (jump tables, literal pools) in the function's own file.

A function whose body is C compiles its own .rodata (jump tables, float and string literals). The link script and
splat.yaml place that data where a `literals_<function>` fact says; this module writes that fact. The hand edit it
replaces was: a `literals_` row per version, the old stand-in (INCLUDE_RODATA or a word array) gone, the link script
and splat.yaml regenerated. Here the row is written (versions/<v>/aliases.txt, with the size of the data); `sync`
drops the superseded stand-ins and regenerates the link script and yamls, so what is committed is what `unbake sync`
writes. Bytes the old piece held after the function's own data (a pad or foreign word) are not in the compiled
.rodata: they get a name of their own and an INCLUDE_RODATA stand-in at the top of the file that starts after them.
When that file also owns data before the table (its one .rodata section would be linked in front of the table), the
names it owns from the leftover on move to a new sibling file (`<stem>_<n>`) that takes the stand-in instead.

A table or pool that other functions also read is placed with the function asked for, its first owner: an assembly
reader links by the symbol's address and a C reader declares the symbol `extern`, so both keep reading the same bytes;
the plan's note names them; a word array of another file for the same bytes is superseded by the row (sync drops it).
What it refuses (nothing is written then): a function that reads two places of the ROM more than a table's length
apart (`place.apart`, final: one `literals_` row cannot hold both, so it keeps its stand-ins); leftover bytes with no
file after them to hold a stand-in and, when the version's object is built and newer than its source, a compiled
.rodata that is not the ROM's bytes (size, alignment, content).
"""

from __future__ import annotations

import bisect
import itertools
import re
import struct
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from unbake import api, symbols, tree
from unbake import config as configuration
from unbake.contracts import Config, Finding, Refusal

TABLE_MAX = 0x400  # bytes: a longer run of code addresses is no jump table
_NAME = re.compile(r"[A-Za-z_]\w*")
_PIECE = re.compile(r"^\s*-\s*\[0x([0-9A-Fa-f]+),\s*(\w+),\s*([^\]\s]+)\]", re.M)
_SHT_PROGBITS, _SHT_REL = 1, 9


@dataclass(frozen=True)
class Placement:
    version: str
    function: str
    path: str  # the file that compiles the function in this version
    vram: int
    rom: int
    end: int  # rom offset where the function's own data ends
    names: tuple[str, ...]  # the data rows the function reads (or whose words are its jump table)
    note: str = ""
    rest: int = 0  # rom offset of the next row when bytes lie between `end` and it (kept as a stand-in of their own)
    after: str = ""  # the file that owns the data after them: their stand-in goes at the top of it
    segment: str = ""
    split_from: str = ""  # when that file also owns data before the table: `after` is a new sibling taking `split`
    split: tuple[str, ...] = ()  # the names split_from owns at or after `rest`, moved into `after`
    junk: tuple[tuple[str, int], ...] = ()  # (name, rom) of unread rows inside the table
    pads: tuple[
        tuple[str, str], ...
    ] = ()  # (name, file) of the junk rows a file defines, all zero bytes: dropped from it
    tail: str = (
        ""  # the alias row that keeps the bytes of a shared piece after the function's data as a piece of their own
    )

    @property
    def row(self) -> str:
        size = f" size:0x{self.end - self.rom:X}"
        return f"{tree.LITERALS}{self.function} = 0x{self.vram:08X}; // rom:0x{self.rom:X}{size}"

    @property
    def leftover(self) -> str:
        """The row that names the bytes after the function's data, up to the next row: a pad or foreign word the
        compiled .rodata does not hold, which stays behind a stand-in (sync gives it one)."""
        if self.rest <= self.end:
            return ""
        at = self.vram + self.end - self.rom
        return f"D_auto_rodata_{self.function}_{at:08X} = 0x{at:08X}; // rom:0x{self.end:X}"

    @property
    def stand_in(self) -> str:
        return self.leftover.partition(" = ")[0]

    @property
    def file(self) -> str:
        return f"versions/{self.version}/aliases.txt"

    @property
    def junk_files(self) -> tuple[str, str]:
        return f"versions/{self.version}/symbol_addrs.txt", self.file


def jump_target(words: Sequence[int], code: Sequence[tuple[int, int, str]]) -> str | None:
    """The function every word of `words` points into (a jump table's cases are addresses inside its own function),
    by the (vram start, vram end, name) runs of `code` (as the ROM's `text_mask` spells code addresses); None when the
    words are not all inside one."""
    if len(words) < 2:
        return None
    found = set()
    for word in words:
        if word % 4:
            return None
        at = bisect.bisect_right(code, (word, 1 << 40, "")) - 1
        if at < 0 or not code[at][0] <= word < code[at][1]:
            return None
        found.add(code[at][2])
    return next(iter(found)) if len(found) == 1 else None


def _elf(path: Path) -> tuple[bytes, int, set[int]] | None:
    """(.rodata bytes, its alignment, offsets of its relocations) of a big-endian ELF32 object, None when the object
    has no .rodata or is no such file."""
    try:
        d = path.read_bytes()
    except OSError:
        return None
    if d[:4] != b"\x7fELF" or d[4] != 1 or d[5] != 2:
        return None
    (shoff,) = struct.unpack_from(">I", d, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from(">HHH", d, 0x2E)
    rows = [struct.unpack_from(">10I", d, shoff + i * shentsize) for i in range(shnum)]
    strs = d[rows[shstrndx][4] : rows[shstrndx][4] + rows[shstrndx][5]]
    names = [strs[r[0] :].split(b"\0")[0].decode() for r in rows]
    found = [i for i, n in enumerate(names) if n in (".rodata", ".rdata") and rows[i][1] == _SHT_PROGBITS]
    if not found:
        return None
    k = found[0]
    reloc = {
        off
        for r in rows
        if r[1] == _SHT_REL and r[7] == k
        for off in (struct.unpack_from(">I", d, r[4] + 8 * j)[0] for j in range(r[5] // 8))
    }
    return d[rows[k][4] : rows[k][4] + rows[k][5]], rows[k][8], reloc


def _pieces(root: Path, v: str) -> list[tuple[int, str, str]]:
    """(rom, splat type, name) of every piece of the generated versions/<v>/splat.yaml, sorted."""
    path = root / f"versions/{v}/splat.yaml"
    if not path.is_file():
        return []
    return sorted((int(m[1], 16), m[2], m[3]) for m in _PIECE.finditer(path.read_text()))


def _piece_tail(pieces: Sequence[tuple[int, str, str]], rom: int, end: int) -> tuple[int, int, str] | None:
    """(start, stop, name) of the data piece holding `rom` when it runs past `end` (bytes after the function's own data
    that only this piece holds), else None. REV1-ONE-40 01:3x: de/eu/eu-x/us held Nav_PrepareProbeSegments' 10 words
    and func_80210EFC_de's 8-word pool in one 18-word array; the 0x28 row superseded it and the 8 words vanished."""
    at = bisect.bisect_right(pieces, (rom, "\uffff")) - 1
    if at < 0 or at + 1 >= len(pieces):
        return None
    start, kind, name = pieces[at]
    stop = pieces[at + 1][0]
    if kind != "rodata" or name.rpartition("/")[2].startswith(tree.LITERALS) or stop <= end:
        return None
    return start, stop, name


def _is_rodata(pieces: Sequence[tuple[int, str, str]], rom: int) -> bool:
    at = bisect.bisect_right(pieces, (rom, "￿")) - 1
    return at >= 0 and pieces[at][1] == "rodata"


_REF = re.compile(r"\b([A-Za-z_]\w*)\b(?:\s*\+\s*(0[xX][0-9A-Fa-f]+|\d+))?")


def _asm_refs(root: Path, config: Config, v: str, function: str, table: Mapping[str, Any]) -> dict[str, int]:
    """name -> the smallest byte offset past it that the extracted assembly of `function` in version `v` uses (`sym`: 0,
    `sym + 0xC`: 12): the data the function reads begins there, not at the start of a row that also holds foreign bytes
    before it. Empty: no such file."""
    try:
        file = root / api.asm_path(config, v, function)
    except Refusal:
        return {}
    if not file.is_file():
        return {}
    text = re.sub(r"/\*.*?\*/", "", file.read_text())
    refs: dict[str, int] = {}
    for m in _REF.finditer(text):
        if m[1] in table:
            at = int(m[2], 0) if m[2] else 0
            refs[m[1]] = min(refs.get(m[1], at), at)
    return refs


def _asm_names(root: Path, config: Config, v: str, function: str, table: Mapping[str, Any]) -> set[str]:
    """The symbol names the extracted assembly of `function` in version `v` mentions (none: no such file)."""
    return set(_asm_refs(root, config, v, function, table))


_WORD = re.compile(r"[A-Za-z_]\w*")
_INDEX: dict[tuple[str, str], dict[str, list[Path]]] = {}


def _index(root: Path, v: str) -> dict[str, list[Path]]:
    """word -> the extracted assembly files of version v that contain it, in path order. Built once per plan call: the
    alternation regex it replaces cost 14 s per call, 143 of the 154 CPU s of one sync --place."""
    key = (str(root), v)
    if key not in _INDEX:
        index: dict[str, list[Path]] = {}
        for file in sorted((root / "asm" / v / "nonmatchings").rglob("*.s")):
            for word in set(_WORD.findall(file.read_text(errors="replace"))):
                index.setdefault(word, []).append(file)
        _INDEX[key] = index
    return _INDEX[key]


def _readers(
    root: Path, v: str, names: Collection[str], skip: str, left: Collection[Path] = ()
) -> dict[str, list[str]]:
    """name -> the functions of version `v` (other than `skip`) whose extracted assembly mentions it. `left` are the
    files an earlier name of a function left behind (symbols.superseded_asm): the same function, not a second reader."""
    if not names:
        return {}
    index = _index(root, v)
    found: dict[str, list[str]] = {}
    for n in names:
        for file in index.get(n, ()):
            if file.name != skip and file not in left:
                found.setdefault(n, []).append(file.stem)
    return found


def _extent(
    root: Path,
    current: Any,
    v: str,
    table: Mapping[str, Any],
    rows: Sequence[tuple[str, int, int]],
    function: str,
    zero: Callable[[str], bool] = lambda n: False,
) -> tuple[list[str], int, int, int, list[str], list[str]]:
    """(names, vram, rom, end, inside, junk) of the data rows of the function: the span they cover, rows that are
    not its own inside the span that something reads or a file owns (inside), and those nothing reads (junk). A row
    a file owns is junk too when `zero` says its ROM bytes are all zero: a pad the file defines as a zero array."""
    names = [n for n, _, _ in rows]
    first, last = names[0], names[-1]
    vram, rom = table[first][0] + rows[0][1], (table[first][1] or 0) + rows[0][1]
    end = (table[last][1] or 0) + rows[-1][1] + rows[-1][2]
    inside = sorted(
        n
        for n, f in table.items()
        if f[1] is not None
        and rom < f[1] < end
        and n not in names
        and f[3] not in ("func", "label")
        and not n.startswith(tree.LITERALS)
        and f[1] not in {table[x][1] for x in names}
    )
    junk: list[str] = []
    if inside:  # a row between the function's own is junk when nothing reads it and no file owns it
        seen = _readers(root, v, inside, f"{function}.s", set(symbols.superseded_asm(root, v, table)))
        junk = [n for n in inside if n not in seen and (not current.owned(v, n) or zero(n))]
        inside = [n for n in inside if n not in junk]
    return names, vram, rom, end, inside, junk


def _apart(table: Mapping[str, Any], rows: Sequence[tuple[str, int, int]]) -> str:
    """The sentence for a span that is really two places: the function reads rows more than a table's length apart (the
    ROM keeps them in different pieces, so no one pool can hold both)."""
    for (a, skip, size), (b, _, _) in itertools.pairwise(rows):
        gap = (table[b][1] or 0) - ((table[a][1] or 0) + skip + size)
        if gap > TABLE_MAX:
            return f"it reads {a} (0x{table[a][1]:X}) and {b} (0x{table[b][1]:X}), 0x{gap:X} bytes apart"
    return ""


def _shared_rows(
    root: Path, v: str, table: Mapping[str, Any], rows: Sequence[tuple[str, int, int]], function: str
) -> set[str]:
    """The names among `rows` that another function's assembly reads too."""
    aka = {n: [m for m, f in table.items() if f[0] == table[n][0]] for n, _, _ in rows}
    left = set(symbols.superseded_asm(root, v, table))
    read = _readers(root, v, {m for ms in aka.values() for m in ms}, f"{function}.s", left)
    return {n for n, _, _ in rows if any(read.get(m) for m in aka[n])}


def _trim_shared(
    root: Path,
    current: Any,
    v: str,
    table: Mapping[str, Any],
    rows: Sequence[tuple[str, int, int]],
    function: str,
    zero: Callable[[str], bool] = lambda n: False,
) -> tuple[list[tuple[str, int, int]], tuple[Any, ...]] | None:
    """When the span of `rows` holds foreign rows, drop the shared rows at its ends one at a time (a constant other
    functions read stays its own shared row, declared extern by them) until the span is clean: (rows, extent), or None
    when no trimming makes it clean."""
    shared = _shared_rows(root, v, table, rows, function)
    cur = list(rows)
    while len(cur) > 1:
        if cur[-1][0] in shared:
            cur = cur[:-1]
        elif cur[0][0] in shared:
            cur = cur[1:]
        else:
            return None
        got = _extent(root, current, v, table, cur, function, zero)
        if not got[4]:
            return cur, got
    return None


def _extern_rows(
    config: Config,
    v: str,
    table: Mapping[str, Any],
    rows: Sequence[tuple[str, int, int]],
    path: str,
    cache: dict[tuple[str, str], str],
) -> set[str]:
    """The names among `rows` that the C of `path`, as version `v` compiles it, declares `extern`: the function reads
    them as another file's data and the bytes stay with their owner, so no pool of the function is there to place
    (Pak_CreateSaveSnapshot reads 16 of the 18 bytes of g_pakSingleSaveBlockName; Libc_FormatInteger reads the tables
    D_800D5300 and D_800D5314 in de, where it is C without literals of its own)."""
    if (path, v) not in cache:
        try:
            cache[path, v] = api.preprocess(config, path, v)
        except Refusal:
            cache[path, v] = ""
    text = cache[path, v]
    out: set[str] = set()
    for n, _, _ in rows:
        names = "|".join(re.escape(m) for m, f in table.items() if f[0] == table[n][0]) or re.escape(n)
        if re.search(rf"\bextern\b[^;{{}}()]*\b(?:{names})\b\s*(?:\[[^\]]*\]\s*)*[;,]", text):
            out.add(n)
    return out


def plan(config: Config, current: Any, functions: Sequence[str]) -> list[Placement]:
    """The rows that place the read-only data of `functions` in the files that compile them, per version; raises a
    Refusal naming every reason when any function has data it may not take, or nothing to place."""

    root = config.project.root
    _INDEX.clear()  # the assembly may have changed since an earlier plan in this process
    mask = configuration.load_resource("rom.toml").get("text_mask", 0xFFFFFFFF)
    found: list[Placement] = []
    refused: list[Finding] = []
    siblings: dict[str, str] = {}
    externs: dict[tuple[str, str], str] = {}  # (file, version) -> the file's C as the version compiles it
    tails: list[tuple[int, int]] = []  # (index into found, rom where the shared piece's rest starts)
    owners: dict[
        str, set[str]
    ] = {}  # function -> the owner address of the pool piece that follows its data in a version
    nothing = {f: True for f in functions}
    for v in config.project.versions:
        table = api.symbols(config, v)
        try:
            image = configuration.rom(config.project, v).read_bytes()
        except (OSError, Refusal):
            image = b""
        pieces = _pieces(root, v)
        compiled = {
            n: m
            for m in current.modules
            if not configuration.is_assembly(m.path)
            for n in m.owns.get(v, ())
            if m.sections.get(v, {}).get(n, ".text") == ".text" and n not in m.stand_ins.get(v, ())
        }
        roms = sorted({f[1] for f in table.values() if f[1] is not None})
        spans = sorted(
            (f[0] & mask, (f[0] & mask) + size, n)
            for n, f in table.items()
            if f[3] == "func" and f[1] is not None and (size := f[2] or (_after(roms, f[1]) - f[1]))
        )

        def zero(name: str, table: Any = table, roms: Any = roms, image: Any = image) -> bool:
            at = table[name][1]
            return bool(image) and at is not None and _after(roms, at) > at and not any(image[at : _after(roms, at)])

        for function in functions:
            module = compiled.get(function)
            if module is None:
                continue  # not C in this version: its assembly keeps its data
            own = tree.LITERALS + function
            if own in table and current.owned(v, own) == module.path:
                nothing[function] = False
                if bad := _placed(root, v, module.path, own, table, roms):
                    refused.append(
                        Finding(
                            "place.object", f"{v}: {function}: {bad}", path=module.path, unit=function, versions=(v,)
                        )
                    )
                continue  # this function's data is already placed (another function's row in the file does not count)
            rows = _data_rows(table, roms, image, spans, pieces, function, _asm_refs(root, config, v, function, table))
            if rows and (extern := _extern_rows(config, v, table, rows, module.path, externs)):
                rows = [r for r in rows if r[0] not in extern]  # read through an extern: not this function's data
            if not rows:
                continue  # nothing of its own here; no version placing anything is the place.none refusal below
            nothing[function] = False
            names, vram, rom, end, inside, junk = _extent(root, current, v, table, rows, function, zero)
            # a constant other functions read, far from the data (REV1-ONE-54 D_800C913C_de), is no part of it
            if inside and (trimmed := _trim_shared(root, current, v, table, rows, function, zero)):
                rows, (names, vram, rom, end, inside, junk) = trimmed
            last = names[-1]
            if apart := _apart(table, rows):  # final: two places of the ROM, no one literals_ row can hold both
                refused.append(
                    Finding(
                        "place.apart",
                        f"{v}: {function}: {apart}: two places, not one pool, so the function keeps its stand-ins",
                        path=module.path,
                        unit=function,
                        versions=(v,),
                    )
                )
            elif (
                inside
            ):  # a single row cannot hold bytes of other data between the function's own: name them, write nothing
                refused.append(
                    Finding(
                        "place.gap",
                        f"{v}: {function}: its data spans 0x{rom:X}-0x{end:X}, which holds rows it does not read: "
                        + ", ".join(f"{n} (0x{table[n][1]:X})" for n in inside[:6])
                        + "; they are read by other code or owned by a file, so they are no junk to remove",
                        path=module.path,
                        unit=function,
                        versions=(v,),
                    )
                )
            rest = _after(roms, table[last][1] or 0)
            if (
                rest > end and image
            ):  # what follows the function's data is another function's table: not ours, no stand-in
                follow = [int.from_bytes(image[i : i + 4], "big") for i in range(end, rest - 3, 4)]
                if _foreign_table(follow, spans, function) == 0:
                    rest = end
            held = _piece_tail(pieces, rom, max(end, rest))
            if held:  # the piece the row supersedes holds more than the function's data: split it at the data's end
                tails.append((len(found), max(end, rest)))
            aka = {n: [m for m, f in table.items() if f[0] == table[n][0]] for n in names}  # second names, same bytes
            left = set(symbols.superseded_asm(root, v, table))
            shared: list[str] = []
            readers = _readers(root, v, {m for ms in aka.values() for m in ms}, f"{function}.s", left)
            for n in names:
                readers[n] = sorted({r for m in aka[n] for r in readers.get(m, ())})
                if readers[n]:  # an assembly reader links by the symbol's address; a C one declares it extern
                    seen = [f"{r} ({'C: extern' if r in compiled else 'assembly'})" for r in readers[n][:3]]
                    shared.append(f"{n} also read by {', '.join(seen)}")
            note, bad = _object(root, v, module.path, image, rom, end, vram)
            note = "; ".join([note, *shared]) if shared else note
            if bad:
                refused.append(
                    Finding("place.object", f"{v}: {function}: {bad}", path=module.path, unit=function, versions=(v,))
                )
            after, segment, split_from = "", "", ""
            split: tuple[str, ...] = ()
            if rest > end:
                after = _owner_from(current, v, table, rest, module.path)
                segment = next((n.partition("/")[0] for r, _, n in reversed(pieces) if r <= rom), "")
                if not after or not segment:
                    refused.append(
                        Finding(
                            "place.leftover",
                            f"{v}: {function}: 0x{rest - end:X} bytes lie after its data before the next row and no "
                            "other file owns the data after them to hold a stand-in",
                            path=module.path,
                            unit=function,
                            versions=(v,),
                        )
                    )
                elif split := _trailing(current, v, table, after, rom, rest):
                    if after not in siblings:
                        siblings[after] = _sibling(root, after, set(siblings.values()))
                    split_from, after = after, siblings[after]
            elif (inherited := _owner_from(current, v, table, end, module.path)) and (
                split := _trailing(current, v, table, inherited, rom, end)
            ):  # no bytes left over, but the file that owns the data after the pool also owns data before it
                if inherited not in siblings:
                    siblings[inherited] = _sibling(root, inherited, set(siblings.values()))
                split_from, after = inherited, siblings[inherited]
            found.append(
                Placement(
                    v,
                    function,
                    module.path,
                    vram,
                    rom,
                    end,
                    tuple(names),
                    note,
                    rest,
                    after,
                    segment,
                    split_from,
                    split,
                    tuple((n, table[n][1] or 0) for n in junk),
                    tuple((n, current.owned(v, n)) for n in junk if current.owned(v, n)),
                )
            )
            if (token := _next_pool(pieces, rom)) is not None:
                owners.setdefault(function, set()).add(token)
    for index, start in tails:  # split the shared piece the way a version that already splits it names it
        p = found[index]
        name = _tail_name(p, start, owners.get(p.function, ()))
        where = (
            "" if _POOL.search(name) else f" // rom:0x{start:X}"
        )  # a pool name spells its offset; any other needs it
        found[index] = replace(p, tail=f"{name} = 0x{p.vram + start - p.rom:08X};{where}")
    refused.extend(
        Finding("place.none", f"{f} has no read-only data of its own to place in any version", unit=f)
        for f, empty in nothing.items()
        if empty
    )
    if refused:
        raise Refusal(*refused)
    return found


_POOL = re.compile(r"D_auto_src_rw_pool_([0-9A-Fa-f]{8})_[0-9A-F]+_\w+$")


def _next_pool(pieces: Sequence[tuple[int, str, str]], rom: int) -> str | None:
    """The owner address in the name of the piece right after the one holding `rom` when that piece is a pool of its own
    (`D_auto_src_rw_pool_<owner>_<ROM>_<version>`: a version that already splits the function's data from the next
    function's pool), else None."""
    at = bisect.bisect_right(pieces, (rom, "\uffff")) - 1
    if at < 0 or at + 1 >= len(pieces):
        return None
    m = _POOL.search(pieces[at + 1][2])
    return m[1] if m else None


def _tail_name(p: Placement, start: int, owners: Collection[str]) -> str:
    """The name of the row that keeps the bytes after the function's data in version `p.version`: the one a version
    that already splits there gives its piece, with this version's ROM offset and name; when no version splits there
    (or two disagree) the row is named for the function and the address it starts at."""
    if len(owners) != 1:
        return f"D_auto_rodata_{p.function}_{p.vram + start - p.rom:08X}"
    return f"D_auto_src_rw_pool_{next(iter(owners))}_{start:X}_{p.version.replace('-', '_')}"


def _owner_from(current: Any, v: str, table: Mapping[str, Any], rom: int, own: str) -> str:
    """The file that owns the first data name at or after `rom` (not the function's own file): the compiled .rodata of
    a file is one section, listed where its first byte lies, so bytes that stay behind a stand-in go at the top of the
    file that starts after them, never into the file before them (the compiled data lies between)."""
    best, path = 1 << 62, ""
    for m in current.modules:
        for n in m.owns.get(v, ()):
            fact = table.get(n)
            if fact is None or fact[1] is None or fact[3] == "func" or fact[1] < rom:
                continue
            if fact[1] < best:
                best, path = fact[1], m.path
    return "" if path == own else path


def _trailing(current: Any, v: str, table: Mapping[str, Any], path: str, rom: int, rest: int) -> tuple[str, ...]:
    """The data names `path` owns in `v` at or after `rest`, when it also owns data before the table at `rom` (one
    .rodata section cannot lie on both sides of the table); () when it owns none before."""
    module = next((m for m in current.modules if m.path == path), None)
    if module is None:
        return ()
    owned = sorted(
        (fact[1], n)
        for n in module.owns.get(v, ())
        if (fact := table.get(n)) is not None and fact[1] is not None and fact[3] != "func"
    )
    if not any(r < rom for r, _ in owned):
        return ()
    return tuple(n for r, n in owned if r >= rest)


def _sibling(root: Path, path: str, taken: Collection[str]) -> str:
    """The next unused `<stem>_<n><suffix>` beside `path`."""
    base = Path(path)
    stem, n = re.sub(r"_\d+$", "", base.stem), 2
    while (name := (base.parent / f"{stem}_{n}{base.suffix}").as_posix()) in taken or (root / name).exists():
        n += 1
    return name


def _after(roms: Sequence[int], rom: int) -> int:
    at = bisect.bisect_right(roms, rom)
    return roms[at] if at < len(roms) else rom


def _foreign_table(words: Sequence[int], spans: Sequence[tuple[int, int, str]], function: str) -> int:
    """Index of the first word where two or more consecutive words are jump targets into functions other than `function`
    (another function's table), or len(words)."""
    for i in range(len(words) - 1):
        a, b = jump_target([words[i], words[i + 1]], spans), jump_target([words[i + 1], words[i + 1]], spans)
        if a is not None and a != function and a == b:
            return i
    return len(words)


def _data_rows(
    table: Mapping[str, Any],
    roms: Sequence[int],
    image: bytes,
    spans: Sequence[tuple[int, int, str]],
    pieces: Sequence[tuple[int, str, str]],
    function: str,
    read: Collection[str] | Mapping[str, int],
) -> list[tuple[str, int, int]]:
    """The rodata rows of a version that belong to `function` as (name, bytes skipped at the row's start, length): the
    ones its assembly reads and the ones whose leading words are its jump table, by address (one row per address;
    `literals_` rows never). Each is bounded to what the function owns: a row read only past an offset (`sym + 0xC`)
    starts there (the bytes before are another function's), and a pool stops where another function's jump table
    begins, so a row is never sized to the next splat row across foreign data."""
    placed = {f[1] for n, f in table.items() if n.startswith(tree.LITERALS) and f[1] is not None}
    offsets = read if isinstance(read, Mapping) else dict.fromkeys(read, 0)
    picked: dict[int, tuple[str, int, int]] = {}
    for n, fact in table.items():
        rom = fact[1]
        if rom is None or fact[3] in ("func", "label") or n.startswith(tree.LITERALS) or rom in placed:
            continue
        end = _after(roms, rom)
        skip = 0
        if n in offsets and offsets[n] > 0:
            skip = min(offsets[n] & ~3, max(end - rom - 4, 0))
        words = [int.from_bytes(image[i : i + 4], "big") for i in range(rom + skip, end - 3, 4)] if image else []
        pad = (-(fact[0] + skip) % 16) // 4  # zero words up to the 16-byte boundary a compiled jump table starts at
        if (
            n in offsets
            and 0 < pad < len(words)
            and not any(words[:pad])
            and jump_target([words[pad]] * 2, spans) == function
        ):  # alignment padding before the table is not the function's data (10-24 de row began 3 words early)
            skip += 4 * pad
            words = words[pad:]
        run = next((i for i, w in enumerate(words) if jump_target([w, w], spans) != function), len(words))
        if n in offsets and image and run * 4 == end - rom - skip and run:
            # the table runs on past the next symbol: rows cut inside a jump table (a facts symbol on one of its
            # words) are no boundary of it, whatever name they carry (J007 eu: 0x24 bytes cut at 0xC)
            more = [int.from_bytes(image[i : i + 4], "big") for i in range(end, rom + skip + TABLE_MAX, 4)]
            run += next((i for i, w in enumerate(more) if jump_target([w, w], spans) != function), len(more))
        is_table = run >= 2 and 4 * run <= TABLE_MAX
        if n in offsets and (not pieces or _is_rodata(pieces, rom)):
            size = 4 * run if is_table else min(end - rom - skip, 4 * _foreign_table(words, spans, function))
            picked.setdefault(rom, (n, skip, size))
        elif is_table and 4 * run == end - rom:
            picked.setdefault(rom, (n, 0, end - rom))
    rows: list[tuple[str, int, int]] = []
    covered = 0  # a row inside the bytes of the row before it (a symbol on a word of a jump table) is part of it
    for r in sorted(picked):
        n, skip, size = picked[r]
        if r >= covered:
            rows.append(picked[r])
            covered = r + skip + size
    return rows


def _placed(root: Path, v: str, path: str, own: str, table: Mapping[str, Any], roms: Sequence[int]) -> str:
    """The refusal when the built object of the file is newer than its source and its .rodata is not the size of the
    `literals_` placement it already has (the link would overlap the next piece or leave a hole); "" when the sizes
    agree or the object is not built after the last edit (then the link proves it)."""
    obj, src = root / "build" / v / f"{path}.o", root / path
    if not obj.is_file() or not src.is_file() or obj.stat().st_mtime < src.stat().st_mtime:
        return ""
    read = _elf(obj)
    rom = table[own][1] or 0
    placed = table[own][2] or _after(roms, rom) - rom
    measured = len(read[0]) if read else 0
    if measured == placed:
        return ""
    return (
        f"its placement {own} is 0x{placed:X} bytes but its built object's .rodata is 0x{measured:X} bytes: "
        "the whole placement must be updated (its size and the rows after it), so nothing is proved"
    )


def _object(root: Path, v: str, path: str, image: bytes, rom: int, end: int, vram: int) -> tuple[str, str]:
    """(note, refusal) from the built object of the file: its .rodata must fit the ROM bytes it takes the place of."""
    obj = root / "build" / v / f"{path}.o"
    src = root / path
    if not obj.is_file() or not src.is_file() or obj.stat().st_mtime < src.stat().st_mtime or not image:
        return "object not built after the last edit: the link proves the bytes", ""
    read = _elf(obj)
    if read is None:
        # a fresh object with no .rodata carries none of the pool: its C still reads the bytes through an extern of
        # another unit, and moving the rows broke all five ROMs (10-24 TOOL-SPEED-14, an extern read of a 4 byte pool)
        return "", (
            f"its built object has no .rodata but the ROM data it takes the place of is 0x{end - rom:X} bytes: "
            "the function reads the pool through an extern, so write the literal in its C first"
        )
    data, align, reloc = read
    size = len(data)
    if size != end - rom:
        return "", f"its compiled .rodata is 0x{size:X} bytes but the ROM data it takes the place of is 0x{end - rom:X}"
    if align > 1 and vram % min(align, 8):
        return "", f"its compiled .rodata is {align}-aligned but the data lies at 0x{vram:08X}"
    for i in range(0, size - 3, 4):
        if i not in reloc and data[i : i + 4] != image[rom + i : rom + i + 4]:
            return "", f"its compiled .rodata differs from the ROM at 0x{rom + i:X}"
    return "compiled .rodata == ROM bytes", ""


def sources(placements: Sequence[Placement], texts: Mapping[str, str]) -> dict[str, bytes]:
    """The source files that gain the stand-in of each placement's leftover bytes (`INCLUDE_RODATA`, under the
    version's macro) at the top, after the includes, once; a file with no include_asm.h gets it."""
    from unbake import padsplit  # padsplit imports repo, which imports this module

    out: dict[str, list[str]] = {}
    moving: dict[tuple[str, str], dict[str, set[str]]] = {}
    for p in placements:
        if p.split:
            names = moving.setdefault((p.split_from, p.after), {})
            for n in p.split:
                names.setdefault(n, set()).add(tree.macro(p.version))  # only this version's pieces of the name move
    for (source, sibling), names in moving.items():
        kept, new = _split(padsplit, texts.get(source, ""), names)
        out[source], out[sibling] = kept.splitlines(), new.splitlines()
    for p in placements:  # the zero pads a data file defines inside the pool: the compiled .rodata holds their bytes
        for name, path in p.pads:
            kept = "\n".join(out.get(path, texts.get(path, "").splitlines())) + "\n"
            try:
                hulls = padsplit.hulls(kept, name, {tree.macro(p.version)})
            except Refusal:
                continue  # an earlier version's pass dropped it from this file
            for start, end, _ in reversed(hulls):
                kept = (kept[:start].rstrip() + "\n" + kept[end:].lstrip("\n")).lstrip("\n")
            out[path] = _EMPTY_IF.sub("", kept).splitlines()
    for p in placements:
        if not p.leftover:
            continue
        lines = out.get(p.after, texts.get(p.after, "").splitlines())
        line = f'INCLUDE_RODATA("{p.segment}", {p.stand_in});'
        if line in lines:
            continue
        if not any("include_asm.h" in x for x in lines):
            lines.insert(0, '#include "include_asm.h"')
        at = max((i + 1 for i, x in enumerate(lines) if x.startswith("#include")), default=0)
        lines[at:at] = [f"#if defined({tree.macro(p.version)})", line, "#endif"]
        out[p.after] = lines
    return {path: ("\n".join(lines).rstrip("\n") + "\n").encode() for path, lines in out.items()}


_EMPTY_IF = re.compile(r"^[ \t]*#\s*if[^\n]*\n(?:[ \t]*\n)*[ \t]*#\s*endif[^\n]*\n?(?:[ \t]*\n)?", re.M)


def _split(padsplit: Any, text: str, names: Mapping[str, Collection[str]]) -> tuple[str, str]:
    """(`text` without `names`, a new file of its #include lines and each name with the #if lines around it): not
    padsplit.split_file's head, which keeps a data file's leading word arrays (they are definitions, not header)."""
    kept = text
    blocks: list[str] = []
    for name, macros in names.items():
        found = []
        for start, end, wrappers in reversed(padsplit.hulls(kept, name, macros)):  # last first: earlier offsets hold
            found.append("".join(w + "\n" for w in wrappers) + kept[start:end] + "\n#endif" * len(wrappers))
            kept = (kept[:start].rstrip() + "\n" + kept[end:].lstrip("\n")).lstrip("\n")
        blocks.extend(reversed(found))
    kept = _EMPTY_IF.sub("", kept)  # the #if around a moved name, now holding nothing
    includes = [line for line in text.splitlines() if line.startswith("#include")]
    return kept, "\n".join(includes) + "\n\n" + "\n".join(blocks) + "\n"


def rows(placements: Sequence[Placement], texts: Mapping[str, str]) -> dict[str, bytes]:
    """aliases.txt per version with each placement's row added (once)."""
    out: dict[str, str] = {}
    for p in placements:
        for path in p.junk_files if p.junk else ():
            text = out.get(path, texts.get(path, ""))
            gone = re.compile(r"^\s*(?:" + "|".join(re.escape(n) for n, _ in p.junk) + r")\s*=")
            kept = "".join(x for x in text.splitlines(True) if not gone.match(x))
            if kept != text:
                out[path] = kept
        for row in (p.row, p.leftover, p.tail):
            text = out.get(p.file, texts.get(p.file, ""))
            if row and row not in text.splitlines():
                out[p.file] = text + ("" if not text or text.endswith("\n") else "\n") + row + "\n"
    return {path: text.encode() for path, text in out.items()}
