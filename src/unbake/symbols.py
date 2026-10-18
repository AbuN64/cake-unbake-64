"""Symbol facts per version (versions/<v>/symbol_addrs.txt, splat format) and the cross-version name join.

A line reads `name = 0xVRAM; // type:func rom:0x... size:0x...`. Each version owns its file; the join gives a name
the spelling the source version (names_from) uses for the same thing, found from equal code, not from the name."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from itertools import pairwise
from typing import Any

from unbake import effort, pool, store
from unbake.contracts import Config, Finding, Refusal, digest, sha256

Symbol = tuple[int, int | None, int | None, str]  # vram, rom, size, type ("func" or the data type)
Correspondence = tuple[str, str, str, str]  # (x, a, y, b): x in version a's code is y in version b's
KIND = "symbols.join"
_LINE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(0x[0-9A-Fa-f]+|\d+)(\s*;)(?:(\s*//)(.*))?$")
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WORD = re.compile(r"/\*[^*]*?\b([0-9A-Fa-f]{8})\s*\*/\s+([a-z][\w.]*)")
_OPERANDS = re.compile(r"%(?:hi|lo|gp_rel)\((\w+)\)|\bjal\s+(\w+)|\.word\s+(\w+)")
_IMMEDIATE = frozenset(
    [
        "lui",
        "addiu",
        "ori",
        "lw",
        "sw",
        "lh",
        "lhu",
        "sh",
        "lb",
        "lbu",
        "sb",
        "lwc1",
        "swc1",
        "ldc1",
        "sdc1",
        "ld",
        "sd",
    ]
)


# ---------------------------------------------------------------- the facts
def path(version: str) -> str:
    return f"versions/{version}/symbol_addrs.txt"


def parse(text: str) -> dict[str, Symbol]:
    """name -> (vram, rom, size, type). A line without `type:` is data."""
    out: dict[str, Symbol] = {}
    for line in text.splitlines():
        found = _LINE.match(line)
        if found:
            attrs = dict(a.split(":", 1) for a in (found[7] or "").split() if ":" in a)
            rom, size = (int(attrs[k], 0) if k in attrs else None for k in ("rom", "size"))
            out[found[2]] = (int(found[4], 0), rom, size, attrs.get("type", "data"))
    return out


def load(config: Config, version: str) -> dict[str, Symbol]:
    try:
        return parse((config.project.root / path(version)).read_text(errors="replace"))
    except FileNotFoundError:
        return {}


def segment_at(config: Config, version: str, rom: int) -> str | None:
    """The segment of the version whose rom range holds `rom`: a segment ends where the next row begins."""
    project: Any = config.project
    rows = sorted((s.rom, s.name) for s in project.version[version].segments)
    held = [name for start, name in rows if start <= rom]
    return held[-1] if held else None


def asm_path(config: Config, version: str, name: str) -> str:
    """Project-relative path of a symbol's extracted assembly: nonmatchings/ for code, data/ for data."""
    row = load(config, version).get(name)
    if row is None or row[1] is None:
        raise Refusal(
            Finding("symbols.unknown", f"{name} has no rom address in {version}", unit=name, versions=(version,))
        )
    segment = segment_at(config, version, row[1])
    if segment is None:
        raise Refusal(
            Finding("tree.segments", f"{name} lies in no segment of {version}", unit=name, versions=(version,))
        )
    return f"asm/{version}/{'nonmatchings' if row[3] == 'func' else 'data'}/{segment}/{name}.s"


def by_address(table: Mapping[str, int]) -> defaultdict[int, list[str]]:
    names: defaultdict[int, list[str]] = defaultdict(list)
    for name, address in table.items():
        names[address].append(name)
    return names


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text))


def rewrite(text: str, old: str, new: str) -> str:
    """text with every token equal to old replaced; names inside longer identifiers are left alone."""
    return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, text)


def rename_lines(text: str, renames: Mapping[str, str]) -> str:
    """The symbol file with the names of the lines renamed; every other byte is kept."""
    out = []
    for line in text.splitlines(keepends=True):
        found = _LINE.match(line.rstrip("\n"))
        if found and found[2] in renames:
            line = line.replace(found[2], renames[found[2]], 1)
        out.append(line)
    return "".join(out)


# ---------------------------------------------------------------- the join
def _asm(config: Config, version: str, name: str) -> str:
    try:
        return (config.project.root / asm_path(config, version, name)).read_text(errors="replace")
    except FileNotFoundError:
        return ""


def _mask(text: str) -> bytes:
    """The code of an asm file with immediates and jump targets cleared: equal in versions that move data."""
    words = []
    for word, op in _WORD.findall(text):
        value = int(word, 16)
        value &= 0xFFFF0000 if op in _IMMEDIATE else 0xFC000000 if op in {"j", "jal"} else 0xFFFFFFFF
        words.append(value.to_bytes(4, "big"))
    return b"".join(words)


def _align(item: tuple[Any, ...]) -> dict[str, str]:
    """Names of one version mapped to the other's: unique equal masked code anchors, the rest pair between."""
    (names_a, masks_a), (names_b, masks_b) = item
    index: list[defaultdict[bytes, list[int]]] = [defaultdict(list), defaultdict(list)]
    for table, masks in zip(index, (masks_a, masks_b), strict=True):
        for i, mask in enumerate(masks):
            table[mask].append(i)
    matched = dict(
        sorted((x[0], index[1][key][0]) for key, x in index[0].items() if len(x) == 1 == len(index[1].get(key, ())))
    )
    masks = (masks_a, masks_b)
    bags: dict[tuple[int, int], Counter[bytes]] = {}

    def bag(side: int, i: int) -> Counter[bytes]:
        if (side, i) not in bags:
            bags[side, i] = Counter(masks[side][i][o : o + 4] for o in range(0, len(masks[side][i]), 4))
        return bags[side, i]

    def share(i: int, j: int) -> float:
        return sum((bag(0, i) & bag(1, j)).values()) * 4 / max(len(masks_a[i]), len(masks_b[j]), 1)

    for (la, lb), (ra, rb) in pairwise([(-1, -1), *matched.items(), (len(masks_a), len(masks_b))]):
        floor = lb
        for i in range(la + 1, ra):
            scores = sorted(((share(i, j), -j) for j in range(floor + 1, rb)), reverse=True)
            if (
                scores
                and scores[0][0] >= 0.6
                and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.15 or masks_b[-scores[0][1]] == masks_a[i])
            ):
                floor = matched[i] = -scores[0][1]
    return {names_a[i]: names_b[j] for i, j in matched.items()}


def _votes_job(item: tuple[Any, ...]) -> Counter[tuple[str, str]]:
    """Operand names the aligned functions use at the same place: how often x (in a) meets y (in b)."""
    rows = item
    counts: Counter[tuple[str, str]] = Counter()
    for left, right in rows:
        operands = [[next(x for x in m if x) for m in _OPERANDS.findall(text)] for text in (left, right)]
        if len(operands[0]) == len(operands[1]):
            counts.update((x, y) for x, y in zip(*operands, strict=True) if x != y)
    return counts


def _agreed(table: Mapping[str, Symbol], counts: Counter[tuple[str, str]]) -> dict[str, tuple[str, int]]:
    targets: defaultdict[str, set[str]] = defaultdict(set)
    for x, y in counts:
        targets[x].add(y)
    return {
        x: (y, n)
        for (x, y), n in counts.items()
        if len(targets[x]) == 1 and y in table and (x not in table or (n > 1 and table[x][0] != table[y][0]))
    }


def _functions(table: Mapping[str, Symbol]) -> list[str]:
    return [n for _, n in sorted((s[1], n) for n, s in table.items() if s[3] == "func" and s[1] is not None)]


def correspondences(config: Config) -> tuple[Correspondence, ...]:
    """(x, a, y, b): name x in version a's code and name y in version b for the same thing, all pairings agreeing.
    The functions pair by masked code; the names their operands use then vote. Cached on the asm content."""
    with effort.stage("symbols.correspondences"):
        project: Any = config.project
        versions: tuple[str, ...] = tuple(project.versions)
        source = project.names_from
        tables = {v: load(config, v) for v in versions}
        names = {v: _functions(tables[v]) for v in versions}
        texts = {v: {n: _asm(config, v, n) for n in names[v]} for v in versions}
        key = digest(
            [
                source,
                [
                    (v, sorted((n, sha256(t.encode())) for n, t in texts[v].items()), sorted(tables[v].items()))
                    for v in versions
                ],
            ]
        )

        def produce() -> bytes:
            return json.dumps(_correspondences(config, source, versions, tables, names, texts)).encode()

        rows = json.loads(store.cached(config, KIND, key, produce))
        return tuple((x, a, y, b) for x, a, y, b in rows)


def _correspondences(
    config: Config,
    source: str,
    versions: Sequence[str],
    tables: Mapping[str, Mapping[str, Symbol]],
    names: Mapping[str, list[str]],
    texts: Mapping[str, Mapping[str, str]],
) -> list[Correspondence]:
    directions = [(a, b) for v in versions if v != source for a, b in ((source, v), (v, source))]
    masks = {v: (names[v], [_mask(texts[v][n]) for n in names[v]]) for v in versions}
    found = pool.map(
        config, "symbols.align", _align, [(masks[a], masks[b]) for a, b in directions], lambda item: digest(item)
    )
    jobs, owners = [], []
    for (a, b), pairs in zip(directions, found, strict=True):
        rows = [(texts[a][x], texts[b][y]) for x, y in pairs.items()]
        for i in range(0, len(rows), 64):
            jobs.append(rows[i : i + 64])
            owners.append((a, b))
    parts = pool.map(config, "symbols.votes", _votes_job, jobs, lambda item: digest(item))
    total: dict[tuple[str, str], Counter[tuple[str, str]]] = defaultdict(Counter)
    for owner, part in zip(owners, parts, strict=True):
        total[owner].update(part)
    return sorted((x, a, y, b) for (a, b), merged in total.items() for x, (y, _) in _agreed(tables[b], merged).items())


def join(config: Config) -> tuple[Correspondence, ...]:
    """Rename, in every other version's symbol_addrs.txt, the names the source version spells otherwise for the same
    thing: x of the source code is y in version b, x is a symbol of the source and y one of b, b has no x. Returns
    the renames made as (x, source, y, b)."""
    project: Any = config.project
    source = project.names_from
    with effort.stage("symbols.join"):
        tables = {v: load(config, v) for v in project.versions}
        made: list[Correspondence] = []
        for x, a, y, b in correspondences(config):
            if a == source and x in tables[a] and y in tables[b] and x not in tables[b] and x != y:
                made.append((x, a, y, b))
        for b in sorted({m[3] for m in made}):
            renames = {y: x for x, _, y, v in made if v == b}
            file = config.project.root / path(b)
            file.write_text(rename_lines(file.read_text(), renames))
        return tuple(made)


def referenced(text: str, table: Collection[str]) -> set[str]:
    """The table names a source text uses."""
    return tokens(text) & set(table)
