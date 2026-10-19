"""Symbol facts per version (versions/<v>/symbol_addrs.txt, splat format) and the cross-version name join.

A line reads `name = 0xVRAM; // type:func rom:0x... size:0x...`. Each version owns its file; the join gives a name
the spelling the source version (names_from) uses for the same thing, found from equal code, not from the name.
Generic names/types input adds exact-address and measured-layout evidence through the same symbol codec."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from unbake import config as configuration
from unbake import effort, pool, store
from unbake.contracts import Config, Finding, Json, Refusal, digest, sha256

Symbol = tuple[int, int | None, int | None, str]  # vram, rom, size, type ("func" or the data type)
Correspondence = tuple[str, str, str, str]  # (x, a, y, b): x in version a's code is y in version b's
KIND = "symbols.join"
_LINE = re.compile(r"^(\s*)([\w.$@]+)(\s*=\s*)(0x[0-9A-Fa-f]+|\d+)(\s*;)(?:(\s*//)(.*))?$")
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_IDENTIFIER = _TOKEN
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


def inside_data(kind: str, vram: int) -> bool:
    """A data symbol off a word boundary names a place inside the data before it: splat writes no piece that starts
    there (a piece holds whole words, the bytes up to the next word are lost), so the name is a label of that piece,
    never a piece or a stand-in of its own."""
    return kind != "func" and vram % 4 != 0


# ---------------------------------------------------------------- the facts
def path(version: str) -> str:
    return f"versions/{version}/symbol_addrs.txt"


def aliases_path(version: str) -> str:
    """The facts splat never reads: a second name at an address symbol_addrs.txt already names, or a name a C file
    defines inside bytes splat sees as one piece. The tree, the report and the link read them like symbol_addrs."""
    return f"versions/{version}/aliases.txt"


def spans(config: Config, version: str) -> tuple[tuple[int, int, int], ...]:
    """(vram start, vram end, rom start) of each loaded segment of a version: a segment ends where the next begins."""
    rows = sorted(config.project.version[version].segments, key=lambda s: s.rom)
    return tuple(
        (s.vram, s.vram + after.rom - s.rom, s.rom)
        for s, after in pairwise(rows)
        if s.vram is not None and s.kind != "bin"
    )


def merged(
    text: str,
    aliases: str,
    held: Sequence[tuple[int, int, int]] = (),
    where: tuple[str, str] = ("symbol_addrs.txt", "aliases.txt"),
) -> dict[str, Symbol]:
    """symbol_addrs.txt facts, then the aliases.txt facts of names it does not list. An alias written without a rom
    takes the one the segment holding its vram gives (none when no segment or several do)."""
    out = parse(text, where[0])
    for name, (vram, rom, size, kind) in parse(aliases, where[1]).items():
        if rom is None:
            found = [start + vram - low for low, high, start in held if low <= vram < high]
            rom = found[0] if len(found) == 1 else None
        out.setdefault(name, (vram, rom, size, kind))
    return out


def identifier(name: str, where: str) -> None:
    """Refuse a fact whose name is no C identifier. A fact names what the C names: a stand-in, then the body that
    replaces it. An assembler's spelling (a local `.L` label, `$` or `@`) is no name any source can define, so the
    piece it starts takes its default name, `func_<VRAM>` (code) or `D_<VRAM>` (data), the version suffix as the
    project spells it."""
    if not _IDENTIFIER.fullmatch(name):
        raise Refusal(
            Finding(
                "symbols.name",
                f"{name} is no C identifier: a symbol fact names what the C can name",
                path=where,
                unit=name,
                action="give the row the piece's default name (func_<VRAM> or D_<VRAM>) and its stand-in the same",
            )
        )


def parse(text: str, where: str = "symbol_addrs.txt") -> dict[str, Symbol]:
    """name -> (vram, rom, size, type). A line without `type:` is data. A row named with no C identifier is refused,
    never dropped: every reader of the file reads the same rows."""
    out: dict[str, Symbol] = {}
    for number, line in enumerate(text.splitlines(), 1):
        found = _LINE.match(line)
        if found:
            identifier(found[2], f"{where}:{number}")
            attrs = dict(a.split(":", 1) for a in (found[7] or "").split() if ":" in a)
            rom, size = (int(attrs[k], 0) if k in attrs else None for k in ("rom", "size"))
            out[found[2]] = (int(found[4], 0), rom, size, attrs.get("type", "data"))
    return out


def load(config: Config, version: str) -> dict[str, Symbol]:
    """symbol_addrs.txt and aliases.txt of one version (either may be absent)."""
    texts = []
    for name in (path(version), aliases_path(version)):
        try:
            texts.append((config.project.root / name).read_text(errors="replace"))
        except FileNotFoundError:
            texts.append("")
    return merged(texts[0], texts[1], spans(config, version), (path(version), aliases_path(version)))


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


HANDWRITTEN = "/* Handwritten function */"  # what splat writes above the extracted assembly of a handwritten function


def handwritten(config: Config, version: str, name: str) -> bool:
    """Whether the project's extracted assembly marks `name` as handwritten in `version`: its bytes are no compiler's
    output, so no C for it is judged by how closely a compiler's code agrees with them."""
    return HANDWRITTEN in _asm(config, version, name)


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


# Generic names/types input joins the same symbol facts; Recipe owns source patch application.
def _input_name(value: str) -> str:
    identifier(value, "names input")
    return value


def _shape(row: Json) -> tuple[Any, ...] | None:
    members = row.get("members", [])
    if not row.get("complete") or not members or any(m.get("size") is None for m in members):
        return None
    if any(m.get("offset_unit", "bytes") != "bytes" for m in members):
        return None
    return row["size"], tuple((m["offset"], m["size"], m["type"]) for m in members)


def names_plan(document: Json, tables: dict[str, dict[str, tuple[Any, ...]]], bindings: Json) -> Json:
    """Build deterministic symbol/type moves, preserving every unmatched fact and reason."""
    configuration.validate("symbols", document, "names input")
    if document.get("schema") != 1:
        raise Refusal(Finding("names.schema", "expected names/types schema 1"))
    for key in ("functions", "structs", "typedefs", "enums"):
        if not isinstance(document.get(key), list):
            raise Refusal(Finding("names.schema", f"{key} must be a list"))
    moves: list[Json] = []
    unmatched: list[Json] = []
    assigned: dict[str, str] = {}
    used: dict[str, str] = {}

    def add(op: str, old: str, new: str) -> None:
        if old == new:
            return
        if (old in assigned and assigned[old] != new) or (new in used and used[new] != old):
            raise Refusal(Finding("names.conflict", f"conflicting rename {old} -> {new}"))
        if old not in assigned:
            assigned[old], used[new] = new, old
            moves.append({"op": op, "subject": old, "target": new})

    for row in document["functions"]:
        name = _input_name(row["name"])
        table = tables.get(row["version"], {})
        found = [n for n, values in table.items() if values[0] == row["address"] and values[3] == "func"]
        if len(found) != 1:
            unmatched.append(
                {
                    "kind": "function",
                    "name": name,
                    "version": row["version"],
                    "address": row["address"],
                    "reason": "address has no unique function symbol",
                }
            )
            continue
        if name in table and name != found[0]:
            raise Refusal(Finding("names.collision", f"{name} already names another symbol"))
        add("symbol", found[0], name)

    accepted_tags: set[tuple[str, str]] = set()
    for row in document["structs"]:
        name = row["name"]
        candidates = [b for b in bindings.get("types", []) if b["original"] == name and b["version"] == row["version"]]
        if not _IDENTIFIER.fullmatch(name) or len(candidates) != 1 or _shape(row) is None:
            unmatched.append(
                {
                    "kind": row["kind"],
                    "name": name,
                    "version": row["version"],
                    "reason": "no unique complete measured layout binding",
                }
            )
            continue
        bound = candidates[0]
        if _shape(bound) != _shape(row):
            unmatched.append(
                {
                    "kind": row["kind"],
                    "name": name,
                    "version": row["version"],
                    "reason": "member offsets, sizes or types disagree",
                }
            )
            continue
        current = _input_name(bound["name"])
        aliases = [
            t["name"]
            for t in document["typedefs"]
            if t["version"] == row["version"] and t["type"]["tag"] == name and not t["type"]["modifiers"]
        ]
        aliases = sorted(set(aliases))
        if len(aliases) != 1:
            unmatched.append({"kind": "typedef", "name": name, "reason": "no unique typedef for bound tag"})
            continue
        alias = _input_name(aliases[0])
        accepted_tags.add((row["version"], name))
        add("type", current, alias)
        # Recipe's word-bounded type move also accepts a tag-qualified spelling.
        prefix = "union" if row["kind"] == "union_tag" else "struct"
        add("type", f"{prefix} {alias}", f"{prefix} {name}")
        for old, new in zip(bound["members"], row["members"], strict=True):
            if old["name"] != new["name"]:
                unmatched.append(
                    {
                        "kind": "field",
                        "name": new["name"],
                        "owner": name,
                        "current": old["name"],
                        "reason": "existing rename path has no scoped member operation",
                    }
                )
    unmatched.extend(
        {
            "kind": "typedef",
            "name": t["name"],
            "version": t["version"],
            "reason": "no verified current typedef identity",
        }
        for t in document["typedefs"]
        if (t["version"], t["type"]["tag"]) not in accepted_tags
    )
    unmatched.extend(
        {
            "kind": "enum",
            "name": e["name"],
            "version": e["version"],
            "reason": "no verified current enum identity; equal integer values are insufficient",
        }
        for e in document["enums"]
    )
    return {"moves": moves, "unmatched": unmatched}


def names_command(params: Json) -> Json:
    """Read a reader-neutral catalog; no project files are written."""
    from unbake.tree import LITERALS

    root = Path(params["project"])
    try:
        document = json.loads(Path(params["input"]).read_text())
        configuration.validate("symbols", document, str(params["input"]))
        bindings = json.loads(Path(params["bindings"]).read_text()) if params.get("bindings") else {"types": []}
        versions = {r["version"] for r in document["functions"]}
        tables = {v: parse((root / path(v)).read_text(), path(v)) for v in sorted(versions)}
        result = names_plan(document, tables, bindings)
        renames = {m["subject"]: m["target"] for m in result["moves"] if m["op"] == "symbol"}
        # Alias facts and compiler-owned literal pools follow the same symbol identity.
        # Reuse the symbol codec; Recipe's symbol move currently covers symbol_addrs only.
        renames.update({LITERALS + old: LITERALS + new for old, new in list(renames.items())})
        changed = {}
        for alias_file in sorted((root / "versions").glob("*/aliases.txt")):
            before = alias_file.read_text()
            after = rename_lines(before, renames)
            if after != before:
                changed[alias_file.relative_to(root).as_posix()] = after
        return {**result, "symbol_files": changed}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise Refusal(Finding("names.input", str(error))) from error
