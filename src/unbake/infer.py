"""Infer object groups and cross-version symbol addresses."""

from __future__ import annotations

import json
import re
import zlib
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from functools import cache
from importlib import resources
from itertools import groupby, pairwise
from typing import Any

import rabbitizer

from unbake import config, effort, layout, native, ownership, pool, recipes, store, symbols, versions
from unbake import evidence as readings
from unbake.contracts import (
    Finding,
    Group,
    Member,
    Placement,
    Plan,
    Refusal,
    SdkHit,
    Snapshot,
    UnitSpec,
    code_identity,
    digest,
    sha256,
)

Scan = tuple[str, frozenset[int], frozenset[str], bool, dict[str, Any]]  # name, loads, callees, padded, usage
Evidence = tuple[dict[str, int], list[tuple[str, str, str]], set[tuple[str, str]]]  # order, joins, cuts

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
_OPERANDS = re.compile(r"%(?:hi|lo|gp_rel)\((\w+)\)|\bjal\s+(\w+)|\.word\s+(\w+)")


def _placement(snapshot: Snapshot, member: Member) -> Placement:
    held = [p for p in member.placements if p.section == ".text"]
    return next((p for p in held if p.version == snapshot.config.project.names_from), held[0])


def _code(snapshot: Snapshot, p: Placement) -> list[tuple[int, rabbitizer.Instruction]]:
    blob = versions.rom_bytes(snapshot.versions[p.version], p.rom_start, p.rom_end)
    return [
        (w, rabbitizer.Instruction(w, vram=p.vram + i))
        for i in range(0, len(blob) - 3, 4)
        for w in [int.from_bytes(blob[i : i + 4], "big")]
    ]


def masked_code(code: Sequence[tuple[int, rabbitizer.Instruction]]) -> bytes:
    return b"".join(
        (
            w
            & (
                0xFFFF0000
                if ins.getOpcodeName() in _IMMEDIATE
                else 0xFC000000
                if ins.getOpcodeName() in {"jal", "j"}
                else 0xFFFFFFFF
            )
        ).to_bytes(4, "big")
        for w, ins in code
    )


def _segment(snapshot: Snapshot, p: Placement) -> str:
    return next(name for name, start, end, _ in snapshot.versions[p.version].segments if start <= p.rom_start < end)


_WIDTH = {
    "lb": 1,
    "lbu": 1,
    "sb": 1,
    "lh": 2,
    "lhu": 2,
    "sh": 2,
    "lw": 4,
    "sw": 4,
    "lwc1": 4,
    "swc1": 4,
    "ld": 8,
    "sd": 8,
    "ldc1": 8,
    "sdc1": 8,
}


def _shape(text: str) -> str:
    calls = re.findall(r"\bjal\s+([\w.$]+)", text)
    references = re.findall(r"%(?:hi|lo)\(([\w.$+]+)\)", text)
    seen: dict[str, int] = {}
    evidence = (
        calls,
        [seen.setdefault(r, len(seen)) for r in references],
        re.findall(r"[\s,](-?0x[0-9A-Fa-f]+|-?\d+)\s*(?:$|[,(])", text, re.M),
    )
    return digest((re.findall(r"^\s*/\*[^*]*\*/\s+([a-z][\w.]*)", text, re.M), evidence)) if any(evidence) else ""


def _asm(snapshot: Snapshot, version: str, name: str) -> bytes:
    path = versions.asm_path(snapshot.config, version, name).relative_to(snapshot.config.project.root)
    return snapshot.read(path.as_posix())


def _scan_job(item: tuple[Any, ...]) -> list[Scan]:
    """Per function of one chunk: rodata loads, calls, whether padding ends it, and its usage of arguments and steps."""
    snapshot, version, rows, addresses, spans = item
    gp = snapshot.versions[version].symbols.get("_gp")
    out: list[Scan] = []
    for name, p in rows:
        code = _code(snapshot, p)
        registers: dict[int, int] = {0: 0, **({28: gp} if gp is not None else {})}
        tags = {28: "full"}  # "hi" after a lui, "full" for an address built on one; read only while a register is known
        args: dict[int, tuple[int, int]] = {4 + n: (n, 0) for n in range(4)}
        indexed: dict[int, int] = {}
        stale: set[int] = set()
        pending: int | None = None
        pending_target = ""
        loads: set[int] = set()
        callees: set[str] = set()
        use: dict[str, set[Any]] = {"accesses": set(), "calls": set(), "args": set(), "steps": set()}
        for i, (w, ins) in enumerate(code):
            op, rs, rt, imm = w >> 26, (w >> 21) & 31, (w >> 16) & 31, w & 65535
            mnemonic, address, signed = ins.getOpcodeName(), None, imm if imm < 32768 else imm - 65536
            dest, width = (w >> 11) & 31 if op == 0 else rt, _WIDTH.get(mnemonic)
            store = bool(width) and mnemonic[0] == "s"
            if width and rs in registers and tags.get(rs) in {"hi", "full"}:
                high = tags[rs] == "hi"
                use["accesses"].add(
                    (
                        (registers[rs] + signed) & 0xFFFFFFFF,
                        width,
                        mnemonic,
                        "abs" if high else "based",
                        None if high else registers[rs],
                    )
                )
            elif width and rs in indexed:
                use["accesses"].add(((indexed[rs] + signed) & 0xFFFFFFFF, width, mnemonic, "indexed", indexed[rs]))
            elif width and rs in args:
                use["args"].add((args[rs][0], args[rs][1] + signed, width))
            if store and rt in registers and tags.get(rt) == "full":
                use["accesses"].add((registers[rt], 4, mnemonic, "taken", None))
            full = rs in registers and tags.get(rs) == "full"
            if mnemonic == "addiu" and rs == rt != 0 and (rs in args or full):
                use["steps"].add((f"a{args[rs][0]}" if rs in args else registers[rs], signed))
            moved: tuple[int, int] | None = None
            indexing: int | None = None
            if op == 0 and w & 63 in {0x21, 0x25}:  # addu or or: a move, or a full address plus an index register
                pairs = ((rs, rt), (rt, rs))
                moved = next((args[r] for r, o in pairs if o == 0 and r in args), None)
                indexing = next(
                    (registers[r] for r, o in pairs if tags.get(r) == "full" and r in registers and o not in registers),
                    None,
                )
            elif mnemonic == "addiu" and rs in args:
                moved = (args[rs][0], args[rs][1] + signed)
            if not (store or op in {1, 2, 3, 4, 5, 6, 7}):
                args.pop(dest, None)
                indexed.pop(dest, None)
                stale.discard(dest)
                if moved:
                    args[dest] = moved
                if indexing is not None:
                    indexed[dest] = indexing
            target = addresses.get(((p.vram + i * 4) & 0xF0000000) | ((w & 0x03FFFFFF) << 2)) if op == 3 else None
            if target not in {None, name}:
                callees.add(target)
            if mnemonic == "lui":
                registers[rt], tags[rt] = imm << 16, "hi"
            elif mnemonic in _IMMEDIATE and rs in registers:
                address = ((registers[rs] | imm) if mnemonic == "ori" else registers[rs] + signed) & 0xFFFFFFFF
                if mnemonic in {"addiu", "ori"}:
                    tags[rt] = "full" if tags.get(rs) in {"hi", "full"} else ""
                    registers[rt] = address
                elif mnemonic.startswith("l"):
                    registers.pop(rt, None)
            elif op not in {2, 3, 4, 5, 6, 7}:
                registers.pop(dest, None)
            registers[0] = 0
            if address is not None and any(a <= address < b for a, b in spans):
                loads.add(address)
            if pending == i:  # the delay slot of a call has run: the argument registers are as the callee sees them
                values = {n: registers[4 + n] for n in range(4) if 4 + n in registers and 4 + n not in stale}
                use["calls"].add((pending_target, tuple(sorted(values.items()))))
                use["accesses"].update(
                    (values[n], 0, "jal", "taken", None) for n in values if tags.get(4 + n) == "full"
                )
                stale.update(range(4, 8))
            if target is not None:
                pending, pending_target = i + 1, target
        words = [w for w, _ in code]
        zeros = len(words) - next((i for i, w in enumerate(reversed(words)) if w), len(words))
        padding = len(words) - zeros - (1 if zeros and words[zeros - 1] == 0x03E00008 else 0)
        out.append(
            (
                name,
                frozenset(loads),
                frozenset(callees),
                bool(zeros and padding > 0),
                {
                    "mask": masked_code(code),
                    "shape": _shape(_asm(snapshot, version, name).decode(errors="replace")),
                    **{key: sorted(found, key=repr) for key, found in use.items()},
                },
            )
        )
    return out


def _cap() -> int:
    """The most functions an inferred module holds (repo.toml [layout] module_cap)."""
    return int(config.load_resource("repo.toml")["layout"]["module_cap"])


def _evidence(
    snapshot: Snapshot, functions: Iterable[Member], version: str, cap: int, scanned: Sequence[Scan]
) -> Evidence:
    held = [(m, p) for m in functions for p in m.placements if p.version == version and p.section == ".text"]
    held.sort(key=lambda row: row[1].rom_start)
    order = {m.name: i for i, (m, _) in enumerate(held)}
    segments = {m.name: _segment(snapshot, p) for m, p in held}
    loads: defaultdict[int, set[str]] = defaultdict(set)
    callers: defaultdict[str, set[str]] = defaultdict(set)
    for name, addresses, callees, *_ in scanned:
        for address in addresses:
            loads[address].add(name)
        for callee in callees:
            callers[callee].add(name)
    padded = {row[0]: row[3] for row in scanned}
    joins: list[tuple[str, str, str]] = []
    cuts: set[tuple[str, str]] = set()
    for names, signal, reach in [(n, "rodata", cap) for n in loads.values() if len(n) > 1] + [
        (n | {callee}, "callee", min(cap, 16)) for callee, n in callers.items()
    ]:
        first, last = min(names, key=order.__getitem__), max(names, key=order.__getitem__)
        if len({segments[n] for n in names}) == 1 and order[last] - order[first] < reach:
            joins.append((first, last, signal))
    for (left, lp), (right, rp) in pairwise(held):
        if (
            padded[left.name]
            and lp.rom_end == rp.rom_start
            and rp.vram % 16 == 0
            and segments[left.name] == segments[right.name]
        ):
            cuts.add((left.name, right.name))
    return order, joins, cuts


def _partition(
    members: Sequence[Member], evidence: Mapping[str, Evidence], cap: int
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    index = {m.name: i for i, m in enumerate(members)}
    joined: list[set[str]] = [set() for _ in members[1:]]
    cut: list[str | None] = [None] * max(len(members) - 1, 0)
    weight, direct = [0] * len(cut), set()
    for i, (left, right) in enumerate(pairwise(members)):
        if set(right.holders()) != set(evidence) or (i == 0 and set(left.holders()) != set(evidence)):
            joined[i].add("version")
        holding = set(left.holders()) & set(right.holders())
        if holding and all(
            (left.name, right.name) in evidence[v][2] and evidence[v][0][right.name] == evidence[v][0][left.name] + 1
            for v in holding
        ):
            cut[i] = "padding"
    for _, joins, _ in evidence.values():
        cover = [0] * len(cut)
        for first, last, signal in joins:
            low, high = sorted((index.get(first, -1), index.get(last, -1)))
            if low >= 0 and not any(cut[low:high]):
                direct.add((low, high))
                for i in range(low, high):
                    joined[i].add(signal)
                    cover[i] += 1
        weight = [max(a, b) for a, b in zip(weight, cover, strict=True)]
    ends = [i for i in range(len(members)) if i == len(cut) or cut[i] or not joined[i] - {"version"}]
    pending = list(zip([0, *(e + 1 for e in ends[:-1])], ends, strict=True))
    while pending:
        low, high = pending.pop()
        if high - low + 1 <= cap or any(a <= low and high <= b for a, b in direct):
            continue
        weakest = min(range(low, high), key=lambda i: (weight[i], abs(2 * i + 1 - low - high), i))
        cut[weakest] = "chain"
        pending.extend(((low, weakest), (weakest + 1, high)))
    ahead = [1] * len(members)
    for i in range(len(cut) - 1, -1, -1):
        if not cut[i] and joined[i] - {"version"}:
            ahead[i] = ahead[i + 1] + 1
    result: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    current: list[str] = []
    signals: set[str] = set()
    atom: list[str] = []
    atom_signals: set[str] = set()
    for i, m in enumerate(members):
        atom.append(m.name)
        if (
            i < len(cut)
            and not cut[i]
            and (joined[i] - {"version"} or (joined[i] and len(current) + len(atom) + ahead[i + 1] <= cap))
        ):
            atom_signals.update(joined[i])
            continue
        if current and len(current) + len(atom) > cap:
            result.append((tuple(current), tuple(sorted(signals | {"cap"}))))
            current, signals = [], set()
        current.extend(atom)
        signals.update(atom_signals)
        atom, atom_signals = [], set()
        if i < len(cut) and (boundary := cut[i]):
            result.append((tuple(current), tuple(sorted(signals | {boundary}))))
            current, signals = [], set()
    if current:
        result.append((tuple(current), tuple(sorted(signals))))
    return result


def scan(snapshot: Snapshot) -> dict[str, list[Scan]]:
    """Per version the scan rows of its functions in ROM order: (name, loads, callees, padded, usage)."""
    functions = [m for m in snapshot.layout.members.values() if m.kind == "function"]
    held = {
        v: sorted(
            (
                (p.rom_start, m.name, p)
                for m in functions
                for p in m.placements
                if p.version == v and p.section == ".text"
            ),
            key=lambda row: row[:2],
        )
        for v in snapshot.versions
    }
    chunks: list[tuple[Any, ...]] = []
    for v, rows in held.items():
        addresses = {p.vram: name for _, name, p in rows}
        spans = [
            (p.vram, p.vram + p.rom_end - p.rom_start)
            for m in snapshot.layout.members.values()
            for p in m.placements
            if p.version == v
            and p.section == ".rodata"
            and any(r.start <= p.rom_start < r.end for r in snapshot.config.project.resident[v])
        ]
        chunks += [
            (snapshot, v, [(name, p) for _, name, p in rows[i : i + 128]], addresses, spans)
            for i in range(0, len(rows), 128)
        ]
    done = pool.map(snapshot.config, "infer.evidence", _scan_job, chunks, _scan_key)
    scanned: dict[str, list[Scan]] = defaultdict(list)
    for chunk, rows in zip(chunks, done, strict=True):
        scanned[chunk[1]].extend(rows)
    return scanned


def _edges(
    snapshot: Snapshot, run: Sequence[Member], sdk_names: Collection[str], evidence: Mapping[str, Evidence], cap: int
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """The evidence modules of a run, which is first cut where two landed units differ in toolchain or an sdk
    function meets a game function."""
    pieces, current = [], [run[0]] if run else []
    for left, right in pairwise(run):
        a, b = (layout.unit_of(snapshot, x.name) for x in (left, right))
        edge = "sdk" if (left.name in sdk_names) != (right.name in sdk_names) else ""
        edge = edge or ("toolchain" if a and b and a.toolchain != b.toolchain else "")
        if edge:
            pieces.append((current, edge))
            current = []
        current.append(right)
    if current:
        pieces.append((current, ""))
    out = []
    for piece, edge in pieces:
        modules = _partition(piece, evidence, cap)
        if edge:
            modules[-1] = (modules[-1][0], tuple(sorted({*modules[-1][1], edge})))
        out.extend(modules)
    return out


def groups(snapshot: Snapshot, identified: Collection[str] | None = None) -> tuple[Group, ...]:
    """CONTRACTS 2.4: authored and proven modules stay. The rest are cut by evidence (shared rodata, shared callees,
    version holders, padded object boundaries, the layout cap) and always at a toolchain change between two landed
    units and at an sdk edge, so no module holds sdk and game code together."""
    with effort.stage("infer.groups"):
        # a kept module keeps the rows that still exist: a name the symbol table renamed or removed leaves its list
        members = snapshot.layout.members
        kept = [
            replace(g, members=tuple(n for n in g.members if n in members))
            for g in snapshot.layout.groups.values()
            if g.evidence in {"authored", "proven"}
        ]
        excluded = {n for g in kept for n in g.members}
        functions = [m for m in snapshot.layout.members.values() if m.kind == "function"]
        cap = _cap()
        scanned = scan(snapshot)
        evidence = {v: _evidence(snapshot, functions, v, cap, scanned[v]) for v in snapshot.versions}
        source = snapshot.config.project.names_from
        sdk_names = sdk(snapshot) if identified is None else identified
        segments = {m.name: _segment(snapshot, _placement(snapshot, m)) for m in functions}
        placed = defaultdict(list)
        for m in sorted(functions, key=lambda m: (segments[m.name], _placement(snapshot, m).vram, m.name)):
            if set(m.holders()) == set(evidence):
                placed[segments[m.name]].append(m)
        line = {segment: [m.name for m in row] for segment, row in placed.items()}
        by_name = {m.name: m for m in functions}
        after: dict[str, dict[str | None, str | None]] = {}  # per segment: name -> next name, None -> head
        for segment, row in line.items():
            after[segment] = dict(zip([None, *row], [*row, None], strict=True))
        for v in [source, *(v for v in evidence if v != source)]:
            cursor: dict[str, str | None] = {}
            for m in sorted((m for m in functions if m.name in evidence[v][0]), key=lambda m: evidence[v][0][m.name]):
                segment, links = segments[m.name], after.setdefault(segments[m.name], {None: None})
                if m.name not in links:
                    before = cursor.get(segment)
                    links[m.name], links[before] = links[before], m.name  # O(1) insert after the last one seen
                cursor[segment] = m.name
        for segment, links in after.items():
            ordered, at = [], links[None]
            while at is not None:
                ordered.append(by_name[at])
                at = links[at]
            placed[segment] = ordered
        result, taken = list(kept), {g.name for g in kept}
        for segment in sorted(placed):
            for protected, run in groupby(placed[segment], key=lambda m: m.name in excluded):
                if protected:
                    continue
                for names, signals in _edges(snapshot, list(run), sdk_names, evidence, cap):
                    first = _placement(snapshot, snapshot.layout.members[names[0]])
                    base = f"code_{first.vram:08X}"  # one address can open runs in two segments: names stay unique
                    name = next(n for n in (base, *(f"{base}_{i}" for i in range(2, len(taken) + 3))) if n not in taken)
                    taken.add(name)
                    result.append(
                        Group(
                            name,
                            segment,
                            names,
                            "inferred",
                            signals,
                            False,
                            f"src/{name}.c",
                            f"include/{segment}/{name}.h",
                        )
                    )
        return tuple(result)


SDK_COMPARES = [0]  # signature rows compared with member code, for the work-count tests


@cache
def _shipped_catalog() -> tuple[dict[str, Any], str]:
    """The packaged catalog of SDK function signatures (any N64 game's libultra, libgcc, libm) and its digest."""
    node = resources.files("unbake.resources") / "data" / "sdk" / "catalog.json"
    raw = node.read_bytes()
    document = json.loads(raw)
    config.validate("sdk", document, "sdk/catalog.json")
    return document, sha256(raw)


def _catalog(snapshot: Snapshot) -> tuple[list[dict[str, Any]], str]:
    """The shipped rows plus the rows of host.sdk_catalog (it adds; a row it repeats changes nothing), with a digest."""
    shipped, shipped_digest = _shipped_catalog()
    path = snapshot.config.host.sdk_catalog
    if path is None:
        return shipped["signatures"], shipped_digest
    raw = path.read_bytes()
    document = json.loads(raw)
    config.validate("sdk", document, str(path))
    return [*shipped["signatures"], *document["signatures"]], digest((shipped_digest, sha256(raw)))


def sdk(snapshot: Snapshot) -> dict[str, SdkHit]:
    """The functions that exactly one signature row identifies, by member name."""
    rows, catalog_digest = _catalog(snapshot)
    return effort.memo(("sdk", snapshot.digest, catalog_digest), lambda: _identify(snapshot, rows))


def _identify(snapshot: Snapshot, rows: list[dict[str, Any]]) -> dict[str, SdkHit]:
    with effort.stage("infer.sdk"):
        by_crc: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        by_words: defaultdict[int, list[tuple[dict[str, Any], list[int], list[int]]]] = defaultdict(list)
        for row in rows:
            if "words" in row:
                words, masks = [int(w, 16) for w in row["words"]], [int(w, 16) for w in row["masks"]]
                if len(words) == len(masks):
                    by_words[len(words)].append((row, words, masks))
            else:
                by_crc[(row["size"], row["crc_head"])].append(row)
        found: dict[str, SdkHit] = {}
        for m in snapshot.layout.members.values():
            if m.kind != "function":
                continue
            code = _code(snapshot, _placement(snapshot, m))
            masked, matched = masked_code(code), []
            for row, words, masks in by_words.get(len(code), ()):
                SDK_COMPARES[0] += 1
                if all((w & ~mask) == (s & ~mask) for (w, _), s, mask in zip(code, words, masks, strict=True)):
                    matched.append(row)
            if len(masked) >= 8:
                head, body = zlib.crc32(masked[:8]), zlib.crc32(masked)
                for row in by_crc.get((len(masked), head), ()):
                    SDK_COMPARES[0] += 1
                    if row["crc_body"] == body:
                        matched.append(row)
            if len(matched) == 1:
                row = matched[0]
                found[m.name] = SdkHit(row["name"], row.get("module", ""), row.get("library", ""))
        return found


_PASSES = 3  # rows from claims, claims on those rows, and one pass that proves they agree


def _scan_key(item: tuple[Any, ...]) -> str:
    snapshot, version, rows, addresses, spans = item
    return digest(
        (
            snapshot.versions[version].rom_sha256,
            snapshot.versions[version].symbols.get("_gp"),
            rows,
            addresses,
            spans,
            [_asm(snapshot, version, n) for n, _ in rows],
        )
    )


def _masks(snapshot: Snapshot, wanted: Collection[str]) -> dict[str, tuple[list[str], list[bytes]]]:
    return {
        v: ([r[0] for r in rows], [r[4]["mask"] for r in rows]) for v, rows in scan(snapshot).items() if v in wanted
    }


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
        return sum((bag(0, i) & bag(1, j)).values()) * 4 / max(len(masks_a[i]), len(masks_b[j]))

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


def pairs(snapshot: Snapshot, directions: Sequence[tuple[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    """Per (a, b) direction the functions of a mapped to their counterparts in b, cached on masked code."""
    with effort.stage("infer.pairs"):
        masks = _masks(snapshot, sorted({v for direction in directions for v in direction}))
        found = pool.map(snapshot.config, "infer.align", _align, [(masks[a], masks[b]) for a, b in directions], digest)
        return dict(zip(directions, found, strict=True))


def _votes_key(job: Any) -> str:
    snapshot, a, b, rows = job
    return digest((a, b, [_asm(snapshot, v, n) for left, right in rows for v, n in ((a, left), (b, right))]))


def _votes_job(item: tuple[Any, ...]) -> Counter[tuple[str, str]]:
    snapshot, a, b, rows = item
    counts: Counter[tuple[str, str]] = Counter()
    for left, right in rows:
        operands = []
        for v, name in ((a, left), (b, right)):
            text = _asm(snapshot, v, name).decode()
            operands.append([next(x for x in match if x) for match in _OPERANDS.findall(text)])
        if len(operands[0]) == len(operands[1]):
            counts.update((x, y) for x, y in zip(*operands, strict=True) if x != y)
    return counts


def _agreed(snapshot: Snapshot, b: str, counts: Counter[tuple[str, str]]) -> dict[str, tuple[str, int, int]]:
    targets: defaultdict[str, set[str]] = defaultdict(set)
    for x, y in counts:
        targets[x].add(y)
    symbols = snapshot.versions[b].symbols
    return {
        x: (y, symbols[y], n)
        for (x, y), n in counts.items()
        if len(targets[x]) == 1 and y in symbols and (x not in symbols or (n > 1 and symbols[x] != symbols[y]))
    }


def correspondence(snapshot: Snapshot, member: str) -> tuple[tuple[str, str, str, str], ...]:
    return _correspondences(snapshot, member)


def correspondences(snapshot: Snapshot) -> tuple[tuple[str, str, str, str], ...]:
    return _correspondences(snapshot, None)


def _correspondences(snapshot: Snapshot, member: str | None) -> tuple[tuple[str, str, str, str], ...]:
    """(x, a, y, b): name x in version a's code and name y in version b for the same thing, all pairings agreeing."""
    with effort.stage("infer.correspondences"):
        source = snapshot.config.project.names_from
        directions = [(a, b) for v in snapshot.versions if v != source for a, b in ((source, v), (v, source))]
        jobs = [
            (snapshot, a, b, rows[i : i + 64])
            for (a, b), found in pairs(snapshot, directions).items()
            for rows in [[(x, y) for x, y in found.items() if member is None or member in (x, y)]]
            for i in range(0, len(rows), 64)
        ]
        total: dict[tuple[str, str], Counter[tuple[str, str]]] = defaultdict(Counter)
        parts = pool.map(snapshot.config, "infer.votes", _votes_job, jobs, _votes_key)
        for (_, a, b, _), part in zip(jobs, parts, strict=True):
            total[a, b].update(part)
        return tuple(
            sorted(
                (x, a, y, b)
                for (a, b), merged in total.items()
                for x, (y, _, _) in _agreed(snapshot, b, merged).items()
            )
        )


def _tree(snapshot: Snapshot) -> str:
    """The plan compiles every unit, so it depends on every source and header, not only on the layout and symbols."""
    root = snapshot.config.project.root
    return digest(
        [
            (str(p), p.read_bytes())
            for place in ("src", "include")
            for p in sorted((root / place).rglob("*"))
            if p.is_file()
        ]
    )


def plan(snapshot: Snapshot) -> Plan:
    with effort.stage("infer.plan"):
        key = digest((snapshot.digest, code_identity(_plan), _tree(snapshot), _catalog(snapshot)[1]))
        return store.cached_object(snapshot.config, "infer", key, lambda: _plan(snapshot))


def _neighbour_group(
    snapshot: Snapshot, unit: UnitSpec, memberships: Mapping[str, str], groups: Mapping[str, Group]
) -> str:
    """The group of the unit's members; a unit whose group is gone takes the one of the member placed just before its
    first member in the first version that holds it, else just after."""
    own = next((memberships[n] for n in unit.members if n in memberships), unit.group)
    members = snapshot.layout.members
    held = sorted(snapshot.layout.held(unit.members))
    if own in groups or not held or not unit.members:
        return own
    version = held[0]
    start = min((p.rom_start for p in members[unit.members[0]].placements if p.version == version), default=None)
    if start is None:
        return own
    rows = sorted(
        (p.rom_start, n)
        for n, m in members.items()
        if n not in unit.members
        for p in m.placements
        if p.version == version
    )
    before = [n for at, n in rows if at < start]
    after = [n for at, n in rows if at > start]
    for name in [*reversed(before[-1:]), *after[:1]]:
        group = memberships.get(name, members[name].group)
        if group in groups:
            return group
    return own


def _plan(snapshot: Snapshot) -> Plan:
    identified = sdk(snapshot)
    base, inferred = snapshot.digest, groups(snapshot, identified)
    updated = {
        g.name: replace(
            g,
            sdk=bool(g.members)
            and all(n in identified for n in g.members if snapshot.layout.members[n].kind == "function"),
        )
        for g in inferred
    }
    grouped = {n for g in updated.values() for n in g.members}
    for g in snapshot.layout.groups.values():
        data = tuple(
            n
            for n in g.members
            if n not in grouped and n in snapshot.layout.members and snapshot.layout.members[n].kind != "function"
        )
        if data:
            current = updated.get(g.name)
            updated[g.name] = replace(current or g, members=(*current.members, *data) if current else data)
    memberships = {n: g.name for g in updated.values() for n in g.members}
    # a unit owns the data it emits
    assigned, claimed, found = ownership.claims(snapshot, units=dict(snapshot.layout.units))
    cut_writes: dict[str, bytes | None] = {}
    for _ in range(_PASSES):  # its rows are generated from its claims: write them now, and plan on the rows they give,
        rows = layout.claim_rows(snapshot, claimed)  # so one plan holds both
        if not rows:
            break
        cut_writes.update(rows)
        snapshot = layout.overlay(
            snapshot, {**rows, "layout.toml": layout.dump_map(replace(snapshot.layout, units=assigned))}
        )
        assigned, claimed, found = ownership.claims(snapshot, units=dict(snapshot.layout.units))
    else:
        raise Refusal(
            Finding(
                "layout.nonconvergent",
                "the rows generated from the claims moved the claims again",
                missing=tuple(sorted(cut_writes)),
                action="report the units of the files named",
            )
        )
    if any(f.blocking for f in found):  # landed C that does not compile: nothing else is worth proving
        return Plan(
            "layout",
            base,
            {},
            (),
            tuple(f for f in found if f.blocking),
            (),
            "infer: refused",
            digest((base, "refused", found)),
        )
    members = {
        n: replace(m, group=memberships[n]) if n in memberships else m for n, m in snapshot.layout.members.items()
    }
    units, more = ownership.exact(snapshot, assigned)  # and is built only where its bytes are proven
    debt = tuple(f for f in [*found, *more] if not f.blocking)
    units = {
        path: replace(u, group=_neighbour_group(snapshot, u, memberships, updated)) for path, u in units.items()
    }  # a unit follows its members into the groups inferred now
    value = replace(
        snapshot.layout, groups=updated, members=members, units=units, digest=digest((updated, members, units))
    )
    text, writes = layout.dump_map(value), dict(cut_writes)
    if text != snapshot.read("layout.toml"):
        writes["layout.toml"] = text
    sdk_units = sum(g.sdk for g in updated.values())
    message = f"infer: {len(updated)} groups, {sdk_units} sdk units, {len(cut_writes)} files cut"
    return Plan("layout", base, writes, (), (), debt, message, digest((base, writes, message)))


def _evidence_key(item: tuple[Any, ...]) -> str:
    return native.stamp(*item)  # everything the unit's build and the ROM text it reads are


def _evidence_job(item: tuple[Any, ...]) -> dict[str, tuple[set[int], bool]]:
    snapshot, unit, version = item
    members, record = snapshot.layout.members, snapshot.versions[version]
    held = [
        p
        for n in unit.members
        if members[n].kind == "function"
        for p in members[n].placements
        if p.version == version and p.section == ".text"
    ]
    try:
        with store.work(snapshot.config) as work:
            blob = native.objects(snapshot, unit, version, recipes.resolve(snapshot.config, unit, {}), work)[
                0
            ].read_bytes()
    except Refusal:  # a unit that does not compile gives no evidence
        return {}
    first = min(held, key=lambda p: p.rom_start)
    rom = versions.rom_bytes(record, first.rom_start, max(p.rom_end for p in held))
    return readings.read(blob, rom, first.vram)


def addresses(snapshot: Snapshot, debt: Sequence[Any]) -> tuple[dict[tuple[str, str], tuple[int, bool]], list[Finding]]:
    """(name, version) -> (address, is a function) for every name whose address the table lacks or gets wrong
    where the ROM code of the units that use it agrees on one address, the exact units included. A unit's vote counts
    only if its code in every other version it holds forms the address the table gives there and the voted address
    is not another identity's in that version; the votes that fail are findings, never addresses."""
    with effort.stage("infer.addresses"):
        table = symbols.load(snapshot.read, snapshot.config.project.versions, store.content(snapshot.config).cached)
        units = snapshot.layout.units
        failing = sorted(
            {
                tuple(row.split(" ", 2)[:2])
                for f in debt
                if "symbols.unknown" in f.reason or "differ in .text" in f.reason
                for row in f.missing
            }
        )

        def votes(jobs: list[tuple[Any, ...]]) -> dict[tuple[str, str, str], tuple[set[int], bool]]:
            jobs = [
                job
                for job in jobs
                if any(
                    p.version == job[2] and p.section == ".text"
                    for n in job[1].members
                    if snapshot.layout.members[n].kind == "function"
                    for p in snapshot.layout.members[n].placements
                )
            ]
            found: dict[tuple[str, str, str], tuple[set[int], bool]] = {}
            for (_, unit, version), result in zip(
                jobs, pool.map(snapshot.config, "infer.evidence", _evidence_job, jobs, _evidence_key), strict=True
            ):
                for name, row in result.items():
                    found[unit.path, name, version] = row
            return found

        def held(unit: UnitSpec) -> list[str]:
            return sorted(snapshot.layout.held(unit.members) - set(unit.withheld))

        failed = set(failing)
        first = votes([(snapshot, units[path], version) for path, version in failing])
        # the other versions of a unit that voted are read too, so a vote can be checked against what it forms there
        beside = votes(
            [
                (snapshot, units[path], w)
                for path in sorted({p for p, _, _ in first})
                for w in held(units[path])
                if (path, w) not in failed
            ]
        )
        agreed: dict[tuple[str, str], set[int]] = defaultdict(set)
        for (path, name, version), (found, _) in first.items():
            if (path, version) in failed:
                agreed[name, version] |= found
        keys = {k for k, found in agreed.items() if len(found) == 1 and table.get(k[0], {}).get(k[1]) not in found}
        if not keys:
            return {}, []
        names = {name for name, _ in keys}
        users = [
            (snapshot, unit, version)
            for unit in units.values()
            if unit.path.endswith(".c")
            for version in held(unit)
            if (unit.path, version) not in failed
            and names & symbols.tokens(snapshot.read(unit.path).decode(errors="replace"))
        ]
        evidence = {**votes(users), **beside, **first}
        owners = {v: symbols.by_address(symbols.declared(table, v)) for v in {k[2] for k in evidence}}
        findings: list[Finding] = []
        wanted: dict[tuple[str, str], list[tuple[set[int], bool]]] = defaultdict(list)
        for (path, name, version), (found, call) in sorted(evidence.items()):
            if (name, version) not in keys:
                continue
            if len(found) == 1:
                (address,) = found
                other = [n for n in owners[version].get(address, ()) if n != name]
                contradicted = any(
                    w != version
                    and w in table.get(name, {})
                    and (path, name, w) in evidence
                    and evidence[path, name, w][0] != {table[name][w]}
                    for w in held(units[path])
                )
                if contradicted or other:
                    what = ", ".join(other) if other else name + " elsewhere"
                    findings.append(
                        Finding(
                            "symbols.misnamed",
                            f"{path} spells {name} where its {version} code forms 0x{address:X}, the address of {what}",
                            path=path,
                            unit=name,
                        )
                    )
                    continue
            wanted[name, version].append((found, call))
        result = {
            key: (next(iter(rows[0][0])), any(call for _, call in rows))
            for key, rows in sorted(wanted.items())
            if len({a for found, _ in rows for a in found}) == 1
        }
        return result, findings
