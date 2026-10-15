"""Where modules live: libraries, subsystems and data owners from the call and load graph, plus cohort hints."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from unbake.contracts import digest

LABELS = ("engine_runtime", "physics_math", "rendering_graphics", "audio", "gameplay_design", "ui_menus", "data_tables")
RECIPE = "subsystems-1"


@dataclass(frozen=True)
class Entity:
    id: str
    kind: str
    name: str
    version: str = ""
    path: str = ""
    storage: tuple[int, int] | None = None
    execution: tuple[int, int] | None = None
    processor: str | None = None


@dataclass(frozen=True)
class Evidence:
    id: str
    kind: str
    subjects: tuple[str, ...]
    origin_ref: str
    input_digest: str
    correlation_key: str
    labels: tuple[str, ...] = ()
    strength: int = 0
    semantic_anchor: bool = False
    polarity: str = "support"
    # Positive semantic support flows from the first subject to consumers only.
    propagate: bool = False
    unresolved_reason: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Family:
    id: str
    seed: str
    members: tuple[str, ...]
    basis: str


@dataclass(frozen=True)
class Facts:
    entities: tuple[Entity, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    families: tuple[Family, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    # These are existing artifact identities, not timestamps or checkout paths.
    meaning_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Membership:
    entity_id: str
    support_by_label: tuple[tuple[str, int], ...]
    state: str
    primary: str | None
    tier: str
    provisional: bool
    evidence_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    conflicts: tuple[str, ...]
    calibrated_probability: float | None = None


@dataclass(frozen=True)
class Snapshot:
    schema: int
    inference_recipe: str
    meaning_key: str
    evidence_key: str
    entities: tuple[Entity, ...]
    memberships: tuple[Membership, ...]
    families: tuple[Family, ...]
    boundary_edges: tuple[Evidence, ...]
    reverse_dependencies: tuple[tuple[str, tuple[str, ...]], ...]
    missing_inputs: tuple[str, ...]

    def document(self) -> dict[str, Any]:
        return asdict(self)

    def for_subject(self, subject: str) -> tuple[Membership, ...]:
        ids = {row.id for row in self.entities if row.name == subject or row.id == subject}
        return tuple(row for row in self.memberships if row.entity_id in ids)


def canonical(facts: Facts) -> Facts:
    entities = {row.id: row for row in facts.entities}
    if len(entities) != len(facts.entities):
        raise ValueError("duplicate subsystem entity identity")
    evidence = []
    seen: set[str] = set()
    for row in sorted(facts.evidence, key=lambda row: row.id):
        if row.id in seen:
            raise ValueError("duplicate subsystem evidence identity")
        seen.add(row.id)
        if not set(row.subjects) <= entities.keys():
            raise ValueError("unbound subsystem evidence endpoint")
        if not set(row.labels) <= set(LABELS) or not 0 <= row.strength <= 8:
            raise ValueError("invalid subsystem support")
        if not row.origin_ref or not row.input_digest or not row.correlation_key:
            raise ValueError("subsystem evidence requires provenance")
        if row.polarity not in {"support", "conflict"}:
            raise ValueError("invalid subsystem evidence polarity")
        evidence.append(row)
    family_members: dict[str, set[str]] = defaultdict(set)
    family_rows = {}
    for family in sorted(facts.families, key=lambda row: row.id):
        if family.seed not in entities or not set(family.members) <= entities.keys():
            raise ValueError("unbound subsystem family")
        family_rows[family.id] = family
        family_members[family.id].update(family.members)
    families = [
        Family(key, row.seed, tuple(sorted(family_members[key])), row.basis) for key, row in sorted(family_rows.items())
    ]
    return Facts(
        tuple(entities[key] for key in sorted(entities)),
        tuple(evidence),
        tuple(families),
        tuple(sorted(set(facts.missing_inputs))),
        tuple(sorted(set(facts.meaning_refs))),
    )


def _keys(facts: Facts) -> tuple[str, str]:
    meaning = asdict(facts)
    # Provenance-only movement changes evidence context, not inferred meaning.
    for row in meaning["evidence"]:
        row.pop("origin_ref")
    meaning_key = digest([RECIPE, meaning])
    return meaning_key, digest([meaning_key, asdict(facts)])


def aggregate(facts: Facts) -> Snapshot:
    """Deterministic three-round directed propagation; tiers are uncalibrated priors."""
    facts = canonical(facts)
    ids = [row.id for row in facts.entities]
    direct = {key: dict.fromkeys(LABELS, 0) for key in ids}
    anchor_origins: dict[tuple[str, str], set[str]] = defaultdict(set)
    classes: dict[tuple[str, str], set[str]] = defaultdict(set)
    conflicts: dict[str, set[str]] = defaultdict(set)
    evidence_ids: dict[str, set[str]] = defaultdict(set)
    links: list[Evidence] = []
    dependencies: dict[str, set[str]] = defaultdict(set)
    counted: set[tuple[str, str, str]] = set()
    for row in facts.evidence:
        for key in row.subjects:
            evidence_ids[key].add(row.id)
            if row.polarity == "conflict":
                conflicts[key].add(row.id)
        if row.propagate and len(row.subjects) > 1:
            for target in row.subjects[1:]:
                dependencies[row.subjects[0]].add(target)
        if row.unresolved_reason or row.polarity == "conflict":
            continue
        for key in row.subjects:
            for label in row.labels:
                vote = key, label, row.correlation_key
                if vote in counted:
                    continue
                counted.add(vote)
                direct[key][label] = min(16000, direct[key][label] + row.strength * 1000)
                classes[key, label].add(row.kind)
                if row.semantic_anchor:
                    anchor_origins[key, label].add(row.correlation_key)
        if row.propagate and len(row.subjects) > 1:
            links.append(row)
    # Degree is per relationship class and distinct endpoint, never per observed access.
    neighbors: dict[tuple[str, str], set[str]] = defaultdict(set)
    unique_links: dict[tuple[str, str, tuple[str, ...]], Evidence] = {}
    for row in links:
        unique_links.setdefault((row.kind, row.correlation_key, row.subjects), row)
    links = list(unique_links.values())
    for row in links:
        source, *targets = row.subjects
        for target in targets:
            neighbors[row.kind, source].add(target)
            neighbors[row.kind, target].add(source)
    support = direct
    origins = anchor_origins
    traced_classes = classes
    for _ in range(3):
        contributions: dict[tuple[str, str, str], int] = defaultdict(int)
        next_origins = {key: set(value) for key, value in anchor_origins.items()}
        next_classes = {key: set(value) for key, value in classes.items()}
        for row in links:
            source, *targets = row.subjects
            for target in targets:
                degree = max(len(neighbors[row.kind, source]), len(neighbors[row.kind, target]))
                weight = 1000 * row.strength // (1 + degree)
                for label in LABELS:
                    value = weight * support[source][label] // (16000 * len(targets))
                    contributions[target, label, row.kind] += value
                    if value and origins.get((source, label)):
                        next_origins.setdefault((target, label), set()).update(origins[source, label])
                        next_classes.setdefault((target, label), set()).add(row.kind)
        inherited: dict[tuple[str, str], int] = defaultdict(int)
        for (target, label, _kind), value in contributions.items():
            inherited[target, label] += min(8000, value)
        support = {
            key: {label: min(16000, direct[key][label] + min(6000, inherited[key, label])) for label in LABELS}
            for key in ids
        }
        origins, traced_classes = next_origins, next_classes
    memberships = []
    for key in ids:
        ordered = sorted(LABELS, key=lambda label: (-support[key][label], label))
        tiers = {}
        for label in LABELS:
            has_anchor = bool(origins.get((key, label)))
            direct_anchor = bool(anchor_origins.get((key, label)))
            independent = len(traced_classes.get((key, label), ())) >= 2
            ticks = support[key][label]
            tiers[label] = (
                "high"
                if direct_anchor
                and ticks >= 8000
                and (
                    independent
                    or any(
                        row.kind == "verified_sdk"
                        and row.semantic_anchor
                        and row.strength == 8
                        and key in row.subjects
                        and label in row.labels
                        and not row.unresolved_reason
                        and row.polarity == "support"
                        for row in facts.evidence
                    )
                )
                else "medium"
                if has_anchor and ticks >= 5000 and (direct_anchor or independent)
                else "low"
                if ticks
                else "unknown"
            )
            if conflicts[key] and ticks:
                tiers[label] = "low"
        strongest, second = ordered[:2]
        top = support[key][strongest]
        medium = [label for label in ordered if tiers[label] in {"medium", "high"}]
        # Weak flat vectors never manufacture a mixed semantic claim.
        state = "mixed" if len(medium) > 1 else "assigned" if medium else "unknown"
        provisional = bool(top and 4 * (top - support[key][second]) < top)
        memberships.append(
            Membership(
                key,
                tuple((label, support[key][label] // 16) for label in LABELS),
                state,
                strongest if medium and not provisional else None,
                tiers[strongest],
                provisional,
                tuple(sorted(evidence_ids[key])),
                tuple(sorted(set().union(*(origins.get((key, label), set()) for label in LABELS)))),
                tuple(sorted(conflicts[key])),
            )
        )
    meaning_key, evidence_key = _keys(facts)
    return Snapshot(
        1,
        RECIPE,
        meaning_key,
        evidence_key,
        facts.entities,
        tuple(memberships),
        facts.families,
        tuple(row for row in facts.evidence if len(row.subjects) > 1),
        tuple((key, tuple(sorted(value))) for key, value in sorted(dependencies.items())),
        facts.missing_inputs,
    )


# ---------------------------------------------------------------- placement (TREE.md sections 1 and 2)

LEAK = 0.10  # a library region sends at most this share of its calls out
FAN_IN = 3  # and is called from outside at least this many times as often as it calls out
MIN_MODULES = 6  # a library region has at least this many modules and fewer than half the segment
DIR_MIN, DIR_MAX = 4, 32  # files per subsystem directory


@dataclass(frozen=True)
class Node:
    """One module as the placement sees it, in ROM order within its segment."""

    name: str
    segment: str
    vram: int
    base: str  # file name of the module's source
    data: bool = False
    library: str = ""  # the library the sdk catalog names, "" when it names none
    calls: tuple[str, ...] = ()  # modules its functions call, one entry per call
    loads: tuple[str, ...] = ()  # data modules its functions load


@dataclass(frozen=True)
class Placed:
    name: str
    source: str  # target source path
    header: str  # target header path, "" to leave it
    subsystem: str
    library: str
    cohort: str = ""  # advisory label, never part of the path


def regions(order: Sequence[str], calls: Mapping[str, Mapping[str, int]], size: int) -> list[tuple[int, int]]:
    """Leaf-closed runs [low, high) of a ROM-ordered module list: few calls leave, many come in, big enough."""
    index = {name: i for i, name in enumerate(order)}
    callers: dict[int, dict[int, int]] = defaultdict(dict)
    for name, row in calls.items():
        for callee, weight in row.items():
            if name in index and callee in index and callee != name:
                callers[index[callee]][index[name]] = weight
    found = []
    for low in range(len(order)):
        inside = out = fan = 0
        best = 0
        for high in range(low, len(order)):
            for callee, weight in calls.get(order[high], {}).items():
                at = index.get(callee, -1)
                if at < 0 or at == high:
                    continue
                if low <= at < high:
                    inside, fan = inside + weight, fan - weight
                else:
                    out += weight
            for caller, weight in callers[high].items():
                if low <= caller < high:
                    inside, out = inside + weight, out - weight
                else:
                    fan += weight
            total = inside + out
            big = high - low + 1 >= MIN_MODULES and 2 * (high - low + 1) < size
            if big and total and out <= LEAK * total and fan >= FAN_IN * out and fan > 0:
                best = high + 1
        if best:
            found.append((low, best))
    taken: set[int] = set()
    chosen = []
    for low, high in sorted(found, key=lambda r: (r[0] - r[1], r[0])):
        if not taken & set(range(low, high)):
            taken.update(range(low, high))
            chosen.append((low, high))
    return sorted(chosen)


def runs(count: int, weight: Mapping[tuple[int, int], float]) -> list[tuple[int, int]]:
    """Contiguous runs [low, high) of DIR_MIN..DIR_MAX modules that maximise modularity; one run when it fits."""
    if count <= DIR_MAX:
        return [(0, count)] if count else []
    degree = [0.0] * (count + 1)
    near: defaultdict[int, dict[int, float]] = defaultdict(dict)
    for (a, b), w in weight.items():
        degree[a] += w
        degree[b] += w
        near[max(a, b)][min(a, b)] = w
    prefix = [0.0]
    for d in degree[:count]:
        prefix.append(prefix[-1] + d)
    total = prefix[-1] or 1.0
    best = [float("-inf")] * (count + 1)
    cut = [0] * (count + 1)
    best[0] = 0.0
    inner: dict[tuple[int, int], float] = {}
    for low in range(count):
        held = 0.0
        for high in range(low + 1, min(count, low + DIR_MAX) + 1):
            held += sum(w for a, w in near[high - 1].items() if a >= low)
            inner[low, high] = held
    for high in range(1, count + 1):
        for low in range(max(0, high - DIR_MAX), high - DIR_MIN + 1):
            if best[low] == float("-inf"):
                continue
            share = (prefix[high] - prefix[low]) / total
            score = best[low] + 2 * inner[low, high] / total - share * share
            if score > best[high]:
                best[high], cut[high] = score, low
    if best[count] == float("-inf"):
        return [(0, count)]
    out, high = [], count
    while high:
        out.append((cut[high], high))
        high = cut[high]
    return out[::-1]


def place(nodes: Sequence[Node]) -> dict[str, Placed]:
    """The new source and header path of every module: library regions under src/sdk, game code in contiguous
    subsystems (sys_<VRAM> below the segment, which a one-segment game drops), data beside its main loader."""
    code = [n for n in nodes if not n.data]
    game_segments = sorted({n.segment for n in code})
    level = len(game_segments) > 1
    placed: dict[str, Placed] = {}
    where: dict[str, str] = {}  # module -> subsystem path
    for segment in game_segments:
        line = sorted((n for n in code if n.segment == segment), key=lambda n: (n.vram, n.name))
        calls = {n.name: Counter(n.calls) for n in line}
        library = {n.name: n.library for n in line if n.library}
        for low, high in regions([n.name for n in line], calls, len(line)):
            tag = f"lib_{line[low].vram:08X}"
            for n in line[low:high]:
                library.setdefault(n.name, tag)
        for n in line:
            if name := library.get(n.name):
                placed[n.name] = Placed(n.name, f"src/sdk/{name}/{n.base}", f"include/sdk/{name}.h", "", name)
        game = [n for n in line if n.name not in placed]
        at = {n.name: i for i, n in enumerate(game)}
        weight: defaultdict[tuple[int, int], float] = defaultdict(float)
        loaders: defaultdict[str, set[int]] = defaultdict(set)
        for n in game:
            for callee in n.calls:
                if callee in at and callee != n.name:
                    weight[tuple(sorted((at[n.name], at[callee])))] += 1  # type: ignore[index]
            for datum in n.loads:
                loaders[datum].add(at[n.name])
        for who in loaders.values():
            if 2 <= len(who) <= 8:
                for a in who:
                    for b in who:
                        if a < b:
                            weight[a, b] += 1 / (len(who) - 1)
        for low, high in runs(len(game), weight):
            sub = f"sys_{game[low].vram:08X}"
            path = f"{segment}/{sub}" if level else sub
            for n in game[low:high]:
                where[n.name] = path
                placed[n.name] = Placed(n.name, f"src/game/{path}/{n.base}", f"include/game/{path}.h", path, "")
    owner = {n.name: where[n.name] for n in code if n.name in where}
    for n in nodes:
        if not n.data:
            continue
        votes = Counter(owner[c.name] for c in code if n.name in c.loads and c.name in owner)
        top = votes.most_common(2)
        if not top:
            continue
        if len(top) == 1 or top[0][1] >= 2 * top[1][1]:
            path = top[0][0]
        else:
            path = f"{n.segment}/shared" if level else "shared"
        placed[n.name] = Placed(n.name, f"src/game/{path}/{n.base}", "", path, "")
    return placed


def cohorts(nodes: Sequence[Node]) -> dict[str, str]:
    """Advisory label per module from the call graph: the aggregate of directed call support, "" when unanchored."""
    ids = {n.name: f"module:{n.name}" for n in nodes}
    entities = tuple(Entity(ids[n.name], "module", n.name) for n in nodes)
    rows = []
    for n in nodes:
        for callee in sorted(set(n.calls) & ids.keys() - {n.name}):
            key = digest([n.name, callee])
            rows.append(
                Evidence(
                    "call:" + key,
                    "direct_call",
                    (ids[callee], ids[n.name]),
                    n.name,
                    key,
                    key,
                    strength=3,
                    propagate=True,
                )
            )
    result = aggregate(Facts(entities, tuple(rows)))
    return {row.entity_id.removeprefix("module:"): row.primary or "" for row in result.memberships}


def nodes(snapshot: Any) -> list[Node]:
    """The placement graph of a snapshot: one node per module, calls and loads read from the cached scan."""
    from bisect import bisect_right

    from unbake import infer
    from unbake.report import Segments

    lay, project = snapshot.layout, snapshot.config.project
    base = project.names_from
    scans: dict[str, Any] = {}
    for version in (base, *(v for v in snapshot.versions if v != base)):
        for row in infer.scan(snapshot).get(version, ()):
            scans.setdefault(row[0], row)
    hits = infer.sdk(snapshot)
    owner = {m: g.name for g in lay.groups.values() for m in g.members if m in lay.members}
    holder = Segments(snapshot)
    spans = sorted(
        (p.vram, p.vram + p.size, m.name)
        for m in lay.members.values()
        for p in m.placements
        if p.section != ".text" and p.version == m.reference(base)
    )
    starts = [s[0] for s in spans]

    def loaded(address: int) -> str:
        at = bisect_right(starts, address) - 1
        return spans[at][2] if at >= 0 and address < spans[at][1] else ""

    out = []
    for g in sorted(lay.groups.values(), key=lambda g: g.name):
        live = [lay.members[m] for m in g.members if m in lay.members]
        if not live:
            continue
        funcs = [m for m in live if m.kind == "function"]
        first = min(live, key=lambda m: min(p.rom_start for p in m.placements if p.version == m.reference(base)))
        vram = next(p.vram for p in first.placements if p.version == first.reference(base))
        names = [loaded(a) for m in funcs if m.name in scans for a in sorted(scans[m.name][1])]
        libraries = Counter(hits[m.name].library for m in funcs if m.name in hits and hits[m.name].library)
        out.append(
            Node(
                g.name,
                holder.of(first) or g.segment,
                vram,
                g.source.rpartition("/")[2],
                not funcs,
                libraries.most_common(1)[0][0] if libraries and 2 * sum(libraries.values()) >= len(funcs) else "",
                tuple(owner[c] for m in funcs if m.name in scans for c in sorted(scans[m.name][2]) if c in owner),
                tuple(owner[n] for n in names if n in owner),
            )
        )
    return out


def moves(snapshot: Any) -> list[dict[str, str]]:
    """The full move stream (`unbake move -` input) that restructures the tree: sources first, then headers."""
    placed, groups = place(nodes(snapshot)), snapshot.layout.groups  # nodes are the layout's modules
    sources = [
        {"op": "source", "subject": n, "target": p.source}
        for n, p in sorted(placed.items())
        if groups[n].source != p.source
    ]
    headers = [
        {"op": "header", "subject": n, "target": p.header}
        for n, p in sorted(placed.items())
        if p.header and groups[n].header != p.header
    ]
    return sources + headers
