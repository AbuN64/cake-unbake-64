"""Blobs: the hand-kept facts about ROM content that is not code or data (`blobs.toml`), and the ROM byte partition.

A blob records game content or an explicit accounting classification with a physical ROM range.
The file holds facts only (ranges,
formats, states, evidence by sha256), never bytes. `sweep` classifies every ROM byte exactly once."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from unbake import config as configuration
from unbake import effort, journal, rnc, store
from unbake.contracts import Config, Finding, Json, Plan, Refusal, digest, sha256, sha256_file

Spans = list[tuple[int, int]]
LEFTOVER_FORMATS = (
    "sn-debug",
    "earlier-link",
    "copy",
    "mips-stream",
    "rnc-repack",
)  # what a `leftover` row (build residue read by a native reader) may be
CONTENT = (
    "graphics",
    "model",
    "audio",
    "world",
    "text",
)  # the types the published map carries; accounting rows are not content
RUN_MIN = 16  # the shortest one-value run that proves padding inside a gap of other bytes
_ORDER = (
    "name",
    "aliases",
    "type",
    "format",
    "codec",
    "state",
    "limit",
    "stored_size",
    "decoded_size",
    "parent",
    "decoded",
)


def path_of(config: Config) -> str:
    """The repo-relative path of the blobs file: config.toml [paths] blobs over repo.toml's."""
    return str(configuration.paths(config.project)["blobs"])


def _read(config: Config) -> bytes | None:
    path = Path(config.project.root) / path_of(config)
    return path.read_bytes() if path.is_file() else None


def _rom_size(config: Config, version: str) -> int | None:
    path = Path(config.project.root) / "roms" / f"baserom.{version}.z64"
    return path.stat().st_size if path.is_file() else None


def sha(config: Config) -> str:
    """The sha256 of blobs.toml, empty when the project has none: the key every blob-derived cache carries."""
    data = _read(config)
    return sha256(data) if data is not None else ""


def parse(config: Config, data: bytes, file: str | None = None) -> Json:
    """The checked document of blobs.toml bytes: schema, parents and ranges are verified, every failure blocking."""
    file = file or path_of(config)
    try:
        document = configuration.toml("blobs", data, file, "blobs.schema")
    except Refusal as refusal:
        raise Refusal(*(Finding("blobs.schema", f.reason, path=f.path) for f in refusal.findings)) from None
    found = check(config, document, file)
    if found:
        raise Refusal(*found)
    return document


def load(config: Config) -> Json:
    """The project's blobs document (`{"schema": 1, "blob": {...}}`), once per content; no file means no blobs."""
    data = _read(config)
    if data is None:
        return {"schema": 1, "blob": {}}
    return effort.memo(("blobs", sha256(data), tuple(config.project.version)), lambda: parse(config, data))


def check(config: Config, document: Json, file: str | None = None) -> tuple[Finding, ...]:
    """Blocking findings of a document: a parent that is missing or cyclic, a child with ROM bytes of its own, a range
    that is empty, outside the ROM or in an unknown version."""
    file = file or path_of(config)
    found: list[Finding] = []
    blob = document.get("blob", {})
    for name, row in blob.items():
        parent = row.get("parent")
        if parent is not None:
            seen, at = {name}, parent
            while at in blob and at not in seen:
                seen.add(at)
                at = blob[at].get("parent")
            if parent not in blob or at in seen:
                found.append(Finding("blobs.parent", f"blob {name} has a missing or cyclic parent {parent}", path=file))
            if row.get("rom") or "decoded" not in row:
                found.append(
                    Finding("blobs.schema", f"blob {name} lives in its parent: it needs decoded, no rom", path=file)
                )
        if row.get("type") == "leftover" and row.get("format") not in LEFTOVER_FORMATS:
            found.append(
                Finding(
                    "blobs.schema",
                    f"leftover blob {name} needs sn-debug, earlier-link, copy, mips-stream or rnc-repack",
                    path=file,
                )
            )
        if row.get("format") == "pad" or "limit" in row:
            found += _pad_findings(name, row, file)
        for v, spans in row.get("rom", {}).items():
            size = _rom_size(config, v) if v in config.project.version else None
            if v not in config.project.version:
                found.append(
                    Finding("blobs.range", f"blob {name} names the unknown version {v}", path=file, versions=(v,))
                )
            for start, end in spans:
                if start >= end or (size is not None and end > size):
                    found.append(
                        Finding(
                            "blobs.range",
                            f"blob {name} range {start:#x}..{end:#x} is empty or outside the ROM",
                            path=file,
                            versions=(v,),
                        )
                    )
    return tuple(found)


def _pad_findings(name: str, row: Json, file: str) -> list[Finding]:
    """A pad row is a decoded fill with a `limit`, and no range of it is longer than that limit."""
    if row["type"] != "fill" or row.get("format") != "pad" or row["state"] != "decoded" or "limit" not in row:
        return [
            Finding("blobs.schema", f"blob {name} needs type fill, format pad, state decoded and a limit", path=file)
        ]
    return [
        Finding(
            "blobs.range",
            f"blob {name} range {a:#x}..{b:#x} is over its limit {row['limit']}",
            path=file,
            versions=(v,),
        )
        for v, spans in row.get("rom", {}).items()
        for a, b in spans
        if b - a > row["limit"]
    ]


def ranges(document: Json, version: str) -> tuple[Spans, Spans]:
    """(decoded, identified) physical ranges of one version. A blob in the `observed` state, and a blob inside
    its parent's stream, adds no bytes."""
    out: dict[str, Spans] = {"decoded": [], "identified": []}
    for row in document.get("blob", {}).values():
        if row["state"] in out and "parent" not in row:
            out[row["state"]] += [(a, b) for a, b in row.get("rom", {}).get(version, ())]
    return out["decoded"], out["identified"]


def no_row(partition: Json, document: Json, version: str) -> int:
    """Physical bytes without a row, separate from undecoded or unidentified bytes.

    Code, data, platform and automatic fill need no blob row. Every physical blob
    row, including observed accounting rows, records its range without granting
    decoded or identified credit. Compressed children have no physical range.
    """
    held = [tuple(span) for spans in partition["spans"].values() for span in spans]
    for row in document.get("blob", {}).values():
        held.extend(tuple(span) for span in row.get("rom", {}).get(version, ()))
    return length(subtract([(0, partition["rom_size"])], merge(held)))


def census(document: Json, version: str | None = None) -> Json:
    """Blobs counted by type and state; with a version, only the blobs located in it (their children included)."""
    blob = document.get("blob", {})

    def located(row: Json) -> bool:
        return (
            version is None
            or version in row.get("rom", {})
            or (row.get("parent") in blob and located(blob[row["parent"]]))
        )

    counts: dict[str, Counter[str]] = {}
    for row in blob.values():
        if located(row):
            counts.setdefault(row["type"], Counter())[row["state"]] += 1
    return {t: dict(sorted(c.items())) for t, c in sorted(counts.items())}


def rows(config: Config) -> list[Json]:
    """Every blob as a row with its id, sorted by id."""
    return [{"id": name, **row} for name, row in sorted(load(config).get("blob", {}).items())]


# ---------------------------------------------------------------- the byte partition
def merge(spans: Iterable[tuple[int, int]]) -> Spans:
    """Sorted, disjoint, touching ranges joined."""
    out: Spans = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def subtract(whole: Spans, cut: Spans) -> Spans:
    """`whole` without `cut` (both merged)."""
    out: Spans = []
    at = 0
    for start, end in whole:
        while at < len(cut) and cut[at][1] <= start:
            at += 1
        low, i = start, at
        while i < len(cut) and cut[i][0] < end:
            if cut[i][0] > low:
                out.append((low, cut[i][0]))
            low = max(low, cut[i][1])
            i += 1
        if low < end:
            out.append((low, end))
    return out


def length(spans: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in spans)


def fill(rom: bytes, gaps: Spans) -> Spans:
    """The proven padding inside the unknown `gaps` of `rom` (owner decision B1: credited matched, no sixth type).
    Proof is by layout, never by content guess. A gap is split into its maximal one-value runs, read from the gap's
    end backwards. A run is padding when it reaches the end of the ROM (cart padding, credited from the 16-byte
    boundary at or above its start unless it is the whole gap), or when it ends where the next item's alignment
    puts it (its length is below the largest power of two dividing its end) and that end is the gap's end or the
    start of the padding run after it. A gap that is one run is the whole-gap rule; a run inside a longer gap must
    be at least `RUN_MIN` bytes, so that a last byte equal to its neighbour by chance proves nothing. The build
    emits these bytes like any other, so `make check` proves them with the rest of the ROM."""
    out: Spans = []
    size = len(rom)
    for start, end in gaps:
        runs = [(start + m.start(), start + m.end()) for m in re.finditer(rb"(.)\1*", rom[start:end], re.S)]
        following = end  # the end a run must meet: the gap's end, then the start of the padding run after it
        for low, high in reversed(runs):
            if high != following or (len(runs) > 1 and high - low < RUN_MIN):
                break
            if high == size:
                low = low if len(runs) == 1 else max(low, (low + 15) & ~15)
                if low >= high:
                    break
            elif high - low >= (high & -high):
                break
            out.append((low, high))
            following = low
    return sorted(out)


def sweep(
    rom_size: int,
    code: Spans,
    data: Spans,
    platform: Spans,
    decoded: Spans,
    identified: Spans,
    rom: bytes | None = None,
) -> Json:
    """Classify every byte of `[0, rom_size)` once, in this precedence: platform, code, data, decoded blobs,
    identified blobs, proven fill (with the ROM bytes, see `fill`), unknown. Identified platform bytes (boot, SDK
    microcode) are platform in every version, whatever symbol covers them: a symbol over them is no credit of ours
    (`platform_overlap` counts those bytes; the report names the symbols). Fill counts as matched blob bytes: it is
    inside `blobs` and `blobs_decoded`, and alone in `blobs_fill`. `overlap` is credited bytes counted twice,
    `outside` ranges beyond the ROM; both must be 0. `spans` holds the disjoint ranges of each class for drawing."""

    def clamp(spans: Spans) -> Spans:
        return [(max(0, a), min(b, rom_size)) for a, b in spans if min(b, rom_size) > max(0, a)]

    raw = code + data
    kept = clamp(raw)
    platform_m = merge(clamp(platform))
    off = [part for span in kept for part in subtract([span], platform_m)]  # each credited span off the platform
    code_m = subtract(merge(clamp(code)), platform_m)
    data_m = subtract(subtract(merge(clamp(data)), code_m), platform_m)
    credit = merge(code_m + data_m)
    held = merge(credit + platform_m)
    dec = subtract(merge(clamp(decoded)), held)
    ide = subtract(subtract(merge(clamp(identified)), held), dec)
    every = merge(clamp(decoded) + clamp(identified))
    pad = fill(rom, subtract([(0, rom_size)], merge(held + dec + ide))) if rom is not None else []
    parts = {
        "rom_size": rom_size,
        "code": length(code_m),
        "data": length(data_m),
        "platform": length(platform_m),
        "blobs_decoded": length(dec) + length(pad),
        "blobs_identified": length(ide),
        "blobs_fill": length(pad),
    }
    parts["blobs"] = parts["blobs_decoded"] + parts["blobs_identified"]
    parts["unknown"] = rom_size - parts["code"] - parts["data"] - parts["platform"] - parts["blobs"]
    return {
        **parts,
        "overlap": length(off) - length(credit),
        "platform_overlap": length(merge(kept)) - length(subtract(merge(kept), platform_m)),
        "outside": length(raw) - length(kept),
        "shadowed": length(every) - length(subtract(every, credit)),
        "spans": {
            "code": code_m,
            "data": data_m,
            "platform": platform_m,
            "decoded": dec,
            "identified": ide,
            "fill": pad,
        },
    }


# ---------------------------------------------------------------- writing blobs.toml
def _int(value: int) -> str:
    return f"{value:#x}"


def _value(key: str, value: object) -> str:
    if key == "aliases":
        assert isinstance(value, list)
        items = ", ".join(
            f"{{ label = {json.dumps(a['label'])}, semantic = {str(a['semantic']).lower()} }}" for a in value
        )
        return f"[{items}]"
    if key == "decoded":
        assert isinstance(value, list)
        return f"[{_int(value[0])}, {_int(value[1])}]"
    return json.dumps(value)


def emit(document: Json) -> bytes:
    """blobs.toml as deterministic text: ids sorted, keys in a fixed order, ROM offsets in hex."""
    lines = ["schema = 1"]
    for name, row in sorted(document.get("blob", {}).items()):
        lines += ["", f"[blob.{json.dumps(name)}]"]
        lines += [f"{k} = {_value(k, row[k])}" for k in _ORDER if k in row]
        for v, spans in sorted(row.get("rom", {}).items()):
            lines.append(f"rom.{json.dumps(v)} = [" + ", ".join(f"[{_int(a)}, {_int(b)}]" for a, b in spans) + "]")
        if row.get("evidence"):
            items = [
                "{ "
                + ", ".join(f"{k} = {json.dumps(e[k])}" for k in ("artifact", "sha256", "pointer") if k in e)
                + " }"
                for e in row["evidence"]
            ]
            lines.append("evidence = [" + ", ".join(items) + "]")
    return ("\n".join(lines) + "\n").encode()


# ---------------------------------------------------------------- import
@dataclass
class _Reads:
    config: Config
    reads: dict[str, str | None] = field(default_factory=dict)


def _rom_sha(config: Config, version: str) -> str:
    path = Path(config.project.root) / "roms" / f"baserom.{version}.z64"
    return effort.memo(("rom-sha256", str(path)), lambda: sha256_file(path))


def superseded_ids(document: Json, packet: Json) -> list[str]:
    """The rows of `document` that the packet's `supersedes` removes: only ids still present (a re-landed packet finds
    them gone already, so landing stays idempotent)."""
    return [n for n in packet.get("supersedes", ()) if n in document["blob"]]


def supersede_findings(document: Json, packet: Json, superseded: list[str], file: str) -> list[Finding]:
    """Blocking findings for a packet's `supersedes`. Only an observed row (an accounting claim that a span has a row,
    D8) can be replaced; the packet must not also write that id; and in every version the packet's own credited rows
    (decoded or identified, not children) must cover every byte of the row, so no byte loses its row and none is both
    observed and decoded."""
    found: list[Finding] = []
    for name in packet.get("supersedes", ()):
        if name in packet["blob"]:
            found.append(
                Finding("blobs.supersede", f"blob {name} is both superseded and written by the packet", path=file)
            )
    covers = {
        v: merge(
            [
                span
                for row in packet["blob"].values()
                if row["state"] != "observed"
                for span in row.get("rom", {}).get(v, ())
            ]
        )
        for v in {v for n in superseded for v in document["blob"][n].get("rom", {})}
    }
    for name in superseded:
        row = document["blob"][name]
        if row["state"] != "observed":
            found.append(
                Finding(
                    "blobs.supersede",
                    f"blob {name} is {row['state']}, only an observed row can be superseded",
                    path=file,
                )
            )
            continue
        # Retired unclassified observations carry no credit or semantic ownership.
        # Explicit removal leaves uncovered bytes unknown; it must not force a fake type.
        if row.get("type") == "unclassified":
            continue
        for v, spans in sorted(row.get("rom", {}).items()):
            loose = length(subtract(merge([tuple(s) for s in spans]), covers[v]))
            if loose:
                found.append(
                    Finding(
                        "blobs.supersede",
                        f"blob {name} in {v}: {loose} bytes are not covered by the packet's decoded or identified rows",
                        path=file,
                        versions=(v,),
                    )
                )
    return found


def import_plan(config: Config, packet: Json) -> Plan:
    """The reviewed plan that merges an evidence packet into blobs.toml: the packet's ROM and artifact sha256 must match
    what the project holds and what each row cites; a row already present is replaced by the packet's."""
    file = path_of(config)
    configuration.validate("blobs.packet", packet, "packet")
    old = _read(config)
    document = load(config)
    blocking: list[Finding] = []
    for v, want in packet["rom_sha256"].items():
        if v not in config.project.version or _rom_size(config, v) is None or _rom_sha(config, v) != want:
            blocking.append(
                Finding("blobs.evidence", f"the packet's ROM sha256 for {v} is not this project's ROM", versions=(v,))
            )
    pins = packet.get("artifacts", {})
    for name, row in packet["blob"].items():
        for item in row.get("evidence", ()):
            if pins.get(item["artifact"]) != item["sha256"]:
                blocking.append(
                    Finding(
                        "blobs.evidence",
                        f"blob {name} cites {item['artifact']} with a sha256 the packet does not pin",
                        path=file,
                    )
                )
    superseded = superseded_ids(document, packet)
    blocking += supersede_findings(document, packet, superseded, file)
    kept = {n: row for n, row in document["blob"].items() if n not in superseded}
    merged = {"schema": 1, "blob": {**kept, **packet["blob"]}}
    blocking += check(config, merged)
    blocking += _overlap_findings(merged, document, file)
    writes = {file: emit(merged)}
    added = sum(n not in document["blob"] for n in packet["blob"])
    message = f"Add {added} blobs to blobs.toml" if added else "Update blobs.toml"
    if superseded:
        message += f", superseding {len(superseded)} observed rows"
    body = ("blobs", sha256(old) if old is not None else "", writes, (), tuple(blocking), (), message)
    return Plan(*body, digest(body))


def apply_import(config: Config, plan: Plan) -> Json:
    """Write a plan (never Git), read it back and name the git add/commit that records it."""
    file = path_of(config)
    old = _read(config)
    reads = _Reads(config, {file: sha256(old) if old is not None else None})
    with store.exclusive(config, "structure", wait=True):
        files = journal.apply(config, plan, reads)
    return {
        "files": list(files),
        "message": plan.message,
        "add": "git add " + " ".join(files),
        "commit": f'git commit -m "{plan.message}"',
    }


# ---------------------------------------------------------------- land
_KEPT = ("rom_size", "code", "data", "platform", "blobs", "blobs_decoded", "blobs_identified", "unknown")


def _partitions(config: Config) -> Json:
    tree = import_module("unbake.tree").capture(config)
    parts = import_module("unbake.report").partitions(config, tree)
    return {v: {k: p[k] for k in (*_KEPT, "overlap", "outside")} for v, p in parts.items()}


def _counts(have: Json, packet: Json) -> Json:
    return {
        "added": sum(n not in have for n in packet["blob"]),
        "unchanged": sum(have.get(n) == r for n, r in packet["blob"].items()),
        "superseded": len(superseded_ids({"blob": have}, packet)),
    }


def land(config: Config, packet: Json) -> Json:
    """Land a facts-only packet on the current tree: import it, read the file back, audit the source rules against
    HEAD and check the partition of every version, no ROM build (blobs.toml is no build input: the build key reads
    src, include, symbols and the extract recipe). A failed check puts the old file back and refuses, so a landing is
    all or nothing. Returns the git commands, the counts and the per-version credit delta."""
    file = path_of(config)
    with effort.stage("blobs.land"):
        old = _read(config)
        plan = import_plan(config, packet)
        if plan.blocking:
            raise Refusal(*plan.blocking)
        before = _partitions(config)
        have = load(config)["blob"]
        counts = _counts(have, packet)
        text = plan.writes[file]
        assert text is not None
        done = (
            apply_import(config, plan)
            if text != old
            else {"files": [], "message": plan.message, "add": "", "commit": ""}
        )
        try:
            with effort.stage("blobs.land.readback"):
                if _read(config) != text:
                    raise Refusal(Finding("blobs.land", "blobs.toml read back differs from the plan", path=file))
                document = parse(config, text)
                wrong = sorted(n for n, r in packet["blob"].items() if document["blob"].get(n) != r)
                if wrong:
                    raise Refusal(
                        Finding(
                            "blobs.land", f"rows read back differ from the packet: {', '.join(wrong[:5])}", path=file
                        )
                    )
                gone = sorted(n for n in packet.get("supersedes", ()) if n in document["blob"])
                if gone:
                    raise Refusal(Finding("blobs.land", f"superseded rows read back: {', '.join(gone[:5])}", path=file))
            with effort.stage("blobs.land.audit"):
                audit = import_module("unbake.policy").audit_command(
                    {"project": str(config.project.root), "against": "HEAD"}
                )
            with effort.stage("blobs.land.partition"):
                after = _partitions(config)
                for v, p in after.items():
                    held = sum(p[k] for k in ("code", "data", "platform", "blobs", "unknown"))
                    if held != p["rom_size"] or p["overlap"] or p["outside"]:
                        raise Refusal(
                            Finding(
                                "blobs.land",
                                f"{v}: partition sums to {held} of {p['rom_size']} (overlap {p['overlap']}, "
                                f"outside {p['outside']})",
                                path=file,
                                versions=(v,),
                            )
                        )
        except Exception:
            if old is None:
                (Path(config.project.root) / file).unlink(missing_ok=True)
            else:
                store.write(Path(config.project.root) / file, old)
            raise
        delta = {
            v: {k: after[v][k] - before[v][k] for k in ("blobs_decoded", "blobs_identified", "unknown")}
            for v in after
            if v in before
        }
        return {**counts, **done, "audit": {"paths": audit["paths"]}, "delta": delta, "partition": after}


_SHA = re.compile(r"[0-9a-f]{64}")


def _overlaps(document: Json) -> dict[tuple[str, str, str], tuple[int, int, int, int]]:
    """(version, row, row) -> the two ranges, for every pair of credited rows of a version that claim a common byte
    (a container and the rows inside it are such pairs; the project has them and `verify` only refuses new ones)."""
    claimed: dict[str, list[tuple[int, int, str]]] = {}
    for name, row in document["blob"].items():
        if row["state"] in ("decoded", "identified") and "parent" not in row:
            for v, spans in row.get("rom", {}).items():
                claimed.setdefault(v, []).extend((a, b, name) for a, b in spans)
    out: dict[tuple[str, str, str], tuple[int, int, int, int]] = {}
    for v, spans in claimed.items():
        spans.sort()
        for i, (a, b, one) in enumerate(spans):
            for c, d, other in spans[i + 1 :]:
                if c >= b:
                    break
                low, high = sorted((one, other))
                out[(v, low, high)] = (a, b, c, d)
    return out


def _overlap_findings(document: Json, previous: Json, file: str) -> list[Finding]:
    """Import and post-write verification refuse the same new duplicate claims, before any capture or write."""
    known = _overlaps(previous)
    return [
        Finding(
            "blobs.range",
            f"{v}: {one} {a:#x}..{b:#x} and {other} {c:#x}..{d:#x} claim the same bytes",
            path=file,
            versions=(v,),
        )
        for (v, one, other), (a, b, c, d) in _overlaps(document).items()
        if (v, one, other) not in known
    ]


def _pad_bytes_findings(config: Config, document: Json, file: str) -> list[Finding]:
    """Every range of a pad row, rebuilt as zero bytes, must equal the ROM bytes of its version."""
    found: list[Finding] = []
    for name, row in document["blob"].items():
        if row.get("format") != "pad":
            continue
        for v, spans in row.get("rom", {}).items():
            path = Path(config.project.root) / "roms" / f"baserom.{v}.z64"
            if not path.is_file():
                found.append(
                    Finding("blobs.range", f"blob {name} cannot be proved: no ROM for {v}", path=file, versions=(v,))
                )
                continue
            rom = path.read_bytes()
            for a, b in spans:
                if rom[a:b] != bytes(b - a):
                    found.append(
                        Finding(
                            "blobs.range",
                            f"blob {name} range {a:#x}..{b:#x} is not zero in the {v} ROM",
                            path=file,
                            versions=(v,),
                        )
                    )
    return found


def verify(config: Config, base: bytes | None = None) -> tuple[Finding, ...]:
    """The checks `land` makes on the blobs.toml of the working tree, for a change that did not come through `land`
    (a hand-written or merged file): the schema and ranges, the canonical form (the file is what `emit` writes for its
    own facts), a well-formed sha256 on every piece of evidence and evidence on every credited row, no pair of credited
    rows that claim a common byte and did not at `base` (the blobs.toml bytes of the base revision), and the partition
    of every version (the classes sum to the ROM, with no overlap, no byte outside the ROM and none both credited and
    shadowed). Never builds or reads an artifact."""
    file = path_of(config)
    data = _read(config)
    if data is None:
        return ()
    try:
        document = parse(config, data)
    except Refusal as refused:
        return refused.findings
    found: list[Finding] = []
    if emit(document) != data:
        found.append(Finding("blobs.land", "blobs.toml is not the canonical emit of its own facts", path=file))
    for name, row in document["blob"].items():
        for item in row.get("evidence", ()):
            if not _SHA.fullmatch(str(item.get("sha256", ""))):
                found.append(Finding("blobs.evidence", f"blob {name} cites a malformed sha256", path=file))
        if row["state"] in ("decoded", "identified") and "parent" not in row and not row.get("evidence"):
            found.append(Finding("blobs.evidence", f"blob {name} is {row['state']} without evidence", path=file))
    found += _pad_bytes_findings(config, document, file)
    try:
        previous = parse(config, base) if base else {"blob": {}}
    except Refusal:
        previous = {"blob": {}}
    found.extend(_overlap_findings(document, previous, file))
    if found:
        return tuple(found)
    parts = import_module("unbake.report").partitions(config, import_module("unbake.tree").capture(config))
    for v, p in parts.items():
        held = sum(p[k] for k in ("code", "data", "platform", "blobs", "unknown"))
        if held != p["rom_size"] or p["overlap"] or p["outside"] or p["shadowed"]:
            found.append(
                Finding(
                    "blobs.land",
                    f"{v}: partition sums to {held} of {p['rom_size']} (overlap {p['overlap']}, "
                    f"outside {p['outside']}, shadowed {p['shadowed']})",
                    path=file,
                    versions=(v,),
                )
            )
    return tuple(found)


# ---------------------------------------------------------------- the command
def _packet(path: str) -> Json:
    try:
        packet: Json = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as error:
        raise Refusal(Finding("blobs.packet", f"the packet cannot be read: {error}", path=path)) from error
    return packet


def run(config: Config, params: Json) -> Json:
    """`unbake blobs`: list the blobs and the partition of every version; `import PACKET` merges an evidence packet;
    `land PACKET` imports it as a checked, all-or-nothing facts-only landing."""
    with effort.stage("blobs.run"):
        action = list(params.get("action") or ())
        if action[:1] == ["prove"]:
            if len(action) != 3:
                raise Refusal(Finding("blobs.proof", "prove takes REQUEST RECEIPT"))
            from unbake import blob_proof

            return blob_proof.run(config, action[1], action[2])
        if action == ["scan"]:
            return _result(config, {"scanned": scan(config)}, bool(params.get("json")))
        if action == ["scan", "pad"]:
            return _result(config, {"scanned": import_module("unbake.blob_pad").scan(config)}, bool(params.get("json")))
        if action == ["scan", "stale"]:
            return _result(
                config, {"scanned": import_module("unbake.blob_stale").scan(config)}, bool(params.get("json"))
            )
        if action[:1] == ["import"]:
            if len(action) != 2:
                raise Refusal(Finding("blobs.packet", "import takes exactly one packet file"))
            packet = _packet(action[1])
            plan = import_plan(config, packet)
            if plan.blocking:
                raise Refusal(*plan.blocking)
            have = load(config)["blob"]
            counts = _counts(have, packet)
            done = apply_import(config, plan)
            return _result(config, {"imported": {**counts, **done}}, bool(params.get("json")))
        if action[:1] == ["land"]:
            if len(action) != 2:
                raise Refusal(Finding("blobs.packet", "land takes exactly one packet file"))
            return _result(config, land(config, _packet(action[1])), bool(params.get("json")))
        if action:
            raise Refusal(
                Finding(
                    "blobs.packet",
                    f"unknown {action[0]}; use prove REQUEST RECEIPT, import PACKET, land PACKET or scan [stale]",
                )
            )
        return _result(config, {}, bool(params.get("json")))


def scan(config: Config) -> Json:
    """`unbake blobs scan`: per version, decode every catalogued RNC stream (both CRCs and `decoded_size` must hold)
    and list the RNC streams that decode whole inside unknown bytes as candidates for typed rows (start, stored and
    unpacked size, sha256 of the decoded bytes: facts, never bytes). Replaces the lane-side stream scripts."""
    tree = import_module("unbake.tree").capture(config)
    parts = import_module("unbake.report").partitions(config, tree)
    document = load(config)["blob"]
    out: dict[str, Any] = {}
    for v, p in parts.items():
        rom = (Path(config.project.root) / "roms" / f"baserom.{v}.z64").read_bytes()
        checked, failed = 0, []
        for name, row in sorted(document.items()):
            if row.get("codec") not in {"rnc1", "rnc2"} or "parent" in row or v not in row.get("rom", {}):
                continue
            start = row["rom"][v][0][0]
            checked += 1
            try:
                if row["codec"] != f"rnc{rnc.header(rom, start).method}":
                    raise rnc.RncError("RNC method differs from declared codec")
                size = len(rnc.unpack(rom, start))
            except rnc.RncError as error:
                failed.append({"blob": name, "reason": str(error)})
                continue
            if "decoded_size" in row and row["decoded_size"] != size:
                failed.append({"blob": name, "reason": f"decodes to {size} bytes, the row says {row['decoded_size']}"})
        spans = p["spans"]
        held = merge([s for k in ("code", "data", "platform", "decoded", "identified", "fill") for s in spans[k]])
        candidates: list[Json] = []
        for low, high in subtract([(0, len(rom))], held):
            for at, head in rnc.find(rom, low, high):
                try:
                    body = rnc.unpack(rom, at)
                except rnc.RncError:
                    continue
                candidates.append(
                    {
                        "rom": [at, at + head.stored],
                        "stored_size": head.stored,
                        "codec": f"rnc{head.method}",
                        "decoded_size": len(body),
                        "sha256": sha256(body),
                    }
                )
        out[v] = {
            "rnc": {"checked": checked, "failed": failed},
            "candidates": candidates,
            "candidate_bytes": sum(c["stored_size"] for c in candidates),
            "fill": p["blobs_fill"],
        }
    return out


def _result(config: Config, extra: Json, with_rows: bool) -> Json:
    document = load(config)
    tree = import_module("unbake.tree").capture(config)
    report = import_module("unbake.report")
    keep = (
        "rom_size",
        "code",
        "data",
        "platform",
        "blobs",
        "blobs_decoded",
        "blobs_identified",
        "blobs_fill",
        "unknown",
        "blobs_no_row",
        "shadowed",
    )
    parts = report.partitions(config, tree)
    result: dict[str, Any] = {
        "count": len(document["blob"]),
        "census": census(document),
        "partition": {v: {k: p[k] for k in keep} for v, p in parts.items()},
        "debt": [
            asdict(f)
            for v, p in parts.items()
            for f in shadows(config, v, merge(p["spans"]["code"] + p["spans"]["data"]))
        ],
        **extra,
    }
    if with_rows:
        result["rows"] = rows(config)
    return result


def shadows(config: Config, version: str, credit: Spans) -> tuple[Finding, ...]:
    """Non-blocking findings for blobs under a credited symbol: those bytes stay counted as code or data."""
    file = path_of(config)
    found = []
    for name, row in sorted(load(config)["blob"].items()):
        mine = merge(row.get("rom", {}).get(version, ()))
        if row["state"] != "observed" and (hit := length(mine) - length(subtract(mine, credit))):
            message = f"blob {name} has {hit} bytes under a credited symbol"
            found.append(Finding("blobs.shadowed", message, path=file, versions=(version,), blocking=False))
    return tuple(found)
