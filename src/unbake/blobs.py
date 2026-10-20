"""Blobs: the hand-kept facts about ROM content that is not code or data (`blobs.toml`), and the ROM byte partition.

A blob is graphics, model, audio, world or text content with a physical ROM range. The file holds facts only (ranges,
formats, states, evidence by sha256), never bytes. `sweep` classifies every ROM byte exactly once."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from unbake import config as configuration
from unbake import effort, journal, rnc, store
from unbake.contracts import Config, Finding, Json, Plan, Refusal, digest, sha256, sha256_file

FILE = "blobs.toml"
Spans = list[tuple[int, int]]
_ORDER = ("name", "aliases", "type", "format", "codec", "state", "stored_size", "decoded_size", "parent", "decoded")


def _read(config: Config) -> bytes | None:
    path = Path(config.project.root) / FILE
    return path.read_bytes() if path.is_file() else None


def _rom_size(config: Config, version: str) -> int | None:
    path = Path(config.project.root) / "roms" / f"baserom.{version}.z64"
    return path.stat().st_size if path.is_file() else None


def sha(config: Config) -> str:
    """The sha256 of blobs.toml, empty when the project has none: the key every blob-derived cache carries."""
    data = _read(config)
    return sha256(data) if data is not None else ""


def parse(config: Config, data: bytes, file: str = FILE) -> Json:
    """The checked document of blobs.toml bytes: schema, parents and ranges are verified, every failure blocking."""
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


def check(config: Config, document: Json, file: str = FILE) -> tuple[Finding, ...]:
    """Blocking findings of a document: a parent that is missing or cyclic, a child with ROM bytes of its own, a range
    that is empty, outside the ROM or in an unknown version."""
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


def ranges(document: Json, version: str) -> tuple[Spans, Spans]:
    """(decoded, identified) physical ranges of one version. A blob in the `observed` state, and a blob inside
    its parent's stream, adds no bytes."""
    out: dict[str, Spans] = {"decoded": [], "identified": []}
    for row in document.get("blob", {}).values():
        if row["state"] in out and "parent" not in row:
            out[row["state"]] += [(a, b) for a, b in row.get("rom", {}).get(version, ())]
    return out["decoded"], out["identified"]


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
    Proof is by layout, never by content guess: a whole gap of one byte value that ends where the next item's alignment
    puts it (its length is below the largest power of two dividing its end), or the maximal one-value run that reaches
    the end of the ROM (cart padding) and starts on a 16-byte boundary. The build emits these bytes like any other, so
    `make check` proves them with the rest of the ROM."""
    out: Spans = []
    size = len(rom)
    for start, end in gaps:
        value = rom[end - 1]
        if rom.count(value, start, end) == end - start and (end == size or end - start < (end & -end)):
            out.append((start, end))
        elif end == size:
            low = end - 1
            while low > start and rom[low - 1] == value:
                low -= 1
            low = (low + 15) & ~15
            if low < end:
                out.append((low, end))
    return out


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


def import_plan(config: Config, packet: Json) -> Plan:
    """The reviewed plan that merges an evidence packet into blobs.toml: the packet's ROM and artifact sha256 must match
    what the project holds and what each row cites; a row already present is replaced by the packet's."""
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
                        path=FILE,
                    )
                )
    merged = {"schema": 1, "blob": {**document["blob"], **packet["blob"]}}
    blocking += check(config, merged)
    writes = {FILE: emit(merged)}
    added = sum(n not in document["blob"] for n in packet["blob"])
    message = f"Add {added} blobs to blobs.toml" if added else "Update blobs.toml"
    body = ("blobs", sha256(old) if old is not None else "", writes, (), tuple(blocking), (), message)
    return Plan(*body, digest(body))


def apply_import(config: Config, plan: Plan) -> Json:
    """Write a plan (never Git), read it back and name the git add/commit that records it."""
    old = _read(config)
    reads = _Reads(config, {FILE: sha256(old) if old is not None else None})
    with store.exclusive(config, "structure", wait=True):
        files = journal.apply(config, plan, reads)
    return {"files": list(files), "add": "git add " + " ".join(files), "commit": f'git commit -m "{plan.message}"'}


# ---------------------------------------------------------------- the command
def run(config: Config, params: Json) -> Json:
    """`unbake blobs`: list the blobs and the partition of every version; `import PACKET` merges an evidence packet."""
    with effort.stage("blobs.run"):
        action = list(params.get("action") or ())
        if action == ["scan"]:
            return _result(config, {"scanned": scan(config)}, bool(params.get("json")))
        if action == ["migrate"]:
            from unbake import blob_migration

            return _result(config, {"migrated": blob_migration.run(config)}, bool(params.get("json")))
        if action[:1] == ["import"]:
            if len(action) != 2:
                raise Refusal(Finding("blobs.packet", "import takes exactly one packet file"))
            try:
                packet = json.loads(Path(action[1]).read_bytes())
            except (OSError, ValueError) as error:
                raise Refusal(Finding("blobs.packet", f"the packet cannot be read: {error}", path=action[1])) from error
            plan = import_plan(config, packet)
            if plan.blocking:
                raise Refusal(*plan.blocking)
            have = load(config)["blob"]
            counts = {
                "added": sum(n not in have for n in packet["blob"]),
                "unchanged": sum(have.get(n) == r for n, r in packet["blob"].items()),
            }
            done = apply_import(config, plan)
            return _result(config, {"imported": {**counts, **done}}, bool(params.get("json")))
        if action:
            raise Refusal(Finding("blobs.packet", f"unknown action {action[0]}; use import PACKET, scan or migrate"))
        return _result(config, {}, bool(params.get("json")))


def scan(config: Config) -> Json:
    """`unbake blobs scan`: per version, decode every catalogued RNC1 stream (both CRCs and `decoded_size` must hold)
    and list the RNC1 streams that decode whole inside unknown bytes as candidates for typed rows (start, stored and
    unpacked size, sha256 of the decoded bytes: facts, never bytes). Replaces the lane-side stream scripts."""
    tree = import_module("unbake.tree").capture(config)
    parts = import_module("unbake.report").partitions(config, tree)
    document = load(config)["blob"]
    out: Json = {}
    for v, p in parts.items():
        rom = (Path(config.project.root) / "roms" / f"baserom.{v}.z64").read_bytes()
        checked, failed = 0, []
        for name, row in sorted(document.items()):
            if row.get("codec") != "rnc1" or "parent" in row or v not in row.get("rom", {}):
                continue
            start = row["rom"][v][0][0]
            checked += 1
            try:
                size = len(rnc.unpack(rom, start))
            except rnc.RncError as error:
                failed.append({"blob": name, "reason": str(error)})
                continue
            if "decoded_size" in row and row["decoded_size"] != size:
                failed.append({"blob": name, "reason": f"decodes to {size} bytes, the row says {row['decoded_size']}"})
        spans = p["spans"]
        held = merge([s for k in ("code", "data", "platform", "decoded", "identified", "fill") for s in spans[k]])
        candidates = []
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
                        "decoded_size": len(body),
                        "sha256": sha256(body),
                    }
                )
        out[v] = {
            "rnc1": {"checked": checked, "failed": failed},
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
    found = []
    for name, row in sorted(load(config)["blob"].items()):
        mine = merge(row.get("rom", {}).get(version, ()))
        if row["state"] != "observed" and (hit := length(mine) - length(subtract(mine, credit))):
            message = f"blob {name} has {hit} bytes under a credited symbol"
            found.append(Finding("blobs.shadowed", message, path=FILE, versions=(version,), blocking=False))
    return tuple(found)
