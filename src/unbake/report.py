"""Layout credit, work lists, objdiff reports and README progress."""

from __future__ import annotations

import json
import re
import struct
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from typing import Any

import rabbitizer

from unbake import adapters, effort, headers, infer, land, layout, libasm, policy, pool, store, versions
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Member, Project, Refusal, Snapshot, UnitSpec, digest, sha256


def _pct(part: float, whole: float) -> float:
    return (10000 * int(part) // int(whole)) / 100 if whole else 0.0


def _f32(value: float) -> float:
    single = struct.unpack(">f", struct.pack(">f", value))[0]
    for digits in range(1, 10):
        shortest = float(f"{single:.{digits}g}")
        if struct.unpack(">f", struct.pack(">f", shortest))[0] == single:
            return shortest
    return float(single)


def _text_starts(snapshot: Snapshot) -> dict[tuple[str, int], str]:
    """The first member whose text starts at each (version, ROM offset), built once per command."""

    def build() -> dict[tuple[str, int], str]:
        starts: dict[tuple[str, int], str] = {}
        for name, member in snapshot.layout.members.items():
            for p in member.placements:
                if p.section == ".text":
                    starts.setdefault((p.version, p.rom_start), name)
        return starts

    return effort.memo(("text-starts", snapshot.digest), build)


def _text_spans(snapshot: Snapshot) -> dict[str, list[tuple[int, int, str]]]:
    """Per version the (vram start, vram end, member) of every text placement in address order, built once."""

    def build() -> dict[str, list[tuple[int, int, str]]]:
        spans: dict[str, list[tuple[int, int, str]]] = {}
        for name, member in snapshot.layout.members.items():
            for p in member.placements:
                if p.section == ".text":
                    spans.setdefault(p.version, []).append((p.vram, p.vram + p.size, name))
        return {v: sorted(rows) for v, rows in spans.items()}

    return effort.memo(("text-spans", snapshot.digest), build)


def _member_at(snapshot: Snapshot, version: str, vram: int) -> tuple[int, str] | None:
    """The (start, member) of the text member of `version` that holds `vram`."""
    rows = _text_spans(snapshot).get(version, [])
    at = bisect_right(rows, (vram, 1 << 64, "")) - 1
    return (rows[at][0], rows[at][2]) if at >= 0 and vram < rows[at][1] else None


def _ends(word: int) -> bool:  # an unconditional jr, j or b, after which execution never reaches the next member
    return (word >> 26 == 0 and word & 0x3F == 8) or word >> 26 == 2 or word >> 16 == 0x1000


_ENTRY_WORDS = 64
_DEFINED_AT_ENTRY = frozenset({0, 4, 5, 6, 7, 28, 29, 31})  # $zero, $a0-$a3, $gp, $sp, $ra
_SAVED = frozenset({*range(16, 24), 30})  # $s0-$s7, $fp: defined only to be stored to the stack


def _reads_before_writing(words: Sequence[int], vram: int) -> bool:
    """Whether the straight-line entry block (up to the first jump or branch and its delay slot) reads a register
    the calling convention leaves undefined at entry: such a symbol is a point inside a function."""
    defined, slot = set(_DEFINED_AT_ENTRY), False
    for index, word in enumerate(words):
        instruction = rabbitizer.Instruction(word, vram + 4 * index)
        if not instruction.isValid():
            return False
        reads = [
            int(getattr(instruction, name).value)
            for name, reading in (
                ("rs", instruction.readsRs()),
                ("rt", instruction.readsRt()),
                ("rd", instruction.readsRd()),
            )
            if reading
        ]
        saving = instruction.doesStore() and instruction.readsRs() and int(instruction.rs.value) == 29
        if any(r not in defined and not (r in _SAVED and saving) for r in reads):
            return True
        defined.update(
            int(getattr(instruction, name).value)
            for name, writing in (
                ("rs", instruction.modifiesRs()),
                ("rt", instruction.modifiesRt()),
                ("rd", instruction.modifiesRd()),
            )
            if writing
        )
        if slot:
            return False
        slot = instruction.hasDelaySlot()
    return False


def _epilogue(words: Sequence[int]) -> bool:
    """Whether the entry block (up to the first jump and its delay slot) hands stack back before it ever takes any:
    `addiu sp, sp, +N` ahead of every negative adjustment is the tail of a function whose frame opened earlier."""
    slot = False
    for word in words:
        if word >> 16 == 0x27BD:  # addiu sp, sp, imm
            return word & 0xFFFF != 0 and word & 0x8000 == 0
        if slot:
            return False
        slot = versions.delay_slot(word)
    return False


def _not_c(words: Sequence[int], vram: int, never: Sequence[str]) -> str:
    """Why the words cannot be compiler output, or "": a non-jump instruction writing $zero (a destination, never a
    source) is data; a coprocessor-0 or system instruction is what C cannot emit without inline assembly."""
    for index, word in enumerate(words):
        instruction = rabbitizer.Instruction(word, vram + 4 * index)
        if not word or not instruction.isValid():
            continue
        if instruction.getOpcodeName() in never:
            return "system code C cannot emit"
        if not instruction.hasDelaySlot() and any(
            int(getattr(instruction, name).value) == 0
            for name, writing in (("rt", instruction.modifiesRt()), ("rd", instruction.modifiesRd()))
            if writing
        ):
            return "data misclassified as code"
    return ""


def split_slot(snapshot: Snapshot, member: str) -> str | None:
    """The member that `member` runs into: the one holding its last delay slot, or the one it falls through into. A
    member whose entry reads an undefined register is "the middle of a function"; one writing $zero is data;
    coprocessor-0 or system code is named so."""
    never = adapters.row(snapshot.config.project.toolchain)["never_emits"]
    reference = snapshot.layout.members[member].reference(snapshot.config.project.names_from)
    for place in snapshot.layout.members[member].placements:
        if place.section != ".text" or place.size < 4:
            continue
        version = snapshot.versions[place.version]
        if place.version == reference:
            text = versions.rom_bytes(version, place.rom_start, place.rom_end)
            if why := _not_c(
                [int.from_bytes(text[i : i + 4], "big") for i in range(0, len(text) - 3, 4)], place.vram, never
            ):
                return why
        last = int.from_bytes(versions.rom_bytes(version, place.rom_end - 4, place.rom_end), "big")
        pair = place.size >= 8 and int.from_bytes(
            versions.rom_bytes(version, place.rom_end - 8, place.rom_end - 4), "big"
        )
        # a lone instruction that is no jump has no room for a return: it runs into the next member
        if versions.delay_slot(last) or (place.size >= 8 and not _ends(pair)) or (place.size == 4 and last):
            return _text_starts(snapshot).get((place.version, place.rom_end), "an unowned range")
        data = versions.rom_bytes(version, place.rom_start, min(place.rom_end, place.rom_start + 4 * _ENTRY_WORDS))
        entry = [int.from_bytes(data[i : i + 4], "big") for i in range(0, len(data) - 3, 4)]
        if _reads_before_writing(entry, place.vram) or _epilogue(entry):
            return "the middle of a function"
        if pair >> 26 == 2 and place.size >= 8:  # the member ends in `j`: a jump to the inside of another member
            target = (place.vram + place.size - 4 & 0xF0000000) | (pair & 0x3FFFFFF) << 2
            held = _member_at(snapshot, place.version, target)
            if held and held[1] != member and held[0] != target:
                return "the middle of a function"
    return None


def _credited(project: Project, unit: UnitSpec) -> bool:
    """Whether the unit's source counts as matched code: decompiled source, or assembly in a library directory."""
    kind = configuration.kind(unit.kind)
    return bool(kind["decompiled"] or (kind["credit"] == "code" and configuration.library(project, unit.path)))


def _inventory(snapshot: Snapshot) -> Iterator[tuple[str, Member, UnitSpec | None, bool, str]]:
    for name, member in snapshot.layout.members.items():
        if (unit := layout.unit_of(snapshot, name)) is None and member.state == "bin":
            continue  # opaque ROM bytes are neither code nor data
        matched = (
            unit is not None and _credited(snapshot.config.project, unit) and not unit.withheld
        )  # built in every version
        yield name, member, unit, matched, snapshot.layout.groups[member.group].segment if member.group else "unknown"


def _version_matched(project: Project, unit: UnitSpec | None, version: str) -> bool:
    """A withheld holder does not remove the exact credit of the unit's other holders."""
    return bool(unit is not None and _credited(project, unit) and version not in unit.withheld)


Buckets = dict[str, dict[str, int]]  # version -> credit counters


def _tally(b: Json) -> Json:
    return {
        **b,
        "fuzzy_bytes": (fuzzy := b["code_fuzzy"] + b["data_fuzzy"]),
        "code_percent": _pct(b["code_matched"], b["code_total"]),
        "data_percent": _pct(b["data_matched"], b["data_total"]),
        "fuzzy_percent": _pct(b["code_matched"] + b["data_matched"] + fuzzy, b["code_total"] + b["data_total"]),
    }


def _check(snapshot: Snapshot) -> Json | None:
    def read() -> Json | None:
        with suppress(FileNotFoundError, ValueError):
            loaded: Json = json.loads(snapshot.read(".unbake/check.json"))
            return loaded
        return None

    return effort.memo(("check", snapshot.digest), read)


def _broken(
    snapshot: Snapshot,
) -> frozenset[str]:  # units the last check could not compile, on the tree it checked: no credit
    document = _check(snapshot) or {}
    return frozenset(document.get("failing", ())) if document.get("digest") == snapshot.digest else frozenset()


def _tallies(snapshot: Snapshot) -> tuple[Buckets, dict[str, Buckets], dict[str, Buckets]]:
    """(per version, per segment, per kind) credit buckets of the inventory, summed in one pass."""
    zero = dict.fromkeys(
        (
            "code_total",
            "code_matched",
            "data_total",
            "data_matched",
            "functions_total",
            "functions_matched",
            "functions_fuzzy",
            "code_fuzzy",
            "data_fuzzy",
        ),
        0,
    )
    totals = {v: dict(zero) for v in snapshot.config.project.versions}
    dirty, broken = policy.unclean(snapshot), _broken(snapshot)
    segments: defaultdict[str, Buckets] = defaultdict(dict)
    kinds: defaultdict[str, Buckets] = defaultdict(dict)
    for name, member, unit, _matched, segment in _inventory(snapshot):
        for v in member.holders():
            built = _version_matched(snapshot.config.project, unit, v)
            exact = built and name not in dirty  # a function an unclean finding withholds earns none; data keeps it
            placements = [p for p in member.placements if p.version == v]
            code = sum(p.size for p in placements if p.section == ".text")
            data = sum(p.size for p in placements if p.section in (".data", ".rodata"))
            score = 0.0 if exact else snapshot.layout.fuzzy.get(name, {"scores": {}})["scores"].get(v, 0.0)
            if unit is not None and unit.path in broken and not exact:
                score = 0.0  # it does not compile: no credit
            elif built and not exact:
                score = 1.0  # exact bytes in unclean source: fuzzy at most
            for bucket in (
                totals[v],
                segments[segment].setdefault(v, dict(zero)),
                kinds[unit.kind if unit else member.state].setdefault(v, dict(zero)),
            ):
                for prefix, size in (("code_", code), ("data_", data), ("functions_", int(member.kind == "function"))):
                    bucket[prefix + "total"] += size
                    bucket[prefix + "matched"] += size if exact else 0
                bucket["functions_fuzzy"] += int(member.kind == "function" and not exact and score > 0)
                bucket["code_fuzzy"] += round(score * code)
                bucket["data_fuzzy"] += round(score * data)
    return totals, segments, kinds


def progress(snapshot: Snapshot) -> Json:
    """The per-version credit a published file shows: `current` without its debt and boundary census."""
    with effort.stage("report.progress"):
        return {"versions": {v: _tally(b) for v, b in _tallies(snapshot)[0].items()}}


def current(snapshot: Snapshot) -> Json:
    with effort.stage("report.current"):
        totals, segments, kinds = _tallies(snapshot)
        debt = {"stale": True, "action": "unbake check"}
        try:
            document = _check(snapshot)
            if document is not None and document["commit"] == snapshot.commit:
                debt = {
                    "commit": document["commit"],
                    "counts": document["counts"],
                    "type_debt": document.get("type_debt", {}),
                }
        except (KeyError, TypeError):
            pass
        counts = layout.boundary_plan(snapshot)[1]
        boundary = {key: counts[key] for key in ("prelude", "split", "merge")}
        return {
            "commit": snapshot.commit,
            "versions": {v: _tally(b) for v, b in totals.items()},
            "segments": {s: {v: _tally(b) for v, b in rows.items()} for s, rows in segments.items()},
            "kinds": {k: {v: _tally(b) for v, b in rows.items()} for k, rows in kinds.items()},
            "debt": debt,
            "boundary": boundary,
            "pending_boundary": sum(row["applied"] for row in boundary.values()),
        }


def items(snapshot: Snapshot, params: Json) -> list[dict[str, Any]]:
    with effort.stage("report.items"):
        segment, shard, count = (params.get(k) for k in ("segment", "shard", "count"))
        if segment is not None and (
            not isinstance(segment, str) or segment not in {g.segment for g in snapshot.layout.groups.values()}
        ):
            raise Refusal(Finding("report.request", "segment must be a segment name of the layout"))
        if count is not None and (type(count) is not int or count < 1):
            raise Refusal(Finding("report.request", "count must be an integer at least 1"))
        index, modulus = 0, 1
        if shard is not None and (
            not isinstance(shard, str)
            or not (m := re.fullmatch(r"([0-9]+)/([0-9]+)", shard))
            or not 0 <= (index := int(m[1])) < (modulus := int(m[2]))
        ):
            raise Refusal(Finding("report.request", "shard must be I/K with 0 <= I < K"))
        rows: list[dict[str, Any]] = []
        inventory = list(_inventory(snapshot))
        library = {
            name
            for name, member, _unit, done, _sid in inventory
            if member.kind == "function" and not done and libasm.recognised(snapshot, name)
        }
        slots = {
            name: None if name in library else split_slot(snapshot, name)
            for name, member, *_ in inventory
            if member.kind == "function"
        }
        successors = set(slots.values())
        for name, member, unit, matched, sid in inventory:
            sections = (".text",) if member.kind == "function" else (".data", ".rodata")
            if matched or not any(p.section in sections for p in member.placements):
                continue
            if member.kind == "function" and (
                (member.state == "hasm" and name not in library) or slots[name] is not None or name in successors
            ):  # handwritten assembly is no target
                continue
            if segment is not None and sid != segment:
                continue
            if int(sha256(name.encode())[:8], 16) % modulus != index:
                continue
            version = member.reference(snapshot.config.project.names_from)
            placement = next(p for p in member.placements if p.version == version)
            rows.append(
                {
                    "member": name,
                    "kind": "library-asm" if name in library else member.kind,
                    "segment": sid,
                    "size": placement.size,
                    "address": placement.vram,
                    **({"source": unit.path} if unit is not None and unit.withheld else {}),
                }
            )
        # biggest first, fragments last
        rows.sort(key=lambda r: (r["size"] < 16, -r["size"], r["address"], r["member"]))
        return rows[:count]


def _sum_measures(units: Sequence[Json]) -> Json:
    counters = (
        "total_code",
        "matched_code",
        "complete_code",
        "total_data",
        "matched_data",
        "complete_data",
        "total_functions",
        "matched_functions",
        "total_units",
        "complete_units",
    )
    result: dict[str, Any] = {k: sum(int(u["measures"].get(k, 0)) for u in units) for k in counters}
    for field in ("matched_code", "complete_code", "matched_functions", "matched_data", "complete_data"):
        total = result["total_" + field.split("_", 1)[1]]
        result[field + "_percent"] = _f32(100 * result[field] / total) if total else 0.0
    weighted = sum(int(u["measures"].get("total_code", 0)) * u["measures"].get("fuzzy_match_percent", 0) for u in units)
    result["fuzzy_match_percent"] = _f32(weighted / result["total_code"]) if result["total_code"] else 0.0
    for field in counters[:6]:
        result[field] = str(result[field])
    return result


def objdiff(
    snapshot: Snapshot,
    report: Json,
    version: str,
    rules: Mapping[str, Sequence[str]] | None = None,
    broken: frozenset[str] | None = None,
) -> Json:
    """The version's report. The unclean rules per withheld function and the units that do not compile are read once
    by the caller and passed in: every version needs the same two answers."""
    with effort.stage("report.objdiff"):
        units: list[dict[str, Any]] = []
        rules = policy.unclean_rules(snapshot) if rules is None else rules
        broken = _broken(snapshot) if broken is None else broken
        for name, member, unit, matched, sid in _inventory(snapshot):
            placements = [p for p in member.placements if p.version == version]
            if not placements:
                continue
            built = (
                unit is not None
                and unit.path not in broken
                and _version_matched(snapshot.config.project, unit, version)
            )
            matched = built and name not in rules
            dirty = built and not matched  # exact bytes, a function an unclean finding withholds: fuzzy 100
            function = member.kind == "function"
            sections = (".text",) if function else (".data", ".rodata")
            size = sum(p.size for p in placements if p.section in sections)
            fuzzy = snapshot.layout.fuzzy.get(name) if not matched and not (unit and unit.path in broken) else None
            percent = 100.0 if matched or dirty else _f32(fuzzy["scores"].get(version, 0.0) * 100) if fuzzy else 0.0
            counter = "total_code" if function else "total_data"
            prefix = counter.removeprefix("total_")
            measures: dict[str, Any] = {counter: str(size), "total_units": 1}
            if function:
                measures.update(total_functions=1, fuzzy_match_percent=percent)
            if matched:
                for key in ("matched_", "complete_"):
                    measures[key + prefix] = str(size)
                    measures[key + prefix + "_percent"] = 100.0
                measures["complete_units"] = 1
                if function:
                    measures.update(matched_functions=1, matched_functions_percent=100.0)
            metadata: dict[str, Any] = {"complete": matched, "progress_categories": [sid]}
            sourced = unit is not None and _credited(snapshot.config.project, unit)
            if unit is not None and sourced:
                metadata["source_path"] = unit.path
            elif fuzzy:
                metadata["source_path"] = fuzzy["path"]
            address = next((p.vram for p in placements if p.section == ".text"), placements[0].vram)
            functions: list[dict[str, Any]] = [
                {"name": name, "size": str(size), "address": str(address), "metadata": {}}
            ]
            if sourced or fuzzy:
                functions[0]["fuzzy_match_percent"] = percent
            section_rows = (
                [{"name": ".text", "size": str(size), "fuzzy_match_percent": percent, "metadata": {}}]
                if function
                else [
                    {"name": p.section, "size": str(p.size), "fuzzy_match_percent": percent, "metadata": {}}
                    for p in placements
                ]
            )
            units.append(
                {
                    "name": name,
                    "measures": measures,
                    "sections": section_rows,
                    "functions": functions if function else [],
                    "metadata": metadata,
                }
            )
        addresses = {
            name: min(p.vram for p in m.placements if p.version == version)
            for name, m in snapshot.layout.members.items()
            if version in m.holders()
        }
        units.sort(key=lambda u: (addresses[u["name"]], u["name"]))
        present = {u["metadata"]["progress_categories"][0] for u in units}
        categories = [
            {
                "id": sid,
                "name": sid,
                "measures": _sum_measures([u for u in units if sid in u["metadata"]["progress_categories"]]),
            }
            for sid in sorted(present)
        ]
        result = {"measures": _sum_measures(units), "units": units, "categories": categories, "version": 2}
        configuration.validate("objdiff", result, f"versions/{version}/report.json")
        return result


def _line(label: str, bucket: Json, cells: int, kind: str = "bytes") -> str:
    parts = ("code", "data") if kind == "bytes" else (kind,)
    matched = sum(bucket[p + "_matched"] for p in parts)
    total = sum(bucket[p + "_total"] for p in parts)
    percent = _pct(matched, total)
    fuzzy = _pct(matched + sum(bucket[p + "_fuzzy"] for p in parts), total)
    solid = int(percent // (100 / cells))
    partial = max(0, min(cells - solid, int(fuzzy // (100 / cells)) - solid))
    bar = "█" * solid + "▒" * partial + "░" * (cells - solid - partial)
    suffix = f" (~{fuzzy:.2f}%)" if fuzzy != percent else ""
    return f"{label} [{bar}]  {percent:5.2f}%{suffix}  {matched:,} of {total:,}"


def readme(text: str, snapshot: Snapshot, report: Json) -> str:
    with effort.stage("report.readme"):
        settings = configuration.load_resource("repo.toml")["readme"]
        begin, end, cells = (settings[k] for k in ("begin", "end", "bar"))
        if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) >= text.index(end):
            raise Refusal(
                Finding(
                    "repo.readme_markers", "README.md needs one begin and one end progress marker", path="README.md"
                )
            )
        project, versions = snapshot.config.project, report["versions"]
        total = {
            k: sum(versions[v][k] for v in project.versions)
            for k in (
                "code_total",
                "code_matched",
                "data_total",
                "data_matched",
                "functions_total",
                "functions_matched",
                "functions_fuzzy",
                "code_fuzzy",
                "data_fuzzy",
            )
        }
        width = max(map(len, ("all", *project.versions))) + 1
        lines = [
            _line(label.ljust(width), bucket, cells) + " bytes"
            for label, bucket in [("all", total), *((v, versions[v]) for v in project.versions)]
        ]
        blocks = ["<pre><code>" + "</code><br><code>".join(lines) + "</code></pre>"]
        for v in project.versions:
            meta = project.version_files[v].meta
            details = ", ".join(meta[k] for k in ("cartridge_id", "region") if meta.get(k))
            header = f"| {v}" + (f" ({details})" if details else "") + "."
            header += f" {meta['description']}" if meta.get("description") else ""
            header += f" SHA256 `{snapshot.versions[v].rom_sha256}` |"
            rows = [_line(k.ljust(9), versions[v], cells, k) for k in ("code", "data", "functions")]
            blocks.append(header + "\n|---|\n| <pre><code>" + "</code><br><code>".join(rows) + "</code></pre> |")
        return text[: text.index(begin) + len(begin)] + "\n" + "\n\n".join(blocks) + "\n" + text[text.index(end) :]


_Item = tuple[Snapshot, Json, str, Mapping[str, Sequence[str]], frozenset[str]]
STAMP = "versions/stamp.json"  # beside the reports: objdiff refuses a report.json field it does not know


def _objdiff_job(item: _Item) -> bytes:
    document = objdiff(*item)
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _objdiff_key(item: _Item) -> str:
    """Everything one version's report reads: the snapshot, the credit it shows, the sources judged unclean and the
    units the last check could not compile. Warm, the pool dispatches nothing."""
    snapshot, report, version, rules, broken = item
    return digest((snapshot.digest, version, report, sorted(rules.items()), sorted(broken)))


def stamp(snapshot: Snapshot) -> Json:
    """What the report.json files are made by and from (versions/stamp.json): the tool's code and the tracked text
    they read (the map, every unit and header), so a reader tells files a later commit left behind from current ones."""
    paths = sorted({*snapshot.layout.units, *headers.sources(snapshot)})
    tree = digest((snapshot.peek("layout.toml"), [(p, snapshot.peek(p)) for p in paths]))
    return {"tool": store.code(), "tree": tree}


def stale(snapshot: Snapshot) -> list[str]:
    """The versions whose report.json was not made from this tree: a reader refuses its numbers."""
    made = None
    with suppress(ValueError, AttributeError):
        made = json.loads(snapshot.peek(STAMP) or b"{}").get("tree")
    return [] if made == stamp(snapshot)["tree"] else list(snapshot.config.project.versions)


def files(snapshot: Snapshot) -> dict[str, bytes]:
    with effort.stage("report.files"):
        report = progress(snapshot)
        ids = snapshot.config.project.versions
        rules, broken = policy.unclean_rules(snapshot), _broken(snapshot)  # once, not again in every version's job
        blobs = pool.map(
            snapshot.config,
            "report.objdiff",
            _objdiff_job,
            [(snapshot, report, v, rules, broken) for v in ids],
            _objdiff_key,
        )
        result = {f"versions/{v}/report.json": blob for v, blob in zip(ids, blobs, strict=True)}
        result[STAMP] = json.dumps(stamp(snapshot), sort_keys=True).encode() + b"\n"
        result["README.md"] = readme(snapshot.read("README.md").decode(), snapshot, report).encode()
        return result


def run(config: Config, params: Json) -> Json:
    with effort.stage("report.run"):
        snapshot = layout.capture(config)
        if params["next"]:
            return {"items": items(snapshot, params)}
        return {**current(snapshot), "inbox": len(land.inbox(config)), "stale": stale(snapshot)}


def _strings(rom: bytes, start: int, limit: int = 4, width: int = 40) -> list[str]:
    """The printable ASCII runs of at least four characters at the start of a rodata block, at most `limit`."""
    runs = re.findall(rb"[\x20-\x7e]{4,}", rom[start : start + 256])
    return [run[:width].decode() for run in runs[:limit]]


class Segments:
    """Segments by VRAM: a member sits in the names_from segment whose VRAM range holds its names_from placement (the
    first holder's when names_from lacks it), so per-version spans of one range are one segment."""

    def __init__(self, snapshot: Snapshot) -> None:
        self.versions = snapshot.versions
        self.base = snapshot.config.project.names_from
        order = [self.base, *(v for v in self.versions if v != self.base)]
        self.ranges = [(t[3], t[3] + t[2] - t[1], t[0]) for v in order for t in self.versions[v].segments]

    def of(self, member: Any) -> str:
        holder = member.reference(self.base)
        vram = next(p.vram for p in member.placements if p.version == holder)
        return next((name for lo, hi, name in self.ranges if lo <= vram < hi), "")

    def listed(self, used: set[str]) -> list[dict[str, Any]]:
        """Each used segment with the span every version holds of its VRAM range."""
        out = []
        for name in sorted(used - {""}):
            lo, hi, _ = next(r for r in self.ranges if r[2] == name)
            versions: dict[str, list[int]] = {}
            for v, ver in self.versions.items():
                span = next((t for t in ver.segments if t[0] == name), None) or next(
                    (t for t in ver.segments if t[3] < hi and lo < t[3] + t[2] - t[1]), None
                )
                if span:
                    versions[v] = list(span[1:])
            out.append({"name": name, "versions": versions})
        return out


def map_command(config: Config, params: Json) -> Json:
    """Write the layout map with its objective evidence to .unbake/map.json (CONTRACTS 3.2); nothing tracked changes."""
    with effort.stage("report.map"):
        snapshot = layout.capture(config)
        lay, project = snapshot.layout, config.project
        order = [project.names_from, *(v for v in snapshot.versions if v != project.names_from)]
        scans: dict[str, Any] = {}
        for version in order:
            for row in infer.scan(snapshot).get(version, ()):
                scans.setdefault(row[0], row)
        sdk = infer.sdk(snapshot)
        roms = {v: snapshot.versions[v].rom.read_bytes() for v in snapshot.versions}
        spans: dict[str, list[tuple[int, int, Any]]] = defaultdict(list)
        for m in lay.members.values():
            for p in m.placements:
                if p.section != ".text":
                    spans[p.version].append((p.vram, p.vram + p.size, p))
        starts = {v: sorted(rows, key=lambda r: r[0]) for v, rows in spans.items()}
        names = {id(p): m.name for m in lay.members.values() for p in m.placements}
        place = Segments(snapshot)
        rows = []
        for name, m in sorted(lay.members.items()):
            unit, fuzzy = layout.unit_of(snapshot, name), lay.fuzzy.get(name)
            scanned = scans.get(name)
            version = m.reference(project.names_from)
            loads: list[str] = []
            strings: list[str] = []
            for address in sorted(scanned[1]) if scanned else ():
                table = starts.get(version, [])
                at = bisect_right([r[0] for r in table], address) - 1
                if at >= 0 and address < table[at][1]:
                    p = table[at][2]
                    loads.append(names[id(p)])
                    strings += _strings(roms[version], p.rom_start + address - p.vram)
            rows.append(
                {
                    "name": name,
                    "kind": m.kind,
                    "state": "c" if unit else m.state,
                    "module": m.group,
                    "unit": unit.path if unit else None,
                    "fuzzy": fuzzy["path"] if fuzzy else None,
                    "segment": place.of(m) or "-",
                    "sdk": name in sdk,
                    "sdk_name": sdk[name].name if name in sdk else "",
                    "sdk_module": sdk[name].module if name in sdk else "",
                    "sdk_library": sdk[name].library if name in sdk else "",
                    "padded": bool(scanned and scanned[3]),
                    "placements": [
                        {
                            "version": p.version,
                            "section": p.section,
                            "rom_start": p.rom_start,
                            "rom_end": p.rom_end,
                            "vram": p.vram,
                        }
                        for p in m.placements
                    ],
                    "calls": sorted(scanned[2]) if scanned else [],
                    "loads": sorted(set(loads)),
                    "strings": strings[:4],
                }
            )
        included, declared = headers.included_by(snapshot), headers.declares(snapshot)
        owners = defaultdict(list)
        for g in lay.groups.values():
            owners[g.header].append(g.name)
        files = headers.sources(snapshot)
        document = {
            "schema": 1,
            "digest": lay.digest,
            "versions": list(snapshot.versions),
            "names_from": project.names_from,
            "segments": place.listed({place.of(m) for m in lay.members.values()}),
            "modules": [
                {
                    "name": g.name,
                    "segment": (
                        place.of(live)
                        if (live := next((lay.members[m] for m in g.members if m in lay.members), None))
                        else ""
                    )
                    or g.segment,  # a member no version has is skipped: the group still lists it
                    "subsystem": g.subsystem,
                    "library": g.library,
                    "source": g.source,
                    "header": g.header,
                    "evidence": g.evidence,
                    "sdk": g.sdk,
                    "members": list(g.members),
                }
                for g in sorted(lay.groups.values(), key=lambda g: g.name)
            ],
            "units": [
                {
                    "path": u.path,
                    "kind": u.kind,
                    "module": u.group,
                    "members": list(u.members),
                    "toolchain": u.toolchain,
                    "options": u.options,
                    "withheld": list(u.withheld),
                }
                for u in sorted(lay.units.values(), key=lambda u: u.path)
            ],
            "members": rows,
            "headers": [
                {
                    "path": h,
                    "modules": sorted(owners.get(h, ())),
                    "declares": sorted(set(declared.get(h, ()))),
                    "includes": sorted(t for t, by in included.items() if h in by),
                    "included_by": sorted(included.get(h, ())),
                }
                for h in files
            ],
            "counts": {
                "src_c": len(headers.sources(snapshot, "src", ".c")),
                "include_h": len(files),
                "units": len(lay.units),
                "modules": len(lay.groups),
                "members": len(lay.members),
            },
        }
        path = project.root / ".unbake/map.json"
        store.write(path, json.dumps(document, sort_keys=True).encode())
        result: dict[str, Any] = {"path": ".unbake/map.json", "digest": lay.digest, "counts": document["counts"]}
        if params.get("moves"):
            result["moves"] = _moves_stream(snapshot)
        return result


def _moves_stream(snapshot: Snapshot) -> Json:
    """Write the restructure move stream to .unbake/moves.jsonl; its count and files per target folder."""
    from unbake.layout.subsystems import moves

    rows = [{"layout": snapshot.layout.digest}, *moves(snapshot)]
    out = snapshot.config.project.root / ".unbake/moves.jsonl"
    store.write(out, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())
    folders = Counter(r["target"].rpartition("/")[0] for r in rows if r.get("op") == "source")
    return {"path": ".unbake/moves.jsonl", "count": len(rows) - 1, "folders": dict(sorted(folders.items()))}
