"""Progress from the source tree: credit per name and version, the README block and the map for Recipe Maker."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

from unbake import blobs, effort, platform, present, rescore, store
from unbake import config as configuration
from unbake.contracts import Finding, Json, Refusal, sha256_file


def _pct(part: float, whole: float) -> float:
    return (10000 * int(part) // int(whole)) / 100 if whole else 0.0


class Symbol(NamedTuple):
    """What a credit row knows of its name: function or data, the module (source path) holding it, whether it is
    C (`c`), a stand-in (`stand-in`), a NON_MATCHING body behind a stand-in (`fuzzy`) or in no module, and its ROM
    offset."""

    kind: str
    group: str
    state: str
    rom: int


def _capture(config: Any) -> Any:
    return import_module("unbake.tree").capture(config)


def facts(config: Any) -> rescore.Facts:
    """Per version the symbol_addrs facts: name -> (vram, rom, size, type), each size the file leaves out filled in."""
    read = import_module("unbake.api").symbols
    return {v: _sized(config, v, read(config, v)) for v in config.project.version}


def _sized(config: Any, v: str, table: Mapping[str, Any]) -> dict[str, Any]:
    """`table` with a missing size taken from the ROM: a name of a code or data segment holds the bytes up to the next
    name's address, the last one up to its segment's end; a name in a bin segment holds none, and a size the file gives
    is cut at its segment's end. Every byte from a segment's first name on is counted once, whatever the names in
    between."""
    cuts = import_module("unbake.versions").segments(config, v)
    kinds = [s.kind for s in config.project.version[v].segments]
    starts = sorted({row[1] for row in table.values() if row[1] is not None})
    out = dict(table)
    for name, (vram, rom, size, kind) in table.items():
        if rom is None:
            continue
        at = bisect_right([c[1] for c in cuts], rom) - 1
        if at < 0 or kinds[at] == "bin":
            out[name] = (vram, rom, 0, kind)
            continue
        if size is not None:
            out[name] = (vram, rom, max(0, min(size, cuts[at][2] - rom)), kind)
            continue
        later = bisect_right(starts, rom)
        end = min(cuts[at][2], starts[later]) if later < len(starts) else cuts[at][2]
        out[name] = (vram, rom, max(0, end - rom), kind)
    return out


def _check(config: Any, tree: Any) -> Json | None:
    """The last check's document when it checked this tree (`digest`), else None."""

    path = Path(config.project.root) / ".unbake/check.json"

    with suppress(FileNotFoundError, ValueError, TypeError):
        loaded: Json = json.loads(path.read_bytes())
        same_tree = loaded.get("digest") == tree.digest
        if "build_inputs" in loaded:
            valid = loaded["build_inputs"] == import_module("unbake.build").project_inputs(config.project)
            if valid:
                return (
                    loaded
                    if same_tree
                    else {k: loaded[k] for k in ("exact", "failing", "build_inputs", "build_failed") if k in loaded}
                )
            return {**loaded, "exact": []} if same_tree else None
        if same_tree:
            return loaded
    return None


def record(config: Any, tree: Any, **fields: Any) -> None:
    """Keep what a check learned of this tree in `.unbake/check.json` (`digest`, `exact`, `failing`, and what a source
    check adds): what the file holds of another tree is dropped, so a verdict never outlives the text it judged."""
    path = Path(config.project.root) / ".unbake/check.json"
    kept = _check(config, tree) or {}
    document = {"exact": [], "failing": [], **kept, **fields, "digest": tree.digest}
    store.write(path, json.dumps(document, sort_keys=True).encode())


def record_build(root: Path, inputs: str, exact: list[str], failed: str | None = None) -> None:
    """The build verdict in the same check document as source debt; each has its own input identity. `failed` names a
    build that did not complete (make exited nonzero): it says nothing about how much code matches."""
    path = root / ".unbake/check.json"
    kept: Json = {}
    with suppress(FileNotFoundError, ValueError, TypeError):
        kept = json.loads(path.read_bytes())
    document = {**kept, "exact": sorted(exact), "failing": [], "build_inputs": inputs, "build_failed": failed}
    store.write(path, json.dumps(document, sort_keys=True).encode())


def exact_versions(config: Any, tree: Any) -> frozenset[str]:
    """The versions whose last link matched the ROM (the `link` verdict the check recorded for this tree)."""
    return frozenset((_check(config, tree) or {}).get("exact", ()))


def _broken(config: Any, tree: Any) -> frozenset[str]:  # modules the last check could not compile: no credit
    return frozenset((_check(config, tree) or {}).get("failing", ()))


Buckets = dict[str, dict[str, int]]  # version -> credit counters


def _tally(b: Json) -> Json:
    return {
        **b,
        "fuzzy_bytes": (fuzzy := b["code_fuzzy"] + b["data_fuzzy"]),
        "code_percent": _pct(b["code_matched"], b["code_total"]),
        "data_percent": _pct(b["data_matched"], b["data_total"]),
        "fuzzy_percent": _pct(b["code_matched"] + b["data_matched"] + fuzzy, b["code_total"] + b["data_total"]),
    }


class Credit(NamedTuple):
    """The credit one name earns in one version: its code and data bytes, the exact code bytes that earn credit
    (`kept`), whether it is exact and its fuzzy score."""

    name: str
    member: Symbol
    unit: Any
    segment: str
    version: str
    code: int
    data: int
    kept: int
    exact: bool
    score: float
    platform: int = 0  # bytes of the symbol that are identified platform (boot, SDK microcode): counted there


def percent(credit: Credit) -> float:
    """A function's match in one version, 0..100 with two decimals: 100 for exact bytes, else its fuzzy score."""
    return 100.0 if credit.exact or credit.score >= 1.0 else round(credit.score * 100, 2)


def fuzzy(credit: Credit) -> bool:
    """Whether an unmatched credit row earns a positive measured score, independent of source form."""
    return not credit.exact and credit.score > 0


def build_credits(
    versions: list[str],
    tree: Any,
    facts: rescore.Facts,
    exact: frozenset[str],
    scored: Mapping[str, Mapping[str, float]],
    broken: frozenset[str],
    extents: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[Credit, ...]:
    """One row per (name, version) of the symbol_addrs facts. A name is exact in a version when C defines it and the
    version links exact; otherwise it earns its cached fuzzy score, and none when its module does not compile. A name
    with no ROM address (an absolute symbol) holds no ROM bytes and gets no row. A data name no file owns (a label)
    that lies inside the compiled size of a C-built object (`extents`: version -> name -> ELF st_size) is that
    object's bytes and earns its credit; a label outside every compiled extent stays unowned."""
    modules = {m.path: m for m in tree.modules}
    rows = []
    for v in versions:
        held = [item for item in facts.get(v, {}).items() if item[1][1] is not None]
        owners = {name: tree.owner(v, name) for name, _ in held}
        sizes = (extents or {}).get(v, {})
        end, container = -1, None  # end of the bytes credited so far; the C object whose compiled size covers here
        for name, (_vram, rom, size, kind) in sorted(held, key=lambda i: (i[1][1], owners[i[0]] is None, i[0])):
            path = owners[name]
            module = modules.get(path) if path else None
            stand_in = module is None or name in module.stand_ins.get(v, ())
            if rom < end:
                size = 0  # it starts inside an earlier row's extent (a label, an alias): those bytes are counted once
            elif path is None and kind != "func" and container and rom < container[0]:
                _, path, module, stand_in = container  # a label inside a C object's compiled size: the object's bytes
            if size:
                end = rom + size
            if name in sizes and module and not stand_in:
                container = (rom + sizes[name], path, module, stand_in)
            built = (
                module is not None
                and not stand_in
                and v in exact
                and (not configuration.is_assembly(module.path) or getattr(module, "library", False))
            )
            function = kind == "func"
            state = "unowned" if module is None else "stand-in" if stand_in else "c"
            state = "fuzzy" if module is not None and stand_in and name in module.fuzzy else state
            code, data = (size, 0) if function else (0, size)
            score = 0.0 if built else scored.get(name, {}).get(v, 0.0)
            if module is not None and module.path in broken and not built:
                score = 0.0  # it does not compile: no credit
            rows.append(
                Credit(
                    name,
                    Symbol("function" if function else "data", path or "", state, rom),
                    module,
                    module.segment if module else "unknown",
                    v,
                    code,
                    data,
                    code if built else 0,
                    built,
                    score,
                )
            )
    return tuple(rows)


def _extents(config: Any, tree: Any) -> dict[str, dict[str, int]]:
    """Per version the compiled extent of every data name a C module defines, measured in build/<v>/<module>.o with
    `versions.defined`: the symbol's ELF st_size, else (the old compiler writes none) the bytes up to the next symbol
    of its section or the section's end. A version or module with no built object has none: report never guesses."""
    defined, root = import_module("unbake.versions").defined, Path(config.project.root)
    out: dict[str, dict[str, int]] = {}
    for module in tree.modules:
        for v, names in module.owns.items():
            obj = root / "build" / v / f"{module.path}.o"
            if not obj.is_file() or not names:
                continue
            found, sections = defined(obj)
            for name in set(names) - set(module.stand_ins.get(v, ())):
                if name not in found:
                    continue
                section, value, size = found[name]
                after = [other for sec, other, _ in found.values() if sec == section and other > value]
                size = size or min(after, default=sections.get(section, 0)) - value
                if size > 0:
                    out.setdefault(v, {})[name] = size
    return out


def credits(config: Any, tree: Any) -> tuple[Credit, ...]:
    """Every (name, version) credit row, computed once per tree digest: the README, the overview and the map all read
    these same rows."""

    def build() -> tuple[Credit, ...]:
        held = facts(config)
        rows = build_credits(
            list(config.project.version),
            tree,
            held,
            exact_versions(config, tree),
            rescore.rows(config, tree, held),
            _broken(config, tree),
            _extents(config, tree),
        )
        spans = {v: blobs.merge(platform.ranges(config, v)) for v in config.project.version}
        return off_platform(rows, spans)

    return effort.memo(("credits", tree.digest, platform.sha(config)), build)


def _on_platform(row: Credit, spans: Mapping[str, blobs.Spans]) -> int:
    size = row.code + row.data
    if not size:
        return 0
    mine = [(row.member.rom, row.member.rom + size)]
    return size - blobs.length(blobs.subtract(mine, spans.get(row.version, [])))


def off_platform(rows: tuple[Credit, ...], spans: Mapping[str, blobs.Spans]) -> tuple[Credit, ...]:
    """The credit rows with the identified platform bytes (boot, SDK microcode) taken out of each row: those bytes
    are platform in every version, never our code or data, whichever symbol covers them."""
    out = []
    for row in rows:
        if hit := _on_platform(row, spans):
            code, data = (row.code - hit, 0) if row.code else (0, row.data - hit)
            row = row._replace(code=code, data=data, kept=min(row.kept, code), platform=hit)
        out.append(row)
    return tuple(out)


def platform_overlaps(config: Any, tree: Any) -> list[Finding]:
    """A non-blocking finding per symbol that covers identified platform bytes (the report counts them as platform)."""
    return [
        Finding(
            "report.platform-overlap",
            f"{row.version}: {row.name} at rom 0x{row.member.rom:X} covers {row.platform} identified platform bytes "
            "(boot or SDK microcode): they count as platform, not as matched code or data",
            unit=row.name,
            versions=(row.version,),
            action=f"cut {row.name} where the platform span starts, or drop the row",
            blocking=False,
        )
        for row in credits(config, tree)
        if row.platform
    ]


def scores(config: Any, tree: Any) -> dict[str, dict[str, float]]:
    """name -> version -> cached fuzzy score of every NON_MATCHING function (what `unbake score` and the API read)."""
    return rescore.rows(config, tree, facts(config))


def _tallies(config: Any, tree: Any) -> tuple[Buckets, dict[str, Buckets], dict[str, Buckets]]:
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
            "code_asm",
            "functions_asm",
        ),
        0,
    )
    rows = credits(config, tree)
    totals = {v: dict(zero) for v in config.project.version}
    segments: defaultdict[str, Buckets] = defaultdict(dict)
    kinds: defaultdict[str, Buckets] = defaultdict(dict)
    for row in rows:
        for bucket in (
            totals[row.version],
            segments[row.segment].setdefault(row.version, dict(zero)),
            kinds[row.member.state].setdefault(row.version, dict(zero)),
        ):
            function = row.member.kind == "function" and row.code > 0
            for prefix, size, earned in (
                ("code_", row.code, row.kept),
                ("data_", row.data, row.data if row.exact else 0),
                ("functions_", int(function), int(row.exact and function)),
            ):
                bucket[prefix + "total"] += size
                bucket[prefix + "matched"] += earned
            assembly = row.unit is not None and configuration.is_assembly(row.unit.path)  # library assembly
            bucket["code_asm"] += row.kept if assembly else 0
            bucket["functions_asm"] += int(assembly and row.exact and function)
            bucket["functions_fuzzy"] += int(function and fuzzy(row))
            bucket["code_fuzzy"] += round(row.score * (row.code - row.kept))
            bucket["data_fuzzy"] += round(row.score * row.data)
    return totals, segments, kinds


def partition(config: Any, v: str, table: Mapping[str, Any], rom_size: int) -> Json:
    """Every byte of one version's ROM once: the credit rows of `table` (code, then data), platform spans, then
    the decoded and identified blobs, then proven fill (`blobs.fill`, read from the ROM), then unknown. `overlap` and
    `outside` (credited bytes counted twice or beyond the ROM) must be 0; `spans` holds the disjoint ranges of each
    class."""
    held = [(rom, rom + size, kind) for _vram, rom, size, kind in table.values() if rom is not None and size]
    code = [(a, b) for a, b, kind in held if kind == "func"]
    data = [(a, b) for a, b, kind in held if kind != "func"]
    document = blobs.load(config)
    decoded, identified = blobs.ranges(document, v)
    path = Path(config.project.root) / "roms" / f"baserom.{v}.z64"
    rom = path.read_bytes() if path.is_file() else None
    rom = rom if rom is not None and len(rom) == rom_size else None
    result = blobs.sweep(rom_size, code, data, platform.ranges(config, v), decoded, identified, rom)
    result["blobs_no_row"] = blobs.no_row(result, document, v)
    return result


def _rom_size(config: Any, v: str) -> int | None:
    path = Path(config.project.root) / "roms" / f"baserom.{v}.z64"
    return path.stat().st_size if path.is_file() else None


def partitions(config: Any, tree: Any) -> dict[str, Json]:
    """The partition of every version whose ROM is present, once per tree, blobs file and ROM size.
    A byte counted twice, a range beyond the ROM or parts not summing to the ROM size: blocking `report.partition`."""

    def build() -> dict[str, Json]:
        tables: dict[str, dict[str, Any]] = {v: {} for v in config.project.version}
        for row in credits(config, tree):  # the rows the report counts, so the partition shows what it sums
            kind = "func" if row.member.kind == "function" else "data"
            tables[row.version][row.name] = (0, row.member.rom, row.code + row.data, kind)
        out = {}
        for v in config.project.version:
            if (size := _rom_size(config, v)) is None:
                continue
            parts = partition(config, v, tables[v], size)
            total = sum(parts[k] for k in ("code", "data", "platform", "blobs", "unknown"))
            if parts["overlap"] or parts["outside"] or parts["unknown"] < 0 or total != size:
                why = (
                    f"{parts['overlap']} bytes counted twice, {parts['outside']} beyond the ROM, sum {total} of {size}"
                )
                raise Refusal(Finding("report.partition", f"{v}: {why}", versions=(v,)))
            out[v] = parts
        return out

    sizes = tuple(_rom_size(config, v) for v in config.project.version)
    return effort.memo(("partition", tree.digest, blobs.sha(config), platform.sha(config), sizes), build)


def accounts(config: Any, tree: Any) -> dict[str, Json]:
    """Per version the blob fields of the report: ROM size, platform, blobs (decoded, identified), unknown, census."""
    document = blobs.load(config)
    return {
        v: {
            "rom_size": p["rom_size"],
            "platform_total": p["platform"],
            "blobs_total": p["blobs"],
            "blobs_decoded": p["blobs_decoded"],
            "blobs_identified": p["blobs_identified"],
            "blobs_fill": p["blobs_fill"],
            "blobs_no_row": p["blobs_no_row"],
            "unknown_total": p["unknown"],
            "blobs": blobs.census(document, v),
        }
        for v, p in partitions(config, tree).items()
    }


def layers(config: Any, tree: Any, v: str) -> Json:
    """The disjoint ranges of one version, by class (code, data, platform, decoded, identified), for drawing."""
    spans: Json = partitions(config, tree).get(v, {}).get("spans", {})
    return spans


def _versions(config: Any, tree: Any, totals: Buckets) -> dict[str, Json]:
    held = accounts(config, tree)
    return {v: {**_tally(b), **held.get(v, {})} for v, b in totals.items()}


def progress(config: Any, tree: Any) -> Json:
    """The per-version credit the README shows."""
    with effort.stage("report.progress"):
        return {"versions": _versions(config, tree, _tallies(config, tree)[0])}


def current(config: Any, tree: Any) -> Json:
    with effort.stage("report.current"):
        totals, segments, kinds = _tallies(config, tree)
        debt: Json = {"stale": True, "action": "unbake check"}
        document = _check(config, tree)
        if document is not None and "counts" in document:
            debt = {"digest": tree.digest, "counts": document["counts"], "type_debt": document.get("type_debt", {})}
        return {
            "digest": tree.digest,
            "versions": _versions(config, tree, totals),
            "segments": {s: {v: _tally(b) for v, b in rows.items()} for s, rows in segments.items()},
            "kinds": {k: {v: _tally(b) for v, b in rows.items()} for k, rows in kinds.items()},
            "debt": debt,
        }


def _line(label: str, bucket: Json, cells: int, kind: str = "bytes") -> str:
    parts = ("code", "data") if kind == "bytes" else (kind,)
    matched = sum(bucket[p + "_matched"] for p in parts)
    total = sum(bucket[p + "_total"] for p in parts)
    asm = bucket.get(kind + "_asm", 0)  # matched library assembly, shown apart from matched C
    suffix = f" ({asm:,} asm)" if asm else ""
    return _bar(label, matched, matched + sum(bucket[p + "_fuzzy"] for p in parts), total, cells) + suffix


def _rom_line(label: str, bucket: Json, cells: int) -> str:
    """The whole ROM: matched code, data and blobs plus the platform bytes, reaching over fuzzy code and identified
    blobs, against the exact ROM size."""
    sure = bucket["code_matched"] + bucket["data_matched"] + bucket["blobs_decoded"] + bucket["platform_total"]
    reach = sure + bucket["code_fuzzy"] + bucket["data_fuzzy"] + bucket["blobs_identified"]
    return _bar(label, sure, reach, bucket["rom_size"], cells)


def _bar(label: str, matched: int, reach: int, total: int, cells: int) -> str:
    """One art bar: █ for `matched` bytes, ▒ for those up to `reach`, ░ for the rest of `total`."""
    percent = _pct(matched, total)
    fuzzy = _pct(reach, total)
    solid = int(percent // (100 / cells))
    partial = max(0, min(cells - solid, int(fuzzy // (100 / cells)) - solid))
    bar = "█" * solid + "▒" * partial + "░" * (cells - solid - partial)
    suffix = f" (~{fuzzy:.2f}%)" if fuzzy != percent else ""
    return f"{label} [{bar}]  {percent:5.2f}%{suffix}  {matched:,} of {total:,}"


_BLOB_KEYS = (
    "rom_size",
    "platform_total",
    "blobs_total",
    "blobs_decoded",
    "blobs_identified",
    "blobs_fill",
    "blobs_no_row",
    "unknown_total",
)


def _blobs_line(label: str, held: Json, cells: int) -> str:
    """Blobs: █ decoded (proven fill included), ▒ identified, ░ the rest, over the ROM bytes outside code, data and
    platform."""
    return _bar(label, held["blobs_decoded"], held["blobs_total"], held["blobs_total"] + held["unknown_total"], cells)


def _rom_sha256(config: Any, version: str) -> str:
    path = configuration.rom(config.project, version)
    return effort.memo(("rom-sha256", str(path)), lambda: sha256_file(path))


def readme(text: str, config: Any, report: Json) -> str:
    with effort.stage("report.readme"):
        settings = configuration.load_resource("repo.toml")["readme"]
        begin, end, cells = (settings[k] for k in ("begin", "end", "bar"))
        if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) >= text.index(end):
            raise Refusal(
                Finding(
                    "repo.readme_markers", "README.md needs one begin and one end progress marker", path="README.md"
                )
            )
        project, versions = config.project, report["versions"]
        total = {
            k: sum(versions[v].get(k, 0) for v in project.version)
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
                "code_asm",
                "functions_asm",
            )
        }
        counted = all("rom_size" in versions[v] for v in project.version)  # every ROM was present for the partition
        width = max(map(len, ("all", *project.version))) + 1
        if counted:
            total.update({k: sum(versions[v][k] for v in project.version) for k in _BLOB_KEYS})
        line = _rom_line if counted else _line
        lines = [
            line(label.ljust(width), bucket, cells) + " bytes"
            for label, bucket in [("all", total), *((v, versions[v]) for v in project.version)]
        ]
        blocks = ["<pre><code>" + "</code><br><code>".join(lines) + "</code></pre>"]
        if project.readme_map:
            image = f'<img src="{present.overview_path(project.readme_map)}" width="{settings["map_width"]}">'
            link, cartridge = (
                project.readme_map_link,
                configuration.rom_meta(project, project.readme_map).get("cartridge_id"),
            )
            anchor = f"#{cartridge.lower()}" if cartridge else ""  # the site names each ROM by its cartridge id
            blocks.append(f'<a href="{link}{anchor}">{image}</a>' if link else image)
        for v in project.version:
            meta = configuration.rom_meta(project, v)
            details = ", ".join(meta[k] for k in ("cartridge_id", "region") if meta.get(k))
            header = f"| {v}" + (f" ({details})" if details else "") + "."
            header += f" {meta['description']}" if meta.get("description") else ""
            header += f" SHA256 `{_rom_sha256(config, v)}` |"
            rows = [_line(k.ljust(9), versions[v], cells, k) for k in ("code", "data", "functions")]
            if "rom_size" in versions[v]:
                rows.append(_blobs_line("blobs".ljust(9), versions[v], cells))
            blocks.append(header + "\n|---|\n| <pre><code>" + "</code><br><code>".join(rows) + "</code></pre> |")
        text = text[: text.index(begin) + len(begin)] + "\n" + "\n\n".join(blocks) + "\n" + text[text.index(end) :]
        first, last = settings["dependencies_begin"], settings["dependencies_end"]
        if text.count(first) == 1 and text.count(last) == 1 and text.index(first) < text.index(last):
            links = "\n".join(_dependencies(config))
            text = text[: text.index(first) + len(first)] + "\n" + links + "\n" + text[text.index(last) :]
        return text


def _dependencies(config: Any) -> list[str]:
    """The README list of the build tools the generated Makefile fetches: one bullet per tool, its short name linked
    to its source repository. Both come from the tool's `dependency` row, never from its download: the README links
    no archive or binary. A fetched tool without that row, or with a download for its repository: `repo.dependency`."""
    project = config.project
    rows = configuration.toolchains()
    rules = getattr(project, "file", {})
    used = sorted({project.toolchain, *(r["toolchain"] for r in rules.values() if r.get("toolchain"))})
    fetched = [(id, rows[id]) for id in used if rows.get(id, {}).get("downloads")]
    for row in configuration.load_resource("repo.toml")["make"]["tools"].values():
        if "download" in row and any(f"{{{row['host']}}}" in str(rows.get(id)) for id in used):
            fetched.append((row["host"], row))
    return [_dependency(id, row) for id, row in fetched]


def _dependency(id: str, row: Mapping[str, Any]) -> str:
    dependency = row.get("dependency")
    repository = str((dependency or {}).get("repository", ""))
    if not dependency or "/releases/" in repository or repository.endswith((".gz", ".zip", ".xz", ".bz2", ".tar")):
        raise Refusal(
            Finding(
                "repo.dependency",
                f"{id}: the README dependency needs a `dependency` row with a short name and its source repository "
                f"URL (not a release download), got {dependency!r}",
                path="README.md",
            )
        )
    return f"- [{dependency['name']}]({repository})"


def publish_hold(config: Any, tree: Any) -> Finding | None:
    """The one rule for moving published progress (README block, overview, progress.json): a non-blocking finding
    when it must stay as it is, else None. Progress moves only on the verdict of a completed build: with no verdict
    every version reads as unmatched and a failed build matched nothing, so the numbers would be zeros, not progress."""
    check = _check(config, tree)
    if check is None:
        return Finding(
            "sync.unchecked",
            "no check verdict for this tree, so the README progress block, overview and progress.json were not touched",
            path="README.md",
            blocking=False,
            action="run make check, then unbake sync",
        )
    if failure := check.get("build_failed"):
        return Finding(
            "sync.build_failed",
            f"the last build failed ({failure}), which says nothing about matching, so the README progress block, "
            "overview and progress.json were not touched",
            path="README.md",
            blocking=False,
            action="fix the build, run make check, then unbake sync",
        )
    return None


def progress_files(config: Any, tree: Any) -> dict[str, bytes]:
    """progress.json and each version's map file for this tree, by project path."""
    from unbake import progress as published

    now = current(config, tree)
    document = published.document(config, tree, now)
    configuration.validate("progress", document, "progress.json")
    out = {"progress.json": (json.dumps(document, sort_keys=True) + "\n").encode()}
    for version, row in published.maps(config, tree, now).items():
        path = published.map_path(version)
        configuration.validate("progress.map", row, path)
        out[path] = (json.dumps(row, sort_keys=True) + "\n").encode()
    return out


def files(config: Any, tree: Any, settle: bool = False) -> dict[str, bytes]:
    """README.md with its progress block, the one image it embeds and an existing progress.json with its map files;
    empty when `publish_hold` holds the tree, so the numbers on disk stay as they are. `settle`: refuse unless the
    README written would come out the same again (the rest reads only the tree, which writing them does not change)."""
    with effort.stage("report.files"):
        if publish_hold(config, tree):
            return {}
        document = progress(config, tree)
        root = Path(config.project.root)
        text = readme((root / "README.md").read_text(), config, document)
        if settle and readme(text, config, document) != text:
            raise Refusal(
                Finding(
                    "sync.changed",
                    "1 generated or stand-in files differ from `unbake sync`",
                    missing=("README.md",),
                    action="run unbake sync and commit the result",
                )
            )
        out = {"README.md": text.encode()}
        out.update(present.files(config, tree))
        if (root / "progress.json").is_file():
            out.update(progress_files(config, tree))
        return out


def run(config: Any, params: Json) -> Json:
    with effort.stage("report.run"):
        pending = sum(p.is_file() for p in (Path(config.project.root) / ".unbake/inbox").glob("*.json"))
        from unbake import progress

        tree = _capture(config)
        result = {**current(config, tree), "inbox": pending}
        notes = [asdict(f) for f in platform_overlaps(config, tree)]
        if hold := publish_hold(config, tree):
            result["notes"] = [asdict(hold), *notes]
        else:
            for path, content in progress_files(config, tree).items():
                store.write(config.project.root / path, content)
            if notes:
                result["notes"] = notes
        if params.get("next"):
            result["next"] = progress.unmatched(config, tree)
        return result


def _moves_stream(tree: Any) -> Json:
    """The restructure census Recipe Maker reads: modules per folder, and how many names each version still stands in
    for."""
    folders = Counter(m.path.rpartition("/")[0] for m in tree.modules)
    return {"count": len(tree.modules), "folders": dict(sorted(folders.items()))}


def _fakes_stream(tree: Any) -> Json:
    """The census of what is not yet C: per version the names behind stand-ins and the NON_MATCHING names."""
    versions = sorted({v for m in tree.modules for v in m.stand_ins})
    return {
        "stand_ins": {v: sum(len(m.stand_ins.get(v, ())) for m in tree.modules) for v in versions},
        "fuzzy": sum(len(m.fuzzy) for m in tree.modules),
    }


def map_command(config: Any, params: Json) -> Json:
    """Write the module map to .unbake/map.json; nothing tracked changes. `fakes` and `moves` add their census."""
    with effort.stage("report.map"):
        tree = _capture(config)
        if params.get("fakes"):
            return _fakes_stream(tree)
        document = {
            "schema": 3,
            "digest": tree.digest,
            "versions": list(config.project.version),
            "names_from": config.project.names_from,
            "modules": [
                {
                    "path": m.path,
                    "segment": m.segment,
                    "owns": {v: list(names) for v, names in m.owns.items()},
                    "fuzzy": list(m.fuzzy),
                    "stand_ins": {v: list(names) for v, names in m.stand_ins.items()},
                }
                for m in sorted(tree.modules, key=lambda m: m.path)
            ],
            "counts": {"modules": len(tree.modules), "fuzzy": sum(len(m.fuzzy) for m in tree.modules)},
        }
        store.write(Path(config.project.root) / ".unbake/map.json", json.dumps(document, sort_keys=True).encode())
        result: dict[str, Any] = {"path": ".unbake/map.json", "digest": tree.digest, "counts": document["counts"]}
        if params.get("moves"):
            result["moves"] = _moves_stream(tree)
        return result
