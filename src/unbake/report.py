"""Progress from the source tree: credit per name and version, the README block and the map for Recipe Maker."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

from unbake import config as configuration
from unbake import effort, perfect, present, rescore, store
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


def _facts(config: Any) -> rescore.Facts:
    """Per version the symbol_addrs facts: name -> (vram, rom, size, type)."""
    read = import_module("unbake.api").symbols
    return {v: read(config, v) for v in config.project.version}


def _check(config: Any, tree: Any) -> Json | None:
    """The last check's document when it checked this tree (`digest`), else None."""

    path = Path(config.project.root) / ".unbake/check.json"

    def read() -> Json | None:
        with suppress(FileNotFoundError, ValueError, TypeError):
            loaded: Json = json.loads(path.read_bytes())
            if loaded.get("digest") == tree.digest:
                return loaded
        return None

    stamp = path.stat().st_mtime_ns if path.exists() else 0  # a check made since the last read is read again
    return effort.memo(("check", tree.digest, stamp), read)


def record(config: Any, tree: Any, **fields: Any) -> None:
    """Keep what a check learned of this tree in `.unbake/check.json` (`digest`, `exact`, `failing`, and what a source
    check adds): what the file holds of another tree is dropped, so a verdict never outlives the text it judged."""
    path = Path(config.project.root) / ".unbake/check.json"
    kept: Json = {}
    with suppress(FileNotFoundError, ValueError, TypeError):
        loaded: Json = json.loads(path.read_bytes())
        if loaded.get("digest") == tree.digest:
            kept = loaded
    document = {"exact": [], "failing": [], **kept, **fields, "digest": tree.digest}
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
        "perfect_percent": _pct(b["perfect_bytes"], b["code_total"] + b["data_total"]),
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


def percent(credit: Credit) -> float:
    """A function's match in one version, 0..100 with two decimals: 100 for exact bytes, else its fuzzy score."""
    return 100.0 if credit.exact or credit.score >= 1.0 else round(credit.score * 100, 2)


def build_credits(
    versions: list[str],
    tree: Any,
    facts: rescore.Facts,
    exact: frozenset[str],
    scored: Mapping[str, Mapping[str, float]],
    broken: frozenset[str],
) -> tuple[Credit, ...]:
    """One row per (name, version) of the symbol_addrs facts. A name is exact in a version when C defines it and the
    version links exact; otherwise it earns its cached fuzzy score, and none when its module does not compile."""
    modules = {m.path: m for m in tree.modules}
    rows = []
    for v in versions:
        for name, (_vram, rom, size, kind) in sorted(facts.get(v, {}).items(), key=lambda item: (item[1][1], item[0])):
            path = tree.owner(v, name)
            module = modules.get(path) if path else None
            stand_in = module is None or name in module.stand_ins.get(v, ())
            built = module is not None and not stand_in and v in exact
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


def credits(config: Any, tree: Any) -> tuple[Credit, ...]:
    """Every (name, version) credit row, computed once per tree digest: the README, the overview and the map all read
    these same rows."""

    def build() -> tuple[Credit, ...]:
        facts = _facts(config)
        return build_credits(
            list(config.project.version),
            tree,
            facts,
            exact_versions(config, tree),
            rescore.rows(config, tree, facts),
            _broken(config, tree),
        )

    return effort.memo(("credits", tree.digest), build)


def scores(config: Any, tree: Any) -> dict[str, dict[str, float]]:
    """name -> version -> cached fuzzy score of every NON_MATCHING function (what `unbake score` and the API read)."""
    return rescore.rows(config, tree, _facts(config))


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
            "perfect_bytes",
        ),
        0,
    )
    rows = credits(config, tree)
    perfect_modules = perfect.census(tree, rows, exact_versions(config, tree))[1]
    totals = {v: dict(zero) for v in config.project.version}
    segments: defaultdict[str, Buckets] = defaultdict(dict)
    kinds: defaultdict[str, Buckets] = defaultdict(dict)
    for row in rows:
        whole = bool(row.member.group) and row.member.group in perfect_modules  # its module is one perfect object
        for bucket in (
            totals[row.version],
            segments[row.segment].setdefault(row.version, dict(zero)),
            kinds[row.member.state].setdefault(row.version, dict(zero)),
        ):
            function = row.member.kind == "function"
            for prefix, size, earned in (
                ("code_", row.code, row.kept),
                ("data_", row.data, row.data if row.exact else 0),
                ("functions_", int(function), int(row.exact and function)),
            ):
                bucket[prefix + "total"] += size
                bucket[prefix + "matched"] += earned
            bucket["functions_fuzzy"] += int(function and not row.exact and row.score > 0)
            bucket["code_fuzzy"] += round(row.score * (row.code - row.kept))
            bucket["data_fuzzy"] += round(row.score * row.data)
            bucket["perfect_bytes"] += (row.code + row.data) if whole else 0
    return totals, segments, kinds


def progress(config: Any, tree: Any) -> Json:
    """The per-version credit the README shows."""
    with effort.stage("report.progress"):
        return {
            "versions": {v: _tally(b) for v, b in _tallies(config, tree)[0].items()},
            "perfect": perfect.census(tree, credits(config, tree), exact_versions(config, tree))[0],
        }


def current(config: Any, tree: Any) -> Json:
    with effort.stage("report.current"):
        totals, segments, kinds = _tallies(config, tree)
        debt: Json = {"stale": True, "action": "unbake check"}
        document = _check(config, tree)
        if document is not None and "counts" in document:
            debt = {"digest": tree.digest, "counts": document["counts"], "type_debt": document.get("type_debt", {})}
        return {
            "digest": tree.digest,
            "versions": {v: _tally(b) for v, b in totals.items()},
            "segments": {s: {v: _tally(b) for v, b in rows.items()} for s, rows in segments.items()},
            "kinds": {k: {v: _tally(b) for v, b in rows.items()} for k, rows in kinds.items()},
            "debt": debt,
            "perfect": perfect.census(tree, credits(config, tree), exact_versions(config, tree))[0],
        }


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


def _perfect_line(label: str, perfect_bytes: int, total: int, cells: int) -> str:
    """The second art bar: bytes of modules that are one proven object with proven flags and only exact members."""
    percent = _pct(perfect_bytes, total)
    shade = int(percent // (100 / cells))
    return f"{label} [{'▒' * shade}{'░' * (cells - shade)}]  {percent:5.2f}%  {perfect_bytes:,} of {total:,} bytes"


def _rom_sha256(config: Any, version: str) -> str:
    path = Path(config.project.root) / "roms" / f"baserom.{version}.z64"
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
            k: sum(versions[v][k] for v in project.version)
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
        width = max(map(len, ("all", *project.version))) + 1
        lines = [
            _line(label.ljust(width), bucket, cells) + " bytes"
            for label, bucket in [("all", total), *((v, versions[v]) for v in project.version)]
        ]
        lines.append(
            _perfect_line(
                "perfect".ljust(width),
                sum(versions[v]["perfect_bytes"] for v in project.version),
                total["code_total"] + total["data_total"],
                cells,
            )
        )
        blocks = ["<pre><code>" + "</code><br><code>".join(lines) + "</code></pre>"]
        if project.readme_map:
            image = f'<img src="{present.overview_path(project.readme_map)}" width="{settings["map_width"]}">'
            link, cartridge = (project.readme_map_link, project.version[project.readme_map].meta.get("cartridge_id"))
            anchor = f"#{cartridge.lower()}" if cartridge else ""  # the site names each ROM by its cartridge id
            blocks.append(f'<a href="{link}{anchor}">{image}</a>' if link else image)
        for v in project.version:
            meta = project.version[v].meta
            details = ", ".join(meta[k] for k in ("cartridge_id", "region") if meta.get(k))
            header = f"| {v}" + (f" ({details})" if details else "") + "."
            header += f" {meta['description']}" if meta.get("description") else ""
            header += f" SHA256 `{_rom_sha256(config, v)}` |"
            rows = [_line(k.ljust(9), versions[v], cells, k) for k in ("code", "data", "functions")]
            blocks.append(header + "\n|---|\n| <pre><code>" + "</code><br><code>".join(rows) + "</code></pre> |")
        text = text[: text.index(begin) + len(begin)] + "\n" + "\n\n".join(blocks) + "\n" + text[text.index(end) :]
        first, last = settings["dependencies_begin"], settings["dependencies_end"]
        if text.count(first) == 1 and text.count(last) == 1 and text.index(first) < text.index(last):
            table = "\n".join(_dependencies(config))
            text = text[: text.index(first) + len(first)] + "\n" + table + "\n" + text[text.index(last) :]
        return text


def _dependencies(config: Any) -> list[str]:
    """The README table of the pinned build tools, from the pins the generated Makefile fetches: each row names the
    tool by its own identity and links the exact archive by its sha256 (where it is hosted shows only in the URL)."""
    project = config.project
    rows = configuration.toolchains()
    rules = getattr(project, "file", {})
    used = sorted({project.toolchain, *(r["toolchain"] for r in rules.values() if r.get("toolchain"))})
    lines = ["| Tool | Used for | Pinned archive |", "|---|---|---|"]
    for id in used:
        for download in rows.get(id, {}).get("downloads", [])[:1]:
            title = rows[id].get("title", id)
            lines.append(f"| {title} | compiles C (`{id}`) | {_archive(download)} |")
    for row in configuration.load_resource("repo.toml")["make"]["tools"].values():
        if "download" in row and any(f"{{{row['host']}}}" in str(rows.get(id)) for id in used):
            lines.append(
                f"| {row.get('title', row['host'])} | assembles and links every unit | {_archive(row['download'])} |"
            )
    return lines


def _archive(download: Mapping[str, Any]) -> str:
    name = str(download["url"]).rsplit("/", 1)[-1]
    return f"[{name}]({download['url']}) sha256 `{str(download['sha256'])[:12]}`"


def files(config: Any, tree: Any) -> dict[str, bytes]:
    """README.md with its progress block and the one image it embeds; nothing else is written."""
    with effort.stage("report.files"):
        document = progress(config, tree)
        source = (Path(config.project.root) / "README.md").read_text()
        return {"README.md": readme(source, config, document).encode(), **present.files(config, tree)}


def run(config: Any, params: Json) -> Json:
    with effort.stage("report.run"):
        pending = sum(p.is_file() for p in (Path(config.project.root) / ".unbake/inbox").glob("*.json"))
        return {**current(config, _capture(config)), "inbox": pending}


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
