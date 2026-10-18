"""Published progress schema 2: ROM accounting and a source-linked map from the report's credit rows."""

from __future__ import annotations

from typing import Any

from unbake import blobs, libasm, publish, report
from unbake import config as configuration
from unbake.contracts import Finding, Json, Refusal

_C_STATES = frozenset({"c", "fuzzy"})


def locations(config: Any, tree: Any) -> dict[tuple[str, str], int]:
    """Definition lines in their owning sources, scanned once per module."""
    found: dict[tuple[str, str], int] = {}
    for module in tree.modules:
        text = (config.project.root / module.path).read_text(encoding="utf-8", errors="replace")
        if configuration.is_assembly(module.path):
            for name in libasm.scan(text):
                for number, line in enumerate(text.splitlines(), 1):
                    if line.strip().startswith(name + ":"):
                        found.setdefault((module.path, name), number)
            continue
        for item in publish.items(text):
            if item.kind == "function":
                found.setdefault((module.path, item.name), text.count("\n", 0, item.start) + 1)
    return found


def document(config: Any, tree: Any, current: Json) -> Json:
    """One cart per version and one map for names_from; all totals come from the same report."""
    carts = []
    for version, row in current["versions"].items():
        if "rom_size" not in row:
            raise Refusal(Finding("report.rom", "progress.json requires the ROM byte partition", versions=(version,)))
        carts.append(
            {
                "id": config.project.version[version].meta.get("cartridge_id", ""),
                "version": version,
                "rom_size": row["rom_size"],
                **{
                    kind: {part: row[f"{kind}_{part}"] for part in ("matched", "fuzzy", "total")}
                    for kind in ("code", "data", "functions")
                },
                "blobs": {
                    "decoded": row["blobs_decoded"],
                    "identified": row["blobs_identified"],
                    "total": row["blobs_total"],
                    "by_type": row["blobs"],
                },
                "platform": row["platform_total"],
                "unknown": row["unknown_total"],
            }
        )
    version = config.project.names_from
    lines = locations(config, tree)
    fns = []
    for row in report.credits(config, tree):
        if row.version != version or row.code <= 0:
            continue
        path = row.member.group
        line = lines.get((path, row.name)) if row.member.state in _C_STATES else None
        fns.append([row.name, row.member.rom, row.code, report.percent(row), path if line else None, line])
    spans = report.layers(config, tree, version)
    content = [
        [name, start, end - start, row["state"], row["type"]]
        for name, row in blobs.load(config)["blob"].items()
        for start, end in row.get("rom", {}).get(version, ())
    ]
    return {
        "schema": 2,
        "carts": carts,
        "map": {
            "version": version,
            "rom_size": current["versions"][version]["rom_size"],
            "fns": sorted(fns, key=lambda row: (row[1], row[0])),
            "blobs": sorted(content, key=lambda row: (row[1], row[0])),
            "spans": {kind: [list(span) for span in spans.get(kind, [])] for kind in ("data", "platform")},
        },
    }


def unmatched(config: Any, tree: Any) -> list[Json]:
    """Unmatched functions, smallest first, with a deterministic name/version tie break."""
    return sorted(
        (
            {
                "name": row.name,
                "version": row.version,
                "size": row.code,
                "percent": report.percent(row),
                "path": row.member.group or None,
                "state": "fuzzy" if report.fuzzy(row) else "unmatched",
            }
            for row in report.credits(config, tree)
            if row.code > 0 and not row.exact
        ),
        key=lambda row: (row["size"], row["name"], row["version"]),
    )
