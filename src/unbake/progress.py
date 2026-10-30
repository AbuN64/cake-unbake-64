"""Published progress schema 2: ROM accounting and a source-linked map from the report's credit rows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from unbake import blobs, libasm, publish, report
from unbake import config as configuration
from unbake.contracts import Finding, Json, Refusal

_C_STATES = frozenset({"c", "fuzzy"})
# What a map's scores mean. Bump it whenever the same source and ROM would score differently (a consumer comparing two
# maps, like the train's exact ratchet, then re-bases instead of blaming the change that happened to land first):
# 1 = byte scores; 2 = an exact build through a FAKEMATCH scores `fakematch.SCORE`, below 100 (10-25);
# 3 = a function the build does not compile as exact shows at most `report.FUZZY_MAX` (10-25);
# 4 = inline asm in a compiled function scores like a FAKEMATCH unless the SDK exception allows it (10-25).
SCORER = 4


def locations(config: Any, tree: Any) -> dict[tuple[str, str], int]:
    """Definition lines in their owning sources, scanned once per module."""
    found: dict[tuple[str, str], int] = {}
    for module in tree.modules:
        text = (config.project.root / module.path).read_text(encoding="utf-8", errors="replace")
        if configuration.is_assembly(module.path):
            for name in libasm.scan(text):
                for number, line in enumerate(text.splitlines(), 1):
                    if line.strip().startswith((name + ":", f"LEAF({name})", f"XLEAF({name})", f"NESTED({name},")):
                        found.setdefault((module.path, name), number)
            continue
        for item in publish.items(text):
            if item.kind == "function":
                found.setdefault((module.path, item.name), text.count("\n", 0, item.start) + 1)
    return found


def header(raw: bytes) -> Json:
    """The parsed fields of a 64-byte cartridge header: PI domain 1 timing (0x00), clock rate (0x04), entry point
    (0x08), libultra release (0x0C), CRC1/CRC2 (0x10, 0x14), internal name (0x20), game ID and country (0x3B-0x3E)
    and version (0x3F). Fields only: no raw bytes, and nothing past the header (the boot code is not read)."""
    if len(raw) < 0x40:
        raise ValueError("a cartridge header is 64 bytes")

    def word(at: int) -> int:
        return int.from_bytes(raw[at : at + 4], "big")

    major, letter = raw[0x0E], raw[0x0F]
    return {
        "pi": {"release": raw[1] >> 4, "page_size": raw[1] & 0xF, "pulse_width": raw[2], "latency": raw[3]},
        "clock_rate": word(0x04),
        "entry_point": word(0x08),
        "release": f"{major // 10}.{major % 10}{chr(letter)}" if 0x41 <= letter <= 0x5A else "",
        "crc1": word(0x10),
        "crc2": word(0x14),
        "name": raw[0x20:0x34].decode("ascii", "replace").rstrip(" \0"),
        "game_id": raw[0x3B:0x3F].decode("ascii", "replace"),
        "country": configuration.load_resource("rom.toml")["country"].get(chr(raw[0x3E]), f"country {chr(raw[0x3E])}"),
        "version": raw[0x3F],
    }


def _header(config: Any, version: str) -> Json:
    """The header fields of a version's ROM: the same file the ROM partition measured."""
    with (Path(config.project.root) / "roms" / f"baserom.{version}.z64").open("rb") as rom:
        return header(rom.read(0x40))


def _mention(config: Any, path: str, name: str) -> int | None:
    """The first line of a source that spells `name` as a word, or None when it cannot be read."""
    try:
        text = (Path(config.project.root) / path).read_text(encoding="utf-8", errors="replace")
    except (OSError, AttributeError, TypeError):
        return None
    for number, line in enumerate(text.splitlines(), 1):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", line):
            return number
    return None


def _map(config: Any, tree: Any, current: Json, version: str, lines: dict[tuple[str, str], int], rows: Any) -> Json:
    fns = []
    for row in rows:
        if row.version != version or row.code <= 0:
            continue
        path = row.member.group
        has_c = row.member.state in _C_STATES and bool(path)
        line = lines.get((path, row.name)) if has_c else None
        if has_c and line is None:  # defined by a macro or renamed by it: the first line that spells the name
            line = _mention(config, path, row.name)
        fns.append([row.name, row.member.rom, row.code, report.percent(row), path if has_c else None, line])
    spans = report.layers(config, tree, version)
    content = [
        [name, start, end - start, row["state"], row["type"]]
        for name, row in blobs.load(config)["blob"].items()
        if row["type"] in blobs.CONTENT
        for start, end in row.get("rom", {}).get(version, ())
    ]
    spans = {kind: [list(span) for span in spans.get(kind, [])] for kind in ("data", "platform", "fill")}
    spans["leftover"] = [  # build residue read by a native reader: accounted bytes, not content
        [start, end]
        for start, end in blobs.merge(
            tuple(span)
            for row in blobs.load(config)["blob"].values()
            if row["type"] == "leftover"
            for span in row.get("rom", {}).get(version, ())
        )
    ]
    return {
        "schema": 1,
        "scorer": SCORER,
        "version": version,
        "rom_size": current["versions"][version]["rom_size"],
        "fns": sorted(fns, key=lambda row: (row[1], row[0])),
        "blobs": sorted(content, key=lambda row: (row[1], row[0])),
        "spans": spans,
    }


def map_path(version: str) -> str:
    """Where a version's published map lives, beside its overview image."""
    return f"docs/map-{version}.json"


def maps(config: Any, tree: Any, current: Json) -> dict[str, Json]:
    """One source-linked map per version, by version, from the same credit rows as the carts."""
    lines = locations(config, tree)
    rows = report.credits(config, tree)
    return {version: _map(config, tree, current, version, lines, rows) for version in current["versions"]}


def document(config: Any, tree: Any, current: Json) -> Json:
    """One cart per version: the ROM size, then its parts (code, data, blobs as matched or decoded of total, and the
    header block: cartridge header and boot code bytes with the header's fields), and the function counts. What the
    parts do not cover is not stored. All totals come from the same report; the maps are their own files (`maps`,
    `map_path`), so progress.json stays a small summary."""
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
                    kind: {
                        **{part: row[f"{kind}_{part}"] for part in ("matched", "total")},
                        **({"asm": row[f"{kind}_asm"]} if row.get(f"{kind}_asm") else {}),  # matched library asm
                    }
                    for kind in ("code", "data", "functions")
                },
                "blobs": {"decoded": row["blobs_decoded"], "total": row["blobs_total"]},
                "linker_fill": {"bytes": row["linker_fill_total"]},
                "header": {"bytes": row["platform_total"], **_header(config, version)},
            }
        )
    return {"schema": 2, "carts": carts}


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
