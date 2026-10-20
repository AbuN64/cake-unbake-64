"""Explicit one-time blob taxonomy migration. Normal readers accept only the current five types."""

from __future__ import annotations

import re
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import tomlkit

from unbake import blobs, journal, platform, store
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Plan, Refusal, digest, sha256

_TYPES = {
    "texture": "graphics",
    "sprite": "graphics",
    "font": "graphics",
    "layout": "graphics",
    "model": "model",
    "sound": "audio",
    "sequence": "audio",
    "level": "world",
}
_CURRENT = frozenset(
    {
        "graphics",
        "model",
        "audio",
        "world",
        "text",
        "sn-debug",
        "stale-code",
        "stale-data",
        "fill",
        "metadata",
        "unclassified",
    }
)


@dataclass
class _Reads:
    config: Config
    reads: dict[str, str | None]


def _format(old: str, value: str) -> str:
    return (
        value
        if re.search(r"(?<!\w)" + re.escape(old) + r"(?!\w)", value, re.I)
        else f"{old}: {value}"
        if value
        else old
    )


def _known(config: Config, document: Json, system: Json) -> dict[str, list[tuple[int, int]]]:
    known = {}
    for v in config.project.version:
        decoded, identified = blobs.ranges(document, v)
        known[v] = blobs.merge([*platform.ranges(config, v, system), *decoded, *identified])
    return known


def run(config: Config) -> Json:
    """Migrate reviewed types with preserved detail, atomic writes and exact byte-coverage readback."""
    root = config.project.root
    paths = (blobs.FILE, platform.FILE)
    raw = {p: (root / p).read_bytes() if (root / p).is_file() else None for p in paths}
    try:
        old = tomllib.loads((raw[blobs.FILE] or b"schema = 1\n").decode())
        edited: Any = tomlkit.parse((raw[blobs.FILE] or b"schema = 1\n").decode())
    except (ValueError, UnicodeError) as error:
        raise Refusal(Finding("blobs.migrate", f"cannot read legacy blobs: {error}", path=blobs.FILE)) from error
    rows = old.get("blob", {})
    if not isinstance(rows, dict) or any(not isinstance(row, dict) for row in rows.values()):
        raise Refusal(Finding("blobs.migrate", "blob rows must be tables", path=blobs.FILE))
    unsupported = sorted(
        n for n, row in rows.items() if row.get("type") not in _CURRENT | _TYPES.keys() | {"microcode"}
    )
    if unsupported:
        raise Refusal(
            Finding(
                "blobs.migrate",
                "classify these rows or remove them so their bytes remain unknown",
                path=blobs.FILE,
                missing=tuple(unsupported),
            )
        )
    checked = deepcopy(old)
    checked["blob"] = {n: row for n, row in checked.get("blob", {}).items() if row["type"] != "microcode"}
    for row in checked["blob"].values():
        row["type"] = _TYPES.get(row["type"], row["type"])
    configuration.validate("blobs", checked, blobs.FILE)
    if findings := blobs.check(config, checked):
        raise Refusal(*findings)
    system = platform.load(config)
    before = _known(config, old, system)
    proposed: Any = tomlkit.parse((raw[platform.FILE] or tomlkit.dumps(system).encode()).decode())
    changed = moved = 0
    for name, row in rows.items():
        kind = row["type"]
        if kind == "microcode":
            if row.get("state") == "observed" or not row.get("rom") or "parent" in row or "decoded" in row:
                raise Refusal(
                    Finding("blobs.migrate", f"{name}: platform needs reviewed physical RSP spans", path=blobs.FILE)
                )
            target = deepcopy(row)
            del target["type"]
            target["format"] = _format(kind, target.get("format", ""))
            target["sha256"] = {}
            for version, spans in row["rom"].items():
                if version not in config.project.version:
                    raise Refusal(Finding("blobs.migrate", f"{name}: unknown version {version}", path=blobs.FILE))
                hashes = []
                for start, end in spans:
                    if not 0 <= start < end <= config.project.version[version].size:
                        raise Refusal(Finding("blobs.migrate", f"{name}: invalid platform range", path=blobs.FILE))
                    hashes.append(sha256(platform.read(config, version, start, end)))
                target["sha256"][version] = hashes
            if "rsp" not in proposed["platform"]:
                proposed["platform"]["rsp"] = {}
            rsp = proposed["platform"]["rsp"]
            if name in rsp and rsp[name] != target:
                raise Refusal(
                    Finding("blobs.migrate", f"{name}: conflicts with an existing platform record", path=platform.FILE)
                )
            rsp[name] = target
            del edited["blob"][name]
            moved += 1
        elif kind in _TYPES:
            edited["blob"][name]["type"] = _TYPES[kind]
            edited["blob"][name]["format"] = _format(kind, row.get("format", ""))
            changed += int(edited["blob"][name] != row)
    new = tomlkit.dumps(edited).encode()
    migrated = blobs.parse(config, new)
    system_new = tomllib.loads(tomlkit.dumps(proposed))
    platform.validate(config, system_new)
    after = _known(config, migrated, system_new)
    if before != after:
        raise Refusal(
            Finding("blobs.migrate", "migration changes covered ROM bytes; nothing was written", path=blobs.FILE)
        )
    writes = {blobs.FILE: new}
    if moved:
        writes[platform.FILE] = tomlkit.dumps(proposed).encode()
    writes = {p: data for p, data in writes.items() if data != raw[p]}
    reads = _Reads(config, {p: sha256(data) if data is not None else None for p, data in raw.items()})
    body = ("blobs", sha256(raw[blobs.FILE] or b""), writes, (), (), (), "Migrate blob types and platform spans")
    plan = Plan(*body, digest(body))
    with store.exclusive(config, "structure", wait=True), journal.held(config):
        if changed_paths := journal.changed(reads, paths):
            raise Refusal(Finding("journal.changed", "Migration inputs changed; rerun", missing=changed_paths))
        if writes:
            journal.apply(config, plan, reads)
        readback = _known(config, blobs.load(config), platform.load(config))
    if readback != before:
        raise Refusal(
            Finding("blobs.migrate", "readback byte coverage differs from the reviewed migration", path=blobs.FILE)
        )
    return {
        "files": sorted(writes),
        "types": changed,
        "platform": moved,
        "before": {v: blobs.length(spans) for v, spans in before.items()},
        "after": {v: blobs.length(spans) for v, spans in readback.items()},
        "readback": True,
    }
