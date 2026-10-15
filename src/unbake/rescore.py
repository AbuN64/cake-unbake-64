"""Fuzzy scores kept current: a write, made once per drain, never by a reader."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256

from unbake import compare, delta, effort, layout, pool, versions
from unbake.contracts import Json, Proof, Refusal, Snapshot, code_identity, digest


def _key_job(item: tuple[Snapshot, str, Json, str]) -> tuple[str, dict[str, str]] | None:
    snapshot, member, row, _ = item
    try:
        unit, view = compare.bind(snapshot, snapshot.config.project.root / row["path"], member)
    except (Refusal, FileNotFoundError):  # a candidate that no longer binds keeps its row as it is
        return None
    return member, compare.fuzzy_keys(snapshot.config, [(view, unit)])[0]


def _row_key(item: tuple[Snapshot, str, Json, str]) -> str:
    """The cache key of one row's keys: a module function, so the workers can compute it too. It names what the row's
    score reads (its text, the headers it reaches, its member's rows and the versions), never the whole tree, so a
    landing elsewhere re-keys no row."""
    snapshot, member, row, reach = item
    entry = snapshot.layout.members.get(member)
    facts = [(v, versions.facts_digest(r)) for v, r in sorted(snapshot.versions.items())]
    text = snapshot.peek(row["path"])
    return digest((code_identity(_key_job), snapshot.config.digest, member, entry, reach, facts, text))


_INCLUDE = re.compile(rb'^\s*#\s*include\s*"([^"]+)"', re.M)


def _reached(snapshot: Snapshot, path: str) -> dict[str, bytes | None]:
    """Every project header a file includes, transitively, by its #include lines (None: named but missing)."""
    seen: dict[str, bytes | None] = {}
    todo = [path]
    while todo:
        for name in _INCLUDE.findall(snapshot.peek(todo.pop()) or b""):
            header = f"include/{name.decode()}"
            if header not in seen:
                seen[header] = snapshot.peek(header)
                todo += [header] * (seen[header] is not None)
    return seen


def reach(snapshot: Snapshot, path: str) -> str:
    """The digest of every project header a file includes, transitively, by its #include lines."""
    return digest(sorted(_reached(snapshot, path).items()))


def _item_key(item: tuple[Snapshot, str, Json]) -> str:
    snapshot, member, row = item
    return _row_key((snapshot, member, row, reach(snapshot, row["path"])))


def _keys_job(item: tuple[Snapshot, str, Json]) -> tuple[tuple[str, dict[str, str]] | None, dict[str, str]]:
    """A row's keys and the pins of what its row key reads: its text and every header it reaches, found or not."""
    snapshot, member, row = item
    files = [row["path"], *_reached(snapshot, row["path"])]
    return _key_job((snapshot, member, row, "")), {f: _pin(snapshot, f) for f in files}


def _pin(snapshot: Snapshot, path: str) -> str:
    """A file's pin as the row key reads it: through the snapshot ("missing" when it has none)."""
    content = snapshot.peek(path)
    return "missing" if content is None else sha256(content).hexdigest()


def _measure_job(item: tuple[Snapshot, str, Json]) -> tuple[Proof, ...]:
    snapshot, member, row = item
    unit, view = compare.bind(snapshot, snapshot.config.project.root / row["path"], member)
    return compare.measure_bound(snapshot.config, [(view, unit)], {})[0]


def rows(snapshot: Snapshot) -> dict[str, Json]:
    """The fuzzy rows of the snapshot with moved rows scored again; only a row whose per-version key moved is measured.
    The keys are computed on all cores and cached per row, so an unchanged drain dispatches nothing."""
    current = snapshot.layout.fuzzy
    facts = [(v, versions.facts_digest(r)) for v, r in sorted(snapshot.versions.items())]
    # only a row whose text, reached headers or member row moved is keyed again: a tool install or a landing elsewhere
    # re-derives no row (a row's record is a fact of its inputs, not of the tool code)
    found = delta.map(
        snapshot.config,
        "rescore.keys",
        _keys_job,
        [(snapshot, m, r) for m, r in sorted(current.items())],
        key=_item_key,
        ident=lambda item: item[1],
        stamp=lambda item: delta.fingerprint((item[1], snapshot.layout.members.get(item[1]), item[2]["path"])),
        pin=lambda path: _pin(snapshot, path),
        context=(snapshot.config.digest, facts),
        code=False,
    )
    keys = {k[0]: k[1] for k in found if k}
    moved = [m for m, key in keys.items() if current[m].get("key") != key]
    if not moved:
        return dict(current)
    # each moved row binds and measures on a worker (the measure inside is keyed and runs inline there), so the binds
    # are not a serial loop of the parent
    scored = pool.map(snapshot.config, "rescore.measure", _measure_job, [(snapshot, m, current[m]) for m in moved])
    proofs = dict(zip(moved, scored, strict=True))
    fuzzy = dict(current)
    for member in moved:
        scores = {p.version: p.score for p in proofs[member] if p.member == member}
        fuzzy[member] = {**current[member], "scores": scores, "key": keys[member]}
    return fuzzy


def writes(snapshot: Snapshot) -> Mapping[str, bytes]:
    """layout.toml with the rescored rows, or nothing when no row moved."""
    with effort.stage("rescore.rows"):
        fuzzy = rows(snapshot) if snapshot.layout.fuzzy else snapshot.layout.fuzzy
    return (
        {}
        if fuzzy == snapshot.layout.fuzzy
        else {"layout.toml": layout.dump_map(replace(snapshot.layout, fuzzy=fuzzy))}
    )
