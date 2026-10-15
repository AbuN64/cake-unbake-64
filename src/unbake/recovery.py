"""Whole-program shared-record recovery command, using existing cache and submit owners."""

from __future__ import annotations

import pickle
import re
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from unbake import compare, headers, infer, land, layout, policy, pool, recipes, report, store, view
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Refusal, Snapshot, code_identity, digest
from unbake.families.structs import (
    _derive_key,
    _is_view,
    _object_usage,
    _prefix_job,
    _prepare_job,
    _prepare_key,
    _rewrite_job,
    _scan_batch,
    _type_job,
    _type_key,
    classes_job,
    components,
    derive,
    enrich,
    graph,
    identity_table,
    object_evidence,
)
from unbake.families.structs import families as families
from unbake.families.structs import rewrite as rewrite
from unbake.families.structs import source_evidence as source_evidence


def _keyed(item: tuple[Any, ...]) -> str:
    key: str = item[-1]
    return key


def _conflict_job(item: tuple[Any, ...]) -> list[Any]:
    return headers._disagreement_rows(item[:-1])[1]


def _contexts(snapshot: Snapshot, authored: Json) -> tuple[dict[str, tuple[str, int, str]], list[Finding]]:
    paths = [*headers.sources(snapshot), *(p for p in snapshot.layout.units if p.endswith(".c"))]
    scans = headers._scanned(snapshot, paths)
    catalogue = {
        name: (path, line, text)
        for path, scan in reversed(list(zip(paths, scans, strict=True)))
        if path.endswith(".h")
        for name, line, text in reversed(scan["declared"])
    }
    # Partition by name, keeping all declarations of a name together. Parsing happens once per file.
    width = max(1, snapshot.config.host.workers * 4)
    buckets = [[{**s, "declared": [], "defined": [], "defs": [], "functions": []} for s in scans] for _ in range(width)]
    for i, scan in enumerate(scans):
        for kind in ("declared", "defined", "defs", "functions"):
            for row in scan[kind]:
                name = row if kind == "defined" else row[0]
                buckets[int(digest(name)[:8], 16) % width][i][kind].append(row)
    defined = tuple(n for n, r in authored["function"].items() if r["evidence"] == "landed")
    common = digest((paths, defined, [s["words"] for s in scans]))
    jobs = [
        (paths, rows, defined, digest((common, [{k: v for k, v in r.items() if k != "words"} for r in rows])))
        for rows in buckets
        if any(r["declared"] or r["defs"] for r in rows)
    ]
    disagreements = [
        f for rows in pool.map(snapshot.config, "recovery.conflicts", _conflict_job, jobs, _keyed) for f in rows
    ]
    return catalogue, disagreements


def _family_job(item: tuple[Any, ...]) -> dict[str, Any]:
    snapshot, job, authored, catalogue, usages, disagreements, objects, _ = item
    enrich(snapshot, job, authored, catalogue, usages, disagreements, objects)
    return derive(job)


def _family_batch(item: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [_family_job(row) for row in item[0]]


def _parts(config: Config, key: str) -> tuple[str, ...]:
    """The fragment digests of a stored prefix: neighbouring prefixes in this order share their leading fragments."""
    value = store.get(config, "recovery.prefix", key)
    row = pickle.loads(value) if value else None
    return tuple(row[0]) if isinstance(row, tuple) else ()


def _sources(snapshot: Snapshot, real_root: str, kinds: Json) -> list[Any]:
    config = snapshot.config
    jobs: list[Any] = []
    for unit in snapshot.layout.units.values():
        if not unit.path.endswith(".c") or not kinds[unit.kind]["decompiled"]:
            continue
        held = compare.holders(snapshot, unit)
        if held:
            jobs.append((snapshot, unit, held, recipes.resolve(config, unit, {}), real_root))
    prepared = pool.map(config, "recovery.views", _prepare_job, jobs, _prepare_key)
    by_version: defaultdict[str, list[Any]] = defaultdict(list)
    inputs: dict[str, Any] = {}
    prefixes: dict[str, Any] = {}
    for job, rows in zip(jobs, prepared, strict=True):
        for version, row in zip(job[2], rows, strict=True):
            by_version[version].append(row)
            inputs.setdefault(row[0], job)
            if row[1] is not None:
                prefixes.setdefault(row[1], job)
    ordered = [
        row
        for version in sorted(by_version, key=lambda v: v != config.project.names_from)
        for row in by_version[version]
    ]
    pool.map(
        config,
        "recovery.prefix",
        _prefix_job,
        [(job, key) for key, job in prefixes.items() if store.get(config, "recovery.prefix", key) is None],
    )
    owner = dict(ordered)
    by_prefix: defaultdict[str | None, list[str]] = defaultdict(list)
    for key in dict.fromkeys(key for key, _ in ordered):
        by_prefix[owner[key]].append(key)
    parts = {key: _parts(config, key) for key in prefixes}
    batches = [
        (prefix, tuple((inputs[k], k) for k in members))
        for prefix, members in sorted(by_prefix.items(), key=lambda row: (parts.get(row[0] or "", ()), row[0] or ""))
        for members in ([members] if prefix is not None else [[k] for k in members])
    ]
    rows = pool.map(
        config,
        "recovery.source",
        _scan_batch,
        [(members, digest((prefix, [k for _, k in members]))) for prefix, members in batches],
        _keyed,
    )
    found = {
        key: row
        for (_, members), batch in zip(batches, rows, strict=True)
        for (_, key), row in zip(members, batch, strict=True)
    }
    return [found[key] for key, _ in ordered]


def _analysis(snapshot: Snapshot, real_root: str, kinds: Json) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Evidence, every family and every family's derivation: cached whole under the digest of all it reads."""
    config = snapshot.config
    pins = [
        view.closure(snapshot, unit, version)
        for unit in snapshot.layout.units.values()
        if unit.path.endswith(".c") and kinds[unit.kind]["decompiled"]
        for version in sorted(compare.holders(snapshot, unit))
    ]
    key = digest((code_identity(_derive_all), snapshot.digest, pins))
    return store.cached_object(config, "recovery.analysis", key, lambda: _derive_all(snapshot, real_root, kinds))


def _derive_all(snapshot: Snapshot, real_root: str, kinds: Json) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    config = snapshot.config
    evidence = _sources(snapshot, real_root, kinds)
    authored = pool.map(config, "recovery.types", _type_job, [snapshot], _type_key)[0]
    catalogue, disagreements = _contexts(snapshot, authored)
    # Assembly evidence uses infer's existing keyed ROM scan, not a second disassembler.
    asm = infer.scan(snapshot)
    usages = {version: {row[0]: row[4] for row in rows} for version, rows in asm.items()}
    object_jobs = [
        (dict(r.symbols), usages.get(v, {}), digest((dict(r.symbols), usages.get(v, {}))))
        for v, r in snapshot.versions.items()
    ]
    objects = dict(
        zip(snapshot.versions, pool.map(config, "recovery.objects", object_evidence, object_jobs, _keyed), strict=True)
    )
    reads = {f"param:{fn}:{k}" for rows in usages.values() for fn, row in rows.items() for k, *_ in row.get("args", ())}
    proposed, spec = pool.map(config, "recovery.graph", graph, [(evidence, objects, reads)], _derive_key)[0]
    table = identity_table(spec, pool.map(config, "recovery.classes", classes_job, components(spec)), objects)
    context = digest(
        (
            snapshot.layout.digest,
            authored,
            catalogue,
            usages,
            disagreements,
            [(v, r.rom_sha256, dict(r.symbols)) for v, r in snapshot.versions.items()],
        )
    )
    conflicts_by_name = defaultdict(list)
    for finding in disagreements:
        conflicts_by_name[finding.unit].append(finding)
    globals_ = {
        d["name"] for job in proposed for d in job["declarations"] if d["kind"] == "global" and d.get("external")
    }
    object_uses = {name: _object_usage(name, objects) for name in globals_}
    jobs = []
    for job in proposed:
        functions = {u["function"] for u in job["casts"]}
        names = {d["name"] for d in job["declarations"] if d["kind"] == "global" and d.get("external")}
        tag = "struct Recovered_" + job["id"]
        # Only a family's own evidence travels to its worker; the whole ROM graph is never pickled per family.
        wanted = {"function": functions, "struct": job["records"]}
        jobs.append(
            (
                snapshot,
                job,
                {k: {n: authored[k][n] for n in chosen if n in authored[k]} for k, chosen in wanted.items()},
                {tag: catalogue[tag]} if tag in catalogue else {},
                {v: {n: rows[n] for n in sorted(functions) if n in rows} for v, rows in usages.items()},
                [f for n in sorted(names) for f in conflicts_by_name[n]],
                {n: object_uses[n] for n in names},
                digest((context, job)),
            )
        )
    # Families are small: one pooled item carries a slice of them, so dispatch never idles the workers.
    width = max(1, len(jobs) // (config.host.workers * 4))
    slices = [(jobs[i : i + width], digest([j[-1] for j in jobs[i : i + width]])) for i in range(0, len(jobs), width)]
    derived = [r for rows in pool.map(config, "recovery.family", _family_batch, slices, _keyed) for r in rows]
    return evidence, proposed, derived, table


def clones(snapshot: Snapshot) -> dict[str, str]:
    """Every record the last recovery run found to duplicate another, with its canonical and that record's header."""
    value = store.get(snapshot.config, "recovery.clones", "table")
    return pickle.loads(value) if value else {}


def run(config: Config, params: Json) -> Json:
    start = time.monotonic()
    out = Path(params.get("out") or Path("/tmp") / ("unbake-recovery-" + digest(str(config.project.root))[:16]))
    out = out.resolve()
    if out.is_relative_to(config.project.root.resolve()):
        raise Refusal(Finding("land.request", "Recovery output must be outside the live project.", path=str(out)))
    real_root = str(config.project.root.resolve())
    snapshot = layout.capture(config)
    before = report.progress(snapshot)
    kinds = configuration.kinds()
    exclude = set(params.get("exclude") or ())
    evidence, proposed, derived, identities = _analysis(snapshot, real_root, kinds)
    wanted = set(params.get("family") or ())
    functions = set(params.get("function") or ())  # families whose casts sit in these functions
    chosen = [
        i
        for i, job in enumerate(proposed)
        if (not wanted or job["id"] in wanted or job["identity"] in wanted)
        and (not functions or functions & {u["function"] for u in job["casts"]})
    ]
    selected = [proposed[i] for i in chosen]
    results = [deepcopy(derived[i]) for i in chosen]
    emitted, files = [], []
    for result in results:
        if result["conflicts"]:
            continue
        needed = (
            {u["path"] for u in result["mapping"]}
            | {d["path"] for d in result["declarations"]}
            | {c["owner"] for c in result["record_changes"]}
        )
        if needed & exclude:
            result["conflicts"].append({"reason": "family reaches excluded files", "paths": sorted(needed & exclude)})
            continue
        pinned = needed | result["sources"].keys()
        pinned.update(
            o["path"]
            for owners in result["record_owners"].values()
            for o in owners
            if o["path"].startswith(("include/", "src/"))
        )
        contents = {p: snapshot.read(p) for p in pinned}
        stale = [p for p, pin in result["sources"].items() if digest(contents[p]) != pin]
        if stale:
            result["conflicts"].append({"reason": "source evidence changed during analysis", "paths": stale})
            continue
        rewritten = pool.map(config, "recovery.rewrite", _rewrite_job, [(result, contents)], _derive_key)[0]
        if rewritten["conflicts"]:
            result["conflicts"].extend(rewritten["conflicts"])
            continue
        writes = rewritten["writes"]
        # Only complete source retirements are submitted; a source can belong to several unresolved families.
        leftovers = [
            p
            for p, raw in writes.items()
            if p.endswith(".c")
            and any(
                _is_view(n)
                for n in re.findall(r"\(\s*(?:struct\s+)?(\w+)\s*\*\s*\)", policy.scopes(raw.decode())["code"])
            )
        ]
        if leftovers:
            result["conflicts"].append({"reason": "source also reaches unresolved families", "paths": leftovers})
            continue
        home = out / result["id"]
        for path, raw in writes.items():
            store.write(home / path, raw)
        sources = sorted(p for p in writes if p.endswith(".c"))
        if not sources:
            continue
        primary = sources[0]
        group = [home / primary, *(Path(p + "=" + str(home / p)) for p in sorted(writes) if p != primary)]
        files.extend(group)
        emitted.append(
            {
                "id": result["id"],
                "files": sorted(writes),
                "primary": str(home / primary),
                "inputs": {p: digest(raw) for p, raw in contents.items()},
            }
        )
        result["affected_files"] = sorted(writes)
    submissions, drain = None, None
    if params.get("submit") and files:
        # The live tree can change while analysis runs. Stale inputs are never admitted.
        for family in emitted:
            for path, pin in family["inputs"].items():
                if digest((config.project.root / path).read_bytes()) != pin:
                    raise Refusal(Finding("journal.changed", "Recovery input changed during analysis.", path=path))
        submissions = compare.submit_command(
            config,
            {
                "files": files,
                "bulk": True,
                "function": None,
                "toolchain": None,
                "flag": (),
                "omit_flag": (),
                "withhold": False,
                "note": "whole-program shared struct recovery",
                "batch_data": None,
                "prove_extras": True,
            },
        )
        admitted = [row["id"] for row in submissions["submissions"] if row["id"] and row["exact"]]
        drain = [land.drain(config, identity) for identity in admitted]
    census = {
        "families": len(proposed),
        "selected": len(selected),
        "casts": sum(len(job["casts"]) for job in proposed),
        "files": len({u["path"] for job in proposed for u in job["casts"]}),
        "ready": len(emitted),
        "source_parse_conflicts": [c for row in evidence for c in row["conflicts"]],
    }
    store.put(
        config,
        "recovery.clones",
        "table",
        pickle.dumps(
            {
                m["name"].removeprefix("struct "): f"{row['canonical'].removeprefix('struct ')} ({row['owner_header']})"
                for row in identities
                for m in row["members"]
                if m["name"] != row["canonical"]
            }
        ),
    )
    return {
        "census": census,
        "families": results,
        "candidates": emitted,
        "identities": identities,
        "submissions": submissions,
        "drain": {
            "landed": [r for d in drain or () for r in d.get("landed", ())],
            "refused": [r for d in drain or () for r in d.get("refused", ())],
            "running": any(d.get("running") for d in drain or ()),
        },
        "before": before,
        "after": report.progress(layout.capture(config)) if drain else None,
        "wall_seconds": time.monotonic() - start,
    }
