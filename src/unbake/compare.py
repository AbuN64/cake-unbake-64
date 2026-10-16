"""Measure one unit in every holding version and report the gaps against the ROMs."""

from __future__ import annotations

import difflib
import json
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any

from unbake import (
    alloc,
    effort,
    headers,
    land,
    layout,
    moved,
    native,
    ownership,
    policy,
    pool,
    process,
    recipes,
    sched,
    store,
    symbols,
    symptoms,
    versions,
    view,
)
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Proof, Recipe, Refusal, Snapshot, UnitSpec, digest


def holders(snapshot: Snapshot, unit: UnitSpec) -> tuple[str, ...]:
    return tuple(sorted(snapshot.layout.held(unit.members)))


def _export(item: tuple[Any, ...]) -> None:
    snapshot, unit, version, recipe, work, proofs, target = item
    target.mkdir(parents=True)
    stem = Path(unit.path).stem
    for name in (f"{stem}.i", f"{stem}.s", f"{stem}.o", "linked.elf", "text.bin", "rodata.bin", "data.bin", "bss.size"):
        if (work / name).exists():
            shutil.copy2(work / name, target / name)
    for extra in sorted(list(work.glob("*.greg")) + list(work.glob("*.lreg"))):
        shutil.copy2(extra, target / extra.name)
    held = sorted(
        (
            p
            for n in unit.members
            for p in snapshot.layout.members[n].placements
            if p.version == version and p.section == ".text"
        ),
        key=lambda p: p.rom_start,
    )
    if held:
        first, last = held[0], held[-1]
        original = versions.rom_bytes(snapshot.versions[version], first.rom_start, last.rom_end)
        (target / "original.bin").write_bytes(original)
        objdump = process.tool(snapshot.config, "mips_objdump")
        for source, name in (("text.bin", "linked.dis"), ("original.bin", "original.dis")):
            if (target / source).exists():
                process.run(
                    "compare.disassemble",
                    [
                        str(objdump),
                        "-D",
                        "-b",
                        "binary",
                        "-m",
                        "mips",
                        "-EB",
                        f"--adjust-vma=0x{first.vram:X}",
                        str(target / source),
                    ],
                    target,
                    stdout_path=target / name,
                    tmp=process.scratch(snapshot.config.project.root),
                )
        built, wanted = target / "linked.dis", target / "original.dis"
        if built.exists() and wanted.exists() and built.read_bytes() != wanted.read_bytes():  # not exact: the diff
            lines = difflib.unified_diff(
                wanted.read_text(errors="replace").splitlines(),
                built.read_text(errors="replace").splitlines(),
                "original.dis",
                "linked.dis",
                lineterm="",
                n=3,
            )
            (target / "diff.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for place in sorted(
        (
            p
            for n in unit.members
            for p in snapshot.layout.members[n].placements
            if p.version == version and p.section in (".rodata", ".data")
        ),
        key=lambda p: p.rom_start,
    ):
        name = f"original{place.section}.bin"  # the target bytes next to the section the build made
        with (target / name).open("ab") as stream:
            stream.write(versions.rom_bytes(snapshot.versions[version], place.rom_start, place.rom_end))
    document = {"recipe": asdict(recipe), "proofs": [asdict(p) for p in proofs if p.version == version]}
    (target / "result.json").write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def _key(item: tuple[Any, ...]) -> str:
    """What a measurement reads, named without building it: a warm one never travels to a worker."""
    snapshot, unit, version, recipe, *_ = item
    return native.stamp(snapshot, unit, version, recipe)


def _known_key(stamps: dict[Path, str], item: tuple[Any, ...]) -> str:
    return stamps[item[4]]


def measure(snapshot: Snapshot, unit: UnitSpec, overrides: Json, out: Path | None) -> tuple[Proof, ...]:
    return measure_many(snapshot, unit, [overrides], out)[0]


def _derived(
    bound: Sequence[tuple[Snapshot, UnitSpec]], trial: Sequence[Recipe | None]
) -> list[tuple[Snapshot, UnitSpec]]:
    """ownership.derive of every pair: every tree state and unit is one derive_pairs, so N units are one dispatch."""
    return ownership.derive_pairs(bound, recipe=list(trial))


def measure_bound(
    config: Config, bound: Sequence[tuple[Snapshot, UnitSpec]], overrides: Json
) -> list[tuple[Proof, ...]]:
    """One proof tuple per (snapshot, unit) pair, every pair and version in one keyed native dispatch."""
    jobs: list[Any] = []
    spans: list[int] = []
    recipes_ = [recipes.resolve(config, unit, overrides) for _, unit in bound]
    with store.work(config) as work:
        for n, ((_, unit), recipe, (derived_snapshot, derived)) in enumerate(
            zip(bound, recipes_, _derived(bound, recipes_), strict=True)
        ):
            held = holders(derived_snapshot, derived)
            spans.append(len(held))
            jobs.extend(
                (derived_snapshot, derived, v, recipe, work / f"{n}-{digest(unit.path)[:16]}" / v, False) for v in held
            )
        # Different candidate trees can read identical proof inputs. Concurrent cache misses would build
        # every copy, whereas sequential groups reused the first result. Name them once, dispatch each
        # distinct proof once, then restore every caller's pair/version order (including refusals).
        stamps = pool.map(config, "compare.stamps", _key, jobs) if jobs else []
        unique: dict[str, tuple[Any, ...]] = {}
        for stamp, job in zip(stamps, jobs, strict=True):
            unique.setdefault(stamp, job)
        known = {job[4]: stamp for stamp, job in unique.items()}
        measured = (
            pool.map(config, "compare.measure", native.prove_args, list(unique.values()), partial(_known_key, known))
            if unique
            else []
        )
        by_stamp = dict(zip(unique, measured, strict=True))
        results = iter(by_stamp[stamp] for stamp in stamps)
        return [tuple(p for _ in range(count) for p in next(results)) for count in spans]


def measure_units(snapshot: Snapshot, units: Sequence[UnitSpec], overrides: Json) -> tuple[Proof, ...]:
    """Prove a coordinated change set in one existing keyed native dispatch."""
    groups = measure_bound(snapshot.config, [(snapshot, u) for u in units], overrides)
    return tuple(p for group in groups for p in group)


def fuzzy_keys(config: Config, bound: Sequence[tuple[Snapshot, UnitSpec]]) -> list[dict[str, str]]:
    """Per pair and held version: what the score read, the pinned proof stamp plus the candidate text, never built."""
    keys = []
    recipes_ = [recipes.resolve(config, unit, {}) for _, unit in bound]
    for (snapshot, unit), recipe, (derived_snapshot, derived) in zip(
        bound, recipes_, _derived(bound, recipes_), strict=True
    ):
        text = snapshot.read(unit.path)
        keys.append(
            {
                v: digest((native.stamp(derived_snapshot, derived, v, recipe, pinned=True), text))
                for v in holders(derived_snapshot, derived)
            }
        )
    return keys


def measure_many(
    snapshot: Snapshot, unit: UnitSpec, variants: Sequence[Json], out: Path | None = None
) -> list[tuple[Proof, ...]]:
    """One proof tuple per override variant, all variants and versions in one pool map; `out` exports variant 0."""
    with effort.stage("compare.measure"):
        config = snapshot.config
        if out is not None and out.exists():
            raise Refusal(
                Finding(
                    "compare.out", reason=f"{out} already exists", unit=unit.path, action="pass a new --out directory"
                )
            )
        recipes_ = [recipes.resolve(config, unit, o) for o in variants]
        # the data its source emits is part of what must match, placed by the first variant's recipe
        snapshot, unit = ownership.derive(snapshot, unit, recipe=recipes_[0])
        held = holders(snapshot, unit)
        with store.work(config) as work:
            jobs = [
                (snapshot, unit, v, r, work / (v if i == 0 else f"{i}-{v}"), i == 0 and out is not None)
                for i, r in enumerate(recipes_)
                for v in held
            ]  # an export links, cached or not
            results = pool.map(config, "compare.measure", native.prove_args, jobs, None if out is not None else _key)
            proofs = [
                tuple(p for batch in results[i * len(held) : (i + 1) * len(held)] for p in batch)
                for i in range(len(variants))
            ]
            if out is not None:
                staging = out.parent / (".tmp-" + out.name)
                if staging.exists():
                    shutil.rmtree(staging)
                pool.map(
                    config,
                    "compare.export",
                    _export,
                    [(snapshot, unit, v, recipes_[0], work / v, proofs[0], staging / v) for v in held],
                )
                os.replace(staging, out)
        return proofs


def gaps(snapshot: Snapshot, unit: UnitSpec, proofs: Sequence[Proof]) -> tuple[Finding, ...]:
    found = {(p.member, p.version): p for p in proofs}
    findings = []
    for name in unit.members:
        bad: list[str] = []
        badproofs: list[Proof] = []
        lost: list[str] = []
        for v in snapshot.layout.members[name].holders():
            proof = found.get((name, v))
            if proof is None:
                bad.append(v)
                lost.append(f"version {v}: no measurement")
            elif not proof.exact:
                bad.append(v)
                badproofs.append(proof)
                lost.extend(proof.missing)
        if lost:
            facts = symptoms.merge([p.symptoms for p in badproofs] + [symptoms.from_missing(lost)])
            findings.append(
                Finding(
                    "land.not_exact",
                    reason=f"{name} does not match the ROM in every holding version",
                    unit=name,
                    versions=tuple(bad),
                    missing=tuple(lost),
                    symptoms=facts,
                )
            )
    return tuple(findings)


def _gaps_of(snapshot: Snapshot, units: Sequence[UnitSpec], proofs: Sequence[Proof]) -> tuple[Finding, ...]:
    return tuple(f for unit in units for f in gaps(snapshot, unit, proofs))


def _standing(
    snapshot: Snapshot,
    primary: UnitSpec,
    found: Sequence[Finding],
    proofs: Sequence[Proof],
    overrides: Json,
    everyone: bool = False,
) -> tuple[tuple[Finding, ...], set[tuple[str, str]]]:
    """A change set gates its other units on no regression: a member HEAD already measures inexact in a version, which
    the change leaves no worse there, is a standing gap of HEAD and not this change's (it lands as it stands); the
    primary must be exact, unless `everyone` (a change that alters no code intent, such as a retype) gates it the same
    way. Returns the gaps that block and the (member, version) pairs that stand."""
    heads = {f.unit: layout.unit_of(snapshot, f.unit) for f in found if everyone or f.unit not in primary.members}
    paths = sorted({u.path for u in heads.values() if u is not None})
    if not paths:
        return tuple(found), set()
    before = {(p.member, p.version): p for p in _baseline(snapshot, paths, overrides)}
    after = {(p.member, p.version): p for p in proofs}
    blocking, standing = [], set()
    for gap in found:
        held = [
            (gap.unit, v)
            for v in gap.versions
            if heads.get(gap.unit) is not None
            and (old := before.get((gap.unit, v))) is not None
            and not old.exact
            and (new := after.get((gap.unit, v))) is not None
            and new.score >= old.score
        ]
        if gap.unit in heads and held and len(held) == len(gap.versions):
            standing.update(held)
        else:
            blocking.append(gap)
    return tuple(blocking), standing


def bind(
    snapshot: Snapshot, file: Path, function: str | None, notes: list[str] | None = None
) -> tuple[UnitSpec, Snapshot]:
    """The unit a candidate file lands as and the snapshot holding it. The candidate is read through the table of moved
    paths and symbols first; each rewrite made is appended to `notes` as `old->new`."""
    root = snapshot.config.project.root.resolve()
    path = file.resolve()
    rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else None
    if rel is not None and rel in snapshot.layout.units:
        return snapshot.layout.units[rel], snapshot
    if not file.is_file():
        raise Refusal(
            Finding(
                "land.request",
                reason=f"the file {file} does not exist",
                path=str(file),
                action="pass an existing source file",
            )
        )
    content, rewritten = moved.rewrite(snapshot, file.read_bytes(), rel or "")
    content, renamed = _member_name(snapshot, file, content)
    if notes is not None:
        notes.extend([*rewritten, *renamed])
    given = [*headers.defined(content), *headers.variables(content)]
    named = [n for n in given if n in snapshot.layout.members]
    if not named and function is None:
        raise Refusal(
            Finding(
                "land.request",
                reason="the source defines no member of the layout: it defines "
                + (", ".join(given) or "no function or storage"),
                action="define a member of the layout or pass --function NAME",
            )
        )
    member = function or named[0]
    if member not in snapshot.layout.members:
        raise Refusal(
            Finding("land.request", reason=f"{member} is not a member of the layout", action="pass --function NAME")
        )
    shared = layout.source_unit(snapshot, member, content)
    landed = next(
        (u for u in sorted(snapshot.layout.units.values(), key=lambda u: u.path) if member in u.members), None
    )
    if shared is not None:
        unit, writes = shared
    elif landed and Path(landed.path).suffix == file.suffix:
        unit, writes = landed, {landed.path: content}
    else:  # a source of another language than the landed unit's (C for library assembly) takes a unit of its own
        options = layout.unit_options(snapshot, member, content)
        if not options:
            held = f": {landed.path} ({landed.kind}) holds it and takes no {file.suffix} source" if landed else ""
            raise Refusal(
                Finding(
                    "land.request",
                    reason=f"{member} has no standalone unit option{held}",
                    action="pass --function NAME" if not landed else f"score a {Path(landed.path).suffix} source",
                )
            )
        unit, writes = options[-1]
    if rel and any(Path(rel).is_relative_to(folder) for folder in getattr(snapshot.config.project, "library_dirs", ())):
        writes = {p: b for p, b in writes.items() if p not in (unit.path, "layout.toml")}
        unit = replace(unit, path=rel)
        units = {p: u for p, u in snapshot.layout.units.items() if not set(u.members) & set(unit.members)}
        units[rel] = unit
        writes.update({rel: content, "layout.toml": layout.dump_map(replace(snapshot.layout, units=units))})
    return unit, layout.overlay(snapshot, writes)


def _member_name(snapshot: Snapshot, file: Path, content: bytes) -> tuple[bytes, list[str]]:
    """A candidate copied from a fuzzy draft defines the draft's name, which can be an older boundary's address name
    (the draft src/fuzzy/func_X.c of member func_Y defines func_X): with no member defined, the function named like a
    draft's file is spelled as that draft's member (`old->new`)."""
    defined = headers.defined(content)
    if any(n in snapshot.layout.members for n in defined):
        return content, []
    drafts = {Path(str(r.get("path", ""))).stem: m for m, r in snapshot.layout.fuzzy.items()}
    pairs = [(n, drafts[n]) for n in defined if drafts.get(n, n) != n and drafts[n] in snapshot.layout.members]
    if not pairs:
        return content, []
    old, new = pairs[0]
    return symbols.rewrite(content.decode(errors="replace"), old, new).encode(), [f"{old}->{new}"]


def _extra(snapshot: Snapshot, spec: Path) -> tuple[str, bytes]:
    """The project path a change-set file replaces and its new text (`PATH=FILE`, a project file, a landed unit)."""
    root = snapshot.config.project.root.resolve()
    target, equals, source = str(spec).partition("=")
    if equals and not spec.exists():
        path = (root / target).resolve()
        fresh = path.is_relative_to(root / "include") and target.endswith(".h")
        if target.endswith(".c") and not path.is_file():
            raise Refusal(
                Finding(
                    "land.request",
                    f"{target} would be a new source; a change-set entry creates headers only.",
                    path=target,
                    action="land the source as a candidate",
                )
            )
        if not path.is_relative_to(root) or not (path.is_file() or fresh):
            raise Refusal(
                Finding(
                    "land.request",
                    f"{target} is no file of the project.",
                    path=target,
                    action="name an existing project file left of =",
                )
            )
        if not Path(source).is_file():
            raise Refusal(
                Finding(
                    "land.request",
                    f"the file {source} does not exist.",
                    path=source,
                    action="pass the new text of the project file right of =",
                )
            )
        return path.relative_to(root).as_posix(), Path(source).read_bytes()
    if spec.resolve().is_relative_to(root):
        return spec.resolve().relative_to(root).as_posix(), spec.read_bytes()
    unit = next((u for u in snapshot.layout.units.values() if spec.stem in u.members), None)
    if unit is None:
        raise Refusal(
            Finding(
                "land.request",
                f"{spec} is outside the project and is no landed unit's source.",
                path=str(spec),
                action="pass PATH=FILE, a file inside the project or a landed unit's new text",
            )
        )
    return unit.path, spec.read_bytes()


def _proof_set(
    snapshot: Snapshot, bound: Snapshot, unit: UnitSpec, extras: Json, strict: bool, every: bool = False
) -> dict[str, Any]:
    """The units a change set proves: `changed` (the primary and the extra sources, which must match exactly) and
    `consumers` (every other unit including a changed header and using a declaration it alters, which must not get
    worse). A change set with no header extra has no consumers. `strict` (recovery)
    proves the primary exactly even alone."""
    changed_headers = [p for p in extras if p.endswith(".h")]
    includers = {
        p
        for p in headers.consumers(bound, changed_headers)
        if p != unit.path and p not in extras and p in snapshot.layout.units
    }
    if every:  # every includer, as the landing proves them: a header edit can break a unit that uses none of it
        consumers = sorted(includers)
    else:
        consumers = sorted(land.declaration_users(snapshot, bound, includers, changed_headers) if includers else ())
    changed = sorted({unit.path, *(p for p in extras if p in bound.layout.units)})
    proving = strict or len(changed) > 1 or bool(consumers)
    return {"changed": changed, "consumers": consumers, "proving": proving, "wall": 0.0}


def _baseline(snapshot: Snapshot, consumers: Sequence[str], overrides: Json) -> tuple[Proof, ...]:
    """The consumers' proofs in the tree as it is. Every group of a bulk chunk shares that tree, so a consumer is
    derived, keyed and measured once per command, not once per group naming it."""
    kept: dict[str, tuple[Proof, ...]] = effort.memo(("compare.baseline", snapshot.digest, digest(overrides)), dict)
    todo = [c for c in consumers if c not in kept]
    if todo:
        units = [snapshot.layout.units[c] for c in todo]
        proofs = measure_units(snapshot, units, overrides)
        for c in todo:
            kept[c] = tuple(p for p in proofs if p.unit == c)
    return tuple(p for c in consumers for p in kept[c])


def _worse(before: Sequence[Proof], after: Sequence[Proof]) -> dict[tuple[str, str], list[str]]:
    """The members per (unit, version) measured worse after than before; a member with no after proof is worse."""
    later = {(p.unit, p.member, p.version): p for p in after}
    found: dict[tuple[str, str], list[str]] = {}
    for old in before:
        new = later.get((old.unit, old.member, old.version))
        if new is None or new.score < old.score or (old.exact and not new.exact):
            found.setdefault((old.unit, old.version), []).append(old.member)
    return found


def _regressions(
    snapshot: Snapshot, consumers: Sequence[str], proofs: Sequence[Proof], overrides: Json, sibling: str = ""
) -> tuple[Finding, ...]:
    """A finding for each consumer and version measured worse against the overlays than without them. The units
    whose preprocessed text is unchanged hit the measurement cache, so only real consumers compile twice. `sibling`
    names the bulk member whose refusal leaves the consumer's old source in place."""
    if not consumers:
        return ()
    found = _worse(_baseline(snapshot, consumers, overrides), proofs)
    kept = f" (its bulk draft {sibling} was refused, so its old source stays)" if sibling else ""
    return tuple(
        Finding(
            "land.consumer_regressed",
            reason=f"{path} no longer builds as before in version {version} with the changed headers{kept}",
            unit=path,
            versions=(version,),
            missing=tuple(sorted(members)),
            action="fix the header change or the consumer so it builds as it did",
        )
        for (path, version), members in sorted(found.items())
    )


def _kin_regressions(candidate: str, own: Sequence[Proof], after: Sequence[Proof]) -> tuple[Finding, ...]:
    """A bulk member's draft measured worse in the candidate's tree (with the draft applied) than in its own tree:
    the candidate really regresses that sibling, and the finding names the pair."""
    return tuple(
        Finding(
            "land.consumer_regressed",
            reason=f"{candidate} makes bulk member {path} build worse in version {version} than its own change does",
            path=candidate,
            unit=path,
            versions=(version,),
            missing=tuple(sorted(members)),
            action=f"make the header changes of {candidate} and {path} agree",
        )
        for (path, version), members in sorted(_worse(own, after).items())
    )


def _attempt(
    config: Config, files: Sequence[Path], params: Json, overrides: Json, out: Path | None, origin: str
) -> Json:
    """Measure the first file (the others are a change set landed with it), submit it and drain: the one path."""
    done = _admit(config, layout.capture(config), files, params, overrides, out, origin)
    return {**done, "drain": land.drain(config, done["submission"].id) if done["submission"] else None}


def _prepare(
    snapshot: Snapshot, files: Sequence[Path], params: Json
) -> tuple[dict[str, bytes], UnitSpec, Snapshot, list[str]]:
    """The candidate bound over its change set: its extras, unit, bound tree state and respelled names."""
    extras = dict(_extra(snapshot, f) for f in files[1:])
    rewritten: list[str] = []
    unit, bound = bind(
        layout.overlay(snapshot, extras) if extras else snapshot, files[0], params["function"], rewritten
    )
    for path in (p for p in extras if p.endswith(".h") and snapshot.peek(p) is None):
        if not headers.consumers(bound, [path]):
            raise Refusal(
                Finding("land.request", "the new header is included by no source", path=path, action="include it")
            )
    return extras, unit, bound, rewritten


def _admit(
    config: Config,
    snapshot: Snapshot,
    files: Sequence[Path],
    params: Json,
    overrides: Json,
    out: Path | None,
    origin: str,
    prepared: tuple[dict[str, bytes], UnitSpec, Snapshot, list[str]] | None = None,
) -> Json:
    """Prove and submit one candidate (a bulk proves every group first, then settles and enters them)."""
    done = _prove(config, snapshot, files, params, overrides, out, prepared, {})
    return _enter(config, snapshot, files, params, overrides, origin, done)


def _prove(
    config: Config,
    snapshot: Snapshot,
    files: Sequence[Path],
    params: Json,
    overrides: Json,
    out: Path | None,
    prepared: tuple[dict[str, bytes], UnitSpec, Snapshot, list[str]] | None,
    kin: dict[str, tuple[Snapshot, UnitSpec]],
) -> Json:
    """Everything admission decides before submitting. `kin` (bulk) maps every other member's unit path to its tree
    and unit: a consumer that is a member is proved with its draft applied, never as its old source. `pending` maps a
    sibling's unit path to the regressions of its old source, which refuse this candidate only if the sibling is
    refused."""
    staged = _stage(config, snapshot, files, params, overrides, prepared, kin)
    started = time.monotonic()
    batch = staged["batch"] if staged["proved"]["proving"] else None  # alone it measures (and exports) itself
    measured = None
    if batch is not None:
        measured = (
            measure_bound(config, batch, overrides)
            if staged["siblings"]
            else [measure_units(staged["bound"], [u for _, u in batch], overrides)]
        )
    return _finish(config, snapshot, params, overrides, out, staged, measured, time.monotonic() - started)


def _stage(
    config: Config,
    snapshot: Snapshot,
    files: Sequence[Path],
    params: Json,
    overrides: Json,
    prepared: tuple[dict[str, bytes], UnitSpec, Snapshot, list[str]] | None,
    kin: dict[str, tuple[Snapshot, UnitSpec]],
) -> Json:
    """A candidate's tree, unit and proof set, and `batch`: snapshot/unit pairs measured with every other candidate.
    Sibling drafts retain their own tree and the candidate tree with their writes laid underneath its own."""
    extras, unit, bound, rewritten = prepared or _prepare(snapshot, files, params)
    bound, unit = ownership.derive(bound, unit, recipe=recipes.resolve(config, unit, overrides))
    proved = _proof_set(snapshot, bound, unit, extras, bool(params.get("prove_extras")))
    units = [unit if p == unit.path else bound.layout.units[p] for p in proved["changed"]] if proved["proving"] else []
    siblings = {c: kin[c] for c in proved["consumers"] if c in kin} if proved["proving"] else {}
    batch = (
        _kin_pairs(bound, units, proved["consumers"], siblings)
        if siblings
        else [(bound, u) for u in ([*units, *(bound.layout.units[p] for p in proved["consumers"])] or [unit])]
    )
    # Recipe errors belong to this candidate, before any shared measurement dispatch can raise them for the chunk.
    for candidate in [*(u for _, u in batch), *(snapshot.layout.units[c] for c in proved["consumers"])]:
        recipes.resolve(config, candidate, overrides)
    return {
        "extras": extras,
        "unit": unit,
        "bound": bound,
        "rewritten": rewritten,
        "proved": proved,
        "units": units,
        "siblings": siblings,
        "batch": batch,
    }


def _finish(
    config: Config,
    snapshot: Snapshot,
    params: Json,
    overrides: Json,
    out: Path | None,
    staged: Json,
    measured: Sequence[tuple[Proof, ...]] | None,
    wall: float,
) -> Json:
    """Everything admission decides from a staged candidate and, when it was batched, its measured proofs."""
    extras, unit, bound, rewritten = staged["extras"], staged["unit"], staged["bound"], staged["rewritten"]
    proved, units, siblings = staged["proved"], staged["units"], staged["siblings"]
    pending: dict[str, tuple[Finding, ...]] = {}
    started = time.monotonic()
    if proved["proving"]:
        assert measured is not None  # every proving candidate was measured before its verdict is decided
        if siblings:
            everyone, split = _kin_findings(snapshot, staged, measured, overrides)
            pending = {c: f for c, f in split.items() if c in siblings and f}
            found_here = split.get("", ())
        else:
            everyone = tuple(p for group in measured for p in group)
            found_here = _regressions(snapshot, proved["consumers"], everyone, overrides)
        proofs = tuple(p for p in everyone if p.unit in proved["changed"])
        alike = bool(params.get("no_regression"))
        found, standing = _standing(snapshot, unit, _gaps_of(bound, units, proofs), proofs, overrides, alike)
        proofs = tuple(p for p in proofs if (p.member, p.version) not in standing)
        found += found_here
    else:
        proofs = (
            tuple(p for group in measured for p in group)
            if measured is not None
            else measure(bound, unit, overrides, out)
        )
        found = gaps(bound, unit, proofs)
    proved["wall"] = round(wall + time.monotonic() - started, 3)
    code = [n for n in unit.members if bound.layout.members[n].kind == "function"]
    # its data rows are not another member; a bulk source names no single one, the proof set covers them all
    member = params["function"] or (code[0] if len(code) == 1 or (params.get("bulk") and code) else None)
    if member is None:
        raise Refusal(
            Finding(
                "land.request",
                reason="the file holds several members; pass --function",
                unit=unit.path,
                action="pass --function NAME",
            )
        )
    return {
        "unit": unit,
        "proofs": proofs,
        "found": found,
        "member": member,
        "submission": None,
        "consumers": {"count": len(proved["consumers"]), "wall": proved["wall"]},
        "rewritten": rewritten,
        "extras": extras,
        "proving": proved["proving"],
        "pending": pending,
        "tree": bound,
    }


def _kin_pairs(
    bound: Snapshot, units: Sequence[UnitSpec], consumers: Sequence[str], siblings: dict[str, tuple[Snapshot, UnitSpec]]
) -> list[tuple[Snapshot, UnitSpec]]:
    """The change set and old consumers, then each sibling draft in its own tree and under the candidate's writes.
    Keep this order so each group's verdict can split its results after the shared measurement dispatch."""
    plain = [c for c in consumers if c not in siblings]
    after = {
        c: layout.overlay(bound, {p: b for p, b in tree.overlays.items() if p not in bound.overlays})
        for c, (tree, _) in siblings.items()
    }
    pairs = [(bound, u) for u in (*units, *(bound.layout.units[c] for c in (*plain, *siblings)))]
    pairs += [siblings[c] for c in siblings] + [(after[c], siblings[c][1]) for c in siblings]
    return pairs


def _kin_findings(
    snapshot: Snapshot, staged: Json, measured: Sequence[tuple[Proof, ...]], overrides: Json
) -> tuple[tuple[Proof, ...], dict[str, tuple[Finding, ...]]]:
    """Split one candidate's measured pairs. Findings under "" refuse now; those under a sibling path hold only if
    that sibling is refused. No measurements are dispatched while deciding the group's verdict."""
    siblings = staged["siblings"]
    plain = [c for c in staged["proved"]["consumers"] if c not in siblings]
    here = len(staged["units"]) + len(plain) + len(siblings)
    everyone = tuple(p for group in measured[:here] for p in group)
    own = measured[here : here + len(siblings)]
    applied = measured[here + len(siblings) :]
    found: dict[str, tuple[Finding, ...]] = {"": _regressions(snapshot, plain, everyone, overrides)}
    for n, c in enumerate(siblings):
        found[""] += _kin_regressions(staged["unit"].path, own[n], applied[n])
        found[c] = _regressions(snapshot, [c], everyone, overrides, c)
    return everyone, found


def _source_rules(snapshot: Snapshot, done: Json) -> tuple[Finding, ...]:
    """The source rules a candidate refused before entry breaks, so its one refusal lists them with its proofs'
    findings and a fix of the bytes is not followed by a resubmit that only then names a rule."""
    tree, unit = done["tree"], done["unit"]
    group = tree.layout.groups.get(unit.group)
    sdk = bool(group and group.sdk)
    text = tree.peek(unit.path)
    if text is None:  # no source text to read rules from
        return ()
    rules = policy.fresh(snapshot, unit.path, policy.evaluate(tree, unit.path, text.decode(), None, sdk), sdk)
    return tuple(f for f in rules if f.blocking and f not in done["found"])


def prove_groups(
    config: Config, snapshot: Snapshot, groups: Sequence[Sequence[Path]], params: Json
) -> list[tuple[Finding, ...]]:
    """Per alternative change set (each a primary and its PATH=FILE extras, each judged alone against the same tree)
    the findings that would refuse it, proving nothing into the queue: every set's units and every unit including a
    header it changes (as the landing proves them) measured in ONE dispatch on the pool, gated as admission gates them
    (`_standing`, `_regressions`, the source rules)."""
    overrides = _overrides(params)
    chunk_params = {**params, "function": None, "bulk": True}
    prepared: list[Any] = []
    for group in groups:
        try:
            prepared.append(_prepare(snapshot, group, chunk_params))
        except Refusal as refusal:
            prepared.append(refusal)
    live = [n for n, p in enumerate(prepared) if not isinstance(p, Refusal)]
    derived = ownership.derive_pairs(
        [(prepared[n][2], prepared[n][1]) for n in live],
        recipe=[recipes.resolve(config, prepared[n][1], overrides) for n in live],
    )
    sets: dict[int, Any] = {}
    pairs: list[tuple[Snapshot, UnitSpec]] = []
    for n, (bound, unit) in zip(live, derived, strict=True):
        extras = prepared[n][0]
        proved = _proof_set(snapshot, bound, unit, extras, True, every=True)
        units = [unit if p == unit.path else bound.layout.units[p] for p in proved["changed"]]
        everyone = [*units, *(bound.layout.units[p] for p in proved["consumers"])]
        sets[n] = (bound, unit, units, proved, len(pairs), len(everyone))
        pairs += [(bound, u) for u in everyone]
    measured = measure_bound(config, pairs, overrides) if pairs else []
    judged: dict[int, tuple[tuple[Proof, ...], tuple[Proof, ...], tuple[Finding, ...]]] = {}
    heads: set[str] = set()
    for n, (bound, _unit, units, proved, at, count) in sets.items():
        everyone_ = tuple(p for group in measured[at : at + count] for p in group)
        proofs = tuple(p for p in everyone_ if p.unit in proved["changed"])
        judged[n] = (everyone_, proofs, _gaps_of(bound, units, proofs))
        heads.update(proved["consumers"])
        heads.update(u.path for f in judged[n][2] if (u := layout.unit_of(snapshot, f.unit)) is not None)
    if heads:  # every set's HEAD proofs in ONE dispatch (memoised for the gates below), never one dispatch per set
        _baseline(snapshot, sorted(heads), overrides)
    out: list[tuple[Finding, ...]] = []
    for n, done in enumerate(prepared):
        if isinstance(done, Refusal):
            out.append(tuple(done.findings))
            continue
        bound, unit, units, proved, at, count = sets[n]
        everyone_, proofs, gaps_ = judged[n]
        found, standing = _standing(snapshot, unit, gaps_, proofs, overrides, bool(params.get("no_regression")))
        found += _regressions(snapshot, proved["consumers"], everyone_, overrides)
        kept = [p for p in proofs if (p.member, p.version) not in standing]
        if not kept:
            found += (Finding("land.not_exact", reason=f"{unit.path} has no proof that stands", unit=unit.path),)
        out.append(found + _source_rules(snapshot, {"tree": bound, "unit": unit, "found": found}))
    return out


def _enter(
    config: Config, snapshot: Snapshot, files: Sequence[Path], params: Json, overrides: Json, origin: str, done: Json
) -> Json:
    """Submit a proved candidate: a proving change set only when it has proofs and no finding."""
    call = _entry(snapshot, files, params, overrides, origin, done)
    if call is None:  # refused unentered: name its source rules too
        return {**done, "found": done["found"] + _source_rules(snapshot, done), "submission": None}
    keys = ("request", "unit", "proofs", "source", "origin", "extras", "withhold", "snapshot")
    return {**done, "submission": land.submit(config, *(call[k] for k in keys))}


def _entry(
    snapshot: Snapshot, files: Sequence[Path], params: Json, overrides: Json, origin: str, done: Json
) -> dict[str, Any] | None:
    """The `land.submit` arguments of a proved candidate, or None when it is refused before entry."""
    if done["proving"] and (not done["proofs"] or done["found"]):
        return None
    request = {"file": str(files[0]), "function": done["member"], "overrides": overrides, "note": params["note"] or ""}
    return {
        "request": request,
        "unit": done["unit"],
        "proofs": done["proofs"],
        "source": moved.rewrite(snapshot, files[0].read_bytes())[0],
        "origin": origin,
        "extras": done["extras"],
        "withhold": bool(params["withhold"]),
        "snapshot": snapshot,
    }


def _overrides(params: Json) -> dict[str, Any]:
    return {
        "add": list(params.get("flag") or ()),
        "omit": list(params.get("omit_flag") or ()),
        **({"toolchain": params["toolchain"]} if params.get("toolchain") else {}),
    }


def run(config: Config, params: Json) -> Json:
    with effort.stage("compare.run"):
        overrides = _overrides(params)
        out = Path(params["out"]) if params["out"] else None
        done = _attempt(config, [Path(params["file"])], params, overrides, out, "compare")
        unit, proofs, found, submission = done["unit"], done["proofs"], done["found"], done["submission"]
        score = {v: min(p.score for p in proofs if p.version == v) for v in sorted({p.version for p in proofs})}
        return {
            "unit": unit.path,
            "members": list(unit.members),
            "exact": all(p.exact for p in proofs) and not found,
            "score": score,
            "proofs": [asdict(p) for p in proofs],
            "gaps": [asdict(f) for f in found],
            "submitted": submission.id if submission else None,
            "consumers": done["consumers"],
            "rewritten": done["rewritten"],
            "drain": done["drain"],
        }


def submit_command(config: Config, params: Json) -> Json:
    with effort.stage("compare.submit_command"):
        if params.get("batch_data"):
            from unbake import batch_data

            return batch_data.run(config, params)
        if not params["files"]:
            raise Refusal(Finding("land.request", "Pass source files or --batch-data MANIFEST."))
        files = [Path(f) for f in params["files"]]
        if params["bulk"]:
            done = _bulk(config, files, params)
            bad = [f for row in done["submissions"] for f in row["findings"] if "holds several members" in f["reason"]]
            if bad:  # a source bulk can never name one member for is an error, not a silent row
                raise Refusal(*(Finding(**{**f, "origin": None}) for f in bad))
            return done
        done = _attempt(config, files, params, _overrides(params), None, "submit")
        proofs, entry = done["proofs"], done["submission"]
        row = {
            "file": str(files[0]),
            "files": [str(f) for f in files],
            "member": done["member"],
            "id": entry.id if entry else None,
            "operation": entry.operation if entry else "none",
            "exact": all(p.exact for p in proofs),
            "score": min(p.score for p in proofs),
            "findings": [asdict(f) for f in done["found"]],
            "consumers": done["consumers"],
        }
        return {"submissions": [row], "drain": done["drain"]}


def _bulk(config: Config, files: Sequence[Path], params: Json) -> Json:
    """A source with the PATH=FILE files after it is one atomic candidate, admitted on its own per chunk of groups; a
    refusal is its row and blocks no other group. The --toolchain and --flag options apply to every group."""
    groups: list[list[Path]] = []
    for file in files:
        if groups and "=" in str(file) and not file.exists():
            groups[-1].append(file)
        else:
            groups.append([file])
    rows: list[dict[str, Any]] = []
    chunk = configuration.fuzzy()["chunk"]
    for at in range(0, len(groups), chunk):
        snapshot = layout.capture(config)  # the chunk is bounded so what one capture and its overlays hold stays small
        chunk_params = {**params, "function": None, "bulk": True}
        prepared: dict[int, Any] = {}
        trials: dict[int, Recipe] = {}
        for n, group in enumerate(groups[at : at + chunk]):
            try:
                prepared[n] = _prepare(snapshot, group, chunk_params)
                trials[n] = recipes.resolve(config, prepared[n][1], _overrides(params))
            except Refusal as refusal:
                prepared[n] = refusal
        bound = {n: p for n, p in prepared.items() if not isinstance(p, Refusal)}
        # every candidate's tree state is derived in one dispatch; admission and its proofs reuse those derivations
        ownership.derive_pairs([(p[2], p[1]) for p in bound.values()], recipe=[trials[n] for n in bound])
        kin = {p[1].path: (p[2], p[1]) for p in bound.values()}  # every member's draft, by its unit path
        staged: dict[int, Any] = {}
        for n, group in enumerate(groups[at : at + chunk]):
            try:
                if isinstance(prepared[n], Refusal):
                    raise prepared[n]
                others = {path: tree for path, tree in kin.items() if path != prepared[n][1].path}
                staged[n] = _stage(config, snapshot, group, chunk_params, _overrides(params), prepared[n], others)
            except Refusal as refusal:
                staged[n] = refusal
        # Warm the union of actual consumers once, then measure every candidate and sibling draft in one dispatch.
        # Each pair keeps its own snapshot; only scheduling is shared, never candidate writes or verdicts.
        batched = {n: s["batch"] for n, s in staged.items() if not isinstance(s, Refusal)}
        consumers = sorted({c for n in batched for c in staged[n]["proved"]["consumers"]})
        pairs = [pair for batch in batched.values() for pair in batch]
        started = time.monotonic()
        if consumers:
            _baseline(snapshot, consumers, _overrides(params))
        measured = iter(measure_bound(config, pairs, _overrides(params)) if pairs else [])
        wall = time.monotonic() - started
        proofs = {n: [next(measured) for _ in batch] for n, batch in batched.items()}
        proved: dict[int, Any] = {}
        for n in range(len(groups[at : at + chunk])):
            try:
                if isinstance(staged[n], Refusal):
                    raise staged[n]
                proved[n] = _finish(
                    config, snapshot, chunk_params, _overrides(params), None, staged[n], proofs.get(n), wall
                )
            except Refusal as refusal:
                proved[n] = refusal
        _settle(proved, {p[1].path: n for n, p in bound.items()})
        # every entering group's acceptance check runs in ONE dispatch; a refusal at entry is that group's row only
        calls: dict[int, dict[str, Any]] = {}
        for n, group in enumerate(groups[at : at + chunk]):
            done = proved[n]
            if isinstance(done, Refusal):
                continue
            call = _entry(snapshot, group, chunk_params, _overrides(params), "submit", done)
            if call is None:
                proved[n] = {**done, "found": done["found"] + _source_rules(snapshot, done), "submission": None}
            else:
                calls[n] = call
        for n, entered in zip(calls, land.submit_many(config, list(calls.values())), strict=True):
            proved[n] = entered if isinstance(entered, Refusal) else {**proved[n], "submission": entered}
        for n, group in enumerate(groups[at : at + chunk]):
            done = proved[n]
            file = group[0]
            row: dict[str, Any] = {
                "file": str(file),
                "files": [str(f) for f in group],
                "member": file.stem,
                "id": None,
                "operation": "none",
                "exact": False,
                "score": 0.0,
                "findings": [],
                "consumers": {"count": 0, "wall": 0.0},
            }
            if isinstance(done, Refusal):
                row["findings"] = [asdict(f) for f in done.findings]
                row["refused"] = True
            else:
                entry = done["submission"]
                row["refused"] = bool(done["proving"] and (not done["proofs"] or done["found"]))
                row.update(
                    member=done["member"],
                    id=entry.id if entry else None,
                    operation=entry.operation if entry else "none",
                    exact=all(p.exact for p in done["proofs"]),
                    score=min((p.score for p in done["proofs"]), default=0.0),
                    findings=[asdict(f) for f in done["found"]],
                    consumers=done["consumers"],
                )
            rows.append(row)
    return {"submissions": rows, "drain": None}


def _settle(proved: dict[int, Any], members: dict[str, int]) -> None:
    """A candidate proved against a sibling's draft keeps that sibling's old-source regressions when the sibling is
    refused; refusals cascade until none is added."""
    while True:
        refused = {
            n for n, d in proved.items() if isinstance(d, Refusal) or (d["proving"] and (not d["proofs"] or d["found"]))
        }
        added = False
        for n, done in proved.items():
            if n in refused or isinstance(done, Refusal):
                continue
            lost = tuple(f for path, found in done["pending"].items() if members[path] in refused for f in found)
            if lost:
                proved[n] = {**done, "found": done["found"] + lost, "pending": {}}
                added = True
        if not added:
            return


def _compact(diff: Sequence[str], limit: int) -> list[str]:
    """Every hunk of a unified diff, reduced to its header and changed lines, in at most `limit` lines."""
    rows = [line for line in diff if line.startswith(("@@", "-", "+")) and not line.startswith(("---", "+++"))]
    return rows[:limit]


def _compile_failure(unit: UnitSpec, version: str, proofs: Sequence[Proof]) -> Finding:
    """Every compiler error line, which a proof that built nothing carries in its missing list."""
    lines = [ln for p in proofs for m in p.missing for ln in m.splitlines() if ln.strip()]
    return Finding(
        "score.compile",
        reason="; ".join(lines) or "no instructions were built",
        unit=unit.path,
        versions=(version,),
        action="fix the compile error shown",
    )


def score_command(config: Config, params: Json) -> Json:
    """Score candidates in one version and write nothing (no journal, inbox, landing or artifacts): the ordered
    match against the ROM with the differing lines, where they part (diagnostic), the source rules broken, and with
    --alloc the registers and the instruction order. --variants scores a JSON list of {file, overlays} in one process,
    side by side."""
    with effort.stage("compare.score"):
        if params.get("variants"):
            return _score_variants(config, params)
        if not params.get("file"):
            raise Refusal(Finding("land.request", reason="score needs a C source", action="pass FILE or --variants"))
        return _score_one(layout.capture(config), params)


def _variant_params(params: Json, row: Any) -> Json:
    if not isinstance(row, dict) or not isinstance(row.get("file"), str) or not row["file"]:
        raise Refusal(
            Finding(
                "land.request",
                reason=f"--variants rows must be objects with a nonempty file path, got {str(row)[:80]}",
                action="pass a JSON list of {file, overlays}",
            )
        )
    unknown = sorted(set(row) - {"file", "overlays", "flags", "omit_flags", "function"})
    if unknown:  # a misspelt key would score the variant without what it names
        raise Refusal(
            Finding(
                "land.request",
                reason=f"--variants rows take file, overlays, flags, omit_flags, function: not {', '.join(unknown)}",
                action="pass a JSON list of {file, overlays, flags, omit_flags, function}",
            )
        )
    for key in ("flags", "omit_flags"):  # a row's own flag set: one call scores a whole flag lattice
        value = row.get(key)
        if value is not None and (not isinstance(value, list) or any(not isinstance(f, str) or not f for f in value)):
            raise Refusal(
                Finding(
                    "land.request",
                    reason=f"--variants {key} must be a list of nonempty compiler options or null",
                    action='pass e.g. "flags": ["-ffast-math"]',
                )
            )
    if row.get("function") is not None and (not isinstance(row["function"], str) or not row["function"]):
        raise Refusal(Finding("land.request", reason="--variants function must be a member name or null"))
    overlays = row.get("overlays")
    if overlays is not None and (
        not isinstance(overlays, list) or any(not isinstance(o, str) or not o for o in overlays)
    ):
        raise Refusal(
            Finding(
                "land.request",
                reason="--variants overlays must be a list of nonempty path strings or null",
                action="pass overlays as a JSON list of file paths or PATH=FILE strings",
            )
        )
    own = {
        **({"flag": list(row["flags"])} if row.get("flags") is not None else {}),
        **({"omit_flag": list(row["omit_flags"])} if row.get("omit_flags") is not None else {}),
        **({"function": row["function"]} if row.get("function") is not None else {}),
    }
    return {**params, **own, "file": Path(row["file"]), "extras": [Path(o) for o in overlays or ()]}


def _variant_job(item: tuple[Snapshot, Json]) -> Json:
    snapshot, params = item
    try:
        result = _score_one(snapshot, params)
    except Refusal as refusal:
        flags = {"flags": list(params.get("flag") or ()), "omit_flags": list(params.get("omit_flag") or ())}
        return {"file": str(params["file"]), **flags, "ok": False, "findings": [asdict(f) for f in refusal.findings]}
    flags = {"flags": list(params.get("flag") or ()), "omit_flags": list(params.get("omit_flag") or ())}
    return {
        "file": str(params["file"]),
        **flags,
        "ok": True,
        "result": {k: v for k, v in result.items() if k != "proofs"},
    }


def _score_variants(config: Config, params: Json) -> Json:
    """Validate the whole manifest before capturing once, then dispatch each variant as one complete score."""
    manifest = Path(params["variants"])
    try:
        rows = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Refusal(
            Finding(
                "land.request",
                reason=f"--variants is not JSON at line {error.lineno}, column {error.colno}: {error.msg}",
                path=str(manifest),
                line=error.lineno,
                action="fix the JSON in the variants input file",
            )
        ) from error
    except (OSError, UnicodeDecodeError) as error:
        raise Refusal(
            Finding(
                "land.request",
                reason=f"cannot read --variants: {error}",
                path=str(manifest),
                action="pass a readable UTF-8 JSON file",
            )
        ) from error
    if not isinstance(rows, list):
        raise Refusal(
            Finding(
                "land.request",
                reason="--variants must hold a JSON list",
                path=str(manifest),
                action="pass {file, overlays} rows",
            )
        )
    variants = []
    for index, row in enumerate(rows, 1):
        try:
            variants.append(_variant_params(params, row))
        except Refusal as error:
            raise Refusal(
                *(replace(f, path=str(manifest), reason=f"row {index}: {f.reason}") for f in error.findings)
            ) from error
    snapshot = layout.capture(config)
    jobs = [(snapshot, variant) for variant in variants]
    return {"variants": pool.map(config, "compare.variants", _variant_job, jobs)}


def _score_one(snapshot: Snapshot, params: Json) -> Json:
    config = snapshot.config
    extras = dict(_extra(snapshot, Path(spec)) for spec in params.get("extras") or ())  # overlaid, never written
    rewritten: list[str] = []
    unit, bound = bind(
        layout.overlay(snapshot, extras) if extras else snapshot, Path(params["file"]), params["function"], rewritten
    )
    overrides = _overrides(params)
    bound, unit = ownership.derive(bound, unit, loose=True, recipe=recipes.resolve(config, unit, overrides))
    held = holders(bound, unit)
    preferred = config.project.names_from
    version = params["version"] or (preferred if preferred in held else held[0])
    if version not in held:
        raise Refusal(
            Finding(
                "land.request",
                reason=f"{unit.path} is not held by version {version}",
                versions=held,
                action="pass --version one of the holding versions",
            )
        )
    recipe = recipes.resolve(config, unit, overrides)
    code = [n for n in unit.members if bound.layout.members[n].kind == "function"]
    with store.work(config) as work:
        proofs = native.prove(bound, unit, version, recipe, work, True)
        places = sorted(
            (
                p
                for n in unit.members
                for p in bound.layout.members[n].placements
                if p.version == version and p.section == ".text"
            ),
            key=lambda p: p.rom_start,
        )
        built = (work / "text.bin").read_bytes() if places and (work / "text.bin").exists() else b""
        target = versions.rom_bytes(bound.versions[version], places[0].rom_start, places[-1].rom_end) if places else b""
        dumps = native.dumps(bound, unit, version, recipe, work / "alloc") if params["alloc"] else None
    text = bound.read(unit.path).decode()
    source = policy.fresh(
        snapshot,
        unit.path,
        policy.evaluate(
            bound, unit.path, text, view.get(bound, unit, version, recipe), bound.layout.groups[unit.group].sdk
        ),
        bound.layout.groups[unit.group].sdk,
    )
    if not built and proofs and not all(p.exact for p in proofs):
        raise Refusal(_compile_failure(unit, version, proofs))
    everything = symptoms.diff(built, target, places[0].vram).splitlines() if built else []
    shown = _compact(everything, params["lines"] or 40) if params.get("short") else everything[: params["lines"] or 80]
    matched = round(symptoms.score(built, target) * len(target) / 4) if built and target else 0
    result = {
        "unit": unit.path,
        "members": list(unit.members),
        "version": version,
        "versions": list(held),
        "exact": bool(proofs) and all(p.exact for p in proofs),
        "score": min((p.score for p in proofs if bound.layout.members[p.member].kind == "function"), default=0.0),
        "words": {"built": len(built) // 4, "target": len(target) // 4, "matched": matched},
        "shift": symptoms.shift(built, target),
        "source": [asdict(f) for f in source],
        "diff": shown,
        "proofs": [asdict(p) for p in proofs],
        "rewritten": rewritten,
    }
    if dumps:
        name = params["function"] or code[0]
        greg, lreg, asm, passes = dumps
        result["alloc"] = alloc.report(name, greg, lreg, asm, symptoms.register_votes(built, target))
        mine = [p for p in bound.layout.members[name].placements if p.version == version and p.section == ".text"]
        lo = (mine[0].rom_start - places[0].rom_start) // 4 if mine else 0
        words = (mine[-1].rom_end - mine[0].rom_start) // 4 if mine else 0
        every = symptoms.listing(built, places[0].vram)
        result["sched"] = sched.report(name, passes, every[lo : lo + words], everything)
    return result
