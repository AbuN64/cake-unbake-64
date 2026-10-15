"""Resource-driven native object production and per-member ROM proofs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import abucache
from abucache import compile

from unbake import (
    adapters,
    build,
    config,
    declarations,
    effort,
    journal,
    process,
    recipes,
    store,
    symptoms,
    versions,
    view,
)
from unbake.contracts import (
    Config,
    Finding,
    NativeResult,
    Origin,
    Placement,
    Proof,
    Recipe,
    Refusal,
    Snapshot,
    UnitSpec,
    code_identity,
    digest,
    proof_from,
    sha256,
)

_WORD = 0xFFFFFFFF


@contextmanager
def _tools(unit: UnitSpec, *context: Any) -> Iterator[None]:  # context: snapshot, unit, version, recipe or nothing
    try:
        yield
    except Refusal as error:
        raise Refusal(
            *(
                replace(
                    f,
                    reason=f"{f.reason}; unit kind {unit.kind}"
                    if f.key == "native.missing_tool"
                    else view.diagnostic(context[0], context[1], context[2], context[3], f.reason)
                    if context
                    else f.reason,
                )
                for f in error.findings
            )
        ) from error


def _phases(unit: UnitSpec) -> tuple[str, ...]:
    return tuple(config.kind(unit.kind)["phases"])


def _held(snapshot: Snapshot, unit: UnitSpec, version: str) -> tuple[tuple[str, Placement], ...]:
    return tuple(
        (name, p) for name in unit.members for p in snapshot.layout.members[name].placements if p.version == version
    )


def sections(snapshot: Snapshot, unit: UnitSpec, version: str) -> tuple[Placement, ...]:
    """One contiguous ROM/memory range per section owned by this unit in a holder."""
    held, output = _held(snapshot, unit, version), []
    for section in sorted({p.section for _, p in held}):
        placements = sorted((p for _, p in held if p.section == section), key=lambda p: p.vram)
        first, last = placements[0], placements[-1]
        if any(
            a.vram + a.size != b.vram or (section != ".bss" and a.rom_end != b.rom_start)
            for a, b in pairwise(placements)
        ):
            raise Refusal(Finding("link.error", f"owned {section} run is not contiguous", unit=unit.path))
        vram = first.vram & snapshot.config.project.build.get("text_mask", _WORD) if section == ".text" else first.vram
        output.append(
            Placement(
                version,
                section,
                first.rom_start,
                last.rom_end,
                vram,
                sum(p.size for p in placements) if section == ".bss" else 0,
            )
        )
    return tuple(output)


def stamp(
    snapshot: Snapshot,
    unit: UnitSpec,
    version: str,
    recipe: Recipe | None = None,
    *,
    pinned: bool = False,
    code: bool = True,
) -> str:
    """Everything a proof reads, named without building: the unit's source, the declarations its preprocessed text
    reaches, rows, version facts and tools. A stored failure replays under the stamp, so any edit of a header the unit
    includes moves it, reached declaration or not."""
    cfg = snapshot.config
    recipe = recipe or recipes.resolve(cfg, unit, {})
    reach = view.closure(snapshot, unit, version) if pinned else view.reads(snapshot, unit, version, recipe, whole=True)
    record = snapshot.versions[version]
    tools = effort.memo(
        ("tools", recipe.toolchain), lambda: (adapters.host_tools(cfg), adapters.tool_identity(cfg, recipe.toolchain))
    )
    return digest(
        (
            replace(unit, withheld=()),
            reach,
            recipe.digest,
            _held(snapshot, unit, version),
            [snapshot.layout.members[n].state for n in unit.members],
            record.rom_sha256,
            versions.facts_digest(record),
            cfg.project.build,
            cfg.project.version_macros[version],
            tools,
            abucache.__version__,
            code_identity(prove, measure, objects, compiles, warnings, recorded, dumps) if code else "",
        )
    )


def _literal_reads(
    snapshot: Snapshot, version: str, base: Placement | None, built: bytes
) -> tuple[Callable[[int, int], bytes], Callable[[int, int], bytes]]:
    """Readers of `size` bytes at a vram in the built rodata and in the target's, empty outside them."""
    start = base.vram if base else 0

    def ours(address: int, size: int) -> bytes:
        return built[address - start : address - start + size] if base and address >= start else b""

    def theirs(address: int, size: int) -> bytes:
        if not base or not start <= address < start + base.size:
            return b""
        at = base.rom_start + address - start
        return versions.rom_bytes(snapshot.versions[version], at, at + size)

    return ours, theirs


def _gaps(
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, object_sha: str, missing: tuple[str, ...]
) -> tuple[Proof, ...]:
    source_sha = sha256(snapshot.read(unit.path))
    return tuple(
        Proof(
            unit.path,
            name,
            version,
            recipe.digest,
            source_sha,
            object_sha,
            "",
            "",
            False,
            missing,
            0.0,
            symptoms.from_missing(missing),
        )
        for name in dict.fromkeys(n for n, _ in _held(snapshot, unit, version))
    )


_EXACT: ContextVar[bool] = ContextVar(
    "native.exact", default=False
)  # set: an object is reused only from its whole text


@contextmanager
def exactly() -> Iterator[None]:
    """Within, an object is reused only when the whole preprocessed text it was built from is unchanged."""
    token = _EXACT.set(True)
    try:
        yield
    finally:
        _EXACT.reset(token)


def _cacheable(unit: UnitSpec) -> bool:
    return bool(config.kind(unit.kind)["cacheable"])


def _replayed[T](cfg: Config, key: str, produce: Callable[[], T]) -> T:
    """A failed build is as final as an object: its refusal is stored under the key and replayed."""
    refused = store.get(cfg, "refused", key)
    if refused is not None:
        effort.count("refused", True)
        raise Refusal(*(_finding(row) for row in json.loads(refused)))
    try:
        return produce()
    except Refusal as refusal:
        effort.count("refused", False)
        store.put(cfg, "refused", key, json.dumps([asdict(f) for f in refusal.findings]).encode())
        raise


def _compile_inputs(
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, include: tuple[Path, ...], stem: str
) -> tuple[str, tuple[compile.Step, ...], str, str, str]:
    seen = view.get(snapshot, unit, version, recipe, lines=False)
    text = seen.text
    # The assembler reads a file only through a directive the compiled text carries. Without one, the include
    # directories (absolute paths of this checkout) are not an input, so they stay out of the key and every
    # worktree of the project shares its objects.
    reads = re.search(r"\.(?:include|incbin)\b", text) is not None
    steps = adapters.chain_steps(snapshot.config, recipe.toolchain, recipe, include if reads else (), stem)
    tools = str(
        effort.memo(
            ("tool-identity", recipe.toolchain), lambda: adapters.tool_identity(snapshot.config, recipe.toolchain)
        )
    )
    live = view.live_digest(
        snapshot.config, seen
    )  # a declaration the unit never reaches cannot change its object: it is not in the key
    return text, steps, tools, compile.key(live.encode(), f"{stem}.i", steps, tools), seen.unit


def _compiled(
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path, include: tuple[Path, ...], stem: str
) -> tuple[Path, tuple[NativeResult, ...]]:
    """Preprocess, compile, assemble through abucache's cache keyed by preprocessed bytes, commands and tools."""
    cfg = snapshot.config
    if "preprocess" not in _phases(unit):
        raise Refusal(Finding("config.resource", f"unit kind {unit.kind} is cacheable but has no preprocess phase"))
    text, steps, tools, key, own = _compile_inputs(snapshot, unit, version, recipe, include, stem)
    source, obj = work / f"{stem}.i", work / f"{stem}.o"
    source.write_text(text, encoding="utf-8")

    def run(phase: str, argv: Sequence[str], cwd: Path) -> NativeResult:
        return process.run(phase, argv, cwd, tmp=process.scratch(snapshot.config.project.root))

    full = compile.key(
        text.encode(), source.name, steps, tools
    )  # a failure is a fact of the whole text, never of the live part

    def headers_compile() -> (
        bool
    ):  # the headers compile alone: proved once per header text and compiler, not per consumer
        header_text = re.sub(r'(?m)^# \d+ "[^"\n]*"[^\n]*\n', "", declarations.headers(text, own))
        proof = digest((header_text, steps[0].argv, tools, "headers"))

        def prove() -> bytes:
            with store.work(cfg) as home:
                (home / f"{stem}.i").write_text(header_text, encoding="utf-8")
                argv = [w.replace("{in}", f"{stem}.i").replace("{out}", steps[0].out) for w in steps[0].argv]
                result = process.run("compile", argv, home, tmp=process.scratch(cfg.project.root))
            with suppress(Refusal):
                adapters.toolchain(cfg, recipe.toolchain).check(steps[0], result)
                return b"yes" if result.exit == 0 and result.signal is None else b"no"
            return b"no"

        return store.cached(cfg, "header-proof", proof, prove) == b"yes"

    def build() -> tuple[
        bytes, list[NativeResult]
    ]:  # an earlier text's object with the same live part if the headers compile
        stored = store.get(cfg, "unit", full) is not None
        kept = None if _EXACT.get() or stored else store.get(cfg, "live-object", key)
        if kept is not None and headers_compile():
            effort.count("live-object", True)
            return kept, []
        if not _EXACT.get() and not stored:  # a stored unit never tries the shortcut: it is no miss
            effort.count("live-object", False)
        value, results = compile.build(
            store.content(cfg), run, steps, source, tools, work, after=adapters.toolchain(cfg, recipe.toolchain).check
        )
        if store.get(cfg, "unit", full) is not None:  # it was stored
            store.put(cfg, "live-object", key, value)
        return value, results

    value, results = _replayed(cfg, full, build)
    obj.write_bytes(value)
    if results:  # the compile that ran keeps its refused warnings, so nothing compiles again to count them
        found = adapters.toolchain(cfg, recipe.toolchain).refused(results[0].stderr.decode(errors="replace"))
        store.put(cfg, "warnings", digest((key, "warnings")), json.dumps(found).encode())
    return obj, tuple(results)


def compiles(
    item: tuple[Snapshot, UnitSpec, str], home: Path | None = None
) -> tuple[Finding, ...]:  # why the unit does not build (compile, link, owned sizes); empty when it does
    snapshot, unit, version = item
    try:
        with (
            nullcontext(home) if home else store.work(snapshot.config)
        ) as work:  # make links (and asserts owned sizes) only where the unit is not withheld
            obj = objects(snapshot, unit, version, r := recipes.resolve(snapshot.config, unit, {}), work)[0]
            proofs = () if version in unit.withheld else measure(snapshot, unit, version, r, obj, work)
        return tuple(
            Finding("link.error", f"{unit.path}: {m}", unit=unit.path, versions=(version,))
            for m in sorted({m for p in proofs for m in p.missing if "bytes differ" not in m})
        )
    except Refusal as refusal:
        return tuple(replace(f, unit=f.unit or unit.path, versions=f.versions or (version,)) for f in refusal.findings)


def dumps(
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path
) -> tuple[str, str, str, dict[str, str]]:
    """The allocation and scheduler dumps and debug assembly of one compile made outside the cache (a cached object
    has none); the scheduler dumps are empty when the toolchain writes none."""
    stem = Path(unit.path).stem
    text, steps, *_ = _compile_inputs(snapshot, unit, version, recipe, _include(snapshot, unit, version), stem)
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{stem}.i").write_text(text, encoding="utf-8")
    argv = [a.replace("{in}", f"{stem}.i").replace("{out}", f"{stem}.s") for a in steps[0].argv]
    argv += ["-g", "-dg", "-dl", "-dS", "-dR"]
    result = process.run("native.dumps", argv, work, tmp=process.scratch(snapshot.config.project.root))
    found = [work / f"{stem}.i.{kind}" for kind in ("greg", "lreg")]
    if result.exit != 0 or not all(f.is_file() for f in found):
        raise Refusal(
            Finding(
                "land.request",
                reason=f"{recipe.toolchain} writes no register allocation dumps",
                unit=unit.path,
                action="use a gcc 2.x toolchain: pass --toolchain",
            )
        )
    greg, lreg, asm = (f.read_text(errors="replace") for f in (*found, work / f"{stem}.s"))
    passes = {k: f.read_text(errors="replace") for k in ("sched", "sched2") if (f := work / f"{stem}.i.{k}").is_file()}
    return greg, lreg, asm, passes


def recorded(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe) -> tuple[str, ...] | None:
    """The refused warnings the build or a proof kept for this unit; None when none was kept; never compiles."""
    cfg = snapshot.config
    if not _cacheable(unit):
        return ()
    home = cfg.project.root / "build" / version / Path(unit.path).with_suffix("")
    for kept in sorted(home.glob("*/compile.err")):
        return tuple(adapters.toolchain(cfg, recipe.toolchain).refused(kept.read_text(errors="replace")))
    *_, key, _ = _compile_inputs(
        snapshot, unit, version, recipe, _include(snapshot, unit, version), Path(unit.path).stem
    )
    kept_json = store.get(cfg, "warnings", digest((key, "warnings")))
    return None if kept_json is None else tuple(json.loads(kept_json))


def warnings(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path) -> tuple[str, ...]:
    """The refused-view warnings of the unit's own compile, kept by the text compiled."""
    with effort.stage("native.warnings"), _tools(unit, snapshot, unit, version, recipe):
        cfg = snapshot.config
        if not _cacheable(unit):
            return ()
        work.mkdir(parents=True, exist_ok=True)
        stem = Path(unit.path).stem
        text, steps, _, key, _ = _compile_inputs(
            snapshot, unit, version, recipe, _include(snapshot, unit, version), stem
        )

        def _produce() -> bytes:
            (work / f"{stem}.i").write_text(text, encoding="utf-8")
            argv = [w.replace("{in}", f"{stem}.i").replace("{out}", steps[0].out) for w in steps[0].argv]
            result = process.run("compile", argv, work, tmp=process.scratch(cfg.project.root))
            found = adapters.toolchain(cfg, recipe.toolchain).refused(result.stderr.decode(errors="replace"))
            return json.dumps(found).encode()

        return tuple(json.loads(store.cached(cfg, "warnings", digest((key, "warnings")), _produce)))


def _include(snapshot: Snapshot, unit: UnitSpec, version: str) -> tuple[Path, ...]:
    project = snapshot.config.project
    macros = config.load_resource("repo.toml")["splat"]["options"]["generated_asm_macros_directory"]
    return (
        project.root / "include",
        (project.root / unit.path).parent,
        project.root / macros.format(version=version, name=project.name),
    )


def objects(
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path
) -> tuple[Path, tuple[NativeResult, ...]]:
    with effort.stage("native.objects"), _tools(unit, snapshot, unit, version, recipe):
        work.mkdir(parents=True, exist_ok=True)
        phases, stem = _phases(unit), Path(unit.path).stem
        row = config.toolchain(recipe.toolchain)
        try:
            source = snapshot.read(unit.path)
        except FileNotFoundError:
            if unit.path in snapshot.overlays or not journal.changed(
                snapshot, ("layout.toml", *(v.split for v in snapshot.versions.values()))
            ):
                raise
            raise Refusal(
                Finding("journal.changed", config.sentence("journal.changed"), path=unit.path, unit=unit.path)
            ) from None
        obj = work / f"{stem}.o"
        results: list[NativeResult] = []
        include = _include(snapshot, unit, version)
        if _cacheable(unit):
            return _compiled(snapshot, unit, version, recipe, work, include, stem)
        dependencies, pending = {}, [source]
        while pending:
            for name in re.findall(rb'\.include\s+"([^"\n]+)"', pending.pop()):
                path = next((p / name.decode() for p in include if (p / name.decode()).is_file()), None)
                if path and str(path) not in dependencies:
                    data = path.read_bytes()
                    dependencies[str(path)] = sha256(data)
                    pending.append(data)
        key = digest((sha256(source), recipe.digest, recipe.toolchain, row["pins"], sorted(dependencies.items())))

        def _produce() -> bytes:
            current = None
            for phase in (p for p in phases if p in ("assemble", "armips")):
                if current is None:
                    current = work / f"{stem}.s"
                    current.write_bytes(source)
                out = obj if phase == "assemble" else work / f"{stem}.bin"
                results.append(
                    adapters.assembler(snapshot.config, "gnu" if phase == "assemble" else phase).assemble(
                        current, recipe, include, out
                    )
                )
                current = out
            assert current is not None  # the phases hold an assembler
            return current.read_bytes()

        obj.write_bytes(_replayed(snapshot.config, key, lambda: store.cached(snapshot.config, "object", key, _produce)))
        return obj, tuple(results)


def _finding(row: dict[str, Any]) -> Finding:
    origin = Origin(**row["origin"]) if row["origin"] else None
    return Finding(**{**row, "origin": origin, "versions": tuple(row["versions"]), "missing": tuple(row["missing"])})


def measure(
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, obj: Path, work: Path, keep: bool = False
) -> tuple[Proof, ...]:
    """The proofs of one unit in one version; `keep` links again so the linker's outputs are left in `work`."""
    with effort.stage("native.measure"), _tools(unit, snapshot, unit, version, recipe):
        ranges = sections(snapshot, unit, version)
        if not ranges:
            return ()
        from unbake import ownership

        blob = obj.read_bytes()
        linked_ranges = ranges
        if "armips" not in _phases(unit):
            linked_ranges = (*ranges, *ownership.borrowed(snapshot, unit, version, blob, ranges))
        object_sha = sha256(blob)
        source_sha = sha256(snapshot.read(unit.path))
        record = snapshot.versions[version]
        names = versions.undefined(obj) if "armips" not in _phases(unit) else ()
        selected = replace(record, symbols={n: record.symbols[n] for n in names if n in record.symbols})
        symbols_data = build.symbols_ld(replace(snapshot, versions={**snapshot.versions, version: selected}), version)
        row = config.toolchain(recipe.toolchain)
        trim = not row["preserves_padding"] or snapshot.layout.members[unit.members[0]].state == "hasm"
        held = [(n, asdict(p)) for n, p in _held(snapshot, unit, version)]  # member names are part of the proof
        key = digest(
            (
                object_sha,
                [asdict(p) for p in linked_ranges],
                held,
                sha256(symbols_data),
                trim,
                abucache.__version__,
                digest(
                    sorted(
                        (k, v)
                        for k, v in adapters.host_tools(snapshot.config).items()
                        if k in ("mips_ld", "mips_objcopy")
                    )
                ),
                unit.path,
                recipe.digest,
                source_sha,
            )
        )

        def _produce() -> bytes:
            built, proofs = {}, []
            if "armips" in _phases(unit):
                if len(ranges) != 1 or ranges[0].section == ".bss":
                    gaps = _gaps(
                        snapshot, unit, version, recipe, object_sha, (f"version {version}: unproved resource sections",)
                    )
                    return json.dumps([asdict(p) for p in gaps], sort_keys=True).encode()
                built[ranges[0].section] = obj.read_bytes()
            else:
                work.mkdir(parents=True, exist_ok=True)
                symbols = work / "symbols.ld"
                symbols.write_bytes(symbols_data)
                result = adapters.linker(snapshot.config).link(obj, linked_ranges, symbols, work, trim)
                built = {s: p.read_bytes() for s, p in result.outputs.items() if s.startswith(".")}
            by_section = {p.section: p for p in ranges}
            for name in dict.fromkeys(n for n, _ in _held(snapshot, unit, version)):
                material, target, missing, measured = [], [], [], []
                for member, p in _held(snapshot, unit, version):
                    if member != name:
                        continue
                    base, data = by_section[p.section], built.get(p.section)
                    if data is None:
                        missing.append(f"version {version}: unproved {p.section}")
                        continue
                    if p.section == ".bss":
                        if data != str(base.size).encode():
                            missing.append(f"version {version}: unproved .bss size")
                        material.append(data)
                        target.append(str(base.size).encode())
                        continue
                    b = data[p.rom_start - base.rom_start : p.rom_end - base.rom_start]
                    t = versions.rom_bytes(snapshot.versions[version], p.rom_start, p.rom_end)
                    measured.append((b, t, p.section))
                    if len(data) != base.size:
                        missing.append(f"version {version}: {p.section} size {len(data)} != {base.size}")
                    differences = [i for i, (x, y) in enumerate(zip(b, t, strict=False)) if x != y]
                    if differences:
                        missing.append(
                            f"version {version}: {len(differences)} bytes differ at +0x{differences[0]:X} {p.section}"
                        )
                    material.append(b)
                    target.append(t)
                exact = not missing
                reads = _literal_reads(snapshot, version, by_section.get(".rodata"), built.get(".rodata", b""))
                score_pairs = [(symptoms.literals(b, t, *reads) if s == ".text" else b, t) for b, t, s in measured]
                score = (
                    1.0
                    if exact
                    else min(
                        symptoms.score(b"".join(b for b, _ in score_pairs), b"".join(t for _, t in score_pairs)),
                        1.0 - 2**-53,
                    )
                )
                facts = (
                    {}
                    if exact
                    else dict(
                        symptoms.merge(
                            [symptoms.measure(b, t, s) for b, t, s in measured] + [symptoms.from_missing(missing)]
                        )
                    )
                )
                if not exact:
                    facts["score"] = score
                proof = Proof(
                    unit.path,
                    name,
                    version,
                    recipe.digest,
                    source_sha,
                    object_sha,
                    sha256(b"".join(material)),
                    sha256(b"".join(target)),
                    exact,
                    tuple(missing),
                    score,
                    facts,
                )
                proofs.append(proof)
                store.put(snapshot.config, "bytes", proof.built_sha256, b"".join(material))
            return json.dumps([asdict(p) for p in proofs], sort_keys=True).encode()

        if keep:
            blob = _produce()
            store.put(snapshot.config, "proof", key, blob)
        else:
            blob = store.cached(snapshot.config, "proof", key, _produce)
        return tuple(proof_from(p) for p in json.loads(blob))


def prove_args(item: tuple[Any, ...]) -> tuple[Proof, ...]:
    return prove(*item)


def prove_job(item: tuple[Snapshot, UnitSpec, str]) -> tuple[Proof, ...]:
    """A pool job: the proof of the unit in one version, in a private work directory."""
    snapshot, unit, version = item
    with store.work(snapshot.config) as work:
        return prove(snapshot, unit, version, recipes.resolve(snapshot.config, unit, {}), work)


def prove(
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path, keep: bool = False
) -> tuple[Proof, ...]:
    with effort.stage("native.prove"):
        object_sha = ""
        try:
            obj, _ = objects(snapshot, unit, version, recipe, work)
            object_sha = sha256(obj.read_bytes())
            names = versions.undefined(obj) if "armips" not in _phases(unit) else ()
            record = snapshot.versions[version]
            findings = versions.resolve(record, names, unit.path)
            if findings:
                missing = tuple(f"version {version}: unresolved {name}" for name in names if name not in record.symbols)
                return _gaps(snapshot, unit, version, recipe, object_sha, missing)
            return measure(snapshot, unit, version, recipe, obj, work, keep)
        except Refusal as refusal:
            if any(f.key == "native.missing_tool" for f in refusal.findings):
                raise
            missing = tuple(
                f"version {version}: {finding.key}: " + view.diagnostic(snapshot, unit, version, recipe, finding.reason)
                for finding in refusal.findings
            )
            return _gaps(snapshot, unit, version, recipe, object_sha, missing)
