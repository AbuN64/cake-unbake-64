"""Native object production and the proof of one function: a module's object linked alone at the addresses the symbol
facts give and compared with the ROM bytes of the function."""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import abucache
from abucache import compile

from unbake import adapters, config, effort, process, recipes, store, symbols, symptoms, tree, versions, view
from unbake.contracts import (
    Config,
    Finding,
    Json,
    NativeResult,
    Origin,
    Placement,
    Proof,
    Recipe,
    Refusal,
    Snapshot,
    UnitSpec,
    Version,
    code_identity,
    digest,
    file_facts,
    proof_from,
    sha256,
)

_WORD = 0xFFFFFFFF
_INPUT = "unit"  # the one name every compile input has: a rename or a move never reaches a key
_FUZZY = "-DNON_MATCHING"  # a measured module compiles its NON_MATCHING bodies, which the real build never does


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


def stamp(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe | None = None, *, code: bool = True) -> str:
    """Everything a compile reads, named by content alone: the preprocessed text, the recipe and the tools. No path,
    member list or layout row is in the key, so a rename or a move between files keeps it."""
    cfg = snapshot.config
    recipe = recipe or recipes.resolve(cfg, unit, {})
    text = view.get(snapshot, unit, version, recipe, lines=False).text
    tools = effort.memo(("tools", recipe.toolchain), lambda: adapters.tool_identity(cfg, recipe.toolchain))
    return digest(
        (
            sha256(text.encode()),
            recipe.digest,
            tools,
            abucache.__version__,
            code_identity(measure, objects, warnings, recorded, dumps) if code else "",
        )
    )


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
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, include: tuple[Path, ...]
) -> tuple[str, tuple[compile.Step, ...], str, str]:
    seen = view.get(snapshot, unit, version, recipe, lines=False)
    text = seen.text
    # The assembler reads a file only through a directive the compiled text carries. Without one, the include
    # directories (absolute paths of this checkout) are not an input, so they stay out of the key and every
    # worktree of the project shares its objects.
    reads = re.search(r"\.(?:include|incbin)\b", text) is not None
    steps = adapters.chain_steps(snapshot.config, recipe.toolchain, recipe, include if reads else (), _INPUT)
    tools = str(
        effort.memo(
            ("tool-identity", recipe.toolchain), lambda: adapters.tool_identity(snapshot.config, recipe.toolchain)
        )
    )
    return text, steps, tools, compile.key(text.encode(), f"{_INPUT}.i", steps, tools)


def _include(snapshot: Snapshot, version: str) -> tuple[Path, ...]:
    project = snapshot.config.project
    macros = config.load_resource("repo.toml")["splat"]["options"]["generated_asm_macros_directory"]
    return (project.root / "include", project.root / macros.format(version=version, name=project.name))


def objects(
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path
) -> tuple[Path, tuple[NativeResult, ...]]:
    """Preprocess, compile and assemble one module through abucache's cache keyed by the preprocessed bytes, the
    commands and the tools. The same text compiles once, whatever file, version or checkout asks."""
    with effort.stage("native.objects"), _tools(unit, snapshot, unit, version, recipe):
        cfg = snapshot.config
        work.mkdir(parents=True, exist_ok=True)
        text, steps, tools, key = _compile_inputs(snapshot, unit, version, recipe, _include(snapshot, version))
        source, obj = (
            work / f"{_INPUT}.i",
            work / f"{Path(unit.path).stem}.o",
        )  # the input is named by nothing of the file
        source.write_text(text, encoding="utf-8")

        def run(phase: str, argv: Sequence[str], cwd: Path) -> NativeResult:
            return process.run(phase, argv, cwd, tmp=process.scratch(cfg.project.root))

        def build() -> tuple[bytes, list[NativeResult]]:
            value, results = compile.build(
                store.content(cfg),
                run,
                steps,
                source,
                tools,
                work,
                after=adapters.toolchain(cfg, recipe.toolchain).check,
            )
            return value, results

        value, results = _replayed(cfg, key, build)
        obj.write_bytes(value)
        if results:  # the compile that ran keeps its refused warnings, so nothing compiles again to count them
            found = adapters.toolchain(cfg, recipe.toolchain).refused(results[0].stderr.decode(errors="replace"))
            store.put(cfg, "warnings", digest((key, "warnings")), json.dumps(found).encode())
        return obj, tuple(results)


def recorded(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe) -> tuple[str, ...] | None:
    """The refused warnings a compile of this unit kept; None when none was kept; never compiles."""
    cfg = snapshot.config
    home = cfg.project.root / "build" / version / Path(unit.path).with_suffix("")
    for kept in sorted(home.glob("*/compile.err")):
        return tuple(adapters.toolchain(cfg, recipe.toolchain).refused(kept.read_text(errors="replace")))
    *_, key = _compile_inputs(snapshot, unit, version, recipe, _include(snapshot, version))
    kept_json = store.get(cfg, "warnings", digest((key, "warnings")))
    return None if kept_json is None else tuple(json.loads(kept_json))


def warnings(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path) -> tuple[str, ...]:
    """The refused-view warnings of the unit's own compile, kept by the text compiled."""
    with effort.stage("native.warnings"), _tools(unit, snapshot, unit, version, recipe):
        cfg = snapshot.config
        work.mkdir(parents=True, exist_ok=True)
        stem = Path(unit.path).stem
        text, steps, _, key = _compile_inputs(snapshot, unit, version, recipe, _include(snapshot, version))

        def _produce() -> bytes:
            (work / f"{stem}.i").write_text(text, encoding="utf-8")
            argv = [w.replace("{in}", f"{stem}.i").replace("{out}", steps[0].out) for w in steps[0].argv]
            result = process.run("compile", argv, work, tmp=process.scratch(cfg.project.root))
            found = adapters.toolchain(cfg, recipe.toolchain).refused(result.stderr.decode(errors="replace"))
            return json.dumps(found).encode()

        return tuple(json.loads(store.cached(cfg, "warnings", digest((key, "warnings")), _produce)))


def dumps(
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path
) -> tuple[str, str, str, dict[str, str]]:
    """The allocation and scheduler dumps and debug assembly of one compile made outside the cache (a cached object
    has none); the scheduler dumps are empty when the toolchain writes none."""
    stem = Path(unit.path).stem
    text, steps, *_ = _compile_inputs(snapshot, unit, version, recipe, _include(snapshot, version))
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{stem}.i").write_text(text, encoding="utf-8")
    argv = [a.replace("{in}", f"{stem}.i").replace("{out}", f"{stem}.s") for a in steps[0].argv]
    argv += ["-g", "-dg", "-dl", "-dS", "-dR"]
    result = process.run("native.dumps", argv, work, tmp=process.scratch(snapshot.config.project.root))
    found = [work / f"{stem}.i.{kind}" for kind in ("greg", "lreg")]
    if result.exit != 0:  # the candidate does not compile: its errors, never a toolchain verdict
        lines = [ln for ln in result.stderr.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        raise Refusal(
            Finding(
                "score.compile",
                reason="; ".join(lines) or f"the compiler exited {result.exit}",
                unit=unit.path,
                versions=(version,),
                action="fix the compile error shown",
            )
        )
    if not all(f.is_file() for f in found):
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


def _finding(row: dict[str, Any]) -> Finding:
    origin = Origin(**row["origin"]) if row["origin"] else None
    return Finding(**{**row, "origin": origin, "versions": tuple(row["versions"]), "missing": tuple(row["missing"])})


# ---------------------------------------------------------------- the proof of one function
def _gap(unit: UnitSpec, name: str, version: str, recipe: Recipe, source: str, obj: str, why: str) -> Proof:
    missing = (f"version {version}: {why}",)
    return Proof(
        unit.path,
        name,
        version,
        recipe.digest,
        source,
        obj,
        "",
        "",
        False,
        missing,
        0.0,
        symptoms.from_missing(missing),
    )


def _output(table: Any, section: str) -> str | None:
    """The linker section (text, rodata, data, bss) an object section belongs to."""
    return next((f".{k}" for k, row in table.items() if any(fnmatch.fnmatch(section, i) for i in row["inputs"])), None)


def _anchors(
    facts: Any, defined: dict[str, tuple[str, int, int]], table: Any, mask: int
) -> tuple[dict[str, int], dict[str, int]]:
    """(vram of the start of each linker section, its size) from the object's defined symbols the version places: the
    section starts where a symbol's address less its offset in the object says."""
    base: dict[str, int] = {}
    for name, (section, value, _) in sorted(defined.items(), key=lambda kv: kv[1][1]):
        fact, out = facts.get(name), _output(table, section)
        if fact is None or out is None or out == ".bss":
            continue
        base.setdefault(out, (fact[0] & mask if out == ".text" else fact[0]) - value)
    return base, {}


def symbol_facts(cfg: Config, version: str) -> dict[str, symbols.Symbol]:
    """The symbol facts of one version, parsed once per content of the file."""
    data = (cfg.project.root / versions.symbol_file(version)).read_bytes()
    return effort.memo(("native.facts", version, sha256(data)), lambda: symbols.parse(data.decode(errors="replace")))


def version_records(cfg: Config) -> dict[str, Version]:
    """The version records (ROM, segments, symbols), read once per config."""
    root = cfg.project.root
    return effort.memo(("native.versions", cfg.digest), lambda: versions.read(cfg, lambda p: (root / p).read_bytes()))


def candidate(
    config_: Config,
    version: str,
    path: str,
    text: str,
    overrides: Json | None = None,
    overlays: Mapping[str, bytes] | None = None,
) -> tuple[Snapshot, UnitSpec, Recipe, list[str]]:
    """(the snapshot holding `text` as the module `path`, its compile identity, its recipe with NON_MATCHING defined,
    the names it owns in the version), from a parse of the one module: nothing else of the tree is read."""
    data = text.encode()
    written = {**(overlays or {}), path: data}
    snapshot = Snapshot(
        config_,
        "",
        None,
        version_records(config_),
        written,
        digest((config_.digest, sorted((p, sha256(b)) for p, b in written.items()))),
    )
    defs, _, fuzzy = tree.parse(
        config_, path, data, version, lambda p: written[p] if p in written else _disk(config_, p)
    )
    table = symbol_facts(config_, version)
    owned = [n for n in dict.fromkeys([*defs, *fuzzy]) if n in table]
    row = file_facts(config_.project, path)
    options = {k: list(v) for k, v in (("add", row.cflags), ("omit", row.omit)) if v}
    unit = UnitSpec(path, "c", tuple(sorted(owned)), row.toolchain or config_.project.toolchain, options)
    recipe = recipes.resolve(config_, unit, overrides or {})
    recipe = replace(recipe, cppflags=(*recipe.cppflags, _FUZZY), digest=digest((recipe.digest, _FUZZY)))
    return snapshot, unit, recipe, owned


def _disk(cfg: Config, path: str) -> bytes | None:
    try:
        return (cfg.project.root / path).read_bytes()
    except OSError:
        return None


def measure(
    config_: Config,
    version: str,
    name: str,
    path: str,
    text: str,
    overrides: Json | None = None,
    overlays: Mapping[str, bytes] | None = None,
) -> Proof:
    """The proof of `name` in one version: the module `path` as `text` compiled, its object linked alone with each
    section at the address the symbol facts give, and the function's bytes compared with the ROM's."""
    return measure_names(config_, version, [name], path, text, overrides, overlays)[name]


def measure_names(
    config_: Config,
    version: str,
    names: Sequence[str],
    path: str,
    text: str,
    overrides: Json | None = None,
    overlays: Mapping[str, bytes] | None = None,
) -> dict[str, Proof]:
    """`measure` of each name: the module is parsed, preprocessed and compiled once, whatever number of its functions
    are asked for, and only the link and compare are made per function."""
    with effort.stage("native.measure"):
        source = sha256(text.encode())
        snapshot, unit, recipe, owned = candidate(config_, version, path, text, overrides, overlays)
        proofs: dict[str, Proof] = {}
        wanted = [n for n in dict.fromkeys(names) if n in owned]
        for name in names:
            if name not in owned:
                proofs[name] = _gap(unit, name, version, recipe, source, "", f"{name} is not owned by {path}")
        if not wanted:
            return proofs
        with store.work(config_) as work:
            try:
                obj, _ = objects(snapshot, unit, version, recipe, work)
            except Refusal as refusal:
                lines = tuple(
                    f"version {version}: compile.error: " + view.diagnostic(snapshot, unit, version, recipe, f.reason)
                    for f in refusal.findings
                )
                for name in wanted:
                    proofs[name] = replace(
                        _gap(unit, name, version, recipe, source, "", "compile.error"),
                        missing=lines,
                        symptoms=symptoms.from_missing(lines),
                    )
                return proofs
            facts = symbol_facts(config_, version)
            for name in wanted:
                proofs[name] = _link(snapshot, unit, version, name, recipe, source, obj, work, facts)
        return proofs


def _link(
    snapshot: Snapshot,
    unit: UnitSpec,
    version: str,
    name: str,
    recipe: Recipe,
    source: str,
    obj: Path,
    work: Path,
    facts: dict[str, symbols.Symbol],
) -> Proof:
    cfg = snapshot.config
    blob = obj.read_bytes()
    object_sha = sha256(blob)
    fact = facts.get(name)
    defined, sizes = versions.defined(obj)
    if fact is None or fact[1] is None or name not in defined:
        return _gap(unit, name, version, recipe, source, object_sha, f"unproved {name}: the object does not define it")
    table = config.sections()
    mask = config.load_resource("rom.toml").get("text_mask", _WORD)
    starts, _ = _anchors(facts, defined, table, mask)
    totals: dict[str, int] = {}
    for title, size in sizes.items():
        if (out := _output(table, title)) is not None:
            totals[out] = totals.get(out, 0) + size
    if ".text" not in starts:
        return _gap(unit, name, version, recipe, source, object_sha, "unproved .text: no anchor")
    end = starts[".text"] + totals.get(".text", 0)
    for out in totals:  # a section no named symbol places follows the code (the per-object shape)
        starts.setdefault(out, (end + 15) & ~15)
    placed = tuple(
        Placement(version, out, 0, size, starts[out], size if out == ".bss" else 0)
        for out, size in sorted(totals.items())
        if size
    )
    size = fact[2] if fact[2] is not None else defined[name][2]
    rom_view = versions.rom_bytes
    record = snapshot.versions[version]
    target = rom_view(record, fact[1], fact[1] + size)
    section = _output(table, defined[name][0]) or ".text"
    symbols_data = versions.symbols_ld(cfg, version)
    row = config.toolchain(recipe.toolchain)
    key = digest(
        (
            object_sha,
            [asdict(p) for p in placed],
            name,
            size,
            sha256(symbols_data),
            sha256(target),
            abucache.__version__,
            adapters.tool_identity(cfg, recipe.toolchain),
            code_identity(_link),
        )
    )

    def produce() -> bytes:
        work.mkdir(parents=True, exist_ok=True)
        ld_symbols = work / "symbols.ld"
        ld_symbols.write_bytes(symbols_data)
        result = adapters.linker(cfg).link(obj, placed, ld_symbols, work, not row["preserves_padding"])
        built = {s: p.read_bytes() for s, p in result.outputs.items() if s.startswith(".")}
        data = built.get(section, b"")
        offset = (fact[0] & mask if section == ".text" else fact[0]) - starts[section]
        have = data[offset : offset + size]
        missing: list[str] = []
        if len(have) != size:
            missing.append(f"version {version}: {section} size {len(have)} != {size}")
        differences = [i for i, (x, y) in enumerate(zip(have, target, strict=False)) if x != y]
        if differences:
            missing.append(f"version {version}: {len(differences)} bytes differ at +0x{differences[0]:X} {section}")
        exact = not missing
        store.put(cfg, "bytes", sha256(have), have)
        reads = _literal_reads(snapshot, version, starts.get(".rodata"), built.get(".rodata", b""), facts)
        shown = symptoms.literals(have, target, *reads) if section == ".text" and not exact else have
        score = 1.0 if exact else min(symptoms.score(shown, target), 1.0 - 2**-53)
        extra = (
            {}
            if exact
            else dict(symptoms.merge([symptoms.measure(have, target, section), symptoms.from_missing(missing)]))
        )
        if not exact:
            extra["score"] = score
        proof = Proof(
            unit.path,
            name,
            version,
            recipe.digest,
            source,
            object_sha,
            sha256(have),
            sha256(target),
            exact,
            tuple(missing),
            score,
            extra,
        )
        return json.dumps(asdict(proof), sort_keys=True).encode()

    return proof_from(json.loads(store.cached(cfg, "proof", key, produce)))


def _literal_reads(
    snapshot: Snapshot, version: str, start: int | None, built: bytes, facts: Any
) -> tuple[Callable[[int, int], bytes], Callable[[int, int], bytes]]:
    """Readers of `size` bytes at a vram in the built rodata and in the target's, empty outside them. The target's
    rodata is read at the ROM offset of the nearest named rodata symbol at or below the address."""
    rows = sorted((f[0], f[1]) for f in facts.values() if f[1] is not None)

    def ours(address: int, size: int) -> bytes:
        return built[address - start : address - start + size] if start is not None and address >= start else b""

    def theirs(address: int, size: int) -> bytes:
        if start is None or not start <= address < start + len(built):
            return b""
        below = [r for r in rows if r[0] <= address]
        if not below:
            return b""
        vram, rom = below[-1]
        at = rom + address - vram
        return versions.rom_bytes(snapshot.versions[version], at, at + size)

    return ours, theirs
