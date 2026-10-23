"""Native object production and the proof of one function: a module's object linked alone at the addresses the symbol
facts give and compared with the ROM bytes of the function."""

from __future__ import annotations

import fnmatch
import io
import json
import re
import struct
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import abucache
from abucache import compile
from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection

from unbake import adapters, config, effort, libasm, process, recipes, store, symbols, symptoms, tree, versions, view
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
# In every compile key since the canonical-form sharing went (10-22): rows made before it may hold another text's
# object, and this leaves them unreachable.
_EXACT = "\0exact-1"
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
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, include: tuple[Path, ...], align: int = 0
) -> tuple[str, tuple[compile.Step, ...], str, str]:
    seen = view.get(snapshot, unit, version, recipe, lines=False)
    text = seen.text
    # The assembler reads a file only through a directive the compiled text carries. Without one, the include
    # directories (absolute paths of this checkout) are not an input, so they stay out of the key and every
    # worktree of the project shares its objects.
    reads = re.search(r"\.(?:include|incbin)\b", text) is not None
    hilo = file_facts(snapshot.config.project, unit.path).hilo_nops
    steps = adapters.chain_steps(
        snapshot.config, recipe.toolchain, recipe, include if reads else (), _INPUT, align, hilo
    )
    tools = str(
        effort.memo(
            ("tool-identity", recipe.toolchain), lambda: adapters.tool_identity(snapshot.config, recipe.toolchain)
        )
    )
    return text, steps, tools, compile.key(text.encode(), f"{_INPUT}.i", steps, tools + _EXACT)


def _include(snapshot: Snapshot, version: str) -> tuple[Path, ...]:
    project = snapshot.config.project
    macros = config.load_resource("repo.toml")["splat"]["options"]["generated_asm_macros_directory"]
    return (project.root / "include", project.root / macros.format(version=version, name=project.name))


def objects(
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path, align: int = 0
) -> tuple[Path, tuple[NativeResult, ...]]:
    """Preprocess, compile and assemble one module through abucache's cache keyed by the preprocessed bytes, the
    commands and the tools. The same text compiles once, whatever file, version or checkout asks. With `align` (the
    address mod 8 of the unit's data in the ROM) its assembly loses the alignment that start does not meet."""
    with effort.stage("native.objects"), _tools(unit, snapshot, unit, version, recipe):
        cfg = snapshot.config
        work.mkdir(parents=True, exist_ok=True)
        if config.is_assembly(unit.path):
            source, obj = work / "unit.s", work / "unit.o"
            source.write_bytes(snapshot.read(unit.path))
            versioned = replace(recipe, asflags=(*recipe.asflags, *libasm.defsyms(list(cfg.project.versions), version)))
            result = adapters.GnuAssembler(cfg).assemble(source, versioned, _include(snapshot, version), obj)
            return obj, (result,)
        text, steps, tools, key = _compile_inputs(snapshot, unit, version, recipe, _include(snapshot, version), align)
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
            kept = json.dumps(found).encode()
            store.put(cfg, "warnings", digest((key, "warnings")), kept)
        return obj, tuple(results)


def recorded(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe) -> tuple[str, ...] | None:
    """The refused warnings a compile of this unit kept; None when none was kept; never compiles."""
    cfg = snapshot.config
    if config.is_assembly(unit.path):
        return ()
    home = cfg.project.root / "build" / version / Path(unit.path).with_suffix("")
    for kept in sorted(home.glob("*/compile.err")):
        return tuple(adapters.toolchain(cfg, recipe.toolchain).refused(kept.read_text(errors="replace")))
    *_, key = _compile_inputs(snapshot, unit, version, recipe, _include(snapshot, version))
    kept_json = store.get(cfg, "warnings", digest((key, "warnings")))
    return None if kept_json is None else tuple(json.loads(kept_json))


def warnings(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path) -> tuple[str, ...]:
    """The refused-view warnings of the unit's own compile, kept by the text compiled."""
    if config.is_assembly(unit.path):
        return ()
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
    argv += ["-g", "-dg", "-dl", "-dS", "-dR", "-dL"]  # -dL: the loop dump, why a giv was or was not reduced
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
    passes = {
        k: f.read_text(errors="replace") for k in ("sched", "sched2", "loop") if (f := work / f"{stem}.i.{k}").is_file()
    }
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
    facts: Any, defined: dict[str, tuple[str, int, int]], table: Any, mask: int, selected: str = ""
) -> tuple[dict[str, int], dict[str, int]]:
    """(vram of the start of each linker section, its size) from the object's defined symbols the version places: the
    section starts where a symbol's address less its offset in the object says. The selected function anchors its own
    section, so it sits at its address whatever an earlier member of the object measures; the other sections start
    at their earliest placed symbol."""
    base: dict[str, int] = {}
    order = sorted(defined.items(), key=lambda kv: (kv[0] != selected, kv[1][1]))
    for name, (section, value, _) in order:
        fact, out = facts.get(name), _output(table, section)
        if fact is None or out is None or out == ".bss":
            continue
        base.setdefault(out, (fact[0] & mask if out == ".text" else fact[0]) - value)
    return base, {}


def symbol_facts(
    cfg: Config, version: str, overlays: Mapping[str, bytes | None] | None = None
) -> dict[str, symbols.Symbol]:
    """The symbol facts of one version, parsed once per content of the file; a fact file in `overlays` stands in for
    the one on disk."""
    given = (overlays or {}).get(versions.symbol_file(version))
    data = given if given is not None else (cfg.project.root / versions.symbol_file(version)).read_bytes()
    return effort.memo(("native.facts", version, sha256(data)), lambda: symbols.parse(data.decode(errors="replace")))


def version_records(cfg: Config, overlays: Mapping[str, bytes | None] | None = None) -> dict[str, Version]:
    """The version records (ROM, segments, symbols), read once per config and per fact file an overlay replaces."""
    root = cfg.project.root
    given = {p: b for p, b in (overlays or {}).items() if b is not None and p in _fact_files(cfg)}
    return effort.memo(
        ("native.versions", cfg.digest, tuple(sorted((p, sha256(b)) for p, b in given.items()))),
        lambda: versions.read(cfg, lambda p: given[p] if p in given else (root / p).read_bytes()),
    )


def _fact_files(cfg: Config) -> frozenset[str]:
    return frozenset(versions.symbol_file(v) for v in cfg.project.versions)


def candidate(
    config_: Config,
    version: str,
    path: str,
    text: str,
    overrides: Json | None = None,
    overlays: Mapping[str, bytes] | None = None,
    names: Collection[str] | None = None,
) -> tuple[Snapshot, UnitSpec, Recipe, list[str]]:
    """(the snapshot holding `text` as the module `path`, its compile identity, its recipe with NON_MATCHING defined,
    the names it owns in the version), from a parse of the one module: nothing else of the tree is read. With `names`,
    only their NON_MATCHING bodies are compiled."""
    data = (text if names is None else tree.fuzzy_only(text, names)).encode()
    written = {**(overlays or {}), path: data}
    snapshot = Snapshot(
        config_,
        "",
        None,
        version_records(config_, written),
        written,
        digest((config_.digest, sorted((p, sha256(b)) for p, b in written.items()))),
    )
    defs, _, fuzzy = tree.parse(
        config_, path, data, version, lambda p: written[p] if p in written else _disk(config_, p), written
    )
    table = symbol_facts(config_, version, written)
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
        snapshot, unit, recipe, owned = candidate(config_, version, path, text, overrides, overlays, names)
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
            facts = symbol_facts(config_, version, snapshot.overlays)
            spelled = frozenset(_WORD_RE.findall(text))
            for name in wanted:
                proofs[name] = _link(snapshot, unit, version, name, recipe, source, obj, work, facts, spelled)
        return proofs


_WORD_RE = re.compile(r"\b[A-Za-z_]\w*")


@dataclass(frozen=True)
class _Pool:
    """Where the ROM's code reads the candidate's own constant pool: `base`, the vram its object .rodata starts at,
    and `reads`, each (offset in that .rodata, bytes the code takes there; 0 is a string) the function's code names."""

    base: int | None = None
    reads: tuple[tuple[int, int], ...] = ()


_LOAD_SIZE = {
    0x20: 1,
    0x24: 1,
    0x21: 2,
    0x25: 2,
    0x23: 4,
    0x31: 4,
    0x35: 8,
    0x09: 0,
}  # lb lbu lh lhu lw lwc1 ldc1 addiu
_NEAR = 16  # words from the same index the ROM's own %lo of a reference may sit (instructions inserted or dropped)
_LUI, _LO16, _WORD32 = 0x0F, 6, 2  # the lui opcode, R_MIPS_LO16 and R_MIPS_32


def _signed(half: int) -> int:
    return (half & 0xFFFF) - ((half & 0x8000) << 1)


def _formed(code: bytes, at: int) -> int | None:
    """The address the code forms at the %lo instruction at `at`: its low half joined to the lui that feeds its base."""
    word = int.from_bytes(code[at : at + 4], "big")
    for earlier in range(at - 4, max(at - 132, -4), -4):
        before = int.from_bytes(code[earlier : earlier + 4], "big")
        if before >> 26 == _LUI and (before >> 16) & 31 == (word >> 21) & 31:
            return (((before & 0xFFFF) << 16) + _signed(word)) & _WORD
    return None


def _pool_refs(blob: bytes, table: Any) -> tuple[bytes, list[tuple[int, int]], set[int]]:
    """(the object's .text, its (text offset, .rodata offset) per %lo of an address in .rodata, the .rodata offsets
    that hold a relocated word - a jump table's entries); empty for anything but an object with one .rodata."""
    try:
        elf = ELFFile(io.BytesIO(blob))
        names = {i: s.name for i, s in enumerate(elf.iter_sections())}
        pools = [i for i, n in names.items() if _output(table, n) == ".rodata"]
        text = elf.get_section_by_name(".text")
        if len(pools) != 1 or text is None:
            return b"", [], set()
        code, refs, words = text.data(), [], set()
        for section in elf.iter_sections():
            if not isinstance(section, RelocationSection) or section.is_RELA():
                continue
            symbols_ = cast(SymbolTableSection, elf.get_section(section["sh_link"]))
            for r in section.iter_relocations():
                if section["sh_info"] == pools[0] and r["r_info_type"] == _WORD32:
                    words.add(r["r_offset"])
                    continue
                symbol = symbols_.get_symbol(r["r_info_sym"])
                if (
                    names[section["sh_info"]] == ".text"
                    and r["r_info_type"] == _LO16
                    and symbol["st_shndx"] == pools[0]
                ):
                    word = int.from_bytes(code[r["r_offset"] : r["r_offset"] + 4], "big")
                    refs.append((r["r_offset"], symbol["st_value"] + _signed(word)))
        return code, refs, words
    except (ELFError, KeyError, IndexError, ValueError, TypeError, AttributeError, struct.error):
        return b"", [], set()


def _pool(blob: bytes, table: Any, row: tuple[str, int, int], start: int, target: bytes, span: int) -> _Pool:
    """The candidate's .rodata placed where the target's code reads its pool: every %lo of the candidate that the ROM's
    code has at (nearly) the same place forms an address, which less the offset the candidate reads in its .rodata is
    the .rodata start; the most common start is the pool's. The unit then links the way the real build does, with its
    literals and jump tables at the ROM's, so the code reads the same addresses when it is the ROM's code."""
    code, refs, words = _pool_refs(blob, table)
    votes: Counter[int] = Counter()
    reads: set[tuple[int, int]] = set()
    for at, offset in refs:
        if not start <= at < start + span:
            continue
        mine = int.from_bytes(code[at : at + 4], "big")
        for step in sorted(range(-_NEAR, _NEAR + 1), key=abs):
            there = at - start + 4 * step
            same = (
                0 <= there <= len(target) - 4 and int.from_bytes(target[there : there + 4], "big") >> 21 == mine >> 21
            )
            if same and (address := _formed(target, there)) is not None:
                votes[(address - offset) & _WORD] += 1
                break
        run = 0
        while offset in words and offset + 4 * run in words:
            run += 1
        reads.add((offset, 4 * run if run else _LOAD_SIZE.get(mine >> 26, 4)))
    return _Pool(votes.most_common(1)[0][0] if votes else None, tuple(sorted(reads)))


def _superseded(
    pool: _Pool,
    defined: Mapping[str, tuple[str, int, int]],
    facts: Mapping[str, symbols.Symbol],
    table: Any,
    spelled: Collection[str],
    name: str,
) -> bool:
    """Whether the read-only bytes that anchor the object's .rodata are a stand-in the C body of `name` replaces: every
    named read-only symbol of the object comes from an included assembly file (the module's C never spells it), the
    body reads none of their bytes, and the ROM's code reads the body's own pool (a jump table, literals) where those
    bytes lie. The old pool of a function now in C stays behind its `INCLUDE_RODATA` stand-in until it lands (a
    `literals_<name>` row then places the compiled pool there), so anchoring on the stand-in would put the body's pool
    after it and every read of it off by the stand-in's size. C data of the module keeps its anchor."""
    named = sorted(
        (value, fact[0], n)
        for n, (section, value, _) in defined.items()
        if n != name and _output(table, section) == ".rodata" and (fact := facts.get(n)) is not None
    )
    if not named or pool.base is None or not pool.reads or any(n in spelled for _, _, n in named):
        return False
    first = min(offset for offset, _ in pool.reads)
    if first < named[0][0]:
        return False  # the body's pool is not after the stand-in's bytes in the object
    regions = []  # (object offset, vram) runs of the stand-in bytes, each up to the next name or the body's pool
    for k, (value, vram, _) in enumerate(named):
        end = named[k + 1][0] if k + 1 < len(named) else first
        if first <= value or first < end:
            return False  # the body reads a stand-in's bytes: they are the pool's anchor
        regions.append((vram, vram + end - value))
    reads = [(pool.base + offset, pool.base + offset + max(size, 1)) for offset, size in pool.reads]
    return any(lo < stop and start < hi for start, stop in reads for lo, hi in regions)


def _rom_at(cfg: Config, version: str, address: int) -> int | None:
    """The ROM offset a vram address of the version is stored at, by the segment that holds it."""
    for _, rom, end, vram in versions.segments(cfg, version):
        if vram and vram <= address < vram + end - rom:
            return rom + address - vram
    return None


def _pool_gaps(snapshot: Snapshot, version: str, pool: _Pool, built: bytes) -> list[str]:
    """The reads of the function's code whose bytes in the linked pool are not the ROM's at the address it forms."""
    wrong = []
    for offset, size in pool.reads:
        if size == 0:  # an address taken: a string runs to its NUL
            end = built.find(b"\0", offset)
            size = (end if end >= 0 else len(built)) - offset + 1
        rom = _rom_at(snapshot.config, version, (pool.base or 0) + offset)
        if (
            rom is None
            or versions.rom_bytes(snapshot.versions[version], rom, rom + size) != built[offset : offset + size]
        ):
            wrong.append(offset)
    if not wrong:
        return []
    return [
        f"version {version}: {len(wrong)} of {len(pool.reads)} reads of the unit's .rodata differ from the ROM's "
        f"at 0x{pool.base:X}+0x{wrong[0]:X}"
    ]


def _unmet(starts: Mapping[str, int]) -> int:
    """The address mod 8 of the data of the unit where the ROM has it, 0 when it starts on an 8-byte boundary (the
    first of its data sections off one: read-only data, then data)."""
    return next((starts[s] % 8 for s in (".rodata", ".data") if starts.get(s, 0) % 8), 0)


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
    spelled: Collection[str] = (),
    align: int = 0,
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
    starts, _ = _anchors(facts, defined, table, mask, name)
    totals: dict[str, int] = {}
    for title, size in sizes.items():
        if (out := _output(table, title)) is not None:
            totals[out] = totals.get(out, 0) + size
    if ".text" not in starts:
        return _gap(unit, name, version, recipe, source, object_sha, "unproved .text: no anchor")
    end = starts[".text"] + totals.get(".text", 0)
    size = extent(facts, name) or defined[name][2]  # the target's own extent; the object's only when facts have none
    record = snapshot.versions[version]
    target = versions.rom_bytes(record, fact[1], fact[1] + size)
    pool = _pool(blob, table, defined[name], (fact[0] & mask) - starts[".text"], target, max(size, defined[name][2]))
    if pool.base is not None and (
        ".rodata" not in starts or _superseded(pool, defined, facts, table, spelled, name)
    ):  # the unit's own literal pool sits where the ROM's code reads it
        starts[".rodata"] = pool.base
    if not align and (align := _unmet(starts)):
        # the ROM packs the unit's data against its address, where the assembler pads from the section start: the same
        # module assembled without the alignment its start does not meet, then measured again
        again, _ = objects(snapshot, unit, version, recipe, work / f"align{align}", align)
        return _link(
            snapshot, unit, version, name, recipe, source, again, work / f"align{align}", facts, spelled, align
        )
    for out in totals:  # a section no named symbol places follows the code (the per-object shape)
        starts.setdefault(out, (end + 15) & ~15)
    placed = tuple(
        Placement(version, out, 0, size_, starts[out], size_ if out == ".bss" else 0)
        for out, size_ in sorted(totals.items())
        if size_
    )
    section = _output(table, defined[name][0]) or ".text"
    symbols_data = versions.symbols_ld(cfg, version, snapshot.overlays)
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
            code_identity(_link, _pool, _pool_gaps, _superseded),
            align,
            sorted(n for n in defined if n in spelled),
            asdict(pool),
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
        if pool.base is not None and starts[".rodata"] == pool.base:
            missing += _pool_gaps(snapshot, version, pool, built.get(".rodata", b""))
        exact = not missing
        store.put(cfg, "bytes", sha256(have), have)
        reads = _literal_reads(snapshot, version, starts.get(".rodata"), built.get(".rodata", b""))
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


def extent(facts: Mapping[str, symbols.Symbol], name: str) -> int | None:
    """The byte size of `name` in the version's facts: its own `size:`, else the distance in ROM to the next named
    symbol; None when neither is known. It never depends on what a candidate built."""
    fact = facts.get(name)
    if fact is None or fact[1] is None:
        return None
    if fact[2] is not None:
        return fact[2]
    after = [f[1] - fact[1] for f in facts.values() if f[1] is not None and f[1] > fact[1] and f[3] != "label"]
    return min(after) if after else None


def _literal_reads(
    snapshot: Snapshot, version: str, start: int | None, built: bytes
) -> tuple[Callable[[int, int], bytes], Callable[[int, int], bytes]]:
    """Readers of `size` bytes at a vram in the built rodata and in the target's, empty where there is none. The
    target's is the ROM's at any address a segment holds, whatever the built rodata's own extent."""

    def ours(address: int, size: int) -> bytes:
        return built[address - start : address - start + size] if start is not None and address >= start else b""

    def theirs(address: int, size: int) -> bytes:
        rom = _rom_at(snapshot.config, version, address)
        return b"" if rom is None else versions.rom_bytes(snapshot.versions[version], rom, rom + size)

    return ours, theirs
