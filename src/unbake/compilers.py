"""`unbake compilers`: which compiler built each unit of a project, as three steps over the ROM-equal build.

fingerprint  reads the objects the build made (no compiler runs) and classes every function by the shapes its code
             has: how the epilogue is filled, how global addresses are formed, whether single floats use odd FPRs,
             where the callee saves sit. The catalog (toolchains.toml `fingerprint`) says which shapes a compiler
             can emit; this module never names one.
census       recompiles the C functions of each unit with every candidate (each catalog toolchain of the project's
             family with its default flags and its `probe_flags` sets, and the unit's own recipe) and counts the
             instruction lines that differ from the ROM-equal object. It stops after a CPU budget and resumes from
             what it stored, so one call stays under the CPU cap.
propose      joins both and gives the smallest set of directory rows (plus file exceptions) that makes every
             judged C function exact, the rows that equal their directory (dead), the units whose shapes their row
             cannot emit, and the NON_MATCHING drafts a different row makes exact. Advisory: rows land only when
             `make check` passes.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

from unbake import adapters, effort, pool, process, recipes, store
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Recipe, Refusal, UnitSpec, digest, sha256

Row = dict[str, Any]  # a row built here, then written

STATE = Path(".unbake") / "compilers"
FINGERPRINT = STATE / "fingerprint.jsonl"
CENSUS = STATE / "census.jsonl"
PROPOSAL = STATE / "propose.json"
TABLE = STATE / "units.tsv"
CPU_BUDGET = 240  # CPU seconds one census call may spend: a command stays under the owner's five-minute cap
BATCH = 8  # units the census gives the pool at once
_DOUBLE = re.compile(
    r"^(dsll32|dsrl32|dsra32|dsll|dsrl|dsra|ld|sd|daddu|dsubu|daddiu|dmult|dmultu|ddiv|ddivu|dsllv|dsrlv|dsrav)$"
)
_MEMORY = re.compile(r"^(lw|sw|lh|lhu|sh|lb|lbu|sb|lwc1|swc1|ldc1|sdc1|ld|sd|lwl|lwr|swl|swr)$")
_NUMERIC = "no-aliases,gpr-names=numeric"
_SYMBOL = re.compile(r"^[0-9a-f]+ <([\w.$]+)>:")
_INSTRUCTION = re.compile(r"^\s+[0-9a-f]+:\s+[0-9a-f]{8}\s+(\S+)\s*(.*)$")
_RELOC = re.compile(r"^\s+[0-9a-f]+:\s+(R_MIPS_\w+)\s+(\S+)")
_FILE = re.compile(r"^(.+):\s+file format ")
_TARGET = re.compile(r"[0-9a-f]+ <([^>+]*)(\+0x[0-9a-f]+)?>")
_DEFINITION = re.compile(r"^[A-Za-z_][\w \t\*]*?\b(\w+)\s*\([^;{}]*\)\s*\{", re.MULTILINE)
_NOP = "sll $0,$0,0"
Ins = tuple[str, str, str | None, str]  # (opcode, operands, relocation, text with targets named alike)


# ---- reading objects
def _cpu() -> float:
    """CPU seconds this process and the children it has waited for have used."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def _tmp(config: Config) -> Path:
    return process.scratch(config.project.root)


def _objdump(config: Config, files: Sequence[Path], flags: str = _NUMERIC) -> list[str]:
    """The disassembly of each object, in order: one objdump call for the whole batch."""
    out = _tmp(config) / f"objdump-{os.getpid()}.txt"
    process.run(
        "objdump",
        [process.tool(config, "mips_objdump"), "-drz", "-M", flags, *map(str, files)],
        config.project.root,
        tmp=_tmp(config),
        stdout_path=out,
    )
    text = out.read_text(errors="replace")
    out.unlink(missing_ok=True)
    parts: list[list[str]] = []
    for line in text.splitlines():
        if _FILE.match(line):
            parts.append([])
        elif parts:
            parts[-1].append(line)
    return ["\n".join(p) for p in parts] + [""] * (len(files) - len(parts))


def _normal(op: str, args: str, current: str) -> str:
    """One instruction as text that two objects can share: branch targets inside the function are named by offset,
    call targets by nothing."""
    text = f"{op} {args}".strip()
    text = _TARGET.sub(lambda m: "T" if m.group(1) != current else "L" + (m.group(2) or "+0"), text)
    text = re.sub(r"\s+#.*", "", text)
    text = re.sub(r"^(jal)\s+\S+$", r"\1 T", text)
    return re.sub(r"\s+", " ", text).strip()


def parse(listing: str) -> dict[str, list[Ins]]:
    """The functions of one object's disassembly (`objdump -drz -M no-aliases,gpr-names=numeric`) by symbol."""
    functions: dict[str, list[Ins]] = {}
    current = ""
    for line in listing.splitlines():
        head = _SYMBOL.match(line)
        if head:
            current = head.group(1)
            functions[current] = []
            continue
        if not current:
            continue
        found = _INSTRUCTION.match(line)
        if found:
            op, args = found.group(1), found.group(2).split("#")[0].strip()
            functions[current].append((op, args, None, _normal(op, found.group(2), current)))
            continue
        reloc = _RELOC.match(line)
        if reloc and functions[current]:
            op, args, _, text = functions[current][-1]
            functions[current][-1] = (op, args, reloc.group(1), text)
    return functions


def _regs(args: str) -> list[str]:
    return [a.strip() for a in args.split(",")]


def features(ins: Sequence[Ins]) -> Row:
    """Counts of the shapes in one function: epilogue (`epi_filled`: the stack adjust sits in the delay slot of the
    return; `epi_unfilled`: it comes before a `jr; nop`), address formation (`at_hilo`: a `lui` of the assembler
    temporary is the base of a global access; `split_hilo`: another register is; `dst_hilo`: the loaded register is),
    `fp_odd_s` (a single-precision operation names an odd FPR: only a 64-bit FPU mode allows it), `dw_ops`
    (64-bit integer operations), `pro_sched` (instructions before the stack adjust), `save_top` / `save_low` (the
    callee saves end at the top of the frame, or below the locals)."""
    f: Row = {
        "n": len(ins),
        "epi_filled": 0,
        "epi_unfilled": 0,
        "leaf_nop": 0,
        "at_hilo": 0,
        "split_hilo": 0,
        "dst_hilo": 0,
        "fp_odd_s": 0,
        "dw_ops": 0,
        "pro_sched": 0,
        "save_top": 0,
        "save_low": 0,
    }
    for i, (op, args, rel, _) in enumerate(ins):
        if op == "jr" and args == "$31" and i + 1 < len(ins):
            slot = ins[i + 1]
            if slot[0] == "addiu" and _regs(slot[1])[:2] == ["$29", "$29"]:
                f["epi_filled"] += 1
            elif slot[0] == "sll" and slot[1] in ("$0,$0,0", "$0,$0,0x0"):
                before = ins[i - 1] if i else None
                if before and before[0] == "addiu" and _regs(before[1])[:2] == ["$29", "$29"]:
                    f["epi_unfilled"] += 1
                else:
                    f["leaf_nop"] += 1
        if op == "lui" and rel == "R_MIPS_HI16":
            reg = _regs(args)[0]
            for op2, args2, rel2, _ in ins[i + 1 : i + 12]:
                if rel2 == "R_MIPS_LO16" and _MEMORY.match(op2) and args2.endswith(f"({reg})"):
                    key = "at_hilo" if reg == "$1" else "dst_hilo" if _regs(args2)[0] == reg else "split_hilo"
                    f[key] += 1
                    break
        if op.endswith(".s") and any(int(r) % 2 for r in re.findall(r"\$f(\d+)", args)):
            f["fp_odd_s"] += 1
        if _DOUBLE.match(op):
            f["dw_ops"] += 1
    frame = 0
    for i, (op, args, _, _) in enumerate(ins[:12]):
        if op == "addiu" and _regs(args)[:2] == ["$29", "$29"] and args.split(",")[2].startswith("-"):
            f["pro_sched"] = i
            frame = -int(args.split(",")[2], 0)
            break
    if frame:
        top: int | None = None
        for op, args, _, _ in ins[:40]:
            m = re.match(r"^\$(f?)(\d+),(-?\w+)\(\$29\)$", args)
            if op in ("sw", "sd", "swc1", "sdc1") and m:
                fpr, number, off = m.group(1), int(m.group(2)), int(m.group(3), 0)
                if (fpr and number >= 20) or (not fpr and (16 <= number <= 23 or number in (30, 31))):
                    top = off if top is None else max(top, off)
        if top is not None:
            f["save_top" if top >= frame - 8 else "save_low"] += 1
    return f


def shape(record: Mapping[str, Any]) -> str:
    """A function's class as letters: epilogue F filled, U unfilled, X both, - none; address M assembler-temporary base,
    S split, X both, - none; and `+f64` when single floats use odd FPRs."""
    filled, unfilled = record["epi_filled"], record["epi_unfilled"]
    epilogue = "X" if filled and unfilled else "F" if filled else "U" if unfilled else "-"
    macro_, split = record["at_hilo"], record["split_hilo"]
    address = "X" if macro_ and split else "M" if macro_ else "S" if split else "-"
    return epilogue + address + ("+f64" if record["fp_odd_s"] else "")


def shapes(record: Mapping[str, Any]) -> dict[str, set[str]]:
    """What one function shows, in the vocabulary of a catalog `fingerprint`."""
    found: dict[str, set[str]] = {"epilogue": set(), "addresses": set(), "save_area": set(), "fp": set()}
    if record["epi_filled"]:
        found["epilogue"].add("filled")
    if record["epi_unfilled"]:
        found["epilogue"].add("unfilled")
    if record["at_hilo"]:
        found["addresses"].add("macro")
    if record["split_hilo"]:
        found["addresses"].add("split")
    if record["save_top"]:
        found["save_area"].add("top")
    if record["save_low"]:
        found["save_area"].add("below-locals")
    if record["fp_odd_s"]:
        found["fp"].add("fp64")
    return found


def allows(fingerprint: Mapping[str, Any], seen: Mapping[str, Iterable[str]]) -> bool:
    """Whether a compiler of this catalog `fingerprint` can emit every shape in `seen`."""
    for kind in ("epilogue", "save_area"):
        if any(s != fingerprint[kind] for s in seen.get(kind, ())):
            return False
    for s in seen.get("addresses", ()):
        if fingerprint["addresses"] not in (s, "either"):
            return False
    return all(s in fingerprint["fp"] for s in seen.get("fp", ()))


def _units(config: Config, version: str, paths: Sequence[str]) -> list[str]:
    """The compiled sources of a version's build (the `.o` files that come from C), sorted."""
    home = config.project.root / "build" / version
    found = []
    for obj in sorted((home / "src").rglob("*.o")) if (home / "src").is_dir() else ():
        unit = obj.relative_to(home).with_suffix("").as_posix()
        if (
            configuration.is_source(unit)
            and not configuration.is_assembly(unit)
            and (config.project.root / unit).is_file()
        ):
            found.append(unit)
    return [u for u in found if not paths or any(u.startswith(p) for p in paths)]


def _read(config: Config, path: Path) -> list[Row]:
    full = config.project.root / path
    return [json.loads(line) for line in full.read_text().splitlines() if line.strip()] if full.is_file() else []


def _write(config: Config, path: Path, rows: Iterable[Row]) -> None:
    store.write(config.project.root / path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())


def _fingerprint_job(item: tuple[Config, str, list[str]]) -> list[Row]:
    config, version, units = item
    home = config.project.root / "build" / version
    listings = _objdump(config, [home / f"{u}.o" for u in units])
    rows = []
    for unit, listing in zip(units, listings, strict=True):
        for name, ins in parse(listing).items():
            if ins:
                rows.append({"k": "fn", "unit": unit, "fn": name, **features(ins)})
    return rows


def fingerprint(config: Config, params: Json) -> Row:
    """Write `.unbake/compilers/fingerprint.jsonl`: a row per function with its features and class, a row per unit
    and a row per directory with class counts. Needs the built objects of the version, no compiler."""
    with effort.stage("compilers.fingerprint"):
        version = params.get("version") or config.project.names_from
        units = _units(config, version, params.get("paths") or ())
        if not units:
            raise Refusal(
                Finding(
                    "compilers.input",
                    f"no built objects under build/{version}/src to fingerprint",
                    action="run make for the version first",
                )
            )
        batches = [(config, version, units[i : i + 200]) for i in range(0, len(units), 200)]
        rows = [r for part in pool.map(config, "compilers.fingerprint", _fingerprint_job, batches) for r in part]
        by_unit: dict[str, list[Row]] = defaultdict(list)
        for r in rows:
            by_unit[r["unit"]].append(r)
        out: list[Row] = [{"k": "meta", "version": version, "objects": len(units)}]
        out += [{**r, "class": shape(r)} for r in rows]
        dirs: dict[str, Counter[str]] = defaultdict(Counter)
        for unit, fns in sorted(by_unit.items()):
            classes = Counter(shape(r) for r in fns)
            seen: dict[str, set[str]] = defaultdict(set)
            for r in fns:
                for kind, values in shapes(r).items():
                    seen[kind] |= values
            out.append(
                {
                    "k": "unit",
                    "unit": unit,
                    "functions": len(fns),
                    "classes": dict(sorted(classes.items())),
                    "shapes": {k: sorted(v) for k, v in sorted(seen.items())},
                }
            )
            for parent in Path(unit).parents:
                if parent.as_posix() != ".":
                    dirs[parent.as_posix() + "/"].update(Counter(c[:2] for c in classes.elements()))
        out += [{"k": "dir", "dir": d, "classes": dict(sorted(c.items()))} for d, c in sorted(dirs.items())]
        _write(config, FINGERPRINT, out)
        total = Counter(shape(r) for r in rows)
        return {
            "action": "fingerprint",
            "version": version,
            "units": len(by_unit),
            "functions": len(rows),
            "classes": dict(sorted(total.items())),
            "path": FINGERPRINT.as_posix(),
        }


# ---- the census
def _c_functions(text: str) -> set[str]:
    """Names of the functions a preprocessed text defines in C (a top-level `type name(args) {`)."""
    return set(_DEFINITION.findall(text))


def _without_stand_ins(text: str) -> str:
    return "".join(line for line in text.splitlines(keepends=True) if not line.startswith("__asm__("))


def _flags(recipe: Recipe, base: Recipe) -> tuple[list[str], list[str]]:
    """(flags the recipe adds to the base, flags of the base it lacks)."""
    return [f for f in recipe.cflags if f not in base.cflags], [f for f in base.cflags if f not in recipe.cflags]


def _label(toolchain: str, add: Sequence[str], omit: Sequence[str]) -> str:
    return " ".join([toolchain, *add, *(f"(omit {o})" for o in omit)])


def _plain(config: Config, toolchain: str, path: str, add: Sequence[str] = ()) -> Recipe:
    """The recipe of a toolchain with its default flags and the project's, no [file] row, plus `add`."""
    return recipes.resolve(config, UnitSpec(path, "c", (), toolchain, {}), {"add": list(add)} if add else {})


def _toolchains(config: Config, named: str | None) -> list[str]:
    """The candidate toolchains: those named, else every installed catalog toolchain of the family the project's
    default toolchain has."""
    rows = configuration.toolchains()
    if named:
        wanted = [t for t in named.split(",") if t]
        unknown = [t for t in wanted if t not in rows]
        if unknown:
            raise Refusal(Finding("compilers.input", f"toolchains not in toolchains.toml: {', '.join(unknown)}"))
        return wanted
    family = rows[config.project.toolchain]["family"]
    return sorted(
        t
        for t, row in rows.items()
        if row["family"] == family and (config.host.toolchain_root / t / row["cc"]).is_file() and "fingerprint" in row
    )


def _candidates(config: Config, toolchains: Sequence[str], path: str) -> dict[str, Recipe]:
    """Label -> recipe for one unit: every toolchain with its default flags and each of its probe sets, and the unit's
    own recipe. A recipe that two sources give is listed once, under the plainest label."""
    found: dict[str, Recipe] = {}
    for toolchain in toolchains:
        base = _plain(config, toolchain, path)
        found[_label(toolchain, (), ())] = base
        for probe in configuration.toolchain(toolchain).get("probe_flags", ()):
            recipe = _plain(config, toolchain, path, probe)
            add, omit = _flags(recipe, base)
            found[_label(toolchain, add, omit)] = recipe
    own = recipes.of(config, path)
    base = _plain(config, own.toolchain, path)
    add, omit = _flags(own, base)
    found.setdefault(_label(own.toolchain, add, omit), own)
    seen: dict[str, str] = {}
    for label, recipe in sorted(found.items(), key=lambda kv: (len(kv[0]), kv[0])):
        seen.setdefault(recipe.digest, label)
    return {label: recipe for label, recipe in found.items() if seen[recipe.digest] == label}


def _build(config: Config, recipe: Recipe, source: Path, work: Path, stem: str) -> Path | str:
    """The object of a preprocessed text under a recipe, or the first line of why it did not build."""
    chain = adapters.toolchain(config, recipe.toolchain)
    first = work / f"{stem}.{'s' if chain.row['emits_asm'] else 'o'}"
    try:
        chain.compile(source, recipe, first)
        if not chain.row["emits_asm"]:
            return first
        obj = work / f"{stem}.o"
        chain.assemble(first, recipe, [], obj)
        return obj
    except Refusal as refusal:
        return (refusal.findings[0].reason.strip().splitlines() or ["failed"])[-1][:160]


def _lines(want: Sequence[str], have: Sequence[str]) -> int:
    """Instruction lines that differ (the lines of an unified diff with no context), trailing alignment nops aside."""
    a, b = list(want), list(have)
    while a and a[-1] == _NOP and len(a) > len(b):
        a.pop()
    while b and b[-1] == _NOP and len(b) > len(a):
        b.pop()
    diff = difflib.unified_diff(a, b, lineterm="", n=0)
    return sum(1 for x in diff if x[:1] in "+-" and x[:3] not in ("+++", "---"))


def _judge(
    config: Config, ref: Mapping[str, Sequence[Ins]], names: Sequence[str], recipe: Recipe, source: Path, work: Path
) -> Row:
    built = _build(config, recipe, source, work, "cand")
    if isinstance(built, str):
        return {"err": built}
    have = parse(_objdump(config, [built])[0])
    return {"fns": {n: (_lines([i[3] for i in ref[n]], [i[3] for i in have[n]]) if n in have else None) for n in names}}


def _census_job(item: tuple[Config, str, str, list[tuple[str, Recipe]], bool]) -> Row:
    """One unit: every candidate against the ROM-equal object (and, with drafts, against the NON_MATCHING bodies)."""
    config, version, unit, cands, drafts = item
    started = _cpu()
    root = config.project.root
    build = root / "build" / version
    work = _tmp(config) / "compilers" / f"{os.getpid()}-{digest(unit)[:10]}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        text = _without_stand_ins((build / f"{unit}.i").read_text(errors="replace"))
        source = work / "unit.i"
        source.write_text(text)
        ref = parse(_objdump(config, [build / f"{unit}.o"])[0])
        names = sorted(n for n in _c_functions(text) if n in ref)
        record: Row = {"k": "unit", "version": version, "unit": unit, "nfn": len(names), "fns": names, "cands": {}}
        if names:
            for label, recipe in cands:
                record["cands"][label] = _judge(config, ref, names, recipe, source, work)
        if drafts and "NON_MATCHING" in _text(config, unit):
            record["drafts"] = _draft_census(config, version, unit, cands, ref, _c_functions(text), work)
        record["cpu"] = _cpu() - started
        return record
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _draft_census(
    config: Config,
    version: str,
    unit: str,
    cands: Sequence[tuple[str, Recipe]],
    ref: Mapping[str, Sequence[Ins]],
    plain: set[str],
    work: Path,
) -> Row:
    """The functions only the NON_MATCHING branch defines, against the ROM-equal object under each candidate."""
    root = config.project.root
    own = recipes.of(config, unit)
    out = work / "draft.i"
    chain = adapters.toolchain(config, own.toolchain)
    flagged = replace(own, cppflags=(*own.cppflags, "-DNON_MATCHING"))
    try:
        chain.preprocess(root / unit, flagged, version, [root / "include", (root / unit).parent], out)
    except Refusal as refusal:
        return {"err": refusal.findings[0].reason[:160]}
    text = _without_stand_ins(out.read_text(errors="replace"))
    out.write_text(text)
    names = sorted(n for n in _c_functions(text) - plain if n in ref)
    result: Row = {"fns": names, "cands": {}}
    for label, recipe in cands:
        result["cands"][label] = _judge(config, ref, names, recipe, out, work) if names else {"fns": {}}
    return result


def _text(config: Config, unit: str) -> str:
    return (config.project.root / unit).read_text(errors="replace")


def _stamp(config: Config, version: str, unit: str, labels: Mapping[str, Recipe], tools: Mapping[str, str]) -> str:
    build = config.project.root / "build" / version
    return digest(
        (
            sha256((build / f"{unit}.i").read_bytes()),
            sha256((build / f"{unit}.o").read_bytes()),
            sorted((label, r.digest) for label, r in labels.items()),
            sorted({(r.toolchain, tools[r.toolchain]) for r in labels.values()}),
            sha256((config.project.root / unit).read_bytes()),
        )
    )


def census(config: Config, params: Json) -> Row:
    """Recompile the C functions of each unit with every candidate and store the differing lines in
    `.unbake/compilers/census.jsonl`. Units whose inputs are unchanged since the stored row are not recompiled; the
    call stops after `cpu` CPU seconds (default CPU_BUDGET) and a later call resumes."""
    with effort.stage("compilers.census"):
        version = params.get("version") or config.project.names_from
        paths = params.get("paths") or ()
        drafts = bool(params.get("drafts"))
        budget = float(params.get("cpu") or CPU_BUDGET)
        toolchains = _toolchains(config, params.get("toolchains"))
        build = config.project.root / "build" / version
        units = [u for u in _units(config, version, paths) if (build / f"{u}.i").is_file()]
        if not units:
            raise Refusal(Finding("compilers.input", f"no built objects and preprocessed texts under build/{version}"))
        stored = {(r["version"], r["unit"]): r for r in _read(config, CENSUS) if r.get("k") == "unit"}
        todo: list[str] = []
        plan: dict[str, tuple[str, dict[str, Recipe]]] = {}
        tools: dict[str, str] = {}
        for unit in units:
            cands = _candidates(config, toolchains, unit)
            for r in cands.values():
                tools.setdefault(r.toolchain, adapters.tool_identity(config, r.toolchain))
            key = _stamp(config, version, unit, cands, tools)
            plan[unit] = (key, cands)
            row = stored.get((version, unit))
            wants = drafts and "drafts" not in (row or {}) and "NON_MATCHING" in _text(config, unit)
            if row is None or row.get("key") != key or wants:
                todo.append(unit)
        spent, done = 0.0, 0
        while todo and spent < budget:
            chunk, todo = todo[:BATCH], todo[BATCH:]
            items = [(config, version, u, list(plan[u][1].items()), drafts) for u in chunk]
            for unit, row in zip(chunk, pool.map(config, "compilers.census", _census_job, items), strict=True):
                spent += row.pop("cpu", 0.0)
                stored[(version, unit)] = {**row, "key": plan[unit][0]}
                done += 1
        table = {
            label: {"toolchain": r.toolchain, "add": _flags(r, _plain(config, r.toolchain, ""))[0]}
            | {"omit": _flags(r, _plain(config, r.toolchain, ""))[1]}
            for _, cands in plan.values()
            for label, r in cands.items()
        }
        _write(
            config,
            CENSUS,
            [
                *({"k": "cand", "label": label, **facts} for label, facts in sorted(table.items())),
                *(stored[k] for k in sorted(stored)),
            ],
        )
        return {
            "action": "census",
            "version": version,
            "units": len(units),
            "done": done,
            "remaining": len(todo),
            "cpu_seconds": round(spent, 1),
            "candidates": sorted(table),
            "path": CENSUS.as_posix(),
        }


# ---- the proposal
def smallest_rows(
    exact: Mapping[str, frozenset[str]], default: str, current: Mapping[str, str]
) -> list[tuple[str, str]]:
    """(path, tag) rows: a path ending in `/` is a directory row, else a file row. `exact` maps each judged unit to
    the tags (candidate labels) it is exact under. The fewest rows that leave every unit with an exact tag, directory
    rows inherited downward as the build does; the project default costs nothing. Ties keep the inherited tag, then
    the tag more units of the directory use."""
    tree: dict[str, Any] = {"dirs": {}, "files": {}}
    for path, tags in exact.items():
        *folders, name = path.split("/")
        node = tree
        for folder in folders:
            node = node["dirs"].setdefault(folder, {"dirs": {}, "files": {}})
        node["files"][name] = tags
    every = sorted(set().union(*exact.values())) if exact else []

    def under(node: Mapping[str, Any]) -> Counter[str]:
        count: Counter[str] = Counter()
        for tags in node["files"].values():
            count.update(tags)
        for child in node["dirs"].values():
            count.update(under(child))
        return count

    @cache
    def solve(key: int, inherited: str, root: bool) -> tuple[int, tuple[Any, ...]]:
        node = nodes[key]
        popular = under(node)
        order = [inherited] + (
            [] if root else sorted((t for t in every if t != inherited), key=lambda t: (-popular[t], t))
        )
        best: tuple[int, tuple[Any, ...]] | None = None
        for choice in order:
            cost = 0 if choice == inherited else 1
            plan: list[Any] = [("dir", choice)] if choice != inherited else []
            for name, tags in sorted(node["files"].items()):
                if choice not in tags:
                    cost += 1
                    plan.append(("file", name, sorted(tags, key=lambda t: (len(t), t))))
            for name, child in sorted(node["dirs"].items()):
                sub, parts = solve(ids[id(child)], choice, False)
                cost += sub
                plan.append(("sub", name, parts))
            if best is None or cost < best[0]:
                best = (cost, tuple(plan))
        assert best is not None
        return best

    nodes: dict[int, Mapping[str, Any]] = {}
    ids: dict[int, int] = {}

    def number(node: Mapping[str, Any]) -> None:
        ids[id(node)] = len(nodes)
        nodes[ids[id(node)]] = node
        for child in node["dirs"].values():
            number(child)

    number(tree)
    rows: list[tuple[str, str]] = []

    def collect(prefix: str, plan: tuple[Any, ...], where: str) -> None:
        for step in plan:
            if step[0] == "dir":
                rows.append((prefix, step[1]))
            elif step[0] == "file":
                wanted = current.get(prefix + step[1], "")
                rows.append((prefix + step[1], wanted if wanted in step[2] else str(step[2][0])))
            else:
                collect(f"{prefix}{step[1]}/", step[2], step[1])

    _, plan = solve(0, default, True)
    collect("", plan, "")
    return sorted(rows)


def dead_rows(config: Config) -> list[str]:
    """[file], [dir] and [library] rows that give a path what the row above it gives already."""
    return configuration.dead_rows(config.project)


def _exact(row: Mapping[str, Any]) -> dict[str, int]:
    """Tag -> total differing lines for every candidate that built and left no function missing."""
    out: dict[str, int] = {}
    for label, c in row["cands"].items():
        if "fns" in c and all(v is not None for v in c["fns"].values()):
            out[label] = sum(c["fns"].values())
    return out


def proposal(config: Config, version: str, fp: Sequence[Row], cen: Sequence[Row]) -> Row:
    """The proposal as data (see `propose`)."""
    table = {r["label"]: r for r in cen if r.get("k") == "cand"}
    rows = {r["unit"]: r for r in cen if r.get("k") == "unit" and r["version"] == version and r["nfn"]}
    units = {r["unit"]: r for r in fp if r.get("k") == "unit"}
    default = _label(config.project.toolchain, (), ())
    exact = {u: frozenset(t for t, n in _exact(r).items() if n == 0) for u, r in rows.items()}
    current = {u: _current_label(config, u) for u in rows}
    judged = {u: tags for u, tags in exact.items() if tags}
    unresolved = sorted(u for u, tags in exact.items() if not tags)
    chosen = smallest_rows(judged, default, current)
    out_rows = []
    for path, tag in chosen:
        facts = table.get(tag, {"toolchain": tag.split()[0], "add": tag.split()[1:], "omit": []})
        covers = sum(1 for u in judged if u.startswith(path)) if path.endswith("/") else 1
        out_rows.append(
            {
                "path": path,
                "kind": "dir" if path.endswith("/") else "file",
                "tag": tag,
                "toolchain": facts["toolchain"],
                "add": facts["add"],
                "omit": facts["omit"],
                "covers": covers,
            }
        )
    exact_current = sum(1 for u, tags in exact.items() if current[u] in tags)
    disagree = []
    for unit, u in sorted(units.items()):
        label = current.get(unit) or _label(recipes.of(config, unit).toolchain, (), ())
        fingerprint = configuration.toolchain(label.split()[0]).get("fingerprint")
        if fingerprint and not allows(fingerprint, u["shapes"]) and (unit not in rows or label not in exact[unit]):
            disagree.append({"unit": unit, "row": label, "shapes": u["shapes"]})
    draft_rows = []
    for r in sorted((r for r in cen if r.get("k") == "unit" and r["version"] == version), key=lambda r: r["unit"]):
        d, unit = r.get("drafts") or {}, r["unit"]
        for fn in d.get("fns", ()):
            under = sorted(t for t, c in d["cands"].items() if "fns" in c and c["fns"].get(fn) == 0)
            if under:
                now = _current_label(config, unit) in under
                draft_rows.append({"unit": unit, "fn": fn, "exact_under": under, "exact_now": now})
    counts = Counter(t for tags in exact.values() for t in tags)
    return {
        "version": version,
        "judged": len(rows),
        "exact_current": exact_current,
        "unresolved": unresolved,
        "exact_under": dict(sorted(counts.items())),
        "rows": out_rows,
        "dead_rows": dead_rows(config),
        "disagree": disagree,
        "drafts": draft_rows,
        "current": {u: current[u] for u in sorted(current)},
        "units": {u: {"classes": units[u]["classes"] if u in units else {}, "exact": sorted(exact[u])} for u in rows},
    }


def _current_label(config: Config, unit: str) -> str:
    own = recipes.of(config, unit)
    add, omit = _flags(own, _plain(config, own.toolchain, unit))
    return _label(own.toolchain, add, omit)


def _table(config: Config, plan: Row, fp: Sequence[Row], cen: Sequence[Row]) -> bytes:
    """One line per unit that has functions: classes, the toolchains whose catalog fingerprint can emit them, the
    judged functions, the tags exact, the best tag and its differing lines, the unit's current row."""
    version = plan["version"]
    units = {r["unit"]: r for r in fp if r.get("k") == "unit"}
    rows = {r["unit"]: r for r in cen if r.get("k") == "unit" and r["version"] == version}
    catalog = configuration.toolchains()
    lines = ["unit\tfunctions\tclasses\tcan_emit\tjudged\texact\tbest\tbest_lines\tcurrent_row"]
    for unit, u in sorted(units.items()):
        can = sorted(
            t for t, row in catalog.items() if "fingerprint" in row and allows(row["fingerprint"], u["shapes"])
        )
        r = rows.get(unit)
        found = _exact(r) if r and r["nfn"] else {}
        best = min(found.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0])) if found else None
        lines.append(
            "\t".join(
                [
                    unit,
                    str(u["functions"]),
                    " ".join(f"{k}:{v}" for k, v in u["classes"].items()),
                    ",".join(can) or "-",
                    str(r["nfn"]) if r else "0",
                    ",".join(sorted(t for t, n in found.items() if n == 0)) or "-",
                    best[0] if best else "-",
                    str(best[1]) if best else "-",
                    _current_label(config, unit),
                ]
            )
        )
    return ("\n".join(lines) + "\n").encode()


def propose(config: Config, params: Json) -> Row:
    """Join `fingerprint` and `census`: the smallest directory rows (plus file exceptions) that make every judged C
    function exact, the dead rows, the units whose shapes their row cannot emit, the drafts another row makes exact.
    Writes `.unbake/compilers/propose.json` and `units.tsv`."""
    with effort.stage("compilers.propose"):
        version = params.get("version") or config.project.names_from
        fp, cen = _read(config, FINGERPRINT), _read(config, CENSUS)
        if not cen:
            raise Refusal(Finding("compilers.input", "no census yet", action="run `unbake compilers census` first"))
        plan = proposal(config, version, fp, cen)
        store.write(config.project.root / PROPOSAL, json.dumps(plan, sort_keys=True).encode())
        store.write(config.project.root / TABLE, _table(config, plan, fp, cen))
        notes = [
            Finding("compilers.dead_row", f"{p} equals the row above it", path=f"project:file.{p}", blocking=False)
            for p in plan["dead_rows"]
        ] + [
            Finding(
                "compilers.disagree",
                f"{d['unit']}: its code has shapes that {d['row']} cannot emit",
                unit=d["unit"],
                blocking=False,
            )
            for d in plan["disagree"]
        ]
        slim = {k: v for k, v in plan.items() if k not in ("units", "current")}
        return {
            "action": "propose",
            **slim,
            "path": PROPOSAL.as_posix(),
            "table": TABLE.as_posix(),
            "notes": [
                {
                    "key": f.key,
                    "reason": f.reason,
                    "path": f.path,
                    "line": 0,
                    "unit": f.unit,
                    "versions": [],
                    "missing": [],
                    "origin": None,
                    "blocking": False,
                    "action": "",
                    "symptoms": {},
                }
                for f in notes
            ],
        }


def run(config: Config, params: Json) -> Row:
    """`unbake compilers ACTION`: fingerprint, census or propose."""
    actions = {"fingerprint": fingerprint, "census": census, "propose": propose}
    action = params["action"]
    if action not in actions:
        raise Refusal(Finding("compilers.input", f"unknown action {action}: use {', '.join(actions)}"))
    return actions[action](config, params)
