"""`unbake census` and `unbake unify`: one declared type per name, proven by the objects of every version.

`census` lists every declaration of every name read from the preprocessed units of all versions, the names with more
than one type being the clashes, in the CLASH.tsv shape (typecensus). `unify` takes per clashing name the candidate type
(typeplan) that leaves every object of every version byte-identical to the baseline, rewrites the other declarations,
and rotates to the next candidate when one fails. The baseline is the objects of a build of the tree as it is.

A name touches the units whose preprocessed text holds a file it edits or copies its declaration from; only those
objects are deleted and made again. Names whose units do not meet share one batch, so a unit that fails or differs
belongs to exactly one name of its batch and the others are settled by the same build. A build that did not run is
refused, never reported as a pass: an empty log, a make exit with no unit named, or an object neither made nor named
in a failure. A name no candidate satisfies is reported with the first failing unit of each candidate type."""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from unbake import build, pool, process, store, tree
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Refusal
from unbake.typecensus import Decl, Norm, Stmt, clashes, declarations, files_of, index, scan, table
from unbake.typeplan import EXHAUSTED, Plan, Sources, plan, render, sites_of

Unit = tuple[str, str]  # (version, source path)
Facts = tuple[list[str], list[Stmt], frozenset[str]]  # (typedef statements, project statements, project files read)
_FAILED = re.compile(r"\*\*\* \[(?:[^\]\s]+: )?([^\]\s]+\.o)\]")


@dataclass(frozen=True)
class Seams:
    """What touches the real build: tests replace all three."""

    units: Callable[[], Mapping[str, Sequence[str]]]  # version -> compiled source paths
    read: Callable[[Sequence[Unit]], list[Facts]]  # the facts of each unit's preprocessed text
    make: Callable[[Sequence[str]], tuple[int, str]]  # object paths -> (exit code, log)


@dataclass
class Census:
    per: dict[str, dict[str, list[Decl]]] = field(default_factory=dict)
    closure: dict[Unit, frozenset[str]] = field(default_factory=dict)
    norm: Norm = field(default_factory=Norm)


def facts(text: str, unit: str, keep: Callable[[str], bool]) -> Facts:
    """One unit's preprocessed text reduced to what the census reads: it is all a worker sends back."""
    stmts = scan(text)
    return (
        [s[2] for s in stmts if s[2].lstrip().startswith("typedef ")],
        [s for s in stmts if keep(s[0]) and not s[2].lstrip().startswith("typedef ")],
        frozenset(f for f in files_of(text, unit) if keep(f)),
    )


def collect(units: Mapping[str, Sequence[str]], read: Callable[[Sequence[Unit]], list[Facts]]) -> Census:
    """The declarations of every name per version, and the files each unit reads."""
    items = [(v, u) for v in sorted(units) for u in sorted(units[v])]
    held = read(items)
    out = Census()
    for typedefs, _, _ in held:
        for stmt in typedefs:
            out.norm.define(stmt)
    seen: dict[str, dict[str, dict[str, Decl]]] = defaultdict(lambda: defaultdict(dict))
    for (version, unit), (_, stmts, files) in zip(items, held, strict=True):
        out.closure[(version, unit)] = files
        for name, decl in declarations(stmts, out.norm, lambda _f: True):
            seen[name][version][f"{decl[0]}\t{decl[1]}\t{decl[2]}"] = decl
    out.per = {n: {v: sorted(d.values()) for v, d in vers.items()} for n, vers in seen.items()}
    return out


def touched(census: Census, found: Plan) -> frozenset[Unit]:
    """The units whose text holds a file the plan reads or writes."""
    return frozenset(unit for unit, files in census.closure.items() if files & found.files)


def batches(units: Mapping[str, frozenset[Unit]]) -> list[list[str]]:
    """Names grouped so no two of a group touch one unit: the largest first into the first group that has room."""
    groups: list[tuple[set[Unit], list[str]]] = []
    for name in sorted(units, key=lambda n: (-len(units[n]), n)):
        for held, names in groups:
            if held.isdisjoint(units[name]):
                held |= units[name]
                names.append(name)
                break
        else:
            groups.append((set(units[name]), [name]))
    return [sorted(names) for _, names in groups]


def target(unit: Unit) -> str:
    return f"build/{unit[0]}/{unit[1]}.o"


def _sha(path: Path) -> str | None:
    return hashlib.sha1(path.read_bytes()).hexdigest() if path.is_file() else None


def rebuild(root: Path, seams: Seams, units: Iterable[Unit]) -> dict[Unit, str | None]:
    """Delete the objects of `units`, make them, and answer unit -> object hash, None for a unit whose build failed.
    A build that did not run is a Refusal (empty log, exit naming no unit, object neither made nor named)."""
    wanted = {target(u): u for u in units}
    for path in wanted:
        (root / path).unlink(missing_ok=True)
    code, log = seams.make(sorted(wanted))
    named = {m for m in _FAILED.findall(log) if m in wanted}
    if not log.strip():
        raise Refusal(Finding("unify.build", "the build log is empty: make did not run", missing=tuple(sorted(wanted))))
    if code != 0 and not named:
        raise Refusal(Finding("unify.build", f"make exited {code} and named no unit that failed: it did not build"))
    absent = sorted(p for p in wanted if p not in named and not (root / p).is_file())
    if absent:
        raise Refusal(Finding("unify.build", "make neither built nor failed these objects", missing=tuple(absent[:20])))
    return {u: None if p in named else _sha(root / p) for p, u in wanted.items()}


def _write(root: Path, files: Iterable[str], held: Iterable[Plan], src: Sources) -> None:
    held = list(held)
    for path in sorted(files):
        store.write(root / path, render(path, held, src).encode("latin-1"))


def baseline(root: Path, seams: Seams, units: Mapping[str, Sequence[str]]) -> dict[Unit, str]:
    """The object hash of every unit built from the tree as it is: a unit that does not build refuses the run."""
    made = rebuild(root, seams, [(v, u) for v in units for u in units[v]])
    broken = sorted(target(u) for u, h in made.items() if h is None)
    if broken:
        raise Refusal(
            Finding("unify.baseline", "the tree does not build before any rewrite", missing=tuple(broken[:20]))
        )
    return {u: h for u, h in made.items() if h is not None}


def run(root: Path, seams: Seams, names: Sequence[str] | None = None) -> Json:
    """Settle one type for each clashing name (or each of `names`); the tree keeps the rewrites that held."""
    units = seams.units()
    if not any(units.values()):
        raise Refusal(Finding("unify.units", "no compiled unit found: build the tree first (unbake setup)"))
    base = baseline(root, seams, units)
    census = collect(units, seams.read)
    src = Sources(lambda p: (root / p).read_bytes().decode("latin-1"))
    found = clashes(census.per)
    todo = sorted(found if names is None else [n for n in names if n in found])
    rank = dict.fromkeys(todo, 0)
    settled: dict[str, Plan] = {}
    failed: dict[str, dict[str, str]] = defaultdict(dict)
    spent: dict[str, float] = defaultdict(float)
    tried: set[Unit] = set()
    pending, rounds, built = list(todo), 0, 0
    while pending:
        rounds += 1
        plans: dict[str, Plan] = {}
        for name in pending:
            got = plan(name, sites_of(census.per, name), rank[name], src, census.norm)
            if isinstance(got, Plan):
                plans[name] = got
            elif rank[name] == 0 or got != EXHAUSTED:  # running out of candidates is how a rotation ends
                failed[name][""] = got
        pending = []
        units_of = {n: touched(census, p) for n, p in plans.items()}
        for group in batches(units_of):
            started, built = time.monotonic(), built + 1
            files = {f for n in group for f in plans[n].files}
            _write(root, files, [*settled.values(), *(plans[n] for n in group)], src)
            span = set().union(*(units_of[n] for n in group))
            tried |= span
            made = rebuild(root, seams, span)
            for name in group:
                bad = sorted(u for u in units_of[name] if made[u] != base[u])
                spent[name] += (time.monotonic() - started) / len(group)
                if bad:
                    why = "does not build" if made[bad[0]] is None else "object differs"
                    failed[name][plans[name].canon] = f"{bad[0][0]} {bad[0][1]}: {why}"
                    rank[name] += 1
                    pending.append(name)
                else:
                    settled[name] = plans[name]
            _write(root, files, settled.values(), src)  # the names that failed go back to their original text
    back = rebuild(root, seams, tried) if tried else {}
    drift = sorted(target(u) for u, h in back.items() if h != base[u])
    if drift:
        raise Refusal(
            Finding("unify.drift", "after the rewrites an object differs from the baseline", missing=tuple(drift[:20]))
        )
    return {
        "names": len(todo),
        "clashing": len(found),
        "units": sum(len(v) for v in units.values()),
        "unified": {
            n: {"type": p.canon, "why": p.why, "units": len(touched(census, p)), "seconds": round(spent[n], 3)}
            for n, p in sorted(settled.items())
        },
        "unresolved": {n: dict(failed[n]) for n in todo if n not in settled},
        "seconds": {n: round(spent[n], 3) for n in todo},
        "batches": built,
        "rounds": rounds,
        "verified": len(back),
        "files": sorted({f for p in settled.values() for f in p.files}),
    }


def _read_unit(item: tuple[Config, str, str]) -> Facts:
    cfg, version, unit = item
    return facts(tree.preprocess(cfg, unit, version, markers=True), unit, configuration.under_tree)


def seams(cfg: Config) -> Seams:
    """The real build: the project's own units, preprocessor and make."""
    root = cfg.project.root

    def units() -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for v in cfg.project.versions:
            home = root / "build" / v
            rel = (o.relative_to(home).with_suffix("").as_posix() for o in sorted(home.rglob("*.o")) if home.is_dir())
            found[v] = [
                u
                for u in rel
                if configuration.is_source(u) and not configuration.is_assembly(u) and (root / u).is_file()
            ]
        return found

    def read(items: Sequence[Unit]) -> list[Facts]:
        return list(pool.map(cfg, "unify.read", _read_unit, [(cfg, v, u) for v, u in items]))

    def make(targets: Sequence[str]) -> tuple[int, str]:
        store.write(root / "local.mk", build.local_mk(cfg))
        ran = process.make(cfg, list(targets), len(targets))
        return (-1 if ran.exit is None else ran.exit), (ran.stdout + ran.stderr).decode(errors="replace")

    return Seams(units, read, make)


def _names(path: Path | None) -> list[str] | None:
    """The names of a file: the first column of each line (a name per line, or a table in the CLASH.tsv shape)."""
    if path is None:
        return None
    rows = [line.split("\t")[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r and r != "name"]


def census_command(cfg: Config, params: Json) -> Json:
    """`unbake census`: the clashes of the tree in the CLASH.tsv shape, to --out or to the count only."""
    real = seams(cfg)
    units = real.units()
    if not any(units.values()):
        raise Refusal(Finding("unify.units", "no compiled unit found: build the tree first (unbake setup)"))
    census = collect(units, real.read)
    root = cfg.project.root
    texts = {
        p: (root / p).read_bytes().decode("latin-1")
        for r in (*configuration.source_roots(), *configuration.header_roots())
        for p in sorted(x.relative_to(root).as_posix() for x in (root / r).rglob("*") if x.is_file())
        if configuration.is_text(p)
    }
    rows = table(census.per, index(texts))
    out = params.get("out")
    if out:
        store.write(Path(out), rows.encode())
    found = clashes(census.per)
    return {
        "units": sum(len(v) for v in units.values()),
        "names": len(census.per),
        "clashing": len(found),
        "declarations": sum(len(d) for vers in census.per.values() for d in vers.values()),
        "written": str(out) if out else "",
    }


def unify_command(cfg: Config, params: Json) -> Json:
    """`unbake unify`: the rewrite itself; --names restricts it, --out writes one row per name."""
    result = run(cfg.project.root, seams(cfg), _names(params.get("names")))
    out = params.get("out")
    if out:
        rows = ["name\tstatus\ttype\tseconds\tfailing per candidate type\n"]
        for name, secs in sorted(result["seconds"].items()):
            got = result["unified"].get(name)
            bad = "; ".join(f"{t or 'no plan'} => {w}" for t, w in result["unresolved"].get(name, {}).items())
            rows.append(f"{name}\t{'unified' if got else 'unresolved'}\t{got['type'] if got else ''}\t{secs}\t{bad}\n")
        store.write(Path(out), "".join(rows).encode())
    notes = [
        Finding(
            "unify.unresolved",
            f"{n}: no declared type keeps every object: "
            + "; ".join(f"{t or 'no plan'} => {w}" for t, w in why.items()),
            blocking=False,
        )
        for n, why in result["unresolved"].items()
    ]
    return {**result, "notes": [asdict(f) for f in notes]}
