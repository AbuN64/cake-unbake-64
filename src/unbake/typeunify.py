"""`unbake census` and `unbake unify`: one declared type per name, proven by the objects of every version.

`census` lists every declaration of every name read from the preprocessed units of all versions, the names with more
than one type being the clashes, in the CLASH.tsv shape (typecensus). `unify` takes per clashing name the candidate type
(typeplan) that leaves every object of every version byte-identical to the baseline, rewrites the other declarations,
and rotates to the next candidate when one fails. The baseline is the objects of a build of the tree as it is.

A name touches the units whose preprocessed text holds a file it edits or copies its declaration from and that
define or use it (a body or an initializer that spells it, or a definition of it): only those objects are deleted,
made again and compared. Names whose units do not meet share one batch, so a unit that fails or differs belongs to
exactly one name of its batch and the others are settled by the same build. Once, after all rewrites, every unit
that holds a file a kept rewrite touches is made again and compared with the baseline: a unit the finer reading
missed is a drift and refuses the run. One line per name goes to the person watching (picked type,
candidates tried, the failing unit of each, seconds). A build that did not run is
refused, never reported as a pass: an empty log, a make exit with no unit named, or an object neither made nor named
in a failure. A name no candidate satisfies is reported with the first failing unit of each candidate type."""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from unbake import build, effort, pool, process, store, tree
from unbake import config as configuration
from unbake.contracts import Config, Finding, Host, Json, Refusal
from unbake.typecensus import Decl, Norm, Stmt, clashes, declarations, files_of, index, scan, table, tkey
from unbake.typeplan import EXHAUSTED, Plan, Sources, plan, render, sites_of

Unit = tuple[str, str]  # (version, source path)
Facts = tuple[list[str], list[Stmt], frozenset[str], frozenset[str]]  # (typedefs, project statements, files, words)
_FAILED = re.compile(r"\*\*\* \[(?:[^\]\s]+: )?([^\]\s]+\.o)\]")
_WORD = re.compile(r"[A-Za-z_]\w*")
_POINTER = re.compile(r"\(\s*\*")


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
    words: dict[Unit, frozenset[str]] = field(default_factory=dict)  # per unit: what it defines or uses
    norm: Norm = field(default_factory=Norm)
    held: dict[Unit, list[Stmt]] = field(default_factory=dict)  # the statements of the units that read a `watch` file
    by_file: dict[str, dict[str, set[Stmt]]] = field(default_factory=dict)  # version -> watched file -> statements
    named: dict[Stmt, list[tuple[str, Decl]]] = field(default_factory=dict)  # what each statement declares, read once


def _declares_only(stmt: Stmt) -> bool:
    """Whether a statement only declares what it names (typedef, extern, prototype): its words are no use of them."""
    text = stmt[2].lstrip()
    if stmt[3] != "stmt" or "=" in text:
        return False
    return text.startswith(("typedef ", "extern ")) or ("(" in text and not _POINTER.search(text))


def spelled(text: str, stmts: Sequence[Stmt]) -> frozenset[str]:
    """The words a unit defines or uses: those of every statement that defines something, and those of the text outside
    its depth-0 statements (function bodies and initializers). A word only a declaration spells is not in it."""
    inside = Counter(w for s in stmts for w in _WORD.findall(s[2]))
    total = Counter(_WORD.findall("\n".join(ln for ln in text.split("\n") if not ln.startswith("# "))))
    defined = {w for s in stmts if not _declares_only(s) for w in _WORD.findall(s[2])}
    return frozenset(defined | {w for w, n in total.items() if n > inside[w]})


def facts(text: str, unit: str, keep: Callable[[str], bool]) -> Facts:
    """One unit's preprocessed text reduced to what the census reads: it is all a worker sends back."""
    stmts = scan(text)
    return (
        [s[2] for s in stmts if s[2].lstrip().startswith("typedef ")],
        [s for s in stmts if keep(s[0]) and not s[2].lstrip().startswith("typedef ")],
        frozenset(f for f in files_of(text, unit) if keep(f)),
        spelled(text, stmts),
    )


def collect(
    units: Mapping[str, Sequence[str]],
    read: Callable[[Sequence[Unit]], list[Facts]],
    watch: frozenset[str] = frozenset(),
) -> Census:
    """The declarations of every name per version, and the files each unit reads. The statements of the units that
    read a `watch` file are kept (the fold-check composes them)."""
    items = [(v, u) for v in sorted(units) for u in sorted(units[v])]
    held = read(items)
    out = Census()
    spelled_out: set[str] = set()  # a typedef or a statement of a shared header is in every unit: read it once
    for typedefs, _, _, _ in held:
        for typedef in typedefs:
            if typedef not in spelled_out:
                spelled_out.add(typedef)
                out.norm.define(typedef)
    seen: dict[str, dict[str, dict[str, Decl]]] = defaultdict(lambda: defaultdict(dict))
    parsed: dict[str, set[Stmt]] = defaultdict(set)
    for (version, unit), (_, stmts, files, words) in zip(items, held, strict=True):
        out.closure[(version, unit)] = files
        out.words[(version, unit)] = words
        fresh = [s for s in stmts if s not in parsed[version]]
        parsed[version].update(fresh)
        for stmt in fresh:
            if stmt not in out.named:  # a statement of a shared header reads the same in every version
                out.named[stmt] = declarations([stmt], out.norm, lambda _f: True)
            for name, decl in out.named[stmt]:
                seen[name][version][f"{decl[0]}\t{decl[1]}\t{decl[2]}"] = decl
        if files & watch:
            out.held[(version, unit)] = stmts
            of_version = out.by_file.setdefault(version, {})
            for stmt in stmts:
                if stmt[0] in watch:
                    of_version.setdefault(stmt[0], set()).add(stmt)
    out.per = {n: {v: sorted(d.values()) for v, d in vers.items()} for n, vers in seen.items()}
    return out


Fold = tuple[str, str]  # (header folded away, target header it folds into)


def folds_of(path: Path) -> list[Fold]:
    """The (header, target) rows of a fold list: the first two columns of each line (a header row is skipped)."""
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(r[0].strip(), r[1].strip()) for r in rows if len(r) > 1 and r[0] != "header"]


def fold_groups(folds: Iterable[Fold]) -> list[frozenset[str]]:
    """The files that become one header: folds that share a file are one group."""
    groups: list[set[str]] = []
    for pair in folds:
        joined = [g for g in groups if g.intersection(pair)]
        merged = set(pair).union(*joined)
        groups = [g for g in groups if g not in joined] + [merged]
    return [frozenset(g) for g in groups]


def folded(census: Census, groups: Sequence[frozenset[str]]) -> dict[str, dict[str, list[Decl]]]:
    """The declarations of every name that two types meet in one unit once `groups` are one header each: a unit that
    reads a file of a group reads the statements of the whole group. name -> version -> declarations. A unit of the
    built tree has no such name, so each is a conflict the fold makes."""
    found: dict[str, dict[str, dict[str, Decl]]] = defaultdict(lambda: defaultdict(dict))
    for (version, unit), stmts in census.held.items():
        files = census.closure[(version, unit)]
        seen = set(stmts)
        for group in groups:
            if files & group:
                for file in group:
                    seen |= census.by_file.get(version, {}).get(file, set())
        local: dict[str, dict[str, Decl]] = defaultdict(dict)
        for stmt in seen:
            for name, decl in census.named.get(stmt, ()):
                local[name][f"{decl[0]}\t{decl[1]}\t{decl[2]}"] = decl
        for name, decls in local.items():
            if len({tkey(d[2]) for d in decls.values()}) > 1:
                found[name][version].update(decls)
    return {n: {v: sorted(d.values()) for v, d in vers.items()} for n, vers in found.items()}


def reach(census: Census, found: Plan) -> frozenset[Unit]:
    """The units whose text holds a file the plan reads or writes."""
    return frozenset(unit for unit, files in census.closure.items() if files & found.files)


def touched(census: Census, found: Plan) -> frozenset[Unit]:
    """The units of its reach that define or use the name: the only objects the plan can change."""
    return frozenset(u for u in reach(census, found) if found.name in census.words[u])


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
    if not wanted:
        return {}
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


def differing(units: Iterable[Unit], made: Mapping[Unit, str | None], base: Mapping[Unit, str]) -> list[Unit]:
    """The units of `units`, in order, whose new object is missing (the build failed) or not the baseline's."""
    return sorted(u for u in units if made[u] != base[u])


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


def line(name: str, got: Plan | None, failed: Mapping[str, str], seconds: float) -> str:
    """The progress line of a name that is done: the type picked (or none), how many candidates were tried, the failing
    unit of each that failed, the seconds its builds took."""
    tried = sum(1 for canon in failed if canon) + (got is not None)
    why = "; ".join(f"{canon or 'no plan'} => {reason}" for canon, reason in failed.items())
    return (
        f"unify {name}: {'unified as ' + got.canon if got else 'unresolved'}; {tried} candidate(s) tried"
        + (f"; failing {why}" if why else "")
        + f"; {seconds:.1f}s"
    )


def run(root: Path, seams: Seams, names: Sequence[str] | None = None, say: Callable[[str], None] = effort.say) -> Json:
    """Settle one type for each clashing name (or each of `names`); the tree keeps the rewrites that held. `say` takes
    one line per name as soon as it is settled or out of candidates."""
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
    pending, rounds, built, rebuilt = list(todo), 0, 0, 0
    while pending:
        rounds += 1
        plans: dict[str, Plan] = {}
        for name in pending:
            got = plan(name, sites_of(census.per, name), rank[name], src, census.norm)
            if isinstance(got, Plan):
                plans[name] = got
            else:
                if rank[name] == 0 or got != EXHAUSTED:  # running out of candidates is how a rotation ends
                    failed[name][""] = got
                say(line(name, None, failed[name], spent[name]))
        pending = []
        units_of = {n: touched(census, p) for n, p in plans.items()}
        for group in batches(units_of):
            started, built = time.monotonic(), built + 1
            files = {f for n in group for f in plans[n].files}
            _write(root, files, [*settled.values(), *(plans[n] for n in group)], src)
            span = set().union(*(units_of[n] for n in group))
            tried |= span
            rebuilt += len(span)
            made = rebuild(root, seams, span)
            for name in group:
                bad = differing(units_of[name], made, base)
                spent[name] += (time.monotonic() - started) / len(group)
                if bad:
                    why = "does not build" if made[bad[0]] is None else "object differs"
                    failed[name][plans[name].canon] = f"{bad[0][0]} {bad[0][1]}: {why}"
                    rank[name] += 1
                    pending.append(name)
                else:
                    settled[name] = plans[name]
                    say(line(name, plans[name], failed[name], spent[name]))
            _write(root, files, settled.values(), src)  # the names that failed go back to their original text
    reached = set().union(tried, *(reach(census, p) for p in settled.values()))
    back = rebuild(root, seams, reached)
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
        "rebuilt": rebuilt,
        "files": sorted({f for p in settled.values() for f in p.files}),
    }


READ_CHUNK = 16  # units one worker reads per job: the statements of their shared headers travel once


def _read_chunk(items: list[tuple[Config, str, str]]) -> list[Facts]:
    """The facts of several units, equal statements and words being one object: pickled once, held once in the parent
    (every unit of a version repeats the statements of the headers it shares)."""
    same: dict[object, object] = {}
    out: list[Facts] = []
    for cfg, version, unit in items:
        typedefs, stmts, files, words = facts(
            tree.preprocess(cfg, unit, version, markers=True), unit, configuration.under_tree
        )
        out.append(
            (
                [same.setdefault(t, t) for t in typedefs],  # type: ignore[misc]
                [same.setdefault(t, t) for t in stmts],  # type: ignore[misc]
                frozenset(same.setdefault(f, f) for f in files),  # type: ignore[misc]
                frozenset(same.setdefault(w, w) for w in words),  # type: ignore[misc]
            )
        )
    return out


def jobs(host: Host) -> int:
    """The job count of a build step: every slot the host grants this command, whatever number of objects it names (the
    steps of one object are many: preprocess, compile, stamps), so the step never starts below what the host can run."""
    return max(1, host.slots)


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
        chunks = [[(cfg, v, u) for v, u in items[at : at + READ_CHUNK]] for at in range(0, len(items), READ_CHUNK)]
        return [got for chunk in pool.map(cfg, "unify.read", _read_chunk, chunks) for got in chunk]

    def make(targets: Sequence[str]) -> tuple[int, str]:
        store.write(root / "local.mk", build.local_mk(cfg))
        ran = process.make(cfg, list(targets), jobs(cfg.host))
        return (-1 if ran.exit is None else ran.exit), (ran.stdout + ran.stderr).decode(errors="replace")

    return Seams(units, read, make)


def _names(path: Path | None) -> list[str] | None:
    """The names of a file: the first column of each line (a name per line, or a table in the CLASH.tsv shape)."""
    if path is None:
        return None
    rows = [line.split("\t")[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r and r != "name"]


def census_command(cfg: Config, params: Json) -> Json:
    """`unbake census`: the clashes of the tree in the CLASH.tsv shape, to --out or to the count only. With --folds
    (a fold list: header and target per line) the table is the fold-check instead: the names two types meet at in a
    unit once each group of folded headers is one header, `fold-only` where the tree as it is has no clash of the name;
    nothing is written but --out."""
    real = seams(cfg)
    units = real.units()
    if not any(units.values()):
        raise Refusal(Finding("unify.units", "no compiled unit found: build the tree first (unbake setup)"))
    folds = folds_of(Path(params["folds"])) if params.get("folds") else []
    groups = fold_groups(folds)
    census = collect(units, real.read, frozenset().union(*groups))
    root = cfg.project.root
    texts = {
        p: (root / p).read_bytes().decode("latin-1")
        for r in (*configuration.source_roots(), *configuration.header_roots())
        for p in sorted(x.relative_to(root).as_posix() for x in (root / r).rglob("*") if x.is_file())
        if configuration.is_text(p)
    }
    users = index(texts)
    found = clashes(census.per)
    made = folded(census, groups) if folds else {}
    out = params.get("out")
    if out:
        rows = (
            table(made, users, {n: "open" if n in found else "fold-only" for n in made})
            if folds
            else table(census.per, users)
        )
        store.write(Path(out), rows.encode())
    return {
        "units": sum(len(v) for v in units.values()),
        "names": len(census.per),
        "clashing": len(found),
        "declarations": sum(len(d) for vers in census.per.values() for d in vers.values()),
        "written": str(out) if out else "",
        **(
            {"folds": len(folds), "folded": len(made), "fold_only": sorted(n for n in made if n not in found)}
            if folds
            else {}
        ),
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
