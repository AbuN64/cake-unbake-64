"""What if the compiler's scheduler had been told one thing more: the candidate compiled with artificial scheduler
dependences (`--dep A>B`, insn A depends on B so it stays after it) or with the single-set top priority turned off
for chosen insns (`--kind top-priority --insn UID`), made inside the toolchain's own cc1 under a batch gdb, then
scored like any probe. `--search` finds the smallest set of such changes that makes the function exact, or says what
it tried. The scheduler functions are found by symbol name in the cc1 the recipe uses; no address is written here and
no breakpoint is created while the inferior is stopped (that hangs a batch gdb)."""

from __future__ import annotations

import itertools
import json
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unbake import compare, config, effort, native, pool, process, sched, store
from unbake.contracts import Config, Finding, Json, Proof, Refusal

_DEP = re.compile(r"(\d+)>(\d+)(?::(\w+))?")
_MARK = re.compile(r"^WHATIF (\w+)\s*(.*)$", re.M)
_KINDS = ("deps", "top-priority")
_SEARCHES = ("singles", "pairs", "greedy")
_GDB_NAMES = ("schedule_block", "priority", "add_dependence", "adjust_priority", "birthing_insn_p")

# The batch-gdb program of the dependence kind. It runs once per compile: the breakpoints are all created before the
# run (the second one disabled), the stops are plain Breakpoint.stop() filters that create nothing, and the work
# happens in the main flow between `continue`s.
_DEPS_PROGRAM = """
import gdb, json, os
P = json.load(open(os.environ["WHATIF_PARAMS"]))
gdb.execute("set pagination off")
gdb.execute("set confirm off")
def say(text):
    print("WHATIF " + text, flush=True)
def bail(why):
    say("error " + why)
    gdb.execute("quit 3")
def u32(addr):
    return int(gdb.parse_and_eval("*(unsigned int*)%d" % addr)) & 0xffffffff
def i32(addr):
    value = u32(addr)
    return value - (1 << 32) if value >> 31 else value
def where(name):
    try:
        return int(gdb.parse_and_eval("(unsigned long)&" + name)) & 0xffffffff
    except gdb.error:
        bail("no symbol " + name + " in the compiler")
if gdb.lookup_type("void").pointer().sizeof != 4:
    bail("the compiler is not a 32-bit program")
RELOAD, NAME = where("reload_completed"), where("current_function_name")
def function():
    try:
        return gdb.parse_and_eval("*(char**)%d" % NAME).string()
    except (gdb.error, gdb.MemoryError):
        return ""
def argument(n):
    return u32(int(gdb.parse_and_eval("$esp")) + 4 * n)
class Block(gdb.Breakpoint):
    def stop(self):
        if i32(RELOAD) != 0 or function() != P["function"]:
            return False
        return i32(int(gdb.parse_and_eval("$esp")) + 4) == P["block"]
block = Block("*%d" % where(P["symbols"]["schedule_block"]))
first = gdb.Breakpoint("*%d" % where(P["symbols"]["priority"]))
first.enabled = False
add = where(P["symbols"]["add_dependence"])
def alive():
    return gdb.selected_inferior().pid != 0
gdb.execute("run")
if not alive():
    bail("block %d of %s was never scheduled before reload" % (P["block"], P["function"]))
block.enabled = False
first.enabled = True
gdb.execute("continue")
if not alive():
    bail("the scheduler never computed a priority in block %d" % P["block"])
p = argument(1)
while u32(p + 8):
    p = u32(p + 8)
chain = {}
while p:
    chain[i32(p + 4)] = p
    p = u32(p + 12)
for a, b, kind in P["deps"]:
    if a in chain and b in chain:
        gdb.execute("call (void) add_dependence(%d, %d, %d)" % (chain[a], chain[b], kind), to_string=True)
        say("dep %d %d" % (a, b))
    else:
        say("missing %d %d" % (a, b))
first.delete()
gdb.execute("continue")
say("exit %s" % gdb.parse_and_eval("$_exitcode"))
"""

# The top-priority kind is plain gdb commands (a `return` from a function belongs to the command list of its
# breakpoint, which gdb runs only when the stop is processed by its own loop).
_TOP_PROGRAM = """set pagination off
set confirm off
set $want = 0
break {adjust}
commands
silent
set $u = *(int*)(*(int*)($ebp+8)+4)
if (*(int*)&reload_completed == 0 && $_streq(*(char**)&current_function_name, "{function}") && ({uids}))
  set $want = 1
  printf "WHATIF adjust %d\\n", $u
end
continue
end
break {birthing}
commands
silent
if ($want == 1)
  set $want = 0
  return (int) 0
end
continue
end
run
printf "WHATIF exit %d\\n", $_exitcode
quit
"""


@dataclass(frozen=True)
class Outcome:
    """One what-if compile: its proof, the changes the compiler took, those it could not find, the CPU it cost."""

    proof: Proof | None
    taken: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    cpu: float = 0.0
    why: str = ""


def _refuse(reason: str, action: str, key: str = "land.request") -> Refusal:
    return Refusal(Finding(key, reason=reason, action=action))


def parse_deps(specs: Sequence[str], kinds: Mapping[str, int]) -> list[tuple[int, int, str]]:
    """`A>B[:kind]` -> (A, B, kind): insn A depends on insn B. The kind is one the toolchain row names."""
    out = []
    for spec in specs:
        found = _DEP.fullmatch(spec.strip())
        if found is None or (found[3] or "anti") not in kinds:
            raise _refuse(
                f"--dep {spec!r} is not A>B or A>B:KIND (KIND one of {', '.join(sorted(kinds))})",
                "write the insn uids the sched1 dump shows: --dep 41>31",
            )
        out.append((int(found[1]), int(found[2]), found[3] or "anti"))
    return out


def _marks(stdout: bytes) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    for key, rest in _MARK.findall(stdout.decode(errors="replace")):
        seen.setdefault(key, []).append(rest.strip())
    return seen


class Lab:
    """The compile of one candidate, repeated with what-if changes. `run(changes)` is the unit's whole chain with the
    compile step under gdb; no change runs the plain chain, so the baseline is the same pipeline."""

    def __init__(
        self,
        cfg: Config,
        version: str,
        name: str,
        path: str,
        text: str,
        overrides: Json,
        extras: Mapping[str, bytes],
        block: int,
        work: Path,
    ) -> None:
        self.cfg, self.version, self.name, self.path, self.block, self.work = cfg, version, name, path, block, work
        self.snapshot, self.unit, self.recipe, _ = native.candidate(cfg, version, path, text, overrides, extras, [name])
        row = config.toolchain(self.recipe.toolchain)
        self.kinds: dict[str, int] = dict((row.get("whatif") or {}).get("kinds") or {})
        if not self.kinds:
            raise _refuse(
                f"toolchain {self.recipe.toolchain} has no whatif row: its scheduler is not described",
                "use a gcc 2.x toolchain with a [toolchain.<id>.whatif] kinds table, or pass --toolchain",
            )
        self.gdb = shutil.which("gdb")
        if self.gdb is None:
            raise _refuse("gdb is not installed on this host", "install gdb (the what-if drives cc1 through it)")
        self.input, self.steps = native.chain_inputs(self.snapshot, self.unit, version, self.recipe)
        self.text = text
        self._count = itertools.count()

    def baseline_pass(self, work: Path) -> sched.Pass:
        """The sched1 dump of the plain compile, parsed: the insns of each block in the order the scheduler got them."""
        _, _, _, passes = native.dumps(self.snapshot, self.unit, self.version, self.recipe, work)
        if "sched" not in passes:
            raise _refuse(f"{self.recipe.toolchain} writes no sched1 dump", "use a gcc 2.x toolchain: pass --toolchain")
        return sched.Pass(passes["sched"], self.name)

    def _program(self, deps: Sequence[tuple[int, int, str]], tops: Sequence[int], folder: Path) -> tuple[Path, Path]:
        params = {
            "function": self.name,
            "block": self.block,
            "symbols": {n: n for n in _GDB_NAMES},
            "deps": [(a, b, self.kinds[k]) for a, b, k in deps],
        }
        (folder / "params.json").write_text(json.dumps(params), encoding="utf-8")
        if tops:
            uids = " || ".join(f"$u == {u}" for u in tops)
            script = folder / "program.gdb"
            script.write_text(
                _TOP_PROGRAM.format(
                    adjust="adjust_priority", birthing="birthing_insn_p", function=self.name, uids=uids
                ),
                encoding="utf-8",
            )
        else:
            script = folder / "program.py"
            script.write_text(_DEPS_PROGRAM, encoding="utf-8")
        return script, folder / "params.json"

    def run(self, deps: Sequence[tuple[int, int, str]] = (), tops: Sequence[int] = ()) -> Outcome:
        n = next(self._count)
        folder = self.work / f"run{n}"
        folder.mkdir(parents=True)
        (folder / "unit.i").write_text(self.input, encoding="utf-8")
        tmp = process.scratch(self.cfg.project.root)
        cpu, previous, marks = 0.0, "unit.i", {}
        for k, step in enumerate(self.steps):
            argv = [w.replace("{in}", previous).replace("{out}", step.out) for w in step.argv]
            env = {}
            if k == 0 and (deps or tops):
                script, params = self._program(deps, tops, folder)
                flag = ["-x", str(script)]
                argv = [str(self.gdb), "-q", "-batch", *flag, "--args", *argv]
                env = {"WHATIF_PARAMS": str(params)}
            result = process.run(f"whatif.{step.phase}", argv, folder, tmp=tmp, env=env, timeout=300)
            cpu += result.cpu_seconds
            if k == 0:
                marks = _marks(result.stdout)
                if marks.get("error"):
                    raise _refuse(marks["error"][0], "name an insn of a block the scheduler runs", "whatif.gdb")
                ran = result.exit == 0 and (not (deps or tops) or marks.get("exit", ["0"])[-1] == "0")
                if not ran:
                    why = result.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["the compile failed"]
                    return Outcome(None, cpu=cpu, why=why[0])
            elif result.exit != 0:
                return Outcome(None, cpu=cpu, why=f"{step.phase} exited {result.exit}")
            previous = step.out
        proof = native.link_object(
            self.snapshot, self.unit, self.version, self.name, self.recipe, self.text, folder / previous, folder
        )
        taken = tuple(f"{a}>{b}" for a, b, _ in deps if f"{a} {b}" in marks.get("dep", []))
        taken += tuple(f"top:{u}" for u in tops if f"{u}" in marks.get("adjust", []))
        asked = [f"{a}>{b}" for a, b, _ in deps] + [f"top:{u}" for u in tops]
        shutil.rmtree(folder, ignore_errors=True)
        return Outcome(proof, taken, tuple(c for c in asked if c not in taken), cpu)


Spec = tuple[str, str, str, str, Json, Mapping[str, bytes], int]


def _apply(
    kind: str, deps: Sequence[tuple[int, int, str]], tops: Sequence[int], change: Sequence[Any]
) -> tuple[list[Any], list[Any]]:
    """The fixed dependences and top-priority insns with one searched change set added."""
    if kind == "deps":
        return [*deps, *((a, b, "anti") for a, b in change)], list(tops)
    return list(deps), [*tops, *change]


def _job(item: tuple[Config, Spec, str, Sequence[Any], Sequence[int], Sequence[tuple[Any, ...]]]) -> list[Outcome]:
    """One worker's batch: the lab is built once, every change of the batch compiled in it."""
    cfg, spec, kind, deps, tops, batch = item
    with store.work(cfg) as work:
        lab = Lab(cfg, *spec, work)
        return [lab.run(*_apply(kind, deps, tops, change)) for change in batch]


def _label(deps: Sequence[tuple[int, int, str]], tops: Sequence[int]) -> list[str]:
    return [f"{a}>{b}" + ("" if k == "anti" else f":{k}") for a, b, k in deps] + [f"top:{u}" for u in tops]


def _named(kind: str, change: Sequence[Any]) -> list[str]:
    return _label([(a, b, "anti") for a, b in change], []) if kind == "deps" else _label([], change)


def universe(
    pass_: sched.Pass, block: int, kind: str, window: int, only: Sequence[int], anywhere: bool = False
) -> list[Any]:
    """The candidate changes in `block`: for dependences every (A, B) with B scheduled before A within `window` insns
    that A does not already depend on (with `anywhere`, B after A too: the scheduler takes such a dependence), nearest
    first; for top priority each insn. `only` restricts the insns."""
    order = sorted((u for u, b in pass_.block.items() if b == block), key=lambda u: pass_.luid[u])
    if only:
        order = [u for u in order if u in set(only)]
    if kind == "top-priority":
        return order
    held = {u: {d for d, _ in pass_.deps.get(u, ())} for u in order}
    pairs = [
        (a, b)
        for i, a in enumerate(order)
        for b in order[max(0, i - window) : (i + window + 1) if anywhere else i]
        if b != a and b not in held[a]
    ]
    return sorted(pairs, key=lambda p: abs(pass_.luid[p[0]] - pass_.luid[p[1]]))


def search(
    many: Callable[[Sequence[tuple[Any, ...]]], list[Outcome]],
    cands: Sequence[Any],
    mode: str,
    top: int,
    jobs: int,
    cpu_budget: float,
    base: Proof,
) -> dict[str, Any]:
    """Smallest exact change set: every single, then (mode pairs) every pair of the `top` best singles that changed
    the code. Stops at the first exact size, or when the CPU budget is spent (then `exhausted` is false)."""
    spent, tried = [0.0], [0]
    scored: list[tuple[float, tuple[Any, ...], Outcome]] = []

    def sweep(items: Sequence[tuple[Any, ...]]) -> list[tuple[tuple[Any, ...], Outcome]]:
        """The outcome of each change, in rounds of `jobs` batches; a round starts only while CPU budget is left."""
        done: list[tuple[tuple[Any, ...], Outcome]] = []
        for at in range(0, len(items), jobs * 16):
            if spent[0] >= cpu_budget:
                break
            round_ = list(items[at : at + jobs * 16])
            for change, out in zip(round_, many(round_), strict=True):
                spent[0] += out.cpu
                tried[0] += 1
                done.append((change, out))
        return done

    found: list[tuple[Any, ...]] = []
    sizes: dict[int, int] = {}
    if mode == "greedy":
        cands_ = list(cands)
        found, complete, scored = _greedy(sweep, cands_, base, top)
        best = max(scored, key=lambda s: s[0], default=None)
        return {
            "mode": mode,
            "tried": tried[0],
            "sizes": {},
            "candidates": len(cands_),
            "exhausted": complete,
            "cpu_seconds": round(spent[0], 1),
            "exact_sets": found[:10],
            "best": {"changes": best[1], "score": best[0]} if best else None,
        }
    singles = sweep([(c,) for c in cands])
    sizes[1] = len(singles)
    for change, out in singles:
        if out.proof is not None:
            scored.append((out.proof.score, change, out))
            if out.proof.exact:
                found.append(change)
    complete = len(singles) == len(cands)
    if not found and mode == "pairs" and complete:
        useful = sorted(  # the singles that changed the code first (best score first), then the nearest of the rest
            scored,
            key=lambda s: (s[2].proof.built_sha256 == base.built_sha256, -s[0]),  # type: ignore[union-attr]
        )[:top]
        pairs = [a[1] + b[1] for a, b in itertools.combinations(useful, 2)]
        done = sweep(pairs)
        sizes[2] = len(done)
        complete = len(done) == len(pairs)
        for change, out in done:
            if out.proof is not None:
                scored.append((out.proof.score, change, out))
                if out.proof.exact:
                    found.append(change)
    best = max(scored, key=lambda s: s[0], default=None)
    return {
        "mode": mode,
        "tried": tried[0],
        "sizes": {str(k): v for k, v in sizes.items()},
        "candidates": len(cands),
        "exhausted": complete,
        "cpu_seconds": round(spent[0], 1),
        "exact_sets": found[:10],
        "best": {"changes": best[1], "score": best[0]} if best else None,
    }


def _greedy(
    sweep: Callable[[Sequence[tuple[Any, ...]]], list[tuple[tuple[Any, ...], Outcome]]],
    cands: Sequence[Any],
    base: Proof,
    depth: int,
) -> tuple[list[tuple[Any, ...]], bool, list[tuple[float, tuple[Any, ...], Outcome]]]:
    """Grow one change set: each round adds the candidate that raises the score most; at exact, drop every member
    the set can do without. (exact sets, whether every round ran to its end, every scored set)"""
    current: tuple[Any, ...] = ()
    score = base.score
    scored: list[tuple[float, tuple[Any, ...], Outcome]] = []
    for _ in range(depth):
        rows = [r for r in sweep([(*current, c) for c in cands if c not in current]) if r[1].proof is not None]
        if not rows:
            return [], False, scored
        scored += [(o.proof.score, c, o) for c, o in rows if o.proof]
        change, out = max(rows, key=lambda r: r[1].proof.score)  # type: ignore[union-attr]
        if out.proof is None or out.proof.score <= score:
            return [], len(rows) == len(cands) - len(current), scored
        current, score = change, out.proof.score
        if out.proof.exact:
            break
    else:
        return [], True, scored
    for member in list(current):  # irredundant: the set minus any member is no longer exact
        smaller = tuple(m for m in current if m != member)
        if smaller:
            ((_, shorter),) = sweep([smaller]) or [((), Outcome(None))]
            if shorter.proof is not None and shorter.proof.exact:
                current = smaller
    return [current], True, scored


def whatif_command(config_: Config, params: Json) -> Json:
    """Compile the candidate with scheduler changes and score it; with --search find the smallest exact change set."""
    with effort.stage("whatif"):
        kind = params.get("kind") or "deps"
        mode = params.get("search")
        if kind not in _KINDS or (mode and mode not in _SEARCHES):
            raise _refuse(
                f"--kind is one of {', '.join(_KINDS)} and --search one of {', '.join(_SEARCHES)}", "fix the option"
            )
        extras, version, name, placed, _ = compare.probe_placement(config_, params)
        block = int(params.get("block") or 0)
        with store.work(config_) as work:
            lab = Lab(
                config_,
                version,
                name,
                placed.owner.path,
                placed.text,
                compare.probe_overrides(params),
                extras,
                block,
                work / "runs",
            )
            deps = parse_deps(params.get("dep") or (), lab.kinds)
            tops = [int(u) for u in params.get("insn") or ()]
            if (kind == "deps" and tops) or (kind == "top-priority" and deps):
                raise _refuse("--dep belongs to --kind deps and --insn to --kind top-priority", "pass one kind")
            base_out = lab.run()
            if base_out.proof is None:
                raise _refuse(base_out.why, "fix the candidate's compile", "score.compile")
            proof = base_out.proof
            result: dict[str, Any] = {
                "unit": placed.owner.path,
                "members": [name],
                "version": version,
                "kind": kind,
                "block": block,
                "baseline": {"exact": proof.exact, "score": proof.score},
            }
            applied = lab.run(deps, tops) if deps or tops else base_out
            if applied.proof is None:
                raise _refuse(applied.why, "the changed compile failed; try other insns", "whatif.compile")
            chosen, shown_proof = _label(deps, tops), applied.proof
            result["missing"] = list(applied.missing)
            if mode:
                pass_ = lab.baseline_pass(work / "dumps")
                cands = universe(
                    pass_,
                    block,
                    kind,
                    int(params.get("window") or 8),
                    [int(u) for u in params.get("only") or ()],
                    params.get("direction") == "any",
                )
                if not cands:
                    blocks = sorted(set(pass_.block.values()))
                    raise _refuse(
                        f"block {block} of {name} holds no insn the scheduler saw (blocks: {blocks})",
                        "pass --block one of those",
                    )

                jobs = int(params.get("jobs") or 4)
                spec: Spec = (
                    version,
                    name,
                    placed.owner.path,
                    placed.text,
                    compare.probe_overrides(params),
                    extras,
                    block,
                )

                def many(changes: Sequence[tuple[Any, ...]]) -> list[Outcome]:
                    size = -(-len(changes) // jobs)
                    batches = [changes[i : i + size] for i in range(0, len(changes), size)]
                    rows = pool.map(
                        config_, "whatif.batch", _job, [(config_, spec, kind, deps, tops, b) for b in batches]
                    )
                    return [out for row in rows for out in row]

                found: dict[str, Any] = search(
                    many, cands, mode, int(params.get("top") or 40), jobs, float(params.get("cpu_budget") or 240), proof
                )
                exact = found["exact_sets"]
                found["exact_sets"] = [_named(kind, s) for s in exact]
                if found["best"]:
                    found["best"]["changes"] = _named(kind, found["best"]["changes"])
                if exact:
                    minimal = min(exact, key=len)
                    fixed = _apply(kind, deps, tops, minimal)
                    shown_proof = lab.run(*fixed).proof or shown_proof
                    chosen = found["minimal"] = _label(*fixed)
                elif found["exhausted"]:
                    reach = "greedy growth" if mode == "greedy" else f"set of at most {2 if mode == 'pairs' else 1}"
                    found["impossible"] = (
                        f"no {reach} "
                        f"of {kind} changes makes {name} exact: "
                        f"{found['candidates']} candidates in block {block} tried"
                        + (
                            f", then every pair of the {int(params.get('top') or 40)} best singles (changed code first)"
                            if mode == "pairs"
                            else ""
                        )
                    )
                result["search"] = found
            result["changes"] = chosen
            result["exact"], result["score"] = shown_proof.exact, shown_proof.score
            shown, result["words"] = compare.probe_diff(
                config_, version, name, placed.owner.path, shown_proof, extras, params
            )
            result["diff"] = shown
        return result
