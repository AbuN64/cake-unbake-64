"""Which pseudo won each register in a gcc 2.x global allocation, by what priority, and what would flip it."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Any

from unbake.contracts import Json

_ABI = "zero at v0 v1 a0 a1 a2 a3 t0 t1 t2 t3 t4 t5 t6 t7 s0 s1 s2 s3 s4 s5 s6 s7 t8 t9 k0 k1 gp sp fp ra".split()  # noqa: SIM905
HARD = {f"${n}": i for i, n in enumerate(_ABI)} | {f"$f{i}": 32 + i for i in range(32)}
_SHOWN = {i: n for n, i in HARD.items()}


def priority(refs: int, length: int, size: int = 1) -> float:
    """global.c: floor(log2(refs)) * refs / live length * size; the higher is allocated first and takes the register."""
    return math.floor(math.log2(refs)) * refs / length * size if refs > 1 and length > 0 else 0.0


def flip(refs: int, length: int, beat: float) -> Json:
    """What lifts a pseudo of `refs` refs over `length` insns above priority `beat`: more refs or a shorter life."""
    more = next((n - refs for n in range(refs + 1, refs * 4 + 8) if priority(n, length) > beat), None)
    longest = math.ceil(refs * math.floor(math.log2(refs)) / beat) - 1 if refs > 1 and beat > 0 else 0
    return {"more_refs": more, "shorter_by": length - longest if 0 < longest < length else None}


def _edit(ours: Json, theirs: Json, flipped: Json | None) -> str:
    """The C edit that moves register `ours["register"]` from the candidate's pseudo to the target's, or why none
    exists: every differing register gets one sentence naming what to change."""
    reg, o, t = ours["register"], ours["name"], theirs["name"]
    where = " ".join(f"line {r['line']}" for r in theirs["lines"][:3])
    if ours["pseudo"] is None:
        saved = reg.startswith("$s")
        return f"the candidate never uses {reg}; the target keeps `{t}` there. " + (
            "A saved register means one more value live across a call: hold a value in a local before the "
            "first call and use it after the last."
            if saved
            else f"Keep `{t}` in a variable of its own so it gets {reg} (split or merge a temporary around {where})."
        )
    if theirs["pseudo"] is None:
        return (
            f"the candidate holds `{o}` in {reg} and has no pseudo for what the target keeps there: a value the "
            "target keeps live is missing (a local read before a call and used after it, see the frame line)."
        )
    if flipped and "cause" not in flipped:
        by = [f"use `{t}` {flipped['more_refs']} more time(s) (re-read it instead of re-deriving the value)"]
        if flipped["shorter_by"]:
            by.append(f"shorten its life by {flipped['shorter_by']} insns (assign it later or end its last use sooner)")
        return f"raise `{t}` over `{o}`: " + ", or ".join(by) + f"; or give `{o}` one reference less."
    if ours["allocator"] != theirs["allocator"]:
        wide = theirs["allocator"] == "global"
        return (
            f"`{o}` is placed by {ours['allocator']}-alloc, `{t}` by {theirs['allocator']}-alloc: "
            + (
                f"make `{t}` live across basic blocks (assign it before the branch or loop, use it after)"
                if not wide
                else f"confine `{t}` to one basic block (declare it in the inner block where it is used)"
            )
            + f", or the reverse for `{o}`."
        )
    return (
        f"`{t}` already outranks `{o}` (priority {theirs['priority']} vs {ours['priority']}), so it lost {reg} on a "
        f"conflict with a value live at the same time: split `{t}` into two temporaries around {where or 'its uses'}"
        f", or reorder so `{o}` ends before `{t}` starts."
    )


def _fewer(refs: int, length: int, beat: float) -> int | None:
    """How many references fewer a pseudo of `refs` refs over `length` insns needs to fall to priority `beat`."""
    return next((refs - n for n in range(refs - 1, 0, -1) if priority(n, length) <= beat), None)


def _spots(rows: list[Json], site: Callable[[int], str] | None, first_last: bool = False) -> str:
    """The source places of a pseudo's references, `file:line  text`; `first_last` keeps its first and last only."""
    found = [site(r["line"]) for r in rows] if site else []
    found = [t for t in found if t]
    if first_last and len(found) > 2:
        found = [found[0], found[-1]]
    return "; ".join(dict.fromkeys(found))


def _rewrites(ours: Json, theirs: Json, flipped: Json | None, site: Callable[[int], str] | None) -> list[str]:
    """Concrete edits, each naming the file:line to change and what to do there, for a register the candidate gives to
    `ours` and the target to `theirs`."""
    if flipped is None or not site:
        return []
    o, t = ours["name"], theirs["name"]
    out: list[str] = []
    last = _spots(theirs["lines"][-1:], site)
    first = _spots(theirs["lines"][:1], site)
    if flipped.get("cause") == "allocator" and theirs["lines"]:
        wide = theirs["allocator"] == "global"
        out.append(
            f"`{t}` must be placed by {theirs['allocator']}-alloc: its uses are {_spots(theirs['lines'], site)}; "
            + (
                "assign it before the branch or loop that encloses its first use and read it after"
                if wide
                else "declare it in the inner block where it is used"
            )
        )
    if flipped.get("more_refs") and not flipped.get("cause") and last:
        out.append(
            f"add {flipped['more_refs']} more read(s) of `{t}`: at {last} use the variable again where the code "
            "re-derives its value (a repeated expression, a reload through a pointer)"
        )
    if flipped.get("shorter_by") and first:
        out.append(
            f"shorten the life of `{t}` by {flipped['shorter_by']} insns: it starts at {first}; move that statement "
            "down to just before the first statement that reads it"
        )
    if flipped.get("fewer_refs") and ours["lines"]:
        out.append(
            f"give `{o}` {flipped['fewer_refs']} reference(s) fewer: its uses are {_spots(ours['lines'], site)}; "
            "read it once into the expression that needs it, or drop the repeated reads"
        )
    if ours["lines"] and theirs["lines"]:
        a, b = ours["lines"][-1]["line"], theirs["lines"][0]["line"]
        if a > b:  # the two lives overlap in the source: end the candidate's before the target's starts
            out.append(
                f"`{o}` is still live at {_spots(ours['lines'][-1:], site)} when `{t}` starts at {first}: move that "
                f"last use of `{o}` above {first}, or start `{t}` after it"
            )
    return out


def _part(text: str, name: str) -> str:
    return next((p for p in re.split(r"(?m)^;; Function ", text) if p.startswith(name + "\n")), "")


_RTX = re.compile(r"\n(?=\((?:insn|jump_insn|call_insn|note|code_label|barrier)(?:/[a-z]+)? )")
_LINE = re.compile(r'\(note(?:/[a-z]+)? \d+ \d+ \d+ \("[^"]*"\) (\d+)\)')


def _label(vars_: list[str] | None, pseudo: int) -> str:
    """The name of a pseudo from the debug records of its register. One variable names it; several variables share the
    register in different live ranges and the records do not say which one this pseudo is, so it is `unknown` with the
    candidates, never the first one (10-24: a loop's table address was named `previous`)."""
    if not vars_:
        return f"pseudo {pseudo}"
    if len(set(vars_)) == 1:
        return vars_[0]
    return f"unknown pseudo {pseudo} (one of {'/'.join(dict.fromkeys(vars_))})"


def lines(lreg: str, function: str, pseudos: set[int]) -> dict[int, list[Json]]:
    """Per pseudo, its references in each source line of the function's local-alloc dump: {line, depth, refs,
    weighted}. flow counts a reference once per loop the insn sits in plus one (REG_N_REFS), so weighted is refs times
    the loop depth and the weighted sum is the dump's "used N times" that priority reads."""
    depth, line = 1, 0
    found: dict[int, dict[tuple[int, int], int]] = {p: {} for p in pseudos}
    for chunk in _RTX.split(_part(lreg, function)):
        if re.match(r"\(note[ /]", chunk):
            depth += ("NOTE_INSN_LOOP_BEG" in chunk) - ("NOTE_INSN_LOOP_END" in chunk)
            if at := _LINE.match(chunk):
                line = int(at[1])
            continue
        if not re.match(r"\((?:insn|jump_insn|call_insn)(?:/[a-z]+)? ", chunk):
            continue
        body = chunk.split("(expr_list:REG_", 1)[0]  # the notes repeat registers the insn does not reference
        for pseudo in pseudos:
            if count := len(re.findall(rf"\(reg(?:/[a-z]+)*:[A-Z]+ {pseudo}\)", body)):
                key = (line, depth)
                found[pseudo][key] = found[pseudo].get(key, 0) + count
    return {
        p: [{"line": ln, "depth": d, "refs": n, "weighted": n * d} for (ln, d), n in sorted(rows.items())]
        for p, rows in found.items()
    }


def report(
    function: str,
    greg: str,
    lreg: str,
    asm: str,
    votes: dict[str, Counter[str]],
    site: Callable[[int], str] | None = None,
) -> list[Json]:
    """A row per register the target fills with another pseudo than the candidate's winner; `votes` maps each register
    the candidate uses to the one the target uses in the same operand of the aligned instructions. A pseudo local-alloc
    placed (in the dispositions, never in the global order) is named too, with its allocator: "free" means no pseudo
    of the function holds the register at all."""
    graph, local = _part(greg, function), _part(lreg, function)
    listed = re.search(r";; \d+ regs to allocate: (.*)", graph)
    order = [int(t) for t in listed[1].split() if t.isdigit()] if listed else []
    given = graph.partition(";; Register dispositions:")[2].partition(";; Hard regs")[0]
    placed = {int(p): int(r) for p, r in re.findall(r"(\d+) in (\d+)", given)}
    for p, r in re.findall(r"^;; Register (\d+) in (\d+)\.", local, re.M):  # local-alloc's own dump of its choices
        placed.setdefault(int(p), int(r))
    seen = re.findall(r"Register (\d+) used (\d+) times across (\d+) insns", local)
    stats = {int(n): (int(r), int(s)) for n, r, s in seen}
    start = asm.find(f'.stabs "{function}:F')  # the debug records name the variable each register holds
    end = asm.find(":F(", start + len(function) + 12)
    named: dict[int, list[str]] = {}
    for var, reg in re.findall(r'"(\w+):[rP][^"]*",\d+,\d+,\d+,(\d+)', asm[max(start, 0) : end if end > 0 else None]):
        named.setdefault(int(reg), []).append(var)
    winner: dict[int, int] = {}
    for pseudo in order:  # the first of the allocation order to be given a register keeps it
        if pseudo in placed:
            winner.setdefault(placed[pseudo], pseudo)
    by_block: dict[int, int] = {}  # local-alloc placed the pseudos of one block that global allocation never ordered
    for pseudo, reg in sorted(placed.items(), key=lambda t: -priority(*stats.get(t[0], (0, 1)))):
        if pseudo not in order and reg not in winner:
            by_block.setdefault(reg, pseudo)

    by_line = lines(lreg, function, {*winner.values(), *by_block.values()})

    def who(reg: int) -> Json:
        pseudo = winner.get(reg, by_block.get(reg))
        if pseudo is None:  # no pseudo of the candidate holds it
            return {
                "register": _SHOWN[reg],
                "pseudo": None,
                "refs": 0,
                "length": 0,
                "name": "free",
                "priority": 0.0,
                "allocator": None,
                "lines": [],
            }
        refs, span = stats.get(pseudo, (0, 1))
        return {
            "register": _SHOWN[reg],
            "pseudo": pseudo,
            "refs": refs,
            "length": span,
            "name": _label(named.get(reg), pseudo),
            "priority": round(priority(refs, span), 4),
            "allocator": "global" if reg in winner else "local",
            "lines": by_line.get(pseudo, []),
        }

    # every pair the aligned operands disagree on, the minority too: one swapped site among a hundred agreeing ones
    # is still the difference the diff shows; the most voted pairs first
    pairs = sorted(
        ((count, have, want) for have, counted in votes.items() for want, count in counted.items() if want != have),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    rows: list[Json] = []
    for count, have, want in pairs:
        a, b = HARD.get(have), HARD.get(want)
        if a is None or b is None or not {a, b} & (winner.keys() | by_block.keys()):
            continue
        ours, theirs = who(b), who(a)  # the target keeps in b the pseudo the candidate has in a
        beaten = (
            ours["pseudo"] is not None
            and theirs["pseudo"] is not None
            and ours["allocator"] == theirs["allocator"]  # one allocator's priority order decides between them
            and theirs["priority"] < ours["priority"]
        )
        lifted = flip(theirs["refs"], theirs["length"], ours["priority"]) if beaten else None
        if lifted is None and ours["pseudo"] is not None and theirs["pseudo"] is not None:
            # not a plain priority loss (the target's pseudo outranks already, or another allocator places it): the
            # prediction is what lifts it over the candidate's pseudo anyway, and what lowers the candidate's below it
            lifted = {
                **flip(theirs["refs"], theirs["length"], ours["priority"]),
                "fewer_refs": _fewer(ours["refs"], ours["length"], theirs["priority"]),
                "cause": "allocator" if ours["allocator"] != theirs["allocator"] else "conflict",
            }
        rows.append(
            {
                "register": ours["register"],
                "candidate": ours,
                "target": theirs,
                "votes": count,
                "flip": lifted,
                "edit": _edit(ours, theirs, lifted),
                "rewrites": _rewrites(ours, theirs, lifted, site),
            }
        )
    return rows


# Matched functions whose recorded tries went from a register swap to the exact body by one source edit
# (.unbake/recipe-attempts: the rows before and after the edit, scored by recipe try on every version).
PRECEDENTS: tuple[Json, ...] = (
    {
        "function": "func_8024ADD0_de",
        "kind": "fold-temp",
        "cycle": 2,
        "files": "f",
        "from": 0.9701,
        "to": 1.0,
        "edit": "dropped the local `opacity` that held one float expression (`opacity = (f32)blended + ((f32)(alpha - "
        "blended) * 0.5f); result = (u32)opacity;` became `result = (u32)((f32)blended + ((f32)(alpha - blended) * "
        "0.5f));`), which put $f0/$f1 back",
    },
)


def _fit(precedent: Json, size: int, files: str) -> int:
    return int(2 * (precedent["cycle"] == size) + (precedent["files"] == files))


def rotations(rows: list[Json], site: Callable[[int], str] | None = None) -> list[Json]:
    """The register permutations the rows make up: the target keeps in b the pseudo the candidate has in a, and the
    chain a -> b -> ... closes on a (a swap, a rotation of three). Each names the order that decides it in this
    compiler (of two overlapping lives the higher priority takes the lower-numbered free register, so the target's
    register order is the priority order it expects) and carries rewrites ranked by the matched functions in
    PRECEDENTS that fixed a rotation of the same size and register file, then the generic edits unsupported."""
    edge: dict[str, Json] = {}
    for row in rows:
        have = row["target"]["register"]
        both = row["candidate"]["pseudo"] is not None and row["target"]["pseudo"] is not None
        if both and (have not in edge or row["votes"] > edge[have]["votes"]):
            edge[have] = row
    done: set[str] = set()
    found: list[Json] = []
    for start in edge:
        chain = [start]
        while chain[-1] in edge and edge[chain[-1]]["register"] not in chain:
            chain.append(edge[chain[-1]]["register"])
        if start in done or chain[-1] not in edge or edge[chain[-1]]["register"] != start:
            continue
        done.update(chain)
        temps = [
            edge[r]["target"] for r in chain
        ]  # temps[i] sits in chain[i] and belongs in edge[chain[i]]["register"]
        wanted = sorted(temps, key=lambda t: HARD[edge[t["register"]]["register"]])
        names = ", ".join(f"`{t['name']}`" for t in wanted)
        files = "f" if all(t["register"].startswith("$f") for t in temps) else "i"
        anon = ("pseudo", "unknown")
        loose = [t for t in temps if not t["name"].startswith(anon) and len(t["lines"]) <= 2] or temps[:1]
        loose.sort(key=lambda t: (t["name"].startswith(("pseudo", "unknown")), len(t["lines"])))
        generic = [
            {
                "kind": "fold-temp",
                "text": f"fold `{t['name']}` into the expression that reads it, no local: {_spots(t['lines'], site)}",
            }
            for t in loose
        ] + [
            {
                "kind": "reorder",
                "text": f"define them in the order {names}: the first definition of each, earliest first",
            }
        ]
        ranked: list[dict[str, object]] = []
        for item in generic:
            hits = [p for p in PRECEDENTS if p["kind"] == item["kind"]]
            best = max(hits, key=lambda p: (_fit(p, len(chain), files), p["to"] - p["from"]), default=None)
            fit = _fit(best, len(chain), files) if best else 0
            ranked.append(
                {
                    **item,
                    "precedent": best and best["function"],
                    "gain": best and round(best["to"] - best["from"], 4),
                    "score": float(len(hits) * (1 + fit)),
                    "because": best and f"{best['function']}: {best['edit']}",
                }
            )
        ranked.sort(key=lambda r: -(r["score"] if isinstance(r["score"], float) else 0.0))
        found.append(
            {
                "registers": chain,
                "temps": [t["name"] for t in temps],
                "moves": [f"{t['name']} {t['register']} -> {edge[t['register']]['register']}" for t in temps],
                "kind": "swap" if len(chain) == 2 else f"rotation of {len(chain)}",
                "priority_order": [f"{t['name']} {t['priority']}" for t in sorted(temps, key=lambda t: -t["priority"])],
                "starts": {t["name"]: t["lines"][0]["line"] if t["lines"] else None for t in temps},
                "needs": [
                    f"`{a['name']}` ({a['priority']}) outranks `{b['name']}` ({b['priority']})"
                    for i, a in enumerate(wanted)
                    for b in wanted[i + 1 :]
                    if a["priority"] <= b["priority"]
                ],
                "rewrites": ranked,
            }
        )
    return found


# ---------------------------------------------------------------- the per-pseudo table
_REGEXP = re.compile(r"\(reg(/[a-z]+)*:([A-Z]+) (\d+)\)")
_ASSIGN = re.compile(r"(?:^|[;{(]|\b\w+\s+\**)\s*\**(\w+)\s*(?:[-+*/|&^]|<<|>>)?=(?!=)")
_WIDE = {"DF", "DI", "TI"}
_QUOTED = re.compile(r"`(.*)`$")


def _modes(lreg: str, function: str) -> dict[int, tuple[str, bool]]:
    """Per pseudo of the function's local-alloc dump, the machine mode of its first reference and whether it is a user
    variable (gcc marks one `reg/v`)."""
    found: dict[int, tuple[str, bool]] = {}
    for flags, mode, number in _REGEXP.findall(_part(lreg, function)):
        found.setdefault(int(number), (mode, "/v" in flags))
    return found


def _spelled(site: Callable[[int], str] | None, line: int) -> str:
    quoted = _QUOTED.search(site(line)) if site and line else None
    return quoted[1] if quoted else ""


def _source_name(row_lines: list[Json], user: bool, site: Callable[[int], str] | None) -> tuple[str, str]:
    """(name, source text) of a pseudo from the source lines that reference it: a user variable is the identifier its
    first line assigns (`near = ...`, `f32 near = ...`); a temporary is `tmp@LINE` over that line's text."""
    first = row_lines[0]["line"] if row_lines else 0
    text = _spelled(site, first)
    assigned = _ASSIGN.search(text)
    if user and assigned:
        return assigned[1], text
    if assigned and not user:
        return f"tmp@{first}", text
    for r in row_lines[1:]:  # the first line may only read it (a parameter): take the first line that assigns
        later = _ASSIGN.search(_spelled(site, r["line"]))
        if user and later:
            return later[1], text
    return (f"var@{first}" if user else f"tmp@{first}") if first else "?", text


_ARGUMENT = {HARD[r]: i for i, r in enumerate(("$a0", "$a1", "$a2", "$a3"))} | {HARD["$f12"]: 0, HARD["$f14"]: 1}
_COPY = re.compile(r"\(insn(?:/[a-z]+)? \d+ \d+ \d+ \(set \(reg\S* (\d+)\)\s+\(reg\S* (\d+) \w+\)\)")


def _arguments(lreg: str, function: str) -> dict[int, int]:
    """Per pseudo that the function first sets from an argument register, the position of that parameter."""
    out: dict[int, int] = {}
    for chunk in _RTX.split(_part(lreg, function)):
        if (m := _COPY.match(chunk)) and int(m[2]) in _ARGUMENT:
            out.setdefault(int(m[1]), _ARGUMENT[int(m[2])])
    return out


def _parameter(signature: str, index: int) -> str | None:
    """The name of parameter `index` in the one-line signature `f32 f(void *arg0, f32 speed)`."""
    inner = signature.partition("(")[2].rpartition(")")[0].split(",")
    found = re.search(r"(\w+)\s*(?:\[\d*\])?\s*$", inner[index]) if index < len(inner) else None
    return found[1] if found else None


def _holders(placed: dict[int, int], conflicts: list[int], reg: int) -> list[int]:
    return [q for q in conflicts if placed.get(q) == reg]


def placement(greg: str, lreg: str, function: str) -> dict[int, int]:
    """Pseudo -> hard register the candidate gave it: global dispositions, then local-alloc's own choices."""
    given = _part(greg, function).partition(";; Register dispositions:")[2].partition(";; Hard regs")[0]
    placed = {int(p): int(r) for p, r in re.findall(r"(\d+) in (\d+)", given)}
    for p, r in re.findall(r"^;; Register (\d+) in (\d+)\.", _part(lreg, function), re.M):
        placed.setdefault(int(p), int(r))
    return placed


_NUMBER = re.compile(r"\(reg\S* (\d+)")


def _insns(text: str, function: str) -> list[tuple[int, str]]:
    """(uid, rtl) of each insn of the function's dump, in the dump's order."""
    out = []
    for chunk in _RTX.split(_part(text, function)):
        if m := re.match(r"\((?:insn|jump_insn|call_insn)(?:/[a-z]+)? (\d+) ", chunk):
            out.append((int(m[1]), chunk))
    return out


def pseudo_votes(
    function: str,
    lreg: str,
    final: str,
    placed: dict[int, int],
    named: list[list[str]],
    pairs: dict[int, list[tuple[str, str]]],
) -> dict[int, Counter[str]]:
    """Per pseudo, the registers the target has where the candidate has the pseudo's register. A built instruction is
    tied to an insn of the final dump (`final`, hard registers, the uids of the local-alloc dump) by the registers both
    name; its operands holding the register of a pseudo that insn references in the local-alloc dump vote for it."""
    free = {HARD[r] for r in ("$zero", "$at")}
    mine = [
        (u, tuple(sorted({int(n) for n in _NUMBER.findall(rtl.split("(expr_list:REG_", 1)[0]) if int(n) < 64} - free)))
        for u, rtl in _insns(final, function)
    ]
    mine = [(u, k) for u, k in mine if k]
    built = [(i, tuple(sorted({HARD[r] for r in regs if r in HARD} - free))) for i, regs in enumerate(named)]
    built = [(i, k) for i, k in built if k]
    tie = SequenceMatcher(None, [k for _, k in mine], [k for _, k in built], autojunk=False)
    uid_of = {built[b + n][0]: mine[a + n][0] for a, b, size in tie.get_matching_blocks() for n in range(size)}
    users = {
        u: {int(n) for n in _NUMBER.findall(rtl.split("(expr_list:REG_", 1)[0])} & placed.keys()
        for u, rtl in _insns(lreg, function)
    }
    votes: dict[int, Counter[str]] = {}
    for i, uid in uid_of.items():
        for p in users.get(uid, ()):
            hard = _SHOWN.get(placed[p])
            for have, want in pairs.get(i, ()):
                if have == hard:
                    votes.setdefault(p, Counter())[want] += 1
    return votes


def _why(name: str, got: str, want: str, hard: bool, took: list[Json], is_global: bool) -> str:
    """One sentence for a mismatch: what keeps the target register `want` from `name`, which has `got`."""
    first = [t for t in took if t["earlier"]]
    if hard:
        return f"{want} is held by hard register conflict (live across {name})"
    if first:
        return f"{want} was taken first by " + ", ".join(f"{t['name']} ({t['priority']})" for t in first)
    if not is_global:
        return (
            f"{want} is not blocked: local-alloc (find_free_reg) gives {name} the lowest register not live over its "
            f"life, {got}; the target has {got} occupied there by a value the candidate does not keep"
        )
    return (
        f"{want} is not blocked in the candidate: {name} took {got}, the first free register in the order; "
        f"the target keeps another value live in {got} across {name}"
    )


def table(
    function: str, greg: str, lreg: str, votes: dict[int, Counter[str]], site: Callable[[int], str] | None = None
) -> list[Json]:
    """A row per pseudo of the function, the allocator's view of the whole function and not only of the registers that
    differ: source name and text, weighted references, live insns, priority (global.c, size counted for DF/DI pairs),
    local or global allocation and its rank in the allocation order, the hard register the candidate gave it and the one
    the target uses in the aligned operands, a MISMATCH flag, and for a mismatch the pseudos in conflict that hold the
    target's register in the candidate (who took it first), the hard registers live across it, the target's pseudos in
    the candidate's register, and the priority delta that puts it above the first one that took the register."""
    graph, local = _part(greg, function), _part(lreg, function)
    listed = re.search(r";; \d+ regs to allocate: (.*)", graph)
    order = [int(t) for t in listed[1].split() if t.isdigit()] if listed else []
    rank = {p: i for i, p in enumerate(order)}
    placed = placement(greg, lreg, function)
    conflicts = {
        int(n): [int(t) for t in rest.split()] for n, rest in re.findall(r"^;; (\d+) conflicts: (.*)$", graph, re.M)
    }
    stats = {
        int(n): (int(r), int(s)) for n, r, s in re.findall(r"Register (\d+) used (\d+) times across (\d+) insns", local)
    }
    blocks = {
        int(n): int(b) for n, b in re.findall(r"Register (\d+) used \d+ times across \d+ insns in block (\d+)", local)
    }
    modes = _modes(lreg, function)
    params = _arguments(lreg, function)
    pseudos = sorted(({*stats, *conflicts} & placed.keys()) | set(order))
    by_line = lines(lreg, function, set(pseudos))
    hard_of = {i: n for n, i in HARD.items()} | {64: "$hi", 65: "$lo"}

    def shown(reg: int | None) -> str:
        return "mem" if reg is None else hard_of.get(reg, str(reg))

    def prio(p: int) -> float:
        refs, span = stats.get(p, (0, 1))
        return priority(refs, span, 2 if modes.get(p, ("", False))[0] in _WIDE else 1)

    def wanted(p: int) -> tuple[int | None, str, str]:
        """(the target register that differs from the candidate's, most voted first, else the most voted; the names of
        every register the target has at the pseudo's sites; their counts): one site that differs is a mismatch."""
        counted = [(w, c) for w, c in votes.get(p, Counter()).most_common() if w in HARD]
        if not counted:
            return None, "", ""
        differ = [w for w, _ in counted if HARD[w] != placed.get(p)]
        names = "/".join(w for w, _ in counted)
        return HARD[differ[0] if differ else counted[0][0]], names, " ".join(f"{w}x{c}" for w, c in counted)

    rows: list[Json] = []
    for p in pseudos:
        got = placed.get(p)
        refs, span = stats.get(p, (0, 1))
        mode, user = modes.get(p, ("", False))
        name, text = _source_name(by_line.get(p, []), user, site)
        if p in params:
            name = _parameter(text, params[p]) or f"arg{params[p]}"
        want, seen, support = wanted(p)
        wrong = want is not None and want != got
        row: dict[str, Any] = {
            "pseudo": p,
            "name": name,
            "source": text[:70],
            "mode": mode,
            "refs": refs,
            "length": span,
            "priority": round(prio(p), 4),
            "allocator": "global" if p in rank else "local",
            "rank": rank.get(p),
            "register": shown(got),
            "target_register": seen or None,
            "support": support,
            "mismatch": wrong,
            "lines": by_line.get(p, []),
        }
        if wrong and got is not None and want is not None:
            near = conflicts.get(p, [])
            if p in rank:
                took = [q for q in _holders(placed, near, want) if q != p]
                hard = [shown(want)] if want in near and want < 64 else []
            else:  # local-alloc: the pseudos of its own block hold what it may not take
                took = [q for q, b in blocks.items() if b == blocks.get(p) and q != p and placed.get(q) == want]
                hard = []
            first = sorted(took, key=lambda q: (rank.get(q, -1), -prio(q)))
            beat = prio(first[0]) if first else None
            row["took"] = [
                {
                    "pseudo": q,
                    "name": _source_name(by_line.get(q, []), modes.get(q, ("", False))[1], site)[0],
                    "priority": round(prio(q), 4),
                    "allocator": "global" if q in rank else "local",
                    "rank": rank.get(q),
                    "earlier": q not in rank or p not in rank or rank[q] < rank[p],
                }
                for q in first
            ]
            row["hard_conflict"] = hard
            row["rom_gives"] = [
                shown(got) + "->" + str(q) for q in near if q != p and wanted(q)[0] == got and placed.get(q) != got
            ]
            row["delta"] = (
                {"above": round(beat, 4), "needs": round(beat - prio(p), 4), **flip(refs, span, beat)}
                if beat is not None and beat >= prio(p)
                else None
            )
            row["why"] = _why(name, shown(got), shown(want), bool(hard), row["took"], p in rank)
        rows.append(row)
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] if r["rank"] is not None else -r["priority"], r["pseudo"]))
    return rows
