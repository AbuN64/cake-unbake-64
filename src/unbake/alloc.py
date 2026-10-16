"""Which pseudo won each register in a gcc 2.x global allocation, by what priority, and what would flip it."""

from __future__ import annotations

import math
import re
from collections import Counter

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


def _part(text: str, name: str) -> str:
    return next((p for p in re.split(r"(?m)^;; Function ", text) if p.startswith(name + "\n")), "")


def report(function: str, greg: str, lreg: str, asm: str, votes: dict[str, Counter[str]]) -> list[Json]:
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
            }
        refs, span = stats.get(pseudo, (0, 1))
        return {
            "register": _SHOWN[reg],
            "pseudo": pseudo,
            "refs": refs,
            "length": span,
            "name": "/".join(named.get(reg, [f"pseudo {pseudo}"])),
            "priority": round(priority(refs, span), 4),
            "allocator": "global" if reg in winner else "local",
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
        rows.append(
            {
                "register": ours["register"],
                "candidate": ours,
                "target": theirs,
                "votes": count,
                "flip": flip(theirs["refs"], theirs["length"], ours["priority"]) if beaten else None,
            }
        )
    return rows
