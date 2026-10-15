"""Why gcc 2.x placed an instruction where it did: the scheduler dumps (-dS sched1, -dR sched2) name per insn its
priority, its dependencies and the ready list it was picked from, so a moved store reads as a priority or an input
order tie instead of a guess."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from unbake.alloc import HARD
from unbake.contracts import Json

_ASM = re.compile(r"(l|s)[a-z]*[0-9a-z]* (\$\w+), (-?(?:0x[0-9A-Fa-f]+|\d+))\((\$\w+)\)")
_BLOCK = re.compile(r";;\t -- basic block number (\d+) from \d+ to \d+ --")
_READY = re.compile(r";; ready list at T-(\d+): (.*?)(?:, now (.*))?$")
_PRIORITY = re.compile(r";; insn\[\s*(\d+)\]: priority =\s*(-?\d+), ref_count =\s*(-?\d+)")
_INSN = re.compile(r"\((?:jump_|call_)?insn (\d+) \d+ \d+ ")
_LINE = re.compile(r'\(note \d+ \d+ \d+ \("[^"]*"\) (\d+)\)')
_DEPENDS = re.compile(r"insn_list(?::REG_DEP_(ANTI|OUTPUT))? (\d+)")
_KINDS = {None: "true", "ANTI": "anti", "OUTPUT": "output"}


def memory_key(text: str) -> tuple[str, int, int, int] | None:
    """(direction, register, offset, base) of a load or store line `sw $zero, 0x2C($v0)`; None for any other line."""
    found = _ASM.fullmatch(text.strip())
    if not found or found[2] not in HARD or found[4] not in HARD:
        return None
    return found[1], HARD[found[2]], int(found[3], 0), HARD[found[4]]


_MEM = r"\(mem\S* (?:\(plus\S* \(reg\S* (\d+) \S+\) \(const_int (-?\d+)\)\)|\(reg\S* (\d+) \S+\))\)"
_REG = r"(?:\(reg\S* (\d+) \S+\)|\(const_int 0\))"
_STORE = re.compile(_MEM + " " + _REG + r"\)")
_LOAD = re.compile(r"\(reg\S* (\d+) \S+\) " + _MEM + r"\)")


def _rtl_key(rtl: str) -> tuple[str, int, int, int] | None:
    flat = " ".join(rtl.split())
    one = re.match(r"\((?:jump_|call_)?insn \d+ \d+ \d+ \(set (.*?) \d+ \{", flat)
    if not one:
        return None
    if (m := _STORE.fullmatch(one[1])) is not None:
        return "s", int(m[4] or 0), int(m[2] or 0), int(m[1] or m[3])
    if (m := _LOAD.fullmatch(one[1])) is not None:
        return "l", int(m[1]), int(m[3] or 0), int(m[2] or m[4])
    return None


def _part(text: str, name: str) -> str:
    return next((p for p in re.split(r"(?m)^;; Function ", text) if p.startswith(name + "\n")), "")


class Pass:
    """One scheduler dump of one function: the insns in final order, per block their priorities and ready lists."""

    def __init__(self, text: str, function: str) -> None:
        part = _part(text, function)
        cut = [m for m in _BLOCK.finditer(part)]
        self.block: dict[int, int] = {}  # insn -> basic block
        self.priority: dict[int, tuple[int, int]] = {}
        self.luid: dict[int, int] = {}  # position of the insn in the order the pass was given
        self.events: dict[int, list[tuple[int, str, list[str]]]] = defaultdict(list)
        end = cut[-1].end() if cut else 0
        tail = part[end:]
        first_rtl = re.search(r"(?m)^\(", tail)
        rtl_at = end + (first_rtl.start() if first_rtl else len(tail))
        for k, head in enumerate(cut):
            body = part[head.end() : cut[k + 1].start() if k + 1 < len(cut) else rtl_at]
            block, lines = int(head[1]), body.splitlines()
            for n, row in enumerate(lines):
                if (p := _PRIORITY.match(row)) is not None:
                    uid = int(p[1])
                    self.block[uid] = block
                    self.priority[uid] = (int(p[2]), int(p[3]))
                    self.luid[uid] = len(self.luid)
                elif (r := _READY.match(row)) is not None:
                    notes = []
                    for after in lines[n + 1 :]:
                        if not after.startswith(";;") or _READY.match(after) or after.startswith(";; insn["):
                            break
                        notes.append(after)
                    self.events[block].append((int(r[1]), (r[2] or "") + "\0" + (r[3] or ""), notes))
        self.order: list[int] = []
        self.line: dict[int, int] = {}
        self.deps: dict[int, list[tuple[int, str]]] = {}
        self.key: dict[int, tuple[str, int, int, int]] = {}
        at = 0
        for chunk in re.split(r"\n\s*\n", part[rtl_at:]):
            if (m := _LINE.match(chunk.strip())) is not None:
                at = int(m[1])
            elif (m := _INSN.match(chunk.strip())) is not None:
                uid = int(m[1])
                self.order.append(uid)
                self.line[uid] = at
                self.deps[uid] = [(int(d[2]), _KINDS[d[1] or None]) for d in _DEPENDS.finditer(chunk)]
                if (key := _rtl_key(chunk.strip())) is not None:
                    self.key[uid] = key

    def picked(self, uid: int) -> Json | None:
        """The last ready list `uid` sat in (the pass is backward: it leaves there) with the insns picked before and
        after it, and whether the neighbours are tied with it on priority."""
        found = None
        for at, packed, _ in self.events.get(self.block.get(uid, -1), ()):
            ready, _, now = packed.partition("\0")
            weight = {int(a): int(b, 16) for a, b in re.findall(r"(\d+) \(([0-9a-f]+)\)", ready)}
            if uid in weight:
                found = (at, weight, [int(n) for n in now.split()] or list(weight))
        if found is None:
            return None
        at, weight, queue = found
        here = queue.index(uid) if uid in queue else len(queue)

        def row(o: int) -> Json:
            return {"insn": o, "line": self.line.get(o), "priority": weight.get(o), "luid": self.luid.get(o)}

        return {
            "at": f"T-{at}",
            "before": [row(o) for o in queue[:here]][-6:],  # picked first, so placed after it
            "after": [row(o) for o in queue[here + 1 :]][:6],  # picked later, so placed before it
            "priority": weight[uid],
        }


def _weight(priority: int) -> str:
    """Insns grouped to the end of the block (a call's argument setup) carry 0x7f000000+: tied by input order only."""
    return "pinned to the block end" if priority >= 0x7F000000 else str(priority)


def _rival(beside: Json, mine: int, own: int) -> str:
    where = f"line {beside['line']}" if beside["line"] else f"insn {beside['insn']}"
    if beside["priority"] == mine:
        tie = f"ties it at priority {_weight(mine)} and is ordered by input position"
        return f"{where} {tie} (luid {beside['luid']} vs {own})"
    return f"{where} has priority {_weight(beside['priority'])} against its {_weight(mine)}"


def hint(words: int, pick: Json, own: int) -> str:
    """What decides the move: the insn picked just before (it must be picked before this one to land later) or just
    after it (it must be picked after to land earlier); priority is the longest dependency path to the block end."""
    side = pick["before"][-1:] if words > 0 else pick["after"][:1]
    if not side:
        return "nothing else was ready: the order is fixed by its dependencies"
    way = "before" if words > 0 else "after"  # the pass runs backward: picked first means placed last
    place = "later" if words > 0 else "earlier"
    return f"to land {abs(words)} words {place} it must be picked {way} {_rival(side[0], pick['priority'], own)}"


def displacement(dump: Pass, uid: int) -> int:
    """How far a pass carried the insn from the position it was given in: output place minus input place among the insns
    the pass both read and wrote; 0 when the pass left it where it found it."""
    seen = [u for u in dump.order if u in dump.luid]
    if uid not in seen:
        return 0
    return seen.index(uid) - sorted(seen, key=lambda u: dump.luid[u]).index(uid)


def moved(everything: list[str]) -> list[tuple[str, str, str]]:
    """(text, built address, target address) of every instruction the diff shows once on each side, in order."""
    plus: dict[str, list[str]] = defaultdict(list)
    minus: dict[str, list[str]] = defaultdict(list)
    for row in everything:
        if (m := re.fullmatch(r"([+-])([0-9A-F]{8}): (.*?)( -> [0-9A-F]{8})?", row)) is not None:
            (plus if m[1] == "+" else minus)[m[3]].append(m[2])
    return [(t, b, minus[t][i]) for t, adds in plus.items() for i, b in enumerate(adds) if i < len(minus[t])]


def report(function: str, passes: dict[str, str], listing: list[str], everything: list[str]) -> list[Json]:
    """A row per memory instruction the diff shows moved: the insn, its source line, and per scheduler pass its
    priority, dependencies and the ready-list tie that placed it. `listing` is the function's built lines in order."""
    first = int(listing[0][:8], 16) if listing else 0
    texts = [row[10:] for row in listing]
    seen: dict[tuple[str, int, int, int], list[int]] = defaultdict(list)
    for n, text in enumerate(texts):
        if (key := memory_key(text)) is not None:
            seen[key].append(n)
    dumps = {name: Pass(text, function) for name, text in passes.items()}
    final_name = "sched2" if "sched2" in dumps else next(iter(dumps), "")
    final = dumps.get(final_name)
    if final is None:
        return []
    byKey: dict[tuple[str, int, int, int], list[int]] = defaultdict(list)
    for uid in final.order:
        if uid in final.key:
            byKey[final.key[uid]].append(uid)
    rows: list[Json] = []
    for text, built, target in moved(everything):
        key = memory_key(text)
        at = (int(built, 16) - first) // 4
        if key is None or at not in seen[key] or seen[key].index(at) >= len(byKey[key]):
            continue
        uid = byKey[key][seen[key].index(at)]
        row: dict[str, Any] = {
            "asm": text,
            "built": built,
            "target": target,
            "words": (int(target, 16) - int(built, 16)) // 4,
            "insn": uid,
            "line": final.line.get(uid),
        }
        for name, dump in dumps.items():
            if uid not in dump.priority:
                continue
            row[name] = {
                "block": dump.block[uid],
                "priority": dump.priority[uid][0],
                "luid": dump.luid[uid],
                "after": [{"insn": d, "kind": k, "line": dump.line.get(d)} for d, k in dump.deps.get(uid, ())][:6],
                "picked": dump.picked(uid),
            }
            row[name]["displaced"] = displacement(dump, uid)
            if name == final_name and row[name]["picked"]:
                row["hint"] = hint(row["words"], row[name]["picked"], dump.luid[uid])
        row["first_pass"] = next(
            (n for n in sorted(dumps) if isinstance(row.get(n), dict) and row[n]["displaced"]), "input"
        )  # sched1 runs before sched2; "input": neither pass moved it, the source order itself differs
        rows.append(row)
    return rows
