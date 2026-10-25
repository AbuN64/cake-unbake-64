"""`unbake order FILE --function FN`: a statement-order search for one function that is close but not exact.

The candidate's statements (the function body, then each nested block of four lines or more, innermost first) are
moved one at a time inside a window, never across a statement they share a variable with (read/write, write/read,
write/write by identifier, `x[0]` apart from `x[1]`, any `ptr->` store and any call a memory barrier), and each move is
probed like `unbake probe`: one version, one module, about a second. The best order of a round is kept and the next
round starts from it; the climb ends exact, when no move helps, or at --budget probes. The input is never edited: the
best source goes to --out. The dependency guard is an approximation of C semantics; run behave on the result.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from pathlib import Path

from unbake import compare, effort, pool, store
from unbake.contracts import Config, Finding, Json, Refusal

_DECL = re.compile(
    r"\s*(const\s+)?(struct\s+\w+|union\s+\w+|[A-Z]\w*|u8|s8|u16|s16|u32|s32|f32|f64|int|char|void)\b[\s\*]*\w+"
    r"[^=(]*(=[^;]*)?;\s*$"
)
_MEMREF = re.compile(
    r"[A-Za-z_]\w*(?:\s*(?:->|\.)\s*\w+|\s*\[\s*\d+\s*\])*\s*->\s*\w+(?:\s*(?:->|\.)\s*\w+|\s*\[\s*\d+\s*\])*"
)
_CONTROL = re.compile(r"\b(if|for|while|do|switch|return|goto|break|continue)\b")
_NOT_CALL = {"if", "for", "while", "switch", "return", "sizeof"}
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_MIN_BLOCK = 4  # a nested block is climbed on its own from this many lines
_BARRIER = "@*"


def _tokens(text: str) -> set[str]:
    text = _MEMREF.sub(" ", text)
    return set(re.findall(r"[A-Za-z_]\w*", re.sub(r"\[(\d+)\]", r"_\1", text)))


def _memory(text: str) -> set[str]:
    return {"@" + re.sub(r"\s+", "", m) for m in _MEMREF.findall(text)}


def _calls(text: str) -> bool:
    stripped = re.sub(r"\(\s*(?:struct\s+)?\w+\s*\**\s*\)", " ", text)
    return any(m[1] not in _NOT_CALL for m in _CALL.finditer(stripped))


def effects(statement: str) -> tuple[set[str], set[str]]:
    """(written, read) names of one statement; control flow and bare calls read and write everything they touch."""
    if _CONTROL.search(statement) or statement.count("{") or "=" not in statement:
        every = _tokens(statement) | _memory(statement) | {_BARRIER}
        return every, every
    assign = re.match(r"\s*(.*?)\s*([|&+\-*/^]|<<|>>)?=(?!=)(.*)", statement, re.S)
    if assign is None:
        every = _tokens(statement) | _memory(statement) | {_BARRIER}
        return every, every
    left, op, right = assign[1], assign[2], assign[3]
    stored = _memory(left)
    if stored:
        wrote = set(stored)
        read = _tokens(left) | _memory(right) | _tokens(right)
        if op:
            read |= stored
    else:
        key = re.sub(r"\[(\d+)\]", r"_\1", left)
        wrote = set(re.findall(r"[A-Za-z_]\w*", re.split(r"\.|->", key)[0]))
        read = _tokens(right) | _memory(right) | (_tokens(left) - wrote)
        if op or "." in left:
            read |= wrote
        if left.lstrip().startswith("*"):
            wrote |= {_BARRIER}
    if _calls(right):
        wrote |= {_BARRIER}
        read |= {_BARRIER}
    return wrote, read


def _overlap(a: set[str], b: set[str]) -> bool:
    if a & b:
        return True
    ma, mb = [x for x in a if x.startswith("@")], [x for x in b if x.startswith("@")]
    if (_BARRIER in a and mb) or (_BARRIER in b and ma):
        return True
    return any(x.startswith(y) or y.startswith(x) for x in ma for y in mb)


def conflict(first: str, second: str) -> bool:
    """Whether the two statements cannot swap: one writes what the other reads or writes."""
    wa, ra = effects(first)
    wb, rb = effects(second)
    return _overlap(wa, rb) or _overlap(wb, ra) or _overlap(wa, wb)


def statements(lines: Sequence[str]) -> list[str]:
    """The lines of a block body grouped into statements: balanced braces, ending in `;`, `}` or a blank line."""
    out: list[str] = []
    current = ""
    for line in lines:
        current = f"{current}\n{line}" if current else line
        if current.count("{") == current.count("}") and (line.strip().endswith((";", "}")) or not line.strip()):
            out.append(current)
            current = ""
    if current:
        out.append(current)
    return out


def moves(block: Sequence[str], window: int) -> list[list[str]]:
    """Every order of the block one allowed single-statement move away, the smallest moves first."""
    fixed = {i for i, s in enumerate(block) if _DECL.match(s) or not s.strip()}
    found: list[tuple[int, list[str]]] = []
    for i in range(len(block)):
        if i in fixed:
            continue
        for j in range(i - window, i + window + 1):
            if j == i or not 0 <= j < len(block):
                continue
            lo, hi = (j, i) if j < i else (i + 1, j + 1)
            if any(k in fixed for k in range(lo, hi)):
                continue
            if any(conflict(block[i], block[k]) for k in range(lo, hi) if k != i):
                continue
            moved = list(block)
            moved.insert(j, moved.pop(i))
            found.append((abs(j - i), moved))
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for _, candidate in sorted(found, key=lambda x: x[0]):
        if tuple(candidate) not in seen:
            seen.add(tuple(candidate))
            out.append(candidate)
    return out


def body(lines: Sequence[str], name: str) -> tuple[int, int]:
    """(open, close) line indexes of the function's definition: the `{` line and the first `}` in column 0 after it."""
    for k, line in enumerate(lines):
        if re.match(rf"^\S.*\b{re.escape(name)}\(", line) and line.rstrip().endswith("{"):
            for end in range(k, len(lines)):
                if lines[end] == "}":
                    return k, end
    raise Refusal(
        Finding(
            "order.function",
            reason=f"no definition of {name}() found: its `{{` must end the signature line and its `}}` stand alone "
            "in column 0",
            action="pass the candidate file that defines the function",
        )
    )


def blocks(lines: Sequence[str], opened: int, closed: int) -> list[tuple[int, int]]:
    """(open, close) lines of each nested block of at least _MIN_BLOCK lines, innermost first."""
    out: list[tuple[int, int]] = []
    stack: list[int] = []
    for k in range(opened + 1, closed):
        for _ in range(lines[k].count("}")):
            if stack:
                out.append((stack.pop(), k))
        stack.extend([k] * lines[k].count("{"))
    return [(a, b) for a, b in out if b - a >= _MIN_BLOCK]


def _job(item: tuple[Config, Json]) -> Json | None:
    config, params = item
    try:
        got = compare.probe_command(config, params)
    except Refusal:
        return None
    return {"score": got["score"], "exact": bool(got["exact"]), "words": got["words"]}


def order_command(config: Config, params: Json) -> Json:
    """`unbake order FILE --function FN`: climb the statement order of FN toward the ROM; write the best source."""
    with effort.stage("order"):
        source = Path(params["file"])
        if not source.is_file():
            raise Refusal(Finding("order.file", reason=f"{source} does not exist", action="pass the candidate C file"))
        name = params.get("function") or ""
        if not name:
            raise Refusal(Finding("order.function", reason="order needs --function NAME", action="pass --function"))
        window, rounds, budget = (
            int(params.get("window") or 6),
            int(params.get("rounds") or 3),
            int(params.get("budget") or 150),
        )
        out = Path(params["out"]) if params.get("out") else source.with_suffix(".ordered.c")
        lines = source.read_text(encoding="utf-8").split("\n")
        opened, closed = body(lines, name)
        levels: list[tuple[int, int]] = [(opened, closed), *blocks(lines, opened, closed)]
        used, steps = [0], list[Json]()
        with store.work(config) as work:
            counter = itertools.count()

            def probe(texts: Sequence[str]) -> list[Json | None]:
                items = []
                for text in texts:
                    path = work / f"order-{next(counter)}.c"
                    path.write_text(text, encoding="utf-8")
                    items.append(
                        (
                            config,
                            {
                                **{k: params[k] for k in ("version", "toolchain", "flag", "omit_flag") if k in params},
                                "file": path,
                                "function": name,
                                "extras": [],
                                "lines": 0,
                            },
                        )
                    )
                used[0] += len(items)
                return pool.map(config, "order.probe", _job, items)

            first = probe(["\n".join(lines)])[0]
            if first is None:
                raise Refusal(
                    Finding(
                        "order.probe",
                        reason="the file does not probe as it is",
                        action=f"unbake probe {source} --function {name}",
                    )
                )
            best, base = list(lines), first
            for a, b in levels:
                if base["exact"]:
                    break
                for _ in range(rounds):
                    head, tail = best[: a + 1], best[b:]
                    block = statements(best[a + 1 : b])
                    room = budget - used[0]
                    if room <= 0:
                        break
                    candidates = moves(block, window)[:room]
                    texts = ["\n".join(head + c + tail) for c in candidates]
                    scored = probe(texts) if texts else []
                    pick = None
                    for text, cand, got in zip(texts, candidates, scored, strict=True):
                        if (
                            got
                            and (got["exact"] or got["score"] > base["score"])
                            and (pick is None or (got["exact"], got["score"]) > (pick[2]["exact"], pick[2]["score"]))
                        ):
                            pick = (text, cand, got)
                    if pick is None:
                        break
                    base, best = pick[2], pick[0].split("\n")
                    steps.append({"block": [a + 1, b + 1], "score": base["score"], "exact": base["exact"]})
                    if base["exact"]:
                        break
                    # statements moved at this level shift later line numbers only when the block grows: it does not
            improved = bool(steps)
            if improved:
                out.write_text("\n".join(best), encoding="utf-8")
        return {
            "function": name,
            "start": first,
            "best": base,
            "improved": improved,
            "exact": base["exact"],
            "steps": steps,
            "probes": used[0],
            "stopped": "exact" if base["exact"] else "budget" if used[0] >= budget else "no move helps",
            "out": str(out) if improved else "",
            "next": f"unbake probe {out} --function {name}" if improved else "",
        }
