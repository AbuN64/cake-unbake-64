"""`unbake deadcode PATH`: a report (never a gate) of the statements the compiler deletes as dead.

cc1 deletes a statement whose value nothing reads (a store to a local never used, an expression with no effect, code
after a return) and the object shows no trace of it. The C then holds a statement the original author may never have
written, or one a decomp wrote to steer the allocator. The lint compiles the unit with the compiler's RTL dumps and,
per function, follows each insn from the expander's dump (`.rtl`, which carries the source line of every insn)
through the passes to the register-allocation dump (`.greg`). A source line whose real insns were ALL deleted, each by
a pass that removes dead or redundant code (jump, cse, cse2, loop, flow), is listed; so is none that only copied one
register to another (copy propagation), and none any of whose insns the combiner merged into another (that code
still has an effect). The report lists lines to look at; nothing is refused.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from unbake import native, store
from unbake.contracts import Config, Finding, Json, Refusal

PASSES = ("rtl", "jump", "cse", "loop", "cse2", "flow", "combine", "sched", "greg")  # the order cc1 runs them in
_DEAD = frozenset({"jump", "cse", "cse2", "flow", "loop"})  # passes whose deletions mean "nothing needed this"
_FUNCTION = re.compile(r"^;; Function (\S+)", re.M)
_ITEM = re.compile(r"^\((insn|call_insn|jump_insn|note|code_label|barrier) (\d+) ", re.M)
_LINE = re.compile(r'^\(note (\d+) \d+ \d+ \("[^"]*"\) (\d+)\)', re.M)


def functions(dump: str) -> dict[str, str]:
    """The dump split per function: name -> its text."""
    marks = list(_FUNCTION.finditer(dump))
    return {
        m[1]: dump[m.start() : marks[i + 1].start() if i + 1 < len(marks) else len(dump)] for i, m in enumerate(marks)
    }


def insns(text: str) -> dict[int, tuple[int, bool]]:
    """uid -> (source line, whether it only copies a register) for each real insn (a set or call, not a jump or label)
    of one function's dump; the line is the nearest line note before it, 0 when none."""
    out: dict[int, tuple[int, bool]] = {}
    line = 0
    marks = list(re.finditer(r"^\((insn|call_insn|note) (\d+) ", text, re.M))
    for k, row in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        body = text[row.start() : end]
        if row[1] == "note":
            found = re.match(r'\(note \d+ \d+ \d+ \("[^"]*"\) (\d+)\)', body)
            line = int(found[1]) if found else line
        else:
            copy = re.match(
                r"\(insn \d+ \d+ \d+ \(set \(reg[^()]*(?:\([^()]*\))?[^()]*\)\s+\(reg[^()]*(?:\([^()]*\))?[^()]*\)\)",
                body,
            )
            out[int(row[2])] = (line, bool(copy))
    return out


def alive(text: str) -> set[int]:
    """The uids that are real insns in one function's dump."""
    return {int(m[2]) for m in _ITEM.finditer(text) if m[1] in ("insn", "call_insn")}


def dead_lines(dumps: Mapping[str, str], name: str) -> list[Json]:
    """The source lines of `name` whose every insn was deleted by a dead-code pass: [{line, insns, passes}]."""
    per = {p: functions(dumps[p]).get(name, "") for p in PASSES if p in dumps}
    if not per.get("rtl") or "greg" not in per:
        return []
    born = insns(per["rtl"])
    gone: dict[int, str] = {}  # uid -> the pass that deleted it
    before = set(born)
    for p in PASSES[1:]:
        if p not in per:
            continue
        if not per[p]:
            continue
        now = alive(per[p])
        for uid in before - now:
            gone.setdefault(uid, p)
        before = before & now
    by_line: dict[int, list[int]] = defaultdict(list)
    for uid, (line, _copy) in born.items():
        if line:
            by_line[line].append(uid)
    out: list[Json] = []
    head = min((ln for ln, _ in born.values() if ln), default=0)
    for line, uids in sorted(by_line.items()):
        if line == head or any(u not in gone for u in uids):
            continue  # the signature line moves its parameters; a surviving insn means the line has an effect
        passes = sorted({gone[u] for u in uids})
        if all(born[u][1] for u in uids):
            continue  # only register copies: copy propagation, not a statement the compiler found dead
        if all(p in _DEAD for p in passes):
            out.append({"line": line, "insns": len(uids), "passes": passes})
    return out


def read_dumps(folder: Path, stem: str) -> dict[str, str]:
    return {p: f.read_text(errors="replace") for p in PASSES if (f := folder / f"{stem}.i.{p}").is_file()}


def deadcode_command(config: Config, params: Json) -> Json:
    """`unbake deadcode PATH`: list the statements cc1 deletes as dead in each function of PATH (a tree source)."""
    path = str(params["file"])
    root = config.project.root
    source = Path(path) if Path(path).is_absolute() else root / path
    if not source.is_file():
        raise Refusal(Finding("deadcode.file", reason=f"{path} does not exist", action="pass a tree source file"))
    rel = str(source.resolve().relative_to(root.resolve()))
    version = params.get("version") or config.project.names_from
    text = source.read_text(encoding="utf-8")
    wanted = params.get("function")
    snapshot, unit, recipe, _ = native.candidate(
        config, version, rel, text, {"add": ["-da"], "omit": []}, {}, [wanted] if wanted else None
    )
    with store.work(config) as work:
        native.dumps(snapshot, unit, version, recipe, work / "dumps")
        dumps = read_dumps(work / "dumps", Path(rel).stem)
        lines = text.split("\n")
        found: list[Json] = []
        for name in functions(dumps.get("rtl", "")):
            if wanted and name != wanted:
                continue
            for row in dead_lines(dumps, name):
                shown = lines[row["line"] - 1].strip() if 0 < row["line"] <= len(lines) else ""
                found.append({"function": name, **row, "text": shown[:120]})
    return {
        "file": rel,
        "version": version,
        "dead": found,
        "functions": len({r["function"] for r in found}),
        "next": "read each line: a statement the compiler deletes is one the original may not have had",
    }
