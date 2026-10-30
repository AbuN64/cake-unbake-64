"""Human findings, command feedback, and terminal progress."""

from __future__ import annotations

import shlex
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from typing import TextIO

from unbake import config, effort
from unbake.contracts import Finding, Json, StageRecord

_stream: TextIO | None = None
_tty = False
_live = False
_SLOW_START = 10.0  # a stage the previous run spent this long in gets a start line
_DEPTH = 3  # stages this shallow report progress lines


def attach(stream: TextIO, tty: bool) -> None:
    global _stream, _tty
    _stream, _tty = stream, tty
    effort.listen(_event)


def _clear() -> None:
    global _live
    if _live and _stream is not None:
        _stream.write("\r\x1b[K")
        _stream.flush()
    _live = False


def _event(event: str, body: Json) -> None:
    global _live
    if _stream is None:
        return
    path = body.get("path", [])
    if path[:1] == ["job"]:  # a pool worker's job runs once per item: its parent stage reports the totals
        return
    if event == "say":
        _clear()
        _stream.write(f"{body['text']}\n")
        _stream.flush()
        return
    if event == "close" and len(path) <= _DEPTH:
        _clear()
        label = " > ".join(path)
        _stream.write(f"{label} {body['wall']:.1f}s {body['cores']:.2f} cores {body['jobs']} jobs\n")
        _stream.flush()
        return
    if event == "open" and len(path) <= _DEPTH:
        previous = body.get("expected")
        if (previous is None and len(path) <= 2) or (previous is not None and previous >= _SLOW_START):
            _clear()
            note = f" (last run {previous:.0f}s)" if previous is not None else ""
            _stream.write(f"{' > '.join(path)} started{note}\n")
            _stream.flush()
    if not _tty:
        return
    if event == "open":
        text = " > ".join(body["path"])
    elif event == "progress":
        text = f"{body['name']} {body['done']}/{body['total']}"
    else:
        return
    width = shutil.get_terminal_size().columns - 1
    _stream.write("\r\x1b[K" + text[:width])
    _stream.flush()
    _live = True


def standard(f: Finding) -> Json:
    """The one finding shape every answer carries: Code, Severity (Refuse|Warn), File, Line, Message, NextStep."""
    detail = {
        "Unit": f.unit,
        "Versions": list(f.versions),
        "Missing": list(f.missing),
        "Origin": f.origin and asdict(f.origin),
        "Symptoms": f.symptoms,
    }
    return {
        "Code": f.key,
        "Severity": f.severity,
        "File": f.path,
        "Line": f.line,
        "Message": f.reason,
        "NextStep": f.action or config.sentence(f.key),
        **({"Detail": {k: v for k, v in detail.items() if v}} if any(detail.values()) else {}),
    }


def tally(findings: Sequence[Finding]) -> Json:
    """Counts per Severity and per Code and Severity: the summary line of every answer."""
    codes: dict[str, dict[str, int]] = {}
    for f in findings:
        row = codes.setdefault(f.key, {})
        row[f.severity] = row.get(f.severity, 0) + 1
    return {"Refuse": sum(f.blocking for f in findings), "Warn": sum(not f.blocking for f in findings), "Codes": codes}


def tally_line(counts: Json) -> str:
    codes = ", ".join(f"{code} {n} {severity}" for code, row in counts["Codes"].items() for severity, n in row.items())
    return f"summary: {counts['Refuse']} Refuse, {counts['Warn']} Warn" + (f" ({codes})" if codes else "")


def finding(finding: Finding) -> str:
    f = finding
    lines = [f"{f.severity}: {config.sentence(f.key)}."]
    if f.reason:
        lines.append(f"  because {f.reason}")
    if f.path:
        lines.append(f"  in {f.path}" + (f":{f.line}" if f.line > 0 else ""))
    if f.unit:
        lines.append(f"  item {f.unit}")
    if f.versions:
        lines.append("  versions " + ", ".join(f.versions))
    lines.extend(f"  missing {missing}" for missing in f.missing)
    if f.action:
        lines.append(f"  next: {f.action}")
    return "\n".join(lines)


def summary(records: Sequence[StageRecord]) -> list[str]:
    root = next((r for r in reversed(records) if r.kind == "command"), None)
    if root is None:
        return []
    lines = [
        f"took {root.wall_seconds:.2f}s, {root.cores:.2f} cores "
        f"(parent {root.parent_cpu_seconds:.2f}s, workers {root.worker_cpu_seconds:.2f}s, "
        f"native {root.native_cpu_seconds:.2f}s cpu)"
    ]
    waits = [
        f"{seconds:.1f}s for {what}"
        for what, seconds in (("slots", root.slot_wait_seconds), ("locks", root.lock_wait_seconds))
        if seconds >= 0.05
    ]
    if waits:
        lines.append("wait " + ", ".join(waits))
    memory = []
    for label, field in (("parent", "parent"), ("workers", "worker")):
        key = f"resources.memory_{field}_bytes"
        origin = root.memory_limits.get(key)
        memory.append(
            f"{label} peak {getattr(root, field + '_rss_peak_bytes') / 1e6:.1f} MB "
            f"of {root.memory_limit_values.get(key, 0) / 1e6:.1f} MB "
            f"({origin.file if origin else 'unbound'})"
        )
    lines.append("memory: " + ", ".join(memory))
    lines.extend(f"cache: {kind} {hits} warm / {misses} cold" for kind, (hits, misses) in sorted(root.cache.items()))
    lines.append("slowest:")
    lines.extend(
        f"  {' > '.join(r.path)}  {r.wall_seconds:.2f}s  {r.cores:.2f} cores  {r.execution}  items {r.items} "
        f"jobs {r.jobs} workers {r.workers_used}/{r.workers_admitted}"
        + (f" held {r.slot_seconds / r.wall_seconds:.1f}" if r.slot_seconds and r.wall_seconds else "")
        for r in sorted(records, key=lambda r: r.wall_seconds, reverse=True)[:10]
    )
    lines.extend(
        f"slow: {config.sentence(f.key)} {f.reason}"
        for record in records
        for f in record.findings
        if f.key.startswith("budget.")
    )
    return lines


def _written(command: str, result: Json | None) -> list[str]:
    """The files a command wrote into the working tree, uncommitted, and the exact git commands that commit them."""
    result = result or {}
    drain = result.get("drain") or result
    groups = [(row["files"], row["message"]) for row in drain.get("landed", ())]
    if "files" in result and "message" in result:  # any command that wrote files names them and its commit message
        groups.append((result["files"], result["message"]))
    lines = []
    for files, message in groups:
        if files:
            lines += [
                "Written, not committed. To commit:",
                "  git add -A -- " + " ".join(map(shlex.quote, files)),
                "  git commit -m " + shlex.quote(message),
            ]
    return lines


def _pseudos(rows: Sequence[Json]) -> list[str]:
    """The allocation table (score --alloc): every pseudo with its priority and registers, MISMATCH rows explained."""
    if not rows:
        return []
    out = [
        f"{'pseudo':>6} {'source name':<14} {'refs':>4} {'live':>5} {'prio':>7} {'alloc':<6} {'rank':>4} "
        f"{'got':<5} {'ROM':<9} flag"
    ]
    for r in rows:
        out.append(
            f"{r['pseudo']:>6} {r['name'][:14]:<14} {r['refs']:>4} {r['length']:>5} {r['priority']:>7} "
            f"{r['allocator']:<6} {'-' if r['rank'] is None else r['rank']:>4} {r['register']:<5} "
            f"{r['target_register'] or '?':<9} " + ("MISMATCH" if r["mismatch"] else "")
        )
        if r["mismatch"]:
            out.append(f"       {r['source']}" if r["source"] else "       (no source line)")
            out.append(f"       why: {r['why']}")
            for t in r.get("took", ()):
                out.append(
                    f"       conflicts with {t['name']} (pseudo {t['pseudo']}, {t['allocator']}"
                    f"{'' if t['rank'] is None else ' rank ' + str(t['rank'])}, prio {t['priority']}"
                    f"{', placed first' if t['earlier'] else ''}) which holds the ROM register here"
                )
            d = r.get("delta")
            if d:
                out.append(
                    f"       delta: needs priority +{d['needs']} to pass {d['above']}: "
                    f"{d['more_refs']} more weighted refs, or {d['shorter_by']} fewer live insns"
                )
            out.extend(
                f"       the ROM gives {r['register']} to pseudo {g.split('->')[1]}" for g in r.get("rom_gives", ())
            )
    return out


def _score(result: Json) -> list[str]:
    """What a score found besides its diff: where the words part, the source rules, and the registers' priorities."""
    shift, lines = result.get("shift") or {}, []
    if shift.get("first_difference"):
        d, a = shift["first_difference"], shift["aligned"]
        lines.append(
            f"first difference at word {d['index']}: built {d['built']}, target {d['target']}; aligned "
            f"{a['exact']} exact, {a['register_blind']} register-blind of {result['words']['target']}"
        )
    lines.extend(x for x in (shift.get("frame"), shift.get("folded")) if x)
    lines.extend(f"source rule {f['key']} at line {f['line']}: {f['reason']}" for f in result.get("source", ()))
    lines.extend(_pseudos(result.get("pseudos", ())))
    for row in result.get("alloc", ()):
        won, lost, flip = row["candidate"], row["target"], row["flip"]
        lines.append(
            f"{row['register']}: {won['name']} (refs {won['refs']}, length {won['length']}, priority "
            f"{won['priority']}) takes it, the target gives it to {lost['name']} (refs {lost['refs']}, length "
            f"{lost['length']}, priority {lost['priority']})"
            + (f"; flips at +{flip['more_refs']} refs or -{flip['shorter_by']} insns live" if flip else "")
        )
        lines.extend([f"  edit: {row['edit']}"] if row.get("edit") else [])
        for side in (won, lost):  # where each pseudo's weighted references come from: line:refs*depth
            if side.get("lines"):
                spread = " ".join(
                    f"{r['line']}:{r['refs']}" + f"*{r['depth']}" * (r["depth"] > 1) for r in side["lines"]
                )
                lines.append(f"  {side['name']} refs by line: {spread}")
    for turn in result.get("rotations", ()):
        lines.append(f"{turn['kind']} of {' '.join(turn['registers'])}: {'; '.join(turn['moves'])}")
        lines.append("  priority order in the candidate: " + ", ".join(turn["priority_order"]))
        lines.extend(f"  the target's register order needs {need}" for need in turn["needs"])
        for i, edit in enumerate(turn["rewrites"], 1):
            lines.append(f"  rewrite {i} [{edit['kind']}, score {edit['score']}]: {edit['text']}")
            lines.extend([f"    precedent {edit['because']} (+{edit['gain']})"] if edit.get("because") else [])
    lines.extend(f"dump {path}" for path in result.get("dumps", ()))
    for row in result.get("sched", ()):
        way = "earlier" if row["words"] > 0 else "later"
        lines.append(f"line {row['line']}: {row['asm']} sits {abs(row['words'])} words {way} than in the target")
        for name in ("sched", "sched2"):
            if name in row:
                lines.append(f"  {name}: priority {row[name]['priority']}, input position {row[name]['luid']}")
        if row.get("hint"):
            lines.append("  " + row["hint"])
    return lines


def finish(command: str, result: Json | None, findings: Sequence[Finding]) -> None:
    _clear()
    if _stream is None:
        return
    lines = [finding(f) for f in sorted(findings, key=lambda f: not f.blocking)]
    if findings:
        lines.append(tally_line(tally(findings)))
    lines.extend(_written(command, result))
    if command in ("score", "probe"):  # result fields belong to their command's contract (upgrade.source is a URL)
        lines.extend(_score(result or {}))
    lines.extend(summary(effort.closed()))
    for line in lines:
        _stream.write(line + "\n")
    _stream.flush()
