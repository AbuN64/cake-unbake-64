"""Human findings, command feedback, and terminal progress."""

from __future__ import annotations

import shlex
import shutil
from collections.abc import Sequence
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


def finding(finding: Finding) -> str:
    f = finding
    lines = [f"{'Refused' if f.blocking else 'Debt'}: {config.sentence(f.key)}."]
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
    if command in ("setup", "move") and "files" in result:
        groups.append((result["files"], result.get("message", "setup")))
    lines = []
    for files, message in groups:
        if files:
            lines += [
                "Written, not committed. To commit:",
                "  git add -A -- " + " ".join(map(shlex.quote, files)),
                "  git commit -m " + shlex.quote(message),
            ]
    return lines


def _score(result: Json) -> list[str]:
    """What a score found besides its diff: where the words part, the source rules, and the registers' priorities."""
    shift, lines = result.get("shift") or {}, []
    if shift.get("first_difference"):
        d, a = shift["first_difference"], shift["aligned"]
        lines.append(
            f"first difference at word {d['index']}: built {d['built']}, target {d['target']}; aligned "
            f"{a['exact']} exact, {a['register_blind']} register-blind of {result['words']['target']}"
        )
    lines.extend(f"source rule {f['key']} at line {f['line']}: {f['reason']}" for f in result.get("source", ()))
    for row in result.get("alloc", ()):
        won, lost, flip = row["candidate"], row["target"], row["flip"]
        lines.append(
            f"{row['register']}: {won['name']} (refs {won['refs']}, length {won['length']}, priority "
            f"{won['priority']}) takes it, the target gives it to {lost['name']} (refs {lost['refs']}, length "
            f"{lost['length']}, priority {lost['priority']})"
            + (f"; flips at +{flip['more_refs']} refs or -{flip['shorter_by']} insns live" if flip else "")
        )
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
    lines.extend(_written(command, result))
    if command == "score":  # result fields belong to their command's contract (upgrade.source is a URL)
        lines.extend(_score(result or {}))
    lines.extend(summary(effort.closed()))
    for line in lines:
        _stream.write(line + "\n")
    _stream.flush()
