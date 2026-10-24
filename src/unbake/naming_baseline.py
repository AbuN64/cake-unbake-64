"""Reviewed naming debt in project tool config; audit can only shrink it."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from unbake import config, store
from unbake.contracts import Finding, Json, Refusal

PATH = "config/symbol-name-baseline.json"


def read(root: Path) -> Json | None:
    path = root / PATH
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise Refusal(Finding("config.schema", str(error), path=PATH)) from error
    config.validate("symbol-name-baseline", document, PATH)
    return document


def by_file(findings: Sequence[Finding]) -> Json:
    rows: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for finding in findings:
        rows[finding.path][finding.reason].append(finding.line)
    return {
        path: [{"reason": reason, "lines": sorted(lines)} for reason, lines in sorted(reasons.items())]
        for path, reasons in sorted(rows.items())
    }


def split(findings: Sequence[Finding], baseline: Json | None) -> tuple[tuple[Finding, ...], tuple[Finding, ...]]:
    """Line shifts are free; a new reason, path or occurrence is a regression."""
    held = Counter(
        {
            (path, row["reason"]): len(row["lines"])
            for path, rows in (baseline or {}).get("files", {}).items()
            for row in rows
        }
    )
    blocking, debt = [], []
    for finding in sorted(findings, key=lambda f: (f.path, f.line, f.reason)):
        key = finding.path, finding.reason
        if held[key] > 0:
            debt.append(replace(finding, blocking=False))
            held[key] -= 1
        else:
            blocking.append(finding)
    return tuple(blocking), tuple(debt)


def shrink(root: Path, baseline: Json | None, debt: Sequence[Finding]) -> None:
    """Only called after a successful full audit while holding the project write lock."""
    if baseline is not None:
        store.write(root / PATH, (json.dumps({**baseline, "files": by_file(debt)}, indent=2) + "\n").encode())
