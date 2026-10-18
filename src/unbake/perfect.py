"""The perfect metric: a module (one source file) is perfect when no version that links exact still holds a stand-in
for any of its names, so the file is the original object's source in every version proven byte-exact."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from typing import Any

from unbake import effort
from unbake.contracts import Json


def census(tree: Any, credits: Sequence[Any], exact: Collection[str]) -> tuple[dict[str, Json], frozenset[str]]:
    """(per version the perfect census, the perfect modules). `credits` are report.credits rows; `exact` are the
    versions whose link matched the ROM. Memoised by the tree digest."""
    return effort.memo(("perfect.census", tree.digest), lambda: _census(tree, credits, frozenset(exact)))


def _census(tree: Any, credits: Sequence[Any], exact: frozenset[str]) -> tuple[dict[str, Json], frozenset[str]]:
    with effort.stage("perfect.census"):
        perfect = frozenset(m.path for m in tree.modules if not any(m.stand_ins.get(v) for v in exact))
        totals: dict[str, Counter[str]] = {}
        for row in credits:
            counter = totals.setdefault(row.version, Counter())
            size = row.code + row.data
            counter["total"] += size
            if row.member.group in perfect:
                counter["bytes"] += size
        held = {v: {m.path for m in tree.modules if m.owns.get(v)} for v in set(totals) | set(exact)}
        out: dict[str, Json] = {
            v: {
                "bytes": totals.get(v, Counter())["bytes"],
                "total": totals.get(v, Counter())["total"],
                "modules": len(held[v]),
                "perfect_modules": len(held[v] & perfect),
            }
            for v in sorted(held)
        }
        return out, perfect
