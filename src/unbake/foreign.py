"""Whether a NON_MATCHING body is the body of the function it stands for. Two ways it is not: it defines a function
that no symbol of the version names (so it is built into nothing), or, linked at its function's address, it agrees
with the function's own bytes less than any attempt at them would."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from unbake import config as configuration
from unbake import tree
from unbake.contracts import Config, Finding

KEY = "source.non_matching_foreign_body"
# The scores of the 101 measurable NON_MATCHING functions of one real 5-version tree (best version of each) fall in
# a gap: three bodies under 0.03 and one at 0.07, then 0.11 and up with no break. A body that is the function's own
# attempt keeps more than a tenth of its bytes in order; 0.10 splits the gap and leaves every real attempt in. A score
# of exactly 0 is a body that does not compile (158 of the 257 bodies of that tree), which is another finding's to name.
FLOOR = 0.10


def _line(text: str, name: str) -> int:
    """The 1-based line where the NON_MATCHING body of `name` is defined in `text`, 0 when none is."""
    lines = text.splitlines(keepends=True)
    head = re.compile(rf"^[^\n;]*\b{re.escape(name)}\s*\([^;]*$")
    for opened, other, closed in tree.non_matching_blocks(lines):
        for number in range(opened + 1, other if other >= 0 else closed):
            if head.match(lines[number]) and name in tree.defined("".join(lines[opened + 1 : closed])):
                return number + 1
    return 0


def _finding(config: Config, path: str, name: str, reason: str, versions: Sequence[str]) -> Finding:
    text = (config.project.root / path).read_text(encoding="utf-8")
    return Finding(
        KEY,
        f"{configuration.sentence(KEY)}: {reason}",
        path=path,
        line=_line(text, name),
        unit=name,
        versions=tuple(versions),
        action="write the body this function has, or keep the stand-in",
    )


def findings(config: Config, modules: Sequence[Any], scores: Mapping[str, Mapping[str, float]]) -> tuple[Finding, ...]:
    """One finding per NON_MATCHING body of `modules` that is not its function's, given the score of each body per
    version (rescore.scores)."""
    found: list[Finding] = []
    for module in modules:
        names: dict[str, list[str]] = {}
        for version, lost in module.unnamed.items():
            for name in lost:
                names.setdefault(name, []).append(version)
        for name, versions in names.items():
            found.append(
                _finding(config, module.path, name, f"{name} is not a symbol of {', '.join(versions)}", versions)
            )
        for name in module.fuzzy:
            held = scores.get(name, {})
            if held and 0 < max(held.values()) < FLOOR:  # 0 is a body that does not build, which scores nothing
                best = max(held, key=lambda v: held[v])
                found.append(
                    _finding(config, module.path, name, f"{name} scores {held[best]:.0%} against its own bytes", [best])
                )
    return tuple(found)
