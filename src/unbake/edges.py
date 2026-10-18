"""Object edges from the ROM alone. GNU as gives every object's .text a 16-byte alignment and pads its size to 16, so
the class of each adjacent pair of functions in one version proves where objects may start:

- pad: the next start is 16-aligned and zero padding sits before it (a gap, or zero words after the delay slot): a cut
- unal: the next start is not 16-aligned and nothing sits between: no object starts here, a join
- al0: 16-aligned with no padding: open (about one real object end in four looks like this)
- seg / gap: a segment change, or any other gap: a cut

Pairs are matched by name across versions: a cut class in any version proves a cut, an unal in any version proves a
join, both together are a contradiction. The .data section is classified by the same rule (padding there is the
zero fill between objects) and only measured, never used as proof."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from pathlib import Path

from unbake import effort, pool
from unbake.contracts import Json, Snapshot, digest

CUT = frozenset({"pad", "seg", "gap"})

Row = tuple[str, str, int, int, int, int]  # name, section, rom_start, rom_end, vram, segment index
Pair = tuple[str, str, str]  # left, right, class


def classify(gap: int, vram: int, same_segment: bool) -> str:
    """The class of the edge before a function starting at `vram`, `gap` padding bytes after the previous one."""
    if not same_segment:
        return "seg"
    if vram % 16:
        return "unal" if gap == 0 else "gap"
    return "al0" if gap == 0 else "pad" if gap < 16 else "gap"


def verdict(kinds: Collection[str]) -> str:
    """The cross-version verdict of one pair from the classes the versions holding it give it."""
    cut, join = bool(CUT.intersection(kinds)), "unal" in kinds
    return "contradiction" if cut and join else "cut" if cut else "join" if join else "open"


def _trailing_zero_words(blob: bytes) -> int:
    return (len(blob) - len(blob.rstrip(b"\0"))) // 4


def _version_job(item: tuple[str, str, tuple[Row, ...]]) -> tuple[tuple[Pair, ...], tuple[Pair, ...], str]:
    """(.text pairs, .data pairs, first function) of one version, in ROM order."""
    _sha, rom_path, rows = item
    rom = Path(rom_path).read_bytes()
    text = [r for r in rows if r[1] == ".text"]
    pairs: list[Pair] = []
    for (name, _, start, end, _, seg), (name2, _, start2, _, vram2, seg2) in pairwise(text):
        # a function ends jr ra and its delay slot: one trailing zero word is that slot, more are padding
        gap = start2 - end + 4 * max(_trailing_zero_words(rom[start:end]) - 1, 0)
        pairs.append((name, name2, classify(gap, vram2, seg == seg2)))
    data: list[Pair] = []
    left: Row | None = None  # the last real .data member of the current run
    for row in rows:
        if row[1] != ".data":
            left = None  # a run of .data ends at any other section
        else:
            if left is not None:
                data.append((left[0], row[0], classify(row[2] - left[3], row[4], left[5] == row[5])))
            left = row
    return tuple(pairs), tuple(data), text[0][0] if text else ""


def _version_key(item: tuple[str, str, tuple[Row, ...]]) -> str:
    """The ROM's sha and the rows: every input, so a copy of the tree elsewhere reads the same entry."""
    return digest((item[0], item[2]))


_version_key.stable = True  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Edges:
    pairs: Mapping[str, tuple[Pair, ...]]  # version -> .text pairs in ROM order
    data: Mapping[str, tuple[Pair, ...]]  # version -> .data pairs in ROM order (measured only)
    first: Mapping[str, str]  # version -> its first function: a segment start

    @cached_property
    def classes(self) -> dict[tuple[str, str], dict[str, str]]:
        """Per pair of names, the class every version holding it gives it."""
        out: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        for v, rows in self.pairs.items():
            for left, right, c in rows:
                out[left, right][v] = c
        return dict(out)

    @cached_property
    def verdicts(self) -> dict[tuple[str, str], str]:
        return {pair: verdict(seen.values()) for pair, seen in self.classes.items()}

    @cached_property
    def before(self) -> dict[str, dict[str, str]]:
        """Per version, each function's predecessor."""
        return {v: {r: left for left, r, _ in rows} for v, rows in self.pairs.items()}

    @cached_property
    def after(self) -> dict[str, dict[str, str]]:
        """Per version, each function's successor."""
        return {v: {left: r for left, r, _ in rows} for v, rows in self.pairs.items()}

    def member(self, name: str, reference: str) -> tuple[dict[str, str], str]:
        """(class of the edge before `name` per version holding a pair for it, the verdict of that edge in
        `reference`). A version's first function starts a segment."""
        classes = {v: self.classes[self.before[v][name], name][v] for v in self.pairs if name in self.before[v]}
        classes.update({v: "seg" for v, first in self.first.items() if first == name})
        left = self.before.get(reference, {}).get(name)
        return classes, self.verdicts[left, name] if left is not None else "cut"

    def census(self, version: str) -> dict[str, int]:
        """The pairs of one version by verdict, `_other` where only another version proves it."""
        out: Counter[str] = Counter()
        for left, right, own in self.pairs.get(version, ()):
            kind = self.verdicts[left, right]
            if (kind == "cut" and own not in CUT) or (kind == "join" and own != "unal"):
                kind += "_other"
            out[kind] += 1
        cuts = out["cut"] + out["cut_other"]
        keys = ("cut", "cut_other", "join", "join_other", "open", "contradiction")
        return {
            "pairs": sum(out.values()),
            **{k: out[k] for k in keys},
            "objects_min": cuts + 1,
            "objects_estimate": cuts + 1 + round(cuts / 3),  # about one real object end in four looks al0
        }

    def contradictions(self) -> list[Json]:
        return [
            {"left": left, "right": right, "classes": dict(sorted(self.classes[left, right].items()))}
            for (left, right), kind in sorted(self.verdicts.items())
            if kind == "contradiction"
        ]

    def data_census(self, module_of: Mapping[str, str]) -> dict[str, dict[str, dict[str, int]]]:
        """Per version and .data class: pairs, and the pairs whose modules differ. The 16-byte rule holds for
        .data when no module change sits at unal."""
        out: dict[str, dict[str, dict[str, int]]] = {}
        for v, rows in self.data.items():
            seen: dict[str, Counter[str]] = defaultdict(Counter)
            for left, right, c in rows:
                seen[c]["n"] += 1
                seen[c]["module_cut"] += module_of.get(left, "") != module_of.get(right, "")
            out[v] = {c: {"n": n["n"], "module_cut": n["module_cut"]} for c, n in sorted(seen.items())}
        return out


def _rows(snapshot: Snapshot) -> dict[str, list[Row]]:
    """Per version its function .text and .data symbols in ROM order, each with its segment's index. A symbol ends where
    the next one starts, or where its segment does; a name the version's code set holds is a function."""
    rows: dict[str, list[Row]] = {}
    for v, ver in snapshot.versions.items():
        found: list[Row] = []
        for index, (_, start, stop, vram) in enumerate(sorted(ver.segments, key=lambda s: s[1])):
            end = vram + stop - start
            placed = sorted((a, n) for n, a in ver.symbols.items() if vram <= a < end)
            for (address, name), after in zip(placed, [*(a for a, _ in placed[1:]), end], strict=True):
                section = ".text" if name in ver.code else ".data"
                found.append((name, section, start + address - vram, start + after - vram, address, index))
        rows[v] = sorted(found, key=lambda r: (r[2], r[0]))
    return rows


def of(snapshot: Snapshot) -> Edges:
    """The edges of a snapshot, once per digest; each version is one pool item cached by its ROM and rows."""

    def build() -> Edges:
        with effort.stage("edges.classes"):
            rows = _rows(snapshot)
            ids = sorted(rows)
            items = [(snapshot.versions[v].rom_sha256, str(snapshot.versions[v].rom), tuple(rows[v])) for v in ids]
            done = pool.map(snapshot.config, "edges.classes", _version_job, items, _version_key)
            return Edges(
                {v: d[0] for v, d in zip(ids, done, strict=True)},
                {v: d[1] for v, d in zip(ids, done, strict=True)},
                {v: d[2] for v, d in zip(ids, done, strict=True) if d[2]},
            )

    return effort.memo(("edges", snapshot.digest), build)
