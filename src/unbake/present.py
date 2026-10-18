"""The one presentation file the README embeds: the ROM overview picture of the README version."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from unbake import effort, report


def overview_path(version: str) -> str:
    return f"docs/overview-{version}.svg"


# colours: q0 is no draft, q10 is exact, q1..q9 spread the drafts by score
_STYLE = (
    "text{font:6px monospace;fill:#6b675c}.h{fill:#c4472d}"
    ".q0{fill:#d9d4c7}.q1{fill:#e3e2f2}.q2{fill:#d6d6ef}.q3{fill:#c8c9ec}.q4{fill:#babce8}.q5{fill:#acafe4}"
    ".q6{fill:#9ea2df}.q7{fill:#9094d8}.q8{fill:#8186cf}.q9{fill:#7277c4}.q10{fill:#1b1a17}"
    "@media (prefers-color-scheme:dark){text{fill:#a8a293}"
    ".q0{fill:#4a463d}.q1{fill:#35375a}.q2{fill:#3c3f66}.q3{fill:#444772}.q4{fill:#4d517f}.q5{fill:#575b8c}"
    ".q6{fill:#62669a}.q7{fill:#6e72a8}.q8{fill:#7b7fb6}.q9{fill:#888cc4}.q10{fill:#d6d0bd}}"
)


# overview geometry: made to be shown 720 px wide, 64 KiB per row, one block per run of one status
OVERVIEW_WIDTH = 720
OVERVIEW_ROW = 0x10000
OVERVIEW_LEFT = 64
OVERVIEW_BAR = OVERVIEW_WIDTH - OVERVIEW_LEFT - 8
OVERVIEW_HEIGHT = 13
OVERVIEW_PITCH = 17
OVERVIEW_JOIN = 32  # functions this close count as touching: alignment padding does not split a run
FUZZY_HIGH = 0.5
_TONES = (
    ("q10", "exact"),
    ("q7", "fuzzy high"),
    ("q3", "fuzzy low"),
    ("q0", "none"),
    ("b2", "decoded"),
    ("b1", "identified"),
    ("p1", "platform"),
)
_LEGEND = (28, 72, 72, 28)  # block widths of the legend bar: none, fuzzy low, fuzzy high, exact
_BLOB_STYLE = ".b1{fill:#ecdcaa}.b2{fill:#b98b1d}@media (prefers-color-scheme:dark){.b1{fill:#5a4a1e}.b2{fill:#d1a63a}}"
_PLATFORM_STYLE = ".p1{fill:#6f8d96}"
_OVERVIEW_STYLE = (
    _STYLE.replace("text{font:6px", "text{font:10px").replace(".h{fill:#c4472d}", "") + _BLOB_STYLE + _PLATFORM_STYLE
)


def _num1(value: float) -> str:
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return text or "0"


def _tone(score: float, exact: bool) -> int:
    """0 exact, 1 fuzzy high (half or more), 2 fuzzy low, 3 none."""
    if exact:
        return 0
    return 3 if score <= 0 else 1 if score >= FUZZY_HIGH else 2


def overview(credit_rows: Iterable[Any], version: str, layers: Mapping[str, Any] | None = None) -> bytes:
    """The ROM of one version as a picture that reads at 720 px: rows of 64 KiB in ROM order (rows without code, data
    or platform left out), adjacent functions and data of one status merged into one block, decoded and identified
    blobs in their own tones, and a legend of the statuses."""
    blobs = [
        (a, b - a, tone) for tone, key in ((4, "decoded"), (5, "identified")) for a, b in (layers or {}).get(key, ())
    ]
    pieces = sorted(
        [
            (credit.member.rom, credit.code + credit.data, _tone(credit.score, credit.exact))
            for credit in credit_rows
            if credit.version == version and credit.code + credit.data > 0
        ]
        + blobs
        + [(a, b - a, 6) for a, b in (layers or {}).get("platform", ())]
    )
    runs: list[list[int]] = []  # [start, end, tone]
    for start, size, tone in pieces:
        if runs and runs[-1][2] == tone and start - runs[-1][1] <= OVERVIEW_JOIN:
            runs[-1][1] = max(runs[-1][1], start + size)
        else:
            runs.append([start, start + size, tone])
    rows: set[int] = set()
    drawn: list[list[tuple[float, int, float]]] = [
        [] for _ in _TONES
    ]  # per tone the outlines of its blocks: one path each keeps the file small
    scale = OVERVIEW_BAR / OVERVIEW_ROW
    for start, end, tone in runs:
        at = start
        while at < end:
            row = at // OVERVIEW_ROW
            until = min(end, (row + 1) * OVERVIEW_ROW)
            rows.add(row)
            drawn[tone].append((OVERVIEW_LEFT + (at - row * OVERVIEW_ROW) * scale, row, max((until - at) * scale, 1)))
            at = until
    order = {row: n for n, row in enumerate(sorted(rows))}
    lines = [
        f'<text x="0" y="{n * OVERVIEW_PITCH + OVERVIEW_HEIGHT - 3}">0x{row * OVERVIEW_ROW:06X}</text>'
        for row, n in order.items()
    ]
    for (cls, _), blocks in zip(_TONES, drawn, strict=True):
        if blocks:
            shapes = "".join(
                f"M{_num1(x)} {order[row] * OVERVIEW_PITCH}h{_num1(w)}v{OVERVIEW_HEIGHT}h-{_num1(w)}z"
                for x, row, w in blocks
            )
            lines.append(f'<path class="{cls}" d="{shapes}"/>')
    top = len(order) * OVERVIEW_PITCH + 6
    legend, x = [], OVERVIEW_LEFT
    for cls, width in zip(("q0", "q3", "q7", "q10"), _LEGEND, strict=True):  # none, fuzzy low, fuzzy high, exact
        legend.append(f'<rect class="{cls}" x="{x}" y="{top}" width="{width}" height="6"/>')
        x += width + 2
    end = x - 2
    label = top + 17
    legend += [
        f'<text x="{OVERVIEW_LEFT}" y="{label}">0 %</text>',
        f'<text x="{end}" y="{label}" text-anchor="end">100 %</text>',
    ]
    if blobs:  # blob swatches, right of the status bar
        x = end + 40
        for cls, name in (("b1", "blob identified"), ("b2", "blob decoded")):
            legend.append(f'<rect class="{cls}" x="{x}" y="{top}" width="28" height="6"/>')
            legend.append(f'<text x="{x + 32}" y="{top + 6}">{name}</text>')
            x += 150
    height = label + 4
    if (layers or {}).get("platform"):
        legend.append(f'<rect class="p1" x="{OVERVIEW_LEFT}" y="{height + 4}" width="28" height="6"/>')
        legend.append(f'<text x="{OVERVIEW_LEFT + 32}" y="{height + 10}">platform</text>')
        height += 18
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {OVERVIEW_WIDTH} {height}" '
        f'width="{OVERVIEW_WIDTH}" height="{height}"><style>{_OVERVIEW_STYLE}</style>'
    )
    return "\n".join([head, *lines, "".join(legend), "</svg>"]).encode() + b"\n"


def files(config: Any, tree: Any) -> dict[str, bytes]:
    """The README image: docs/overview-<readme_map>.svg, or nothing when the project names no README map."""
    with effort.stage("present.files"):
        version = config.project.readme_map
        if not version:
            return {}
        return {
            overview_path(version): overview(
                report.credits(config, tree), version, report.layers(config, tree, version)
            )
        }
