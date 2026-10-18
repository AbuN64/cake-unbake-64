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
OVERVIEW_SEAM = 1024  # a gap this small between two blobs of one class is a seam: the row still counts as uniform
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
# blobs and platform in the site's tones: its amber identified and decoded blobs, its strong edge colour for boot
_OVERVIEW_STYLE = (
    _STYLE.replace("text{font:6px", "text{font:10px").replace(".h{fill:#c4472d}", "")
    + ".b1{fill:#e3c27a}.b2{fill:#b87a14}.p1{fill:#b9b2a3}"
    + ".bt{fill:#1b1a17}.bt.q10{fill:#fbf9f3}"
    + "@media (prefers-color-scheme:dark){.b1{fill:#7a6232}.b2{fill:#d9a441}.p1{fill:#6a6658}"
    + ".bt.b1,.bt.p1{fill:#f0ead8}.bt.q0{fill:#f0ead8}.bt.q10{fill:#1b1a17}}"
)


def _num1(value: float) -> str:
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return text or "0"


def _tone(score: float, exact: bool) -> int:
    """0 exact, 1 fuzzy high (half or more), 2 fuzzy low, 3 none."""
    if exact:
        return 0
    return 3 if score <= 0 else 1 if score >= FUZZY_HIGH else 2


def _uniform(parts: list[tuple[int, int, int]]) -> bool:
    """A row of one class from edge to edge, the seams between its blobs (gaps up to OVERVIEW_SEAM) not counted."""
    ends = [0, *(e for _, _, e in parts)]
    starts = [s for _, s, _ in parts] + [OVERVIEW_ROW]
    return len({t for t, _, _ in parts}) == 1 and all(b - a <= OVERVIEW_SEAM for a, b in zip(ends, starts, strict=True))


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
    cover: dict[int, list[tuple[int, int, int]]] = {}  # per row its runs as (tone, start, end) inside the row
    drawn: list[list[tuple[float, int, float]]] = [
        [] for _ in _TONES
    ]  # per tone the outlines of its blocks: one path each keeps the file small
    scale = OVERVIEW_BAR / OVERVIEW_ROW
    for start, end, tone in runs:
        at = start
        while at < end:
            row = at // OVERVIEW_ROW
            until = min(end, (row + 1) * OVERVIEW_ROW)
            cover.setdefault(row, []).append((tone, at - row * OVERVIEW_ROW, until - row * OVERVIEW_ROW))
            drawn[tone].append((OVERVIEW_LEFT + (at - row * OVERVIEW_ROW) * scale, row, max((until - at) * scale, 1)))
            at = until
    uniform = {row: parts[0][0] for row, parts in cover.items() if _uniform(parts)}
    slots: list[list[int]] = []  # [first row, last row, tone or -1]: runs of rows of one uniform class are one band
    for row in sorted(cover):
        tone = uniform.get(row, -1)
        if slots and tone >= 0 and slots[-1][2] == tone and slots[-1][1] == row - 1:
            slots[-1][1] = row
        else:
            slots.append([row, row, tone])
    order = {row: n for n, (first, last, _) in enumerate(slots) for row in range(first, last + 1) if row in cover}
    banded = {row for first, last, _ in slots if last > first for row in range(first, last + 1)}
    lines, bands, whole = [], [], []
    for n, (first, last, tone) in enumerate(slots):
        y = n * OVERVIEW_PITCH
        if last > first:
            whole.append((tone, first))
            name = f"0x{first * OVERVIEW_ROW:06X}\u20130x{last * OVERVIEW_ROW:06X}"
            pos = f'x="{OVERVIEW_LEFT + 6}" y="{y + OVERVIEW_HEIGHT - 3}"'
            bands.append(f'<text class="bt {_TONES[tone][0]}" {pos}>{name}</text>')
        else:
            lines.append(f'<text x="0" y="{y + OVERVIEW_HEIGHT - 3}">0x{first * OVERVIEW_ROW:06X}</text>')
    for (cls, _), blocks in zip(_TONES, drawn, strict=True):
        blocks = [b for b in blocks if b[1] not in banded]
        blocks += [(OVERVIEW_LEFT, row, OVERVIEW_BAR) for t, row in whole if _TONES[t][0] == cls]
        if blocks:
            shapes = "".join(
                f"M{_num1(x)} {order[row] * OVERVIEW_PITCH}h{_num1(w)}v{OVERVIEW_HEIGHT}h-{_num1(w)}z"
                for x, row, w in blocks
            )
            lines.append(f'<path class="{cls}" d="{shapes}"/>')
    lines += bands
    top = len(slots) * OVERVIEW_PITCH + 6  # one legend strip: the 0 % to 100 % scale, then a swatch per layer drawn
    legend = [f'<text x="{OVERVIEW_LEFT}" y="{top + 7}">0 %</text>']
    x = OVERVIEW_LEFT + 34
    for cls, width in zip(("q0", "q3", "q7", "q10"), _LEGEND, strict=True):  # none, fuzzy low, fuzzy high, exact
        legend.append(f'<rect class="{cls}" x="{x}" y="{top + 1}" width="{width}" height="6"/>')
        x += width + 2
    legend.append(f'<text x="{x + 4}" y="{top + 7}">100 %</text>')
    x += 52
    for cls, name in (("b1", "identified"), ("b2", "decoded"), ("p1", "platform")):
        if (cls == "p1" and (layers or {}).get("platform")) or (cls != "p1" and blobs):
            legend.append(f'<rect class="{cls}" x="{x}" y="{top + 1}" width="12" height="6"/>')
            legend.append(f'<text x="{x + 16}" y="{top + 7}">{name}</text>')
            x += 16 + 7 * len(name) + 12
    height = top + 11
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
