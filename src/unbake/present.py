"""The small presentation files a site reads live from the repository: progress.json, a memory map per version and
the source line of every function. Same report rows as the README, identical shape for every project."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from contextlib import suppress
from xml.sax.saxutils import escape

from unbake import config as configuration
from unbake import effort, pool, process, report
from unbake.contracts import Json, Refusal, Snapshot

PROGRESS = "progress.json"
SCHEMA = 1
SYMBOLS_SCHEMA = 2  # symbols.json gained percent and words per function


def map_path(version: str) -> str:
    return f"docs/memory-map-{version}.svg"


def overview_path(version: str) -> str:
    return f"docs/overview-{version}.svg"


def symbols_path(version: str) -> str:
    return f"versions/{version}/symbols.json"


def paths(versions: Iterable[str]) -> list[str]:
    return [PROGRESS, *(p for v in versions for p in (map_path(v), overview_path(v), symbols_path(v)))]


def _dump(document: Json) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _committed(snapshot: Snapshot) -> str:
    """The ISO commit time of the tree the files are made from: a pure function of the commit, so a fresh setup of the
    same commit writes the same bytes. Empty when the commit is unknown (an unborn branch, a test tree)."""
    with suppress(Refusal, OSError):
        return process.git(snapshot.config, "show", "-s", "--format=%cI", snapshot.commit).stdout.decode().strip()
    return ""


def progress(snapshot: Snapshot, current: Json, updated: str | None = None) -> bytes:
    """progress.json: per version the cartridge, function and byte counts of the report the README shows. `head` is the
    commit the tree was made from (a file cannot name the commit that holds it)."""
    project, rows = snapshot.config.project, current["versions"]
    drafted = _drafted(snapshot)
    carts = []
    for v in project.versions:
        row = rows[v]
        matched, total, fuzzy = row["code_matched"], row["code_total"], drafted.get(v, 0)
        carts.append(
            {
                "id": project.version_files[v].meta.get("cartridge_id", v),
                "version": v,
                "functions": {"matched": row["functions_matched"], "total": row["functions_total"]},
                "text_pct": row["code_percent"],
                "bytes": {"matched": matched, "fuzzy": fuzzy, "asm": total - matched - fuzzy, "total": total},
                "toolchain": project.toolchain,
            }
        )
    document = {"schema": SCHEMA, "carts": carts}
    with suppress(ValueError, AttributeError, TypeError):  # the same numbers keep the file: setup never rewrites it
        before = json.loads(snapshot.peek(PROGRESS) or b"")
        if {k: v for k, v in before.items() if k not in ("head", "updated")} == document:
            return snapshot.peek(PROGRESS) or b""
    head = snapshot.commit if re.fullmatch(r"[0-9a-f]{40}", snapshot.commit or "") else ""
    when = (_committed(snapshot) if head else "") if updated is None else updated
    return _dump({**document, "head": head, "updated": when})


def _drafted(snapshot: Snapshot) -> dict[str, int]:
    """Per version the code bytes that are not exact but have a draft scoring above zero."""
    out: dict[str, int] = {}
    for row in report.credits(snapshot):
        if row.score > 0 and not row.exact:
            out[row.version] = out.get(row.version, 0) + row.code - row.kept
    return out


# memory map geometry: 16 KiB per row, 16 bytes per unit, a gutter for the row offset
ROW_BYTES = 16384
UNIT = 16
GUTTER = 72
PITCH = 8
HEIGHT = 6
STEPS = 10  # q0 = no draft, q10 = exact, q1..q9 spread the drafts by score
_STYLE = (
    "text{font:6px monospace;fill:#6b675c}.h{fill:#c4472d}"
    ".q0{fill:#d9d4c7}.q1{fill:#e3e2f2}.q2{fill:#d6d6ef}.q3{fill:#c8c9ec}.q4{fill:#babce8}.q5{fill:#acafe4}"
    ".q6{fill:#9ea2df}.q7{fill:#9094d8}.q8{fill:#8186cf}.q9{fill:#7277c4}.q10{fill:#1b1a17}"
    "@media (prefers-color-scheme:dark){text{fill:#a8a293}"
    ".q0{fill:#4a463d}.q1{fill:#35375a}.q2{fill:#3c3f66}.q3{fill:#444772}.q4{fill:#4d517f}.q5{fill:#575b8c}"
    ".q6{fill:#62669a}.q7{fill:#6e72a8}.q8{fill:#7b7fb6}.q9{fill:#888cc4}.q10{fill:#d6d0bd}}"
)


def step(score: float, exact: bool) -> int:
    if exact:
        return STEPS
    if score <= 0:
        return 0
    return 1 + min(STEPS - 2, int(score * (STEPS - 1)))


def _num(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def memory_map(snapshot: Snapshot, version: str) -> bytes:
    """The ROM of one version, byte-proportional: rows of 16 KiB in ROM order (rows without code left out), each
    function as wide as its bytes and shaded by its credit, a hairline where a segment starts, and a title per
    function naming it, its score and its size."""
    pieces: list[tuple[int, int, str, int, float]] = []  # (rom start, size, name, step, percent)
    for credit in report.credits(snapshot):
        if credit.version != version or credit.member.kind != "function":
            continue
        percent = report.percent(credit)
        for p in credit.member.placements:
            if p.version == version and p.section == ".text" and p.size > 0:
                pieces.append((p.rom_start, p.size, credit.name, step(credit.score, credit.exact), percent))
    pieces.sort()
    starts = sorted({s[1] for s in snapshot.versions[version].segments if s[2] > s[1]})
    rows: dict[int, list[str]] = {}
    for start, size, name, q, percent in pieces:
        title = f"<title>{escape(name)} · {_num(percent)}% · {size} B</title>"
        at, end = start, start + size
        while at < end:
            row = at // ROW_BYTES
            until = min(end, (row + 1) * ROW_BYTES)
            x = GUTTER + (at - row * ROW_BYTES) / UNIT
            rows.setdefault(row, []).append(
                f'<rect class="q{q}" x="{_num(x)}" width="{_num(max((until - at) / UNIT, 0.25))}" height="{HEIGHT}">'
                f"{title}</rect>"
            )
            at = until
    lines = []
    for n, row in enumerate(sorted(rows)):
        y = n * PITCH
        lines.append(f'<g transform="translate(0 {y})"><text y="{HEIGHT}">{row * ROW_BYTES:08X}</text>')
        lines.extend(rows[row])
        lines.extend(
            f'<rect class="h" x="{_num(GUTTER + (s - row * ROW_BYTES) / UNIT)}" width="0.5" height="{HEIGHT}"/>'
            for s in starts
            if row * ROW_BYTES <= s < (row + 1) * ROW_BYTES
        )
        lines.append("</g>")
    width, height = GUTTER + ROW_BYTES // UNIT, max(len(rows) * PITCH - (PITCH - HEIGHT), 0)
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        f"<style>{_STYLE}</style>"
    )
    return "\n".join([head, *lines, "</svg>"]).encode() + b"\n"


# overview geometry: made to be shown 720 px wide, 64 KiB per row, one block per run of one status
OVERVIEW_WIDTH = 720
OVERVIEW_ROW = 0x10000
OVERVIEW_LEFT = 64
OVERVIEW_BAR = OVERVIEW_WIDTH - OVERVIEW_LEFT - 8
OVERVIEW_HEIGHT = 13
OVERVIEW_PITCH = 17
OVERVIEW_JOIN = 32  # functions this close count as touching: alignment padding does not split a run
FUZZY_HIGH = 0.5
_TONES = (("q10", "exact"), ("q7", "fuzzy high"), ("q3", "fuzzy low"), ("q0", "none"))
_LEGEND = (28, 72, 72, 28)  # block widths of the legend bar: none, fuzzy low, fuzzy high, exact
_OVERVIEW_STYLE = _STYLE.replace("text{font:6px", "text{font:10px").replace(".h{fill:#c4472d}", "")


def _num1(value: float) -> str:
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return text or "0"


def _tone(score: float, exact: bool) -> int:
    """0 exact, 1 fuzzy high (half or more), 2 fuzzy low, 3 none."""
    if exact:
        return 0
    return 3 if score <= 0 else 1 if score >= FUZZY_HIGH else 2


def overview(snapshot: Snapshot, version: str) -> bytes:
    """The ROM of one version as a picture that reads at 720 px: rows of 64 KiB in ROM order (rows without code left
    out), runs of adjacent functions of one status merged into one block, and a legend of the four statuses."""
    pieces = sorted(
        (p.rom_start, p.size, _tone(credit.score, credit.exact))
        for credit in report.credits(snapshot)
        if credit.version == version and credit.member.kind == "function"
        for p in credit.member.placements
        if p.version == version and p.section == ".text" and p.size > 0
    )
    runs: list[list[int]] = []  # [start, end, tone]
    for start, size, tone in pieces:
        if runs and runs[-1][2] == tone and start - runs[-1][1] <= OVERVIEW_JOIN:
            runs[-1][1] = max(runs[-1][1], start + size)
        else:
            runs.append([start, start + size, tone])
    rows: set[int] = set()
    drawn: list[list[tuple[float, int, float]]] = [
        [],
        [],
        [],
        [],
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
    height = label + 4
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {OVERVIEW_WIDTH} {height}" '
        f'width="{OVERVIEW_WIDTH}" height="{height}"><style>{_OVERVIEW_STYLE}</style>'
    )
    return "\n".join([head, *lines, "".join(legend), "</svg>"]).encode() + b"\n"


_DEFINITION = r"^(?:[A-Za-z_][^;{{}}()#]*?[\s*])?{name}\s*\("  # at column 0, after a return type if any


def _line(text: str, name: str) -> int:
    """The 1-based line where the function named is defined (or, for assembly, labelled); 0 when not found."""
    for pattern in (rf"^[ \t]*(?:glabel|\.globl|\.global)\s+{re.escape(name)}\b", rf"^{re.escape(name)}:"):
        if (match := re.search(pattern, text, re.M)) is not None:
            return text.count("\n", 0, match.start()) + 1
    for match in re.finditer(_DEFINITION.format(name=re.escape(name)), text, re.M):
        rest = text[match.end() :]
        depth, i = 1, 0
        while i < len(rest) and depth:  # skip the parameter list, then a body (not a prototype) must follow
            depth += {"(": 1, ")": -1}.get(rest[i], 0)
            i += 1
        if rest[i:].lstrip().startswith("{"):
            return text.count("\n", 0, match.start()) + 1
    return 0


def symbols(snapshot: Snapshot, version: str) -> bytes:
    """versions/<v>/symbols.json: every function of the version that a tracked source holds (its unit, else its fuzzy
    draft) with the path and the line where it starts, its percent (report.percent) and its words [matched, total].
    Statistics only: no byte, instruction or disassembly of the ROM is ever written. One function per line."""
    found: dict[str, Json] = {}
    for row in report.credits(snapshot):
        if row.version != version or row.member.kind != "function":
            continue
        path = row.unit.path if row.unit is not None else snapshot.layout.fuzzy.get(row.name, {}).get("path")
        if not path:
            continue
        text = _text(snapshot, path)
        if text is None:
            continue
        size = sum(p.size for p in row.member.placements if p.version == version and p.section == ".text")
        share = report.percent(row)
        found[row.name] = {
            "path": path,
            "line": _line(text, row.name) or 1,
            "percent": share,
            "words": [round(share * (size // 4) / 100), size // 4],  # matched, total: counts only, never the bytes
        }
    body = ",\n".join(
        f"{json.dumps(name)}:{json.dumps(found[name], sort_keys=True, separators=(',', ':'))}" for name in sorted(found)
    )
    return f'{{"schema":{SYMBOLS_SCHEMA},"functions":{{\n{body}\n}}}}\n'.encode()


def _text(snapshot: Snapshot, path: str) -> str | None:
    def read() -> str | None:
        data = snapshot.peek(path)
        return None if data is None else data.decode(errors="replace")

    return effort.memo(("present-text", snapshot.digest, path), read)


def files(snapshot: Snapshot, current: Json) -> dict[str, bytes]:
    """Every presentation file of the tree, from the report rows the README shows."""
    with effort.stage("present.files"):
        out = {PROGRESS: progress(snapshot, current)}
        jobs = [
            (snapshot, v, kind) for v in snapshot.config.project.versions for kind in ("map", "overview", "symbols")
        ]
        made = pool.map(snapshot.config, "present.files", _file_job, jobs)  # every version's files on all cores
        for (_, v, kind), blob in zip(jobs, made, strict=True):
            out[{"map": map_path, "overview": overview_path, "symbols": symbols_path}[kind](v)] = blob
        return out


def _file_job(item: tuple[Snapshot, str, str]) -> bytes:
    snapshot, version, kind = item
    make = {"map": memory_map, "overview": overview, "symbols": symbols}[kind]
    return make(snapshot, version)


def recomp(title: str) -> bytes:
    """The starter .recomp.json the recomp.fyi board reads: the game, its platform and the kind of project."""
    rules = configuration.load_resource("boards.toml")["recomp"]
    return (json.dumps({"$schema": rules["schema"], "game": title, **rules["fields"]}, indent=2) + "\n").encode()
