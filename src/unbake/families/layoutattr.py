"""Layout attributes of a preprocessed text, made visible to the parser.

pycparser drops `__attribute__`, so `aligned(N)` and the attributes this tool cannot lay out would vanish from the
tree. `rewrite` replaces each one that matters with a marker member of its record (`__ub_aligned_N`: record alignment,
`__ub_field_aligned_N`: alignment of the next field, `__ub_pack_N`: cap on field alignment, `__ub_unsupported_WHAT`: no
supported layout) and names the typedefs whose own type carries such an attribute."""

from __future__ import annotations

import re
from typing import Any

MARKER = "__ub_"
_TOKEN = re.compile(
    r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|^[ \t]*#[^\n]*|\b__attribute__\b'
    r"|\b(?:struct|union)\b|[{};()=,*\[]|[A-Za-z_]\w*",
    re.M,
)
_PACK = re.compile(r"#\s*pragma\s+pack\s*(?:\(([^)]*)\))?")
_BAD = frozenset(("vector_size", "mode", "ms_struct"))


def _items(group: str) -> list[tuple[str, int]]:
    """The (name, alignment) of each layout attribute in `((a, b(1)))`: alignment 0 means unsupported."""
    found: list[tuple[str, int]] = []
    for item in re.finditer(r"(\w+)\s*(?:\(([^()]*)\))?", group):
        name = item[1].strip("_")
        if name == "aligned":
            found.append(
                (
                    "aligned",
                    int(item[2], 0) if item[2] and re.fullmatch(r"\s*(?:0[xX][\da-fA-F]+|\d+)\s*", item[2]) else 0,
                )
            )
        elif name == "packed":
            found.append(("pack", 1))
        elif name in _BAD:
            found.append((name, 0))
    return found


def _group(text: str, at: int) -> int:
    """The end of the balanced parenthesis group opening at `at`, or -1."""
    depth = 0
    for index in range(at, len(text)):
        depth += (text[index] == "(") - (text[index] == ")")
        if depth == 0:
            return index + 1
    return -1


def rewrite(text: str) -> tuple[str, frozenset[str]]:
    edits: list[tuple[int, int, str]] = []
    tainted: set[str] = set()
    frames: list[dict[str, Any]] = []
    pack = 0
    saved: list[Any] = []
    pending: dict[str, Any] | None = None
    closed: dict[str, Any] | None = None
    stmt, last_end, last_ident = 0, 0, ""

    def mark(frame: dict[str, Any], name: str, value: int | str) -> None:
        edits.append((frame["open"], frame["open"], f"char {MARKER}{name}_{value};"))

    def apply(frame: dict[str, Any], items: list[tuple[str, int]]) -> None:
        for name, value in items:
            if name == "pack":
                mark(frame, "pack", value)
            else:
                mark(frame, "aligned", value) if value else mark(frame, "unsupported", name)

    for match in _TOKEN.finditer(text):
        token, start, end = match[0], match.start(), match.end()
        if token.lstrip().startswith("#"):
            if found := _PACK.match(token.lstrip()):
                args = [a.strip() for a in (found[1] or "").split(",") if a.strip()]
                if args[:1] == ["pop"]:
                    pack = saved.pop() if saved else 0
                else:
                    if args[:1] == ["push"]:
                        saved.append(pack)
                        args = args[1:]
                    pack = int(args[0]) if args and args[0].isdigit() else 0
            continue
        if token == "__attribute__":
            open_ = text.find("((", end)
            stop = _group(text, open_) if open_ >= 0 and not text[end:open_].strip() else -1
            if stop < 0:
                continue
            items = _items(text[open_:stop])
            if not items:
                continue
            blank = re.sub(r"[^\n]", " ", text[start:stop])
            edits.append((start, stop, blank))
            trailer = closed is not None and not text[last_end:start].strip()
            if pending is not None:
                pending["attrs"].extend(items)
            elif closed is not None and trailer:
                apply(closed, items)
            elif frames and frames[-1]["kind"] == "record":
                for name, value in items:
                    if name == "pack":
                        mark(frames[-1], "unsupported", "packed")
                    elif value:
                        edits.append((frames[-1]["field"], frames[-1]["field"], f"char {MARKER}field_aligned_{value};"))
                    else:
                        mark(frames[-1], "unsupported", name)
            elif re.search(r"\btypedef\b", text[stmt:start]):
                tainted.add(last_ident)
            last_end = stop
            # the scanner resumes after the group: its inner words are not declarations
            continue
        if token in ("struct", "union"):
            pending, closed = {"attrs": []}, None
        elif token == "{":
            kind = "record" if pending is not None else "block"
            frames.append(
                {"kind": kind, "open": end, "field": end, "attrs": pending["attrs"] if pending else [], "pack": pack}
            )
            pending, closed, stmt = None, None, end
        elif token == "}":
            frame = frames.pop() if frames else None
            closed = frame if frame and frame["kind"] == "record" else None
            if closed:
                apply(closed, closed["attrs"])
                if closed["pack"] not in (0, 8):
                    mark(closed, "pack", closed["pack"])
            pending, stmt = None, end
        elif token == ";":
            pending, closed, stmt = None, None, end
            if frames:
                frames[-1]["field"] = end
        elif not re.match(r"[A-Za-z_]", token):
            pending, closed = None, None
        else:
            last_ident = token
            if pending is None:
                closed = None
        last_end = end
    for first, last, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        text = text[:first] + replacement + text[last:]
    return text, frozenset(tainted)
