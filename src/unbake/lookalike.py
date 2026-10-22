"""Existing records that look like a padded view: same size, and every named field at the same offset and size."""

from __future__ import annotations

import re
from pathlib import Path

from unbake import cheaders

_LIMIT = 3  # records named in one hint
_Index = dict[str, list[tuple[str, int | None, set[tuple[int, int]]]]]
_cache: dict[str, tuple[cheaders.Headers, _Index]] = {}
_HEAD = re.compile(r"(?:\btypedef\s+)?\bstruct\s*(\w+)?\s*\{")


def _shape(struct: cheaders.Struct) -> tuple[int | None, set[tuple[int, int]]]:
    """(size, {(offset, size) of each named, non-padding field}) of a laid-out record."""
    return struct.size, {(f.offset, f.type.size or 0) for f in struct.fields if not f.pad}


def _index(root: Path) -> tuple[cheaders.Headers, _Index]:
    """The project's headers and, per record size, every complete record the headers and the source files define as
    (name, size, named fields). A source file's own records are read with its text alone (they are file-local)."""
    key = str(root)
    if key in _cache:
        return _cache[key]
    base = cheaders.Headers()
    base.read(root, sources=())
    by_size: _Index = {}

    def keep(headers: cheaders.Headers, names: list[str], where: str) -> None:
        for name in names:
            struct = headers.struct(name)
            if struct is not None and struct.complete and struct.size and not struct.union:
                size, fields = _shape(struct)
                if len(fields) >= 2:
                    by_size.setdefault(str(size), []).append((f"{name} ({where})", size, fields))

    keep(base, [n for n in base._raw if not n.startswith("_")], "headers")
    for path in sorted((root / "src").rglob("*")):
        if path.suffix.lower() not in (".c", ".cpp", ".h"):
            continue
        text = path.read_text(errors="replace")
        if "struct" not in text:
            continue
        local = base.copy()
        before = set(local._raw)
        local.add(text, path.relative_to(root).as_posix())
        keep(local, [n for n in local._raw if n not in before], path.relative_to(root).as_posix())
    _cache[key] = (base, by_size)
    return _cache[key]


def hint(root: Path, path: str, text: str, line: int) -> str:
    """The sentence naming existing records that match the padded struct opened on `line` of `text`, "" when none."""
    base, by_size = _index(root)
    heads = [m for m in _HEAD.finditer(text) if text.count("\n", 0, m.start()) + 1 == line]
    if not heads:
        return ""
    head = heads[0]
    depth, at = 0, head.end() - 1
    while at < len(text):
        depth += (text[at] == "{") - (text[at] == "}")
        at += 1
        if depth == 0:
            break
    alias = re.match(r"\s*(\w+)\s*;", text[at:])
    name = (alias[1] if alias else head[1]) or ""
    local = base.copy()
    local.add(text, path)
    mine = local.struct(name) if name else None
    if mine is None or not mine.complete or not mine.size:
        return ""
    size, fields = _shape(mine)
    if not fields:
        return ""
    close = sorted(
        (len(other) - len(fields), label)  # the record with no further named field first
        for label, other_size, other in by_size.get(str(size), ())
        if other_size == size and fields <= other and not label.startswith(f"{name} (")
    )
    found = [label for _, label in close]
    if not found:
        return ""
    return "an existing record has this size and these field offsets: " + "; ".join(found[:_LIMIT])
