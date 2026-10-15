"""Type-debt checks over code text with comments and strings blanked: each returns {line: subject} per row."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from typing import Any

_SYMBOL = r"func_[0-9A-Fa-f]+"  # the project's function symbol: no record type may be named after one
_NAMED = re.compile(rf"\b\w+?_{_SYMBOL}\w*")  # a name carrying a function symbol (Actor_func_802A274C_de)
_VIEW = re.compile(rf"\b{_SYMBOL}(?:_\w+?)?_S\d+\b")  # a per-function view numbered _S<n>
_TAG = re.compile(rf"\b(?:struct|union)\s+(\w*{_SYMBOL}\w*)")  # a tag that carries a function symbol
_SIZES = {1: "u8 s8 char bool", 2: "u16 s16 short", 4: "u32 s32 int long float f32 void", 8: "u64 s64 double f64"}
_BYTES = {name: size for size, names in _SIZES.items() for name in names.split()}
_PRIMITIVE = frozenset(_BYTES) | {"unsigned", "signed", "const", "volatile", "size_t", "uintptr_t", "intptr_t"}
_INTEGER = frozenset(_PRIMITIVE - {"float", "f32", "double", "f64", "void", "bool"})
_PLACEHOLDER = re.compile(r"^(?:pad\w*|unk_?[0-9A-Fa-f]*|field_[0-9A-Fa-f]+|_[0-9A-Fa-f]+)$", re.I)  # an offset name


def _line(code: str) -> Callable[[int], int]:
    return lambda offset: code.count("\n", 0, offset) + 1


def _body(code: str, start: int) -> tuple[int, int]:
    """The span inside the brace at `start`."""
    depth = 0
    for index in range(start, len(code)):
        depth += (code[index] == "{") - (code[index] == "}")
        if depth == 0:
            return start + 1, index
    return start + 1, len(code)


def _record_names(code: str) -> set[str]:
    """Every name this text declares as a record type: struct and union tags and typedef aliases."""
    names = {m[1] for m in re.finditer(r"\b(?:struct|union|enum)\s+(\w+)", code)}
    names.update(m[1] for m in re.finditer(r"\btypedef\b[^;{]*?(\w+)\s*;", code))
    for head in re.finditer(r"\btypedef\s+(?:struct|union|enum)\s*(?:\w+\s*)?\{", code):
        alias = re.match(r"\}\s*(\w+)\s*;", code[_body(code, head.end() - 1)[1] :])
        if alias:
            names.add(alias[1])
    return names


@functools.cache
def fake_name(name: str) -> bool:
    """Whether a record name is invented per function: it carries a function symbol (role-prefixed or numbered view)."""
    return bool(re.search(_SYMBOL, name))


def duplicate_records(code: str, clones: dict[str, str]) -> dict[int, str]:
    """The first line defining each record that an identity class says duplicates its canonical (`clones` maps a
    record name to its canonical and header)."""
    line = _line(code)
    found: dict[int, str] = {}
    for name in sorted(_record_names(code) & clones.keys()):
        head = re.search(rf"\b(?:struct|union)\s+{re.escape(name)}\s*\{{|\}}\s*{re.escape(name)}\s*;", code)
        if head:
            found.setdefault(line(head.start()), f"{name} duplicates {clones[name]}")
    return found


def fake_type(code: str) -> dict[int, str]:
    """The first line of each distinct invented per-function record name: numbered views, and record types (declared
    here or used as a type) whose name carries a function symbol. A data symbol carrying one is not a type."""
    line = _line(code)
    records = _record_names(code)
    first: dict[str, int] = {}
    for match in _VIEW.finditer(code):
        first.setdefault(match[0], line(match.start()))
    for match in _TAG.finditer(code):
        first.setdefault(match[1], line(match.start(1)))
    for match in _NAMED.finditer(code):
        name = match[0]
        after = code[match.end() :]
        typed = name in records or re.match(r"\s*\**\s*[A-Za-z_]", after) or re.match(r"\s*\*", after)
        if typed and not re.match(r"\s*\(", after):
            first.setdefault(name, line(match.start()))
    found: dict[int, str] = {}
    for name, at in first.items():
        found.setdefault(at, name)
    return found


def _padded(code: str, head: re.Match[str]) -> tuple[bool, int]:
    """Whether the struct opened by `head` is mostly padding arrays, and where it ends."""
    start, stop = _body(code, head.end() - 1)
    pad = named = total = 0
    for field in code[start:stop].split(";"):
        match = re.search(
            r"([A-Za-z_]\w*)\s*(\*?)\s*([A-Za-z_]\w*)\s*(?:\[\s*(0[xX][0-9A-Fa-f]+|\d+)\s*\])?\s*$", field.strip()
        )
        if not match:
            continue
        size = 4 if match[2] else _BYTES.get(match[1], 4) * int(match[4] or "1", 0)
        total += size
        if match[4] and _PLACEHOLDER.match(match[3]):
            pad += size
        else:
            named += 1
    return bool(pad and 2 * pad >= total and named <= 3), stop


_OWNER_RECORD = re.compile(r"Data[A-Z0-9][A-Za-z0-9]*")  # recipe's owner-local record of one data symbol


def owner_record(code: str, name: str) -> bool:
    """Whether `name` is the owner-local record of a data symbol this text declares as it or as a pointer to it: its
    members are every offset the symbol's users access, the bytes between no one reads (owner 2026-10-15), so its
    padding is the evidence, not a view standing in for real fields."""
    return bool(_OWNER_RECORD.fullmatch(name)) and bool(
        re.search(rf"^[ \t]*extern\s+(?:struct\s+)?{name}\s*\*?\s*[A-Za-z_]\w*\s*;", code, re.M)
    )


def _named(code: str, head: re.Match[str]) -> str:
    """The typedef alias, else the tag, of the struct opened by `head`."""
    alias = re.match(r"\}\s*(\w+)\s*;", code[_body(code, head.end() - 1)[1] :])
    tag = re.match(r"struct\s+(\w+)", head[0])
    return alias[1] if alias else tag[1] if tag else ""


def pad_view(code: str) -> dict[int, str]:
    """Struct definitions where pad arrays (offset-named: pad*, unk<hex>, field_<hex>) are at least half the bytes and
    at most three fields are named; a data symbol's owner-local record is none."""
    line = _line(code)
    return {
        line(head.start()): " ".join(head[0].split())
        for head in re.finditer(r"\bstruct\s*(?:\w+\s*)?\{", code)
        if _padded(code, head)[0] and not owner_record(code, _named(code, head))
    }


def placeholder_record(code: str) -> dict[int, str]:
    """Records this unit defines whose every field is an offset placeholder (unk4, field_8, pad0) and that the unit
    then uses by name: a layout invented here instead of the project's real record."""
    line = _line(code)
    found: dict[int, str] = {}
    for head in re.finditer(r"\b(typedef\s+)?struct\s*(\w+)?\s*\{", code):
        start, stop = _body(code, head.end() - 1)
        body = code[start:stop]
        alias = re.match(r"\s*\}\s*(\w+)\s*;", code[stop:]) if head[1] else None
        names = {n for n in (head[2], alias[1] if alias else None) if n}
        if "{" in body or not names:
            continue
        fields = [re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$", f.strip()) for f in body.split(";") if f.strip()]
        if not fields or not all(f and _PLACEHOLDER.match(f[1]) for f in fields):
            continue
        defined = {head.start(2)} if head[2] else set()
        if alias:
            defined.add(stop + alias.start(1))
        if any(m.start() not in defined for n in names for m in re.finditer(rf"\b{re.escape(n)}\b", code)):
            found.setdefault(line(head.start()), " ".join(code[head.start() : head.end()].split()))
    return found


def view_types(code: str) -> set[str]:
    """The names (tag and typedef) of the records this text defines as a union or a pad view."""
    names: set[str] = set()
    for head in re.finditer(r"(?<![\w])(?:(typedef)\s+)?(struct|union)\s*(\w+)?\s*\{", code):
        padded, stop = _padded(code, head) if head[2] == "struct" else (False, _body(code, head.end() - 1)[1])
        alias = re.match(r"\}\s*(\w+)\s*;", code[stop:]) if head[1] else None
        if padded and owner_record(code, alias[1] if alias else head[3] or ""):
            continue
        if head[2] != "union" and not padded:
            continue
        names.update(n for n in (head[3], alias[1] if alias else None) if n)
    return names


def view_uses(code: str, views: set[str]) -> dict[int, str]:
    """The first line on which this text names each of the given view records (a definition is not a use)."""
    if not views:
        return {}
    line = _line(code)
    first: dict[str, int] = {}
    names = "|".join(map(re.escape, sorted(views)))
    for match in re.finditer(rf"\b(?:(struct|union)\s+)?({names})\b(?!\s*\{{)", code):
        first.setdefault(match[2], line(match.start()))
    return {at: name for name, at in first.items()}


def parameters(code: str) -> list[dict[str, Any]]:
    """Every definition parameter declared void* or integer: its function, name, index, span, and each cast of it.
    A cast row holds the span from the cast's open parenthesis through the parameter name, the spelled target
    and its qualifiers."""
    line = _line(code)
    rows: list[dict[str, Any]] = []
    for head in re.finditer(r"^[ \t]*[A-Za-z_][\w \t\*]*?\b(\w+)\s*\(([^()]*)\)\s*\{", code, re.M):
        start, stop = _body(code, head.end() - 1)
        at = head.start(2)
        for index, param in enumerate(head[2].split(",")):
            span = (at, at + len(param))
            at += len(param) + 1
            match = re.fullmatch(r"\s*((?:\w+\s+)*?)(\w+)(?:\s*(\*)\s*|\s+)(\w+)\s*", param)
            if not match:
                continue
            words = set(match[1].split()) | {match[2]}
            plain = (match[2] == "void" and match[3]) or (not match[3] and words <= _INTEGER)
            if not plain:
                continue
            cast = rf"\(\s*((?:const\s+|volatile\s+)*)(?:struct\s+)?(\w+)\s*\*\s*\)\s*(\(?)\s*{match[4]}\b"
            casts = [
                {
                    "span": (start + m.start(), start + m.end()),
                    "target": m[2],
                    "spelled": " ".join(re.sub(r"\s*\*\s*", " *", m[0][1 : m[0].index(")")].strip()).split()),
                    "qualifiers": frozenset(m[1].split()),
                    "open": bool(m[3]),
                }
                for m in re.finditer(cast, code[start:stop])
            ]
            rows.append(
                {
                    "function": head[1],
                    "name": match[4],
                    "index": index,
                    "span": span,
                    "line": line(head.start(1)),
                    "pointer": bool(match[3]),
                    "qualifiers": frozenset(match[1].split()) & {"const", "volatile"},
                    "body": (start, stop),
                    "casts": casts,
                }
            )
    return rows


def param_cast(code: str) -> dict[int, str]:
    """Function parameters declared void* or integer that the body casts to a pointer of a non-primitive type. The
    subject is the function called, "f()": its head holds every such parameter on one line, so retyping one of them
    leaves the row of the others as it was instead of reading as new debt, and no bare name lets the row withhold the
    function's callers."""
    hits: dict[int, str] = {}
    for row in parameters(code):  # an integer parameter's debt is the address-int row of its callers, not this one
        if row["pointer"] and any(c["target"] not in _PRIMITIVE for c in row["casts"]):
            hits.setdefault(row["line"], row["function"] + "()")
    return hits


_ADDRESS_INT = re.compile(
    r"(?<!sizeof)\(\s*((?:(?:unsigned|signed|const|volatile)\s+)*"
    r"(?:u8|s8|u16|s16|u32|s32|u64|s64|int|long|short|size_t|uintptr_t|intptr_t))\s*\)"
    r"\s*&(?!&)(?!\s*\(\s*\(\s*\w+\s*\*\s*\)\s*0\b)\s*\(*\s*(\w*)"
)


def _holders(code: str) -> list[tuple[int, int, str]]:
    """(body start, body end, name) of every function definition of the code."""
    spans = []
    for head in re.finditer(r"^[ \t]*[A-Za-z_][\w \t\*]*?\b(\w+)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*\{", code, re.M):
        start, stop = _body(code, head.end() - 1)
        spans.append((start, stop, head[1]))
    return spans


def address_int(code: str) -> dict[int, str]:
    """An address taken and cast to an integer type (the null-base offsetof idiom is no address): the callee or field
    is mistyped, or a ROM or segment symbol should be an array used without the address-of. Inside a function the
    subject is the function, "f()" with the cast type: a retype that respells the operand (`(s32)&((T *)a)->f` ->
    `(s32)&a->f`) is the same row, never new debt."""
    line = _line(code)
    holders = _holders(code) if _ADDRESS_INT.search(code) else []
    hits: dict[int, str] = {}
    for m in _ADDRESS_INT.finditer(code):
        holder = next((name for start, stop, name in holders if start <= m.start() < stop), "")
        cast = " ".join(m[1].split())
        hits.setdefault(line(m.start()), f"({cast})&{holder}()" if holder else f"({cast})&{m[2]}")
    return hits
