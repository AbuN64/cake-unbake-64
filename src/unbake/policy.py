"""Source rules, candidate admission scope, and project debt."""

from __future__ import annotations

import json
import re
import shutil
import tomllib
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from unbake import config, delta, effort, native, pool, process, recipes, store, tree, versions
from unbake import view as _view
from unbake.contracts import Finding, Json, Refusal, Snapshot, SourceView, UnitSpec, code_identity, digest


def _numbered(text: str) -> Callable[[int], int]:
    """text offset -> 1-based line, without counting newlines from the start for every hit."""
    breaks = [m.start() for m in re.finditer("\n", text)]
    return lambda offset: bisect_left(breaks, offset) + 1


def _blank(text: str) -> str:
    return re.sub(r"[^\n]", " ", text)


def _scopes(text: str) -> dict[str, str]:
    code, comments, directives = list(text), list(_blank(text)), list(text)
    tokens = r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\''
    for match in re.finditer(tokens, text):
        start, end = match.span()
        code[start:end] = _blank(match[0])
        if match[0].startswith(("/*", "//")):
            comments[start:end] = match[0]
            directives[start:end] = _blank(match[0])
    clean = "".join(code).splitlines(keepends=True)
    directive_lines = "".join(directives).splitlines(keepends=True)
    continuation = False
    for index, line in enumerate(directive_lines):
        is_directive = continuation or line.lstrip().startswith("#")
        continuation = is_directive and line.rstrip().endswith("\\")
        if is_directive:
            clean[index] = _blank(clean[index])
        else:
            directive_lines[index] = _blank(line)
    return {"code": "".join(clean), "directives": "".join(directive_lines), "comments": "".join(comments)}


def scopes(text: str) -> dict[str, str]:
    """Expose the existing code/directive/comment masks to source transformations."""
    return _scopes(text)


def _active(source: SourceView, path: str) -> str:
    lines = _view.active_lines(source, path)
    return "\n".join(lines.get(n, "") for n in range(1, max(lines, default=0) + 1))


def _volatile(code: str, ranges: Sequence[dict[str, Any]]) -> set[int]:
    allowed: set[Any] = set()
    cast = r"\(\s*[\w\s]*\bvolatile\b[\w\s]*\*\s*\)"
    for match in re.finditer(cast, code):
        stop = code.find(";", match.end())
        expression = code[match.end() : stop if stop >= 0 else None]
        numbers = [
            int(m[0], 16 if m[0].lower().startswith("0x") else 10)
            for m in re.finditer(r"\b(?:0[xX][\da-fA-F]+|[0-9]+)(?=[uUlL]*\b)", expression)
        ]
        combined = 0
        for number in numbers:
            combined |= number
        if any(r["start"] <= n <= r["end"] for n in [*numbers, combined] for r in ranges):
            allowed.update(match.start() + m.start() for m in re.finditer(r"\bvolatile\b", match[0]))
    line = _numbered(code)
    return {line(m.start()) for m in re.finditer(r"\bvolatile\b", code) if m.start() not in allowed}


def _declarations(code: str) -> dict[str, int]:
    names, start, depth, line = {}, 0, 0, _numbered(code)
    for token in re.finditer(r"[{};]", code):
        if token[0] == "{":
            depth += 1
        elif token[0] == "}":
            depth -= 1
            prefix = code[start : token.start()]
            function_body = re.search(r"\)\s*$", prefix.split("{", 1)[0])
            if depth == 0 and (function_body or not re.search(r"\b(?:typedef|struct|union|enum)\b", prefix)):
                start = token.end()
        elif depth == 0:
            declaration = code[start : token.start()]
            pattern = None
            if re.search(r"\btypedef\b", declaration) or (
                re.search(r"\bextern\b", declaration) and "(" not in declaration
            ):
                pattern = r"\b(\w+)\s*(?:\[[^]]*\]\s*)?$"
            elif "=" not in declaration and "{" not in declaration:
                pattern = r"\b(\w+)\s*\([^;]*\)\s*$"
            match = re.search(r"\(\s*\*\s*(\w+)\s*\)", declaration) if pattern else None
            match = match or (re.search(pattern, declaration) if pattern else None)
            if match:
                names[match[1]] = line(start + match.start(1))
            start = token.end()
    return names


def _header_names(snapshot: Snapshot | None, active: str) -> dict[str, int]:
    """The declarations of one header's active text, kept by that text."""

    def produce() -> dict[str, int]:
        return _declarations(_scopes(active)["code"])

    return store.cached_object(snapshot.config, "header.names", digest(active), produce) if snapshot else produce()


def _shared(code: str, source: SourceView, path: str, snapshot: Snapshot | None = None) -> set[int]:
    header_names: set[Any] = set()
    for header, _ in source.dependencies:
        if header != path and config.is_header(header):
            header_names.update(_header_names(snapshot, _active(source, header)))
    return {line for name, line in _declarations(code).items() if name in header_names}


_LOOP_KEYWORD = re.compile(r"\b(?:for|while)\s*\(")


def _empty_loop_heads(code: str) -> Iterator[tuple[int, str]]:
    """Yield (start, keyword) of loops whose parenthesised head is directly followed by ';'."""
    for match in _LOOP_KEYWORD.finditer(code):
        depth, index = 0, match.end() - 1
        while index < len(code):
            depth += {"(": 1, ")": -1}.get(code[index], 0)
            if depth == 0:
                break
            index += 1
        else:
            continue
        if code[index + 1 :].lstrip().startswith(";"):
            yield match.start(), match[0]


def _closes_do_body(code: str, start: int) -> bool:
    """Whether the loop head at start is the tail of a do { ... } while (...); statement."""
    before = code[:start].rstrip()
    if not before.endswith("}"):
        return False
    depth = 0
    for index in range(len(before) - 1, -1, -1):
        depth += {"}": 1, "{": -1}.get(before[index], 0)
        if depth == 0:
            return re.search(r"\bdo\s*$", before[:index]) is not None
    return False


def _empty_loops(code: str) -> set[int]:
    line = _numbered(code)
    return {
        line(start)
        for start, keyword in _empty_loop_heads(code)
        if not (keyword.startswith("while") and _closes_do_body(code, start))
    }


_PAIR = re.compile(r"\s*\)?\s*,\s*0x[0-9A-Fa-f]{8}\b")


def _raw_gfx(code: str) -> set[int]:
    opcodes = set(config.load_resource("rules.toml")["gfx"]["opcodes"])
    result, line = set(), _numbered(code)
    for match in re.finditer(r"\b0x[0-9A-Fa-f]{8}\b", code):
        if int(match[0], 16) >> 24 not in opcodes:
            continue
        prefix = code[max(code.rfind(";", 0, match.start()), code.rfind("\n", 0, match.start())) + 1 : match.start()]
        if (
            re.search(r"\bw0\s*=\s*\(?\s*$", prefix)
            or re.search(r"\bgs?[DS]PWord\s*\([^;]*$", prefix)
            or (re.search(r"\{\s*\(?\s*$", prefix) and _PAIR.match(code, match.end()))
        ):
            result.add(line(match.start()))
    return result


_C_WORDS = ["if", "else", "while", "for", "do", "switch", "case", "return", "sizeof", "defined", "typeof"]
_NOT_NAMES = frozenset([*_C_WORDS, "goto", "break", "continue", "default", "struct", "union", "enum"])


_NUMBER = re.compile(  # a float is consumed whole so its fragments are no integers; the integer keeps its suffix out
    r"(?<![\w.])(?:(?:\d+\.\d*|\.\d+|\d+(?=[eE][+-]?\d))(?:[eE][+-]?\d+)?[fFlL]?"
    r"|(?P<int>0[xX][0-9A-Fa-f]+|\d+)(?:[uU][lL]{0,2}|[lL]{1,2}[uU]?)?)(?![\w.])"
)
_KSEG0 = 0x80000000  # the first MIPS virtual address a ROM symbol can have: a smaller literal is a count, mask or index


def _integer(text: str) -> int | None:
    """The value of a C integer literal: 0x hex, a leading 0 octal, else decimal; None when it is not one."""
    try:
        return int(text, 16 if text[:2] in ("0x", "0X") else 8 if text[0] == "0" and len(text) > 1 else 10)
    except ValueError:
        return None


_OFFSET = re.compile(
    r"(?:\(\s*(?:char|u8|s8|unsigned char)\s*\*\s*\)\s*)?&?\s*([A-Za-z_]\w*)\s*\+\s*(0[xX][0-9A-Fa-f]+|\d+)\b"
    r"(?!\s*[*/%<>])"  # a literal that is the left operand of a product, quotient or shift is scaled, not a byte offset
)
# An operand of a bitwise operator is a mask, never an address: `&`, `|` or `^` (a literal has no address to take, so
# `& 0x800FFFFF` is the binary and), `~`, or their assignments, before the literal; the binary operator after it.
_MASKED = re.compile(r"(?:(?<![&|])[&|^]=?|~)[\s(]*\Z")
_MASKS = re.compile(r"[\s)]*[&|^](?![&|])")


def _mask(code: str, start: int, end: int) -> bool:
    """Whether the literal at [start, end) is an operand of a bitwise operator."""
    return bool(_MASKED.search(code, max(0, start - 24), start) or _MASKS.match(code, end))


def _sdk(config_: Any, path: str) -> bool:
    """Whether a path sits in a vendored library directory: its source idioms are not ours to judge."""
    return any(Path(path).is_relative_to(folder) for folder in getattr(config_.project, "libraries", ()))


def _tree(snapshot: Snapshot) -> tree.Tree:
    """The tree this snapshot is: the one it holds, else its overlays over the working tree, captured once per
    snapshot digest (which names the overlays), however many rules ask."""
    if snapshot.files is not None and not snapshot.overlays:
        held: tree.Tree = snapshot.files
        return held
    return effort.memo(("policy.tree", snapshot.digest), lambda: tree.capture(snapshot.config, snapshot.overlays))


def _module(snapshot: Snapshot, path: str) -> tree.Module | None:
    return next((m for m in _tree(snapshot).modules if m.path == path), None)


def _held(module: tree.Module) -> list[str]:
    return sorted(v for v, names in module.owns.items() if names)


def _unit(snapshot: Snapshot, module: tree.Module) -> UnitSpec:
    return recipes.unit(snapshot.config, module)


def _extents(snapshot: Snapshot, version: str) -> tuple[list[int], list[int], dict[str, tuple[int, int]]]:
    """The sorted start and end of every symbol address in a version (a symbol ends where the next starts, or where
    its segment does) and each name's extent, once per command."""

    def build() -> tuple[list[int], list[int], dict[str, tuple[int, int]]]:
        ver = snapshot.versions[version]
        segments = [(vram, vram + stop - start) for _, start, stop, vram in ver.segments]
        starts = sorted(set(ver.symbols.values()))
        stops: list[int] = []
        for index, address in enumerate(starts):
            cap = next((end for first, end in segments if first <= address < end), address + 4)
            nxt = starts[index + 1] if index + 1 < len(starts) else cap
            stops.append(min(nxt, cap) if min(nxt, cap) > address else address + 4)
        table = dict(zip(starts, stops, strict=True))
        named = {name: (address, table[address]) for name, address in ver.symbols.items()}
        return starts, stops, named

    return effort.memo(("extents", snapshot.digest, version), build)


def _scalar(unit: str, name: str) -> bool:
    """Whether the translation unit declares `name` only as an object that is no array (an int, a u64, a pointer): its
    `NAME + K` adds to the value it holds, not to its address. Unknown (no declaration seen) is no scalar."""
    found = False
    for m in re.finditer(rf"\b{re.escape(name)}\b\s*([\[(=;,)]?)", unit):
        if m[1] in ("[", "("):
            return False  # an array or a function (or a use that indexes or calls it)
        found = found or (
            m[1] in (";", "=", ",") and _DECLARES.search(unit, max(0, m.start() - 80), m.start()) is not None
        )
    return found


_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*["<]([^">]+)[">]', re.M)


def _included(snapshot: Snapshot, path: str, code: str) -> str:
    """The module's text and the text of every project header it includes, transitively (no preprocessor run: what
    a check without a source view can read of the declarations the module sees)."""
    texts, seen, todo = [code], {path}, [(path, code)]
    while todo and len(seen) < 400:
        at, text = todo.pop()
        for name in _INCLUDE.findall(text):
            for candidate in (f"include/{name}", f"{Path(at).parent.as_posix()}/{name}", f"src/{name}"):
                if candidate in seen:
                    break
                data = snapshot.peek(candidate)
                if data is not None:
                    seen.add(candidate)
                    texts.append(data.decode(errors="replace"))
                    todo.append((candidate, texts[-1]))
                    break
    return "\n".join(texts)


_DECLARES = re.compile(
    r"(?:^|[;{}\n])\s*(?!(?:return|case|goto|sizeof|else|do)\b)(?:extern\s+|static\s+|volatile\s+|const\s+)*"
    r"[A-Za-z_][\w\s]*\**\s*\Z"
)  # a type before the name, no statement keyword


def _address_literals(snapshot: Snapshot, path: str, code: str, view: SourceView | None = None) -> dict[int, str]:
    """Lines that spell a ROM address as a number: a literal in the mapped address space (from _KSEG0) inside a symbol's
    range in a version holding the module, or `NAME + K` that reaches past the data symbol NAME in every holder, unless
    NAME is a scalar the unit declares (an int, u64 or pointer global: the sum is of its value) and no `&` takes its
    address."""
    module = _module(snapshot, path)
    if module is None:
        return {}
    held = _held(module)
    indexes = [_extents(snapshot, v) for v in held]
    line = _numbered(code)
    hits: dict[int, str] = {}
    for match in _NUMBER.finditer(code):
        value = _integer(match["int"]) if match["int"] else None
        if value is None or value < _KSEG0 or _mask(code, match.start(), match.end()):
            continue
        for starts, ends, _ in indexes:
            at = bisect_right(starts, value) - 1
            if at >= 0 and value < ends[at]:
                hits.setdefault(line(match.start()), match["int"])
                break
    unit: str | None = None  # the declarations the module sees, read at the first sum that needs them
    for match in _OFFSET.finditer(code):
        sizes = [
            e[2][match[1]][1] - e[2][match[1]][0]
            for v, e in zip(held, indexes, strict=True)
            if match[1] in e[2] and match[1] not in snapshot.versions[v].code
        ]
        if sizes and (reach := _integer(match[2])) is not None and all(reach >= size for size in sizes):
            if "&" not in match[0]:
                if unit is None:
                    unit = view.text if view is not None else _included(snapshot, path, code)
                if _scalar(unit, match[1]):
                    continue
            hits.setdefault(line(match.start()), " ".join(match[0].split()))
    return hits


_EXTERN = re.compile(r"^[ \t]*extern\b[^;{}]*;", re.M)
_EXTERN_NAME = (
    re.compile(r"\(\s*\*\s*(\w+)\s*\)"),
    re.compile(r"(\w+)\s*\("),
    re.compile(r"(\w+)\s*(?:\[[^\]]*\]\s*)*;$"),
)


def _externs(code: str) -> dict[str, tuple[str, ...]]:
    """Every `extern` declaration of one header's code, by the name it declares, whitespace normalised."""
    found: dict[str, list[str]] = defaultdict(list)
    for match in _EXTERN.finditer(code):
        text = " ".join(match[0].split())
        name = next((m[1] for pattern in _EXTERN_NAME if (m := pattern.search(text))), None)
        if name:
            found[name].append(text)
    return {name: tuple(rows) for name, rows in found.items()}


def _header_externs(snapshot: Snapshot, text: str) -> dict[str, tuple[str, ...]]:
    """The externs of one header, kept by its text (its blob sha) in the `header.names` kind."""
    return store.cached_object(
        snapshot.config, "header.names", digest(("externs", text)), lambda: _externs(_scopes(text)["code"])
    )


def _include_headers(snapshot: Snapshot) -> list[str]:
    root = snapshot.config.project.root
    found = {p.relative_to(root).as_posix() for p in (root / "include").rglob("*") if p.is_file()}
    found |= {p for p, data in snapshot.overlays.items() if data is not None and p.startswith("include/")}
    found -= {p for p, data in snapshot.overlays.items() if data is None}
    return sorted(p for p in found if config.is_header(p))


def header_lint(snapshot: Snapshot, only: Collection[str] | None = None) -> tuple[Finding, ...]:
    """Each extern is declared at most once across include/, and one name has one declaration everywhere."""
    seen: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in _include_headers(snapshot):
        for name, found in _header_externs(snapshot, snapshot.read(path).decode(errors="replace")).items():
            seen[name].extend((path, row) for row in found)
    out: list[Finding] = []
    for name, rows in sorted(seen.items()):
        if len(rows) < 2:
            continue
        first = rows[0]
        for path, row in rows[1:]:
            key = "source.extern-conflict" if row != first[1] else "source.extern-redeclared"
            reason = f"{config.sentence(key)}: {name} (first declared in {first[0]})"
            if only is None or path in only:
                out.append(Finding(key, reason, path=path, unit=name, blocking=False))
    return tuple(out)


# Type-debt checks over code text with comments and strings blanked: each returns {line: subject} per row.
# the project's function symbol, in either case (a rename capitalizes it: Func_802A274C): no record type may be named
# after one
_SYMBOL = r"[Ff]unc_[0-9A-Fa-f]{8}"
# a name carrying a function symbol (Actor_func_802A274C_de, Func_802A274C_Record_SharedActor)
_NAMED = re.compile(rf"\b(?:\w+?_)?{_SYMBOL}\w*")
# a per-function view numbered _S<n> or _View<n>, whatever follows it (func_X_S1_SharedActor)
_VIEW = re.compile(rf"\b{_SYMBOL}(?:_\w+?)?_(?:S|View)\d+(?:_\w+)?\b")
_TAG = re.compile(rf"\b(?:struct|union)\s+(\w*{_SYMBOL}\w*)")  # a tag that carries a function symbol
_SIZES = {1: "u8 s8 char bool", 2: "u16 s16 short", 4: "u32 s32 int long float f32 void", 8: "u64 s64 double f64"}
_BYTES = {name: size for size, names in _SIZES.items() for name in names.split()}
_PRIMITIVE = frozenset(_BYTES) | {"unsigned", "signed", "const", "volatile", "size_t", "uintptr_t", "intptr_t"}
_INTEGER = frozenset(_PRIMITIVE - {"float", "f32", "double", "f64", "void", "bool"})
_PLACEHOLDER = re.compile(  # an offset name: pad*, unk<hex>, unknown<hex>, field_<hex>, _<hex>
    r"^(?:pad\w*|unk(?:nown)?_?[0-9A-Fa-f]*|field_[0-9A-Fa-f]+|_[0-9A-Fa-f]+)$", re.I
)


def _line_of(code: str) -> Callable[[int], int]:
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


def _fake_type(code: str) -> dict[int, str]:
    """The first line of each distinct invented per-function record name: numbered views, and record types (declared
    here or used as a type) whose name carries a function symbol. A data symbol carrying one is not a type."""
    line = _line_of(code)
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
    """Whether the struct opened by `head` is mostly padding arrays, and where it ends. Every declarator of a field
    counts (`u32 a, b, c;` names three, a bitfield names one)."""
    start, stop = _body(code, head.end() - 1)
    pad = named = total = 0
    for field in code[start:stop].split(";"):
        first = re.match(r"\s*(?:(?:const|volatile|unsigned|signed|struct|union|enum)\s+)*([A-Za-z_]\w*)", field)
        for part in field.split(","):
            match = re.search(
                r"(\*?)\s*([A-Za-z_]\w*)\s*(?:\[\s*(0[xX][0-9A-Fa-f]+|\d+)\s*\])?\s*(?::\s*\w+\s*)?$", part.strip()
            )
            if not first or not match or (match[2] == first[1] and not match.start(2)):
                continue
            size = 4 if match[1] else _BYTES.get(first[1], 4) * int(match[3] or "1", 0)
            total += size
            if match[3] and _PLACEHOLDER.match(match[2]):
                pad += size
            else:
                named += 1
    return bool(pad and 2 * pad >= total and named <= 3), stop


_OWNER_RECORD = re.compile(r"Data[A-Z0-9][A-Za-z0-9]*")  # recipe's owner-local record of one data symbol


def _owner_record(code: str, name: str) -> bool:
    """Whether `name` is the owner-local record of a data symbol this text declares as it or as a pointer to it: its
    members are every offset the symbol's users access, the bytes between no one reads, so its
    padding is the evidence, not a view standing in for real fields."""
    return bool(_OWNER_RECORD.fullmatch(name)) and bool(
        re.search(rf"^[ \t]*extern\s+(?:struct\s+)?{name}\s*\*?\s*[A-Za-z_]\w*\s*;", code, re.M)
    )


def _named(code: str, head: re.Match[str]) -> str:
    """The typedef alias, else the tag, of the struct opened by `head`."""
    alias = re.match(r"\}\s*(\w+)\s*;", code[_body(code, head.end() - 1)[1] :])
    tag = re.match(r"struct\s+(\w+)", head[0])
    return alias[1] if alias else tag[1] if tag else ""


def _pad_view(code: str) -> dict[int, str]:
    """Struct definitions where pad arrays (offset-named: pad*, unk<hex>, field_<hex>) are at least half the bytes and
    at most three fields are named; a data symbol's owner-local record is none."""
    line = _line_of(code)
    found: dict[int, str] = {}
    for head in re.finditer(r"\bstruct\s*(?:\w+\s*)?\{", code):
        if _padded(code, head)[0] and not _owner_record(code, name := _named(code, head)):
            # an anonymous record is named by its alias: two distinct views never read as one subject
            spelled = " ".join(head[0].split())
            found[line(head.start())] = spelled if not name or name in spelled else f"struct {name} {{"
    return found


def _placeholder_record(code: str) -> dict[int, str]:
    """Records this unit defines whose every field is an offset placeholder (unk4, field_8, pad0) and that the unit
    then uses by name: a layout invented here instead of the project's real record."""
    line = _line_of(code)
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


def _view_types(code: str) -> set[str]:
    """The names (tag and typedef) of the records this text defines as a union or a pad view."""
    names: set[str] = set()
    for head in re.finditer(r"(?<![\w])(?:(typedef)\s+)?(struct|union)\s*(\w+)?\s*\{", code):
        padded, stop = _padded(code, head) if head[2] == "struct" else (False, _body(code, head.end() - 1)[1])
        alias = re.match(r"\}\s*(\w+)\s*;", code[stop:]) if head[1] else None
        if padded and _owner_record(code, alias[1] if alias else head[3] or ""):
            continue
        if head[2] != "union" and not padded:
            continue
        names.update(n for n in (head[3], alias[1] if alias else None) if n)
    return names


def _views_named(code: str, views: set[str]) -> dict[int, str]:
    """The first line on which this text names each of the given view records (a definition is not a use)."""
    if not views:
        return {}
    line = _line_of(code)
    first: dict[str, int] = {}
    names = "|".join(map(re.escape, sorted(views)))
    for match in re.finditer(rf"\b(?:(struct|union)\s+)?({names})\b(?!\s*\{{)", code):
        first.setdefault(match[2], line(match.start()))
    return {at: name for name, at in first.items()}


def _parameters(code: str) -> list[dict[str, Any]]:
    """Every definition parameter declared void* or integer: its function, name, index, span, and each cast of it.
    A cast row holds the span from the cast's open parenthesis through the parameter name, the spelled target
    and its qualifiers."""
    line = _line_of(code)
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


def _param_cast(code: str) -> dict[int, str]:
    """Function _parameters declared void* or integer that the body casts to a pointer of a non-primitive type. The
    subject is the function called, "f()": its head holds every such parameter on one line, so retyping one of them
    leaves the row of the others as it was instead of reading as new debt, and no bare name lets the row withhold the
    function's callers."""
    hits: dict[int, str] = {}
    for row in _parameters(code):  # an integer parameter's debt is the address-int row of its callers, not this one
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


def _address_int(code: str) -> dict[int, str]:
    """An address taken and cast to an integer type (the null-base offsetof idiom is no address): the callee or field
    is mistyped, or a ROM or segment symbol should be an array used without the address-of. Inside a function the
    subject is the function, "f()" with the cast type: a retype that respells the operand (`(s32)&((T *)a)->f` ->
    `(s32)&a->f`) is the same row, never new debt."""
    line = _line_of(code)
    holders = _holders(code) if _ADDRESS_INT.search(code) else []
    hits: dict[int, str] = {}
    for m in _ADDRESS_INT.finditer(code):
        holder = next((name for start, stop, name in holders if start <= m.start() < stop), "")
        cast = " ".join(m[1].split())
        hits.setdefault(line(m.start()), f"({cast})&{holder}()" if holder else f"({cast})&{m[2]}")
    return hits


_STAND_IN = re.compile(r"\bINCLUDE_(?:ASM|DATA|RODATA|BSS)\s*\(")


def _stand_in_body(code: str) -> dict[int, str]:
    """A stand-in line is a whole item of the file: one inside a function body is inline assembly in disguise."""
    line = _line_of(code)
    found: dict[int, str] = {}
    for match in _STAND_IN.finditer(code):
        if code.count("{", 0, match.start()) > code.count("}", 0, match.start()):
            found.setdefault(line(match.start()), match[0].rstrip("( \t"))
    return found


_RECORD = re.compile(r"\b(?:typedef\s+)?(?:struct|union)\s*(\w*)\s*\{")


def _records(code: str) -> list[tuple[int, str, str]]:
    """(line, name, body with whitespace squeezed) of every record this text defines. A record is named by its typedef
    alias, else its tag; a body with fewer than two fields is not worth comparing."""
    line, found = _line_of(code), []
    for head in _RECORD.finditer(code):
        start, stop = _body(code, head.end() - 1)
        alias = re.match(r"\s*\*?\s*(\w+)\s*[;,\[]", code[stop + 1 :]) if head[0].startswith("typedef") else None
        name = alias[1] if alias else head[1]
        body = " ".join(code[start:stop].split())
        if name and body.count(";") >= 2:
            found.append((line(head.start()), name, body))
    return found


def _record_index(snapshot: Snapshot) -> dict[str, list[tuple[str, str]]]:
    """body -> [(path, name)] over the headers and the modules of the tree, once per snapshot."""

    def build() -> dict[str, list[tuple[str, str]]]:
        index: dict[str, list[tuple[str, str]]] = defaultdict(list)
        paths = [*_include_headers(snapshot), *(m.path for m in _tree(snapshot).modules)]
        for path in sorted(paths):
            data = snapshot.peek(path)
            if data is not None:
                for _, name, body in _records(_scopes(data.decode(errors="replace"))["code"]):
                    index[body].append((path, name))
        return dict(index)

    return effort.memo(("records", snapshot.digest), build)


def _duplicate_records(snapshot: Snapshot, path: str, code: str) -> dict[int, str]:
    """A record whose body is another record's under another name: the one defined first (by path) is the canonical
    one, any later is a duplicate to fold into it. Fed from the tree's own sources, not a recovered census."""
    index, found = _record_index(snapshot), dict[int, str]()
    for number, name, body in _records(code):
        others = sorted(index.get(body, ()))
        canonical = next(((p, n) for p, n in others if n != name), None)
        if canonical and (canonical[0], canonical[1]) < (path, name):
            found.setdefault(number, f"{name} repeats {canonical[1]} ({canonical[0]})")
    return found


_DEBT = {
    "fake-type": _fake_type,
    "pad-view": _pad_view,
    "placeholder-record": _placeholder_record,
    "param-cast": _param_cast,
    "address-int": _address_int,
}


def _hits(
    rule: Json,
    path: str,
    scopes: Mapping[str, str],
    rules: Json,
    sdk_group: bool,
    snapshot: Snapshot | None,
    view: SourceView | None,
) -> dict[int, str]:
    """{line: subject} of one rule, the subject being the text that identifies the finding across edits around it."""
    code = scopes["code"]
    if "regex" in rule:
        if rule.get("files") == "source" and config.is_header(path):
            return {}
        scoped = scopes[rule["scope"]]
        number = _numbered(scoped)
        found: dict[int, str] = {}
        for m in re.finditer(rule["regex"], scoped, re.M):
            found.setdefault(number(m.start() + len(m[0]) - len(m[0].lstrip())), m[0].strip())
        if rule["id"] == "union-view" and found:
            sdk = _sdk_union_lines(scoped)
            found = {n: t for n, t in found.items() if n not in sdk}
        return found
    check = rule["check"]
    if check == "placeholder-record" and config.is_header(path):
        return {}
    if check in _DEBT:
        return dict(_DEBT[check](code))
    if check == "stand-in-body":
        return _stand_in_body(code)
    if check == "duplicate-record":
        return _duplicate_records(snapshot, path, code) if snapshot else {}
    if check == "empty-loop":
        lines = _empty_loops(code)
    elif check == "volatile":
        active = code if sdk_group or not view else _scopes(_active(view, path))["code"]
        lines = set() if sdk_group else _volatile(active, rules["hardware"]["address_ranges"])
    elif check == "shared-declarations":
        lines = _shared(code, view, path, snapshot) if view else set()
    elif check == "address-literal":
        return _address_literals(snapshot, path, code, view) if snapshot else {}
    else:
        lines = _raw_gfx(code)
    return dict.fromkeys(lines, "")


_UNION_DEF = re.compile(r"\btypedef\s+union\b(?:\s+(\w+))?\s*\{")


def _union_typedefs(code: str) -> Iterator[tuple[int, int, set[str]]]:
    """(start, end, names) of each `typedef union ... { ... } Name;` of the code: its tag and typedef name."""
    for m in _UNION_DEF.finditer(code):
        depth, at = 0, m.end() - 1
        for at in range(m.end() - 1, len(code)):
            depth += {"{": 1, "}": -1}.get(code[at], 0)
            if depth == 0:
                break
        tail = re.match(r"\s*(\w+)\s*;", code[at + 1 :])
        yield m.start(), at + 1 + (tail.end() if tail else 0), {n for n in (m[1], tail and tail[1]) if n}


def _sdk_unions() -> frozenset[str]:
    """The union types the SDK itself declares (Gfx, Acmd: 8-byte aligned command words), from the packaged SDK
    headers: a project header that spells one of them is the SDK's type, not a union view."""
    names = {
        n for t in ("sdk/gfx.h", "sdk/acmd.h") for _, _, found in _union_typedefs(config.template(t)) for n in found
    }
    return frozenset(names)


def _sdk_union_lines(code: str) -> set[int]:
    """The lines of the code that define or name an SDK union type."""
    sdk = _sdk_unions()
    lines: set[int] = set()
    for start, end, names in _union_typedefs(code):
        if names & sdk:
            lines.update(range(code.count("\n", 0, start) + 1, code.count("\n", 0, end) + 2))
    for m in re.finditer(r"\bunion\s+(\w+)", code):
        if m[1] in sdk:
            lines.add(code.count("\n", 0, m.start()) + 1)
    return lines


def _findings(
    snapshot: Snapshot | None,
    path: str,
    text: str,
    view: SourceView | None,
    sdk_group: bool,
    only: Collection[str] | None,
) -> tuple[Finding, ...]:
    rules = config.load_resource("rules.toml")
    scopes, lines, findings = {**_scopes(text), "text": text}, text.splitlines(), []
    for rule in rules["rule"]:
        if (only is not None and rule["id"] not in only) or (sdk_group and rule["id"] == "union-view"):
            continue
        if snapshot is None and rule.get("needs"):
            continue
        hits = _hits(rule, path, scopes, rules, sdk_group, snapshot, view)
        comment_lines = scopes["comments"].splitlines()
        for n in sorted(hits):
            adjacent = "\n".join(comment_lines[max(0, n - 2) : n])
            waivers = re.findall(r"/\*\s*FAKEMATCH:((?:[^*]|\*(?!/))*)\*/", adjacent)
            if rule["waivable"] and any(reason.strip() for reason in waivers):
                continue
            subject = hits[n] or (lines[n - 1].strip()[:120] if n <= len(lines) else "")
            findings.append(Finding(f"source.{rule['id']}", f"{rule['sentence']}: {subject}", path=path, line=n))
    return tuple(findings)


def text_findings(path: str, text: str, sdk_group: bool) -> tuple[Finding, ...]:
    """The findings of every rule that reads the text alone, with no snapshot, ROM or host."""
    if config.is_assembly(path) and not sdk_group:
        return (
            Finding("source.library-asm", "assembly source must be inside a configured library directory", path=path),
        )
    return _findings(None, path, text, None, sdk_group, None)


def evaluate(
    snapshot: Snapshot,
    path: str,
    text: str,
    view: SourceView | None,
    sdk_group: bool,
    only: Collection[str] | None = None,
) -> tuple[Finding, ...]:
    with effort.stage("policy.evaluate"):
        if _sdk(snapshot.config, path):
            return ()  # vendored source idioms; compile and ROM proof are still mandatory
        if config.is_assembly(path):
            return text_findings(path, text, False)
        return _findings(snapshot, path, text, view, sdk_group, only)


_IDENT = re.compile(r"[A-Za-z_]\w*")


def _placements(read: Callable[[str], bytes | None]) -> dict[str, frozenset[tuple[str, int]]]:
    """name -> {(version, vram)} of every symbol the version symbol files of one tree place, read through `read`
    (None: no such file). A symbol is where it sits, so its placements are its identity, whatever it is called."""
    raw = read("config.toml")
    places: defaultdict[str, set[tuple[str, int]]] = defaultdict(set)
    for version in tomllib.loads(raw.decode()).get("version", {}) if raw else ():
        file = versions.symbol_file(version)
        data = read(file)
        for row in versions.parse_symbol_addrs(data, file) if data else ():
            places[row.name].add((version, row.vram))
    return {name: frozenset(p) for name, p in places.items()}


def _renames(
    before: Mapping[str, frozenset[tuple[str, int]]], after: Mapping[str, frozenset[tuple[str, int]]]
) -> dict[str, str]:
    """old name -> new name of every symbol whose old spelling is gone while exactly one name of `after` holds one of
    its placements: a rename (recipe naming) keeps the symbol, so the rows that name it keep their identity."""
    held: defaultdict[tuple[str, int], set[str]] = defaultdict(set)
    for name, places in after.items():
        for place in places:
            held[place].add(name)
    out: dict[str, str] = {}
    for name, places in before.items():
        new = {n for place in places for n in held.get(place, ())}
        if name not in after and len(new) == 1:
            out[name] = new.pop()
    return out


def scope(
    before: Sequence[Finding],
    after: Sequence[Finding],
    writes: Collection[str],
    renamed: Mapping[str, str] | None = None,
) -> tuple[tuple[Finding, ...], tuple[Finding, ...]]:
    # A row is its rule and subject, wherever it is: moving text between files is free and a repeated row is new once
    # it outnumbers the old text's, so the repo-wide total of a rule never rises through any path. When a total does
    # rise, the paths blamed are the ones whose own count of that row rose (never one whose count is unchanged), each
    # reported with its count before and after. A subject names symbols by identity, not spelling: an old row that
    # spells a renamed symbol (`renamed`: old -> new name) is read with its new name, so a rename is as free as a move.
    names = renamed or {}

    def spelled(reason: str) -> str:
        return _IDENT.sub(lambda m: names.get(m[0], m[0]), reason) if names else reason

    def rows(findings: Sequence[Finding], read: Callable[[str], str]) -> Counter[tuple[tuple[str, str], str]]:
        return Counter(((f.key, read(f.reason)), f.path) for f in findings)

    old, new = rows(before, spelled), rows(after, str)
    total: Counter[tuple[str, str]] = Counter()
    for (row, _), n in new.items():
        total[row] += n
    for (row, _), n in old.items():
        total[row] -= n
    blocking, debt = [], []
    seen: Counter[tuple[tuple[str, str], str]] = Counter()
    for finding in after:
        row = (finding.key, finding.reason)
        at = (row, finding.path)
        seen[at] += 1
        if finding.path in writes and total[row] > 0 and seen[at] > old[at]:
            blocking.append(replace(finding, blocking=True))
        else:
            debt.append(replace(finding, blocking=False))
    return tuple(blocking), tuple(debt)


def fresh(snapshot: Snapshot, path: str, findings: Sequence[Finding], sdk_group: bool) -> tuple[Finding, ...]:
    """The blocking findings the landed text of path did not already carry: its old debt never blocks its next text."""
    findings = tuple(f for f in findings if f.blocking)
    if not findings:
        return ()
    old = snapshot.peek(path)
    if old is None:
        return scope((), findings, {path})[0]
    viewed = None  # a rule that reads the preprocessed view judges the old text through the old view too
    module = _module(snapshot, path)
    if module is not None and {f.key.removeprefix("source.") for f in findings} & _VIEWED:
        unit = _unit(snapshot, module)
        try:
            viewed = _view.get(
                snapshot, unit, _census_version(snapshot, module), recipes.resolve(snapshot.config, unit, {})
            )
        except Refusal:  # the old text has no view: its view rules found nothing
            viewed = None
    return scope(evaluate(snapshot, path, old.decode(), viewed, sdk_group), findings, {path})[0]


_VIEWED = frozenset({"shared-declarations", "volatile-storage"})  # the checks that read a source view (`_hits`)


def _census_version(snapshot: Snapshot, module: tree.Module) -> str:
    """The version a module is read in: the naming version when it owns names there, else its first holder."""
    holders = _held(module)
    names = snapshot.config.project.names_from
    return names if names in holders or not holders else holders[0]


def _census_unit(item: tuple[Snapshot, tree.Module]) -> tuple[Finding, ...]:
    snapshot, module = item
    unit = _unit(snapshot, module)
    source = _view.get(snapshot, unit, _census_version(snapshot, module), recipes.resolve(snapshot.config, unit, {}))
    sdk = _sdk(snapshot.config, module.path)
    text = snapshot.read(module.path).decode()
    # evaluate reads the unit text and the active lines of the unit and its headers: other edits keep the verdict warm
    files = [module.path, *(p for p, _ in source.dependencies if config.is_header(p))]
    key = digest(
        (
            text,
            sdk,
            config.load_resource("rules.toml"),
            snapshot.config.project.digest,
            code_identity(evaluate, _view_uses, _stand_in_body, _duplicate_records, *_DEBT.values()),
            [_active(source, p) for p in files],
        )
    )
    return store.cached_object(
        snapshot.config,
        "census",
        key,
        lambda: (
            *evaluate(snapshot, module.path, text, source, sdk),
            *_view_uses(module.path, text, snapshot, files[1:]),
        ),
    )


def _view_uses(path: str, text: str, snapshot: Snapshot, reached: Sequence[str]) -> tuple[Finding, ...]:
    """MEASURE ONLY (no rule, never unclean): one finding per union or pad-view record that a header defines and the
    unit names."""
    views = set().union(*(_view_types(_scopes(snapshot.read(p).decode())["code"]) for p in reached))
    found = _views_named(_scopes(text)["code"], views)
    return tuple(
        Finding(
            "source.uses-view-type",
            f"{config.sentence('source.uses-view-type')}: {name}",
            path=path,
            line=at,
            unit=path,
        )
        for at, name in sorted(found.items())
    )


def _census_warnings(item: tuple[Snapshot, tree.Module]) -> tuple[Finding, ...]:
    """The unit's landed view conversions as debt by count, read from what the build kept and never compiled here."""
    snapshot, module = item
    unit = _unit(snapshot, module)
    found = native.recorded(
        snapshot, unit, _census_version(snapshot, module), recipes.resolve(snapshot.config, unit, {})
    )
    if not found:
        return ()
    return (
        Finding(
            "land.view-conversion",
            f"{len(found)} integer-pointer conversions the compiler reports",
            path=module.path,
            missing=found,
            action="type each variable as the pointer it holds",
        ),
    )


def _census_key(item: tuple[Snapshot, tree.Module]) -> str:
    """Everything a build of the unit reads: warm while none of it changed, so the pool dispatches nothing."""
    snapshot, module = item
    reads = _view.closure(snapshot, _unit(snapshot, module), _census_version(snapshot, module))
    return digest(
        (
            reads,
            _sdk(snapshot.config, module.path),
            config.load_resource("rules.toml"),
            snapshot.config.project.digest,
            code_identity(_census_unit, _stand_in_body, _duplicate_records, *_DEBT.values()),
        )
    )


def _census_header(item: tuple[Snapshot, str, bool]) -> tuple[Finding, ...]:
    snapshot, path, sdk = item  # a header is judged by its text alone, so one verdict serves every unit
    return evaluate(snapshot, path, snapshot.read(path).decode(), None, sdk)


def _header_key(item: tuple[Snapshot, str, bool]) -> str:
    snapshot, path, sdk = item
    return digest(
        (
            snapshot.read(path).decode(),
            path,
            sdk,
            config.load_resource("rules.toml"),
            snapshot.config.project.digest,
            code_identity(_census_header, *_DEBT.values()),
        )
    )


def _headers_read(snapshot: Snapshot, wanted: Collection[str] | None = None) -> dict[str, bool]:
    """Every header some module reaches, SDK only when every module that reads it is."""
    headers: dict[str, bool] = {}
    for module in _tree(snapshot).modules:
        if wanted is not None and module.path not in wanted:
            continue
        sdk = _sdk(snapshot.config, module.path)
        for path, _ in _view.closure(snapshot, _unit(snapshot, module), _census_version(snapshot, module)):
            if config.is_header(path):
                headers[path] = headers.get(path, True) and sdk
    return headers


def _changed_findings(snapshot: Snapshot, texts: Mapping[str, str]) -> list[Finding]:
    """The findings of these texts in this snapshot."""
    headers = _headers_read(snapshot) if any(config.is_header(p) for p in texts) else {}
    findings: list[Any] = []
    for path, text in texts.items():
        if not config.is_header(path) and _module(snapshot, path) is None:
            continue
        sdk = _sdk(snapshot.config, path) if not config.is_header(path) else headers.get(path, False)
        if text and not sdk:
            findings.extend(evaluate(snapshot, path, text, None, sdk))
    return findings


def census(
    snapshot: Snapshot,
    also: Sequence[tuple[Any, ...]] = (),
    changes: Mapping[str, tuple[str, str]] | None = None,
    baseline: Snapshot | None = None,
    only: Collection[str] | None = None,
) -> tuple[tuple[Finding, ...], list[list[Any]]]:
    """Every source once and every header once, however many units include it. The pool groups `also` run in the same
    dispatch so their jobs fill the workers the census leaves idle; their results come back beside the findings.
    With changes, compare only those before/after texts, without preprocessing or compiling. With `only`, judge just
    those unit paths and the headers they read: the snapshot-wide listings are filtered, never recomputed per path."""
    with effort.stage("policy.census"):
        if changes == {}:
            return (), []
        if changes is not None:
            before = _changed_findings(baseline or snapshot, {p: old for p, (old, _) in changes.items()})
            after = _changed_findings(snapshot, {p: new for p, (_, new) in changes.items()})
            blocking, debt = scope(before, after, changes)
            return (*blocking, *debt), []
        wanted = None if only is None else frozenset(only)
        items = [(snapshot, m) for m in _tree(snapshot).modules if wanted is None or m.path in wanted]
        headers = _headers_read(snapshot, wanted)
        found = pool.gather(
            snapshot.config,
            [
                ("policy.census", _census_unit, items, _census_key),
                (
                    "policy.census.headers",
                    _census_header,
                    [(snapshot, path, sdk) for path, sdk in sorted(headers.items())],
                    _header_key,
                ),
                # what a build kept is read where the units are judged, on all cores (no key: it changes with the build)
                ("policy.census.warnings", _census_warnings, items),
                *also,
            ],
        )
        unique = {}
        lint = header_lint(snapshot, None if wanted is None else [*sorted(wanted), *headers])
        for findings in (*found[0], *found[1], *found[2], lint):
            for finding in findings:
                unique[(finding.key, finding.path, finding.line, finding.unit)] = replace(finding, blocking=False)
        return tuple(unique.values()), found[3:]


def unclean(snapshot: Snapshot) -> frozenset[str]:
    """The functions a finding of a rules.toml `unclean` rule withholds: no exact credit."""
    return frozenset(unclean_rules(snapshot))


def unclean_rules(snapshot: Snapshot, only: Collection[str] | None = None) -> dict[str, tuple[str, ...]]:
    """Per function a finding of an `unclean` rule of rules.toml withholds (see `withheld`), the keys of the rules
    that fire; with `only`, just the functions of those units are judged."""
    rules = config.load_resource("rules.toml")
    ids = rules.get("unclean", ())
    if not ids:
        return {}

    def judge() -> dict[str, tuple[str, ...]]:
        paths = [m.path for m in _tree(snapshot).modules if only is None or m.path in only]
        items = [(snapshot, path, ids, rules["rule"]) for path in paths]
        if only is not None:  # a few units judged for a caller: no record of the whole project
            verdicts = pool.map(snapshot.config, "policy.unclean", _unclean_unit, items, _unclean_key)
        else:  # a landing judges again only the units whose text, reach or rows moved
            verdicts = delta.map(
                snapshot.config,
                "policy.unclean",
                _unclean_job,
                items,
                key=_unclean_key,
                ident=lambda item: item[1],
                stamp=lambda item: delta.fingerprint(_unclean_rows(item[0], item[1])),
                pin=lambda path: _view.pin(snapshot, path),
                context=(_unclean_context(snapshot), digest((ids, rules["rule"]))),
            )
        return {name: fired for verdict in verdicts for name, fired in verdict}

    return effort.memo(("unclean", snapshot.digest, None if only is None else tuple(sorted(only))), judge)


def _unclean_unit(item: tuple[Snapshot, str, Sequence[str], Any]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    snapshot, path, ids, _ = item
    text = snapshot.read(path).decode()
    functions = credited(snapshot, path, text)
    return tuple(sorted(withheld(text, evaluate(snapshot, path, text, None, False, ids), functions).items()))


def credited(snapshot: Snapshot, path: str, text: str | None = None) -> list[str]:
    """The functions of a unit that earn credit of their own: every function its text defines that has an address in
    the version it is read in. A defined function with no address was inlined into the functions that name it, so its
    findings withhold them instead."""
    module = _module(snapshot, path)
    text = snapshot.read(path).decode() if text is None else text
    placed = snapshot.versions[_census_version(snapshot, module)].symbols if module else {}
    return list(dict.fromkeys(h for _, _, h, _, _ in _top(_scopes(text)["code"]) if h and h in placed))


def pieces(snapshot: Snapshot, member: str, version: str) -> list[tuple[str, int]]:
    """(function, bytes) of a function's .text in one version: one piece, the bytes between its address and the next
    symbol's. A version that does not hold the function has none."""
    extent = _extents(snapshot, version)[2].get(member)
    return [(member, extent[1] - extent[0])] if extent else []


def withheld_bytes(snapshot: Snapshot, rules: Collection[str], member: str, version: str) -> int:
    """The .text bytes of a function in one version that unclean verdicts withhold."""
    if not rules:
        return 0
    return sum(size for name, size in pieces(snapshot, member, version) if name in rules)


_C_KEYWORDS = _NOT_NAMES | {"char", "const", "double", "extern", "float", "int", "long", "short", "signed", "static"}
_C_KEYWORDS |= {"typedef", "unsigned", "void", "volatile", "register", "auto"}


def withheld(text: str, findings: Sequence[Finding], functions: Collection[str]) -> dict[str, tuple[str, ...]]:
    """The functions of one source that its findings withhold credit from, with the keys of the rules: a finding
    inside a function withholds that function. What a finding names (a file-scope record, typedef or global it sits
    in, a type it reports once, a helper it sits in) withholds every function that names it, also through the
    file-scope declarations that name it in turn. A finding no function names withholds every function."""
    statements = list(_top(_scopes(text)["code"]))
    bodies = [(holder, words) for _, _, holder, words, _ in statements if holder in functions]
    declarations = [(words, declares) for _, _, holder, words, declares in statements if holder is None]
    found: defaultdict[str, set[str]] = defaultdict(set)
    for finding in findings:
        empty: set[str] = set()
        holder, declares = next(((s[2], s[4]) for s in statements if s[0] <= finding.line <= s[1]), (None, empty))
        subject = finding.reason.rpartition(": ")[2]
        names = {subject} - _C_KEYWORDS if re.fullmatch(r"[A-Za-z_]\w*", subject) else set()
        own = {holder} & set(functions) if holder else empty
        names |= declares if holder is None else {holder} - own
        grown = True
        while grown:  # a declaration that names a withheld name declares withheld names
            grown = False
            for words, declared in declarations:
                if words & names and not declared <= names:
                    names |= declared
                    grown = True
        hit = own | {name for name, words in bodies if name and words & names}
        for name in hit or functions:
            found[name].add(finding.key)
    return {name: tuple(sorted(keys)) for name, keys in found.items()}


def _top(code: str) -> Iterator[tuple[int, int, str | None, set[str], set[str]]]:
    """(first line, last line, function, words, declared names) of each file-scope statement of masked code: a
    function definition names its function, any other statement the names it declares."""
    line = _numbered(code)
    start = depth = opened = 0
    for token in re.finditer(r"[{};]", code):
        if token[0] == "{":
            opened, depth = token.start() if depth == 0 else opened, depth + 1
            continue
        if token[0] == "}":
            depth = max(depth - 1, 0)
            head = re.search(r"(\w+)\s*\((?:[^()]|\([^()]*\))*\)\s*$", code[start:opened])
            if depth or not head or "=" in code[start:opened]:
                continue
            first = start + len(code[start:opened]) - len(code[start:opened].lstrip())
            yield line(first), line(token.start()), head[1], set(re.findall(r"\w+", code[start : token.end()])), set()
        elif depth:
            continue
        else:
            text = code[start : token.end()]
            first = start + len(text) - len(text.lstrip())
            yield line(first), line(token.start()), None, set(re.findall(r"\w+", text)), _declared(text)
        start = token.end()


def _declared(statement: str) -> set[str]:
    """The names one file-scope declaration declares: record tags, typedef names and globals (not prototypes)."""
    flat = statement
    while (inner := re.sub(r"\{[^{}]*\}", " ", flat)) != flat:
        flat = inner
    names = set(re.findall(r"\b(?:struct|union|enum)\s+(\w+)\s*\{", statement))
    for piece in re.split(r",(?![^()]*\))", flat.rstrip().rstrip(";")):
        piece = piece.split("=", 1)[0]
        pointer = re.search(r"\(\s*\*+\s*(\w+)", piece)
        if not pointer and re.search(r"\w\s*\(", piece):
            continue  # a prototype declares a function its definition names
        name = pointer or re.search(r"(\w+)\s*(?:\[[^\]]*\]\s*)*$", piece)
        if name and name[1] not in _C_KEYWORDS:
            names.add(name[1])
    return names


def _unclean_job(
    item: tuple[Snapshot, str, Sequence[str], Any],
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], dict[str, str]]:
    """A verdict and every file the unit's closure walk read or probed, pinned."""
    snapshot, path = item[0], item[1]
    module = _module(snapshot, path)
    assert module is not None
    unit = _unit(snapshot, module)
    return _unclean_unit(item), _view.asked(snapshot, unit, _view.search(snapshot, _census_version(snapshot, module)))


def _unclean_rows(snapshot: Snapshot, path: str) -> tuple[Any, ...]:
    """The tree row a verdict reads: the module and the names it owns (a symbol's address is judged from them)."""
    return (_module(snapshot, path),)


def _unclean_context(snapshot: Snapshot) -> str:
    """What every verdict reads besides its module: each version's symbol table and segments."""
    return effort.memo(
        ("unclean-context", id(snapshot.config), id(snapshot.versions)),
        lambda: digest([(v, sorted(x.symbols.items()), x.segments) for v, x in sorted(snapshot.versions.items())]),
    )


def _unclean_key(item: tuple[Snapshot, str, Sequence[str], Any]) -> str:
    """One verdict is a fact of the text, every file it reaches, its tree row, the symbol tables, the rules and the job
    that judges it (its result shape moves the key): a header edit moves the verdict of each unit that reads it."""
    snapshot, path, ids, rows = item
    module = _module(snapshot, path)
    assert module is not None
    reach = _view.closure(snapshot, _unit(snapshot, module), _census_version(snapshot, module))
    return digest(
        (
            snapshot.read(path),
            reach,
            _unclean_rows(snapshot, path),
            _unclean_context(snapshot),
            ids,
            rows,
            credited(snapshot, path),
            code_identity(_unclean_unit, credited, *_DEBT.values()),
        )
    )


def counts(findings: Sequence[Finding]) -> dict[str, int]:
    return dict(sorted(Counter(f.key for f in findings).items()))


def _removed_functions(paths: Sequence[str], root: Path, against: str, git: str, tmp: Path) -> int:
    """How many function definitions the changed sources hold in `against` that the working tree no longer defines
    (a moved definition is not removed)."""
    from unbake import publish

    def defined(text: str) -> set[str]:
        return {i.name for i in publish.items(text) if i.kind == "function"}

    old: set[str] = set()
    new: set[str] = set()
    for path in paths:
        shown = process.run("git show", [git, "show", f"{against}:{path}"], root, tmp=tmp)
        if shown.exit == 0:
            old |= defined(shown.stdout.decode(errors="replace"))
        if (root / path).is_file():
            new |= defined((root / path).read_text(errors="replace"))
    return len(old - new)


def _credit_drops(root: Path, against: str, git: str, tmp: Path) -> dict[str, int]:
    """Per version, how many matched functions progress.json holds fewer of than in `against`; empty when either side
    has no progress.json."""
    shown = process.run("git show", [git, "show", f"{against}:progress.json"], root, tmp=tmp)
    if shown.exit != 0 or not (root / "progress.json").is_file():
        return {}

    def matched(text: str | bytes) -> dict[str, int]:
        return {c["version"]: c["functions"]["matched"] for c in json.loads(text).get("carts", ())}

    before, after = matched(shown.stdout), matched((root / "progress.json").read_bytes())
    return {v: n - after[v] for v, n in before.items() if v in after and after[v] < n}


def audit_command(params: Json) -> Json:
    """Refuse the source rule rows the working tree adds over a revision. Reads text and git only: no ROM, host or
    snapshot, so CI runs it with nothing but the checkout."""
    root, against = Path(params["project"] or "."), params["against"]
    git = shutil.which("git")
    if git is None:
        raise Refusal(Finding("native.missing_tool", "git is not installed", path="git"))
    tmp = process.scratch(root)

    def run(*args: str) -> bytes:
        result = process.run("git " + args[0], [git, *args], root, tmp=tmp)
        if result.exit != 0:
            raise Refusal(Finding("native.exit", f"git {' '.join(args)} exit {result.exit}"))
        return result.stdout

    document = tomllib.loads((root / "config.toml").read_text()) if (root / "config.toml").is_file() else {}
    library = tuple(document.get("libraries", ()))
    names = (
        run(
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            against,
            "--",
            ":(icase)*.c",
            ":(icase)*.cpp",
            ":(icase)*.h",
            ":(icase)*.s",
        )
        .decode()
        .split("\0")
    )
    paths = sorted(p for p in names if p and not any(Path(p).is_relative_to(d) for d in library))

    def old_file(path: str) -> bytes | None:
        result = process.run("git show", [git, "show", f"{against}:{path}"], root, tmp=tmp)
        return result.stdout if result.exit == 0 else None

    def new_file(path: str) -> bytes | None:
        return (root / path).read_bytes() if (root / path).is_file() else None

    before: list[Finding] = []
    after: list[Finding] = []
    for path in paths:
        for found, data in ((before, old_file(path)), (after, new_file(path))):
            if data is not None:
                found.extend(text_findings(path, data.decode(errors="replace"), False))
    blocking, _ = scope(before, after, paths, _renames(_placements(old_file), _placements(new_file)))
    if blocking:
        held, now = Counter((f.key, f.path) for f in before), Counter((f.key, f.path) for f in after)
        raise Refusal(
            *(
                replace(f, reason=f"{f.reason} ({f.path}: {f.key.removeprefix('source.')} {held[at]} -> {now[at]})")
                for f in blocking
                for at in [(f.key, f.path)]
            )
        )
    removed = _removed_functions(paths, root, against, git, tmp)
    dropped = _credit_drops(root, against, git, tmp)
    unexplained = {v: n for v, n in dropped.items() if n > removed}
    if unexplained:
        lost = ", ".join(f"{v}: -{n}" for v, n in sorted(unexplained.items()))
        raise Refusal(
            Finding(
                "audit.credit_drop",
                f"progress.json lost matched functions ({lost}) while the changed sources removed {removed}",
                path="progress.json",
                versions=tuple(sorted(unexplained)),
                action="restore the C that earned the credit, or run make check and unbake sync",
            )
        )
    return {"against": against, "paths": paths, "counts_before": counts(before), "counts_after": counts(after)}
