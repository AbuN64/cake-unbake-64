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

from unbake import config, delta, effort, native, pool, process, recipes, store, symbols, types, versions
from unbake import view as _view
from unbake.contracts import Finding, Json, Refusal, Snapshot, SourceView, UnitSpec, code_identity, digest
from unbake.families import typedebt


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
        if header != path and header.endswith(".h"):
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


def _symbol_guards(snapshot: Snapshot, text: str) -> set[int]:
    """Only symbol-selection guards with identical arms and distinct ROM functions are exempt; others are findings."""
    pattern = (
        r"^[ \t]*#[ \t]*if[ \t]+defined\s*\(\s*(VERSION_\w+)\s*\)[ \t]*\n"
        r"([^#]*?)^[ \t]*#[ \t]*else[ \t]*\n([^#]*?)^[ \t]*#[ \t]*endif\b"
    )
    allowed, number = set(), _numbered(text)
    clean = re.sub(
        r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\'',
        lambda m: _blank(m[0]) if m[0].startswith(("/*", "//")) else m[0],
        text,
    )
    for match in re.finditer(pattern, clean, re.M):
        tokens = [
            re.findall(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\S', arm) for arm in match.group(2, 3)
        ]
        if len(tokens[0]) != len(tokens[1]):
            continue
        changed = {(a, b) for a, b in zip(*tokens, strict=True) if a != b}
        if len(changed) != 1:
            continue
        a, b = next(iter(changed))
        members = snapshot.layout.members
        if a not in members or b not in members or any(members[n].kind != "function" for n in (a, b)):
            continue
        code: dict[str, Any] = {}
        arms = set()
        for vid, flags in snapshot.config.project.version_macros.items():
            name = a if f"-D{match[1]}" in flags else b
            places = [
                p
                for p in members[name].placements
                if p.version == vid and p.section == ".text" and snapshot.versions[vid].symbols.get(name) == p.vram
            ]
            if len(places) != 1:
                break
            p = places[0]
            code.setdefault(name, versions.rom_bytes(snapshot.versions[vid], p.rom_start, p.rom_end))
            arms.add(name)
        else:
            if arms == {a, b} and code[a] != code[b]:
                allowed.add(number(match.start()))
    return allowed


_GUARD = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif)\b([^\n]*)$")
_VERSION_ONLY = re.compile(r"(?:\s|defined|VERSION_\w+|[()!]|&&|\|\|)*")


def _truth(expr: str, macros: Collection[str]) -> bool:
    """The value of a version-only `#if` expression (`defined(X)`, `X`, `!`, `&&`, `||`, parentheses) when exactly the
    `-D` macros given are set. A malformed expression raises ValueError."""
    spelled = {"&&": "and", "||": "or", "!": "not", "(": "(", ")": ")"}
    tokens = [t for t in re.findall(r"&&|\|\||[()!]|\w+", expr) if t != "defined"]
    words = [spelled.get(t) or str(f"-D{t}" in macros) for t in tokens]
    try:  # every word is and, or, not, a parenthesis, True or False: Python's precedence is C's
        return bool(eval(" ".join(words), {"__builtins__": {}}))
    except SyntaxError as error:
        raise ValueError(f"malformed expression (unbalanced parenthesis or missing tokens): {expr}") from error


def _guard_expression(found: re.Match[str]) -> str:
    rest = found[2].strip()
    return {"ifdef": f"defined({rest})", "ifndef": f"!defined({rest})"}.get(found[1], rest)


def _data_guards(snapshot: Snapshot, path: str, text: str) -> set[int]:
    """The version guards of a data unit whose bytes truly differ between its holders (proofs must still be exact)."""
    unit = snapshot.layout.units.get(path)
    if unit is None or "VERSION_" not in text:
        return set()
    if config.kind(unit.kind)["credit"] == "code":
        return set()
    differs = any(
        len(
            {
                versions.rom_bytes(snapshot.versions[p.version], p.rom_start, p.rom_end)
                for p in snapshot.layout.members[name].placements
                if p.section != ".bss"
            }
        )
        > 1
        for name in unit.members
    )
    if not differs:
        return set()
    return {
        n
        for n, line in enumerate(text.splitlines(), 1)
        if (m := _GUARD.match(line)) and "VERSION_" in line and _VERSION_ONLY.fullmatch(m[2])
    }


_ELSE = re.compile(r"^[ \t]*#[ \t]*else\b")
_END = re.compile(r"^[ \t]*#[ \t]*endif\b")


def _branches(
    text: str, holders: Collection[str], macros: Json
) -> list[tuple[dict[int, frozenset[str]], list[frozenset[str]]]]:
    """Every version-only `#if` chain of text as ({guard line: holders taking that branch}, [every branch's holders]).
    A branch is taken by the holders whose expression is true and no earlier one was; the `#else` takes the rest. A
    chain with a condition that is not version-only has no truth table and is left out."""
    done: list[tuple[dict[int, frozenset[str]], list[frozenset[str]]]] = []
    stack: list[dict[str, Any]] = []
    for n, line in enumerate(text.splitlines(), 1):
        found = _GUARD.match(line)
        if found and found[1] != "elif":
            stack.append({"lines": {}, "sets": [], "left": frozenset(holders), "ok": True})
        elif _ELSE.match(line) and stack:
            stack[-1]["sets"].append(stack[-1]["left"])
            stack[-1]["left"] = frozenset()
            continue
        elif _END.match(line) and stack:
            chain = stack.pop()
            if chain["ok"]:
                done.append((chain["lines"], [s for s in (*chain["sets"], chain["left"]) if s]))
            continue
        if not found or not stack:
            continue
        chain = stack[-1]
        try:
            if not _VERSION_ONLY.fullmatch(found[2]):
                raise ValueError(found[2])
            raw = frozenset(v for v in holders if _truth(_guard_expression(found), macros[v]))
        except ValueError:
            chain["ok"] = False
            continue
        taken = raw & chain["left"]
        chain["lines"][n] = taken
        chain["sets"].append(taken)
        chain["left"] -= taken
    return done


def _judge(
    snapshot: Snapshot, path: str, text: str, function: str, sides: Sequence[Collection[str]]
) -> tuple[set[int], str]:
    """The guard lines of text a documented N-way difference of function stands for, and why not ("" when it does)."""
    from unbake import layout, ownership

    unit = snapshot.layout.units.get(path)
    if unit is None or function not in unit.members:
        return set(), "Version difference names a function this unit does not hold"
    member = snapshot.layout.members[function]
    holders = set(member.holders())
    parts = [frozenset(s) for s in sides]
    if len(parts) < 2 or not all(parts) or sum(map(len, parts)) != len(holders) or frozenset().union(*parts) != holders:
        return set(), "Version difference must partition every holding version into disjoint sides"
    macros = snapshot.config.project.version_macros
    lines: set[int] = set()
    for found, _ in _branches(text, holders, macros):
        for n, taken in found.items():
            if taken and all(part <= taken or not part & taken for part in parts):
                lines.add(n)
    if not lines:
        return set(), "Version difference must name a guard whose branches are whole sides"
    targets = {
        v: b"".join(
            versions.rom_bytes(snapshot.versions[v], p.rom_start, p.rom_end)
            for p in member.placements
            if p.version == v and p.section == ".text"
        )
        for v in holders
    }
    sets = [{targets[v] for v in part} for part in parts]
    if not all(targets.values()) or any(a & b for i, a in enumerate(sets) for b in sets[i + 1 :]):
        return set(), "Target function bytes do not differ between the documented version sides"
    trial = layout.overlay(snapshot, {path: text.encode()})
    recipe = recipes.resolve(trial.config, unit, unit.options)
    row = {"function": function, "sides": sorted(sorted(part) for part in parts)}
    key = ("version-difference", digest(row), tuple(native.stamp(trial, unit, v, recipe) for v in sorted(holders)))

    def prove() -> bool:
        own, candidate = ownership.derive(trial, replace(unit, withheld=()))
        proofs = [p for v in sorted(holders) for p in native.prove_job((own, candidate, v))]
        return bool(proofs) and {p.version for p in proofs} == holders and all(p.exact for p in proofs)

    if not effort.memo(key, prove):
        return set(), "Documented version difference is not exact in every holding version"
    return lines, ""


def _version_guards(snapshot: Snapshot, path: str, text: str, failures: list[Finding] | None = None) -> set[int]:
    """A documented code guard needs different cart bytes per side and an exact compile in every holder. A difference
    that does not stand is a finding of the text like any rule's (in `failures`), so a change is judged against what
    its base text already had; without `failures` it refuses."""
    unit = snapshot.layout.units.get(path)
    allowed: set[int] = set()
    for row in getattr(snapshot.config.project, "version_differences", ()):
        if unit and row["function"] in unit.members:
            lines, reason = _judge(snapshot, path, text, row["function"], row["sides"])
            if reason:
                failed = Finding("source.file-version-guard", reason, path=path, unit=row["function"])
                if failures is None:
                    raise Refusal(failed)
                failures.append(failed)
            allowed |= lines
    return allowed


def _derived_rows(snapshot: Snapshot) -> list[Json]:
    """The version_differences rows the sources earn: for each code function, every `#if` chain of its unit that splits
    its holders yields a row, kept only when `_judge` accepts it (different bytes per side, exact in every holder)."""
    rows: dict[str, Json] = {}
    kinds = config.kinds()
    for path, unit in snapshot.layout.units.items():
        if kinds[unit.kind]["credit"] != "code" or (text := snapshot.peek(path)) is None or b"VERSION_" not in text:
            continue
        source = text.decode()
        for name in unit.members:
            member = snapshot.layout.members[name]
            holders = member.holders()
            if member.kind != "function" or len(holders) < 2:
                continue
            for _, branches in _branches(source, holders, snapshot.config.project.version_macros):
                sides = sorted(sorted(b) for b in dict.fromkeys(branches) if b)
                if len(sides) > 1 and not _judge(snapshot, path, source, name, sides)[1]:
                    rows[digest((name, sides))] = {"function": name, "sides": sides}
    return sorted(rows.values(), key=lambda r: (r["function"], r["sides"]))


_ROW = "[[project.version_differences]]"


def _rows_text(text: str, rows: Sequence[Json]) -> str:
    """config.toml text with its version_differences rows replaced by rows, in the place the first old row held."""
    kept: list[str] = []
    first: int | None = None
    skipping = False
    for line in text.splitlines():
        if line.strip() == _ROW:
            first = len(kept) if first is None else first
            skipping = True
        elif line.startswith("["):
            skipping = False
        if not skipping:
            kept.append(line)
    if first is None:  # before the first table that follows [project], else at the end
        project = next((i for i, line in enumerate(kept) if line.strip() == "[project]"), None)
        later = range(project + 1, len(kept)) if project is not None else ()
        first = next((i for i in later if kept[i].startswith("[") and not kept[i].startswith("[project.")), len(kept))
    block = []
    for row in rows:
        sides = ", ".join("[" + ", ".join(json.dumps(v) for v in side) + "]" for side in row["sides"])
        block += [_ROW, f"function = {json.dumps(row['function'])}", f"sides = [{sides}]", ""]
    while first and not kept[first - 1].strip() and not block:
        first -= 1
    out = [*kept[:first], *block, *kept[first:]]
    return "\n".join(out).rstrip("\n") + "\n"


_C_WORDS = ["if", "else", "while", "for", "do", "switch", "case", "return", "sizeof", "defined", "typeof"]
_NOT_NAMES = frozenset([*_C_WORDS, "goto", "break", "continue", "default", "struct", "union", "enum"])
_IDENT = re.compile(r"[A-Za-z_]\w*")
_DECLARED = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)|\b([A-Za-z_]\w*)\s*(?=[(\[;,])", re.M)


def _chain_names(text: str, start: int) -> set[str]:
    """The names the version chain opening at line `start` (1-based) defines or declares: its macros and declarators."""
    depth, body = 0, []
    for line in text.splitlines()[start - 1 :]:
        opens = _GUARD.match(line)
        depth += bool(opens and opens[1] != "elif") - bool(_END.match(line))
        body.append(line)
        if depth == 0:
            break
    return {m[1] or m[2] for m in _DECLARED.finditer("\n".join(body))} - _NOT_NAMES


def _header_guards(snapshot: Snapshot, path: str, text: str) -> set[int]:
    """The guard lines of a header whose version-only `#if` chains really split the ROMs: the branches partition the
    versions into sides, and a code function of a unit that mentions a name the chain defines or declares holds bytes
    that differ between every pair of sides. The compile and ROM proof of those units still has to be exact."""
    if not path.endswith(".h") or "VERSION_" not in text or path in snapshot.layout.units:
        return set()
    return effort.memo(
        ("header-guards", snapshot.digest, path, digest(text)), lambda: _header_guard_lines(snapshot, path, text)
    )


def _header_guard_lines(snapshot: Snapshot, path: str, text: str) -> set[int]:
    from unbake import headers as _headers

    macros = snapshot.config.project.version_macros
    chains = _branches(text, set(snapshot.versions) & set(macros), macros)
    if not chains:
        return set()
    users = [snapshot.layout.units[u] for u in _headers.consumers(snapshot, [path])]
    users = [u for u in users if config.kind(u.kind)["credit"] == "code"]
    words: dict[str, set[str]] = {}

    def mentions(unit: UnitSpec) -> set[str]:  # tokenized when first asked, once per unit
        if unit.path not in words:
            words[unit.path] = set(_IDENT.findall((snapshot.peek(unit.path) or b"").decode(errors="replace")))
        return words[unit.path]

    allowed: set[int] = set()
    for found, sides in chains:
        parts = [frozenset(side) for side in sides]
        lines = {n for n, taken in found.items() if taken and all(p <= taken or not p & taken for p in parts)}
        if len(parts) < 2 or not lines:
            continue
        names = _chain_names(text, min(found))
        for unit in users:
            if names & mentions(unit) and any(
                _sides_differ(snapshot, snapshot.layout.members[n], parts)
                for n in unit.members
                if snapshot.layout.members[n].kind == "function"
            ):
                allowed |= lines
                break
    return allowed


def _sides_differ(snapshot: Snapshot, member: Any, parts: Sequence[frozenset[str]]) -> bool:
    """Whether the member's .text bytes in the versions of each side share no value with any other side's."""
    seen: list[set[bytes]] = []
    for part in parts:
        held = {
            versions.rom_bytes(snapshot.versions[p.version], p.rom_start, p.rom_end)
            for p in member.placements
            if p.version in part and p.section == ".text"
        }
        if not held or any(held & earlier for earlier in seen):
            return False  # a side without the member, or one sharing bytes with an earlier side, ends it at once
        seen.append(held)
    return True


def _allowed_guards(snapshot: Snapshot, path: str, text: str) -> set[int]:
    """The lines of `text` the file-version-guard rule lets stand: symbol selection, or proven data differences."""
    if "VERSION_" not in text:
        return set()
    return (
        _symbol_guards(snapshot, text)
        | _data_guards(snapshot, path, text)
        | _version_guards(snapshot, path, text, [])  # a difference that does not stand is evaluate's finding
        | _header_guards(snapshot, path, text)
    )


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
)


def _vram_index(snapshot: Snapshot, version: str) -> tuple[list[int], list[int]]:
    """The sorted vram starts and ends of every placed member in a version, once per command."""

    def build() -> tuple[list[int], list[int]]:
        rows = sorted(
            (p.vram, p.vram + p.size)
            for member in snapshot.layout.members.values()
            for p in member.placements
            if p.version == version and p.size > 0
        )
        return [r[0] for r in rows], [r[1] for r in rows]

    return effort.memo(("vram", snapshot.digest, version), build)


def _address_literals(snapshot: Snapshot, path: str, code: str) -> dict[int, str]:
    """Lines that spell a ROM address as a number: a literal in the mapped address space (from _KSEG0) inside a placed
    member's vram range in a version holding the unit, or `NAME + K` that reaches past the data member NAME in every
    holder."""
    unit = snapshot.layout.units.get(path)
    if unit is None:
        return {}
    held = sorted({p.version for name in unit.members for p in snapshot.layout.members[name].placements})
    indexes = [_vram_index(snapshot, v) for v in held]
    line = _numbered(code)
    hits: dict[int, str] = {}
    for match in _NUMBER.finditer(code):
        value = _integer(match["int"]) if match["int"] else None
        if value is None or value < _KSEG0:
            continue
        for starts, ends in indexes:
            at = bisect_right(starts, value) - 1
            if at >= 0 and value < ends[at]:
                hits.setdefault(line(match.start()), match["int"])
                break
    for match in _OFFSET.finditer(code):
        member = snapshot.layout.members.get(match[1])
        if member is None or member.kind == "function" or not member.placements:
            continue
        if (reach := _integer(match[2])) is not None and all(reach >= p.size for p in member.placements):
            hits.setdefault(line(match.start()), " ".join(match[0].split()))
    return hits


def _hits(
    rule: Json,
    path: str,
    scopes: Mapping[str, str],
    rules: Json,
    sdk_group: bool,
    snapshot: Snapshot | None,
    view: SourceView | None,
    documented: set[int],
) -> dict[int, str]:
    """{line: subject} of one rule, the subject being the text that identifies the finding across edits around it."""
    code = scopes["code"]
    if "regex" in rule:
        if rule.get("files") == "source" and path.endswith(".h"):
            return {}
        scoped = scopes[rule["scope"]]
        number = _numbered(scoped)
        found: dict[int, str] = {}
        for m in re.finditer(rule["regex"], scoped, re.M):
            found.setdefault(number(m.start() + len(m[0]) - len(m[0].lstrip())), m[0].strip())
        if rule["id"] == "union-view" and found:
            sdk = _sdk_union_lines(scoped)
            found = {n: t for n, t in found.items() if n not in sdk}
        if rule["id"] == "file-version-guard" and found and snapshot:
            allowed = _allowed_guards(snapshot, path, scopes["text"]) | documented
            found = {n: t for n, t in found.items() if n not in allowed}
        return found
    check = rule["check"]
    if check == "placeholder-record" and path.endswith(".h"):
        return {}
    if check in ("fake-type", "pad-view", "placeholder-record", "param-cast", "address-int"):
        found = getattr(typedebt, check.replace("-", "_"))(code)
        return dict(found)
    if check == "duplicate-record":
        from unbake import recovery  # recovery reads policy, so the import waits for the first use

        return typedebt.duplicate_records(code, recovery.clones(snapshot)) if snapshot else {}
    if check == "empty-loop":
        lines = _empty_loops(code)
    elif check == "volatile":
        active = code if sdk_group or not view else _scopes(_active(view, path))["code"]
        lines = set() if sdk_group else _volatile(active, rules["hardware"]["address_ranges"])
    elif check == "shared-declarations":
        lines = _shared(code, view, path, snapshot) if view else set()
    elif check == "address-literal":
        return _address_literals(snapshot, path, code) if snapshot else {}
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
    documented: set[int],
) -> tuple[Finding, ...]:
    rules = config.load_resource("rules.toml")
    scopes, lines, findings = {**_scopes(text), "text": text}, text.splitlines(), []
    for rule in rules["rule"]:
        if (only is not None and rule["id"] not in only) or (sdk_group and rule["id"] == "union-view"):
            continue
        if snapshot is None and rule.get("needs"):
            continue
        hits = _hits(rule, path, scopes, rules, sdk_group, snapshot, view, documented)
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
    return _findings(None, path, text, None, sdk_group, None, set())


def evaluate(
    snapshot: Snapshot,
    path: str,
    text: str,
    view: SourceView | None,
    sdk_group: bool,
    only: Collection[str] | None = None,
) -> tuple[Finding, ...]:
    with effort.stage("policy.evaluate"):
        failures: list[Finding] = []
        documented = _version_guards(snapshot, path, text, failures)
        if any(Path(path).is_relative_to(folder) for folder in getattr(snapshot.config.project, "library_dirs", ())):
            return ()  # vendored source idioms; compile and ROM proof are still mandatory
        if only is not None and "file-version-guard" not in only:
            failures.clear()
        return (*failures, *_findings(snapshot, path, text, view, sdk_group, only, documented))


def scope(
    before: Sequence[Finding], after: Sequence[Finding], writes: Collection[str]
) -> tuple[tuple[Finding, ...], tuple[Finding, ...]]:
    # A row is its rule and subject, wherever it is: moving text between files is free and a repeated row is new once
    # it outnumbers the old text's, so the repo-wide total of a rule never rises through any path.
    prior = Counter((f.key, f.reason) for f in before)
    blocking, debt = [], []
    for finding in after:
        row = (finding.key, finding.reason)
        prior[row] -= 1
        if finding.path in writes and prior[row] < 0:
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
    before = evaluate(snapshot, path, old.decode(), None, sdk_group) if old is not None else ()
    return scope(before, findings, {path})[0]


def _census_version(snapshot: Snapshot, unit: UnitSpec) -> str:
    holders = sorted({p.version for name in unit.members for p in snapshot.layout.members[name].placements})
    return snapshot.config.project.names_from if snapshot.config.project.names_from in holders else holders[0]


def _census_unit(item: tuple[Snapshot, UnitSpec]) -> tuple[Finding, ...]:
    snapshot, unit = item
    source = _view.get(snapshot, unit, _census_version(snapshot, unit), recipes.resolve(snapshot.config, unit, {}))
    sdk = snapshot.layout.groups[unit.group].sdk
    text = snapshot.read(unit.path).decode()
    # evaluate reads the unit text and the active lines of the unit and its headers: other edits keep the verdict warm
    files = [unit.path, *(p for p, _ in source.dependencies if p.endswith(".h"))]
    guards = _allowed_guards(snapshot, unit.path, text)
    key = digest(
        (
            text,
            sdk,
            config.load_resource("rules.toml"),
            snapshot.config.project.digest,
            code_identity(evaluate, _view_uses, typedebt),
            guards,
            [_active(source, p) for p in files],
        )
    )
    return store.cached_object(
        snapshot.config,
        "census",
        key,
        lambda: (*evaluate(snapshot, unit.path, text, source, sdk), *_view_uses(snapshot, unit, text, files[1:])),
    )


def _view_uses(snapshot: Snapshot, unit: UnitSpec, text: str, reached: Sequence[str]) -> tuple[Finding, ...]:
    """MEASURE ONLY (no rule, never unclean): one finding per union or pad-view record that a header defines and the
    unit names."""
    views = set().union(*(typedebt.view_types(_scopes(snapshot.read(p).decode())["code"]) for p in reached))
    found = typedebt.view_uses(_scopes(text)["code"], views)
    return tuple(
        Finding(
            "source.uses-view-type",
            f"{config.sentence('source.uses-view-type')}: {name}",
            path=unit.path,
            line=at,
            unit=unit.path,
        )
        for at, name in sorted(found.items())
    )


def _census_warnings(item: tuple[Snapshot, UnitSpec]) -> tuple[Finding, ...]:
    """The unit's landed view conversions as debt by count, read from what the build kept and never compiled here."""
    snapshot, unit = item
    found = native.recorded(snapshot, unit, _census_version(snapshot, unit), recipes.resolve(snapshot.config, unit, {}))
    if not found:
        return ()
    return (
        Finding(
            "land.view-conversion",
            f"{len(found)} integer-pointer conversions the compiler reports",
            path=unit.path,
            missing=found,
            action="type each variable as the pointer it holds",
        ),
    )


def _census_key(item: tuple[Snapshot, UnitSpec]) -> str:
    """Everything a build of the unit reads: warm while none of it changed, so the pool dispatches nothing."""
    snapshot, unit = item
    reads = _view.closure(snapshot, unit, _census_version(snapshot, unit))
    text = snapshot.read(unit.path).decode()
    guards = _allowed_guards(snapshot, unit.path, text)
    from unbake import recovery  # recovery reads policy, so the import waits for the first use

    return digest(
        (
            reads,
            recovery.clones(snapshot),
            snapshot.layout.groups[unit.group].sdk,
            config.load_resource("rules.toml"),
            snapshot.config.project.digest,
            code_identity(_census_unit, typedebt),
            guards,
        )
    )


def _census_header(item: tuple[Snapshot, str, bool]) -> tuple[Finding, ...]:
    snapshot, path, sdk = item  # a header is judged by its text alone, so one verdict serves every unit
    return evaluate(snapshot, path, snapshot.read(path).decode(), None, sdk)


def _header_key(item: tuple[Snapshot, str, bool]) -> str:
    snapshot, path, sdk = item
    text = snapshot.read(path).decode()
    guards = _allowed_guards(snapshot, path, text)
    return digest(
        (
            text,
            path,
            sdk,
            config.load_resource("rules.toml"),
            snapshot.config.project.digest,
            code_identity(_census_header, typedebt),
            guards,
        )
    )


def _changed_findings(snapshot: Snapshot, texts: Mapping[str, str], *, baseline: bool = False) -> list[Finding]:
    """The findings of these texts in this snapshot. The baseline is the committed layout, which may list a unit whose
    group it no longer defines: its old text has no group to classify it by and counts for nothing, so the new text is
    judged strictly. The staged layout itself must be whole."""
    headers: dict[str, Any] = {}
    if any(p.endswith(".h") for p in texts):
        kinds = config.kinds()
        for unit in snapshot.layout.units.values():
            if (
                kinds[unit.kind]["credit"] != "code"
                or "compile" not in kinds[unit.kind]["phases"]
                or not snapshot.layout.held(unit.members)
            ):
                continue
            for path, _ in _view.closure(snapshot, unit, _census_version(snapshot, unit)):
                if path in texts:
                    headers[path] = headers.get(path, True) and snapshot.layout.groups[unit.group].sdk
    findings: list[Any] = []
    for path, text in texts.items():
        owner = snapshot.layout.units.get(path)
        if not path.endswith(".h") and (owner is None or not snapshot.layout.held(owner.members)):
            continue
        if baseline and owner and owner.group not in snapshot.layout.groups:
            continue
        sdk = snapshot.layout.groups[owner.group].sdk if owner else headers.get(path, False)
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
            before = _changed_findings(baseline or snapshot, {p: old for p, (old, _) in changes.items()}, baseline=True)
            after = _changed_findings(snapshot, {p: new for p, (_, new) in changes.items()})
            blocking, debt = scope(before, after, changes)
            return (*blocking, *debt), []
        from unbake import libasm

        kinds = config.kinds()
        wanted = None if only is None else frozenset(only)
        outside = [
            Finding("source.asm-outside-library", config.sentence("source.asm-outside-library"), path=p, unit=p)
            for p in libasm.outside(snapshot)
            if wanted is None or p in wanted
        ] + list(libasm.plan(snapshot)[1] if wanted is None else ())  # its rows name members, never a scoped unit
        items = [
            (snapshot, unit)
            for unit in snapshot.layout.units.values()
            if (wanted is None or unit.path in wanted)
            and kinds[unit.kind]["credit"] == "code"
            and "compile" in kinds[unit.kind]["phases"]
        ]
        unique = {}
        headers: dict[str, Any] = {}
        for _, unit in items:  # a header is judged once, as SDK only when every unit that reads it is
            for path, _ in _view.closure(snapshot, unit, _census_version(snapshot, unit)):
                if path.endswith(".h"):
                    headers[path] = headers.get(path, True) and snapshot.layout.groups[unit.group].sdk
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
        conflicts = (
            types.conflicts(snapshot) if wanted is None else types.conflicts(snapshot, [*sorted(wanted), *headers])
        )
        for findings in (*found[0], *found[1], *found[2], conflicts, outside):
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
    ids, kinds = rules.get("unclean", ()), config.kinds()
    if not ids:
        return {}

    def judge() -> dict[str, tuple[str, ...]]:
        paths = [
            u.path
            for u in snapshot.layout.units.values()
            if kinds[u.kind]["decompiled"] and (only is None or u.path in only)
        ]
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
                pin=lambda path: _view._pin(snapshot, path),
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
    """The functions of a unit that earn credit of their own: its function members and every other function its text
    defines that has an address of its own (a helper compiled inside a member's bytes). A defined function with no
    address was inlined into the functions that name it, so its findings withhold them instead."""
    unit, members = snapshot.layout.units[path], snapshot.layout.members
    own = [n for n in unit.members if members[n].kind == "function"]
    text = snapshot.read(path).decode() if text is None else text
    placed = snapshot.versions[_census_version(snapshot, unit)].symbols if own else {}
    inner = [h for _, _, h, _, _ in _top(_scopes(text)["code"]) if h and h not in own and h in placed]
    return [*own, *dict.fromkeys(inner)]


def withheld_bytes(snapshot: Snapshot, rules: Collection[str], member: str, version: str) -> int:
    """The .text bytes of a function member in one version that unclean verdicts withhold. The member's bytes split at
    the address of each function inside them; a piece goes with the function that starts it (the first with the
    member), so a member keeps the credit of its clean functions when a helper inside it is withheld."""
    if not rules:
        return 0
    places = [p for p in snapshot.layout.members[member].placements if p.version == version and p.section == ".text"]
    starts, total = _function_starts(snapshot, version), 0
    for place in places:
        cuts = [(place.vram, member), *_starts_in(starts, place)]
        for (start, name), (end, _) in zip(cuts, [*cuts[1:], (place.vram + place.size, "")], strict=True):
            total += end - start if name in rules else 0
    return total


def _starts_in(starts: Sequence[tuple[int, str]], place: Any) -> list[tuple[int, str]]:
    """The (address, name) starts strictly inside one placement: its own start belongs to the member."""
    low, high = place.vram, place.vram + place.size
    return list(starts[bisect_right(starts, (low, "\uffff")) : bisect_left(starts, (high, ""))])


def _function_starts(snapshot: Snapshot, version: str) -> list[tuple[int, str]]:
    """(address, name) of every function of the symbol table in one version, sorted: where a function's bytes start. A
    project with no symbol table knows no function inside a member."""

    def starts() -> list[tuple[int, str]]:
        if snapshot.peek(symbols.path()) is None:
            return []
        table = symbols.load(snapshot.read, snapshot.config.project.versions, store.content(snapshot.config).cached)
        located = snapshot.versions[version].symbols
        return sorted((located[n], n) for n, row in table.items() if row.get("kind") == "function" and n in located)

    return effort.memo(("function-starts", snapshot.digest, version), starts)


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
    unit = snapshot.layout.units[path]
    return _unclean_unit(item), _view.asked(snapshot, unit, _view.search(snapshot, _census_version(snapshot, unit)))


def _unclean_rows(snapshot: Snapshot, path: str) -> tuple[Any, ...]:
    """The layout rows a verdict reads: the unit and its members' rows (a version guard is judged on their bytes)."""
    unit = snapshot.layout.units[path]
    return unit, [snapshot.layout.members[n] for n in unit.members]


def _unclean_context(snapshot: Snapshot) -> str:
    """What every verdict reads besides its unit: the record clones a recovery found, the documented version
    differences and the ROMs. The type map is not read: a type map row a landing changes judges nothing again."""
    from unbake import recovery  # recovery reads policy, so the import waits for the first use

    return effort.memo(
        ("unclean-context", id(snapshot.config), id(snapshot.versions)),
        lambda: digest(
            (
                sorted(recovery.clones(snapshot).items()),
                list(getattr(snapshot.config.project, "version_differences", ())),
                sorted((v, x.rom_sha256) for v, x in snapshot.versions.items()),
            )
        ),
    )


def _unclean_key(item: tuple[Snapshot, str, Sequence[str], Any]) -> str:
    """One verdict is a fact of the text, every file it reaches, its layout rows, the clones and version differences,
    the rules and the job that judges it (its result shape moves the key): a header edit moves the verdict of each unit
    that reads the header, a type map edit none."""
    snapshot, path, ids, rows = item
    unit = snapshot.layout.units[path]
    reach = _view.closure(snapshot, unit, _census_version(snapshot, unit))
    return digest(
        (
            snapshot.read(path),
            reach,
            _unclean_rows(snapshot, path),
            _unclean_context(snapshot),
            ids,
            rows,
            credited(snapshot, path),
            code_identity(_unclean_unit, credited, typedebt),
        )
    )


def counts(findings: Sequence[Finding]) -> dict[str, int]:
    return dict(sorted(Counter(f.key for f in findings).items()))


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

    layout_doc = tomllib.loads((root / "layout.toml").read_text()) if (root / "layout.toml").is_file() else {}
    project = (
        tomllib.loads((root / "config.toml").read_text()).get("project", {}) if (root / "config.toml").is_file() else {}
    )
    sdk_of = {g["name"]: g.get("sdk", False) for g in layout_doc.get("group", ())}
    sdk = {u["path"]: sdk_of.get(u["group"], False) for u in layout_doc.get("unit", ())}
    library = tuple(project.get("library_dirs", ()))
    names = run("diff", "--name-only", "-z", "--no-renames", against, "--", "*.c", "*.h").decode().split("\0")
    paths = sorted(p for p in names if p and not any(Path(p).is_relative_to(d) for d in library))
    before: list[Finding] = []
    after: list[Finding] = []
    for path in paths:
        old = process.run("git show", [git, "show", f"{against}:{path}"], root, tmp=tmp)
        new = root / path
        group = sdk.get(path, False)
        if old.exit == 0:
            before.extend(text_findings(path, old.stdout.decode(errors="replace"), group))
        if new.is_file():
            after.extend(text_findings(path, new.read_text(errors="replace"), group))
    blocking, _ = scope(before, after, paths)
    if blocking:
        raise Refusal(*blocking)
    return {"against": against, "paths": paths, "counts_before": counts(before), "counts_after": counts(after)}


def version_config(snapshot: Snapshot) -> bytes:
    """config.toml with its version_differences rows replaced by the rows the sources earn."""
    return _rows_text(snapshot.read("config.toml").decode(), _derived_rows(snapshot)).encode()
