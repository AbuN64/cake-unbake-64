"""Units into one source file: the pure text rule behind `unbake layout merge`.

`merge` takes the units of one module in link (ROM) order and returns the text of the one file that replaces them, plus
what it could not decide. It
  hoists and dedupes the `#include` lines that sit outside any `#if`,
  drops the all-zero pad arrays (a few bytes) that the linker's section alignment reproduces,
  drops the prototypes and `extern` declarations that the merged file itself now defines or already declares,
  drops a prototype that disagrees with a definition standing earlier in the merged text (the definition wins),
  reports what is left as conflicts: a declaration whose type differs from a definition, from another declaration
  of the same name or from one in an included header, and two `static` definitions of one name. Types are compared
  through the `typedef`s of the units and the included headers, and an empty parameter list matches any list.
The text is read as top-level declarations (comments and strings masked, braces and parentheses counted); it needs no
compiler and knows no project."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

Row = dict[str, Any]
PAD_LIMIT = 8
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.S)
_WORD = re.compile(r"[A-Za-z_]\w*")
_PAD = re.compile(
    r"^(?P<const>const[ \t]+)?(?:unsigned[ \t]+char|u8|s8|char)[ \t]+(?P<name>\w+)\[(?P<n>\d+)\]"
    r"[ \t]*=[ \t]*\{(?P<body>[^}]*)\};[ \t]*\n?",
    re.M,
)
_QUALIFIERS = frozenset({"const", "volatile", "struct", "union", "enum", "unsigned", "signed", "register", "static"})
_BASE_TYPES = frozenset({"int", "char", "short", "long", "float", "double", "void"})
_IF = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef)\b")
_ELSE = re.compile(r"^[ \t]*#[ \t]*(else|elif)\b")
_ENDIF = re.compile(r"^[ \t]*#[ \t]*endif\b")
_INCLUDE = re.compile(r"^[ \t]*#[ \t]*include\b.*$")
_IFNDEF = re.compile(r"^[ \t]*#[ \t]*ifndef[ \t]+(\w+)[ \t]*$")
_QUOTED = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"\n]+)"')
Droppable = Callable[[str, str, int, bool], bool]
Header = Callable[[str], str | None]


@dataclass
class Decl:
    """One top-level statement: its kind, name, normalised type text and the span it covers in its unit's body."""

    kind: str  # proto | def | extern | var | other
    name: str
    sig: str
    unit: str
    start: int
    end: int
    static: bool = False
    guard: tuple[str, ...] = ()
    drop: bool = field(default=False, compare=False)


def mask(text: str) -> str:
    """The text with comments and string/char literals blanked (newlines kept), so offsets still line up."""
    return _COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _norm(text: str) -> str:
    text = re.sub(r"\s+", " ", re.sub(r"\b(?:struct|union|enum)\s+(?=[A-Za-z_])", "", text).strip())
    return re.sub(r" ?([*,()\[\]]) ?", r"\1", text)


def _param(raw: str) -> str:
    """A parameter without its name: `View *actor` -> `View*`, `u32 x` -> `u32`, `unsigned int` stays."""
    tokens = re.findall(
        r"[A-Za-z_]\w*|\*|\[[^\]]*\]|.", re.sub(r"\b(?:struct|union|enum)\s+", "", raw.replace("\n", " "))
    )
    tokens = [t for t in tokens if t.strip()]
    kept = [t for t in tokens if t not in _QUALIFIERS]
    if len(kept) >= 2 and _WORD.fullmatch(tokens[-1]) and tokens[-1] not in _BASE_TYPES | _QUALIFIERS:
        tokens = tokens[:-1]
    return "".join(tokens)


def signature(head: str) -> tuple[str, str]:
    """(name, normalised signature without parameter names) of a function head such as `static u32 f(u32 a, b *c)`."""
    open_at = head.find("(")
    name = _WORD.findall(head[:open_at])[-1] if _WORD.findall(head[:open_at]) else ""
    depth, params, current = 0, [], ""
    close_at = open_at
    for at in range(open_at, len(head)):
        char = head[at]
        depth += char == "("
        depth -= char == ")"
        if depth == 0:
            close_at = at
            break
        if depth == 1 and char == ",":
            params.append(current)
            current = ""
        elif at > open_at:
            current += char
    params.append(current)
    kept = [] if [p.strip() for p in params] in ([""], ["void"]) else [_param(p) for p in params]
    ret = " ".join(w for w in head[:open_at].split() if w not in ("extern", "static", "inline"))
    ret = _norm(ret)
    ret = ret[: ret.rfind(name)].rstrip() if name and name in ret else ret
    return name, f"{ret}({','.join(kept)}){_norm(head[close_at + 1 :])}"


def _data_name(head: str) -> tuple[str, str]:
    """(name, type) of a data declaration head (`extern u32 D_1[4]`, `const u8 t[3]`)."""
    base = head.split("=")[0]
    words = _WORD.findall(base)
    name = words[-1] if words else ""
    suffix = _norm(base[base.rfind(name) + len(name) :]) if name else ""
    typed = _norm(" ".join(w for w in base[: base.rfind(name)].split() if w not in ("extern", "static")))
    return name, _norm(f"{typed} {suffix}") if name else ""


def decls(body: str, unit: str) -> list[Decl]:
    """The top-level declarations of one unit's body, with the `#if` guard each sits in."""
    text = mask(body)
    out: list[Decl] = []
    guard: list[str] = []
    buffer, start, depth = "", -1, 0
    at, size = 0, len(text)
    while at < size:
        line_end = text.find("\n", at)
        line_end = size if line_end < 0 else line_end
        line = text[at:line_end]
        if depth == 0 and not buffer.strip() and line.lstrip().startswith("#"):
            while line.rstrip().endswith("\\") and line_end < size:
                nxt = text.find("\n", line_end + 1)
                line_end = size if nxt < 0 else nxt
                line = text[at:line_end]
            if _IF.match(line):
                guard.append(line.strip())
            elif _ELSE.match(line) and guard:
                guard[-1] = guard[-1] + " | " + line.strip()
            elif _ENDIF.match(line) and guard:
                guard.pop()
            at = line_end + 1
            continue
        for offset, char in enumerate(line):
            if not buffer.strip() and not char.isspace():
                start = at + offset
            buffer += char
            if char in "({[":
                depth += 1
            elif char in ")}]":
                depth -= 1
            end_of = at + offset + 1
            if depth == 0 and char == ";":
                out.append(_classify(buffer, unit, start, end_of, guard, False))
                buffer = ""
            elif depth == 0 and char == "}" and _is_function(buffer):
                out.append(_classify(buffer, unit, start, end_of, guard, True))
                buffer = ""
        buffer += "\n"
        at = line_end + 1
    return out


def _is_function(buffer: str) -> bool:
    head = buffer[: buffer.find("{")] if "{" in buffer else ""
    return bool(head.strip()) and "=" not in head and head.rstrip().endswith((")", "const")) and "(" in head


def _classify(buffer: str, unit: str, start: int, end: int, guard: Sequence[str], body: bool) -> Decl:
    text = buffer.strip()
    static = bool(re.match(r"(?:extern\s+)?static\b|(?:\w+\s+)*static\b", text.split("(")[0].split("=")[0]))
    where = tuple(guard)
    if body:
        name, sig = signature(text[: text.find("{")])
        return Decl("def", name, sig, unit, start, end, static, where)
    head = text.rstrip(";").strip()
    if re.match(r"(typedef|struct|union|enum|class|template|using|namespace)\b", head) or not head:
        return Decl("other", "", "", unit, start, end, static, where)
    if "(" in head.split("=")[0] and not re.search(r"\)\s*\(", head.split("=")[0]) and "=" not in head:
        name, sig = signature(head)
        return Decl("proto", name, sig, unit, start, end, static, where)
    name, sig = _data_name(head)
    kind = "extern" if re.match(r"extern\b", head) else "var"
    return Decl(kind, name, sig, unit, start, end, static, where)


def drop_pads(body: str, unit: str, droppable: Droppable) -> tuple[str, list[str]]:
    """Remove the all-zero pad arrays under PAD_LIMIT bytes that `droppable(unit, name, size, const)` accepts, then the
    `#if` blocks left empty; returns the text and the dropped names."""
    dropped: list[str] = []

    def take(found: re.Match[str]) -> str:
        size = int(found["n"])
        zeros = re.fullmatch(r"[\s,]*(?:0[xX]0+|0)(?:[\s,]+(?:0[xX]0+|0))*[\s,]*", found["body"])
        if size < PAD_LIMIT and zeros and droppable(unit, found["name"], size, bool(found["const"])):
            dropped.append(found["name"])
            return ""
        return found.group(0)

    text = _PAD.sub(take, body)
    if dropped:
        text = re.sub(r"^[ \t]*#[ \t]*if[^\n]*\n(?:[ \t]*\n)*[ \t]*#[ \t]*endif[^\n]*\n?", "", text, flags=re.M)
    return text, dropped


def _cut(body: str, spans: Sequence[tuple[int, int]]) -> str:
    for start, end in sorted(spans, reverse=True):
        end += 1 if body[end : end + 1] == "\n" else 0
        body = body[:start] + body[end:]
    return body


def aliases(text: str) -> dict[str, str]:
    """The simple `typedef`s of `text` as name -> type: `typedef unsigned int u32;`, `typedef struct T {..} U;`
    (U -> T), `typedef char *str;`; function-pointer and array typedefs are skipped."""
    masked = mask(text)
    found: dict[str, str] = {}
    for decl in decls(text, ""):
        statement = masked[decl.start : decl.end].strip().rstrip(";").strip()
        if decl.kind != "other" or not statement.startswith("typedef"):
            continue
        brace = statement.find("{")
        if brace >= 0:
            tag = _WORD.findall(statement[len("typedef") : brace])
            tail = statement[statement.rfind("}") + 1 :]
            names = [n.strip() for n in tail.split(",")]
            if tag and tag[-1] not in _QUALIFIERS:
                found.update({n: tag[-1] for n in names if _WORD.fullmatch(n) and n != tag[-1]})
            continue
        if "(" in statement or "[" in statement:
            continue
        words = _WORD.findall(statement)
        if len(words) >= 3:
            base = _norm(statement[len("typedef") : statement.rfind(words[-1])])
            if base and base != words[-1]:
                found[words[-1]] = base
    return found


_SIGNED = re.compile(r"\bsigned\b\s*(?=(?:int|char|short|long)\b)")
_BARE = re.compile(r"\b(un)?signed\b(?!\s+(?:int|char|short|long)\b)")
_INT = re.compile(r"\b(short|long)\s+int\b")


def _canon(sig: str, names: dict[str, str]) -> str:
    """`sig` with every typedef name replaced by its type and the integer spellings made one: `s32` and `int`, `u32` and
    `unsigned int`, `long int` and `long` read alike."""
    for _ in range(8):
        again = _WORD.sub(lambda m: names.get(m[0], m[0]), sig)
        if again == sig:
            break
        sig = again
    sig = _SIGNED.sub("", sig)
    sig = _BARE.sub(lambda m: "unsigned int" if m[1] else "int", sig)
    return _INT.sub(r"\1", sig)


def _alike(one: str, two: str, names: dict[str, str] | None = None) -> bool:
    """Whether two signatures name one type; an array of unknown size matches any size and an empty parameter list
    (unspecified arguments in C) matches any list of the same return type."""
    one, two = (re.sub(r"\[[^\]]*\]", "[]", _canon(t, names or {})) for t in (one, two))
    if one == two:
        return True
    open_one, open_two = one.find("("), two.find("(")
    if open_one < 0 or open_two < 0 or one[:open_one] != two[:open_two]:
        return False
    return one[open_one:].startswith("()") or two[open_two:].startswith("()")


def _units_of(group: Sequence[Decl]) -> str:
    return ", ".join(sorted({d.unit for d in group}))


def _unguard(text: str) -> str:
    """`text` without its include guards: an `#ifndef X` straight followed by `#define X`, wherever it stands, and
    the `#endif` that closes it."""
    lines, out, stack, i = text.split("\n"), [], [], 0
    while i < len(lines):
        line = lines[i]
        if (
            (m := _IFNDEF.match(line))
            and i + 1 < len(lines)
            and re.match(rf"[ \t]*#[ \t]*define[ \t]+{m[1]}\b", lines[i + 1])
        ):
            stack.append(True)
            i += 2
            continue
        if _IF.match(line):
            stack.append(False)
        elif _ENDIF.match(line) and stack and stack.pop():
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _header_decls(includes: Sequence[str], header: Header) -> tuple[list[Decl], dict[str, str]]:
    """The unguarded prototypes and externs the quoted `#include` lines bring in, followed through the headers'
    own includes, and the typedefs of those headers; `header(name)` is the text of an include or None. A header's
    include guard is not a guard."""
    found: list[Decl] = []
    names: dict[str, str] = {}
    seen: set[str] = set()
    todo = [m[1] for line in includes if (m := _QUOTED.match(line))]
    while todo:
        name = todo.pop(0)
        text = None if name in seen else header(name)
        seen.add(name)
        if text is None:
            continue
        text = _unguard(text)
        todo += [m[1] for line in text.split("\n") if (m := _QUOTED.match(line))]
        found += [d for d in decls(text, name) if d.name and not d.guard and d.kind != "other"]
        names.update(aliases(text))
    return found, names


def reconcile(
    parts: Sequence[tuple[str, list[Decl]]], inherited: Sequence[Decl] = (), names: dict[str, str] | None = None
) -> tuple[dict[str, list[Decl]], list[Row], list[str]]:
    """Mark the declarations a merged file no longer needs (`drop`) and list the conflicts left and the prototypes
    dropped although they disagreed with an earlier definition. Only unguarded declarations are touched; a guarded
    one is left in place. `inherited` are the declarations of the included headers: they stand before every unit's
    text; `names` are the typedefs in force."""
    names = names or {}
    conflicts: list[Row] = []
    retyped: list[str] = []
    order = {unit: at for at, (unit, _) in enumerate(parts)}
    every = [d for _, group in parts for d in group if d.name]
    defined: dict[str, list[Decl]] = defaultdict(list)
    for decl in every:
        if decl.kind in ("def", "var") and not decl.guard:
            defined[decl.name].append(decl)
    for name, group in sorted(defined.items()):
        statics = [d for d in group if d.static]
        if len(statics) > 1:
            conflicts.append({"kind": "static", "name": name, "detail": f"static defined in {_units_of(statics)}"})
    declared: dict[tuple[str, str], Decl] = {}
    for decl in inherited:
        declared.setdefault((decl.name, decl.kind), decl)
    for name, group in sorted(defined.items()):
        for decl in group:
            kind = "proto" if decl.kind == "def" else "extern"
            earlier = declared.get((name, kind))
            if earlier is not None and not _alike(earlier.sig, decl.sig, names):
                conflicts.append(
                    {
                        "kind": kind,
                        "name": name,
                        "detail": f"`{decl.sig}` defined in {decl.unit} against `{earlier.sig}` in {earlier.unit}",
                    }
                )
    for decl in every:
        if decl.kind not in ("proto", "extern") or decl.guard:
            continue
        kind = decl.kind
        home = [d for d in defined.get(decl.name, ()) if d.kind == ("def" if kind == "proto" else "var")]
        if home:
            same = [d for d in home if _alike(d.sig, decl.sig, names)]
            if same:
                decl.drop = True
            elif kind == "proto" and (order[home[0].unit], home[0].start) < (order[decl.unit], decl.start):
                decl.drop = True
                retyped.append(
                    f"{decl.name}: `{decl.sig}` in {decl.unit} gives way to `{home[0].sig}` in {home[0].unit}"
                )
            else:
                conflicts.append(
                    {
                        "kind": kind,
                        "name": decl.name,
                        "detail": f"{decl.unit}: `{decl.sig}` against the definition `{home[0].sig}` in {home[0].unit}",
                    }
                )
            continue
        earlier = declared.get((decl.name, decl.kind))
        if earlier is None:
            declared[(decl.name, decl.kind)] = decl
        elif _alike(earlier.sig, decl.sig, names):
            decl.drop = True
        else:
            conflicts.append(
                {
                    "kind": kind,
                    "name": decl.name,
                    "detail": f"`{earlier.sig}` in {earlier.unit} against `{decl.sig}` in {decl.unit}",
                }
            )
    return {unit: [d for d in group if d.drop] for unit, group in parts}, conflicts, retyped


def _span(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else f"{names[0]}..{names[-1]} ({len(names)} names)"


def _grouped(conflicts: Sequence[Row]) -> list[Row]:
    """One conflict per kind and detail: names that disagree the same way (a header and a unit spelling a run of
    constants differently) are one finding that lists its names."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for conflict in conflicts:
        groups[conflict["kind"], conflict["detail"]].append(conflict["name"])
    return [{"kind": kind, "name": _span(names), "detail": detail} for (kind, detail), names in groups.items()]


def merge(units: Sequence[tuple[str, str]], droppable: Droppable, header: Header = lambda name: None) -> Row:
    """The merged text of `units` ((name, text) in link order) and the report:
    {"text", "includes", "pads": [names], "dropped": n, "conflicts": [{kind, name, detail}], "retyped": [text]}:
    `retyped` lists the prototypes dropped for a differing earlier definition, whose callers may need a cast."""
    seen: list[str] = []
    bodies: list[tuple[str, str]] = []
    pads: list[str] = []
    for name, text in units:
        depth, kept = 0, []
        for line in text.split("\n"):
            if _IF.match(line):
                depth += 1
            elif _ENDIF.match(line):
                depth = max(0, depth - 1)
            if depth == 0 and _INCLUDE.match(line):
                if (line := _norm(line)) not in seen:
                    seen.append(line)
                continue
            kept.append(line)
        body, dropped = drop_pads("\n".join(kept), name, droppable)
        pads += dropped
        bodies.append((name, body.strip("\n")))
    parts = [(name, decls(body, name)) for name, body in bodies]
    inherited, names = _header_decls(seen, header)
    for _, body in bodies:
        names.update(aliases(body))
    marked, conflicts, retyped = reconcile(parts, inherited, names)
    out = []
    removed = 0
    for name, body in bodies:
        spans = [(d.start, d.end) for d in marked[name]]
        removed += len(spans)
        out.append(f"/* {name.rsplit('/', 1)[-1]} */\n" + _cut(body, spans).strip("\n"))
    text = "\n".join(seen) + "\n\n" + "\n\n".join(out) + "\n"
    return {
        "text": re.sub(r"\n{3,}", "\n\n", text),
        "includes": len(seen),
        "pads": pads,
        "dropped": removed,
        "conflicts": _grouped(conflicts),
        "retyped": retyped,
    }
