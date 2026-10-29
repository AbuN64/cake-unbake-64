"""A small reader of the C headers a project already has: struct and union layouts, typedefs, extern objects and
function prototypes. It reads declarations only (no preprocessing, no expressions beyond integer arithmetic); a type it
cannot size makes its struct's layout unknown from that field on, and says so instead of guessing."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r"0[xX][0-9A-Fa-f]+|\d+|[A-Za-z_]\w*|<<|>>|[{}();,*\[\]=:+\-/<>&|~!?%^.]")
_BASE = {
    "s8": (1, "s8"), "u8": (1, "u8"), "char": (1, "s8"), "bool": (1, "u8"),
    "s16": (2, "s16"), "u16": (2, "u16"), "short": (2, "s16"),
    "s32": (4, "s32"), "u32": (4, "u32"), "int": (4, "s32"), "long": (4, "s32"), "unsigned": (4, "u32"),
    "f32": (4, "f32"), "float": (4, "f32"),
    "s64": (8, "s64"), "u64": (8, "u64"), "f64": (8, "f64"), "double": (8, "f64"),
}  # fmt: skip
_QUALIFIERS = frozenset({"const", "volatile", "static", "extern", "register", "inline", "signed", "__inline"})
_PAD = re.compile(r"^(_?pad|reserved|unk|unused|padding)", re.IGNORECASE)


@dataclass(frozen=True)
class CType:
    """A resolved type: `kind` is int, float, ptr, struct, enum or void; `size` None when it could not be sized."""

    text: str
    kind: str
    size: int | None
    align: int = 1
    signed: bool | None = None
    pointee: str | None = None  # ptr: the struct (or typedef) name pointed to, when it is one
    struct: str | None = None  # struct: the record's name
    count: int | None = None  # array element count (kind is then the element's)
    stride: int | None = None
    array: bool = False


@dataclass
class Field:
    name: str
    offset: int
    type: CType
    pad: bool = False


@dataclass
class Struct:
    name: str
    size: int | None
    fields: list[Field]
    union: bool = False
    complete: bool = True  # False when a field could not be sized: offsets after it are not trusted
    path: str = ""


@dataclass
class Prototype:
    name: str
    returns: str
    params: list[str]  # type text of each parameter, as written


@dataclass
class Headers:
    structs: dict[str, Struct] = field(default_factory=dict)
    typedefs: dict[str, str] = field(default_factory=dict)  # alias -> type text
    objects: dict[str, str] = field(default_factory=dict)  # global -> type text, e.g. "Foo *" or "u8 [16]"
    prototypes: dict[str, Prototype] = field(default_factory=dict)
    _raw: dict[str, tuple[list[str], bool, str]] = field(default_factory=dict)  # tag -> (body tokens, union, path)
    _cache: dict[str, Struct] = field(default_factory=dict)

    def copy(self) -> Headers:
        """The declarations read so far, to which a file's own may be added without touching these."""
        return Headers(typedefs=dict(self.typedefs), _raw=dict(self._raw))

    # ---- reading

    def read(self, root: Path, patterns: Iterable[str] | None = None, sources: Iterable[str] | None = None) -> None:
        """Declarations of the header files; of the source files only the function signatures (their structs and
        objects are file-local, so two files may name different things alike). By default the header and source roots of
        the project's layout (include and src unless its config.toml says otherwise)."""
        from unbake.config import layout_at  # config does not import this module; the import is local to stay so

        layout = layout_at(root)
        if patterns is None:
            patterns = [f"{r}/**/*{e}" for r in layout["header_roots"] for e in (".H", ".h")]
        if sources is None:
            sources = [f"{r}/**/*{e}" for r in layout["source_roots"] for e in (".CPP", ".c")]
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                self.add(path.read_text(errors="replace"), path.relative_to(root).as_posix())
        for pattern in sources:
            for path in sorted(root.glob(pattern)):
                local = Headers()
                local.add(path.read_text(errors="replace"), path.relative_to(root).as_posix())
                for name, proto in local.prototypes.items():
                    self.prototypes.setdefault(name, proto)

    def add(self, text: str, path: str = "") -> None:
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        text = re.sub(r"//[^\n]*", " ", text)
        text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        tokens = _TOKEN.findall(text)
        i = 0
        while i < len(tokens):
            i = self._declaration(tokens, i, path)

    def _declaration(self, t: list[str], i: int, path: str) -> int:
        start = i
        depth = 0
        while i < len(t):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
            elif t[i] == ";" and depth == 0:
                break
            elif t[i] == ")" and depth == 0 and i + 1 < len(t) and t[i + 1] == "{":  # a function definition
                self._function(t[start : i + 1])
                return _skip_block(t, i + 1)
            i += 1
        self._simple(t[start:i], path)
        return i + 1

    def _function(self, d: list[str]) -> None:
        if "(" not in d:
            return
        at = d.index("(")
        if at == 0 or not re.match(r"[A-Za-z_]", d[at - 1]):
            return
        params = _split_params(d[at + 1 : len(d) - 1])
        self.prototypes.setdefault(d[at - 1], Prototype(d[at - 1], _text(d[: at - 1]), params))

    def _simple(self, d: list[str], path: str) -> None:
        if not d:
            return
        d = [x for x in d if x not in ("extern", "static", "const", "volatile", "inline")] if d[0] != "typedef" else d
        if d[0] == "typedef":
            self._typedef(d[1:], path)
            return
        if d[0] in ("struct", "union") and len(d) >= 3 and d[2] == "{":
            body, rest = _body(d, 2)
            self._register(d[1], body, d[0] == "union", path)
            if rest:
                self._object(d[:2] + rest)
            return
        if "(" in d:
            self._function([*d, ")"] if d[-1] != ")" else d)
            return
        if len(d) == 2 and d[0] in ("struct", "union", "enum"):  # a forward declaration
            return
        self._object(d)

    def _object(self, d: list[str]) -> None:
        names = [x for x in d if re.match(r"[A-Za-z_]", x)]
        if len(d) < 2 or not names:
            return
        if "[" in d:
            name = d[d.index("[") - 1]
            self.objects.setdefault(name, _text(d[: d.index("[") - 1]) + " " + _text(d[d.index("[") :]))
        else:
            self.objects.setdefault(d[-1], _text(d[:-1]))

    def _typedef(self, d: list[str], path: str) -> None:
        if not d:
            return
        if d[0] in ("struct", "union", "enum"):
            tag = d[1] if len(d) > 1 and d[1] != "{" else None
            if "{" in d:
                open_at = d.index("{")
                body, rest = _body(d, open_at)
                alias = rest[-1] if rest else None
                if d[0] == "enum":
                    if alias:
                        self.typedefs[alias] = "int"
                    return
                name = tag or alias
                if name:
                    self._register(name, body, d[0] == "union", path)
                if alias and tag and alias != tag:
                    self.typedefs[alias] = f"struct {tag}"
                return
            if d[0] == "enum":
                self.typedefs[d[-1]] = "int"
                return
        if "(" in d:  # function pointer typedef: a pointer
            for k, x in enumerate(d):
                if x == "*" and k + 1 < len(d) and re.match(r"[A-Za-z_]", d[k + 1]):
                    self.typedefs[d[k + 1]] = "void *"
            return
        if "[" in d:
            at = d.index("[")
            self.typedefs[d[at - 1]] = _text(d[: at - 1]) + " " + _text(d[at:])
            return
        self.typedefs[d[-1]] = _text(d[:-1])

    def _register(self, tag: str, body: list[str], union: bool, path: str) -> None:
        self._raw[tag] = (body, union, path)
        self._cache.pop(tag, None)

    # ---- types

    def resolve(self, text: str, seen: frozenset[str] = frozenset()) -> CType:
        """The CType of a type written as text such as `u8`, `struct Foo *`, `Foo [4]`, `Bar`."""
        words = [w for w in _TOKEN.findall(text) if w not in _QUALIFIERS]
        array = [w for w in words if w == "["]
        if array:
            at = words.index("[")
            element = self.resolve(_text(words[:at]), seen)
            dims = _dimensions(words[at:], self)
            if dims is None or element.size is None:
                return CType(text, "array", None, element.align, array=True)
            count = 1
            for d in dims:
                count *= d
            return CType(text, element.kind, element.size * count, element.align, element.signed, element.pointee,
                         element.struct, count, element.size, True)  # fmt: skip
        if (words and words[-1] == "*") or "*" in words:
            stars = words.count("*")
            base = [w for w in words if w != "*"]
            name = base[-1] if base else None
            if name in ("struct", "union") and len(base) >= 1:
                name = None
            pointee = name if stars == 1 else None
            return CType(text, "ptr", 4, 4, None, pointee)
        if not words:
            return CType(text, "void", 0)
        if words[0] in ("struct", "union") and len(words) >= 2:
            struct = self.struct(words[1])
            return CType(text, "struct", struct.size if struct else None, _struct_align(struct), None, None, words[1])
        if words[0] == "enum":
            return CType(text, "enum", 4, 4, True)
        if words[0] == "void":
            return CType(text, "void", 0)
        low = [w for w in words if w not in ("signed",)]
        if "long" in low and low.count("long") == 2:
            return CType(text, "int", 8, 8, "unsigned" not in low)
        if "double" in low:
            return CType(text, "float", 8, 8, None)
        name = words[-1]
        if low and low[0] == "unsigned" and len(low) >= 2:
            size = {"char": 1, "short": 2, "int": 4, "long": 4}.get(low[-1], 4)
            return CType(text, "int", size, size, False)
        if name in _BASE:
            size, tag = _BASE[name]
            kind = "float" if tag.startswith("f") else "int"
            signed = None if kind == "float" else tag.startswith("s")
            return CType(text, kind, size, min(size, 8), signed)
        if name in self.typedefs and name not in seen:
            inner = self.resolve(self.typedefs[name], seen | {name})
            if inner.kind == "struct":
                return inner
            return inner
        struct = self.struct(name)
        if struct is not None:
            return CType(text, "struct", struct.size, _struct_align(struct), None, None, name)
        return CType(text, "unknown", None)

    def struct(self, name: str) -> Struct | None:
        name = self._tag(name)
        if name in self._cache:
            return self._cache[name]
        if name not in self._raw:
            return None
        body, union, path = self._raw[name]
        self._cache[name] = Struct(name, None, [], union, False, path)  # a recursion guard
        built = self._layout(name, body, union, path)
        self._cache[name] = built
        return built

    def _tag(self, name: str) -> str:
        seen = set()
        while name not in self._raw and name in self.typedefs and name not in seen:
            seen.add(name)
            target = self.typedefs[name].split()
            name = target[-1] if target else name
        return name

    def _layout(self, name: str, body: list[str], union: bool, path: str) -> Struct:
        fields: list[Field] = []
        offset = 0
        biggest = 1
        complete = True
        top = 0
        for decl in _members(body):
            parsed = self._member(decl)
            if parsed is None:
                complete = False
                break
            for fname, ctype in parsed:
                if fname == "" and ctype.struct:
                    inner = self.struct(ctype.struct)
                    if inner is None or not inner.complete:
                        complete = False
                        break
                    align = max(ctype.align, 1)
                    biggest = max(biggest, align)
                    at = 0 if union else (offset + align - 1) // align * align
                    fields.extend(Field(f.name, at + f.offset, f.type, f.pad) for f in inner.fields)
                    top = max(top, at + (inner.size or 0))
                    offset = at + (inner.size or 0)
                    continue
                if ctype.size is None:
                    complete = False
                    break
                align = max(ctype.align, 1)
                biggest = max(biggest, align)
                at = 0 if union else (offset + align - 1) // align * align
                fields.append(Field(fname, at, ctype, bool(_PAD.match(fname))))
                top = max(top, at + ctype.size)
                offset = at + ctype.size
            if not complete:
                break
        size = (top + biggest - 1) // biggest * biggest if complete else None
        return Struct(name, size, fields, union, complete, path)

    def _member(self, d: list[str]) -> list[tuple[str, CType]] | None:
        d = [w for w in d if w not in _QUALIFIERS or w == "signed"]
        if not d or ":" in d:
            return None
        if d[0] in ("struct", "union", "enum") and "{" in d:
            open_at = d.index("{")
            inner, rest = _body(d, open_at)
            if d[0] == "enum":
                return [(rest[-1], CType("enum", "enum", 4, 4, True))] if rest else None
            tag = f"__anon_{len(self._raw)}"
            self._raw[tag] = (inner, d[0] == "union", "")
            ctype = self.resolve(f"struct {tag}" + (" " + _text(rest[1:]) if len(rest) > 1 else ""))
            if not rest:  # an anonymous member: its fields belong to the enclosing record
                return [("", ctype)] if ctype.size is not None else None
            return [(rest[0], self._with_dims(ctype, rest[1:]))]
        if any(d[k] == "(" and d[k + 1] == "*" for k in range(len(d) - 1)):  # a function pointer member
            for k, x in enumerate(d):
                if x == "*" and k + 1 < len(d) and re.match(r"[A-Za-z_]", d[k + 1]):
                    return [(d[k + 1], CType("fn *", "ptr", 4, 4))]
            return None
        out = []
        for part in _commas(d):
            dims = part.index("[") if "[" in part else len(part)
            if dims < 2:
                return None
            name = part[dims - 1]
            base = _text(part[: dims - 1])
            ctype = self.resolve(base + (" " + _text(part[dims:]) if dims < len(part) else ""))
            out.append((name, ctype))
        return out

    def _with_dims(self, ctype: CType, tail: list[str]) -> CType:
        return self.resolve(f"struct {ctype.struct} {_text(tail)}") if tail else ctype

    def locate(self, struct: Struct, offset: int, depth: int = 0) -> tuple[str, Field, int] | None:
        """The innermost field (as a dotted path, the field, its absolute offset) holding byte `offset` of the
        struct, descending into nested records and into arrays of them, or None past its end or in unsized layout."""
        if depth > 6:
            return None
        for f in struct.fields:
            size = f.type.size or 0
            if f.offset <= offset < f.offset + max(size, 1) and f.offset + size > f.offset:
                if f.type.kind == "struct" and f.type.struct:
                    inner = self.struct(f.type.struct)
                    if inner is not None and inner.complete:
                        stride = f.type.stride or f.type.size or 1
                        element = (offset - f.offset) % stride if f.type.array else offset - f.offset
                        index = (offset - f.offset) // stride if f.type.array else None
                        sub = self.locate(inner, element, depth + 1)
                        if sub is not None:
                            label = f.name + (f"[{index}]" if index is not None else "")
                            base = offset - element
                            return f"{label}.{sub[0]}", sub[1], sub[2] + base
                return f.name, f, f.offset
        return None


def _struct_align(struct: Struct | None) -> int:
    return max((f.type.align for f in struct.fields), default=1) if struct else 1


def _text(tokens: list[str]) -> str:
    return " ".join(tokens)


def _skip_block(t: list[str], i: int) -> int:
    depth = 0
    while i < len(t):
        depth += t[i] == "{"
        depth -= t[i] == "}"
        i += 1
        if depth == 0:
            break
    return i


def _body(d: list[str], open_at: int) -> tuple[list[str], list[str]]:
    """(tokens inside the braces opened at open_at, tokens after the closing brace)."""
    depth = 0
    for k in range(open_at, len(d)):
        depth += d[k] == "{"
        depth -= d[k] == "}"
        if depth == 0:
            return d[open_at + 1 : k], d[k + 1 :]
    return d[open_at + 1 :], []


def _members(body: list[str]) -> list[list[str]]:
    out: list[list[str]] = []
    cur: list[str] = []
    depth = 0
    for x in body:
        if x == "{":
            depth += 1
        elif x == "}":
            depth -= 1
        if x == ";" and depth == 0:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(x)
    if cur:
        out.append(cur)
    return out


def _commas(d: list[str]) -> list[list[str]]:
    """`T a, *b, c[2]` as three declarators, each carrying the base type words."""
    parts: list[list[str]] = []
    cur: list[str] = []
    depth = 0
    for x in d:
        depth += x in "[("
        depth -= x in "])"
        if x == "," and depth == 0:
            parts.append(cur)
            cur = []
        else:
            cur.append(x)
    parts.append(cur)
    if len(parts) == 1:
        return parts
    first = parts[0]
    name_at = (first.index("[") if "[" in first else len(first)) - 1
    base = [w for w in first[:name_at] if w != "*"]
    return [first] + [base + p for p in parts[1:]]


def _dimensions(tokens: list[str], headers: Headers) -> list[int] | None:
    dims: list[int] = []
    cur: list[str] = []
    depth = 0
    for x in tokens:
        if x == "[":
            depth += 1
            cur = [] if depth == 1 else [*cur, x]
        elif x == "]":
            depth -= 1
            if depth == 0:
                value = _evaluate(cur, headers)
                if value is None or value < 0:
                    return None
                dims.append(value)
            else:
                cur.append(x)
        elif depth:
            cur.append(x)
    return dims


def _evaluate(tokens: list[str], headers: Headers) -> int | None:
    tokens = list(tokens)
    while "sizeof" in tokens:
        at = tokens.index("sizeof")
        depth, end = 0, at + 1
        while end < len(tokens):
            depth += tokens[end] == "("
            depth -= tokens[end] == ")"
            end += 1
            if depth == 0:
                break
        size = headers.resolve(_text(tokens[at + 2 : end - 1])).size
        if size is None:
            return None
        tokens[at:end] = [str(size)]
    text = " ".join(tokens)
    if (
        not tokens
        or not re.fullmatch(r"[0-9xXa-fA-F\s+\-*/()<>&|~%^]+", text)
        or re.search(r"[a-fA-F]", re.sub(r"0[xX][0-9a-fA-F]+", "", text))
    ):
        return None
    try:
        return int(eval(text.replace("/", "//"), {"__builtins__": {}}, {}))
    except (SyntaxError, ZeroDivisionError, ValueError, TypeError):
        return None


def _split_params(tokens: list[str]) -> list[str]:
    if tokens == ["void"] or not tokens:
        return []
    out: list[str] = []
    cur: list[str] = []
    depth = 0
    for x in tokens:
        depth += x == "("
        depth -= x == ")"
        if x == "," and depth == 0:
            out.append(_text(cur))
            cur = []
        else:
            cur.append(x)
    out.append(_text(cur))
    return out


def parameter_type(param: str) -> str:
    """The type text of one parameter declaration, its name dropped: `struct Foo *arg0` -> `struct Foo *`."""
    words = _TOKEN.findall(param)
    if "(" in words:
        return "void *"
    if "[" in words:
        words = [*words[: words.index("[")], "*"] if len(words) > 2 else words
        return _text(words if words[-1] == "*" else [*words[:-1], "*"])
    if (
        len(words) >= 2
        and re.match(r"[A-Za-z_]", words[-1])
        and (words[-2] == "*" or len(words) > 2 or words[-1] not in _BASE)
    ):
        return _text(words[:-1])
    return _text(words)
