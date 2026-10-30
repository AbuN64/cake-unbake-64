"""The census of declared types: every declaration of every name in the preprocessed units of every version.

A unit is read as its preprocessed text with line markers (`# 12 "include/a.H"`), so each statement keeps the file and
line it was written on, and the files a unit reads are the markers it holds. A statement at depth 0 is parsed into
(name, kind, type): `proto` and `fdef` for functions, `ext` for an extern, `vdef` for a variable defined here. The type
is normalised (typedefs resolved, parameter names and the first array bound dropped) so that two spellings of one type
compare equal. A name declared with more than one type in some version is a clash; `table` writes the clashes in the
CLASH.tsv shape. No project, version or type name is special: the typedefs come from the units themselves."""

from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

Decl = tuple[str, str, str, int]  # (loc "file:line", kind, normalised type, declarators in the statement)
Stmt = tuple[str, int, str, str]  # (file, line, text, hint "body" | "stmt")

HEADER = "name\tntypes\tdeclarations(type => file:line kind, versions)\tusers\tstatus\n"
_KW = frozenset(
    {"const", "volatile", "unsigned", "signed", "struct", "union", "enum", "void", "char", "short", "int", "long"}
    | {"float", "double", "register"}
)
_TAGS = ("struct", "union", "enum")
_ANONYMOUS = re.compile(r"\b(struct|union|enum)\s*(\w*)\s*\{ \}\s*(?=[\w*])")
_FUNCTION = re.compile(r"^([\w\s\*]+?)\b(\w+)\s*\(((?:[^()]|\([^()]*\))*)\)$")
_MORE = re.compile(r"^(\**)\s*(\w+)\s*\(((?:[^()]|\([^()]*\))*)\)$")
_MARK = re.compile(r'# (\d+) "([^"]*)"')


def ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_top(text: str) -> list[str]:
    """The comma-separated parts of `text` outside any bracket."""
    out, depth, cur = [], 0, ""
    for c in text:
        depth += (c in "([{") - (c in ")]}")
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    return [*out, cur]


_BOUND = re.compile(r"\[[^\]]*\]")
_HEX = re.compile(r"\b0[xX][0-9a-fA-F]+[uUlL]*\b")


def decimal_bounds(text: str) -> str:
    """`text` with every hex literal inside an array bound written in decimal: the one spelling `unbake audit` accepts
    (`source.hex-value`), so a bound copied from one declaration into another never carries a hex literal."""
    return _BOUND.sub(lambda b: _HEX.sub(lambda h: str(int(h[0].rstrip("uUlL"), 16)), b[0]), text)


def tkey(typ: str) -> str:
    """The type of an extern: the bound of its first dimension is not part of it, however it is spelled."""
    return _BOUND.sub("[]", typ, count=1) if "(" not in typ else typ


def files_of(text: str, unit: str) -> frozenset[str]:
    """The files a preprocessed text was written from: the unit and every file a line marker names."""
    found = {unit}
    for m in _MARK.finditer(text):
        found.add(unit if m[2] == "<stdin>" else m[2])
    return frozenset(found)


_SPECIAL = re.compile(r"[{};]")
_BODY = re.compile(r"\)\s*(\w+\s+[\w\s\*,\[\]]+;\s*)*\{\s*\}\s*$")


def scan(text: str) -> list[Stmt]:
    """The statements at depth 0 of a preprocessed text with line markers."""
    res: list[Stmt] = []
    cur, line = "?", 1
    depth, buf, start = 0, "", None
    for ln in text.split("\n"):
        if m := _MARK.match(ln):
            line, cur = int(m[1]), m[2]
            continue
        if ln.startswith("# "):
            continue
        at = 0
        for m in _SPECIAL.finditer(ln):  # the text between braces and semicolons is read whole, not per character
            if depth == 0 and m.start() > at:
                if start is None and not ln[at : m.start()].isspace():
                    start = (cur, line)
                buf += ln[at : m.start()]
            at = m.end()
            c = m[0]
            if c == "{":
                depth += 1
                if depth == 1:
                    buf += " { "
            elif c == "}":
                depth -= 1
                if depth == 0:
                    buf += " } "
                    # a function body ends the statement; aggregates continue until ';'
                    # `{ }`: the body is skipped
                    if _BODY.search(buf) and not re.match(r"^\s*(typedef|struct|union|enum)\b", buf) and "=" not in buf:
                        res.append((start[0], start[1], buf, "body") if start else (cur, line, buf, "body"))
                        buf, start = "", None
            elif depth == 0:
                if start:
                    res.append((start[0], start[1], buf, "stmt"))
                buf, start = "", None
        if depth == 0 and at < len(ln):
            if start is None and not ln[at:].isspace():
                start = (cur, line)
            buf += ln[at:]
        buf += " "
        line += 1
    return res


class Norm:
    """Type normalisation over one set of typedefs."""

    def __init__(self, typedefs: Mapping[str, str] | None = None) -> None:
        self.typedefs: dict[str, str] = dict(typedefs or {})

    def define(self, stmt: str) -> None:
        s = ws(stmt)
        if not s.startswith("typedef "):
            return
        m = re.match(r"^typedef\s+(struct|union|enum)\s+(\w+)?\s*(?:\{ \})?\s*(\w+)$", s)
        if m:
            if m[2]:
                self.typedefs.setdefault(m[3], m[1] + " " + m[2])
            return
        m = re.match(r"^typedef\s+([\w\s]+?)\s*(\w+)$", s)
        if m and "(" not in s:
            self.typedefs.setdefault(m[2], m[1])

    def prim(self, t: str) -> str:
        toks = re.findall(r"[A-Za-z_]\w*|\*|\[|\]|\(|\)|,|\.\.\.|\d+", t)
        out = []
        for i, x in enumerate(toks):
            if (toks[i - 1] if i else "") not in _TAGS:
                for _ in range(5):
                    if x not in self.typedefs:
                        break
                    x = self.typedefs[x]
            out.append(x)
        s = " ".join(out)
        for a, b in ((" * ", "*"), (" *", "*"), ("* ", "*"), (" [", "["), (" ]", "]"), ("[ ", "["), (" ( ", "(")):
            s = s.replace(a, b)
        for a, b in ((" )", ")"), ("( ", "("), (" ,", ","), ("long int", "long"), ("short int", "short")):
            s = s.replace(a, b)
        return re.sub(r"\bunsigned\b(?! (int|char|short|long))", "unsigned int", s)

    def param(self, p: str) -> str:
        p = ws(p)
        if p in ("", "void", "..."):
            return p
        if "(" in p:
            return self.prim(re.sub(r"\(\s*\*\s*\w+\s*\)", "(*)", p))
        m = re.search(r"((?:\[[^\]]*\]\s*)+)$", p)
        arr = m[1] if m else ""
        toks = re.findall(r"[A-Za-z_]\w*|\*", p[: len(p) - len(arr)] if arr else p)
        names = [t for t in toks if t != "*"]
        named = len(names) >= 2 and names[-1] not in _KW and names[-1] not in self.typedefs
        if named and names[-2] not in _TAGS and not (names[-2] == "const" and len(names) == 2):
            toks = toks[:-1]
        return self.prim(" ".join(toks) + ("*" if arr else ""))

    def prototypes(self, head: str, kind: str) -> list[tuple[str, str, str, int]]:
        """The functions of `T f(..), *g(..)`: one declaration list sharing the first one's base type; empty for a head
        that is not one."""
        parts = split_top(head)
        first = _FUNCTION.match(parts[0].strip())
        if len(parts) < 2 or not first or not first[1].strip():
            return []
        base = first[1].rstrip(" *")
        out = [(first[2], kind, self.prim(first[1]) + "(" + self.args(first[3]) + ")", len(parts))]
        for part in parts[1:]:
            more = _MORE.match(part.strip())
            if not more:
                return []
            out.append((more[2], kind, self.prim(base + more[1]) + "(" + self.args(more[3]) + ")", len(parts)))
        return out

    def args(self, text: str) -> str:
        args = ",".join(self.param(a) for a in split_top(text))
        return "" if args == "void" else args

    def statement(self, buf: str, hint: str) -> list[tuple[str, str, str, int]]:
        """(name, kind, type, declarators) of every name one statement declares."""
        s = ws(buf)
        if not s or s.startswith("typedef ") or re.match(r"^(static|asm|__asm|inline|__inline)\b", s):
            return []
        ext = s.startswith("extern ")
        s2 = s[7:] if ext else s
        if re.match(r"^(struct|union|enum)\s*\w*\s*(\{ \})?$", s2):
            return []
        s2 = _ANONYMOUS.sub(
            lambda a: f"{a[1]} {a[2] or '__anonymous'} ", s2
        )  # `struct { } name[8] = ..`: a variable of it
        body = s2.endswith("}") and hint == "body"
        init = False
        head = s2
        if body:
            head = s2[: s2.index("{")].strip()
        elif "=" in s2:
            head, init = s2.split("=")[0], True
        head = head.strip()
        kind = "ext" if ext and not init else "vdef"
        m = re.match(r"^([\w\s\*]+?)\(\s*\*\s*(\w+)\s*\)\s*\(((?:[^()]|\([^()]*\))*)\)$", head)
        if m:  # function pointer variable: T (*name)(args)
            return [(m[2], kind, self.prim(m[1]) + "(*)(" + ",".join(self.param(a) for a in split_top(m[3])) + ")", 1)]
        if not init and (listed := self.prototypes(head, "fdef" if body else "proto")):
            return listed
        m = re.match(r"^([\w\s\*]+?)\b(\w+)\s*\(((?:[^()]|\([^()]*\))*)\)$", head)
        if m and m[1].strip() and m[2] not in ("if", "for", "while", "switch", "return", "sizeof") and not init:
            args = ",".join(self.param(a) for a in split_top(m[3]))
            return [
                (m[2], "fdef" if body else "proto", self.prim(m[1]) + "(" + ("" if args == "void" else args) + ")", 1)
            ]
        if body or not re.match(r"^[A-Za-z_*]", head):
            return []
        parts = split_top(head)
        m = re.match(r"^((?:\w+\s+|\w+\s*\*+\s*)*?)(\*{0,}\s*)(\w+)\s*((?:\[[^\]]*\]\s*)*)$", parts[0].strip())
        base = m[1].strip() if m else ""
        if m and not base and (ext or m[2]):  # `extern *p;`: no type word is `int`
            base = "int"
        words = re.findall(r"[A-Za-z_]\w*", base)
        if not m or not base or all(t in ("extern", "const", "volatile") for t in words):
            return []
        out = [
            (
                m[3],
                kind,
                self.prim(base) + m[2].replace(" ", "") + decimal_bounds(ws(m[4]).replace(" ", "")),
                len(parts),
            )
        ]
        for p in parts[1:]:
            m2 = re.match(r"^\s*(\*{0,}\s*)(\w+)\s*((?:\[[^\]]*\]\s*)*)$", p)
            if m2:
                out.append(
                    (
                        m2[2],
                        kind,
                        self.prim(base) + m2[1].replace(" ", "") + decimal_bounds(ws(m2[3]).replace(" ", "")),
                        len(parts),
                    )
                )
        return out


def declarations(stmts: Iterable[Stmt], norm: Norm, keep: Callable[[str], bool]) -> list[tuple[str, Decl]]:
    """(name, Decl) of every declaration written in a file `keep` accepts."""
    found = []
    for file, line, text, hint in stmts:
        if keep(file):
            for name, kind, typ, count in norm.statement(text, hint):
                found.append((name, (f"{file}:{line}", kind, typ, count)))
    return found


def clashes(per: Mapping[str, Mapping[str, Sequence[Decl]]]) -> dict[str, Mapping[str, Sequence[Decl]]]:
    """The names declared with more than one type in some version: name -> version -> declarations."""
    return {name: vers for name, vers in per.items() if any(len({tkey(d[2]) for d in ds}) > 1 for ds in vers.values())}


def table(
    per: Mapping[str, Mapping[str, Sequence[Decl]]],
    users: Mapping[str, Sequence[str]],
    status: str | Mapping[str, str] = "open",
) -> str:
    """The clashes in the CLASH.tsv shape: one row per name with its types, where each is declared and its users. The
    status is one word for every row, or per name (a name the mapping lacks is `open`)."""
    lines = [HEADER]
    for name, vers in sorted(clashes(per).items()):
        groups: OrderedDict[str, dict[str, set[str]]] = OrderedDict()
        for ver, ds in vers.items():
            for loc, kind, typ, _ in ds:
                groups.setdefault(tkey(typ), {}).setdefault(f"{loc} {kind}", set()).add(ver)
        decl = " | ".join(
            f"{t} => " + ", ".join(f"{loc} [{','.join(sorted(v))}]" for loc, v in sorted(locs.items()))
            for t, locs in groups.items()
        )
        held = sorted(users.get(name, ()))
        word = status if isinstance(status, str) else status.get(name, "open")
        lines.append(f"{name}\t{len(groups)}\t{decl}\t{len(held)}: {' '.join(held)}\t{word}\n")
    return "".join(lines)


def index(files: Mapping[str, str]) -> dict[str, list[str]]:
    """word -> the files whose text holds it, for the users column."""
    held: defaultdict[str, set[str]] = defaultdict(set)
    for path, text in files.items():
        for word in set(re.findall(r"[A-Za-z_]\w*", text)):
            held[word].add(path)
    return {w: sorted(p) for w, p in held.items()}
