"""Added-debt checks of rules.toml: hex-value, raw-offset-deref, raw-data, placeholder-name.

Each check reads the code scope of one text (comments, strings and directives blanked, line numbers kept) and returns
{line: subject}; policy.scope turns a row a change adds into a refusal, so the old tree's leftovers never block anyone.
Thresholds come from rules.toml [audit] with the project's config.toml [audit] keys over them. The scanner is lexical:
it reads tokens, brackets and operator precedence, never a compiler, so it runs on the checkout alone (CI, no ROM,
no toolchain).
"""

from __future__ import annotations

import re
import struct
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

from unbake import symbol_names

Json = Mapping[str, Any]

_TOKEN = re.compile(
    r"0[xX][0-9A-Fa-f]+[uUlL]*|\d[\w.]*|[A-Za-z_]\w*|->|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^]=|\+\+|--|\S"
)
_HEX = re.compile(r"0[xX][0-9A-Fa-f]+[uUlL]*$")
_IDENT = re.compile(r"[A-Za-z_]\w*$")
_KEYWORDS = frozenset(
    {
        *("if", "while", "for", "switch", "return", "case", "sizeof", "else", "do", "goto", "default", "break"),
        *("continue", "typedef", "static", "extern", "const", "volatile", "struct", "union", "enum", "register"),
    }
)
_TYPES = frozenset(
    {
        *("void", "char", "short", "int", "long", "float", "double", "signed", "unsigned"),
        *("u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64", "f32", "f64", "vu8", "vu16", "vu32", "vu64"),
        *("const", "volatile", "struct", "union", "enum"),
    }
)
_PRECEDENCE = {
    **dict.fromkeys(("*", "/", "%"), 12),
    **dict.fromkeys(("+", "-"), 11),
    **dict.fromkeys(("<<", ">>"), 10),
    **dict.fromkeys(("<", ">", "<=", ">="), 9),
    **dict.fromkeys(("==", "!="), 8),
    "&": 7,
    "^": 6,
    "|": 5,
    "&&": 4,
    "||": 3,
    **dict.fromkeys(("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="), 1),
}
_BITOPS = frozenset(("&", "|", "^", "&=", "|=", "^="))
_SHIFTS = frozenset(("<<", ">>", "<<=", ">>="))
_COMPARE = frozenset(("==", "!=", "<", ">", "<=", ">="))
_ARITH = frozenset(("+", "-", "*", "/", "%", "+=", "-=", "*=", "/=", "%="))
_OPENERS = {")": "(", "]": "[", "}": "{"}
_PAD_NAME = re.compile(r"(?i)(pad|unk|_?unused|filler|gap|reserved)")
_STATEMENT_END = frozenset((";", "{", "}"))


def _tokens(code: str) -> list[tuple[str, int]]:
    return [(m[0], m.start()) for m in _TOKEN.finditer(code)]


def _matches(tokens: Sequence[tuple[str, int]]) -> dict[int, int]:
    """token index of each bracket -> index of its partner (an unbalanced bracket has none)."""
    stack: list[int] = []
    partner: dict[int, int] = {}
    for at, (text, _) in enumerate(tokens):
        if text in "([{" and len(text) == 1:
            stack.append(at)
        elif text in _OPENERS and stack and tokens[stack[-1]][0] == _OPENERS[text]:
            start = stack.pop()
            partner[start], partner[at] = at, start
    return partner


def _integer(text: str) -> int:
    return int(re.sub(r"[uUlL]+$", "", text), 0)


# ---------------------------------------------------------------- hex-value


def _flag(settings: Json, name: str) -> re.Pattern[str]:
    return _compiled(settings["hex"][name])


@cache
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _as_float(word: int, low: float, high: float) -> bool:
    value = struct.unpack(">f", struct.pack(">I", word & 0xFFFFFFFF))[0]
    return value == value and value not in (float("inf"), float("-inf")) and value != 0 and low <= abs(value) <= high


def _fourcc(value: int) -> bool:
    raw = value.to_bytes(4, "big")
    return (
        all(c < 0x80 and (chr(c).isalnum() or c in b" _") for c in raw)
        and sum(chr(c).isalpha() for c in raw) >= 2
        and raw[0] != 0x20
    )


def _pow2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _hexround(value: int, step: int) -> bool:
    """Byte-structured: every byte 00 or FF, or a run of trailing zero nibbles (packed fields, colors, masks)."""
    return all(c in (0, 0xFF) for c in value.to_bytes(4, "big")) or (value >= step and value % step == 0)


def _decround(value: int) -> bool:
    return value % 100 == 0 or len(str(value)) <= 3


def _value_rule(value: int, ctx: str, detail: str, hexed: Json, flag: re.Pattern[str]) -> tuple[str, str]:
    """(class, verdict) of a literal whose meaning is its value (assignments, inits, arguments, compares, cases,
    arithmetic): the decision table hexcensus pinned. KEEP is legitimate hex, DEC a mechanical rewrite to decimal,
    REVIEW a class a person must read."""
    if value <= hexed["small_max"]:
        return "small", "DEC"
    if value in hexed["sentinels"]:
        return "sentinel", "KEEP"
    low, high = hexed["address"]
    if value >= 0x80000000:
        if low <= value <= high:
            return "address", "REVIEW"
        if hexed["hardware"][0] <= value <= hexed["hardware"][1]:
            return "hwreg", "KEEP"
    if value >= 0x10000:
        if _as_float(value, *hexed["float_range"]):
            return "float-bits", "REVIEW"
        if _fourcc(value):
            return "fourcc", "REVIEW"
        if _hexround(value, hexed["round_step"]) or value >= 0x80000000:
            return "packed", "KEEP"
        return ("large-dec", "DEC") if _decround(value) else ("large", "REVIEW")
    if (ctx in ("ASSIGN", "INIT") or (ctx == "COMPARE" and detail[:2] in ("==", "!="))) and flag.search(detail):
        return "flag-named", "KEEP"
    if _pow2(value) and value >= hexed["pow2_min"]:
        return "pow2", "REVIEW"
    if value >= hexed["pow2_min"] and _hexround(value, hexed["round_step"]):
        return "packed", "KEEP"
    return ("id" if value >= hexed["pow2_min"] else "value"), "DEC"


def _rule(ctx: str, detail: str, value: int, hexed: Json, flag: re.Pattern[str]) -> tuple[str, str]:
    """(class, verdict) of one hex literal by its context (the port of hexcensus.rule)."""
    if ctx == "BITOP":
        return "bitop", "KEEP"
    if ctx == "CALL" and _compiled(hexed["gbi"]).match(detail.split(":")[0]):
        return "gbi", "KEEP"
    small = value < 0x10000
    if ctx == "SHIFTCOUNT":
        return "shiftcount", "DEC"
    if ctx in ("PADSIZE", "STRUCT"):
        return ("struct-layout" if ctx == "PADSIZE" else "struct-offset"), "REVIEW"
    if ctx == "PTRCAST":
        return ("small", "DEC") if value == 0 else ("address", "REVIEW")
    if ctx == "INDEX":
        return ("index", "DEC") if small else ("index-large", "REVIEW")
    if ctx in ("ARRAYSIZE", "COND", "LOGIC", "UNARY"):
        return ctx.lower(), "DEC" if small and value not in hexed["sentinels"] else "REVIEW"
    if ctx == "ARITH" and detail in ("*", "/", "%", "*=", "/=", "%=") and _pow2(value) and value >= hexed["pow2_min"]:
        return "fixed-point", "KEEP"
    if ctx == "OTHER":
        return "other", "REVIEW"
    return _value_rule(value, ctx, detail, hexed, flag)


class _Scan:
    """One text's tokens, bracket partners and the brace kind of every position."""

    def __init__(self, code: str) -> None:
        self.t = _tokens(code)
        self.text = [t for t, _ in self.t]
        self.partner = _matches(self.t)
        self.line_at = _line_starts(code)
        self.braces = self._braces()

    def _statement_start(self, at: int) -> int:
        depth = 0
        for j in range(at - 1, -1, -1):
            tok = self.text[j]
            if tok in (")", "]"):
                depth += 1
            elif tok in ("(", "["):
                depth -= 1
            elif depth <= 0 and tok in _STATEMENT_END:
                return j + 1
        return 0

    def _braces(self) -> list[tuple[str, ...]]:
        """for each token, the kinds of the braces around it: body, init (file-scope or static data), local (a local
        initializer), enum, record."""
        stack: list[str] = []
        out: list[tuple[str, ...]] = []
        for at, tok in enumerate(self.text):
            out.append(tuple(stack))
            if tok == "}" and stack:
                stack.pop()
            elif tok == "{":
                prev = self.text[at - 1] if at else ";"
                start = self._statement_start(at)
                head = self.text[start:at]
                if "enum" in head:
                    kind = "enum"
                elif prev == "=":
                    kind = "local" if "body" in stack and "static" not in head else "init"
                elif prev in (",", "{") and stack and stack[-1] in ("init", "local"):
                    kind = stack[-1]
                elif prev in (")", "else", "do", ";", "}") or (prev == "{" and stack[-1:] == ["body"]):
                    kind = "body"
                elif any(w in head for w in ("struct", "union")):
                    kind = "record"
                else:
                    kind = "body"
                stack.append(kind)
        return out

    def name_before(self, at: int) -> str:
        """The last identifier of the operand that ends just before token `at` (member chains and subscripts)."""
        j = at - 1
        while j >= 0:
            tok = self.text[j]
            if tok in (")", "]") and j in self.partner:
                j = self.partner[j] - 1
            elif _IDENT.match(tok):
                return tok
            else:
                return ""
        return ""

    def name_after(self, at: int) -> str:
        for j in range(at + 1, min(at + 6, len(self.text))):
            if _IDENT.match(self.text[j]) and self.text[j] not in _TYPES:
                return self.text[j]
            if self.text[j] not in ("(", "*", "&", "-", "+", "!", "~"):
                return ""
        return ""

    def is_cast(self, close: int) -> str | None:
        """None, or "ptr"/"float"/"int" when the `)` at `close` ends a cast `(type)` that precedes an operand."""
        if self.text[close] != ")" or close not in self.partner:
            return None
        start = self.partner[close]
        inner = self.text[start + 1 : close]
        after = self.text[close + 1] if close + 1 < len(self.text) else ""
        if not inner or not (_IDENT.match(after) or _HEX.match(after) or after[:1].isdigit() or after == "("):
            return None
        words = [w for w in inner if w != "*"]
        if not words or not all(_IDENT.match(w) for w in words):
            return None
        before = self.text[start - 1] if start else ""
        if _IDENT.match(before) and before not in _KEYWORDS:
            return None  # a call or a declaration's parameter list
        if words[0] not in _TYPES and not (len(words) == 1 and (words[0][0].isupper() or words[0].endswith("_t"))):
            return None
        if "*" in inner:
            return "ptr"
        return "float" if any(w in ("float", "double", "f32", "f64") for w in words) else "int"

    def unary(self, at: int) -> bool:
        """Whether the `+`/`-`/`*`/`&` at `at` has no left operand."""
        prev = self.text[at - 1] if at else ";"
        return prev in _PRECEDENCE or prev in ("(", ",", "[", "{", "?", ":", ";", "}", "return", "case", "!", "~")

    def context(self, at: int) -> tuple[str, str] | None:
        """(context, detail) of the hex literal at token `at`, None when it is data (initializer, enum)."""
        kinds = self.braces[at]
        if "enum" in kinds or (kinds and kinds[-1] == "init"):
            return None
        lo = hi = at
        castkind = ""
        while True:
            left, right = lo - 1, hi + 1
            prev = self.text[left] if left >= 0 else ";"
            nxt = self.text[right] if right < len(self.text) else ";"
            if prev == "[" and nxt in ("]", "-", "+", "/", "*") and self._subscript(left):
                return self._size_or_index(left)
            if prev == "(" and nxt == ")":
                before = self.text[left - 1] if left else ";"
                if _IDENT.match(before) and before not in ("if", "while", "for", "switch", "sizeof", "return"):
                    return "CALL", f"{before}:{self._argument(left, at)}"
                if before in ("if", "while", "for", "switch"):
                    return "COND", before
                lo, hi = left, right
                continue
            if prev == "~":
                return "BITOP", "~"
            if prev in ("+", "-") and self.unary(left):
                lo = left
                continue
            if prev == "!":
                return "UNARY", "!"
            if prev in ("*", "&") and self.unary(left):
                return "UNARY", prev
            kind = self.is_cast(left)
            if kind:
                castkind = castkind or kind
                lo = self.partner[left]
                continue
            op = self._binding(prev, nxt, left)
            if op is not None:
                return self._operator(op, left, right, castkind)
            return self._statement(left, at, prev, castkind)

    def _binding(self, prev: str, nxt: str, left: int) -> tuple[str, bool] | None:
        """(operator, on the left of the literal) of the binary operator the literal is the operand of."""
        before = prev if prev in _PRECEDENCE and not self.unary(left) else None
        after = nxt if nxt in _PRECEDENCE else None
        if before and (not after or _PRECEDENCE[before] >= _PRECEDENCE[after]):
            return before, True
        if after and (not before or _PRECEDENCE[after] > _PRECEDENCE[before]):
            return after, False
        return None

    def _operator(self, found: tuple[str, bool], left: int, right: int, castkind: str) -> tuple[str, str]:
        op, on_left = found
        if op in _BITOPS:
            return "BITOP", op
        if op in _SHIFTS:
            return ("SHIFTCOUNT", op) if on_left else ("BITOP", op)
        other = self.name_before(left) if on_left else self.name_after(right - 1)
        if op in _COMPARE:
            if self._bit_expression(left, right, on_left):
                return "BITOP", "cmp"
            return "COMPARE", f"{op} {self.name_before(left) if not on_left else self.name_after(right - 1)}"
        if op in ("&&", "||"):
            return "LOGIC", op
        if op == "=":
            if on_left:
                return "ASSIGN", self.name_before(left)
            return "OTHER", "lhs="
        if op in _ARITH:
            if op in ("+", "-", "+=", "-=") and (castkind == "ptr" or re.search(r"(?i)addr|base|ptr", other)):
                return "STRUCT", f"ptr{op}"
            return "ARITH", op
        return "OTHER", f"binop{op}"

    def _bit_expression(self, left: int, right: int, on_left: bool) -> bool:
        """`(y & 0xFF) == x`, `x == (y >> 8)`: the other side of a compare is a parenthesized bit operation."""
        if on_left:
            close = left - 1
            if close < 0 or self.text[close] != ")" or close not in self.partner:
                return False
            span = range(self.partner[close] + 1, close)
        else:
            if right + 1 >= len(self.text) or self.text[right + 1] != "(" or right + 1 not in self.partner:
                return False
            span = range(right + 2, self.partner[right + 1])
        depth = 0
        for j in span:
            tok = self.text[j]
            depth += (tok in ("(", "[")) - (tok in (")", "]"))
            if depth == 0 and (tok in _BITOPS or tok in ("<<", ">>")) and not self.unary(j):
                return True
        return False

    def _statement(self, left: int, at: int, prev: str, castkind: str) -> tuple[str, str]:
        if prev == "case":
            return "CASE", ""
        if prev == "return":
            return "RETURN", ""
        if prev in ("(", ","):
            depth = 0
            for j in range(left, -1, -1):
                tok = self.text[j]
                if tok in (")", "]", "}"):
                    depth += 1
                elif tok in ("(", "[", "{"):
                    if depth == 0:
                        before = self.text[j - 1] if j else ";"
                        if tok == "(" and _IDENT.match(before) and before not in _KEYWORDS:
                            return "CALL", f"{before}:{self._argument(j, at)}"
                        if tok == "{":
                            return "INIT", self.name_before(j - 1)
                        return "OTHER", "paren"
                    depth -= 1
        if prev in ("?", ":"):
            start = self._statement_start(at)
            for j in range(at - 1, start - 1, -1):
                if self.text[j] == "=" and self.text[j - 1 : j] and self.text[j - 1] not in _PRECEDENCE:
                    return "ASSIGN", self.name_before(j)
                if self.text[j] == "return":
                    return "RETURN", ""
        if prev == "{":
            return "INIT", self.name_before(left - 1)
        return "OTHER", prev

    def _argument(self, opener: int, at: int) -> int:
        count, depth = 0, 0
        for j in range(opener + 1, at):
            if self.text[j] in ("(", "[", "{"):
                depth += 1
            elif self.text[j] in (")", "]", "}"):
                depth -= 1
            elif self.text[j] == "," and depth == 0:
                count += 1
        return count + 1

    def _subscript(self, opener: int) -> bool:
        return opener in self.partner

    def _size_or_index(self, opener: int) -> tuple[str, str]:
        """ARRAYSIZE or PADSIZE when the `[` at `opener` follows a declared name (`u8 pad[0x20]`), INDEX otherwise."""
        j = opener - 1
        while j >= 0 and self.text[j] == "]" and j in self.partner:
            j = self.partner[j] - 1
        if j < 1 or not _IDENT.match(self.text[j]) or self.text[j] in _KEYWORDS:
            return "INDEX", ""
        before = self.text[j - 1]
        if not (before == "*" or (_IDENT.match(before) and before not in ("return", "case", "else", "sizeof", "goto"))):
            return "INDEX", ""
        close = self.partner[opener]
        name = self.text[j]
        sized = "-" in self.text[opener + 1 : close] or "sizeof" in self.text[opener + 1 : close]
        return ("PADSIZE", name) if _PAD_NAME.match(name) else ("ARRAYSIZE", name + (" -" if sized else ""))


def _line_starts(code: str) -> list[int]:
    return [0] + [m.end() for m in re.finditer("\n", code)]


def hex_values(code: str, settings: Json, verdict: str) -> dict[int, str]:
    """{line: subject} of each hex literal in a value context whose table verdict is `verdict` (DEC: a mechanical
    decimal rewrite, REVIEW: a class a person reads); masks, bit operands, GBI arguments, sentinels, byte-structured and
    hardware values are KEEP. Initializer data is raw-data's, enum values name their constant."""
    from bisect import bisect_right

    hexed = settings["hex"]
    found: dict[int, str] = {}
    flag = _flag(settings, "flag_names")
    scan = _Scan(code)
    for at, (text, offset) in enumerate(scan.t):
        if not _HEX.match(text):
            continue
        context = scan.context(at)
        if context is None:
            continue
        name, action = _rule(*context, _integer(text), hexed, flag)
        if action == verdict:
            found.setdefault(bisect_right(scan.line_at, offset), f"{text} ({name})")
    return found


# ---------------------------------------------------------------- raw-data

_ARRAY = re.compile(
    r"(?P<head>(?:\b(?:static|const|extern|volatile)\s+)*)"
    r"\b(?P<type>(?:unsigned\s+|signed\s+)?(?:char|short|int|long|[us](?:8|16|32|64)))\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*(?:\[[^\]\n]*\]\s*)+=\s*\{"
)
_WIDTH = {"char": 1, "short": 2, "int": 4, "long": 4, "8": 1, "16": 2, "32": 4, "64": 8}
_CODE_OPS = frozenset(
    (
        *(
            0x09,
            0x0F,
            0x23,
            0x2B,
            0x0D,
            0x0C,
            0x03,
            0x02,
            0x04,
            0x05,
            0x24,
            0x25,
            0x28,
            0x29,
            0x31,
            0x39,
            0x0A,
            0x0B,
            0x08,
        ),
        *(0x0E, 0x20, 0x21, 0x2A, 0x2E, 0x06, 0x07, 0x01, 0x10, 0x11, 0x14, 0x15, 0x16, 0x17, 0x1A, 0x1B, 0x35, 0x3D),
    )
)
_CODE_FUNCTS = frozenset(
    (
        *(0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x2A, 0x2B, 0x00, 0x02, 0x03, 0x04, 0x06, 0x07, 0x08, 0x09),
        *(0x10, 0x12, 0x18, 0x19, 0x1A, 0x1B),
    )
)
_SYMBOL_ELEMENT = re.compile(r"^\(\s*(?:u32|unsigned\s+int|void\s*\*)\s*\)\s*&?\s*[A-Za-z_]\w*$|^&\s*[A-Za-z_]\w*$")


def _width(kind: str) -> int:
    """Bytes of one element of a raw scalar type: char 1, short 2, int and long 4, u8..u64 by their bits."""
    word = kind.split()[-1]
    return _WIDTH[word] if word in _WIDTH else _WIDTH[re.sub(r"\D", "", word)]


def _elements(body: str) -> tuple[list[int | None], bool] | None:
    """The values of a flat initializer list (None for a symbol address) and whether any element is a symbol, None
    when an element is neither a number nor a symbol reference (a nested record, an expression)."""
    values: list[int | None] = []
    symbols = False
    for element in (e.strip() for e in body.split(",")):
        if not element:
            continue
        if _SYMBOL_ELEMENT.match(element):
            values.append(None)
            symbols = True
            continue
        number = re.fullmatch(r"(-?)\s*(0[xX][0-9A-Fa-f]+|\d+)[uUlL]*", element)
        if not number:
            return None
        values.append(-int(number[2], 0) if number[1] else int(number[2], 0))
    return values, symbols


def _raw_bytes(values: Sequence[int | None], width: int) -> bytes | None:
    if any(v is None for v in values):
        return None
    size = 8 * width
    return b"".join((v & ((1 << size) - 1)).to_bytes(width, "big") for v in values if v is not None)


def _flt(word: int) -> float:
    return float(struct.unpack(">f", struct.pack(">I", word))[0])


def _short_digits(word: int) -> int:
    value = _flt(word)
    for digits in range(1, 10):
        if struct.pack(">f", float(f"{value:.{digits}g}")) == struct.pack(">f", value):
            return digits
    return 9


def _floats(words: Sequence[int], raw: Json) -> bool:
    """Every word 0 or a float of authored magnitude: at least half of the nonzero words have a short decimal form,
    or the array is long (DATA-TYPED-B's floats rule)."""
    nonzero = [w for w in words if w not in (0, 0x80000000)]
    if len(nonzero) < raw["float_min_words"]:
        return False
    low, high = raw["float_range"]
    if not all(low <= abs(_flt(w)) <= high for w in nonzero):
        return False
    short = sum(1 for w in nonzero if _short_digits(w) <= raw["float_digits"])
    return short * 2 >= len(nonzero) or len(words) >= raw["float_long"]


def _is_code(words: Sequence[int], raw: Json) -> bool:
    """MIPS code: canonical opcodes dominate the nonzero words and one is `jr ra` (DATA-TYPED-B's codeness rule)."""
    nonzero = [w for w in words if w]
    if len(nonzero) < raw["code_min_words"]:
        return False

    def canonical(word: int) -> bool:
        if word == 0x03E00008 or word >> 16 == 0x27BD:
            return True
        op = word >> 26
        return (word & 0x3F) in _CODE_FUNCTS if op == 0 else op in _CODE_OPS

    good = sum(1 for w in nonzero if canonical(w))
    return good / len(nonzero) > raw["code_share"] and any(w == 0x03E00008 for w in nonzero)


def _is_text(data: bytes, raw: Json) -> bool:
    """Printable bytes ending in NUL (one string, or strings joined by NULs)."""
    if len(data) < raw["string_min_bytes"] or data[-1] != 0:
        return False
    body = data.rstrip(b"\0")
    return len(body) >= raw["string_min_chars"] and all(b == 0 or 0x20 <= b < 0x7F or b in (9, 10, 13) for b in data)


def _is_pointer_table(values: Sequence[int | None], raw: Json) -> bool:
    low, high = raw["address"]
    nonzero = [v for v in values if v != 0]
    if len(values) < raw["pointer_min_words"] or not nonzero:
        return False
    return all(v is None or low <= v & 0xFFFFFFFF <= high for v in nonzero)


def raw_data(code: str, settings: Json, certain: bool) -> dict[int, str]:
    """{line: subject} of each word/byte array whose initializer is really another type: a string, a float table, a
    pointer table or MIPS code. `certain` picks the proven kinds (a long string, a table of symbol addresses, code with
    its `jr ra`); the rest (short text, floats, bare addresses) are a reader's judgment. A proven pad class (a short
    array named for its proven cause) stays allowed."""
    raw = settings["rawdata"]
    allowed = _compiled(raw["allowed_names"])
    found: dict[int, str] = {}
    for m in _ARRAY.finditer(code):
        scan_from = m.end() - 1
        depth, end = 0, scan_from
        for end in range(scan_from, len(code)):
            depth += (code[end] == "{") - (code[end] == "}")
            if depth == 0:
                break
        parsed = _elements(code[scan_from + 1 : end])
        if parsed is None:
            continue
        values, symbols = parsed
        width = _width(m["type"])
        if len(values) * width <= raw["pad_max_bytes"] and allowed.search(m["name"]):
            continue
        kind, proven = "", False
        concrete = [v for v in values if v is not None]
        data = _raw_bytes(values, width)
        if data is not None and any(data) and _is_text(data, raw):
            kind, proven = "a string", len(data.rstrip(b"\0")) >= raw["string_certain_chars"]
        elif width == 4 and _is_pointer_table(values, raw) and (symbols or len(concrete) == len(values)):
            kind, proven = "a pointer table", bool(symbols)
        elif width == 4 and data is not None and not symbols:
            words = [v & 0xFFFFFFFF for v in values if v is not None]
            if _is_code(words, raw):
                kind, proven = "MIPS code", True
            elif _floats(words, raw):
                kind = "a float table"
        if kind and proven == certain:
            found[code.count("\n", 0, m.start("name")) + 1] = f"{m['name']} is {kind}"
    return found


# ---------------------------------------------------------------- placeholder-name


def _owner(code: str, binding: symbol_names.Binding) -> str:
    """The record a member belongs to: its tag, else the typedef or variable name after the closing brace."""
    if binding.kind != "member" or binding.owner or not binding.scope:
        return binding.owner
    depth = 0
    for m in re.finditer(r"[{}]", code[binding.scope[-1] :]):
        depth += 1 if m[0] == "{" else -1
        if not depth:
            tail = re.match(r"\s*(\w+)", code[binding.scope[-1] + m.end() :])
            return tail[1] if tail else ""
    return ""


def placeholder_names(code: str, settings: Json) -> dict[int, str]:
    """{line: subject} of each identity declared with an address or placeholder name (func_8XXXXXXX, D_8XXXXXXX,
    *_padding_<addr>, unk_*, sub_*, a numbered *Block17 / *Words1). An identity counts once, at its first
    definition: a reference, an extern or prototype declaration, a size assertion over it and a version-conditional copy
    of its record are the same name, not a new one."""
    names = settings["placeholder"]
    patterns = [_compiled(p) for p in names["patterns"]]
    numbered = _compiled(rf"\b\w*?(?:{'|'.join(names['numbered_suffixes'])})\d+\b")
    found: dict[int, str] = {}
    seen: set[tuple[str, str, str]] = set()
    for b in symbol_names.bindings(code):
        if not b.definition or not (any(p.fullmatch(b.name) for p in patterns) or numbered.fullmatch(b.name)):
            continue
        identity = (b.name, b.kind, _owner(code, b))
        if identity not in seen:
            seen.add(identity)
            found.setdefault(code.count("\n", 0, b.start) + 1, b.name)
    return found


def placeholder_path(path: str, settings: Json) -> str | None:
    """The first component of a project path (file or folder) named by address or placeholder, None for none."""
    names = settings["placeholder"]
    patterns = [_compiled(p) for p in names["patterns"]]
    for part in path.split("/"):
        stem = part.rsplit(".", 1)[0]
        if any(p.fullmatch(stem) for p in patterns):
            return part
    return None


# ---------------------------------------------------------------- guard-gap

_COND = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
_VERSION = re.compile(r"\bVERSION_\w+")
_FUNCTION = re.compile(
    r"\b(?!(?:if|for|while|switch|return|sizeof)\b)[A-Za-z_]\w*\s*\([^;{}()]*(?:\([^()]*\)[^;{}()]*)*\)\s*\{"
)


def _branches(text: str) -> list[tuple[frozenset[str], bool, int, int]]:
    """(versions, negated, first line, last line) of each branch of every `#if` whose condition names a VERSION_ macro.
    An `#else` is the opposite polarity of its `#if`, an `#elif` a new condition."""
    found: list[tuple[frozenset[str], bool, int, int]] = []
    stack: list[list[Any]] = []  # [versions, negated, first line] of the open branch, or None for a plain `#if`
    for number, line in enumerate(text.splitlines(), 1):
        m = _COND.match(line)
        if not m:
            continue
        word, rest = m[1], m[2]
        if word in ("if", "ifdef", "ifndef"):
            versions = frozenset(_VERSION.findall(rest))
            negated = word == "ifndef" or bool(re.match(r"\s*!", rest))
            stack.append([versions, negated, number + 1] if versions else [])
            continue
        if not stack:
            continue
        top = stack[-1]
        if top:
            found.append((top[0], top[1], top[2], number - 1))
        if word == "endif":
            stack.pop()
        elif top:
            if word == "else":
                stack[-1] = [top[0], not top[1], number + 1]
            else:
                versions = frozenset(_VERSION.findall(rest))
                stack[-1] = [versions, bool(re.match(r"\s*!", rest)), number + 1] if versions else []
    return found


def guard_gaps(text: str, code: str, settings: Json) -> dict[int, str]:
    """{line: subject} of each raw word array in one version's branch beside a function written in C for the opposite
    branch of the same version guard (DATA-TYPED-B: a negated guard next to a per-version array)."""
    lines = code.splitlines()
    branches = _branches(text)
    found: dict[int, str] = {}
    for versions, negated, first, last in branches:
        body = "\n".join(lines[first - 1 : last])
        arrays = [(m, _elements(body[m.end() : _close(body, m.end() - 1)])) for m in _ARRAY.finditer(body)]
        words = [(m, p) for m, p in arrays if p is not None and _width(m["type"]) == 4 and len(p[0]) >= 4]
        if not words:
            continue
        for other_versions, other_negated, other_first, other_last in branches:
            if other_versions == versions and other_negated != negated:
                other = "\n".join(lines[other_first - 1 : other_last])
                function = _FUNCTION.search(other)
                if function:
                    name = re.match(r"\w+", function[0])
                    for m, _ in words:
                        at = first + body.count("\n", 0, m.start("name"))
                        found.setdefault(at, f"{m['name']} is raw words where {name[0] if name else 'a function'} is C")
    return found


def _close(body: str, start: int) -> int:
    depth = 0
    for at in range(start, len(body)):
        depth += (body[at] == "{") - (body[at] == "}")
        if depth == 0:
            return at
    return len(body)


# ---------------------------------------------------------------- sdk-name

_DEFINITION = re.compile(
    r"(?m)^(?!\s*(?:typedef|return|else|#))[A-Za-z_][\w \t\*]*?\b([A-Za-z_]\w*)"
    r"\s*(?:\([^;{}]*\)\s*\{|(?:\[[^\]\n]*\]\s*)*=)"
)


@cache
def _sdk_catalog() -> frozenset[str]:
    import json
    from importlib import resources

    node = resources.files("unbake.resources") / "data" / "sdk" / "catalog.json"
    return frozenset(s["name"] for s in json.loads(node.read_text(encoding="utf-8"))["signatures"])


def sdk_names(code: str, settings: Json, library: bool) -> dict[int, str]:
    """{line: subject} of each name a file defines that does not belong there: under a library root a name the SDK
    catalog does not list and no SGI prefix spells; outside one a definition spelling an SGI-reserved name."""
    names = settings["sdk"]
    reserved, prefixes = _compiled(names["reserved"]), _compiled(names["prefixes"])
    found: dict[int, str] = {}
    for m in _DEFINITION.finditer(code):
        name = m[1]
        if name in _KEYWORDS:
            continue
        if library:
            if name not in _sdk_catalog() and not prefixes.match(name):
                found.setdefault(code.count("\n", 0, m.start(1)) + 1, f"{name} is no SDK name")
        elif reserved.match(name):
            found.setdefault(code.count("\n", 0, m.start(1)) + 1, f"{name} is an SDK-reserved name")
    return found
