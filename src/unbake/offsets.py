"""Fields whose name states an offset (`f_A4`, `pad55`, `unk10`, `field_1C`) must sit at that offset. A name that says
one offset on a field the compiler places at another misleads every reader and, in a record that overlays original
data, shifts every later field. The layout is the one the C headers give under the target ABI (cheaders: o32 big-endian
natural alignment); a record whose layout cannot be sized stops being judged at the first field it cannot size."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from unbake import cheaders

# a prefix and the hex offset it carries: `f_A4`, `padB0`, `unk_10`, `field_1C`, `zero0C`, `flag1C`, `off20`. A bare
# suffix must be upper case or digits (`padded` is no offset); after an underscore either case.
_NAMED = re.compile(
    r"^(?:(?:f|pad|padding|unk|unknown|field)_(?:0[xX])?([0-9A-Fa-f]{1,4})"
    r"|(?:f|pad|padding|unk|unknown|field|zero|flag|off)([0-9A-F]{1,4}))$"
)
_PADDING = re.compile(r"^pad(?:ding)?_?[0-9A-Fa-f]")
_COUNTED = re.compile(r"^pad(?:ding)?_?(?:0[xX])?(?:0\d*|[0-9A-Fa-f])$")  # pad1, pad_002: a pad counted, not placed
_RECORD = re.compile(r"\b(typedef\s+)?(struct|union)\s*(\w*)\s*\{")


@dataclass(frozen=True)
class Site:
    line: int
    record: str
    field: str
    named: int
    actual: int

    @property
    def subject(self) -> str:
        return f"{self.record}.{self.field} is named for offset 0x{self.named:X} but sits at 0x{self.actual:X}"


def named_offset(name: str) -> int | None:
    """The offset a field name states, or None when it states none."""
    match = _NAMED.match(name)
    return int(match[1] or match[2], 16) if match else None


def registry(headers: Mapping[str, str]) -> cheaders.Headers:
    """The declarations of every header (path -> text), in path order."""
    found = cheaders.Headers()
    for path in sorted(headers):
        found.add(headers[path], path)
    return found


def _close(code: str, at: int) -> int:
    depth = 0
    for i in range(at, len(code)):
        depth += {"{": 1, "}": -1}.get(code[i], 0)
        if depth == 0:
            return i
    return len(code)


def _top(body: str) -> str:
    """The body with every nested brace pair blanked (newlines kept): what its own members say."""
    out, depth = [], 0
    for ch in body:
        depth += {"{": 1, "}": -1}.get(ch, 0)
        out.append(ch if (depth == 0 and ch != "}") or ch == "\n" else " ")
    return "".join(out)


def sites(base: cheaders.Headers, text: str, directives: str = "", path: str = "") -> list[Site]:
    """Every named-offset field of the file's records that sits elsewhere. `text` has its comments and strings blanked
    (policy.scopes "code") so lines stay, `directives` is the preprocessor lines alone; a record with a preprocessor
    line in it, a union, and a field out of order with the named ones before it are not judged."""
    if not _NAMED_ANY.search(text):
        return []
    local = base.copy()
    local.add(text, path)
    marked = directives.split("\n")
    found: list[Site] = []
    for head in _RECORD.finditer(text):
        if head[2] == "union":
            continue
        end = _close(text, head.end() - 1)
        alias = re.match(r"\s*\*?\s*(\w+)\s*[;,\[]", text[end + 1 :]) if head[1] else None
        name = (alias[1] if alias else head[3]) or ""
        first, last = text.count("\n", 0, head.end()), text.count("\n", 0, end)
        if not name or any(line.strip() for line in marked[first : last + 1]):
            continue
        struct = local.struct(name)
        if struct is None or struct.union:
            continue
        body = _top(text[head.end() : end])
        at, previous, before = 0, 0, -1
        for f in struct.fields:
            last, before = before, f.offset
            named = named_offset(f.name)
            hit = re.compile(rf"\b{re.escape(f.name)}\b").search(body, at)
            if hit is None:
                continue  # a field of an anonymous member: its names are the inner record's own
            at = hit.end()
            if (
                named is None
                or _COUNTED.match(f.name)
                or named < previous
                or (struct.size is not None and named >= struct.size)
            ):
                continue
            previous = named
            # a padding array is named for its own start, for the field it follows or for the offset it runs to
            if named != f.offset and not (_PADDING.match(f.name) and named in (last, f.offset + (f.type.size or 0))):
                line = text.count("\n", 0, head.end() + hit.start()) + 1
                found.append(Site(line, name, f.name, named, f.offset))
    return found


_NAMED_ANY = re.compile(r"\b(?:f|pad|padding|unk|unknown|field|zero|flag|off)_?[0-9A-Fa-f]{1,4}\b")
