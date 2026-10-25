"""Which units a header edit cannot reach: an includer is rebuilt only when the edit can change its compile key.

`decomp data check` used to preprocess every includer of a touched header (3400 units over five versions for a header
many units include: 171 CPU s) and compile those whose key moved. The build's compile key (`hdrkey`, unbake's Makefile)
is the preprocessed unit with every top-level declaration the unit never names, transitively, left out. So an edit
that only changes droppable declarations (extern variables and prototypes, typedefs, struct/union/enum definitions)
changes a unit's key only if the unit's key names one of the names those declarations define:

  key(U) is the closure, from the unit's definitions, over the names the kept items spell. An item the key does not
  define-and-name stays out of the closure. If no name the changed items define (before or after the edit) occurs in the
  old key of U, then no kept item defined a changed item and no kept item names a changed item's definition, so the
  closure over the new text is the same set of items with the same text: the key is equal and the object is too.

The proof holds only when (1) the header's changed region holds nothing but droppable declarations, parsed here from
the raw text; every preprocessor line (a conditional, a define, an include) is an item of its own that must not change,
and the items of every branch are read together (a unit sees one branch: the names of all of them are a superset);
(2) no name involved is a
macro anywhere (a macro could rename the declaration inside the unit); (3) the unit's old key is the key of the
state the header diff starts from: the unit's `.i.ok` is not newer than the pool's build marker (written when the
pool's build ends), so no build after the pool minted the tree has touched it.
Anything else keeps the unit (never a skip on doubt).

Lives in unbake because the key is unbake's (`hdrkey`) and `buildproof` uses it to build only the units a header
edit can reach (a widely included header: 3400 units over five versions, 527 CPU s);
`decomp data check` shares the parser."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unbake import process
from unbake.contracts import Config, Refusal

KEYWORDS = frozenset(
    [
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "int",
        "long",
        "register",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
        "inline",
        "__inline",
        "__inline__",
        "__volatile__",
        "__const",
        "__const__",
        "__signed__",
        "asm",
        "__asm",
        "__asm__",
        "__attribute__",
        "__attribute",
        "__extension__",
        "__typeof__",
        "typeof",
        "restrict",
        "__restrict",
        "_Bool",
    ]
)
_TOKEN = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|[A-Za-z_]\w*|\d\w*|\.\.\.|->|\S')
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_IDENT = re.compile(r"[A-Za-z_]\w*$")


@dataclass
class Parsed:
    items: list[tuple[str, ...]] = field(default_factory=list)
    why: str = ""  # non-empty: the text is outside what this parser proves anything about


def _blank(m: re.Match[str]) -> str:
    return re.sub(r"[^\n]", " ", m.group(0))


def parse(text: str) -> Parsed:
    """The includes, preprocessor lines and top-level items of a header's raw text. `why` names the first thing it does
    not handle. A directive is an item that must not change (`__INC__` for an include, `__PP__` for any other: the
    include guard, a version conditional, a define); a directive inside an item stays one of its tokens."""
    text = _COMMENT.sub(_blank, text.replace("\\\n", "  "))
    out = Parsed()
    code: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s.startswith("#"):
            code.append(line)
            continue
        m = re.match(r'#\s*include\s+(["<][^">]+[">])\s*$', s)
        if m:  # an include is an item in its place: moving an item across it, or changing it, is not analysable
            code.append('__INC__ "' + m.group(1).strip('"<>') + '" ;')
            continue
        words = " ".join(s[1:].split()).replace("\\", "/").replace('"', "'")
        code.append(f'__PP__ "{words}" ;')
    toks = _TOKEN.findall("\n".join(code))
    item: list[str] = []
    depth = {"(": 0, "[": 0, "{": 0}
    pair = {")": "(", "]": "[", "}": "{"}
    for t in toks:
        if t in depth:
            depth[t] += 1
        elif t in pair:
            depth[pair[t]] -= 1
            if depth[pair[t]] < 0:
                out.why = "unbalanced brackets"
                return out
        item.append(t)
        if t == ";" and not any(depth.values()):
            if item[:-1]:
                out.items.append(tuple(item[:-1]))
            item = []
        elif t == "}" and not any(depth.values()) and item[0] not in ("extern", "typedef", "struct", "union", "enum"):
            out.items.append(tuple(item))  # a function body or an initialiser: ends at its brace
            item = []
    if item or any(depth.values()):
        out.why = "text after the last item"
    return out


def _idents(tokens) -> set[str]:
    return {t for t in tokens if _IDENT.match(t) and t not in KEYWORDS}


def _depth0(t: tuple[str, ...], stop: str):
    d = 0
    for i, x in enumerate(t):
        if x in "([{":
            if x == stop and d == 0:
                return i
            d += 1
        elif x in ")]}":
            d -= 1
        elif x == stop and d == 0:
            return i
    return -1


def defs(item: tuple[str, ...]) -> set[str] | None:
    """The names a droppable top-level item defines (a superset is always safe); None when the item is not droppable by
    hdrkey's rule or not understood (a definition, an `extern "C"`, a struct with declarators)."""
    if not item:
        return None
    head = item[0]
    if head == "extern":
        if len(item) < 2 or item[1].startswith('"') or "{" in item:
            return None
        if _depth0(item, ",") >= 0:
            return _idents(item)
        p = _depth0(item, "(")
        if p > 0:
            return {item[p - 1]} if _IDENT.match(item[p - 1]) and item[p - 1] not in KEYWORDS else _idents(item)
        cut = min((i for i in (_depth0(item, "["), _depth0(item, "=")) if i >= 0), default=len(item))
        name = item[cut - 1]
        return {name} if _IDENT.match(name) and name not in KEYWORDS else _idents(item)
    body = _depth0(item, "{")
    if head == "typedef":
        if body >= 0:
            close = _match(item, body)
            tag = item[body - 1] if body >= 2 and item[body - 1] not in KEYWORDS else ""
            rest = item[close + 1 :]
            if "(" in rest or close < 0:
                return _idents(item)
            names = _idents(rest) | ({tag} if tag else set())
            return names | (_idents(item[body + 1 : close]) if item[1] == "enum" else set())
        if "(" in item or _depth0(item, ",") >= 0:
            return _idents(item)
        cut = _depth0(item, "[")
        name = item[(cut if cut >= 0 else len(item)) - 1]
        return {name} if _IDENT.match(name) and name not in KEYWORDS else _idents(item)
    if head in ("struct", "union", "enum"):
        if body < 0:
            return {item[1]} if len(item) == 2 and _IDENT.match(item[1]) else None
        close = _match(item, body)
        if close < 0 or close != len(item) - 1:
            return None  # declarators after the body: a variable definition, kept always
        tag = item[1] if body == 2 and item[1] not in KEYWORDS else ""
        if head != "enum" and not tag:
            return None
        return ({tag} if tag else set()) | (_idents(item[body + 1 : close]) if head == "enum" else set())
    return None


def _match(item: tuple[str, ...], open_at: int) -> int:
    d = 0
    for i in range(open_at, len(item)):
        d += item[i] == "{"
        d -= item[i] == "}"
        if d == 0:
            return i
    return -1


def changed_defs(old: str | None, new: str | None) -> tuple[set[str] | None, str]:
    """(names defined by the items the edit removed or added, "") or (None, why the edit is not analysable)."""
    a, b = parse(old or ""), parse(new or "")
    if a.why or b.why:
        return None, a.why or b.why
    names: set[str] = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a.items, b=b.items, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        for it in (*a.items[i1:i2], *b.items[j1:j2]):
            d = defs(it)
            if d is None:
                return None, f"changed item is not a droppable declaration: {' '.join(it[:6])}"
            names |= d
    return names, ""


BUILT_MARKER = "build/.pool-built-at"  # content: the commit; mtime: the end of the build that made every unit's .i.ok
EXTRACTED_MARKER = "build/.pool-extracted-at"  # content: the commit; mtime: the extraction, before that build
_DEFINE = re.compile(rb"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", re.M)
_SOURCE_SUFFIX = (".h", ".c", ".cpp", ".s")


def macro_hits(root: Path, names: set[str], flags: Sequence[str] = ()) -> set[str]:
    """NAMES that are `#define`d (or -D) anywhere in the tracked files under include/ and src/: a macro could rename the
    declaration inside a unit."""
    hit = {f[2:].split("=")[0] for f in flags if f.startswith("-D")} & names
    if not names:
        return hit
    for path in sorted(process.tracked(root)):
        if not path.startswith(("include/", "src/")) or not path.lower().endswith(_SOURCE_SUFFIX):
            continue
        try:
            body = (root / path).read_bytes()
        except OSError:
            continue
        hit |= {m.decode(errors="replace") for m in _DEFINE.findall(body)} & names
    return hit


def unit_key(root: Path, obj_i: Path, hdrkey: Path) -> str | None:
    """The compile key of a unit's preprocessed `.i`: the `.i.key` make keeps when it is newer than the `.i`, else
    hdrkey's answer (left there as make would). None when no key can be had."""
    key = obj_i.parent / (obj_i.name + ".key")
    try:
        if key.is_file() and key.stat().st_mtime >= obj_i.stat().st_mtime:
            return key.read_text(errors="replace")
        if not hdrkey.is_file():
            return None
        done = process.run("hdrkey", [hdrkey, obj_i], root, tmp=process.scratch(root))
        if done.exit != 0 or not done.stdout:
            return None
        text = done.stdout.decode(errors="replace")
        tmp = key.with_name(key.name + ".new")
        tmp.write_text(text)
        tmp.replace(key)
        return text
    except (OSError, Refusal):
        return None


def narrow(
    config: Config,
    version: str,
    headers: Sequence[str],
    units: Sequence[str],
    object_fmt: str,
    new_text: Callable[[str], str | None],
    flags: Sequence[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    """`narrow_tree` of the project's tree."""
    return narrow_tree(config.project.root, version, headers, units, object_fmt, new_text, flags)


def narrow_tree(
    root: Path,
    version: str,
    headers: Sequence[str],
    units: Sequence[str],
    object_fmt: str,
    new_text: Callable[[str], str | None] | None = None,
    flags: Sequence[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    """(UNITS to keep, stats): drops the units that no edit of HEADERS can reach (module doc). The old text of a header
    is its text at the revision the tree was built at (`build/.pool-built-at`, written when the pool's build ends; else
    the extraction marker `build/.pool-extracted-at`), which must be the tree's own content;
    NEW_TEXT(header) is its text after the change (None: deleted; default: the work tree's file). `object_fmt` is the
    object path template. The one narrowing of the build (`buildproof`) and of `decomp data check`."""
    root = Path(root)
    if new_text is None:

        def new_text(h: str) -> str | None:
            try:
                return (root / h).read_text(errors="replace")
            except OSError:
                return None

    stats: dict[str, Any] = {"headers": len(headers), "includers": len(units), "skipped": 0, "why": ""}
    for name in (
        BUILT_MARKER,
        EXTRACTED_MARKER,
    ):  # a build after the extraction touches the units' .i.ok: the later marker
        try:
            base = (root / name).read_text().strip()
            since = (root / name).stat().st_mtime
            break
        except OSError:
            continue
    else:
        stats["why"] = "no pool marker: tree not minted from a pool"
        return list(units), stats
    if not base or process.tree_id(root, base) != process.tree_id(root, "HEAD") or not process.tree_id(root, "HEAD"):
        stats["why"] = "the tree is not the one the unit keys were made at"
        return list(units), stats
    names: set[str] = set()
    for h in headers:
        if not h.lower().endswith(".h"):
            stats["why"] = f"{h} is not a header"
            return list(units), stats
        got, why = changed_defs(process.show(root, base, h), new_text(h))
        if got is None:
            stats["why"] = f"{h}: {why}"
            return list(units), stats
        names |= got
    mac = macro_hits(root, names, flags)
    if mac:
        stats["why"] = f"{sorted(mac)[0]} is a macro"
        return list(units), stats
    if not names:
        stats["why"] = "no declaration changed"
        stats["skipped"] = len(units)
        return [], stats
    pat = re.compile(r"(?<![\w$])(?:" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")(?![\w$])")
    hdrkey = root / "build/hdrkey"
    keep: list[str] = []
    for u in units:
        unit_i = root / (object_fmt.format(v=version, unit=u)[: -len(".o")] + ".i")
        try:
            pristine = (unit_i.parent / (unit_i.name + ".ok")).stat().st_mtime <= since
        except OSError:
            pristine = False
        key = unit_key(root, unit_i, hdrkey) if pristine else None
        if key is None or pat.search(key):
            keep.append(u)
        else:
            stats["skipped"] += 1
    return keep, stats


def _dependencies(dfile: Path) -> list[str] | None:
    """The files a make dependency file names after its targets: the reading of the Makefile's stamp sweep."""
    try:
        text = dfile.read_text(errors="replace")
    except OSError:
        return None
    names: list[str] = []
    for line in text.replace("\\\n", " ").splitlines():
        _, colon, named = line.partition(": ")
        if colon and named.strip():
            names += named.split()
    return names


def sign(root: Path, stamps: Sequence[str]) -> int:
    """Sign STAMPS (build/<v>/<unit>.i.ok, relative to ROOT) again under the bytes their inputs hold now: append to
    build/.inputs.log the record the Makefile's stamp record writes (path, the stamp's own mtime, the SHA-1 of the
    SHA-1s of the files its .d names). Only for units `narrow_tree` dropped: their preprocessed text and object stay
    valid under the new header bytes; without a fresh record the stamp sweep of the next make drops every one of them
    (same date, other input bytes) and the build remakes them all (10-25: a stamp's ROM check rebuilt the 2,000-4,000
    includers per version the data check had proved unreachable, 186-619 CPU s). The stamp keeps its date, so it stays
    pristine for the next narrowing. Returns the number of records written (a stamp without a .d gets none)."""
    root = Path(root)
    digests: dict[str, str] = {}

    def digest(name: str) -> str:
        if name not in digests:
            try:
                digests[name] = hashlib.sha1((root / name).read_bytes()).hexdigest()
            except OSError:
                digests[name] = "-"
        return digests[name]

    lines = []
    for stamp in stamps:
        deps = _dependencies(root / (stamp[: -len(".i.ok")] + ".d"))
        try:
            when = (root / stamp).stat().st_mtime_ns
        except OSError:
            continue
        if deps is None:
            continue
        key = hashlib.sha1(" ".join(digest(d) for d in deps).encode()).hexdigest()
        lines.append(json.dumps([stamp, when, key]) + "\n")
    if lines:
        fd = os.open(root / "build/.inputs.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, "".join(lines).encode())
        finally:
            os.close(fd)
    return len(lines)
