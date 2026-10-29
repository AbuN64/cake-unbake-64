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

from unbake import config as configuration
from unbake import process
from unbake.contracts import Config, Finding, Refusal

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


_INCLUDE = re.compile(r'\s*#\s*include\s+"([^"]+)"\s*$')


def _guard(text: str) -> tuple[str, list[str]] | None:
    """(the include guard macro, the lines inside it) when ONE `#ifndef G` / `#define G` pair wraps the whole text."""
    lines = _COMMENT.sub(_blank, text.replace("\\\n", "  ")).split("\n")
    at = [i for i, x in enumerate(lines) if x.strip()]
    if len(at) < 3 or not re.match(r"\s*#\s*endif\b", lines[at[-1]]):
        return None
    a, b = re.match(r"\s*#\s*ifndef\s+(\w+)\s*$", lines[at[0]]), re.match(r"\s*#\s*define\s+(\w+)\s*$", lines[at[1]])
    if not (a and b and a.group(1) == b.group(1)):
        return None
    depth = 0
    for i in at:
        if re.match(r"\s*#\s*if", lines[i]):
            depth += 1
        elif re.match(r"\s*#\s*endif\b", lines[i]):
            depth -= 1
            if depth == 0 and i != at[-1]:
                return None
    return a.group(1), lines[at[1] + 1 : at[-1]]


def _inlined(
    root: Path, base: str, text: str | None, new_text: Callable[[str], str | None], flags: Sequence[str] = ()
) -> str | None:
    """TEXT with each `#include "X"` of a header the revision BASE lacks written out as the header's own lines: textual
    inclusion IS the lines in its place, so an edit that moves declarations into a new header leaves the preprocessed
    unit unchanged. Only when the new header is guarded whole by a macro no other file spells (so a unit sees it once,
    here) and each include it makes is one the text made earlier and is itself guarded (a repeated guarded include adds
    nothing). Any other include stays an `__INC__` item, which `changed_defs` refuses."""
    if text is None:
        return None
    dirs = [d[2:] for d in flags if d.startswith("-I") and len(d) > 2] or ["include"]

    def locate(name: str) -> tuple[str, str] | None:
        for d in dirs:
            got = new_text(f"{d}/{name}")
            if got is not None:
                return f"{d}/{name}", got
        return None

    out: list[str] = []
    earlier: set[str] = set()
    for line in text.split("\n"):
        m = _INCLUDE.match(line)
        found = locate(m.group(1)) if m else None
        if not m or found is None:
            out.append(line)
            if m:
                earlier.add(m.group(1))
            continue
        path, body = found
        wrapped = _guard(body)
        ok = wrapped is not None and process.show(root, base, path) is None
        if ok and wrapped is not None:
            guard, inner = wrapped
            spelled = process.spelling(root, guard)
            ok = not [x for x in spelled if x != path]
            for n in (_INCLUDE.match(x) for x in inner):
                sub = locate(n.group(1)) if n else None
                ok = ok and (n is None or (n.group(1) in earlier and sub is not None and _guard(sub[1]) is not None))
        if not ok or wrapped is None:
            out.append(line)
            earlier.add(m.group(1))
            continue
        out.extend(x for x in wrapped[1] if not _INCLUDE.match(x))
        earlier.add(m.group(1))
    return "\n".join(out)


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
        if not configuration.under_tree(path) or not path.lower().endswith(_SOURCE_SUFFIX):
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
    base: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """(UNITS to keep, stats): drops the units that no edit of HEADERS can reach (module doc). The old text of a header
    is its text at the revision the tree was built at (`build/.pool-built-at`, written when the pool's build ends; else
    the extraction marker `build/.pool-extracted-at`), which must be the tree's own content; with BASE, its text at
    BASE, and a unit counts as built there only when its stamp record says so (`_base_made`: the train's tree, whose
    chain is committed over BASE and whose build is the last round's);
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
    since = -1.0  # with BASE given, no date makes a unit pristine: only its record
    for name in () if base else (BUILT_MARKER, EXTRACTED_MARKER):
        # a build after the extraction touches the units' .i.ok: the later marker
        try:
            base = (root / name).read_text().strip()
            since = (root / name).stat().st_mtime
            break
        except OSError:
            continue
    else:
        if not base:
            stats["why"] = "no pool marker: tree not minted from a pool"
            return list(units), stats
    if since >= 0 and (
        not base or process.tree_id(root, base) != process.tree_id(root, "HEAD") or not process.tree_id(root, "HEAD")
    ):
        stats["why"] = "the tree is not the one the unit keys were made at"
        return list(units), stats
    names: set[str] = set()
    for h in headers:
        if not h.lower().endswith(".h"):
            stats["why"] = f"{h} is not a header"
            return list(units), stats
        before = process.show(root, base, h)
        if before is None:  # a header the base lacks: only a changed file includes it
            continue
        got, why = changed_defs(before, _inlined(root, base, new_text(h), new_text, flags))
        if got is None:
            stats["why"] = f"{h}: {why}"
            return list(units), stats
        names |= got
    mac = macro_hits(root, names, flags)
    if mac:
        stats["why"] = f"{sorted(mac)[0]} is a macro"
        return list(units), stats
    pat = (
        re.compile(r"(?<![\w$])(?:" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")(?![\w$])")
        if names
        else None
    )
    if not names:
        stats["why"] = "no declaration changed"
    hdrkey = root / "build/hdrkey"
    made_at_base = _base_made(root, base, headers)
    keep: list[str] = []
    for u in units:
        unit_i = root / (object_fmt.format(v=version, unit=u)[: -len(".o")] + ".i")
        stamp = unit_i.parent / (unit_i.name + ".ok")
        try:
            # a stamp dated 1 s is one the sweep (or `drop`) marked for a rebuild: its .i is no base text
            st = stamp.stat()
            dated = st.st_mtime_ns != DROPPED_NS and st.st_mtime <= since
            pristine = dated or made_at_base(str(stamp.relative_to(root)))
        except (OSError, ValueError):
            pristine = False
        if pristine and pat is None:  # nothing a key could name changed: the unit's object stands as built at BASE
            stats["skipped"] += 1
            continue
        key = unit_key(root, unit_i, hdrkey) if pristine else None
        if key is None or pat.search(key):
            keep.append(u)
        else:
            stats["skipped"] += 1
    return keep, stats


_RECORDS: dict[tuple[str, int, int], dict[str, list]] = {}


def records(root: Path) -> dict[str, list]:
    """The stamp sweep's recorded input keys, {output: [mtime_ns, key]}: build/.inputs.json with build/.inputs.log
    replayed over it (the last record of an output wins), as the sweep reads them. Cached while neither file moves."""
    root = Path(root)
    state_path, log_path = root / "build/.inputs.json", root / "build/.inputs.log"
    try:
        stat = state_path.stat()
        tag = (str(root), stat.st_mtime_ns, log_path.stat().st_size if log_path.exists() else -1)
    except OSError:
        tag = (str(root), -1, log_path.stat().st_size if log_path.exists() else -1)
    if tag in _RECORDS:
        return _RECORDS[tag]
    try:
        state = json.loads(state_path.read_text())
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError):
        state = {}
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            try:
                name, when, key = json.loads(line)
            except (ValueError, TypeError):
                continue
            state[name] = [when, key]
    except OSError:
        pass
    _RECORDS.clear()
    _RECORDS[tag] = state
    return state


def _base_made(root: Path, base: str, headers: Sequence[str]) -> Callable[[str], bool]:
    """A test of whether a unit's stamp (build/<v>/<unit>.i.ok) was made from the tree at BASE whatever its date: its
    record (the date it carries now and the key the Makefile wrote) equals the key of the files its .d names, read as
    BASE holds them (the HEADERS at BASE, every other file as it is). A unit preprocessed again after the pool's build
    marker from the same bytes (a no-op make, a refresh) still narrows."""
    state = records(root)
    digests: dict[str, str] = {}
    for h in headers:
        data = process.blob(root, base, h)
        digests[h] = hashlib.sha1(data).hexdigest() if data is not None else "-"

    digest = _digester(root, digests)

    def made(stamp: str) -> bool:
        rec = state.get(stamp)
        if not rec or len(rec) != 2:
            return False
        try:
            if (root / stamp).stat().st_mtime_ns != rec[0]:
                return False
        except OSError:
            return False
        deps = _dependencies(root / (stamp[: -len(".i.ok")] + ".d"))
        return deps is not None and hashlib.sha1(" ".join(digest(d) for d in deps).encode()).hexdigest() == rec[1]

    return made


def _digester(root: Path, known: dict[str, str] | None = None) -> Callable[[str], str]:
    """The SHA-1 of a file under ROOT ("-" when unreadable), each read once: what the stamp sweep hashes a .d's names by
    (KNOWN: digests given beforehand, as a header's at another revision)."""
    digests = {} if known is None else known

    def digest(name: str) -> str:
        if name not in digests:
            try:
                digests[name] = hashlib.sha1((root / name).read_bytes()).hexdigest()
            except OSError:
                digests[name] = "-"
        return digests[name]

    return digest


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


DROPPED_NS = 10**9  # the date the Makefile's stamp sweep gives a stamp made from other bytes


def drop(root: Path, stamps: Sequence[str]) -> int:
    """Date STAMPS (build/<v>/<unit>.i.ok) long ago, as the stamp sweep drops a stamp made from other bytes, so the next
    make preprocesses those units again whatever header it is told to skip by date (`make -o`): the includers a
    narrowing keeps. Returns how many were dated."""
    done = 0
    for stamp in stamps:
        try:
            os.utime(Path(root) / stamp, ns=(DROPPED_NS, DROPPED_NS))
            done += 1
        except OSError:
            continue
    return done


def includers(root: Path, version: str, headers: Sequence[str]) -> list[str]:
    """The units of VERSION whose dependency file (build/<version>/<unit>.d) names one of HEADERS."""
    found: set[str] = set()
    want = set(headers)
    for dfile in (Path(root) / "build" / version).rglob("*.d"):
        deps = _dependencies(dfile)
        if deps and want & set(deps):
            found.add(str(dfile.relative_to(Path(root) / "build" / version))[: -len(".d")])
    return sorted(found)


def narrow_command(params: dict[str, Any]) -> dict[str, Any]:
    """`unbake narrow --against REV [--version V] [--apply]`: for every header the tree changes over REV (committed or
    not), the includers whose object no edit can change, per version (`narrow_tree` with BASE=REV: only a stamp record
    proves a unit was built from REV's bytes). With --apply, and only when every version narrowed, their stamps are
    signed under the new bytes (`sign`) and every other includer is dated for a rebuild (`drop`): a make told to skip
    the headers by date (`make -o HEADER`) then rebuilds exactly the includers the edit can reach (the train's check
    rebuilt all 9,933 includers x versions of a reshaped wide header for ~925 real compiles, 10-25)."""
    root = Path(params.get("project") or ".")
    against = str(params["against"])
    changed = process.changed(root, against)
    headers = sorted(f for f in changed or () if f.lower().endswith(".h"))
    versions = list(params.get("version") or ()) or sorted(
        p.name for p in (root / "build").iterdir() if p.is_dir() and (root / "versions" / p.name).is_dir()
    )
    out: dict[str, Any] = {"against": against, "headers": headers, "versions": {}, "skip_headers": [], "applied": False}
    if not headers:
        out["why"] = "no header changed" if changed is not None else "git diff failed"
        return out
    plans: dict[str, tuple[list[str], list[str]]] = {}
    for v in versions:
        units = includers(root, v, headers)
        keep, stats = narrow_tree(root, v, headers, units, "build/{v}/{unit}.o", base=against)
        kept = set(keep)
        out["versions"][v] = {k: stats[k] for k in ("includers", "skipped", "why")}
        plans[v] = ([u for u in units if u not in kept], keep)
        if stats["why"] and stats["why"] != "no declaration changed":
            out["why"] = f"{v}: {stats['why']}"
            if params.get("apply"):  # asked to apply and could not: a refusal, not a success with applied false
                raise Refusal(Finding("narrow.unsafe", f"nothing applied: {out['why']}", versions=(v,)))
            return out
    if params.get("apply"):
        stamp = "build/{v}/{u}.i.ok"
        out["signed"] = sign(root, [stamp.format(v=v, u=u) for v, (skip, _) in plans.items() for u in skip])
        out["dropped"] = drop(root, [stamp.format(v=v, u=u) for v, (_, keep) in plans.items() for u in keep])
        out["applied"] = True
    out["skip_headers"] = headers
    return out


def sign(root: Path, stamps: Sequence[str]) -> int:
    """Sign STAMPS (build/<v>/<unit>.i.ok, relative to ROOT) again under the bytes their inputs hold now: append to
    build/.inputs.log the record the Makefile's stamp record writes (path, the stamp's own mtime, the SHA-1 of the
    SHA-1s of the files its .d names). Only for units `narrow_tree` dropped: their preprocessed text and object stay
    valid under the new header bytes; without a fresh record the stamp sweep of the next make drops every one of them
    (same date, other input bytes) and the build remakes them all (10-25: a stamp's ROM check rebuilt the 2,000-4,000
    includers per version the data check had proved unreachable, 186-619 CPU s). The stamp keeps its date, so it stays
    pristine for the next narrowing. Returns the number of records written (a stamp without a .d gets none)."""
    root = Path(root)
    digest = _digester(root)

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


def resign(root: Path, headers: Sequence[str], before: dict[str, list]) -> int:
    """After a tree's diff is reverted (a reused stamp slot reset): sign again each includer stamp of HEADERS whose
    record BEFORE the diff (`records`, read before it was applied) names the stamp's date now and the bytes its inputs
    hold again. No build touched such a stamp since (same date) and that record proves its text was made from these
    bytes; the record the diff's narrowing wrote names the diff's header bytes, so the next make's stamp sweep would
    drop every one (2,000-4,500 includers per version, A2 10-25). Returns the number of records written."""
    root = Path(root)
    versions = sorted(
        p.name for p in (root / "build").iterdir() if p.is_dir() and (root / "versions" / p.name).is_dir()
    )
    digest = _digester(root)

    stamps = []
    for v in versions:
        for u in includers(root, v, headers):
            stamp = f"build/{v}/{u}.i.ok"
            rec = before.get(stamp)
            deps = _dependencies(root / f"build/{v}/{u}.d")
            try:
                same = bool(rec) and len(rec) == 2 and (root / stamp).stat().st_mtime_ns == rec[0]
            except OSError:
                continue
            if (
                same
                and deps is not None
                and hashlib.sha1(" ".join(digest(d) for d in deps).encode()).hexdigest() == rec[1]
            ):
                stamps.append(stamp)
    return sign(root, stamps)
