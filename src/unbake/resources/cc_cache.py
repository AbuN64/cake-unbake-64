"""A compiler front for the build: `python3 -I cc_cache.py CACHE_DIR REAL_CC ARGS...`.

The Makefile compiles one preprocessed source (`X.i`) to one assembly file (`-o X.s`). The same text under the same
flags and compiler always gives the same assembly, wherever the tree is. A commit that touches a header every source
includes changes every .i of the tree, yet a proof tree has compiled exactly those texts already; this front keeps each
result under a hash of (compiler, arguments, text) so the next tree, the seeded tip cache among them, copies the
assembly instead of compiling it again. A miss runs the compiler and stores the result; anything the front cannot
parse (no -o, no input file) runs the compiler as given.

A header edit that only renames (a field, a type, a function) changes the text of every source that includes it yet
leaves the assembly alone, as does an edit that only touches type declarations and static inline functions the source
never names. A miss whose text is the last one compiled under the same compiler and arguments, with identifiers renamed
one for one and none of the old names anywhere in that assembly, or with only unnamed type declarations or static inline
functions added, dropped or changed, copies the last assembly. The ROM check every build ends with stays the
proof: a wrong reuse cannot
ship, it fails the SHA-1.
"""

from __future__ import annotations

import os
import sys
import zlib

TYPE_CHECKING = False  # typing itself is 1 ms of this front's start
if TYPE_CHECKING:
    import re

try:  # the C module hashlib wraps: importing hashlib itself loads OpenSSL, 4 ms of every call
    from _sha2 import sha256
except ImportError:  # pragma: no cover
    from hashlib import sha256
# Everything a call that finds its assembly (or its last one) does not need loads on first use: this front runs once
# per source of a build (about ten thousand times after a header edit), so its start is most of its cost.

MAX_AGE = 4 * 86400  # seconds an entry survives without a hit
PRUNE_ODDS = 400  # one call in this many walks the cache and drops what aged out


def parse(args: list[str]) -> tuple[str, str] | None:
    """(input, output) of a compile command, None when it is not one this front can key."""
    if "-o" not in args or args.index("-o") + 1 >= len(args):
        return None
    out = args[args.index("-o") + 1]
    sources = [a for a in args if a.endswith(".i")]
    return (sources[-1], out) if len(sources) == 1 and out.endswith(".s") else None


def key(compiler: str, args: list[str], text: bytes) -> str:
    """The compiler binary (path, size, mtime), every argument as given (paths name the .file line the assembly
    carries) and the text of the source."""
    info = os.stat(compiler)
    head = "\0".join([compiler, str(info.st_size), str(info.st_mtime_ns), *args]).encode()
    return sha256(head + b"\0" + text).hexdigest()


TOKEN_SOURCE = rb"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\.?[0-9](?:[eEpP][+-]|[\w.])*|[A-Za-z_]\w*|\s+|."
TOKEN: re.Pattern[bytes]  # the three are compiled by patterns() on first use
IDENT: re.Pattern[bytes]
SPACE: re.Pattern[bytes]
COMPILED = False


def patterns() -> None:
    global TOKEN, IDENT, SPACE, COMPILED
    if not COMPILED:
        import re

        TOKEN = re.compile(TOKEN_SOURCE, re.S)
        IDENT = re.compile(rb"[A-Za-z_]\w*")
        SPACE = re.compile(rb"\s+")
        COMPILED = True


KEYWORDS = frozenset(
    b"asm auto break case char const continue default do double else enum extern float for goto if inline int long "
    b"register return short signed sizeof static struct switch typedef union unsigned void volatile while "
    b"__asm__ __asm __attribute__ __attribute __inline __inline__ __volatile__ __volatile __const __signed__ "
    b"__typeof__ __typeof __extension__ __alignof__ __label__ __FUNCTION__ __PRETTY_FUNCTION__ __func__".split()
)


def renamed(old: bytes, new: bytes, assembly: bytes) -> bool:
    """True when `new` is `old` with identifiers renamed one for one (every use of an old name, no new name already in
    use) and no old name occurs, even as a part of a word, in the assembly `old` compiled to: the same compiler then
    writes the same assembly. Whitespace must match exactly."""
    patterns()
    a, b = TOKEN.findall(old), TOKEN.findall(new)
    if len(a) != len(b):
        return False
    mapping: dict[bytes, bytes] = {}
    for x, y in zip(a, b, strict=True):
        if x == y:
            continue
        if not (IDENT.fullmatch(x) and IDENT.fullmatch(y)) or x in KEYWORDS or y in KEYWORDS:
            return False
        if mapping.setdefault(x, y) != y:
            return False
    new_names = set(mapping.values())
    if not mapping or len(new_names) != len(mapping) or new_names & mapping.keys():
        return False
    kept = {x for x, y in zip(a, b, strict=True) if x == y}
    if kept & mapping.keys() or kept & new_names:
        return False
    return not set(IDENT.findall(assembly)) & mapping.keys()


TYPE_HEAD = frozenset((b"typedef", b"struct", b"union", b"enum"))


def chunks(text: bytes) -> list[list[bytes]]:
    """The text as top-level declarations, each a list of non-blank tokens: a chunk ends at a `;` outside every bracket,
    or at the `}` that closes a function body (the first `{` follows a `)`). Blanks between chunks are dropped."""
    patterns()
    out: list[list[bytes]] = []
    now: list[bytes] = []
    depth = 0
    body = False  # inside a function body: its closing brace ends the chunk
    for tok in TOKEN.findall(text):
        if SPACE.fullmatch(tok):
            continue
        now.append(tok)
        if tok in (b"(", b"[", b"{"):
            if tok == b"{" and depth == 0:
                body = now[-2:-1] == [b")"] or (len(now) > 1 and b")" in now[:-1] and b"=" not in now)
            depth += 1
        elif tok in (b")", b"]", b"}"):
            depth -= 1
            if depth == 0 and tok == b"}" and body:
                out.append(now)
                now, body = [], False
        elif tok == b";" and depth == 0:
            out.append(now)
            now = []
    if now:
        out.append(now)
    return out


INLINE = frozenset((b"inline", b"__inline", b"__inline__"))


def unused_inline(chunk: list[bytes]) -> bytes | None:
    """The name of a `static` inline function definition (the compiler writes nothing for one nothing calls, and its
    body numbers no label or local of another function), else None. A body holding another `static` (a local that
    takes a numbered name) or a `;`-less shape is never one."""
    if chunk[-1] != b"}" or b"static" not in chunk[:3] or b"(" not in chunk:
        return None
    first = chunk.index(b"(")
    if not INLINE & set(chunk[:first]) or first < 2 or not IDENT.fullmatch(chunk[first - 1]):
        return None
    if chunk[first - 1] in KEYWORDS or b"static" in chunk[first:] or chunk.count(b"{") < 1:
        return None
    return chunk[first - 1]


def defined(chunk: list[bytes]) -> set[bytes] | None:
    """The names a droppable chunk introduces, None for any other chunk (a function, a variable or tentative
    definition, an extern, an asm statement: it can change the object and is never dropped). Droppable chunks are a
    static inline function, which introduces its own name, and pure type chunks: a
    typedef, a `struct|union|enum [tag] {...};` and a forward `struct tag;`. Names are the tag and declarators outside
    the braces, plus every enumerator of an enum; a struct's field names are not among them, unless its body defines
    another type, when everything in it counts (a larger set only keeps more chunks)."""
    patterns()
    head = chunk[0]
    if head not in TYPE_HEAD:
        name = unused_inline(chunk)
        return None if name is None else {name}
    if head != b"typedef":
        words = [t for t in chunk if t not in (b"{", b"}")]
        braced = b"{" in chunk
        shape = (
            chunk[-1] == b";"
            and (not braced or (chunk[2 if len(chunk) > 2 and chunk[1] != b"{" else 1] == b"{" and chunk[-2] == b"}"))
            and (braced or len(chunk) == 3)
        )
        if not shape or len(words) < 2:
            return None
    names: set[bytes] = set()
    depth = 0
    nested = any(t in TYPE_HEAD for t in chunk[2:] if t != b"typedef") and b"{" in chunk
    for tok in chunk:
        if tok == b"{":
            depth += 1
        elif tok == b"}":
            depth -= 1
        elif IDENT.fullmatch(tok) and tok not in KEYWORDS and (depth == 0 or nested or head == b"enum"):
            names.add(tok)
    return names


def complete(text: bytes) -> bool:
    """True when `text` is whole top-level declarations: brackets balance, never closing more than they opened, and it
    ends at a `;` or a `}` (or holds no token)."""
    patterns()
    depth, last = 0, b""
    for tok in TOKEN.findall(text):
        if SPACE.fullmatch(tok):
            continue
        depth += (tok in (b"(", b"[", b"{")) - (tok in (b")", b"]", b"}"))
        if depth < 0:
            return False
        last = tok
    return depth == 0 and last in (b"", b";", b"}")


def kept(parts: list[list[bytes]], inrest: set[bytes]) -> list[list[bytes]]:
    """The declarations of one side of the changed region that can reach the object: every chunk that is not a pure
    type, and every pure type that a surviving chunk, or the unchanged text around the region, names."""
    patterns()
    defs = [defined(c) for c in parts]
    keep = [d is None or bool(d & inrest) for d in defs]
    used = {t for c, k in zip(parts, keep, strict=True) if k for t in c if IDENT.fullmatch(t)}
    grew = True
    while grew:
        grew = False
        for i, d in enumerate(defs):
            if not keep[i] and d & used:
                keep[i], grew = True, True
                used.update(t for t in parts[i] if IDENT.fullmatch(t))
    return [c for c, k in zip(parts, keep, strict=True) if k]


def common(old: bytes, new: bytes) -> tuple[int, int]:
    """(length of the common prefix, length of the common suffix not overlapping it), by halving over slice compares."""
    low, high = 0, min(len(old), len(new))
    while low < high:
        mid = (low + high + 1) // 2
        low, high = (mid, high) if old[:mid] == new[:mid] else (low, mid - 1)
    prefix = low
    low, high = 0, min(len(old), len(new)) - prefix
    while low < high:
        mid = (low + high + 1) // 2
        low, high = (mid, high) if old[len(old) - mid :] == new[len(new) - mid :] else (low, mid - 1)
    return prefix, low


def hunks(old: bytes, new: bytes) -> list[tuple[int, int, int, int]]:
    """The whole-line ranges where `old` and `new` differ, as (old first line, old end, new first line, new end) in
    line numbers: equal lines at both ends are trimmed, then lines found once in each text anchor the rest (the longest
    run of them in the same order), so two edits far apart are two small hunks, not one that spans everything."""
    from bisect import bisect_left

    a, b = old.split(b"\n"), new.split(b"\n")
    out: list[tuple[int, int, int, int]] = []
    todo = [(0, len(a), 0, len(b))]
    while todo:
        alo, ahi, blo, bhi = todo.pop()
        while alo < ahi and blo < bhi and a[alo] == b[blo]:
            alo, blo = alo + 1, blo + 1
        while alo < ahi and blo < bhi and a[ahi - 1] == b[bhi - 1]:
            ahi, bhi = ahi - 1, bhi - 1
        if alo == ahi and blo == bhi:
            continue
        pairs: list[tuple[int, int]] = []
        if alo < ahi and blo < bhi:
            once_a: dict[bytes, int] = {}
            for i in range(alo, ahi):
                once_a[a[i]] = -1 if a[i] in once_a else i
            once_b: dict[bytes, int] = {}
            for j in range(blo, bhi):
                once_b[b[j]] = -1 if b[j] in once_b else j
            pairs = sorted((i, once_b[line]) for line, i in once_a.items() if i >= 0 and once_b.get(line, -1) >= 0)
        tails: list[int] = []  # patience: the longest increasing run of new-side positions
        back: list[int] = [-1] * len(pairs)
        ends: list[int] = []
        for n, (_, j) in enumerate(pairs):
            at = bisect_left(tails, j)
            if at == len(tails):
                tails.append(j)
                ends.append(n)
            else:
                tails[at], ends[at] = j, n
            back[n] = ends[at - 1] if at else -1
        chain: list[tuple[int, int]] = []
        n = ends[-1] if ends else -1
        while n >= 0:
            chain.append(pairs[n])
            n = back[n]
        if not chain:
            out.append((alo, ahi, blo, bhi))
            continue
        chain.reverse()
        here = (alo, blo)
        for i, j in chain:
            todo.append((here[0], i, here[1], j))
            here = (i + 1, j + 1)
        todo.append((here[0], ahi, here[1], bhi))
    return sorted(out)


def mentioned(names: set[bytes], text: bytes) -> set[bytes]:
    """The names that occur in `text` as whole words."""
    found: set[bytes] = set()
    for name in names:
        at = text.find(name)
        while at >= 0:
            before, after = text[at - 1 : at] if at else b"", text[at + len(name) : at + len(name) + 1]
            if not (before.isalnum() or before == b"_" or after.isalnum() or after == b"_"):
                found.add(name)
                break
            at = text.find(name, at + 1)
    return found


def types_only_differ(old: bytes, new: bytes) -> bool:
    """True when `old` and `new` differ only in type declarations (typedef, struct, union, enum definitions) and
    static inline functions that nothing else in the text names: no .s byte depends on such a declaration (the
    compile has no debug output), so the same compiler writes the same assembly. Each changed hunk, widened to whole
    declarations, is the only text read closely; everything around them is searched for the names they define."""
    patterns()
    a, b = old.split(b"\n"), new.split(b"\n")
    spans: list[list[int]] = [list(h) for h in hunks(old, new)]
    at = 0
    while at < len(spans):
        alo, ahi, blo, bhi = spans[at]
        for _ in range(80):
            if complete(b"\n".join(a[alo:ahi])) and complete(b"\n".join(b[blo:bhi])):
                break
            if alo and blo:
                alo, blo = alo - 1, blo - 1
            if ahi < len(a) and bhi < len(b):
                ahi, bhi = ahi + 1, bhi + 1
        else:
            return False
        spans[at] = [alo, ahi, blo, bhi]
        if at + 1 < len(spans) and spans[at + 1][0] <= ahi:  # the widened hunk reaches the next: one hunk
            nxt = spans.pop(at + 1)
            spans[at] = [alo, max(ahi, nxt[1]), blo, max(bhi, nxt[3])]
            continue
        at += 1
    inside_a = [b"\n".join(a[s[0] : s[1]]) for s in spans]
    inside_b = [b"\n".join(b[s[2] : s[3]]) for s in spans]
    parts = chunks(b"\n".join(inside_a)), chunks(b"\n".join(inside_b))
    names = {n for side in parts for c in side for n in (defined(c) or ())}
    inrest: set[bytes] = set()
    if names:
        edges = [0, *(x for s in spans for x in (s[0], s[1])), len(a)]
        rest = b"\n".join(line for lo, hi in zip(edges[::2], edges[1::2], strict=True) for line in a[lo:hi])
        inrest = mentioned(names, rest)
    return kept(parts[0], inrest) == kept(parts[1], inrest)


def read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def write(path: str, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)


def remember(cache: str, label: str, source: str) -> None:
    """The text last compiled under this compiler and these arguments: a hard link to the .i (make replaces that file
    and never writes it in place), else a copy."""
    entry = os.path.join(cache, "last", label[:2], label[2:])
    part = f"{entry}.{os.getpid()}.part"
    try:
        os.makedirs(os.path.dirname(entry), exist_ok=True)
        try:  # noqa: SIM105 - contextlib is 1.5 ms of this front's start
            os.unlink(part)
        except FileNotFoundError:
            pass
        try:
            os.link(source, part)
        except OSError:
            write(part, read(source))
        os.replace(part, entry)
    except OSError:
        pass


def reuse(cache: str, label: str, compiler: str, args: list[str], text: bytes) -> bytes | None:
    """The assembly of the last text compiled under this label when `text` only renames it, else None."""
    try:
        last = read(os.path.join(cache, "last", label[:2], label[2:]))
        name = key(compiler, args, last)
        assembly = zlib.decompress(read(os.path.join(cache, name[:2], name[2:])))
    except (OSError, zlib.error):
        return None
    if last == text:
        return None
    patterns()
    # the cheap test first: a header edit that only adds, drops or changes type declarations nothing here names
    return assembly if types_only_differ(last, text) or renamed(last, text, assembly) else None


def prune(root: str, now: float) -> int:
    import glob

    root = os.fspath(root)
    dropped = 0
    for entry in [*glob.glob(os.path.join(root, "??", "*")), *glob.glob(os.path.join(root, "last", "??", "*"))]:
        try:
            if now - os.stat(entry).st_mtime > MAX_AGE:
                os.unlink(entry)
                dropped += 1
        except OSError:
            pass
    return dropped


def call(compiler: str, args: list[str]) -> int:
    import subprocess

    return subprocess.call([compiler, *args])


def run(cache: str, compiler: str, args: list[str]) -> int:
    cache = os.fspath(cache)
    parsed = parse(args)
    if parsed is None:
        return call(compiler, args)
    source, out = parsed
    try:
        text = read(source)
        name = key(compiler, args, text)
    except OSError:
        return call(compiler, args)
    entry = os.path.join(cache, name[:2], name[2:])
    label = key(compiler, args, b"")  # the unit: this compiler and these arguments, whatever the text
    try:
        data = zlib.decompress(read(entry))
        write(out, data)
        os.utime(entry)  # a hit keeps it alive
        remember(cache, label, source)
        return 0
    except (OSError, zlib.error):
        pass
    code, made = 0, reuse(cache, label, compiler, args, text)
    if made is not None:
        write(out, made)
    else:
        code = call(compiler, args)
        made = read(out) if code == 0 else None
    if code == 0 and made is not None:
        part = f"{entry}.{os.getpid()}.part"
        try:
            os.makedirs(os.path.dirname(entry), exist_ok=True)
            write(part, zlib.compress(made, 3))
            os.replace(part, entry)
        except OSError:
            pass  # a full or read-only cache costs only the saving
        remember(cache, label, source)
    import random

    if random.randrange(PRUNE_ODDS) == 0:
        import time

        prune(cache, time.time())
    return code


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        sys.stderr.write("usage: cc_cache.py CACHE_DIR REAL_CC ARGS...\n")
        return 2
    return run(argv[1], argv[2], argv[3:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
