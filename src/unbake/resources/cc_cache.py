"""A compiler front for the build: `python3 -I cc_cache.py CACHE_DIR REAL_CC ARGS...`.

The Makefile compiles one preprocessed source (`X.i`) to one assembly file (`-o X.s`). The same text under the same
flags and compiler always gives the same assembly, wherever the tree is. A commit that touches a header every source
includes changes every .i of the tree, yet a proof tree has compiled exactly those texts already; this front keeps each
result under a hash of (compiler, arguments, text) so the next tree, the seeded tip cache among them, copies the
assembly instead of compiling it again. A miss runs the compiler and stores the result; anything the front cannot
parse (no -o, no input file) runs the compiler as given.

A header edit that only renames (a field, a type, a function) changes the text of every source that includes it yet
leaves the assembly alone. A miss whose text is the last one compiled under the same compiler and arguments with
identifiers renamed one for one, and none of the old names anywhere in that assembly, copies the last assembly. The
ROM check every build ends with stays the proof: a wrong reuse cannot ship, it fails the SHA-1.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

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
    return hashlib.sha256(head + b"\0" + text).hexdigest()


TOKEN = re.compile(rb"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\.?[0-9](?:[eEpP][+-]|[\w.])*|[A-Za-z_]\w*|\s+|.", re.S)
IDENT = re.compile(rb"[A-Za-z_]\w*")
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


def remember(cache: Path, label: str, source: str) -> None:
    """The text last compiled under this compiler and these arguments: a hard link to the .i (make replaces that file
    and never writes it in place), else a copy."""
    entry = cache / "last" / label[:2] / label[2:]
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        part = entry.with_name(f"{entry.name}.{os.getpid()}.part")
        part.unlink(missing_ok=True)
        try:
            os.link(source, part)
        except OSError:
            part.write_bytes(Path(source).read_bytes())
        os.replace(part, entry)
    except OSError:
        pass


def reuse(cache: Path, label: str, compiler: str, args: list[str], text: bytes) -> bytes | None:
    """The assembly of the last text compiled under this label when `text` only renames it, else None."""
    try:
        last = (cache / "last" / label[:2] / label[2:]).read_bytes()
        assembly = zlib.decompress((cache / (name := key(compiler, args, last))[:2] / name[2:]).read_bytes())
    except (OSError, zlib.error):
        return None
    return assembly if last != text and renamed(last, text, assembly) else None


def prune(root: Path, now: float) -> int:
    dropped = 0
    for entry in [*root.glob("??/*"), *root.glob("last/??/*")]:
        try:
            if now - entry.stat().st_mtime > MAX_AGE:
                entry.unlink()
                dropped += 1
        except OSError:
            pass
    return dropped


def run(cache: Path, compiler: str, args: list[str]) -> int:
    parsed = parse(args)
    if parsed is None:
        return subprocess.call([compiler, *args])
    source, out = parsed
    try:
        text = Path(source).read_bytes()
        name = key(compiler, args, text)
    except OSError:
        return subprocess.call([compiler, *args])
    entry = cache / name[:2] / name[2:]
    label = key(compiler, args, b"")  # the unit: this compiler and these arguments, whatever the text
    try:
        data = zlib.decompress(entry.read_bytes())
        Path(out).write_bytes(data)
        os.utime(entry)  # a hit keeps it alive
        remember(cache, label, source)
        return 0
    except (OSError, zlib.error):
        pass
    code, made = 0, reuse(cache, label, compiler, args, text)
    if made is not None:
        Path(out).write_bytes(made)
    else:
        code = subprocess.call([compiler, *args])
        made = Path(out).read_bytes() if code == 0 else None
    if code == 0 and made is not None:
        try:
            entry.parent.mkdir(parents=True, exist_ok=True)
            part = entry.with_name(f"{entry.name}.{os.getpid()}.part")
            part.write_bytes(zlib.compress(made, 3))
            os.replace(part, entry)
        except OSError:
            pass  # a full or read-only cache costs only the saving
        remember(cache, label, source)
    if random.randrange(PRUNE_ODDS) == 0:
        prune(cache, time.time())
    return code


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        sys.stderr.write("usage: cc_cache.py CACHE_DIR REAL_CC ARGS...\n")
        return 2
    return run(Path(argv[1]), argv[2], argv[3:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
