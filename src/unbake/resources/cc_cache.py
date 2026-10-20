"""A compiler front for the build: `python3 -I cc_cache.py CACHE_DIR REAL_CC ARGS...`.

The Makefile compiles one preprocessed source (`X.i`) to one assembly file (`-o X.s`). The same text under the same
flags and compiler always gives the same assembly, wherever the tree is. A commit that touches a header every source
includes changes every .i of the tree, yet a proof tree has compiled exactly those texts already; this front keeps each
result under a hash of (compiler, arguments, text) so the next tree, the seeded tip cache among them, copies the
assembly instead of compiling it again. A miss runs the compiler and stores the result; anything the front cannot
parse (no -o, no input file) runs the compiler as given.
"""

from __future__ import annotations

import hashlib
import os
import random
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


def prune(root: Path, now: float) -> int:
    dropped = 0
    for entry in root.glob("??/*"):
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
        name = key(compiler, args, Path(source).read_bytes())
    except OSError:
        return subprocess.call([compiler, *args])
    entry = cache / name[:2] / name[2:]
    try:
        data = zlib.decompress(entry.read_bytes())
        Path(out).write_bytes(data)
        os.utime(entry)  # a hit keeps it alive
        return 0
    except (OSError, zlib.error):
        pass
    code = subprocess.call([compiler, *args])
    if code == 0:
        try:
            entry.parent.mkdir(parents=True, exist_ok=True)
            part = entry.with_name(f"{entry.name}.{os.getpid()}.part")
            part.write_bytes(zlib.compress(Path(out).read_bytes(), 3))
            os.replace(part, entry)
        except OSError:
            pass  # a full or read-only cache costs only the saving
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
