"""A compiler front for the build: `python3 -I cc_cache.py CACHE_DIR REAL_CC ARGS...`.

The Makefile compiles one preprocessed source (`X.i`) to one assembly file (`-o X.s`). The same text under the same
flags and compiler always gives the same assembly, wherever the tree is. A commit that touches a header every source
includes changes every .i of the tree, yet a proof tree has compiled exactly those texts already; this front keeps each
result under a hash of (compiler identity, every argument, the exact bytes of the text) so the next tree copies the
assembly instead of compiling it again. A miss runs the compiler and stores the result; anything the front cannot
parse (no -o, no input file) runs the compiler as given.

The same cache answers the preprocessor (`--unit`, the whole `X.i.ok` recipe in one process): one source under the
same arguments preprocesses to the same text while every file it included has the bytes it had. An entry keeps the
text, the dependency file and the hash of each included file; a hit reads those files again and compares them, so
the versions of a unit and every tree preprocess it once while nothing it includes moved.

The key is exact: no "similar" text is ever answered with another text's assembly (10-22: a reuse of the last text for
texts that only reordered statements served stale assembly, and a stale answer that happens to match passes the ROM
check). The chunk parser below serves tree.py's parse key, never this cache.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import zlib

TYPE_CHECKING = False  # typing itself is 1 ms of this front's start
if TYPE_CHECKING:
    import re

try:  # the C modules hashlib wraps: importing hashlib itself loads OpenSSL, 4 ms of every call
    from _sha1 import sha1  # type: ignore[import-not-found]
    from _sha2 import sha256  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    from hashlib import sha1, sha256
# Everything a call that finds its assembly does not need loads on first use: this front runs once per source of a
# build (about ten thousand times after a header edit), so its start is most of its cost.

MAX_AGE = 4 * 86400  # seconds an entry survives without a hit
PRUNE_ODDS = 400  # one call in this many walks the cache and drops what aged out


def parse(args: list[str]) -> tuple[str, str] | None:
    """(input, output) of a compile command, None when it is not one this front can key."""
    if "-o" not in args or args.index("-o") + 1 >= len(args):
        return None
    out = args[args.index("-o") + 1]
    sources = [a for a in args if a.endswith(".i")]
    return (sources[-1], out) if len(sources) == 1 and out.endswith(".s") else None


IDENTITIES: dict[str, bytes] = {}


def identity(compiler: str) -> bytes:
    """The compiler as its output can depend on it: the bytes of every file beside it (a driver such as IDO's cc runs
    the passes in its own folder), by name. Read once per process."""
    if compiler not in IDENTITIES:
        folder = os.path.dirname(os.path.abspath(compiler))
        digest = sha256(os.path.abspath(compiler).encode())
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                digest.update(b"\0" + name.encode(errors="surrogateescape") + b"\0" + read(path))
        IDENTITIES[compiler] = digest.digest()
    return IDENTITIES[compiler]


def key(compiler: str, args: list[str], text: bytes) -> str:
    """The compiler's identity, every argument as given (paths name the .file line the assembly carries) and the exact
    bytes of the source text."""
    head = "\0".join(args).encode(errors="surrogateescape")
    return str(sha256(identity(compiler) + b"\0" + head + b"\0" + text).hexdigest())


def answer_key(compiler: str, args: list[str], text: bytes) -> str:
    """What a batch answer (`<out>.cc`) is for, written beside it as `<out>.cc.key`: the command line exactly as the
    front records it, then the text. The front (cc_front.sh) takes the answer only when its own command and input give
    the same hash, so an answer is never used for another text or other flags."""
    line = SEPARATOR.join([compiler, *args]) + "\n"
    return str(sha256(line.encode(errors="surrogateescape") + text).hexdigest())


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


def read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def write(path: str, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)


def prune(root: str, now: float) -> int:
    import glob

    root = os.fspath(root)
    dropped = 0
    for entry in glob.glob(os.path.join(root, "??", "*")):
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


def lookup(cache: str, compiler: str, args: list[str], text: bytes) -> bytes | None:
    """The stored assembly of exactly this text under this compiler and these arguments; None: the compiler must run."""
    name = key(compiler, args, text)
    entry = os.path.join(cache, name[:2], name[2:])
    try:
        data = zlib.decompress(read(entry))
        os.utime(entry)  # a hit keeps it alive
        return data
    except (OSError, zlib.error):
        return None


def store(entry: str, made: bytes) -> None:
    part = f"{entry}.{os.getpid()}.part"
    try:
        os.makedirs(os.path.dirname(entry), exist_ok=True)
        write(part, zlib.compress(made, 3))
        os.replace(part, entry)
    except OSError:
        pass  # a full or read-only cache costs only the saving


def maybe_prune(cache: str) -> None:
    import random

    if random.randrange(PRUNE_ODDS) == 0:
        import time

        prune(cache, time.time())


def take_answer(compiler: str, args: list[str], out: str, text: bytes) -> bool:
    """The batch's answer for this unit (`<out>.cc`, see `batch`) moved to `out` when its key is the hash of this very
    command and text; a leftover answer or key of any other kind is dropped."""
    try:
        with open(out + ".cc.key", encoding="ascii") as handle:
            wanted = handle.read().strip()
        if wanted == answer_key(compiler, args, text):
            os.replace(out + ".cc", out)
            _remove(out + ".cc.key")
            return True
    except OSError:
        pass
    _remove(out + ".cc")
    _remove(out + ".cc.key")
    return False


def run(cache: str, compiler: str, args: list[str]) -> int:
    cache = os.fspath(cache)
    parsed = parse(args)
    if parsed is None:
        return call(compiler, args)
    source, out = parsed
    try:
        text = read(source)
        if take_answer(compiler, args, out, text):
            return 0
        name = key(compiler, args, text)
    except OSError:
        return call(compiler, args)
    found = lookup(cache, compiler, args, text)
    if found is not None:
        write(out, found)
        return 0
    code = call(compiler, args)
    if code == 0:
        store(os.path.join(cache, name[:2], name[2:]), read(out))
    maybe_prune(cache)
    return code


def pack(parts: list[bytes]) -> bytes:
    return b"".join(len(p).to_bytes(8, "big") + p for p in parts)


def unpack(data: bytes) -> list[bytes]:
    parts, at = [], 0
    while at < len(data):
        size = int.from_bytes(data[at : at + 8], "big")
        parts.append(data[at + 8 : at + 8 + size])
        at += 8 + size
    return parts


@functools.cache  # a batch asks for the same preprocessor once per unit: the driver is started once
def preprocessor_identity(preprocessor: str) -> bytes:
    """The preprocessor as its output can depend on it: the driver and the program it runs (cc1), by path, size and
    date. Not the files beside it: a host cpp lives in /usr/bin."""
    import subprocess

    names = [os.path.realpath(preprocessor)]
    asked = subprocess.run([preprocessor, "-print-prog-name=cc1"], capture_output=True, check=False)
    found = asked.stdout.decode(errors="surrogateescape").strip()
    if asked.returncode == 0 and os.path.isabs(found):
        names.append(os.path.realpath(found))
    digest = sha256(b"")
    for name in names:
        try:
            info = os.stat(name)
            digest.update(f"{name}\0{info.st_size}\0{info.st_mtime_ns}\0".encode(errors="surrogateescape"))
        except OSError:
            digest.update(name.encode(errors="surrogateescape") + b"\0")
    return bytes(digest.digest())


def preprocess_parse(args: list[str]) -> tuple[str, str] | None:
    """(source, dependency file) of a preprocess command the cache can answer: the source is the last argument, the
    dependency file has a phony rule per header (-MP: the one way every included file is named) and the text goes to
    standard output."""
    if "-MP" not in args or "-o" in args or "-MF" not in args or args.index("-MF") + 1 >= len(args):
        return None
    return (args[-1], args[args.index("-MF") + 1]) if not args[-1].startswith("-") else None


def included(depfile: bytes) -> list[str]:
    """The files a dependency file names beyond the source: its phony rules (`header:`)."""
    names = []
    for line in depfile.decode(errors="surrogateescape").splitlines():
        if line.endswith(":") and not line.startswith((" ", "\t")):
            names.append(line[:-1].replace("\\ ", " "))
    return names


def digests(names: list[str]) -> bytes | None:
    """`<sha256> <name>` per included file; None when one cannot be read (no entry can be made)."""
    lines = []
    for name in names:
        try:
            lines.append(f"{sha256(read(name)).hexdigest()} {name}")
        except OSError:
            return None
    return "\n".join(lines).encode(errors="surrogateescape")


def unchanged(recorded: bytes) -> bool:
    """Every included file an entry names still has the bytes it had."""
    for line in recorded.decode(errors="surrogateescape").splitlines():
        want, _, name = line.partition(" ")
        try:
            if sha256(read(name)).hexdigest() != want:
                return False
        except OSError:
            return False
    return True


def preprocess(cache: str, preprocessor: str, args: list[str]) -> tuple[int, bytes, bytes]:
    """(exit, text, errors) of `preprocessor args`: from the cache when this source under these arguments last gave a
    text whose included files still have their bytes, else by running it and keeping the result. The dependency file
    the arguments name is written either way."""
    import subprocess

    cache = os.fspath(cache)
    parsed = preprocess_parse(args) if cache else None
    if parsed is not None:
        source, depfile = parsed
        try:
            head = preprocessor_identity(preprocessor) + b"\0" + "\0".join(args).encode(errors="surrogateescape")
            name = sha256(b"pp\0" + head + b"\0" + read(source)).hexdigest()
        except OSError:
            parsed = None
    if parsed is None:
        made = subprocess.run([preprocessor, *args], capture_output=True, check=False)
        return made.returncode, made.stdout, made.stderr
    entry = os.path.join(cache, name[:2], name[2:])
    try:
        recorded, made_dep, text, errors = unpack(zlib.decompress(read(entry)))
    except (OSError, zlib.error, ValueError):
        recorded = None
    if recorded is not None and unchanged(recorded):
        os.utime(entry)  # a hit keeps it alive
        write(depfile, made_dep)
        return 0, text, errors
    made = subprocess.run([preprocessor, *args], capture_output=True, check=False)
    if made.returncode == 0:
        try:
            made_dep = read(depfile)
            if (hashes := digests(included(made_dep))) is not None:
                store(entry, pack([hashes, made_dep, made.stdout, made.stderr]))
        except OSError:
            pass
    maybe_prune(cache)
    return made.returncode, made.stdout, made.stderr


def same(a: str, b: str) -> bool:
    """`cmp -s a b`: both files exist and hold the same bytes."""
    try:
        return read(a) == read(b)
    except OSError:
        return False


def newer(a: str, b: str) -> bool:
    """`[ a -nt b ]`: a exists and b is older or missing."""
    try:
        return not os.path.exists(b) or os.stat(a).st_mtime_ns > os.stat(b).st_mtime_ns
    except OSError:
        return False


def compile_key(hdrkey: str, text: str, out: str) -> bool:
    """`hdrkey text > out`; False when the tool is missing or fails."""
    import subprocess

    if not os.access(hdrkey, os.X_OK):
        return False
    with open(out, "wb") as handle:
        return subprocess.run([hdrkey, text], stdout=handle, check=False).returncode == 0


def settle(i: str, hdrkey: str) -> None:
    """Put the text `i.new` where `i` is: the old `i` stays, with its date, when it has the same compile key (a header
    edit that moves no code changes nothing the compiler sees); the key goes beside it (`i.key`)."""
    new, key_file = i + ".new", i + ".key"
    if same(new, i):
        os.unlink(new)
        return
    kept = False
    if compile_key(hdrkey, new, key_file + ".new"):
        kept = (
            os.path.isfile(i)
            and (newer(key_file, i) or compile_key(hdrkey, i, key_file))
            and same(key_file + ".new", key_file)
        )
    else:
        _remove(key_file + ".new")
    if kept:
        os.unlink(new)
        _remove(key_file + ".new")
        return
    os.replace(new, i)
    if os.path.exists(key_file + ".new") and os.path.getsize(key_file + ".new"):
        os.replace(key_file + ".new", key_file)
    else:
        _remove(key_file)
        _remove(key_file + ".new")


def _remove(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


def record_stamp(stamprec: str, stamp: str, *extracted: str) -> int:
    """The stamp's input record (build/.inputs.log): the C tool when it was built, else the sweep script."""
    import subprocess

    if (
        os.access(stamprec, os.X_OK)
        and subprocess.run([stamprec, "record", stamp, *extracted], check=False).returncode == 0
    ):
        return 0
    return subprocess.run(
        [
            "python3",
            "-I",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "make", "stamp_sweep.py"),
            "record",
            stamp,
            *extracted,
        ],
        check=False,
    ).returncode


def unit(
    cache: str,
    stamp: str,
    flags: str,
    hdrkey: str,
    stamprec: str,
    preprocessor: str,
    args: list[str],
    retry: bool = False,
) -> int:
    """The whole recipe of one `X.i.ok` stamp in this one process: preprocess (through the cache), put the text in
    `X.i` (settle), write the flags it was made with, date the stamp and record its inputs. `retry`: the caller's make
    runs the recipe again when this fails, so a failure says nothing here."""
    os.makedirs(os.path.dirname(stamp) or ".", exist_ok=True)
    code, text, errors = preprocess(cache, preprocessor, args)
    if code == 0 or not retry:
        sys.stderr.buffer.write(errors)
        sys.stderr.buffer.flush()
    i = stamp[: -len(".ok")]
    write(i + ".new", text)
    if code != 0:
        return code
    settle(i, hdrkey)
    write(stamp + ".flags", (flags + "\n").encode(errors="surrogateescape"))
    with open(stamp, "ab"):
        pass
    os.utime(stamp)
    return record_stamp(stamprec, stamp)


def trim_text(path: str, size: str) -> None:
    """The .text size of a big-endian ELF32 object cut to `size` bytes when the bytes past it are zero (the object
    assembles to a text the ROM's slot holds in full)."""
    import struct

    n = int(size, 0)
    d = bytearray(read(path))
    o = struct.unpack_from(">I", d, 32)[0]
    e, c, x = struct.unpack_from(">HHH", d, 46)
    names = struct.unpack_from(">I", d, o + x * e + 16)[0]
    text = [
        o + k * e
        for k in range(c)
        if d[names + struct.unpack_from(">I", d, o + k * e)[0] :].split(b"\0")[0] == b".text"
    ]
    start, length = struct.unpack_from(">II", d, text[0] + 16) if text else (0, 0)
    if text and length > n and not d[start + n : start + length].strip(b"\0"):
        struct.pack_into(">I", d, text[0] + 20, n)
        write(path, bytes(d))


def is_file(path: str) -> bool:
    return os.path.isfile(path)


def jump_table(path: str, base: str) -> int:
    """A jump table entry `.word .L12` of the assembly holds its label's address less BASE (a hex number)."""
    import re

    text = read(path)
    made = re.sub(
        rb"^([ \t]*\.word[ \t]+[.$]L[0-9]+)[ \t]*$", lambda m: m[1] + b"-0x" + base.encode(), text, flags=re.M
    )
    if made != text:
        write(path, made)
    return 0


def front_command(words: list[str]) -> bool:
    """Whether `words` are the compiler front, `sh .../cc_front.sh CACHE COMPILER ARGS...`."""
    return len(words) >= 5 and words[0] == "sh" and words[1].endswith("cc_front.sh")


def compile_step(words: list[str]) -> int:
    """The compile command: the front's, while the build only collects, is recorded and failed (125); else the batch's
    answer is moved into place; a unit the batch could not answer is the front's own run (the cache, then the compiler).
    Any other command is just started."""
    import subprocess

    if front_command(words):
        compiler, args = words[3], words[4:]
        if collect := os.environ.get("ABU_CC_COLLECT"):
            line = (SEPARATOR.join([compiler, *args]) + "\n").encode(errors="surrogateescape")
            fd = os.open(
                collect, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666
            )  # one write: lines this short never interleave
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
            return 125
        parsed = parse(args)
        if parsed is not None:
            with contextlib.suppress(OSError):
                if take_answer(compiler, args, parsed[1], read(parsed[0])):
                    return 0
    return subprocess.call(words)


def object_step(words: list[str], state: dict[str, object]) -> int:
    """One step of an object's program: a command to start, or a file operation the chain used to start a process for.

    `@rotate S KEPT`: the last assembly is kept (a KEPT still there says the last run failed: S goes, never into it).
    `@compile COMMAND...`: the compiler front (compile_step).  `@nonempty S`: a compiler that exited 0 without assembly
    has not compiled.  `@jumptable S HEX`: jump_table.
    `@out FILE COMMAND...`: the command's output goes to FILE.  `@settle S KEPT`: the object stands
    as it was when only its .i moved, its size is the wanted one and S is KEPT byte for byte (the rest of the program
    is skipped, `.same` tells the rule its trim is held); KEPT goes either way."""
    import subprocess

    head, rest = words[0], words[1:]
    if head == "@rotate" and len(rest) == 2:
        if is_file(rest[1]):
            _remove(rest[0])
        elif is_file(rest[0]):
            os.replace(rest[0], rest[1])
        return 0
    if head == "@nonempty" and len(rest) == 1:
        return 0 if os.path.exists(rest[0]) and os.path.getsize(rest[0]) > 0 else 1
    if head == "@jumptable" and len(rest) == 2:
        return jump_table(*rest)
    if head == "@compile" and rest:
        return compile_step(rest)
    if head == "@out" and len(rest) >= 2:
        with open(rest[0], "wb") as handle:
            return subprocess.call(rest[1:], stdout=handle)
    if head == "@settle" and len(rest) == 2:
        assembly, kept = rest
        out = str(state["out"])
        same = False
        if not state["changed"] and state["recorded"] == state["textsize"] and is_file(out) and is_file(kept):
            same = read(assembly) == read(kept)
        _remove(kept)
        if same:
            os.utime(out)
            write(out + ".same", b"")
            state["stop"] = True
        return 0
    if head.startswith("@"):
        sys.stderr.write(f"cc_cache: no step {head}\n")
        return 64
    try:
        return subprocess.call(words)
    except OSError as error:
        sys.stderr.write(f"cc_cache: cannot run {words[0]}: {error.strerror}\n")
        return 127


def object_file(
    out: str, version: str, facts: str, changed: str, textsize: str, flags: str, stamprec: str, rest: list[str]
) -> int:
    """The whole recipe of one `X.o` in this one process (the shell ran about ten commands, then the chain's twelve):
    the extracted files it assembles are what they were (`<out>.asm`, their sum; a different sum removes the object),
    the object stands when nothing but its stamp moved and its text size and flags are the ones it was made with, else
    the program (`rest` after the extracted files and a `--`: steps separated by `&&`, see object_step) makes it and
    the trim and `.textsize` follow; then the flags, the sum and the input record."""
    try:
        made = read(f"asm/{version}/.extracted").decode().rstrip("\n")
    except OSError:
        made = ""
    if made and made != facts:
        sys.stderr.write(
            f"asm/{version} was extracted from other symbol_addrs.txt/splat.yaml than this tree holds "
            f"({made}, now {facts}): "
            f"run make extract-{version}\n"
        )
        return 1
    split = rest.index("--") if "--" in rest else len(rest)
    extracted, program = rest[:split], rest[split + 1 :]
    if not program:
        return 64
    digest = sha1()
    for name in extracted:
        with contextlib.suppress(OSError):
            digest.update(read(name))
    total = f"{digest.hexdigest()}  -\n".encode()
    try:
        if read(out + ".asm") != total:
            _remove(out)
    except OSError:
        pass
    flags_line = (flags + "\n").encode(errors="surrogateescape")
    try:
        recorded = read(out + ".textsize").decode(errors="surrogateescape").removesuffix("\n")
    except OSError:
        recorded = ""
    source = out[: -len(".o")] + ".i"
    try:
        stands = not changed and recorded == textsize and os.stat(source).st_mtime_ns < os.stat(out).st_mtime_ns
    except OSError:
        stands = False
    if stands:
        write(out + ".flags", flags_line)
        os.utime(out)
    else:
        state: dict[str, object] = {
            "out": out,
            "changed": changed,
            "textsize": textsize,
            "recorded": recorded,
            "stop": False,
        }
        steps: list[list[str]] = [[]]
        for word in program:
            if word == "&&":
                steps.append([])
            else:
                steps[-1].append(word)
        for words in steps:
            code = object_step(words, state) if words else 0
            if code:
                return code
            if state["stop"]:
                break
        if os.path.exists(out + ".same"):  # the object stands as it was (the compile's own check): its trim is held
            _remove(out + ".same")
        else:
            if textsize:
                trim_text(out, textsize)
            write(out + ".textsize", (textsize + "\n").encode(errors="surrogateescape"))
        write(out + ".flags", flags_line)
    write(out + ".asm", total)
    return record_stamp(stamprec, out, *extracted)


INLINE_UNITS = (
    64  # units one make preprocesses itself before it asks for a batch: the first run of a small edit stays one run
)


def defer(manifest: str, row: list[str]) -> bool:
    """Collect mode (resources/cc_front.sh explains the two makes): record the unit's command for the batch, unless
    fewer than INLINE_UNITS units came first; True: the unit is recorded and fails (125)."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    fd = os.open(manifest + ".n", flags, 0o644)
    try:
        os.write(fd, b".")
        if os.fstat(fd).st_size <= INLINE_UNITS:
            return False
    finally:
        os.close(fd)
    fd = os.open(manifest, flags, 0o644)
    try:
        os.write(fd, (SEPARATOR.join(row) + "\n").encode(errors="surrogateescape"))  # one write: never interleaves
    finally:
        os.close(fd)
    return True


SEPARATOR = "\x1f"  # between the fields of a compile command the front records (cc_front.sh)


def commands(manifest: str) -> list[tuple[str, list[str]]]:
    """(compiler, arguments) of every compile command the front recorded, in the order written."""
    with open(manifest, encoding="utf-8", errors="surrogateescape") as handle:
        rows = [line.rstrip("\n").split(SEPARATOR) for line in handle]
    return [(row[0], row[1:]) for row in rows if row and row[0]]


def batch(cache: str, root: str, manifest: str, index: int, count: int) -> int:
    """Resolve, without the compiler, every recorded compile whose number is `index` modulo `count`: its assembly goes
    to `<out>.cc` and its answer key (`answer_key`) to `<out>.cc.key`; the front moves the answer into place only for
    exactly that command and input. A compile the cache cannot answer is left to the front, which runs the compiler.
    Returns how many it resolved."""
    done = 0
    for number, (compiler, args) in enumerate(commands(manifest)):
        parsed = parse(args)
        if parsed is None or number % count != index:
            continue
        source, out = os.path.join(root, parsed[0]), os.path.join(root, parsed[1]) + ".cc"
        try:
            text = read(source)
            found = lookup(cache, compiler, args, text)
            if found is None:
                continue
            write(out, found)
            write(out + ".key", (answer_key(compiler, args, text) + "\n").encode())
        except OSError:
            continue
        done += 1
    return done


def units(cache: str, manifest: str, index: int, count: int) -> int:
    """Make, without the preprocessor's own make recipe, every recorded stamp whose number is `index` modulo `count`
    (`unit`): a stamp that fails here is left to the second make, which runs the recipe again and names the error.
    Returns how many it made."""
    done = 0
    for number, (stamp, row) in enumerate(commands(manifest)):
        if number % count == index and len(row) >= 4:
            flags, hdrkey, stamprec, preprocessor, *args = row
            done += unit(cache, stamp, flags, hdrkey, stamprec, preprocessor, args, retry=True) == 0
    return done


def main(argv: list[str]) -> int:
    if len(argv) == 7 and argv[1] == "--batch":
        batch(argv[2], argv[3], argv[4], int(argv[5]), int(argv[6]))
        return 0
    if len(argv) == 6 and argv[1] == "--units":
        units(argv[2], argv[3], int(argv[4]), int(argv[5]))
        return 0
    if len(argv) >= 10 and argv[1] == "--object":
        # --object OUT VERSION FACTS CHANGED TEXTSIZE FLAGS STAMPREC [EXTRACTED...] -- PROGRAM...
        return object_file(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9:])
    if len(argv) == 4 and argv[1] == "--trim":
        trim_text(argv[2], argv[3])
        return 0
    if len(argv) >= 9 and argv[1] == "--unit":
        # --unit CACHE_DIR STAMP FLAGS HDRKEY STAMPREC PREPROCESSOR ARGS...
        if (collect := os.environ.get("ABU_PP_COLLECT")) and defer(collect, argv[3:]):
            return 125
        stamp, flags, hdrkey, stamprec, preprocessor = argv[3:8]
        return unit(argv[2], stamp, flags, hdrkey, stamprec, preprocessor, argv[8:])
    if len(argv) < 3:
        sys.stderr.write("usage: cc_cache.py CACHE_DIR REAL_CC ARGS...\n")
        return 2
    return run(argv[1], argv[2], argv[3:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
