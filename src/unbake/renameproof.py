"""`unbake renameproof`: a change that only renames identifiers is proved by its rename map, not by a build.

A rename of a shared header or symbol fact changes every preprocessed unit that reads it, so the build recompiles
thousands of units whose objects cannot change. The proof is the same statement made once, over the files the change
touches: the files after the change equal the files before it with one rename map applied, token by token, and the
map is one-to-one. Every unit preprocesses from those files, so every unit then preprocesses to its old text with the
map applied; every object is the old object under the map; the ROM bytes are equal. Anything else is not a pure
rename and refuses, so the normal build proof runs instead.

The pieces of the proof, each a reason to refuse:
- every changed file is text and pairs with its old file: no file is added or removed (a declared inert path may);
- C files compare as tokens (comments and spacing are free; a directive keeps its line ends), every other text file
  compares as words with its line indents; a token that differs must be an identifier on both sides, and the
  (old, new) pairs read off the differences are the map; one old name has one new name and no two share one;
- the map applied to every old token gives the new file (a name renamed in one place and left in another refuses);
- a string differs only inside inline assembly text, where the map is the symbol rename it stands for; any other
  string, number or operator is equal exactly;
- no old name remains as a code token in an unchanged file, and no new name was already a code token in the old tree
  (that would capture or collide); a swap inside the map is allowed;
- a renamed identifier is never pasted (`##`) and is stringified (`#x`) only into inline assembly text, through
  every macro call that holds it as an argument, to the depth the macro definitions go;
- no name, old or new, is one the compiler knows as a builtin (`sqrtf`, `memcpy`): it expands those inline, so a
  rename across them changes the objects; a project that compiles with `-fno-builtin` is exempt;
- where the tree reads `__LINE__`, every changed token stays on its line.
The map is read from the change itself, so no project, name or file is special."""

from __future__ import annotations

import bisect
import fnmatch
import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from unbake import process
from unbake.contracts import Finding, Json, Refusal

C_SUFFIXES = (".c", ".cpp", ".cc", ".h", ".hpp")
NOT_CODE = (":(exclude)*.md", ":(exclude)*.tsv", ":(exclude)*.json", ":(exclude)*.svg")  # prose and tables of names
ASM_MARK = "\0"
ASM_WORDS = frozenset({"asm", "__asm", "__asm__"})
ASM_QUALIFIERS = frozenset({"volatile", "__volatile", "__volatile__"})
NOT_MACROS = frozenset({"if", "while", "for", "switch", "return", "sizeof", "defined", "__attribute__", "else"})
_MATH = "sqrt sin cos tan asin acos atan atan2 exp log log10 pow fabs floor ceil fmod ldexp frexp modf sinh cosh tanh"
_LIBC = (
    "abs labs llabs ffs alloca memcpy memmove memset memcmp memchr strlen strcpy strncpy strcat strncat strcmp strncmp"
    " strchr strrchr strstr strspn strcspn strpbrk index rindex bcopy bzero bcmp printf sprintf puts putchar abort"
    " exit setjmp longjmp va_start va_end va_arg"
)
# names a C compiler expands or folds itself: a rename across them changes code
BUILTINS = frozenset(_LIBC.split() + _MATH.split() + [n + s for n in _MATH.split() for s in ("f", "l")])
SHOWN = 6  # reasons named in the refusal
DEPTH = 6  # macro definitions followed from a call argument

_C = re.compile(
    r"""(?P<skip>[ \t\r\f\v]+|\\\r?\n|//[^\n]*|/\*.*?\*/)
    |(?P<nl>\n)
    |(?P<str>"(?:\\.|[^"\\\n])*")
    |(?P<chr>'(?:\\.|[^'\\\n])*')
    |(?P<id>[A-Za-z_$][A-Za-z_0-9$]*)
    |(?P<num>\.?[0-9](?:[eEpP][+-]|[\w.])*)
    |(?P<op>.)""",
    re.S | re.X,
)
_TEXT = re.compile(
    r"""(?P<skip>[ \t\r\f\v]+)|(?P<nl>\n)|(?P<id>[A-Za-z_$][A-Za-z_0-9$]*)|(?P<num>\.?[0-9][\w.]*)|(?P<op>.)""",
    re.S | re.X,
)
_IDENT = re.compile(r"[A-Za-z_$][A-Za-z_0-9$]*\Z")
_WORD = re.compile(r"[A-Za-z_$][A-Za-z_0-9$]*")


def _is_c(path: str) -> bool:
    return path.lower().endswith(C_SUFFIXES)


def _tokens(text: str, c: bool, starts: list[int] | None = None) -> list[str]:
    """The tokens of a file; `starts` collects each token's offset. C: no comments or spacing, a line end only in a
    directive, a string inside an inline assembly statement marked. Text: words, numbers, punctuation, the indent of
    each line and every line end."""
    out: list[str] = []
    add = out.append
    line_start, directive = True, False
    asm_wait, asm_depth = False, 0
    size = len(text)
    for m in (_C if c else _TEXT).finditer(text):
        kind, token = m.lastgroup, m.group()
        if kind == "skip":
            if not c and line_start and m.end() < size and text[m.end()] != "\n":
                add(" " + token)  # an indent is structure in a text file
                if starts is not None:
                    starts.append(m.start())
            continue
        if kind == "nl":
            if not c or directive:
                add("\n")
                if starts is not None:
                    starts.append(m.start())
            line_start, directive, asm_wait = True, False, False
            continue
        if c:
            if kind == "id":
                if token in ASM_WORDS:
                    asm_wait = True
                elif asm_wait and token not in ASM_QUALIFIERS:
                    asm_wait = False
            elif kind == "op":
                if token == "(":
                    if asm_depth:
                        asm_depth += 1
                    elif asm_wait:
                        asm_depth, asm_wait = 1, False
                elif token == ")" and asm_depth:
                    asm_depth -= 1
                else:
                    asm_wait = False
                if token == "#" and line_start:
                    directive = True
            elif kind == "str" and asm_depth:
                token = ASM_MARK + token
        line_start = False
        add(token)
        if starts is not None:
            starts.append(m.start())
    return out


def _is_id(token: str) -> bool:
    return bool(_IDENT.match(token))


def _asm_flags(tokens: Sequence[str]) -> list[bool]:
    """For each token, whether it stands inside the parentheses of an inline assembly statement."""
    flags: list[bool] = []
    wait, depth = False, 0
    for token in tokens:
        flags.append(depth > 0)
        if token in ASM_WORDS:
            wait = True
        elif wait and token in ASM_QUALIFIERS:
            continue
        elif token == "(":
            if depth:
                depth += 1
            elif wait:
                depth, wait = 1, False
        elif token == ")" and depth:
            depth -= 1
            if not depth:
                flags[-1] = True
        else:
            wait = False
    return flags


NOT_TYPES = frozenset(
    {"return", "goto", "case", "else", "do", "sizeof", "typeof", "defined", "if", "while", "for", "switch", "default"}
)
DECL_END = frozenset({";", "=", ",", ")", "["})


def _spans(tokens: Sequence[str]) -> tuple[list[int], list[tuple[int, int]]]:
    """The function definitions of a C token list: for each token the index of the function whose parameters or body
    holds it (-1 for none), and each function's (first, last) token. A function is a `{` at file level that follows a
    `)`; directive lines are skipped."""
    owner = [-1] * len(tokens)
    spans: list[tuple[int, int]] = []
    depth, start, directive = 0, -1, False
    for n, token in enumerate(tokens):
        if directive:
            directive = token != "\n"
            continue
        if token == "#":
            directive = True
        elif token == "{":
            if depth == 0 and n and tokens[n - 1] == ")":
                close, j = 0, n - 1
                while j >= 0:
                    close += (tokens[j] == ")") - (tokens[j] == "(")
                    if not close:
                        break
                    j -= 1
                start = max(j, 0)
            depth += 1
        elif token == "}" and depth:
            depth -= 1
            if not depth and start >= 0:
                spans.append((start, n))
                owner[start : n + 1] = [len(spans) - 1] * (n + 1 - start)
                start = -1
    return owner, spans


def _member(tokens: Sequence[str], n: int) -> bool:
    """A name after `.` or `->` is a member, which has its own namespace."""
    return n > 0 and (tokens[n - 1] == "." or (n > 1 and tokens[n - 1] == ">" and tokens[n - 2] == "-"))


def _declares(tokens: Sequence[str], lo: int, n: int) -> bool:
    """True when the identifier at `n` is declared there (a parameter or a local): it follows a type word or `*`
    inside a statement or parameter list that starts with two words, and is followed by `;`, `=`, `,`, `)` or `[`."""
    if n + 1 >= len(tokens) or tokens[n + 1] not in DECL_END or _member(tokens, n):
        return False
    depth, j, comma = 0, n - 1, n
    while j > lo:
        token = tokens[j]
        if token in (")", "]"):
            depth += 1
        elif token in ("(", "["):
            if not depth:
                break
            depth -= 1
        elif not depth and token in (";", "{", "}"):
            break
        elif not depth and token == "=":
            return False
        elif not depth and token == "," and comma == n:
            comma = j
        j -= 1
    first = j + 1
    if comma != n and any(t not in ("*", "const", "volatile") for t in tokens[comma + 1 : n]):
        return False
    stop = first
    while stop < n and tokens[stop] not in (",", "=", "[") and (_is_id(tokens[stop]) or tokens[stop] == "*"):
        stop += 1
    lead = tokens[first:stop]
    if not lead or not _is_id(lead[0]) or lead[0] in NOT_TYPES:
        return False
    if comma == n:
        return stop == n and any(_is_id(t) for t in lead)
    return sum(_is_id(t) for t in lead) >= 2


Macros = dict[str, list[tuple[list[str], list[str]]]]


class _Repo:
    """Git on the project with a private index (the working tree's own index is never touched)."""

    def __init__(self, root: Path) -> None:
        git = shutil.which("git")
        if git is None:
            raise Refusal(Finding("native.missing_tool", "git is not installed", path="git"))
        self.git, self.root, self.tmp = git, root, process.scratch(root)
        self.holder = Path(tempfile.mkdtemp(dir=self.tmp, prefix="renameproof-"))
        self.env = {"GIT_INDEX_FILE": str(self.holder / "index")}

    def close(self) -> None:
        shutil.rmtree(self.holder, ignore_errors=True)

    def run(self, *args: str, stdin: bytes = b"", ok: Sequence[int] = (0,), own: bool = True) -> bytes:
        result = process.run(
            "git " + args[0], [self.git, *args], self.root, tmp=self.tmp, stdin=stdin, env=self.env if own else {}
        )
        if result.exit not in ok:
            text = result.stderr.decode(errors="replace").strip()[:200]
            raise Refusal(Finding("native.exit", f"git {' '.join(args[:3])} exit {result.exit}: {text}"))
        return result.stdout

    def blobs(self, specs: Sequence[str]) -> dict[str, bytes | None]:
        """`git cat-file --batch`: each spec's bytes, None when it names nothing."""
        out = self.run("cat-file", "--batch", stdin="".join(s + "\n" for s in specs).encode())
        found: dict[str, bytes | None] = {}
        at = 0
        for spec in specs:
            end = out.index(b"\n", at)
            header = out[at:end].split()
            at = end + 1
            if len(header) == 3 and header[1] == b"blob":
                size = int(header[2])
                found[spec] = out[at : at + size]
                at += size + 1
            else:
                found[spec] = None
        return found


class _Proof:
    def __init__(self, repo: _Repo, against: str, patches: Sequence[Path], inert: Sequence[str]) -> None:
        self.repo, self.patches, self.inert = repo, patches, tuple(inert)
        tree = repo.run("rev-parse", "--verify", "--end-of-options", f"{against}^{{tree}}", own=False)
        self.base = tree.decode().strip()  # a tree: the state after earlier steps is one too
        commit = repo.run("rev-parse", "--verify", "--end-of-options", f"{against}^{{commit}}", ok=(0, 128), own=False)
        self.commit = commit.decode().strip() or self.base
        self.reasons: list[Finding] = []
        self.map: dict[str, str] = {}  # names renamed everywhere they are visible
        self.local: dict[tuple[str, int, str], str] = {}  # (file, function, name) -> new name, for a local or parameter
        self.owner: dict[str, list[int]] = {}
        self.spans: dict[str, list[tuple[int, int]]] = {}
        self.texts: dict[str, tuple[str, str]] = {}
        self.old: dict[str, list[str]] = {}
        self.new: dict[str, list[str]] = {}

    def refuse(self, reason: str, path: str = "") -> None:
        self.reasons.append(Finding("renameproof.not_rename", reason, path=path, action="prove it with the build"))

    # ------------------------------------------------------------------ the two sides
    def load(self) -> tuple[list[str], list[str]]:
        """The after state in the private index: the patches applied in order to the base, else the working tree's
        tracked files. Returns the changed paths and the inert ones."""
        repo = self.repo
        repo.run("read-tree", self.base)
        for patch in self.patches:
            repo.run("apply", "--cached", "--whitespace=nowarn", str(patch.resolve()))
        if not self.patches:
            repo.run("add", "-u")
        rows = repo.run("diff", "--cached", "--no-renames", "--name-status", "-z", self.base).decode().split("\0")
        changed, inert = [], []
        for status, path in zip(rows[0::2], rows[1::2], strict=False):
            if any(fnmatch.fnmatch(path, glob) for glob in self.inert):
                inert.append(path)
            elif status == "M":
                changed.append(path)
            else:
                self.refuse(f"a file is added or removed (status {status}), which no rename does", path)
        return sorted(changed), sorted(inert)

    def read(self, changed: Sequence[str]) -> None:
        old = self.repo.blobs([f"{self.base}:{p}" for p in changed])
        new = self.repo.blobs([f":0:{p}" for p in changed])
        for path in changed:
            before, after = old[f"{self.base}:{path}"], new[f":0:{path}"]
            if before is None or after is None or b"\0" in before or b"\0" in after:
                self.refuse("the file is binary or unreadable as text", path)
                continue
            self.texts[path] = (before.decode("latin-1"), after.decode("latin-1"))

    # ------------------------------------------------------------------ the map
    def derive(self, line_sensitive: bool) -> None:
        pairs: dict[str, str] = {}
        sources: dict[str, str] = {}
        local: dict[tuple[str, int, str], str] = {}
        local_sources: dict[tuple[str, int, str], str] = {}
        declared: dict[tuple[str, int, str], bool] = {}
        for path, (before, after) in self.texts.items():
            c = _is_c(path)
            a_at: list[int] | None = [] if line_sensitive else None
            b_at: list[int] | None = [] if line_sensitive else None
            a, b = _tokens(before, c, a_at), _tokens(after, c, b_at)
            self.old[path], self.new[path] = a, b
            if len(a) != len(b):
                self.refuse(f"the token count changed ({len(a)} -> {len(b)})", path)
                continue
            owner, spans = _spans(a) if c else ([-1] * len(a), [])
            self.owner[path], self.spans[path] = owner, spans
            for n, (x, y) in enumerate(zip(a, b, strict=True)):
                if x == y or (x[:1] == ASM_MARK and y[:1] == ASM_MARK):
                    continue  # equal, or an assembly string judged under the finished map
                if not (_is_id(x) and _is_id(y)):
                    self.refuse(f"token {n} changed from {x[:40]!r} to {y[:40]!r}, which is not an identifier", path)
                    break
                fn = owner[n]
                if fn >= 0 and not _member(a, n):
                    key = (path, fn, x)
                    if key not in declared:
                        lo, hi = spans[fn]
                        declared[key] = any(
                            _declares(a, lo, k) for k in range(lo, hi) if a[k] == x and not _member(a, k)
                        )
                    if declared[key]:  # a local or parameter: its rename belongs to this function alone
                        if local.setdefault(key, y) != y:
                            self.refuse(f"{x} is renamed to both {local[key]} and {y} in one function", path)
                        elif local_sources.setdefault((path, fn, y), x) != x:
                            self.refuse(f"{local_sources[(path, fn, y)]} and {x} are both renamed to {y}", path)
                        continue
                if pairs.setdefault(x, y) != y:
                    self.refuse(f"{x} is renamed to both {pairs[x]} and {y}", path)
                elif sources.setdefault(y, x) != x:
                    self.refuse(f"{sources[y]} and {x} are both renamed to {y}", path)
            if a_at is not None and b_at is not None:
                lines_a = [m.start() for m in re.finditer("\n", before)]
                lines_b = [m.start() for m in re.finditer("\n", after)]
                if any(
                    bisect.bisect_left(lines_a, p) != bisect.bisect_left(lines_b, q)
                    for p, q in zip(a_at, b_at, strict=True)
                ):
                    self.refuse("a token moved to another line where the tree reads __LINE__", path)
        self.map = pairs
        self.local = local

    def pairs(self) -> list[tuple[str, str]]:
        return sorted({*self.map.items(), *((key[2], new) for key, new in self.local.items())})

    def apply(self) -> None:
        """The map over every old file must give the new file: a partial rename and an asm string both end here."""
        names = self.map

        def text(token: str) -> str:
            return ASM_MARK + _WORD.sub(lambda m: names.get(m.group(), m.group()), token[1:])

        for path, before in self.old.items():
            owner = self.owner[path]
            mapped = [
                text(t)
                if t[:1] == ASM_MARK
                else self.local.get((path, owner[n], t), names.get(t, t))
                if owner[n] >= 0 and not _member(before, n)
                else names.get(t, t)
                for n, t in enumerate(before)
            ]
            if mapped != self.new[path]:
                at = next(n for n, (x, y) in enumerate(zip(mapped, self.new[path], strict=True)) if x != y)
                self.refuse(
                    f"token {at} is {self.new[path][at][:40]!r} where the map gives {mapped[at][:40]!r}"
                    " (a name renamed in one place and left in another, or a string that is not assembly)",
                    path,
                )

    # ------------------------------------------------------------------ the rest of the tree
    def _hits(self, names: Sequence[str], rev: str | None) -> list[str]:
        pattern = self.repo.holder / "names"
        pattern.write_text("".join(n + "\n" for n in sorted(names)))
        args = ["grep", "-I", "-l", "-w", "-F", "-z", "-f", str(pattern)]
        if rev:
            out = self.repo.run(*args, rev, "--", ".", *NOT_CODE, ok=(0, 1), own=False)
        else:
            out = self.repo.run(*args, "--cached", "--", ".", *NOT_CODE, ok=(0, 1))
        files = [f for f in out.decode("latin-1").split("\0") if f]
        return [f.split(":", 1)[1] if rev else f for f in files]

    def capture(self) -> None:
        """No new name was a code token in the old tree and no old name remains one in an unchanged file."""
        keys, values = set(self.map), set(self.map.values())
        for names, rev, spec, what in (
            (values - keys, self.base, f"{self.base}:", "already a code token in the old tree (it would capture)"),
            (keys - values, None, ":0:", "still a code token after the change"),
        ):
            if not names:
                continue
            paths = [p for p in self._hits(sorted(names), rev) if rev or p not in self.texts]
            blobs = self.repo.blobs([spec + p for p in paths])
            for path in paths:
                data = blobs[spec + path]
                if data is None or b"\0" in data:
                    continue
                tokens = _tokens(data.decode("latin-1"), _is_c(path))
                found = names & set(tokens)
                if found and rev:
                    found = self.visible(tokens, found, keys) if _is_c(path) else found
                if found:
                    self.refuse(f"{sorted(found)[0]} is {what}", path)

    def visible(self, tokens: list[str], names: set[str], keys: set[str]) -> set[str]:
        """The new names a file uses where a renamed global could be captured: a use that is not a member and not in
        a function which declares the name itself and uses no renamed global is another scope's name."""
        owner, spans = _spans(tokens)
        free = self.directive_names(keys) if keys else set()
        clash: set[str] = set()
        inside: dict[int, bool] = {}
        for n, token in enumerate(tokens):
            if token not in names or token in clash or _member(tokens, n):
                continue
            fn = owner[n]
            if fn < 0 or keys & free:
                clash.add(token)
                continue
            lo, hi = spans[fn]
            if fn not in inside:
                inside[fn] = bool(keys & set(tokens[lo : hi + 1]))
            if inside[fn] or not any(
                _declares(tokens, lo, k) for k in range(lo, hi) if tokens[k] == token and not _member(tokens, k)
            ):
                clash.add(token)
        return clash

    def directive_names(self, names: set[str]) -> set[str]:
        """The names among `names` that a preprocessor directive of the old tree uses as a macro, in a macro body or
        in a condition (not as a macro parameter): inside a function they could mean the local."""
        if not names:
            return set()
        alt = "|".join(re.escape(n) for n in sorted(names))
        word = rf"\<({alt})\>"
        args = ["grep", "-I", "-l", "-z", "-E", "-e", rf"^[[:space:]]*#.*{word}", "-e", rf"{word}.*\\[[:space:]]*$"]
        out = self.repo.run(*args, self.base, "--", ".", ok=(0, 1), own=False)
        paths = [f.split(":", 1)[1] for f in out.decode("latin-1").split("\0") if f]
        blobs = self.repo.blobs([f"{self.base}:{p}" for p in paths])
        used: set[str] = set()
        for path in paths:
            if not _is_c(path):
                continue
            tokens = _tokens((blobs[f"{self.base}:{path}"] or b"").decode("latin-1"), True)
            n = 0
            while n < len(tokens):
                if tokens[n] != "#":
                    n += 1
                    continue
                end = tokens.index("\n", n) if "\n" in tokens[n:] else len(tokens)
                row = tokens[n + 1 : end]
                params: set[str] = set()
                if row[:1] == ["define"] and row[2:3] == ["("] and ")" in row:
                    params = set(row[3 : row.index(")")])
                    row = row[:2] + row[row.index(")") + 1 :]
                used |= (set(row) - params) & names
                n = end + 1
        return used

    def scoped(self) -> None:
        """A local or parameter rename meets only its own function: the new name must not be used there, and neither
        name may be a macro or a macro's free name anywhere (a macro expands inside the function)."""
        if not self.local:
            return
        values = set(self.map.values())
        for (path, fn, old), new in sorted(self.local.items()):
            tokens = self.old[path]
            lo, hi = self.spans[path][fn]
            used = {t for k, t in enumerate(tokens[lo : hi + 1], lo) if not _member(tokens, k)}
            swapped = any(key[:2] == (path, fn) and key[2] == new for key in self.local)
            if new in used and not swapped:
                self.refuse(f"{new} is already used in the function that renames {old} to it", path)
            elif new in values:
                self.refuse(f"{new} is also the new name of a renamed global", path)
        names = {key[2] for key in self.local} | set(self.local.values())
        for name in sorted(self.directive_names(names)):
            self.refuse(f"{name} is a macro name or appears in a preprocessor directive, where it could mean the local")

    def builtins(self) -> None:
        """A name the compiler knows as a builtin must not enter or leave the map, unless the project's flags turn
        builtins off."""
        config = self.repo.blobs([f"{self.base}:config.toml"])[f"{self.base}:config.toml"] or b""
        if b"-fno-builtin" in config:
            return
        every = {name for pair in self.pairs() for name in pair}
        crossing = sorted(n for n in every if n in BUILTINS or n.startswith("__builtin_"))
        if crossing:
            self.refuse(f"{', '.join(crossing[:5])} is a name the compiler expands itself, so a rename changes code")

    def line_sensitive(self) -> bool:
        return bool(self._hits(["__LINE__"], self.base))

    # ------------------------------------------------------------------ macros
    def calls(self) -> dict[tuple[str, int], str]:
        """(macro, argument position) of every call in a changed C file that holds a renamed name as an argument,
        with the file it is in; a paste of a renamed name refuses here."""
        keys = {old for old, _ in self.pairs()}
        found: dict[tuple[str, int], str] = {}
        for path, tokens in self.old.items():
            if not _is_c(path) or not keys & set(tokens):
                continue
            stack: list[tuple[str | None, int]] = []
            for n, token in enumerate(tokens):
                if token == "\n":
                    stack.clear()
                elif token == "(":
                    before = tokens[n - 1] if n else ""
                    stack.append((before if _is_id(before) and before not in NOT_MACROS else None, 0))
                elif token == ")":
                    if stack:
                        stack.pop()
                elif token == "," and stack:
                    stack[-1] = (stack[-1][0], stack[-1][1] + 1)
                elif token in keys:
                    if tokens[max(n - 2, 0) : n] == ["#", "#"] or tokens[n + 1 : n + 3] == ["#", "#"]:
                        self.refuse(f"{token} is pasted with ##, which the map does not follow", path)
                    for name, index in stack:
                        if name is not None:
                            found.setdefault((name, index), path)
        return found

    def definitions(self, names: Sequence[str], known: Macros) -> None:
        """The `#define NAME(` rows of the old tree for the names not looked up yet; a name with none is a function."""
        todo = sorted(set(names) - set(known))
        for name in todo:
            known[name] = []
        if not todo:
            return
        pattern = r"^[[:space:]]*#[[:space:]]*define[[:space:]]+(" + "|".join(todo) + r")[[:space:]]*\("
        out = self.repo.run("grep", "-I", "-l", "-z", "-E", pattern, self.base, "--", ".", ok=(0, 1), own=False)
        paths = [f.split(":", 1)[1] for f in out.decode("latin-1").split("\0") if f]
        blobs = self.repo.blobs([f"{self.base}:{p}" for p in paths])
        wanted = set(todo)
        for path in paths:
            tokens = _tokens((blobs[f"{self.base}:{path}"] or b"").decode("latin-1"), True)
            n = 0
            while n + 4 < len(tokens):
                if tokens[n : n + 2] == ["#", "define"] and tokens[n + 2] in wanted and tokens[n + 3] == "(":
                    end = tokens.index("\n", n) if "\n" in tokens[n:] else len(tokens)
                    close = tokens.index(")", n + 3)
                    params = [t for t in tokens[n + 4 : close] if t != ","]
                    known[tokens[n + 2]].append((params, tokens[close + 1 : end]))
                    n = end
                n += 1

    def safe(self, name: str, index: int, known: Macros, depth: int) -> bool:
        """True when argument `index` of the macro is expanded as itself or stringified only into assembly text."""
        self.definitions([name], known)
        rows = known[name]
        if not rows:
            return True  # a function: its argument is code
        if depth > DEPTH:
            return False
        for params, body in rows:
            variadic = bool(params) and params[-1] == "..."
            if index >= len(params) and not variadic:
                return False
            param = "__VA_ARGS__" if variadic and index >= len(params) - 1 else params[index]
            flags = _asm_flags(body)
            stack: list[tuple[str | None, int]] = []
            for n, token in enumerate(body):
                if token == "(":
                    before = body[n - 1] if n else ""
                    stack.append((before if _is_id(before) and before not in NOT_MACROS else None, 0))
                elif token == ")":
                    if stack:
                        stack.pop()
                elif token == "," and stack:
                    stack[-1] = (stack[-1][0], stack[-1][1] + 1)
                elif token == param:
                    if body[max(n - 2, 0) : n] == ["#", "#"] or body[n + 1 : n + 3] == ["#", "#"]:
                        return False
                    if n and body[n - 1] == "#" and not flags[n]:
                        return False
                    if any(c is not None and not self.safe(c, i, known, depth + 1) for c, i in stack):
                        return False
        return True

    def stringified(self) -> None:
        sites = self.calls()
        known: Macros = {}
        self.definitions(sorted({name for name, _ in sites}), known)
        for (name, index), path in sorted(sites.items()):
            if not self.safe(name, index, known, 0):
                self.refuse(
                    f"{name} takes a renamed name as argument {index} and pastes or stringifies it outside"
                    " inline assembly, so the rename reaches a string or a new identifier",
                    path,
                )


def _check(
    repo: _Repo, against: str, patches: Sequence[Path], inert: Sequence[str]
) -> tuple[_Proof, list[str], list[str]]:
    proof = _Proof(repo, against, patches, inert)
    changed, skipped = proof.load()
    proof.read(changed)
    if not proof.reasons:
        proof.derive(proof.line_sensitive())
    if not proof.reasons:
        proof.apply()
    if not proof.reasons:
        proof.builtins()
        proof.capture()
        proof.scoped()
        proof.stringified()
    if not changed and not skipped and not proof.reasons:
        proof.refuse("the change touches no file the build reads")
    return proof, changed, skipped


def _refusal(proof: _Proof) -> Refusal:
    shown = proof.reasons[:SHOWN]
    if len(proof.reasons) > SHOWN:
        shown.append(Finding("renameproof.not_rename", f"{len(proof.reasons) - SHOWN} more reasons"))
    return Refusal(*shown)


def prove(root: Path, against: str, patches: Sequence[Path], inert: Sequence[str], steps: bool = False) -> Json:
    """The proof of one change (patches over `against`, else the working tree over it); a Refusal names why it is
    not a pure rename. With `steps` each patch is its own change over the state the patches before it leave, and the
    result lists each one's verdict: the batches that fall back to the build are exactly the refused rows."""
    repo = _Repo(root)
    try:
        if steps:
            return _steps(repo, against, patches, inert)
        proof, changed, skipped = _check(repo, against, patches, inert)
        if proof.reasons:
            raise _refusal(proof)
        names = proof.pairs()
        return {
            "against": against,
            "base": proof.commit,
            "tree": repo.run("write-tree").decode().strip(),
            "patches": [str(p) for p in patches],
            "files": len(changed),
            "renames": len(names),
            "map": [list(pair) for pair in names],
            "inert": skipped,
        }
    finally:
        repo.close()


def _steps(repo: _Repo, against: str, patches: Sequence[Path], inert: Sequence[str]) -> Json:
    """Each patch over the state every patch before it leaves (the whole stack, as it will land): its verdict says
    whether the build may skip it; a refused patch still moves the state on, and one that does not apply stops it."""
    rows: list[dict[str, Any]] = []
    state, base, files, renames = against, "", 0, set()
    for patch in patches:
        step = _Repo(repo.root)
        row: dict[str, Any] = {"patch": str(patch), "pure": False, "files": 0, "renames": 0, "inert": [], "reasons": []}
        try:
            proof, changed, skipped = _check(step, state, [patch], inert)
            base = base or proof.commit
            row.update(files=len(changed), renames=len(proof.pairs()), inert=skipped)
            row["reasons"] = [f.reason for f in proof.reasons[:SHOWN]]
            if not proof.reasons:
                row["pure"] = True
                files += len(changed)
                renames.update(proof.pairs())
            state = step.run("write-tree").decode().strip()
        except Refusal as refusal:
            row["reasons"] = [f.reason for f in refusal.findings[:SHOWN]]
        finally:
            step.close()
        rows.append(row)
    return {
        "against": against,
        "base": base,
        "tree": repo.run("rev-parse", "--verify", f"{state}^{{tree}}", own=False).decode().strip(),
        "patches": [str(p) for p in patches],
        "files": files,
        "renames": len(renames),
        "map": [list(pair) for pair in sorted(renames)],
        "inert": [],
        "steps": rows,
    }


def rename_command(params: Json) -> Json:
    """The command: no host or ROM is read, only git and text."""
    patches = [Path(p) for p in params.get("patch") or ()]
    return prove(
        Path(params["project"] or "."),
        params["against"],
        patches,
        list(params.get("inert") or ()),
        bool(params.get("steps")),
    )
