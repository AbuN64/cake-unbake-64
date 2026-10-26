"""Widen: port the C one version matches to the versions whose ROM holds the same code (`compare FILE --widen V,..`).

A function exact in the reference version (`--from`, default the project's names_from) often has its C in a conditional
that compiles it only there, while the other versions build an INCLUDE_ASM stand-in or a NON_MATCHING draft. Where V's
ROM holds the same code up to the addresses the link writes, the port is mechanical, and this does it in one step:

1. guards: every conditional that decides a piece of the function (its definition, draft or stand-in), in FILE and in
   every other source file that holds a piece of it, is rewritten so that V takes the branch the reference takes; the
   reference and every other version take the branches they took. A condition that does not name V gets the smallest
   edit (`defined(REF)` becomes `defined(REF) || defined(V)`); one that does is written again from its truth table over
   the versions and NON_MATCHING. Every other function those conditionals enclose comes along and is measured too.
2. data symbols: a name the ported C spells that V's tables do not place is read from the code, as N64DecompTools'
   trial.counterpart does: where the reference's ROM words build the name's address (a `lui` and its low half, or a
   `jal`), V's paired words build V's. The pairing is word for word when the two functions are the same length, else
   by an alignment of their opcodes. A name no word reads, or that reads as two addresses, refuses. The row goes to
   V's aliases.txt when V already names that address, else to its symbol_addrs.txt (missing.line).
3. `sync --place` for the ported functions that have read-only data of their own, then sync.
4. each ported function is scored in each V (`compare.score_functions`); one that is not exact puts every file back
   (and syncs again) and refuses.
The result names what was written and the diff (`# MSG:` first line, ready for ROUND-INBOX). With `check`: the edits and
symbols it would make, nothing written.
"""

from __future__ import annotations

import difflib
import re
import struct
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from unbake import api, process, publish, store
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Refusal, macro

_WORD = re.compile(r"\b[A-Za-z_]\w*\b")
_DEFINED = re.compile(r"\bdefined\s*\(\s*(\w+)\s*\)|\bdefined\s+(\w+)")
_ALIAS = re.compile(r"(?m)^[ \t]*#[ \t]*define[ \t]+(\w+)[ \t]+(\w+)[ \t]*$")
EXACT = 1.0  # Proof.score: the matched fraction, 1.0 iff exact
MSG_MAX = 72


# ---------------------------------------------------------------- conditions
def _condition(line: str) -> tuple[str, str]:
    """(directive word, condition text) of a conditional line; `#ifdef X` reads as `defined(X)`."""
    found = re.match(r"[ \t]*#[ \t]*(\w+)[ \t]*(.*)", line.rstrip("\n"))
    word, rest = (found[1], found[2]) if found else ("", "")
    rest = re.sub(r"/\*.*?\*/|//.*", "", rest).strip()
    if word == "ifdef":
        return word, f"defined({rest})"
    if word == "ifndef":
        return word, f"!defined({rest})"
    return word, rest


def evaluator(condition: str, known: Collection[str]) -> Callable[[Collection[str]], bool] | None:
    """A function of the defined macros for a condition over KNOWN macros only (bare or under `defined`), None when it
    reads anything else (another macro, a value comparison)."""
    names: list[str] = []

    def slot(name: str) -> str:
        if name not in known:
            raise KeyError(name)
        names.append(name)
        return f" _d[{len(names) - 1}] "

    try:
        expr = _DEFINED.sub(lambda m: slot(m[1] or m[2]), condition)
        expr = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: m[0] if m[0] == "_d" else slot(m[0]), expr)
    except KeyError:
        return None
    if re.search(r"[^\s()!&|0-9_d\[\]]", expr):
        return None
    expr = expr.replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"!(?!=)", " not ", expr)
    try:
        code = compile(expr.strip() or "0", "<condition>", "eval")
    except SyntaxError:
        return None

    def run(defined: Collection[str]) -> bool:
        return bool(eval(code, {"__builtins__": {}}, {"_d": [n in defined for n in names]}))

    return run


@dataclass(frozen=True)
class Universe:
    """The versions a condition is judged over, the reference and the versions it widens to."""

    versions: tuple[str, ...]
    ref: str
    widen: tuple[str, ...]
    non_matching: str = "NON_MATCHING"

    @property
    def known(self) -> set[str]:
        return {macro(v) for v in self.versions} | {self.non_matching}

    def cases(self) -> list[tuple[str, bool]]:
        return [(v, nm) for v in self.versions for nm in (False, True)]

    def env(self, v: str, nm: bool) -> set[str]:
        return {macro(v)} | ({self.non_matching} if nm else set())

    def as_ref(self, v: str) -> str:
        return self.ref if v in self.widen else v


def _paren(expr: str) -> str:
    return f"({expr})" if "||" in expr or "&&" in expr else expr


def _set_expr(held: Sequence[str], versions: Sequence[str]) -> str:
    if len(held) == len(versions):
        return "1"
    if not held:
        return "0"
    pos = " || ".join(f"defined({macro(v)})" for v in held)
    out = [v for v in versions if v not in held]
    neg = "!" + (
        f"defined({macro(out[0])})" if len(out) == 1 else "(" + " || ".join(f"defined({macro(v)})" for v in out) + ")"
    )
    return pos if len(held) <= len(out) else neg


def render(table: Mapping[tuple[str, bool], bool], u: Universe) -> str:
    """The condition of a truth table over (version, NON_MATCHING), as plain `defined()` terms."""
    vs = list(u.versions)
    on = [v for v in vs if table[(v, True)]]
    off = [v for v in vs if table[(v, False)]]
    nm = f"defined({u.non_matching})"
    if on == off:
        return _set_expr(off, vs)
    if len(on) == len(vs):
        return nm if not off else f"{nm} || {_paren(_set_expr(off, vs))}"
    if not off:
        return f"{nm} && {_paren(_set_expr(on, vs))}"
    if not on:
        return f"!{nm} && {_paren(_set_expr(off, vs))}"
    if len(off) == len(vs):
        return f"!{nm} || {_paren(_set_expr(on, vs))}"
    return f"({nm} && {_paren(_set_expr(on, vs))}) || (!{nm} && {_paren(_set_expr(off, vs))})"


def _widened_text(condition: str, u: Universe) -> str:
    """`defined(REF)` -> `defined(REF) || defined(V)..`, parenthesized unless it is an operand of `||` or alone."""
    wide = " || ".join(f"defined({macro(v)})" for v in (u.ref, *u.widen))
    ref = re.escape(macro(u.ref))
    pattern = re.compile(rf"\bdefined\s*\(\s*{ref}\s*\)|\bdefined\s+{ref}\b")
    out, last = [], 0
    for found in pattern.finditer(condition):
        before = condition[: found.start()].rstrip()
        after = condition[found.end() :].lstrip()
        bare = (not before or before.endswith(("(", "||"))) and (not after or after.startswith((")", "||")))
        out += [condition[last : found.start()], wide if bare else f"({wide})"]
        last = found.end()
    return "".join(out) + condition[last:]


def rewrite(line: str, u: Universe) -> str | None:
    """The conditional line with V taking the reference's branch, the line itself when nothing changes for any version,
    None when the condition names V beside a macro this cannot judge."""
    word, condition = _condition(line)
    if word not in ("if", "elif", "ifdef", "ifndef"):
        return line
    words = set(_WORD.findall(condition))
    v_named = bool({macro(v) for v in u.widen} & words)
    indent = re.match(r"[ \t]*", line)[0]  # type: ignore[index]
    head = "elif" if word == "elif" else "if"
    newline = "\n" if line.endswith("\n") else ""
    judge = evaluator(condition, u.known)
    if judge is None:  # another macro: `defined(REF)` -> `defined(REF) || defined(V)` is sound while V is not named
        if v_named:
            return None
        if macro(u.ref) not in words:
            return line
        return f"{indent}#{head} {_widened_text(condition, u)}{newline}"
    want = {(v, nm): judge(u.env(u.as_ref(v), nm)) for v, nm in u.cases()}
    if all(want[c] == judge(u.env(*c)) for c in u.cases()):
        return line
    if not v_named:
        small = _widened_text(condition, u)
        check = evaluator(small, u.known)
        if check is not None and all(check(u.env(*c)) == want[c] for c in u.cases()):
            return f"{indent}#{head} {small}{newline}"
    return f"{indent}#{head} {render(want, u)}{newline}"


# ---------------------------------------------------------------- pieces of a function in a file
@dataclass
class Conditional:
    lines: list[int]  # the #if line, then each #elif / #else line
    end: int  # the #endif line


def conditionals(lines: Sequence[str]) -> list[Conditional]:
    stack: list[Conditional] = []
    out: list[Conditional] = []
    for number, line in enumerate(lines):
        word = _condition(line)[0] if re.match(r"[ \t]*#", line) else ""
        if word in ("if", "ifdef", "ifndef"):
            stack.append(Conditional([number], -1))
        elif word in ("elif", "else") and stack:
            stack[-1].lines.append(number)
        elif word == "endif" and stack:
            done = stack.pop()
            done.end = number
            out.append(done)
    return out


def _stand_in(name: str) -> re.Pattern[str]:
    return re.compile(rf"\bINCLUDE_ASM\s*\(\s*\"[^\"]*\"\s*,\s*{re.escape(name)}\s*\)")


def pieces(text: str, names: Collection[str]) -> dict[str, list[tuple[int, int]]]:
    """name -> (first line, last line) of each definition (under its name or a `#define ALIAS NAME` spelling) and each
    INCLUDE_ASM stand-in of NAMES in TEXT."""
    aliases = {a: n for a, n in _ALIAS.findall(text) if n in names}
    out: dict[str, list[tuple[int, int]]] = {}
    for item in publish.items(text):
        body = text[item.start : item.end]
        span = (text.count("\n", 0, item.start), text.count("\n", 0, max(item.end - 1, item.start)))
        if item.kind == "function" and (item.name in names or item.name in aliases):
            out.setdefault(aliases.get(item.name, item.name), []).append(span)
        elif item.kind != "function":
            for n in names:
                if _stand_in(n).search(body):
                    out.setdefault(n, []).append(span)
    return out


def enclosed(text: str) -> dict[str, list[tuple[int, int]]]:
    """name -> spans of every function definition and INCLUDE_ASM stand-in of TEXT."""
    out: dict[str, list[tuple[int, int]]] = {}
    for item in publish.items(text):
        span = (text.count("\n", 0, item.start), text.count("\n", 0, max(item.end - 1, item.start)))
        if item.kind == "function":
            out.setdefault(item.name, []).append(span)
        else:
            for found in re.finditer(r"\bINCLUDE_ASM\s*\(\s*\"[^\"]*\"\s*,\s*(\w+)\s*\)", text[item.start : item.end]):
                out.setdefault(found[1], []).append(span)
    return out


@dataclass
class Edit:
    path: str
    before: str
    after: str
    names: set[str] = field(default_factory=set)


def widen_text(text: str, names: Collection[str], u: Universe) -> tuple[str, set[str], list[str]]:
    """(new text, every function name the rewritten conditionals decide, the lines it could not judge)."""
    lines = text.splitlines(keepends=True)
    conds = conditionals(lines)
    spans = [s for found in pieces(text, names).values() for s in found]
    chosen = [c for c in conds if any(c.lines[0] < a and b < c.end for a, b in spans)]
    unjudged: list[str] = []
    changed: list[Conditional] = []
    for c in chosen:
        hit = False
        for number in c.lines:
            new = rewrite(lines[number], u)
            if new is None:
                unjudged.append(f"line {number + 1}: {lines[number].strip()}")
            elif new != lines[number]:
                lines[number], hit = new, True
        if hit:
            changed.append(c)
    decided = {
        n for n, found in enclosed(text).items() for a, b in found for c in changed if c.lines[0] < a and b < c.end
    }
    drop: set[int] = set()
    for c in changed:
        taken = _taken(lines, c, u)
        if taken is None:
            continue
        bounds = [*c.lines, c.end]
        for k in range(len(c.lines)):
            if k != taken:
                drop.update(range(bounds[k], bounds[k + 1]))
        drop.update({c.end} if taken < 0 else {bounds[taken], c.end})  # -1: no version builds any branch
    return "".join(line for n, line in enumerate(lines) if n not in drop), decided, unjudged


def _taken(lines: Sequence[str], c: Conditional, u: Universe) -> int | None:
    """The branch (index into C.lines; -1 for none) every version takes with and without NON_MATCHING, None when they
    differ or a condition cannot be judged: such a conditional is only its branch's lines (or nothing)."""
    judges = []
    for number in c.lines:
        word, condition = _condition(lines[number])
        judge = (lambda _d: True) if word == "else" else evaluator(condition, u.known)
        if judge is None:
            return None
        judges.append(judge)
    picks = {next((k for k, j in enumerate(judges) if j(u.env(v, nm))), -1) for v, nm in u.cases()}
    return picks.pop() if len(picks) == 1 else None


# ---------------------------------------------------------------- reading symbols from the code (trial.counterpart)
_LOW = frozenset({0x09, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x37, 0x31, 0x35}) | frozenset(
    {0x28, 0x29, 0x2A, 0x2B, 0x2E, 0x3F, 0x39, 0x3D}
)
_WRITES_RT = frozenset(
    {0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x37}
)


def _writes(w: int) -> int | None:
    op = w >> 26
    if op == 0:
        return (w >> 11) & 31 or None
    if op in _WRITES_RT:
        return (w >> 16) & 31
    if op == 0x03:
        return 31
    return None


def _low(w: int) -> int:
    return (w & 0xFFFF) - (0x10000 if w & 0x8000 else 0)


def addresses(words: Sequence[int]) -> list[tuple[int, int, int]]:
    """[(high, low, address)]: a `lui` and a later instruction that adds a signed low half to its register before
    anything else writes it (trial.addresses)."""
    found = []
    for at, word in enumerate(words):
        if word >> 26 != 0x0F:
            continue
        register = (word >> 16) & 31
        for later in range(at + 1, len(words)):
            w = words[later]
            if w >> 26 in _LOW and (w >> 21) & 31 == register:
                found.append((at, later, (((word & 0xFFFF) << 16) + _low(w)) & 0xFFFFFFFF))
            if _writes(w) == register:
                break
    return found


def _key(w: int) -> tuple[int, int]:
    return (w >> 26, w & 0x3F if w >> 26 == 0 else 0)


def pair(ref: Sequence[int], other: Sequence[int]) -> dict[int, int]:
    """ref word index -> OTHER's word index of the same instruction: word for word at equal lengths, else by opcode
    alignment (the role trial's aligner plays)."""
    if len(ref) == len(other):
        return {i: i for i in range(len(ref)) if _key(ref[i]) == _key(other[i])}
    out: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(None, [_key(w) for w in ref], [_key(w) for w in other], autojunk=False)
    for a, b, n in matcher.get_matching_blocks():
        out.update({a + k: b + k for k in range(n)})
    return out


def counterpart(
    address: int, extent: int, ref: Sequence[int], ref_vram: int, other: Sequence[int], other_vram: int
) -> set[int]:
    """The addresses OTHER builds where REF builds one inside [ADDRESS, ADDRESS + EXTENT) (a field of the symbol reads
    as the symbol), or jumps to ADDRESS; each less the same offset, so a reading of the symbol's own address."""
    paired = pair(ref, other)
    found: set[int] = set()
    for high, low, built in addresses(ref):
        if address <= built < address + max(extent, 1) and high in paired and low in paired:
            h, lo = other[paired[high]], other[paired[low]]
            if h >> 26 == 0x0F and lo >> 26 == ref[low] >> 26:
                found.add(((((h & 0xFFFF) << 16) + _low(lo)) - (built - address)) & 0xFFFFFFFF)
    for i, w in enumerate(ref):
        jumps = w >> 26 in (0x02, 0x03) and ((ref_vram + 4 * i) & 0xF0000000) | ((w & 0x3FFFFFF) << 2) == address
        if jumps and i in paired and other[paired[i]] >> 26 == w >> 26:
            o = other[paired[i]]
            found.add(((other_vram + 4 * paired[i]) & 0xF0000000) | ((o & 0x3FFFFFF) << 2))
    return found


# ---------------------------------------------------------------- the command
Table = Mapping[str, tuple[int, int | None, int | None, str]]


def _words(image: bytes, table: Table, name: str) -> tuple[list[int], int] | None:
    """(the ROM words, the vram) of the function NAME of a version's table, None when it has no row with a ROM."""
    row = table.get(name)
    if row is None or row[1] is None:
        return None
    vram, rom, size = row[0], row[1], row[2]
    if not size:
        later = sorted(r[1] for r in table.values() if r[3] == "func" and r[1] is not None and r[1] > rom)
        size = (later[0] - rom) if later else 0
    size = min(size, max(len(image) - rom, 0)) // 4 * 4
    return list(struct.unpack_from(f">{size // 4}I", image, rom)), vram


def _extent(table: Table, name: str) -> int:
    vram, _, size, _ = table[name]
    if size:
        return size
    later = [r[0] for n, r in table.items() if r[0] > vram]
    return min(min(later) - vram, 0x10000) if later else 4


def derive(
    names: Collection[str], spelled: Collection[str], ref: tuple[Table, bytes], other: tuple[Table, bytes]
) -> tuple[dict[str, int], list[str]]:
    """({symbol: address in OTHER}, refusals) for each name the ported C spells that the reference places and OTHER's
    tables do not, read from the words of the ported functions."""
    (rt, rimg), (ot, oimg) = ref, other
    missing = sorted(n for n in spelled if n in rt and n not in ot and n not in names)
    found: dict[str, set[int]] = {m: set() for m in missing}
    for fn in sorted(names):
        here, there = _words(rimg, rt, fn), _words(oimg, ot, fn)
        if here is None or there is None:
            continue
        for m in missing:
            found[m] |= counterpart(rt[m][0], _extent(rt, m), here[0], here[1], there[0], there[1])
    derived = {m: min(v) for m, v in found.items() if len(v) == 1}
    refused = [
        f"{m}: "
        + ("read as " + ", ".join(f"0x{a:08X}" for a in sorted(v)) if v else "no ported word builds its address")
        for m, v in found.items()
        if len(v) != 1
    ]
    return derived, refused


def files_spelling(config: Config, names: Collection[str], where: Sequence[str] = ("src",)) -> list[str]:
    """The tracked files under WHERE that spell any of NAMES as a word."""
    if not names:
        return []
    try:
        out = process.git(config, "grep", "-lw", *(a for n in sorted(names) for a in ("-e", n)), "--", *where)
    except Refusal:  # git grep exits 1 when nothing matches
        return []
    return sorted(out.stdout.decode().split())


def write_diff(config: Config, written: Collection[str], head: str, target: Path) -> Path:
    """The diff of WRITTEN (a file the command created included) with the `# MSG:` HEAD line, at TARGET."""
    paths = sorted(written)
    new = process.git(config, "ls-files", "--others", "--exclude-standard", "--", *paths).stdout.decode().split()
    if new:  # a file sync created (a split sibling) is in the diff too
        process.git(config, "add", "--intent-to-add", "--", *new)
    diff = process.git(config, "diff", "--", *paths).stdout.decode()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"# MSG: {head}"[: MSG_MAX + len("# MSG: ")] + "\n" + diff)
    return target


def _universe(config: Config, params: Json) -> Universe:
    versions = tuple(config.project.versions)
    ref = params.get("from") or config.project.names_from
    widen = tuple(v.strip() for v in str(params["widen"]).split(",") if v.strip())
    bad = [v for v in (ref, *widen) if v not in versions]
    if bad or not widen or ref in widen:
        raise Refusal(
            Finding("widen.request", f"--widen takes versions other than --from {ref} among {', '.join(versions)}")
        )
    return Universe(versions, ref, widen, config.project.non_matching or "NON_MATCHING")


def plan(config: Config, path: str, function: str | None, u: Universe) -> tuple[list[Edit], set[str], list[str]]:
    """(the source edits, the ported functions, refusals): FILE's functions decided by a conditional that names the
    reference (or FUNCTION), and every function the rewritten conditionals also decide, in every file holding one."""
    root = config.project.root
    text = (root / path).read_text(encoding="utf-8")
    if function:
        names = {function}
    else:
        lines = text.splitlines(keepends=True)
        ref_conds = [
            c
            for c in conditionals(lines)
            if any(macro(u.ref) in _WORD.findall(_condition(lines[n])[1]) for n in c.lines)
        ]
        names = {
            n
            for n, found in enclosed(text).items()
            for a, b in found
            for c in ref_conds
            if c.lines[0] < a and b < c.end
        }
    if not names:
        raise Refusal(Finding("widen.none", f"{path}: no function sits in a conditional that names {macro(u.ref)}"))
    edits: dict[str, Edit] = {}
    refused: list[str] = []
    while True:
        grown = False
        for file in sorted({path, *files_spelling(config, names)}):
            before = edits[file].before if file in edits else (root / file).read_text(encoding="utf-8")
            after, decided, unjudged = widen_text(before, names, u)
            refused += [f"{file} {x}" for x in unjudged if f"{file} {x}" not in refused]
            if after != before or file in edits:
                edits[file] = Edit(file, before, after, decided)
            if decided - names:
                names |= decided
                grown = True
        if not grown:
            break
    return [e for e in edits.values() if e.after != e.before], names, refused


def _diff_text(edits: Sequence[Edit]) -> str:
    return "".join(
        "".join(difflib.unified_diff(e.before.splitlines(True), e.after.splitlines(True), f"a/{e.path}", f"b/{e.path}"))
        for e in edits
    )


def _spelled(edits: Sequence[Edit], names: Collection[str]) -> set[str]:
    out: set[str] = set()
    for e in edits:
        for item in publish.items(e.after):
            if item.kind == "function" and item.name in names:
                out |= set(_WORD.findall(e.after[item.start : item.end]))
    return out


def run(config: Config, params: Json) -> Json:
    from unbake import compare, missing, place, repo, symbols, tree

    u = _universe(config, params)
    root = config.project.root
    try:
        path = str(Path(params["file"]).resolve().relative_to(root.resolve()))
    except ValueError:
        raise Refusal(Finding("widen.request", f"{params['file']} is not a file of the project {root}")) from None
    edits, names, refused = plan(config, path, params.get("function"), u)
    if refused:
        raise Refusal(*[Finding("widen.condition", f"cannot judge {r}") for r in refused])
    if not edits:
        raise Refusal(Finding("widen.none", f"{path}: every version in {', '.join(u.widen)} already builds this C"))
    tables = {v: api.symbols(config, v) for v in (u.ref, *u.widen)}
    images = {v: configuration.rom(config.project, v).read_bytes() for v in (u.ref, *u.widen)}
    absent = [
        f"{v} has no row named {n} (its code there has another name: rename the row first)"
        for v in u.widen
        for n in sorted(names)
        if n not in tables[v] and n in tables[u.ref]
    ]
    if absent:
        raise Refusal(*[Finding("widen.row", a) for a in absent])
    # a function with no row in the reference (a static helper) is built inside the object the reference matches
    carried = sorted(n for n in names if n not in tables[u.ref])
    names = {n for n in names if n in tables[u.ref]}
    spelled = _spelled(edits, names | set(carried))
    derived: dict[str, dict[str, int]] = {}
    bad: list[str] = []
    for v in u.widen:
        derived[v], why = derive(names, spelled, (tables[u.ref], images[u.ref]), (tables[v], images[v]))
        bad += [f"{v} {w}" for w in why]
    shown = {
        "from": u.ref,
        "widen": list(u.widen),
        "ported": sorted(names),
        "carried": carried,
        "files": [e.path for e in edits],
        "derived": {v: {s: f"0x{a:08X}" for s, a in d.items()} for v, d in derived.items()},
    }
    if bad:
        raise Refusal(*[Finding("widen.symbol", b) for b in bad])
    if params.get("check"):
        return {**shown, "written": [], "diff": _diff_text(edits)}
    rows: dict[str, list[Json]] = {}
    for v, found in derived.items():
        by_vram = {r[0] for r in tables[v].values()}
        for s, a in sorted(found.items()):
            rom, _ = missing.rom_of_ram(config, v, a)
            rows.setdefault(v, []).append({"name": s, "address": a, "rom": rom, "aliased": a in by_vram})
    touched = {e.path for e in edits} | {symbols.path(v) for v in rows} | {symbols.aliases_path(v) for v in rows}
    written: set[str] = set()
    synced = False
    with store.journal() as undone:
        try:
            for e in edits:
                store.write(root / e.path, e.after.encode())
            for v, rs in rows.items():
                missing.append(
                    config,
                    symbols.aliases_path(v),
                    [r for r in rs if r["aliased"]],
                    "name",
                    lambda r: missing.line(
                        {"class": "b", "kind": "data", "proposed": r["name"], "address": r["address"]}
                    ),
                )
                missing.append(
                    config,
                    symbols.path(v),
                    [r for r in rs if not r["aliased"]],
                    "name",
                    lambda r: missing.line(
                        {"class": "b", "kind": "data", "proposed": r["name"], "address": r["address"], "rom": r["rom"]}
                    ),
                )
            pools: list[str] = []
            current = tree.capture(config)
            for n in sorted(names):
                try:
                    place.plan(config, current, [n])
                    pools.append(n)
                except Refusal as refusal:
                    if not all(f.key == "place.none" for f in refusal.findings):
                        raise
            synced = True
            done = repo.sync(config, {"place": pools} if pools else {})
            written |= set(done.get("written", ()))
            scores: dict[str, dict[str, float]] = {}
            for v in u.widen:
                for e in edits:
                    held = [n for n in sorted(names) if n in tables[v] and pieces(e.after, [n]).get(n)]
                    for n in held:
                        try:
                            got = compare.score_functions(config, e.path, [n], v).get(n, 0.0)
                        except Refusal:
                            continue  # this file no longer builds it in V: the other file holding it does
                        scores.setdefault(n, {})[v] = max(got, scores.get(n, {}).get(v, 0.0))
            for n in sorted(names):
                for v in u.widen:
                    if n in tables[v]:
                        scores.setdefault(n, {}).setdefault(v, 0.0)
            short = [f"{n} {v} {s:.2f}" for n, by in sorted(scores.items()) for v, s in sorted(by.items()) if s < EXACT]
            if short:
                raise Refusal(*[Finding("widen.score", f"not exact after the widening: {s}") for s in short])
        except BaseException:
            store.undo(undone)  # every file the widening and the placing wrote, as it was
            if synced:
                repo.sync(config, {})
            raise
    written |= touched
    head = f"Port {sorted(names)[0]}{f' +{len(names) - 1}' if len(names) > 1 else ''} to {','.join(u.widen)}"
    out = Path(params["out"]) if params.get("out") else root / ".unbake" / "widen"
    target = write_diff(config, written, head, out / f"{Path(path).stem}-{'-'.join(u.widen)}.diff")
    return {**shown, "scores": scores, "written": sorted(written), "diff": str(target)}
