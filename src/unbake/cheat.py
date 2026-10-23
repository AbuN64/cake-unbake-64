"""The no-cheat check: a matching build proves the bytes, this proves the source that made them is real C. Over a
revision range it refuses assembly, raw words, pragmas, section attributes, an INCLUDE_ASM that was not there, a
non-C file under the source roots, edits to the build and compare infrastructure and any Makefile change that is not
a per-object compiler row or the Makefile `sync` generates for the head tree. A stand-in line that is exactly what
`sync` writes for a symbol no C owns passes: the tool knows what it generates, so a hand-written one does not.
Reads git and the tree's facts only: no ROM, no build. The rules are data (repo.toml [cheat])."""

from __future__ import annotations

import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from unbake import api, build, process, symbols, tree
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Refusal, digest

_HUNK = re.compile(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@")
_STAND_IN = re.compile(r'^\s*INCLUDE_(?:DATA|RODATA)\(\s*"([^"]+)"\s*,\s*(\w+)\s*\)\s*;\s*$')
_TOKEN = re.compile(r"\s*(defined|[A-Za-z_]\w*|\d+[uUlL]*|0[xX][0-9a-fA-F]+|&&|\|\||==|!=|<=|>=|[!()<>]|.)")
_OPENERS = ("if", "ifdef", "ifndef")
_CONDITIONAL = re.compile(r"\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)")


def _and(a: bool | None, b: bool | None) -> bool | None:
    if a is False or b is False:
        return False
    return True if (a is True and b is True) else None


def _or(a: bool | None, b: bool | None) -> bool | None:
    if a is True or b is True:
        return True
    return False if (a is False and b is False) else None


def _evaluate(expr: str, non_matching: bool) -> bool | None:
    """An `#if` expression with NON_MATCHING defined or not and every other macro unknown: True, False or None."""
    toks = [m.group(1) for m in _TOKEN.finditer(re.sub(r"/\*.*?\*/|//.*", "", expr))]
    at = [0]

    def peek() -> str | None:
        return toks[at[0]] if at[0] < len(toks) else None

    def eat() -> str:
        at[0] += 1
        return toks[at[0] - 1]

    def primary() -> bool | None:
        t = eat()
        if t == "(":
            v = disjunction()
            if eat() != ")":
                raise ValueError(expr)
            return v
        if t == "!":
            v = primary()
            return None if v is None else not v
        if t == "defined":
            paren = peek() == "("
            if paren:
                eat()
            name = eat()
            if paren and eat() != ")":
                raise ValueError(expr)
            return non_matching if name == "NON_MATCHING" else None
        if re.fullmatch(r"\d+[uUlL]*", t):
            return int(re.sub(r"[uUlL]", "", t)) != 0
        if re.fullmatch(r"0[xX][0-9a-fA-F]+", t):
            return int(t, 16) != 0
        if re.fullmatch(r"[A-Za-z_]\w*", t):
            return None  # an unknown macro: its comparison operands stay unknown
        raise ValueError(expr)

    def comparison() -> bool | None:
        v = primary()
        while peek() in ("==", "!=", "<", ">", "<=", ">="):
            eat()
            primary()
            v = None
        return v

    def conjunction() -> bool | None:
        v = comparison()
        while peek() == "&&":
            eat()
            v = _and(v, comparison())
        return v

    def disjunction() -> bool | None:
        v = conjunction()
        while peek() == "||":
            eat()
            v = _or(v, conjunction())
        return v

    value = disjunction()
    if peek() is not None:
        raise ValueError(expr)
    return value


def _condition(kind: str, rest: str, non_matching: bool) -> bool | None:
    if kind in ("ifdef", "ifndef"):
        if rest.split()[0] != "NON_MATCHING":
            return None
        return non_matching if kind == "ifdef" else not non_matching
    return _evaluate(rest, non_matching)


def non_matching_lines(text: str) -> set[int]:
    """1-based numbers of the lines inside a branch that is dead without NON_MATCHING and live with it (other macros
    unknown). Any condition it cannot read keeps its lines out of the set: a parse doubt is never an exemption."""
    out: set[int] = set()
    stack: list[dict[str, Any]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        full = lines[i]
        m = _CONDITIONAL.match(full)
        while m and full.rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            full = full.rstrip()[:-1] + " " + lines[i]
            m = _CONDITIONAL.match(full)
        if m:
            kind, rest = m.group(1), m.group(2).strip()
            try:
                if kind in ("if", "ifdef", "ifndef"):
                    c0, c1 = _condition(kind, rest, False), _condition(kind, rest, True)
                    stack.append(
                        {
                            "off": c0 is True,
                            "on": c1 is True,
                            "dead_off": c0 is False,
                            "dead_on": c1 is False,
                            "bad": False,
                        }
                    )
                elif kind == "elif":
                    f = stack[-1]
                    c0, c1 = _condition("if", rest, False), _condition("if", rest, True)
                    f["dead_off"], f["dead_on"] = f["off"] or c0 is False, f["on"] or c1 is False
                    f["off"], f["on"] = f["off"] or c0 is True, f["on"] or c1 is True
                elif kind == "else":
                    f = stack[-1]
                    f["dead_off"], f["dead_on"] = f["off"], f["on"]
                else:
                    stack.pop()
            except (ValueError, IndexError):
                if kind in _OPENERS:
                    stack.append({"off": False, "on": False, "dead_off": False, "dead_on": False, "bad": True})
                elif stack:
                    stack[-1].update(dead_off=False, dead_on=False, bad=True)
        elif (
            stack
            and not any(f["bad"] for f in stack)
            and any(f["dead_off"] for f in stack)
            and not any(f["dead_on"] for f in stack)
        ):
            out.add(i + 1)
        i += 1
    return out


class _Git:
    """git over a project root through process.run, text out; a failed command refuses."""

    def __init__(self, root: Path) -> None:
        self.root, self.tmp = root, process.scratch(root)
        found = shutil.which("git")
        if found is None:
            raise Refusal(Finding("native.missing_tool", "git is not installed", path="git"))
        self.exe = found

    def __call__(self, *args: str, ok: Sequence[int] = (0,)) -> str:
        result = process.run("git " + args[0], [self.exe, *args], self.root, tmp=self.tmp)
        if result.exit not in ok:
            raise Refusal(Finding("native.exit", f"git {' '.join(args)} exit {result.exit}"))
        return result.stdout.decode(errors="replace")


def _added(git: _Git, base: str, head: str, roots: Sequence[str]) -> list[str]:
    """The lines `head` adds under the source roots over `base` (a renamed file shows only the lines it changes),
    without those inside a NON_MATCHING-only branch (dead without NON_MATCHING, live with it). A file whose branches
    cannot be read keeps every added line in scope."""
    diff = git("diff", "-M", "-l0", "-U0", base, head, "--", *roots)
    out: list[str] = []
    path: str | None = None
    exempt: dict[str, set[int]] = {}
    cur = 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif m := _HUNK.match(line):
            cur = int(m.group(1))
        elif line.startswith("+"):
            if path is not None and path not in exempt:
                try:
                    exempt[path] = non_matching_lines(git("show", f"{head}:{path}"))
                except Refusal:
                    exempt[path] = set()
            if not (path is not None and cur in exempt[path]):
                out.append(line[1:])
            cur += 1
    return out


def _makefile_generated(config: Config, git: _Git, head: str, rows: re.Pattern[str]) -> bool:
    """HEAD's Makefile, per-object rows stripped, is byte-identical to the Makefile `sync` generates for the head tree
    (also rows stripped), generated from git objects only and never written."""
    root = config.project.root
    project = configuration.load_project(
        root,
        lambda p: git("show", f"{head}:{p}").encode(),
        lambda: git("ls-tree", "-r", "--name-only", head).splitlines(),
    )
    generated = build.makefile(Config(project, config.host, digest((project.digest, config.host.digest)))).decode()

    def strip(text: str) -> list[str]:
        return [ln for ln in text.splitlines() if not rows.search(ln)]

    return bool(strip(generated)) and strip(generated) == strip(git("show", f"{head}:Makefile"))


def _defined_at(git: _Git, base: str, name: str, roots: Sequence[str]) -> bool:
    """Whether a file under the roots at `base` defines `name` in C."""
    files = git("grep", "-lw", name, base, "--", *roots, ok=(0, 1)).splitlines()
    return any(name in tree.defined(git("show", f)) for f in (x.split(":", 1)[1] for x in files if ":" in x))


def _generated(config: Config, git: _Git, base: str, segment: str, name: str, roots: Sequence[str]) -> bool:
    """A data stand-in `sync` writes: `name` is a fact at the head tree with a ROM address inside `segment`, not
    code, and no C at `base` defined it (a C definition turned into a stand-in is a hidden body, not a symbol the
    tool exposed)."""
    for v in config.project.versions:
        fact = api.symbols(config, v).get(name)
        if fact and fact[1] is not None and fact[3] != "func" and symbols.segment_at(config, v, fact[1]) == segment:
            return not _defined_at(git, base, name, roots)
    return False


def _fact_text(git: _Git, rev: str, version: str) -> list[str]:
    """A version's symbol_addrs.txt and aliases.txt at `rev` ("" for one that is absent there)."""
    return [git("show", f"{rev}:{p}", ok=(0, 128)) for p in (symbols.path(version), symbols.aliases_path(version))]


def _duplicate_names(config: Config, git: _Git, base: str, head: str) -> dict[str, list[str]]:
    """version -> the names `head` spells on more than one fact row that `base` did not (one name, two rows: the
    second address never reaches the link image or the port's boot image)."""
    out: dict[str, list[str]] = {}
    for v in config.project.versions:
        new = symbols.duplicates(*_fact_text(git, head, v)) - symbols.duplicates(*_fact_text(git, base, v))
        if new:
            out[v] = sorted(new)
    return out


def _tree_findings(config: Config) -> tuple[Finding, ...]:
    """What the head tree refuses to capture (a `literals_` row no file owns, an unparsable source, ...): the check
    every command that reads the tree makes, so a change the tree cannot hold is refused here, before it lands."""
    try:
        tree.capture(config)
    except Refusal as refused:
        return refused.findings
    return ()


def cheat_command(config: Config, params: Json) -> Json:
    """Refuse what `head` (default HEAD) adds over `against` that could fake a match. The facts read are the working
    tree's, so it must be checked out at `head`."""
    base, head = params["against"], params.get("head") or "HEAD"
    rules = configuration.load_resource("repo.toml")["cheat"]
    git = _Git(config.project.root)
    if git("status", "--porcelain", "--", *rules["facts"]).strip() and head == "HEAD":
        raise Refusal(Finding("cheat.dirty", "the tree's facts differ from HEAD: check out the revision first"))
    roots = rules["source"]
    findings: list[Finding] = []
    infra = [p for p in git("diff", "--name-only", base, head, "--", *rules["infra"]).splitlines() if p]
    if infra:
        findings.append(Finding("cheat.infra", "build/compare infrastructure changed", missing=tuple(infra)))
    rows = re.compile(rules["makefile_row"])
    changed = [
        ln
        for ln in git("diff", "-U0", base, head, "--", "Makefile").splitlines()
        if ln[:1] in "+-" and not ln.startswith(("+++", "---")) and not rows.match(ln[1:])
    ]
    makefile = "unchanged"
    if changed:
        makefile = "generated" if _makefile_generated(config, git, head, rows) else ""
        if not makefile:
            findings.append(
                Finding("cheat.makefile", "Makefile change beyond per-object compiler rows", missing=tuple(changed[:8]))
            )
    scan = re.compile("|".join(rules["added"]))
    stand_ins: list[str] = []
    hits: list[str] = []
    for line in _added(git, base, head, roots):
        if not scan.search(line):
            continue
        found = _STAND_IN.match(line)
        if found and _generated(config, git, base, found[1], found[2], roots):
            stand_ins.append(found[2])
        else:
            hits.append(line.strip())
    if hits:
        findings.append(Finding("cheat.added", "added asm/raw-word/pragma/section lines", missing=tuple(hits[:8])))

    def count(rev: str) -> int:
        rows = git("grep", "-c", "INCLUDE_ASM(", rev, "--", *roots, ok=(0, 1)).splitlines()
        return sum(int(row.rsplit(":", 1)[1]) for row in rows)

    before, after = count(base), count(head)
    if after > before:
        findings.append(Finding("cheat.include-asm", f"INCLUDE_ASM count rose {before} -> {after}"))
    suffix = re.compile(rules["source_suffix"])
    odd = [
        p
        for p in git("diff", "-M", "-l0", "--name-only", "--diff-filter=A", base, head, "--", *roots).splitlines()
        if p and not suffix.search(p)
    ]
    if odd:
        findings.append(Finding("cheat.non-c", "non-C files added under the source roots", missing=tuple(odd[:8])))
    for v, names in _duplicate_names(config, git, base, head).items():
        findings.append(
            Finding(
                "cheat.duplicate-name",
                f"{v}: a symbol name is spelled on more than one fact row",
                missing=tuple(names[:8]),
                action="give each address its own name (rename the new row, e.g. suffix _2)",
            )
        )
    if head == "HEAD":  # the facts and sources captured are the working tree's: the head only when it is checked out
        findings.extend(_tree_findings(config))
    if findings:
        raise Refusal(*findings)
    return {"against": base, "head": head, "makefile": makefile, "stand_ins": sorted(set(stand_ins))}
