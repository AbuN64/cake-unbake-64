"""`unbake layout`: move a project's files to a target folder layout by rules, and carry work across the move.

plan    reads a rules file (`[[rule]]` rows: `match` a file or a directory ending in `/`, `to` where it goes, and an
        optional `row` for the destination directory) and the tracked files, and writes `.unbake/layout/plan.json`
        and `map.tsv` (old path, new path). The longest `match` wins; a path no rule names stays. Nothing else moves.
apply   moves the files of the plan, rewrites `config.toml` (row keys, `libraries`, `-I` flags, the rules' directory and
        library rows; a per-file row that equals the row above it goes), respells the per-file and per-folder flag
        rules of an existing Makefile (sync leaves it alone) and runs sync. It writes files only, never git
        state: `git add -A` is the caller's step.
carry   turns a worktree's uncommitted work into the new layout: untracked files move in place and the uncommitted
        edits of tracked files are written as a patch with the new paths, to apply after the worktree is on the new
        layout.
reorg   the one-command form: plan from a `--map` file (old path, new path per line; headers included), apply, rewrite
        every `#include` of a moved header (and the relative includes of moved files), and with `--prove` run
        `make check` once so every ROM is compared after the single move.
mergecheck  reads a module map (unit, module) and the built objects and says which merges must stay separate files.
verify  compares the objects of a build made before the move with the build after it, through the map, and the link
        script of both.
Everything about one project's folders is in its rules file; this module names no folder."""

from __future__ import annotations

import json
import os
import posixpath
import re
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tomlkit
from elftools.elf.elffile import ELFFile

from unbake import config as configuration
from unbake import effort, process, repo, store
from unbake import mergecheck as merging
from unbake.contracts import Config, FileFacts, Finding, Json, Refusal, digest, file_facts

Row = dict[str, Any]
STATE = Path(".unbake") / "layout"
PLAN = STATE / "plan.json"
MAP = STATE / "map.tsv"
CARRY = STATE / "carry.patch"
_KINDS = ("dir", "library")
_PASSES = 6
_SKIPPED = frozenset({".symtab", ".strtab", ".shstrtab", ".mdebug", ".comment", ".gnu.attributes", ".pdr", ".reginfo"})


def _refuse(reason: str, **fields: Any) -> Refusal:
    return Refusal(Finding("layout.input", reason, **fields))


# ---- rules
def rules_from(text: str, file: str = "rules") -> Row:
    """The validated rules file: `rules` (each a `match`, a `to`, and an optional `row` for the destination directory),
    `rows` (standalone `[[row]]` tables for directories no rule moves into) and `libraries`, the library list the
    config keeps afterwards when the file gives one. A row has a `path`, a `kind` (dir or library) and any of
    toolchain, cflags, omit and, for a library, origin."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise _refuse(f"{file} is not valid TOML: {error}") from error
    rows = document.get("rule", [])
    seen: set[str] = set()
    problems = []
    for index, rule in enumerate(rows):
        where = f"{file}:rule[{index}]"
        match, to = rule.get("match"), rule.get("to")
        if not isinstance(match, str) or not isinstance(to, str) or not match or not to:
            problems.append(Finding("layout.input", "a rule needs a match and a to", path=where))
            continue
        if match.endswith("/") and not to.endswith("/"):
            problems.append(Finding("layout.input", f"{match} is a directory: its to must end in /", path=where))
        if match in seen:
            problems.append(Finding("layout.input", f"{match} is matched by two rules", path=where))
        seen.add(match)
        row = rule.get("row")
        if row is not None and (row.get("kind") not in _KINDS or not to.endswith("/")):
            problems.append(Finding("layout.input", "a row needs kind dir or library and a directory to", path=where))
    standalone = document.get("row", [])
    for index, row in enumerate(standalone):
        where = f"{file}:row[{index}]"
        if row.get("kind") not in _KINDS or not str(row.get("path", "")).endswith("/"):
            problems.append(
                Finding("layout.input", "a row needs kind dir or library and a path ending in /", path=where)
            )
    libraries = document.get("libraries")
    if libraries is not None and not all(isinstance(f, str) and f.endswith("/") for f in libraries):
        problems.append(Finding("layout.input", "libraries must be directories ending in /", path=f"{file}:libraries"))
    if problems:
        raise Refusal(*problems)
    return {"rules": [dict(r) for r in rows], "rows": [dict(r) for r in standalone], "libraries": libraries}


def rules_from_map(text: str, file: str = "map") -> list[Row]:
    """The rules a map file stands for: one `old<TAB>new` line per file (a `#` line is a comment), the file named whole.
    A new path that ends in / keeps the file name."""
    rules: list[Row] = []
    seen: set[str] = set()
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.rstrip("\n").split("\t")
        where = f"{file}:{number}"
        if len(cells) < 2 or not cells[0].strip() or not cells[1].strip():
            problems.append(Finding("layout.input", "a map line is `old path<TAB>new path`", path=where))
            continue
        old, new = cells[0].strip(), cells[1].strip()
        if old in seen:
            problems.append(Finding("layout.input", f"{old} is mapped twice", path=where))
        seen.add(old)
        if old != new:
            rules.append({"match": old, "to": new})
    if problems:
        raise Refusal(*problems)
    return rules


def place(rules: Sequence[Mapping[str, Any]], path: str) -> str:
    """Where a path goes: by the rule with the longest match, else where it is. A directory match keeps the rest of the
    path; a file match goes to `to` (a directory keeps the file name)."""
    best: Mapping[str, Any] | None = None
    for rule in rules:
        match = rule["match"]
        if (path == match or (match.endswith("/") and path.startswith(match))) and (
            best is None or len(match) > len(best["match"])
        ):
            best = rule
    if best is None:
        return path
    match, to = str(best["match"]), str(best["to"])
    if match.endswith("/"):
        return to + path[len(match) :]
    return to + path.rsplit("/", 1)[-1] if to.endswith("/") else to


def moves(rules: Sequence[Mapping[str, Any]], tracked: Sequence[str]) -> tuple[dict[str, str], list[Row]]:
    """(old -> new for every tracked path a rule moves, the destination rows the moving rules ask for). Refuses two
    paths landing on one, and a path landing on a file that stays."""
    mapping = {old: new for old in sorted(tracked) if (new := place(rules, old)) != old}
    problems = []
    landing: dict[str, str] = {}
    for old, new in mapping.items():
        if new in landing:
            problems.append(Finding("layout.collision", f"{old} and {landing[new]} both move to {new}", path=old))
        landing[new] = old
        if new in tracked and new not in mapping:
            problems.append(Finding("layout.collision", f"{old} would replace {new}, which stays", path=old))
    if problems:
        raise Refusal(*problems)
    rows: dict[str, Row] = {}
    for old in mapping:
        rule = _rule_of(rules, old)
        if rule is not None and rule.get("row"):
            folder = rule["to"] if rule["to"].endswith("/") else rule["to"].rsplit("/", 1)[0] + "/"
            rows[folder] = {"path": folder, **rule["row"]}
    return mapping, [rows[k] for k in sorted(rows)]


def _rule_of(rules: Sequence[Mapping[str, Any]], path: str) -> Mapping[str, Any] | None:
    found = [r for r in rules if path == r["match"] or (r["match"].endswith("/") and path.startswith(r["match"]))]
    return max(found, key=lambda r: len(r["match"])) if found else None


def folded(tracked: Sequence[str], mapping: Mapping[str, str]) -> dict[str, str]:
    """Old directory -> new directory for every directory whose files all moved the same way (the rest of each path
    unchanged), so a row, a library or a `-I` flag naming the directory can name the new one."""
    found: dict[str, set[str | None]] = defaultdict(set)
    for old in tracked:
        new = mapping.get(old, old)
        parts = old.split("/")
        for depth in range(1, len(parts)):
            folder = "/".join(parts[:depth]) + "/"
            rest = old[len(folder) :]
            found[folder].add(new[: -len(rest)] if new.endswith(rest) and new != rest else None)
    out: dict[str, str] = {}
    for folder, news in found.items():
        (only,) = news if len(news) == 1 else (None,)
        if only is not None and only != folder:
            out[folder] = only
    return out


# ---- plan
def _tracked(config: Config) -> list[str]:
    return sorted(process.tracked(config.project.root))


def plan(config: Config, params: Json) -> Row:
    """Write the plan and the map for a rules file; moves nothing."""
    with effort.stage("layout.plan"):
        path, mapped = params.get("rules"), params.get("map")
        if not path and not mapped:
            raise _refuse("plan needs --rules FILE or --map FILE")
        root = config.project.root
        given = rules_from(Path(path).read_text(), str(path)) if path else {"rules": [], "rows": [], "libraries": None}
        rules = [*given["rules"], *(rules_from_map(Path(mapped).read_text(), str(mapped)) if mapped else [])]
        tracked = _tracked(config)
        mapping, rows = moves(rules, tracked)
        taken = {r["path"] for r in rows}
        clash = sorted(r["path"] for r in given["rows"] if r["path"] in taken)
        if clash:
            raise _refuse(f"rows given twice (by a rule and by a [[row]]): {', '.join(clash)}")
        rows = sorted([*rows, *given["rows"]], key=lambda r: r["path"])
        document = {"schema": 1, "rules": rules, "moves": mapping, "rows": rows, "libraries": given["libraries"]}
        store.write(root / PLAN, json.dumps(document, sort_keys=True, indent=1).encode())
        store.write(root / MAP, "".join(f"{o}\t{n}\n" for o, n in sorted(mapping.items())).encode())
        return {
            "action": "plan",
            "moved": len(mapping),
            "kept": len(tracked) - len(mapping),
            "rows": rows,
            "path": PLAN.as_posix(),
            "map": MAP.as_posix(),
        }


# ---- includes
_TEXT = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".inc"})
_DIRECTIVE = re.compile(r'^([ \t]*#[ \t]*include[ \t]*)(["<])([^">\n]+)([">])', re.M)
_MIN_VOTES = 2


def _roots(texts: Mapping[str, str], tracked: set[str]) -> list[str]:
    """The directories includes are written against besides the including file's own: every directory R where some
    include `X` names exactly one tracked file `R/X` that is not next to its includer, at least twice, most used
    first."""
    by_base: dict[str, list[str]] = defaultdict(list)
    for t in tracked:
        by_base[posixpath.basename(t)].append(t)
    votes: dict[str, int] = defaultdict(int)
    for path, text in texts.items():
        for name in {m.group(3) for m in _DIRECTIVE.finditer(text)}:
            if posixpath.normpath(posixpath.join(posixpath.dirname(path), name)) in tracked:
                continue
            hits = [t for t in by_base.get(posixpath.basename(name), []) if t == name or t.endswith("/" + name)]
            if len(hits) == 1:
                votes[hits[0][: len(hits[0]) - len(name)]] += 1
    return [r for r, n in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0])) if n >= _MIN_VOTES]


def includes(texts: Mapping[str, str], tracked: Sequence[str], mapping: Mapping[str, str]) -> dict[str, str]:
    """The `#include` lines of the text files (old path -> text) written for the layout after the move: an include
    that names a moved file, or that is relative to a file that moved, is spelled again the way it was written (next
    to the includer, or under the same include directory) for the new places. Returns the files whose text changes."""
    known = set(tracked)
    roots = _roots(texts, known)
    changed: dict[str, str] = {}
    for path, text in texts.items():
        here = mapping.get(path, path)

        def respell(match: re.Match[str], path: str = path, here: str = here) -> str:
            name = match.group(3)
            near = posixpath.normpath(posixpath.join(posixpath.dirname(path), name))
            if near in known:
                target, root = near, None
            else:
                root = next((r for r in roots if r + name in known), None)
                if root is None:
                    return match.group(0)
                target = root + name
            new = mapping.get(target, target)
            if root is not None and new.startswith(root):
                spelled = new[len(root) :]
            elif root is not None and (other := next((r for r in roots if new.startswith(r)), None)) is not None:
                spelled = new[len(other) :]
            else:
                spelled = posixpath.relpath(new, posixpath.dirname(here) or ".")
            if root is None and spelled == name and posixpath.dirname(here) == posixpath.dirname(path):
                return match.group(0)
            return match.group(1) + match.group(2) + spelled + match.group(4)

        out = _DIRECTIVE.sub(respell, text)
        if out != text:
            changed[path] = out
    return changed


_MAKE_RULE = re.compile(r"(build/(?:nmraw/)?\$\(v\)/)(\S+?)(%?)(\.o\):)")


def makefile_rules(text: str, tracked: Sequence[str], mapping: Mapping[str, str]) -> tuple[str, int]:
    """A Makefile's per-file and per-folder flag rules (`build/$(v)/<file>.o` and `<dir>/%.o`, also in the second
    object tree) named for the layout after the move. Sync leaves an existing Makefile alone, so a rule keyed to an old
    path would stop applying to its moved source. Returns the new text and the count of rules respelled."""
    dirs = folded(tracked, mapping)
    count = 0

    def respell(match: re.Match[str]) -> str:
        nonlocal count
        old = match.group(2)
        new = dirs.get(old) if match.group(3) else mapping.get(old)
        if not new or new == old:
            return match.group(0)
        count += 1
        return match.group(1) + new + match.group(3) + match.group(4)

    return _MAKE_RULE.sub(respell, text), count


def _texts(root: Path, tracked: Sequence[str]) -> dict[str, str]:
    return {t: (root / t).read_text(errors="surrogateescape") for t in tracked if Path(t).suffix.lower() in _TEXT}


# ---- apply
def _read_plan(config: Config, params: Json) -> Row:
    path = Path(params.get("plan") or config.project.root / PLAN)
    if not path.is_file():
        raise _refuse(f"no plan at {path}", action="run `unbake layout plan --rules FILE` first")
    document: Row = json.loads(path.read_text())
    return document


def _facts(row: Mapping[str, Any]) -> FileFacts:
    return FileFacts(row.get("toolchain"), tuple(row.get("cflags", ())), tuple(row.get("omit", ())))


def _new_dir(path: str, dirs: Mapping[str, str]) -> str | None:
    """The new name of a `-I` path, or None when the path is not a moved directory."""
    folder = path.rstrip("/") + "/"
    return dirs[folder].rstrip("/") if folder in dirs else None


def _flags(words: Sequence[Any], dirs: Mapping[str, str]) -> list[str]:
    """Compiler words with the `-I` paths of moved directories named in the new layout."""
    return ["-I" + new if str(w).startswith("-I") and (new := _new_dir(str(w)[2:], dirs)) else str(w) for w in words]


def _rows(document: Any) -> dict[str, tuple[str, FileFacts]]:
    """path -> (table, facts) for every [file], [dir] and [library] row of a parsed config."""
    return {
        path: (kind, _facts(table[path]))
        for kind in ("file", "dir", "library")
        for table in [document.get(kind) or {}]
        for path in table
    }


def _effective(rows: Mapping[str, FileFacts], default: str, path: str) -> tuple[str, frozenset[str], frozenset[str]]:
    """What the build gives a source: its deepest row's toolchain (else the project's) and flags."""
    row = file_facts(SimpleNamespace(files=rows), path)  # type: ignore[arg-type]
    return row.toolchain or default, frozenset(row.cflags), frozenset(row.omit)


def rewrite(
    text: str,
    tracked: Sequence[str],
    mapping: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    default: str,
    libraries: Sequence[str] | None = None,
) -> tuple[str, Row]:
    """The text of config.toml after the move, and what changed: file rows renamed by the map, directory rows by the
    folded directories, `libraries` and `-I` flags likewise, the plan's rows added; a source whose toolchain or flags
    the new rows would change keeps its old ones as a file row; a row that now equals the row above it (or the project
    default) is dropped, library rows aside (they mark a boundary)."""
    dirs = folded(tracked, mapping)
    document: Any = tomlkit.parse(text)
    before = {
        path: FileFacts(f.toolchain, tuple(_flags(f.cflags, dirs)), f.omit) for path, (_, f) in _rows(document).items()
    }
    problems, renamed = [], 0
    for kind in ("file", "dir", "library"):
        table = document.get(kind)
        if table is None:
            continue
        for key in list(table):
            new = mapping.get(key) or dirs.get(key) or key
            if key.endswith("/") and key not in dirs and any(t.startswith(key) and t in mapping for t in tracked):
                problems.append(
                    Finding("layout.split", f"[{kind}] row {key} covers files that move to different places")
                )
            if new != key:
                item = table[key]
                table.remove(key)
                table.add(new, item)
                renamed += 1
    for array in (document.get("cppflags"), document.get("cflags")):
        if array is not None:
            for index, word in enumerate(_flags(array, dirs)):
                array[index] = word
    for kind in ("file", "dir", "library"):
        for body in (document.get(kind) or {}).values():
            if "cflags" in body:
                body["cflags"] = _flags(body["cflags"], dirs)
    flat = document.get("libraries")
    if libraries is not None:
        document.pop("libraries", None)
        if libraries:
            document["libraries"] = list(libraries)
    elif flat is not None:
        for index, folder in enumerate(list(flat)):
            if folder in dirs:
                flat[index] = dirs[folder]
            elif any(t.startswith(folder) and t in mapping for t in tracked):
                problems.append(Finding("layout.split", f"library {folder} covers files that move to different places"))
    if problems:
        raise Refusal(*problems)
    added = []
    for row in rows:
        kind = row["kind"]
        if kind not in document:
            document.add(kind, tomlkit.table(True))
        if row["path"] not in document[kind]:
            body = tomlkit.table()
            for key in ("toolchain", "cflags", "omit", "origin"):
                if key in row:
                    body.add(key, row[key])
            document[kind].add(row["path"], body)
            added.append(row["path"])
    after = {path: f for path, (_, f) in _rows(document).items()}
    kept = []
    for old in tracked:
        if not configuration.is_source(old) or configuration.is_assembly(old):
            continue
        new = mapping.get(old, old)
        want = _effective(before, default, old)
        if want != _effective(after, default, new):
            if "file" not in document:
                document.add("file", tomlkit.table(True))
            body = tomlkit.table()
            body.add("toolchain", want[0])
            old_row = file_facts(SimpleNamespace(files=before), old)  # type: ignore[arg-type]
            body.add("cflags", list(old_row.cflags))
            if old_row.omit:
                body.add("omit", list(old_row.omit))
            document["file"].add(new, body)
            after[new] = _facts(body)
            kept.append(new)
    # directory and library rows read before the file rows they sit above
    if "file" in document:
        files = document.pop("file")
        document.add("file", files)
    libraries_now = set(document.get("library") or {})
    dead = [p for p in configuration.dead_paths(after, default) if p not in libraries_now]
    for path in dead:
        for kind in ("file", "dir", "library"):
            if document.get(kind) is not None and path in document[kind]:
                document[kind].remove(path)
    for kind in ("file", "dir", "library"):
        if document.get(kind) is not None and not len(document[kind]):
            document.pop(kind)
    return tomlkit.dumps(document), {
        "renamed": renamed,
        "added": added,
        "exceptions": sorted(kept),
        "dropped": sorted(dead),
    }


def _git(config: Config, *args: str) -> bytes:
    git = process.tool(config, "git")
    done = process.run("git " + args[0], [git, *args], config.project.root, tmp=process.scratch(config.project.root))
    if done.exit != 0:
        raise Refusal(Finding("native.exit", f"git {' '.join(args)} exit {done.exit}"))
    return done.stdout


def apply(config: Config, params: Json) -> Row:
    """Carry out a plan on the working tree: config.toml rewritten and loaded again first (so a mistake refuses before
    any file moves), then the files, then sync."""
    with effort.stage("layout.apply"):
        root = config.project.root
        document = _read_plan(config, params)
        mapping: dict[str, str] = document["moves"]
        tracked = _tracked(config)
        missing = sorted(o for o in mapping if not (root / o).is_file())
        if missing:
            raise _refuse(f"{len(missing)} files of the plan are not in the tree", missing=tuple(missing[:20]))
        dirty = sorted(
            {p for p in _git(config, "diff", "--name-only", "-z", "HEAD").decode().split("\0") if p in mapping}
        )
        if dirty:
            raise _refuse(
                f"{len(dirty)} files the plan moves have uncommitted edits: commit or carry them",
                missing=tuple(dirty[:20]),
            )
        new_tracked = sorted({mapping.get(t, t) for t in tracked})
        makefile = root / "Makefile"
        rules, ruled = makefile_rules(makefile.read_text(), tracked, mapping) if makefile.is_file() else ("", 0)
        respelled = includes(_texts(root, tracked), tracked, mapping)
        text, changed = rewrite(
            (root / "config.toml").read_text(),
            tracked,
            mapping,
            document["rows"],
            config.project.toolchain,
            document.get("libraries"),
        )

        def read(path: str) -> bytes:
            return text.encode() if path == "config.toml" else (root / path).read_bytes()

        project = configuration.load_project(root, read, lambda: new_tracked)  # refuses a rewrite that does not load
        for old in sorted(mapping):
            target = root / mapping[old]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(root / old, target)
            folder = (root / old).parent
            while folder != root and not any(folder.iterdir()):
                folder.rmdir()
                folder = folder.parent
        for old, body in respelled.items():
            store.write(root / mapping.get(old, old), body.encode(errors="surrogateescape"))
        store.write(root / "config.toml", text.encode())
        if ruled:
            store.write(makefile, rules.encode())
        effort.forget("pin", "includes", "closure", "listing")
        loaded = Config(project, config.host, digest((project.digest, config.host.digest)))
        try:
            written = repo.sync(loaded, {})["written"]
            held: list[Row] = []
        except (
            Refusal
        ) as refusal:  # a table the tree leaves unplaced is the project's, not the move's: `literals` places it
            if not all(f.key in _TABLES for f in refusal.findings):
                raise
            written, held = [], [{"key": f.key, "reason": f.reason} for f in refusal.findings]
        return {
            "action": "apply",
            "moved": len(mapping),
            "config": changed,
            "includes": len(respelled),
            "makefile": ruled,
            "written": written,
            "held": held,
            "next": "git add -A; "
            + ("unbake layout literals (places the tables sync held); " if held else "")
            + "unbake sync --check",
        }


# ---- literals
_TABLES = frozenset({"sync.literals", "sync.literals-span"})
_ADD = re.compile(r"^add `(?P<row>[^`]+)` to (?P<file>\S+)")


def literals(config: Config, params: Json) -> Row:
    """Sync until it passes, adding to the versions' symbol files each row sync's own findings name (a jump table's
    `literals_<function>` row, a split of a data row that runs over one): a table of a function the tree compiles is
    that function's file's read-only data, and these rows are how a project says so. Stops when sync passes or a
    finding is of any other kind."""
    with effort.stage("layout.literals"):
        root = config.project.root
        added: list[str] = []
        for _ in range(_PASSES):
            try:
                written = repo.sync(config, {})["written"]
                return {"action": "literals", "rows": added, "written": written}
            except Refusal as refusal:
                if not all(f.key in _TABLES for f in refusal.findings):
                    raise
                fresh = 0
                for finding in refusal.findings:
                    found = _ADD.match(finding.action)
                    if not found:
                        raise
                    target = root / found["file"]
                    body = target.read_text()
                    if found["row"] not in body.splitlines():
                        store.write(target, (body.rstrip("\n") + "\n" + found["row"] + "\n").encode())
                        added.append(f"{found['file']}: {found['row']}")
                        fresh += 1
                if not fresh:
                    raise
        raise _refuse(f"sync still holds tables after {_PASSES} passes", missing=tuple(added[-20:]))


# ---- carry
_DIFF = re.compile(r"^diff --git a/(\S+) b/(\S+)$")
_SIDE = re.compile(r"^(--- a/|\+\+\+ b/|rename from |rename to )(\S+)$")


def carry_patch(patch: str, rules: Sequence[Mapping[str, Any]], mapping: Mapping[str, str]) -> str:
    """A git patch with every path of its headers placed in the new layout (the map first, else the rules). Only the
    header lines of each file are read: a line of a hunk that looks like one is content."""

    def go(path: str) -> str:
        return mapping.get(path) or place(rules, path)

    out = []
    header = False
    for line in patch.splitlines(keepends=True):
        text = line.rstrip("\n")
        if found := _DIFF.match(text):
            header = True
            line = f"diff --git a/{go(found[1])} b/{go(found[2])}\n"
        elif header and (side := _SIDE.match(text)):
            line = f"{side[1]}{go(side[2])}\n"
        elif text.startswith(("@@", "GIT binary patch", "Binary files")):
            header = False
        out.append(line)
    return "".join(out)


def carry(config: Config, params: Json) -> Row:
    """Untracked files that a plan moves go to their new places now; the uncommitted edits of tracked files are written
    as a patch with the new paths. The tracked files themselves are not touched."""
    with effort.stage("layout.carry"):
        root = config.project.root
        document = _read_plan(config, params)
        rules, mapping = document["rules"], document["moves"]
        untracked = [p for p in _git(config, "ls-files", "-o", "--exclude-standard", "-z").decode().split("\0") if p]
        moved, clash = [], []
        for old in untracked:
            new = place(rules, old)
            if new == old:
                continue
            if (root / new).exists():
                clash.append(old)
                continue
            (root / new).parent.mkdir(parents=True, exist_ok=True)
            os.replace(root / old, root / new)
            moved.append(new)
        patch = _git(config, "diff", "HEAD", "--binary", "--no-renames").decode(errors="surrogateescape")
        out = Path(params.get("patch") or root / CARRY)
        store.write(out, carry_patch(patch, rules, mapping).encode(errors="surrogateescape"))
        return {
            "action": "carry",
            "moved": sorted(moved),
            "clash": sorted(clash),
            "patch": str(out),
            "next": "git stash, move the tree to the new layout, then git apply --3way " + str(out),
        }


# ---- verify
def _contents(path: Path) -> dict[str, bytes]:
    with path.open("rb") as handle:
        return {s.name: bytes(s.data()) for s in ELFFile(handle).iter_sections() if s.name and s.name not in _SKIPPED}


def verify(config: Config, params: Json) -> Row:
    """Objects built before the move against objects built after it, through the plan's map: every section the same.
    `old` is a directory holding the earlier `build/` tree and the earlier link script."""
    with effort.stage("layout.verify"):
        root = config.project.root
        old = params.get("old")
        if not old:
            raise _refuse("verify needs --old DIR (a copy of the build tree and link script from before the move)")
        before = Path(old)
        mapping: dict[str, str] = _read_plan(config, params)["moves"]
        same = differ = absent = 0
        bad: list[str] = []
        for version in config.project.versions:
            for source, target in mapping.items():
                one, two = before / "build" / version / f"{source}.o", root / "build" / version / f"{target}.o"
                if not one.is_file():
                    continue
                if not two.is_file():
                    absent += 1
                    bad.append(f"{version}:{target} (not built)")
                elif _contents(one) == _contents(two):
                    same += 1
                else:
                    differ += 1
                    bad.append(f"{version}:{target}")
        name = f"{config.project.name}.ld"
        script = (before / name).read_text() if (before / name).is_file() else ""
        for source, target in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            script = script.replace(f"{source}.o(", f"{target}.o(")
        link = (root / name).is_file() and sorted(script.split()) == sorted((root / name).read_text().split())
        return {
            "action": "verify",
            "same": same,
            "differ": differ,
            "absent": absent,
            "bad": bad[:50],
            "link_script": link,
        }


def prove(config: Config, params: Json) -> Row:
    """The ROMs of every version, built and compared once: `make check` on the moved tree, refused with its tail when
    any ROM differs or the build fails."""
    with effort.stage("layout.prove"):
        root = config.project.root
        jobs = int(params.get("jobs") or 4)
        make = process.tool(config, "make")
        done = process.run("make check", [make, f"-j{jobs}", "check"], root, tmp=process.scratch(root), timeout=3600)
        tail = [line for line in (done.stdout + done.stderr).decode(errors="replace").splitlines() if line.strip()]
        if done.exit != 0:
            raise _refuse(f"make check exit {done.exit} after the move", missing=tuple(tail[-20:]))
        return {"proved": True, "check": tail[-5:]}


def reorg(config: Config, params: Json) -> Row:
    """The one-command form: plan from the map (and rules), apply it with includes respelled, prove the ROMs once."""
    with effort.stage("layout.reorg"):
        planned = plan(config, params)
        applied = apply(config, params)
        out: Row = {
            "action": "reorg",
            "moved": planned["moved"],
            "kept": planned["kept"],
            "rows": planned["rows"],
            "path": planned["path"],
            "map": planned["map"],
            "config": applied["config"],
            "includes": applied["includes"],
            "makefile": applied["makefile"],
            "written": applied["written"],
            "held": applied["held"],
            "next": applied["next"],
        }
        if params.get("prove"):
            out.update(prove(config, params))
        return out


def mergecheck(config: Config, params: Json) -> Row:
    """Which modules of a module map may become one source file and which must stay separate files, from the link
    order and the built objects of every version (`ld` and `obj` are patterns over {version} and {unit})."""
    with effort.stage("layout.mergecheck"):
        path = params.get("modules")
        if not path:
            raise _refuse("mergecheck needs --modules FILE (a TSV with `unit` and `module` columns)")
        root = config.project.root
        try:
            rows = merging.read_modules(Path(path).read_text())
            seqs, statics = merging.facts_of(
                root,
                config.project.versions,
                params.get("ld") or "build/{version}/link.ld",
                params.get("obj") or "build/{version}/{unit}.o",
            )
        except (OSError, ValueError) as error:
            raise _refuse(f"mergecheck: {error}") from error

        def identity(unit: str) -> Any:
            if unit.lower().endswith(".s"):
                return "assembler"
            row = file_facts(config.project, unit)
            return (row.toolchain or config.project.toolchain, tuple(sorted(row.cflags)), tuple(sorted(row.omit)))

        modules = merging.verdicts(rows, seqs, statics, identity)
        separate = sum(1 for m in modules if m["verdict"] == "separate")
        return {"action": "mergecheck", "modules": modules, "separate": separate, "mergeable": len(modules) - separate}


def run(config: Config, params: Json) -> Row:
    """`unbake layout ACTION`: plan, apply, literals, carry, verify, reorg or mergecheck."""
    actions = {
        "plan": plan,
        "apply": apply,
        "literals": literals,
        "carry": carry,
        "verify": verify,
        "reorg": reorg,
        "mergecheck": mergecheck,
    }
    action = params["action"]
    if action not in actions:
        raise _refuse(f"unknown action {action}: use {', '.join(actions)}")
    return actions[action](config, params)
