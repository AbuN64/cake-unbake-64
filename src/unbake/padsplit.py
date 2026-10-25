"""An alignment pad word in front of a function: the original linker rounds an object's code up to 16 bytes and leaves
garbage there, so splat starts the function one word early and no C body can match (every byte and jump target sits
four off). The cure is a layout split: the pad word is data of the object before, the function's row starts a word
later and the function lives in a file of its own (one object cannot hold text on both sides of data). This module
writes those edits as a patch, proved under the exclusive tree lock and then taken back off the tree."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from unbake import batch_data, effort, journal, publish, repo, tree, treelock
from unbake.contracts import Config, Finding, Json, Refusal

_OPEN = re.compile(r"#\s*(?:if|ifdef|ifndef)\b")
_MID = re.compile(r"#\s*(?:else|elif)\b")
_END = re.compile(r"#\s*endif\b")


def _refuse(reason: str, **more: object) -> Refusal:
    return Refusal(Finding("land.request", reason, **more))  # type: ignore[arg-type]


def symbol_rows(text: str, member: str, rom: int, vram: int) -> str:
    """symbol_addrs text with a row for the pad word (`D_auto_<member>_padding_<ROM>`, one word at `vram`) in front of
    `member`'s row, which now starts a word later (vram, rom and any size: shifted by 4)."""
    lines = text.splitlines(keepends=True)
    found = re.compile(rf"^{re.escape(member)}(\s*=\s*)0x([0-9A-Fa-f]+);(.*)$")
    for n, line in enumerate(lines):
        row = found.match(line.rstrip("\n"))
        if row is None:
            continue
        comment = row[3]
        if (
            int(row[2], 16) != vram
            or (m := re.search(r"rom:0x([0-9A-Fa-f]+)", comment)) is None
            or int(m[1], 16) != rom
        ):
            raise _refuse(f"{member}'s row is not at vram 0x{vram:X} rom 0x{rom:X}: {line.strip()}", unit=member)
        comment = re.sub(r"rom:0x[0-9A-Fa-f]+", f"rom:0x{rom + 4:X}", comment)
        comment = re.sub(
            r"size:0x([0-9A-Fa-f]+)", lambda s: f"size:0x{int(s[1], 16) - 4:X}", comment
        )  # a row with a size keeps its end
        pad = f"D_auto_{member}_padding_{rom:X} = 0x{vram:X}; // rom:0x{rom:X}\n"
        lines[n : n + 1] = [pad, f"{member}{row[1]}0x{vram + 4:X};{comment}\n"]
        return "".join(lines)
    raise _refuse(f"no row of {member} in symbol_addrs.txt", unit=member)


def _kinds(text: str) -> list[tuple[publish.Item, str]]:
    return [(i, text[i.start : i.end]) for i in publish.items(text)]


def _guards(rows: Sequence[tuple[publish.Item, str]]) -> list[list[list[str]]]:
    """Per row, the conditionals open at it, outermost first, each as its lines (#if..., then any #elif/#else)."""
    open_: list[list[str]] = []
    out: list[list[list[str]]] = []
    for item, body in rows:
        out.append([list(g) for g in open_])
        if item.kind != "directive":
            continue
        if _OPEN.match(body):
            open_.append([body])
        elif _MID.match(body) and open_:
            open_[-1].append(body)
        elif _END.match(body) and open_:
            open_.pop()
    return out


def _ours(guards: Sequence[Sequence[str]], macros: Collection[str] | None) -> bool:
    """A piece belongs to the versions `macros` when the innermost conditional around it that names a version names one
    of them (in an #else branch: when it leaves one of them out), or when no conditional names a version."""
    if macros is None:
        return True
    named = re.compile(r"\bVERSION_\w+")
    for g in reversed(guards):
        if not any(named.search(line) for line in g):
            continue
        branch = g[-1]
        if re.match(r"#\s*else\b", branch):
            return any(not re.search(rf"\b{re.escape(m)}\b", " ".join(g[:-1])) for m in macros)
        return any(re.search(rf"\b{re.escape(m)}\b", branch) for m in macros)
    return True


def hulls(text: str, member: str, macros: Collection[str] | None = None) -> list[tuple[int, int, list[str]]]:
    """(start, end, wrappers) per run of `text` holding the pieces of `member` (its definition, its NON_MATCHING draft,
    its INCLUDE_ASM stand-in) with its #if/#else/#endif balanced, and the `#if` lines of the conditionals that still
    enclose it. With `macros` (version macros), only the pieces those versions compile: a name defined once per
    version under `#ifdef VERSION_X` moves for the asked versions, the other versions' pieces stay, and each run is
    apart. A refusal when another function sits inside a run."""
    rows = _kinds(text)
    guards = _guards(rows)
    word = re.compile(rf"\b{re.escape(member)}\b")
    allm = [n for n, (i, body) in enumerate(rows) if i.kind in ("function", "decl") and word.search(body)]
    if not allm:
        raise _refuse(f"no piece of {member} in its file", unit=member)
    groups: list[list[int]] = [[]]
    for n in allm:
        if _ours(guards[n], macros):
            groups[-1].append(n)
        elif groups[-1]:
            groups.append([])
    groups = [g for g in groups if g]
    if not groups:
        raise _refuse(f"no piece of {member} for {', '.join(sorted(macros or ()))} in its file", unit=member)
    return [_run(rows, g[0], g[-1], member) for g in groups]


def hull(text: str, member: str) -> tuple[int, int, list[str]]:
    """The one run holding every piece of `member` (see hulls)."""
    return hulls(text, member)[0]


def _condition(line: str) -> str:
    """The expression an #if/#ifdef/#ifndef/#elif line tests."""
    m = re.match(r"#\s*(ifdef|ifndef|if|elif)\b\s*(.*?)\s*(?://.*|/\*.*)?$", line.strip())
    if m is None:
        return line
    return {"ifdef": f"defined({m[2]})", "ifndef": f"!defined({m[2]})"}.get(m[1], m[2])


def _wrappers(rows: Sequence[tuple[publish.Item, str]], lo: int) -> list[str]:
    """The #if lines open at row lo. A run inside an #else/#elif branch gets the line that holds in
    that branch (`#if !(first) && (this)`), never its #if, which names the other branch."""
    stack: list[list[Any]] = []  # [#if line, conditions passed, this branch condition or None]
    for n in range(lo):
        body = rows[n][1] if rows[n][0].kind == "directive" else ""
        if _OPEN.match(body):
            stack.append([body, [], _condition(body)])
        elif _MID.match(body) and stack:
            stack[-1][1].append(stack[-1][2])
            stack[-1][2] = _condition(body) if re.match(r"#\s*elif\b", body) else None
        elif _END.match(body) and stack:
            stack.pop()
    out = []
    for line, passed, cur in stack:
        terms = [f"!({c})" for c in passed] + ([f"({cur})"] if cur else [])
        out.append("#if " + " && ".join(terms) if passed else line)
    return out


def _run(rows: Sequence[tuple[publish.Item, str]], lo: int, hi: int, member: str) -> tuple[int, int, list[str]]:
    while True:
        depth, left = 0, False
        for n in range(lo, hi + 1):
            body = rows[n][1] if rows[n][0].kind == "directive" else ""
            if _OPEN.match(body):
                depth += 1
            elif _END.match(body) or _MID.match(body):
                if depth == 0:
                    left = True
                    break
                depth -= _END.match(body) is not None
        if left:  # an #else or #endif whose #if is before the run: take the #if in
            need = 0
            for n in range(lo - 1, -1, -1):
                body = rows[n][1] if rows[n][0].kind == "directive" else ""
                need += _END.match(body) is not None
                if _OPEN.match(body):
                    if need == 0:
                        lo = n
                        break
                    need -= 1
            else:
                raise _refuse(f"{member}'s conditional has no opening #if", unit=member)
        elif depth > 0:  # an #if whose #endif is after the run
            hi += 1
            if hi >= len(rows):
                raise _refuse(f"{member}'s conditional is never closed", unit=member)
        else:
            break
    others = [i.name for i, _ in rows[lo : hi + 1] if i.kind == "function" and i.name != member]
    if others:
        raise _refuse(f"{member}'s pieces are interleaved with {others[0]}: move them to a file of their own by hand")
    return rows[lo][0].start, rows[hi][0].end, _wrappers(rows, lo)


def head(text: str) -> str:
    """What a file needs in front of its first definition or stand-in: its includes, types and externs, balanced."""
    rows = _kinds(text)
    cut = next(
        (i.start for i, body in rows if i.kind == "function" or (i.kind == "decl" and body.startswith("INCLUDE_"))),
        len(text),
    )
    kept, depth, opened = text[:cut], 0, []
    for m in re.finditer(r"^[ \t]*#[^\n]*", kept, re.M):
        if _OPEN.match(m[0].strip()):
            opened.append(m.start())
        elif _END.match(m[0].strip()) and opened:
            opened.pop()
        depth = len(opened)
    return kept[: opened[0]] if depth else kept


def split_file(text: str, members: str | Sequence[str]) -> tuple[str, str]:
    """(`text` without the members, a new file holding them): the new file has the old one's head and, around each
    member, every `#if` that enclosed it, closed again after it."""
    names, blocks = [members] if isinstance(members, str) else list(members), []
    kept, top = text, head(text).rstrip()
    for member in names:
        start, end, wrappers = hull(kept, member)
        blocks.append("".join(w + "\n" for w in wrappers) + kept[start:end] + "\n#endif" * len(wrappers))
        kept = (kept[:start].rstrip() + "\n" + kept[end:].lstrip("\n")).lstrip("\n")
    return kept, f"{top}\n\n" + "\n".join(blocks) + "\n"


def _sibling(config: Config, path: str, taken: set[str]) -> str:
    """The next unused `<stem>_<n><suffix>` beside `path`."""
    base = Path(path)
    stem = re.sub(r"_\d+$", "", base.stem)
    n = 2
    while (name := (base.parent / f"{stem}_{n}{base.suffix}").as_posix()) in taken or (
        config.project.root / name
    ).exists():
        n += 1
    return name


def _rows(path: Path) -> list[dict[str, object]]:
    try:
        rows = json.loads(path.read_text())
        ok = isinstance(rows, list) and rows and all(
            isinstance(r, dict) and isinstance(r.get("member"), str) and isinstance(r.get("version"), str)
            and isinstance(r.get("rom"), int) and isinstance(r.get("vram"), int)
            for r in rows
        )  # fmt: skip
        if not ok:
            raise ValueError("expected a nonempty list of {member, version, rom, vram} rows")
        return list(rows)
    except (OSError, ValueError) as error:
        raise _refuse(f"Invalid pad manifest: {error}", path=str(path)) from error


def run(config: Config, params: Json) -> Json:
    """submit --pad-split FILE: FILE lists {member, version, rom, vram} per function that has the pad word (the vram and
    rom are its current row's: the pad's own). The patch (--patch) holds the symbol rows, the functions moved to a file
    of their own (one per module) and the stand-in `unbake sync` writes for each pad word."""
    rows = _rows(Path(params["pad_split"]).resolve())
    with ExitStack() as held:
        held.enter_context(effort.stage("publish.pad_split"))
        held.enter_context(
            treelock.hold(config.project.root, exclusive=True, who="pad-split")
        )  # the tree, then the journal
        held.enter_context(journal.held(config))
        captured = tree.capture(config)
        reads = publish.Reads(config, {})
        old: dict[str, bytes | None] = {}
        writes: dict[str, bytes | None] = {}
        for row in rows:
            path = f"versions/{row['version']}/symbol_addrs.txt"
            if path not in old:
                old[path] = publish.read_file(reads, path)
            text = (writes.get(path) or old[path] or b"").decode()
            writes[path] = symbol_rows(text, str(row["member"]), int(str(row["rom"])), int(str(row["vram"]))).encode()
        moved: dict[str, list[str]] = {}
        for row in rows:
            owners = publish.owners(captured, str(row["member"]), str(row["version"]))
            if len(owners) != 1:
                raise _refuse(f"{len(owners)} files own {row['member']} in {row['version']}: split needs one")
            moved.setdefault(owners[0].path, [])
            if row["member"] not in moved[owners[0].path]:
                moved[owners[0].path].append(str(row["member"]))
        homes: dict[str, str] = {}
        for module, members in moved.items():
            old[module] = publish.read_file(reads, module)
            kept, new = split_file((old[module] or b"").decode(), members)
            sibling = _sibling(config, module, {*writes})
            old[sibling] = None
            writes[module], writes[sibling] = kept.encode(), new.encode()
            homes.update({m: sibling for m in members})
        names = ", ".join(sorted({str(r["member"]) for r in rows}))
        plan = publish.make_plan("publish", captured.digest, writes, tuple(sorted(writes)), (), f"split pad: {names}")
        before = {**repo.generated(config, captured), **repo.stand_ins(config, captured)}
        files = journal.apply(config, plan, reads)
        try:
            now = tree.capture(config)
            after = {**repo.generated(config, now), **repo.stand_ins(config, now)}  # what `unbake sync` would write
            shifted = {p: b for p, b in after.items() if before.get(p) != b}  # only what the split moves: no old drift
            files = (
                *files,
                *journal.apply(
                    config, publish.make_plan("generated", plan.base, shifted, (), (), ""), publish.Reads(config, {})
                ),
            )
            now = tree.capture(config)
            lost = [
                f"{r['member']} in {r['version']}"
                for r in rows
                if now.owner(str(r["version"]), str(r["member"])) != homes[str(r["member"])]
            ]
            if lost:
                raise _refuse(f"after the split {lost[0]} is not owned by its new file")
            default = f".unbake/pad-split/{plan.digest}.patch"
            made = batch_data.patch(config, files, plan.base, Path(params.get("patch") or default))
        except BaseException:
            batch_data.restore(config, files, plan.base)
            raise
        return {
            "submissions": [],
            "drain": {"running": False, "landed": [], "refused": []},
            "patch": {"path": str(made), "files": list(files), "message": plan.message},
        }
