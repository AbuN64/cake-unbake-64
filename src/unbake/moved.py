"""moved: the paths and symbols that moves renamed, so a candidate written before a move still binds after it."""

from __future__ import annotations

import os
import posixpath
import re
from collections.abc import Iterable, Mapping

from unbake import config, effort, headers, process, store, symbols
from unbake.contracts import Config, Finding, Json, Refusal, Snapshot

STREAM = "moved"
_LITERAL = re.compile(
    r'(^[ \t]*#[ \t]*include[^\n]*|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/)', re.M | re.S
)


def record(config: Config, rows: Iterable[tuple[str, str, str]], digest: str) -> int:
    """Append one row per (kind "path"|"symbol", old, new) change, each tagged with the layout digest after the move."""
    count = 0
    for kind, old, new in rows:
        store.append(config, STREAM, {"kind": kind, "old": old, "new": new, "digest": digest})
        count += 1
    return count


def table(config: Config) -> tuple[dict[str, str], dict[str, str]]:
    """Old -> last name for paths and symbols: a->b->c reads a->c and b->c, a name moved back is live again. The stream
    only grows, so it is read once per size and mtime within a command, not once per bound candidate."""
    path = config.project.root / ".unbake" / f"{STREAM}.jsonl"  # store.rows reads this one file
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return {}, {}
    return effort.memo((STREAM, str(path), info.st_size, info.st_mtime_ns), lambda: _table(config))


def _table(config: Config) -> tuple[dict[str, str], dict[str, str]]:
    found: dict[str, dict[str, str]] = {"path": {}, "symbol": {}}
    for row in store.rows(config, STREAM):
        names = found.get(row.get("kind", ""))
        if names is None:
            continue
        old, new = row["old"], row["new"]
        for key, value in list(names.items()):
            if value == old:
                names[key] = new
        names[old] = new
        names.pop(new, None)
        for key in [k for k, v in names.items() if k == v]:
            del names[key]
    return found["path"], found["symbol"]


def rewrite(snapshot: Snapshot, text: bytes, at: str = "") -> tuple[bytes, list[str]]:
    """The candidate text spelled for the moved paths and symbols, and the `old->new` of each change made. An old name
    that exists again in the snapshot (a later move or generate re-created it) is live, never respelled."""
    every_path, every_name = table(snapshot.config)
    paths = {old: new for old, new in every_path.items() if snapshot.peek(old) is None}
    names = {old: new for old, new in every_name.items() if old not in snapshot.layout.members}
    if not paths and not names:
        return text, []
    code, done = text.decode(errors="replace"), list[str]()
    for old, new in sorted(paths.items()):
        if posixpath.basename(old) not in code:
            continue
        changed = headers.rewrite_includes(code, {old: new}, at)
        if changed != code:
            code, done = changed, [*done, f"{old}->{new}"]
    parts = _LITERAL.split(code)  # odd parts are include lines, strings and comments: never respelled
    used = set().union(*(symbols.tokens(p) for p in parts[::2])) & names.keys()
    for old in sorted(used):
        parts[::2] = [symbols.rewrite(p, old, names[old]) for p in parts[::2]]
        done.append(f"{old}->{names[old]}")
    code = "".join(parts)
    return (code.encode() if done else text), done


def note(config: Config, rewritten: list[str], source: str) -> Json:
    """The ledger body of a rewrite made while landing (the submit result lists its own)."""
    body: Json = {"rewritten": rewritten, "source": source}
    store.log(config, "drain", body)
    return body


def _refuse(path: str, target: str, why: str) -> Refusal:
    return Refusal(
        Finding(
            "land.rebase",
            f"{path} -> {target}: {why}",
            path=target,
            action=f"rebase the change set onto {target} and submit the source with its current-path extras again",
        )
    )


def _extra_text(snapshot: Snapshot, text: bytes, old: str, target: str) -> bytes:
    if not config.is_text(target):
        return text
    text = rewrite(snapshot, text, old)[0]
    if target == old:
        return text
    paths, _ = table(snapshot.config)

    def include(match: re.Match[str]) -> str:
        for base in (posixpath.dirname(old), "include", "src"):
            path = posixpath.normpath(posixpath.join(base, match[1]))
            path = paths.get(path, path) if snapshot.peek(path) is None else path
            if snapshot.peek(path) is not None:
                if path == target:
                    return ""
                name = (
                    path.removeprefix("include/")
                    if path.startswith("include/")
                    else posixpath.relpath(path, posixpath.dirname(target))
                )
                return match[0].replace(f'"{match[1]}"', f'"{name}"', 1)
        return match[0]

    spelled = headers._INCLUDE.sub(include, text.decode())
    return (headers.wrapped(target, headers._unguarded(spelled)) if config.is_header(target) else spelled).encode()


def extras(snapshot: Snapshot, changes: Mapping[str, bytes], base: str | None = None) -> dict[str, bytes]:
    """Normalize a whole change set before overlay. Recorded submissions apply their base-to-candidate deltas to
    current destinations, including merged headers; fresh requests have no historical base and must name live paths.
    Any missing provenance, deleted destination or merge conflict refuses before a project write."""
    if not changes:
        return {}
    paths, _ = table(snapshot.config)
    revision = ""
    if base is not None:
        try:
            revision = (
                process.git(snapshot.config, "rev-parse", "--verify", f"{base}^{{commit}}").stdout.decode().strip()
            )
        except Refusal:
            old = next(iter(changes))
            target = paths.get(old, old) if snapshot.peek(old) is None else old
            raise _refuse(old, target, f"recorded base {base} is unavailable") from None
    out: dict[str, bytes] = {}
    for old, supplied in sorted(changes.items()):
        target = paths.get(old, old) if snapshot.peek(old) is None else old
        if base is None and target != old:
            raise _refuse(old, target, "the retired path has no recorded submission base")
        fresh = _extra_text(snapshot, supplied, old, target)
        if base is None:
            out[target] = fresh
            continue
        listed = process.git(snapshot.config, "ls-tree", "-z", revision, "--", old).stdout
        before = process.git(snapshot.config, "show", f"{revision}:{old}").stdout if listed else None
        before = _extra_text(snapshot, before, old, target) if before is not None else None
        current = out.get(target, snapshot.peek(target))
        if config.is_source(target) and target not in snapshot.layout.units:
            raise _refuse(old, target, "the current layout has no source unit at this destination")
        if before == fresh or current == fresh:
            if current is not None:
                out[target] = current
            continue
        if before is None:
            if current is not None:
                raise _refuse(old, target, "the new file collides with an existing destination")
            out[target] = fresh
        elif current is None:
            raise _refuse(old, target, "the recorded file was removed without a live destination")
        elif current == before:
            out[target] = fresh
        else:
            with store.work(snapshot.config) as work:
                names = [work / name for name in ("current", "base", "candidate")]
                for path, data in zip(names, (current, before, fresh), strict=True):
                    path.write_bytes(data)
                merged = process.run(
                    "git merge-file",
                    [process.tool(snapshot.config, "git"), "merge-file", "-p", *map(str, names)],
                    snapshot.config.project.root,
                    tmp=work,
                )
            if merged.exit != 0:
                raise _refuse(old, target, "the recorded change conflicts with the current destination")
            out[target] = merged.stdout
    return out


def check_primary(snapshot: Snapshot, path: str, changes: Mapping[str, bytes]) -> None:
    """A remapped extra must not be silently replaced by the primary candidate at the same destination."""
    if path in changes and snapshot.read(path) != changes[path]:
        raise _refuse(path, path, "the primary source and an extra both replace this source; combine their edits")
