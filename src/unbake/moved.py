"""moved: the paths and symbols that moves renamed, so a candidate written before a move still binds after it."""

from __future__ import annotations

import os
import posixpath
import re
from collections.abc import Iterable

from unbake import effort, headers, store, symbols
from unbake.contracts import Config, Json, Snapshot

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
