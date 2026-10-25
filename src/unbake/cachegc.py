"""Size-capped trim of the host content cache: least recently used entries out first, until it fits `cache.gc_bytes`.
abucache's own collect drops what no live manifest names; everything a live tree's manifest names stays, so a host
that builds many trees only grows. This keeps the same index (cache.db and its manifest rows) and adds the recency
order it lacks: an entry's last use is the newest manifest naming it (every command's finish writes one, and
store.touch marks the keys a command needs without reading them) or, when none names it, its store time. Entries
stored in the last half hour are kept (their command is running; abucache's 6 h grace would shield a whole batch
cycle's output). Manifests older than the last entry deleted name nothing left and go too. A deleted entry is only a
miss that recomputes. A trim takes the collect's own lock, so it never runs beside a collect. It reads the index with
sqlite, not abucache's classes.
`python -m unbake.cachegc [--host H] [--cap BYTES] [--dry-run]` prints one JSON line."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from unbake import config, store
from unbake.contracts import Json

TARGET = 0.9  # a trim deletes down to this share of the cap, so the next few commands do not trim again
PROTECT = 1800  # entries stored this recently are kept: their command is still running (a deleted one is only a miss)
BATCH = 5000
_MANIFEST, _MANIFEST_END = "manifest:", "manifest;"  # abucache's manifest keys form one key range
_SPECIAL = (_MANIFEST, "collect:")


def _sizes(db: sqlite3.Connection, directory: Path) -> tuple[int, int]:
    """(bytes of the value files, bytes of the index and its journal) now."""
    (files,) = db.execute("SELECT COALESCE(SUM(size), 0) FROM Cache WHERE filename IS NOT NULL").fetchone()
    return int(files), sum(p.stat().st_size for p in directory.glob("cache.db*") if p.is_file())


def _manifests(db: sqlite3.Connection, directory: Path) -> tuple[dict[str, float], list[tuple[float, int, float, int]]]:
    """(key -> the time of the newest manifest naming it, every manifest as (time, rowid, store_time, file bytes)).
    A big manifest is a value file, a small one sits in the row."""
    newest: dict[str, float] = {}
    found: list[tuple[float, int, float, int]] = []
    query = "SELECT value, filename, rowid, store_time, size FROM Cache WHERE key >= ? AND key < ?"
    for raw, name, rowid, stored, size in db.execute(query, (_MANIFEST, _MANIFEST_END)).fetchall():
        try:
            body = json.loads(raw if name is None else (directory / name).read_bytes())
        except (OSError, ValueError, TypeError):
            continue  # a manifest being replaced or torn names nothing for this trim
        found.append((body["at"], rowid, stored, size if name else 0))
        for key in body["keys"]:
            if newest.get(key, 0.0) < body["at"]:
                newest[key] = body["at"]
    return newest, found


def trim(directory: Path, cap: int, now: float | None = None, dry_run: bool = False) -> Json:
    """Delete the least recently used entries of the cache at `directory` until its value files and index fit
    TARGET * cap. Returns what it measured and did."""
    now = now or time.time()
    if not (directory / "cache.db").is_file():
        return {"skipped": f"no cache at {directory}"}
    with open(directory / "collect.lock", "a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"skipped": "a collect is running"}
        db = sqlite3.connect(directory / "cache.db", timeout=120, isolation_level=None)
        try:
            files, index = _sizes(db, directory)
            report: Json = {"directory": str(directory), "cap": cap, "before": files + index, "index": index}
            if files + index <= cap:
                return {**report, "deleted": 0, "freed": 0, "after": files + index}
            used, manifests = _manifests(db, directory)
            rows = db.execute("SELECT rowid, key, store_time, size, filename FROM Cache").fetchall()
            old = sorted(
                (used.get(key, 0.0) or stored, rowid, stored, size if filename else 0)
                for rowid, key, stored, size, filename in rows
                if isinstance(key, str) and not key.startswith(_SPECIAL) and now - stored >= PROTECT
            )
            del rows, used
            goal, freed, doomed, cutoff = int(cap * TARGET), 0, [], 0.0
            for used_at, rowid, stored, size in old:
                if files + index - freed <= goal:
                    break
                doomed.append((rowid, stored, size))
                freed += size
                cutoff = used_at
            for at, rowid, stored, size in manifests:
                if at <= cutoff:  # every key it names was used later than it, or is gone: the newer manifests keep them
                    doomed.append((rowid, stored, size))
                    freed += size
            gone = 0 if dry_run else _delete(db, directory, doomed)
            if not dry_run:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return {
                **report,
                "deleted": len(doomed) if dry_run else gone,
                "freed": freed,
                "after": files + index - freed,
                "dry_run": dry_run,
            }
        finally:
            db.close()


def _delete(db: sqlite3.Connection, directory: Path, doomed: list[tuple[int, float, int]]) -> int:
    """Delete the (rowid, store_time, size) rows in short transactions, then their value files. A row stored again
    since the scan has a new store_time and stays."""
    count = 0
    for start in range(0, len(doomed), BATCH):
        names: list[str] = []
        db.execute("BEGIN IMMEDIATE")
        for rowid, stored, _ in doomed[start : start + BATCH]:
            found = db.execute(
                "SELECT filename FROM Cache WHERE rowid = ? AND store_time = ?", (rowid, stored)
            ).fetchall()
            if found:
                db.execute("DELETE FROM Cache WHERE rowid = ?", (rowid,))
                count += 1
                names.extend(name for (name,) in found if name)
        db.execute("COMMIT")
        for name in names:
            with contextlib.suppress(FileNotFoundError):
                os.remove(directory / name)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unbake.cachegc", description=__doc__)
    parser.add_argument("--host", type=Path, help="host config, default ~/.config/unbake/unbake.toml")
    parser.add_argument("--cap", type=int, help="bytes; default the host config's cache.gc_bytes (8 GB when unset)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    path = args.host or config.default_host()
    if path is None:
        parser.exit(2, f"no host config: {config.NO_HOST}\n")
    host = config.load_host(path)
    cap = args.cap or host.cache_gc_bytes
    if not cap:
        parser.exit(2, "no cap: set cache.gc_bytes in the host config or pass --cap\n")
    started = time.perf_counter()
    result = trim(store.content_directory(host), cap, dry_run=args.dry_run)
    sys.stdout.write(json.dumps({**result, "seconds": round(time.perf_counter() - started, 1)}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
