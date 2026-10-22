"""Atomic publication of independent extracted data, with one snapshot and pool dispatch."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

from unbake import effort, journal, process, publish, store, tree, treelock
from unbake.contracts import Config, Finding, Json, Refusal, digest


def _manifest(path: Path) -> list[dict[str, str]]:
    try:
        rows = json.loads(path.read_text())
        if (
            not isinstance(rows, list)
            or not rows
            or any(
                not isinstance(r, dict)
                or set(r) != {"member", "file"}
                or any(not isinstance(v, str) or not v for v in r.values())
                for r in rows
            )
            or len({r["member"] for r in rows}) != len(rows)
        ):
            raise ValueError("expected a nonempty list of unique {member, file} records")
        return [{"member": r["member"], "file": str((path.parent / r["file"]).resolve())} for r in rows]
    except (OSError, ValueError) as error:
        raise Refusal(Finding("land.request", f"Invalid data manifest: {error}", path=str(path))) from error


def _patch(config: Config, files: tuple[str, ...], base: str, path: Path) -> Path:
    """The batch as a patch file over HEAD, then the tracked tree back as it was: the batch was proved (written, built
    and linked whole) under the exclusive tree lock, and no other lane ever sees its files."""
    root = config.project.root
    path = path if path.is_absolute() else root / path
    head = process.git(config, "diff", "--binary", "--full-index", "HEAD", "--", *files).stdout
    store.write(path, head)
    reads, old = publish.Reads(config, {}), {}
    for name in files:
        shown = process.git(config, "show", f"HEAD:{name}")
        old[name] = shown.stdout if shown.exit == 0 else None  # a file HEAD lacks goes again
        publish.read_file(reads, name)
    journal.apply(config, publish.make_plan("restore", base, old, (), (), "restore"), reads)
    return path


def run(config: Config, params: Json) -> Json:
    if params["files"] or params["function"]:
        raise Refusal(Finding("land.request", "--batch-data cannot be combined with files or --function."))
    rows = _manifest(Path(params["batch_data"]).resolve())
    with ExitStack() as held:
        held.enter_context(effort.stage("publish.batch_data"))
        if not held.enter_context(store.exclusive(config, "land", wait=False)):
            raise Refusal(Finding("land.request", "A landing is running; retry the data batch when idle."))
        held.enter_context(store.shared(config, "structure"))  # the drain's order: land, then structure
        held.enter_context(journal.held(config))
        held.enter_context(treelock.hold(config.project.root, exclusive=True, who="batch-data"))
        captured = tree.capture(config)
        reads = publish.Reads(config, {})
        old: dict[str, bytes | None] = {}
        texts: dict[str, str] = {}
        versions: set[str] = set()
        for row in rows:
            name = row["member"]
            for owner in publish.owners(captured, name):
                path = owner.path
                if path not in texts:
                    old[path] = publish.read_file(reads, path)
                    texts[path] = (old[path] or b"").decode()
                texts[path] = publish.splice(
                    texts[path], owner.segment, name, Path(row["file"]).read_text(), False, "data"
                )
                versions.update(owner.versions)
        writes: dict[str, bytes | None] = {p: t.encode() for p, t in texts.items()}
        blocking, debt = publish.judged(config, writes, old)
        publish.refuse_blocking(blocking)
        message = f"publish {len(rows)} data members"
        plan = publish.make_plan("publish", captured.digest, writes, tuple(sorted(writes)), debt, message)
        landing = publish.Landing(plan, reads, tuple(sorted(versions)), "", (), old)
        files = publish.apply_checked(config, landing)
        patch = _patch(config, files, plan.base, Path(params.get("patch") or f".unbake/batch-data/{plan.digest}.patch"))
        submissions = []
        for row in rows:
            identity = digest((plan.digest, row["member"]))
            submissions.append(
                {
                    "file": row["file"],
                    "files": [row["file"]],
                    "member": row["member"],
                    "id": identity,
                    "operation": "publish",
                    "exact": True,
                    "score": 1.0,
                    "findings": [],
                    "consumers": {"count": 0, "wall": 0.0},
                }
            )
        return {
            "submissions": submissions,
            "drain": {"running": False, "landed": [], "refused": []},
            "patch": {"path": str(patch), "files": list(files), "message": plan.message},
        }
