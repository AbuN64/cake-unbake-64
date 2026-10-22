"""Atomic publication of independent extracted data, with one snapshot and pool dispatch."""

from __future__ import annotations

import json
from collections.abc import Sequence
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


def committed(config: Config, name: str) -> bytes | None:
    """The bytes HEAD has of a project file, None when HEAD lacks it."""
    if not process.git(config, "ls-tree", "HEAD", "--", name).stdout.strip():
        return None
    return process.git(config, "show", f"HEAD:{name}").stdout


def restore(config: Config, files: Sequence[str], base: str) -> None:
    """The tracked tree back as HEAD has it: each file rewritten to its committed bytes (one HEAD lacks removed)."""
    reads, old = publish.Reads(config, {}), {}
    for name in files:
        old[name] = committed(config, name)
        publish.read_file(reads, name)
    journal.apply(config, publish.make_plan("restore", base, old, (), (), "restore"), reads)


def patch(config: Config, files: Sequence[str], base: str, path: Path) -> Path:
    """The files' change over HEAD as a patch file (a file HEAD lacks as a new file), then the tracked tree back as it
    was: the change was proved (written, built whole) under the exclusive tree lock and no other lane saw its files."""
    root = config.project.root
    path = path if path.is_absolute() else root / path
    body = process.git(config, "diff", "--binary", "--full-index", "HEAD", "--", *files).stdout
    for name in files:
        if committed(config, name) is None and (root / name).is_file():
            text = (root / name).read_text(encoding="utf-8").splitlines(keepends=True)
            tail = "" if not text or text[-1].endswith("\n") else "\n\\ No newline at end of file\n"
            head = f"diff --git a/{name} b/{name}\nnew file mode 100644\n--- /dev/null\n+++ b/{name}\n"
            body += (head + f"@@ -0,0 +1,{len(text)} @@\n" + "".join("+" + line for line in text) + tail).encode()
    store.write(path, body)
    restore(config, files, base)
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
        made = patch(config, files, plan.base, Path(params.get("patch") or f".unbake/batch-data/{plan.digest}.patch"))
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
            "patch": {"path": str(made), "files": list(files), "message": plan.message},
        }
