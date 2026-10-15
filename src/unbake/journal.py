import base64
import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path

from unbake import effort, store
from unbake.contracts import Config, Finding, Plan, Refusal, Snapshot, sha256

_HELD: set[Path] = set()  # the projects whose journal lock this process holds across a whole landing


@contextmanager
def held(config: Config) -> Iterator[None]:
    """Hold the journal lock until the landing is written: nothing else writes meanwhile."""
    with store.exclusive(config, "journal", wait=True):
        _HELD.add(config.project.root)
        try:
            yield
        finally:
            _HELD.discard(config.project.root)


def _lock(config: Config) -> AbstractContextManager[bool | None]:
    return nullcontext() if config.project.root in _HELD else store.exclusive(config, "journal", wait=True)


@contextmanager
def reading(config: Config) -> Iterator[None]:
    """Read the project under the shared journal lock: a reader sees the old file set or the new one, never a mix.

    A swap a crash left unfinished is rolled back first, so no reader ever sees a half-written set.
    """
    if config.project.root in _HELD:
        yield
        return
    while True:
        with store.shared(config, "journal"):
            if not (config.project.root / ".unbake" / "journal.json").exists():
                yield
                return
        recover(config)


def _stage(path: Path, content: bytes) -> str:
    """Write the new bytes beside the target, durable, ready for a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".unbake-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return name


def _stage_all(root: Path, writes: dict[str, bytes | None]) -> dict[str, str]:
    """Stage every new content before the lock is taken, so the swap under it is only renames."""
    staged: dict[str, str] = {}
    try:
        for path, content in writes.items():
            if content is not None:
                staged[path] = _stage(root / path, content)
    except BaseException:
        _discard(staged)
        raise
    return staged


def _discard(staged: dict[str, str]) -> None:
    for name in staged.values():
        Path(name).unlink(missing_ok=True)


def _swap(root: Path, writes: dict[str, bytes | None], staged: dict[str, str]) -> None:
    """Under the journal lock: rename the staged files into place in one quick pass, then make the entries durable."""
    for path, content in writes.items():
        if content is None:
            (root / path).unlink(missing_ok=True)
        else:
            os.replace(staged[path], root / path)
            del staged[path]
    for directory in {(root / path).parent for path in writes if (root / path).parent.exists()}:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)  # the directory entries are durable too
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _install(path: Path, content: bytes | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if content is None:
        path.unlink(missing_ok=True)
    elif _read(path) == content:
        return
    else:
        fd, name = tempfile.mkstemp(prefix=".unbake-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)  # the directory entry is durable too
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _sha(content: bytes | None) -> str | None:
    return None if content is None else sha256(content)


def changed(snapshot: Snapshot, paths: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        p for p in paths if p in snapshot.reads and _sha(_read(snapshot.config.project.root / p)) != snapshot.reads[p]
    )


def apply(config: Config, plan: Plan, snapshot: Snapshot) -> tuple[str, ...]:
    """Write the plan's files into the working tree only (never Git): each path must hold what the snapshot read."""
    with effort.stage("journal.apply"):
        if plan.blocking:
            raise ValueError("A blocking plan cannot be applied.")
        root = config.project.root
        paths = sorted(plan.writes)
        staged = _stage_all(root, {path: plan.writes[path] for path in paths})
        try:
            with _lock(config):
                return _apply_locked(config, plan, snapshot, paths, staged)
        finally:
            _discard(staged)


def _apply_locked(
    config: Config, plan: Plan, snapshot: Snapshot, paths: list[str], staged: dict[str, str]
) -> tuple[str, ...]:
    root = config.project.root
    journal = root / ".unbake" / "journal.json"
    _rollback(config)  # whoever writes clears what a crash left behind
    if moved := changed(snapshot, paths):
        raise Refusal(
            Finding(
                "journal.changed",
                "The project changed under a planned write.",
                missing=moved,
                action="run the command again",
            )
        )
    manifest = {
        "plan": plan.digest,
        "before": {
            path: base64.b64encode(old).decode() if (old := _read(root / path)) is not None else None for path in paths
        },
    }
    _install(journal, json.dumps(manifest, sort_keys=True).encode())
    _swap(root, {path: plan.writes[path] for path in paths}, staged)
    mismatched = tuple(path for path in paths if _read(root / path) != plan.writes[path])
    if mismatched:
        raise Refusal(Finding("journal.readback", "Written files differ from the plan.", missing=mismatched))
    snapshot.reads.update({path: _sha(plan.writes[path]) for path in paths})  # the snapshot now holds them
    _install(journal, None)
    return tuple(paths)


def _rollback(config: Config) -> bool:
    """Under journal.lock: put back the old bytes of a crashed apply."""
    root = config.project.root
    journal = root / ".unbake" / "journal.json"
    content = _read(journal)
    if content is None:
        return False
    for path, old in sorted(json.loads(content)["before"].items()):
        _install(root / path, base64.b64decode(old) if old is not None else None)
    _install(journal, None)
    return True


def recover(config: Config) -> bool:
    with effort.stage("journal.recover"), _lock(config):
        return _rollback(config)
