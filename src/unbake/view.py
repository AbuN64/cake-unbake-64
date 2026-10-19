"""Preprocessed source views pinned to active project dependencies."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import zlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from unbake import adapters, effort, pool, store
from unbake import config as configuration
from unbake.contracts import Config, Finding, Recipe, Refusal, Snapshot, SourceView, UnitSpec, digest, macro, sha256


def _sha(value: bytes) -> str:
    return sha256(value)


def pin(snapshot: Snapshot, path: str) -> str:
    """The hash of a project file as the snapshot reads it, hashed once per command until store.write replaces one."""
    try:
        if path in snapshot.overlays or snapshot.tree:  # the snapshot digest covers what it overlays or holds
            return effort.memo(("pin", snapshot.digest, path), lambda: _sha(snapshot.read(path)))
        full = f"{snapshot.config.project.root}/{path}"
        stat = Path(full).stat()
        return effort.memo(("pin", full, stat.st_mtime_ns, stat.st_size), lambda: _sha(Path(full).read_bytes()))
    except FileNotFoundError:
        return "missing"


def _path(marker: str, root: Path, overlay: Path | None, real: bool = True) -> tuple[str, bool]:
    if marker.startswith("<") and marker.endswith(">"):
        return marker, False
    path = Path(marker)
    path = path if path.is_absolute() else root / path
    path = path.resolve() if real else Path(os.path.normpath(path))  # real follows links; a closure only needs names
    for base in (overlay, root):
        if base is not None and path.is_relative_to(base):
            return path.relative_to(base).as_posix(), True
    return str(path), False


_VIEW_DIR = re.compile(r"^build/views/[0-9a-f]{16}/")  # the directory a view with overlaid files is made in

_DIRECTIVE = re.compile(rb'^[ \t]*(?:#[ \t]*include|\.include)[ \t]*([<"])([^>"\n]+)[>"]', re.M)


def directives(content: bytes) -> tuple[tuple[bytes, bytes], ...]:
    return tuple(_DIRECTIVE.findall(content))


def _link_relative(overlay: Path, root: Path, path: str, content: bytes) -> None:
    """A quoted include is found beside the file naming it: link such files of the real tree into the overlay."""
    for _, name in directives(content):
        relative = os.path.normpath(Path(path).parent / name.decode())
        real, target = root / relative, overlay / relative
        if not relative.startswith("..") and real.is_file() and _link(target, real):
            _link_relative(overlay, root, relative, real.read_bytes())


def _link(target: Path, real: Path) -> bool:
    """Link `real` at `target` unless something is there; True when this call made the link."""
    if target.exists() or target.is_symlink():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.name}.{uuid.uuid4().hex}")
    pending.unlink(missing_ok=True)
    pending.symlink_to(real)
    os.replace(pending, target)  # atomic: another job making the same link finds it whole, never half-made
    return True


def headers(snapshot: Snapshot, unit: UnitSpec, argv: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The walk of `_walk`, made once per command and checked on every use: it is kept with the pin of every file it
    asked for (a probed name found nowhere pins as missing), so it holds exactly while none of them changed."""
    return _walked(snapshot, unit, argv)[0]


def asked(snapshot: Snapshot, unit: UnitSpec, argv: Sequence[str]) -> dict[str, str]:
    """Every file the walk of `headers` read or probed (the source, each reached file and each name found nowhere),
    with its pin: the walk holds while none of them moved."""
    return dict(_walked(snapshot, unit, argv)[1])


def _walked(
    snapshot: Snapshot, unit: UnitSpec, argv: Sequence[str]
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], dict[str, str]]:
    root = str(snapshot.config.project.root)
    identity = snapshot.digest if snapshot.overlays or snapshot.tree else ""  # the live tree is checked by its pins
    return effort.memo_valid(
        ("closure", root, unit.path, tuple(argv), identity),
        lambda: _walk(snapshot, unit, argv),
        lambda kept: all(pin(snapshot, path) == held for path, held in kept[1].items()),
    )


def _walk(
    snapshot: Snapshot, unit: UnitSpec, argv: Sequence[str]
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], dict[str, str]]:
    """(the project files the unit can include, the paths a quoted or forced name found nowhere would be read from)
    under the command's search flags (-iquote, -I, -isystem, -include, -imacros), read through the snapshot so a planned
    file counts. An unfound <name> is the toolchain's own; an unfound quoted name is the caller's to cover."""

    asked: dict[str, str] = {}

    def text(path: str) -> tuple[tuple[bytes, bytes], ...] | None:
        """What one file includes, read and parsed once per snapshot."""

        def read() -> tuple[tuple[bytes, bytes], ...] | None:
            content = snapshot.peek(path)
            return None if content is None else directives(content)

        held = asked[path] = pin(snapshot, path)
        return effort.memo(("includes", str(snapshot.config.project.root), path, held), read)

    dirs: dict[str, list[str]] = {"-iquote": [], "-I": [], "-isystem": []}
    forced: list[str] = []
    words = iter(argv)
    for word in words:
        flag = next((f for f in ("-include", "-imacros", "-iquote", "-isystem", "-I") if word.startswith(f)), None)
        if flag is None or word == "-I-":
            continue
        value = word[len(flag) :] or next(words, "")  # each of these flags takes the next word when bare
        if value:
            (forced if flag in ("-include", "-imacros") else dirs[flag]).append(value)
    angle = (*dirs["-I"], *dirs["-isystem"])
    found: set[str] = set()
    missing: set[str] = set()
    pending = [unit.path]

    def reach(name: str, bases: Sequence[str], quoted: bool) -> None:
        options = [os.path.normpath(os.path.join(base, name)) for base in bases]
        candidate = next((c for c in options if text(c) is not None), None)
        if candidate is None:
            missing.update(options if quoted else ())
        elif candidate not in found:
            found.add(candidate)
            pending.append(candidate)

    for name in forced:  # searched first in the working directory, then as a quoted include of the source
        reach(name, (".", os.path.dirname(unit.path), *dirs["-iquote"], *angle), True)
    while pending:
        path = pending.pop()
        for delimiter, raw in text(path) or ():
            quoted = delimiter == b'"'
            reach(raw.decode(), (os.path.dirname(path), *dirs["-iquote"], *angle) if quoted else angle, quoted)
    return (tuple(sorted(found - {unit.path})), tuple(sorted(missing))), asked


def closure(snapshot: Snapshot, unit: UnitSpec, version: str) -> tuple[tuple[str, str], ...]:
    """(path, pin) of the unit's source and every project file it can reach, pinned by content as the snapshot reads it,
    so two snapshots that give the unit the same files share every cache entry built from them."""
    return _closure(snapshot, unit, version)


def search(snapshot: Snapshot, version: str) -> list[str]:
    """The search flags a closure of the version walks with."""
    return _search(snapshot, version)


def _search(snapshot: Snapshot, version: str) -> list[str]:
    cfg = snapshot.config
    macros = configuration.load_resource("repo.toml")["splat"]["options"]["generated_asm_macros_directory"]
    return ["-Iinclude", "-Isrc", "-I" + macros.format(version=version, name=cfg.project.name)]


def _closure(snapshot: Snapshot, unit: UnitSpec, version: str) -> tuple[tuple[str, str], ...]:
    return tuple((n, pin(snapshot, n)) for n in [unit.path, *headers(snapshot, unit, _search(snapshot, version))[0]])


def _lines(
    text: str, root: Path, overlay: Path | None, source: str
) -> tuple[tuple[tuple[int, str, int], ...], list[str]]:
    lines: list[tuple[int, str, int]] = []
    files = {source}
    current, number = source, 1
    for output, text_line in enumerate(text.splitlines(), 1):
        marker = text_line.startswith("# ") and re.match(r'^# (\d+) "([^"]+)"', text_line)
        if marker:
            current, project_file = _path(marker[2], root, overlay)
            number = int(marker[1])
            if project_file:
                files.add(current)
        else:
            lines.append((output, current, number))
            number += 1
    return tuple(lines), [source, *sorted(files - {source})]


_CUT = 256  # a chunk ends after a line whose checksum is 0 mod this: about 8 KB of preprocessed text
_LEAST = 16  # lines before a chunk may end, so a repeated one-line pattern cannot make single-line chunks
_HOME = "\0"  # stands for the key a view's overlay directory is named by


def _chunks(text: str) -> list[str]:
    """The text cut where its own lines say so, so the same declarations expanded in another unit's view are the same
    chunks and are stored once. Joining the chunks with a newline gives the text back."""
    chunks, current = [], []
    for line in text.split("\n"):
        current.append(line)
        if len(current) >= _LEAST and zlib.crc32(line.encode()) % _CUT == 0:
            chunks.append("\n".join(current))
            current = []
    if current:
        chunks.append("\n".join(current))
    return chunks


def _kept(config: Config, key: str) -> tuple[str, list[tuple[str, str]]] | None:
    """A stored view: its record names the chunks of its text; a chunk the cache no longer holds makes it a miss."""
    blob = store.get(config, "view", key)
    if blob is None:
        return None
    record = json.loads(blob)
    parts = [store.get(config, "view.chunk", name) for name in record["chunks"]]
    if any(part is None for part in parts):
        return None
    text = b"\n".join(part for part in parts if part is not None).decode()
    return (text.replace(_HOME, key[:16]) if record["home"] else text), record["deps"]


def _keep(config: Config, key: str, overlaid: bool, text: str, deps: list[tuple[str, str]]) -> None:
    """Chunks first, then the record that names them, so a stored record never names a chunk that was not written."""
    home = overlaid and _HOME not in text
    parts = [p.encode() for p in _chunks(text.replace(key[:16], _HOME) if home else text)]
    names = [sha256(part)[:16] for part in parts]
    store.put_many(config, "view.chunk", zip(names, parts, strict=True))
    store.put(config, "view", key, json.dumps({"chunks": names, "deps": deps, "home": home}).encode())


_FORMAT = 1  # bump when the text a view keeps is made differently (marker rewriting, overlay naming)


def _code() -> str:
    """The bytes of this module, the one place a view's text is made (marker rewriting, overlay naming, chunking)."""
    return sha256(Path(__file__).read_bytes())


def _identity(config: Config, recipe: Recipe) -> str:
    """What makes a preprocessed text beyond the files it reads: the preprocessor command, the bytes of the tools and
    of the module that makes views, never the rest of the tool, so an install that leaves these alone keeps every
    stored view."""
    return str(
        effort.memo(
            ("view-identity", recipe.toolchain),
            lambda: digest(
                (
                    _FORMAT,
                    _code(),
                    adapters.tool_identity(config, recipe.toolchain),
                    adapters.row(recipe.toolchain).get("preprocess"),
                )
            ),
        )
    )


def _key(
    snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe
) -> tuple[str, tuple[tuple[str, str], ...], bool]:
    """The key a view is kept by, named without preprocessing or loading it: every file the build can reach, pinned. A
    view that reads an overlaid file is made in a directory named by that key, so every snapshot giving the unit the
    same files gets the same bytes."""
    try:
        source_hash = _sha(snapshot.read(unit.path))
    except FileNotFoundError:
        raise Refusal(Finding("preprocess.error", "Source file is missing.", path=unit.path)) from None
    reads = closure(snapshot, unit, version)
    overlaid = snapshot.tree or bool(snapshot.overlays.keys() & {name for name, _ in reads})
    key = digest(
        (
            unit.path,
            source_hash,
            recipe.digest,
            version,
            macro(version),
            _identity(snapshot.config, recipe),
            reads,
            overlaid,
            snapshot.tree,
        )
    )
    return key, reads, overlaid


def get(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, *, lines: bool = True) -> SourceView:
    """The preprocessed view; lines=False skips the per-line marker table that only active_lines needs."""
    with effort.stage("view.get"):
        config = snapshot.config
        if configuration.is_assembly(unit.path):
            text = snapshot.read(unit.path).decode()
            key = _sha(text.encode())
            return SourceView(
                unit.path,
                version,
                key,
                text,
                ((unit.path, key),),
                tuple((n, unit.path, n) for n in range(1, len(text.splitlines()) + 1)) if lines else (),
            )
        root = config.project.root.resolve()
        key, reads, overlaid = _key(snapshot, unit, version, recipe)
        overlay = root / "build" / "views" / key[:16] if overlaid else None

        def produce() -> tuple[str, list[tuple[str, str]]]:
            """Made in a directory of its own, removed after preprocessing: the content cache keeps text and deps."""
            home = root / "build" / "views" / uuid.uuid4().hex[:16]
            try:
                return made(home)
            finally:
                shutil.rmtree(home, ignore_errors=True)

        def made(home: Path) -> tuple[str, list[tuple[str, str]]]:
            if overlay is not None:
                selected = {path: snapshot.overlays[path] for path, _ in reads if path in snapshot.overlays}
                if snapshot.tree:
                    selected = {path: snapshot.peek(path) for path, _ in reads}
                # Views are keyed by their reads: unrelated overlays can include entire generated build files.
                for path, content in selected.items():
                    target = home / path
                    if content is not None and not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content)
                # Install replacements before linking their relative includes, so a real-tree link cannot hide one.
                for path, content in selected.items():
                    if content is not None and not snapshot.tree:
                        _link_relative(home, root, path, content)
                # A quoted include is found beside the file naming it, so every file the unit reads sits in the view
                # directory: a real header naming an overlaid one by a relative path finds the overlaid one.
                if not snapshot.tree:
                    for path, _ in reads:
                        if path not in selected and (root / path).is_file():
                            _link(home / path, root / path)
            include = [root / "include", root / "src"]
            if overlay is not None:
                include = [home / "include", include[0], home / "src", include[1]]
            if snapshot.tree:
                include = [home / "include", home / "src"]
            source = (home if overlay is not None else root) / unit.path
            work = home / ".work"
            work.mkdir(parents=True, exist_ok=True)
            out = work / "view.i"
            toolchain = adapters.toolchain(config, recipe.toolchain)
            # -P drops the line markers
            marked = replace(recipe, cppflags=tuple(f for f in recipe.cppflags if f != "-P"))
            result = toolchain.preprocess(source, marked, version, include, out=out)
            findings = list(toolchain.diagnose(result))
            stderr = result.stderr.decode(errors="replace")
            if not findings and (result.exit != 0 or result.signal is not None):
                findings.append(Finding("preprocess.error", stderr or "Preprocessing failed.", path=unit.path))
            missing = re.search(r'(\S+): No such file or directory|Cannot open include file "(\S+)"', stderr)
            if missing and (findings or result.exit != 0 or result.signal is not None):
                header = missing[1] or missing[2]
                findings.append(
                    Finding("view.include", f"Include file {header} is missing.", path=unit.path, missing=(header,))
                )
            if findings:
                raise Refusal(*findings)
            text = out.read_text()
            if overlay is not None:
                text = text.replace(str(home), str(overlay))  # the key names the directory in the text
            # a line marker names its file from the project root: the text is then the same in every checkout,
            # and the cached view and object keys (which hold no root) can be shared by all of them
            text = re.sub(rf'(?m)^(# \d+ "){re.escape(str(root))}/', r"\1", text)
            _, files = _lines(text, root, overlay, unit.path)
            files += [name for name in headers(snapshot, unit, _search(snapshot, version))[0] if name not in files]
            deps = [(path, pin(snapshot, path)) for path in files]
            return text, deps

        found = _kept(config, key)
        store.content(config).count("view", found is not None)
        if found is None:
            found = produce()
            _keep(config, key, overlay is not None, *found)
        text, deps = found
        table = _lines(text, root, overlay, unit.path)[0] if lines else ()
        return SourceView(unit.path, version, key, text, tuple((path, sha) for path, sha in deps), table)


def stored(config: Config, unit: str, version: str, key: str) -> SourceView | None:
    """The view `get` stored under its key, without its line table; None when the store no longer holds it. A process
    that keeps only a view's key (a move's parent) has its workers load the text here instead of holding it."""
    found = _kept(config, key)
    if found is None:
        return None
    text, deps = found
    return SourceView(unit, version, key, text, tuple((path, sha) for path, sha in deps), ())


def _one(item: tuple[Snapshot, UnitSpec, str, Recipe]) -> SourceView:
    return get(*item)


def all_versions(snapshot: Snapshot, unit: UnitSpec, recipe: Recipe, versions: Sequence[str]) -> dict[str, SourceView]:
    with effort.stage("view.all_versions"):
        results = pool.map(
            snapshot.config, "view.all_versions", _one, [(snapshot, unit, version, recipe) for version in versions]
        )
        return dict(zip(versions, results, strict=True))


_by_file: list[Any] = []  # the one view whose lines were last grouped by file, and that grouping


def active_lines(view: SourceView, path: str) -> dict[int, str]:
    with effort.stage("view.active_lines"):
        if not _by_file or _by_file[0] is not view:  # a policy pass asks for every header of one view in turn
            text = view.text.splitlines()
            grouped: dict[str, dict[int, str]] = {}
            for output, file, source in view.lines:
                grouped.setdefault(file, {})[source] = text[output - 1]
            _by_file[:] = [view, grouped]
        return dict(_by_file[1].get(path, {}))


def diagnostic(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, text: str) -> str:
    """Map physical view lines and out-of-file coordinates; valid source coordinates stay source coordinates. Each
    marker, file length and the view's line map are worked out once per call: a compiler log names the same few
    files thousands of times."""
    root = snapshot.config.project.root.resolve()
    paths: dict[str, str] = {}
    lengths: dict[str, int | None] = {}
    origins: list[dict[int, tuple[str, int]]] = []

    def fix(match: re.Match[str]) -> str:
        if match[1] not in paths:
            paths[match[1]] = _VIEW_DIR.sub("", _path(match[1], root, None)[0])
        path, line = paths[match[1]], int(match[2])
        if path not in lengths:
            raw = None if path.endswith(".i") else snapshot.peek(path)
            lengths[path] = None if raw is None else raw.count(b"\n") + (not raw.endswith(b"\n") and bool(raw))
        size = lengths[path]
        if not (path.endswith(".i") or configuration.is_header(path)) or (size is not None and line <= size):
            return f"{path}:{line}"
        if not origins:
            source = get(snapshot, unit, version, recipe)
            origins.append({})
            for output, p, n in source.lines:
                origins[0].setdefault(output, (p, n))
        path, line = origins[0].get(line, (path, line))
        return f"{path}:{line}"

    return re.sub(r"([\w./-]+):(\d+)", fix, text)
