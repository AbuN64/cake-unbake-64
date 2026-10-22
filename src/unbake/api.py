"""The read-only surface Recipe Maker imports: the source tree, the symbol facts, cached scores and the caches and
parallel map the tree's readers share. Nothing else of unbake is public."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from unbake import adapters, effort, human, pool, process, recipes, store, tree
from unbake import config as _config
from unbake import symbols as _symbols
from unbake import versions as _versions
from unbake.contracts import Config, Finding, Json, Refusal, Snapshot, Span, StageRecord
from unbake.tree import Module, Tree, capture

__all__ = [
    "Config",
    "Finding",
    "Json",
    "Module",
    "Refusal",
    "Tree",
    "asm_path",
    "bind",
    "build_context",
    "cached",
    "capture",
    "closed",
    "command",
    "load",
    "memo",
    "pool_map",
    "preprocess",
    "score",
    "snapshot",
    "stage",
    "summary",
    "symbols",
    "tree_lock",
]


def load(root: Path, host: Path) -> Config:
    return _config.load(root, host)


def symbols(config: Config, v: str) -> dict[str, tuple[int, int | None, int | None, str]]:
    """Name -> (vram, rom, size, type) of one version's versions/<v>/symbol_addrs.txt."""
    return _symbols.load(config, v)


def asm_path(config: Config, v: str, name: str) -> str:
    """The project-relative path of the extracted assembly of a symbol: nonmatchings/ for code, data/ for data."""
    return _symbols.asm_path(config, v, name)


def score(config: Config, name: str) -> Mapping[str, float]:
    """The cached fuzzy score of a function per version (the `score` store kind); report is loaded on first use."""
    from unbake import report

    return report.scores(config, capture(config), (name,)).get(name, {})


def snapshot(config: Config) -> Snapshot:
    """A contracts.Snapshot of the working tree, holding the captured Tree."""
    held = capture(config)
    root = config.project.root
    found = _versions.read(config, lambda path: (root / path).read_bytes())
    fields: dict[str, Any] = {"files": held, "versions": found, "overlays": {}, "digest": held.digest}
    return Snapshot(config=config, commit=_head(config), **fields)


def _head(config: Config) -> str:
    return process.git(config, "rev-parse", "HEAD").stdout.decode().strip()


def preprocess(config: Config, path: str, version: str, text: bytes | None = None, markers: bool = False) -> str:
    """The source `path` as the build's compiler reads it in `version` (the Makefile's preprocess step, byte for byte);
    `text` stands for the file's contents, `markers` keeps the line markers a toolchain's -P drops."""
    return tree.preprocess(config, path, version, text, markers)


def build_context(config: Config, name: str, version: str, text: bytes | None = None) -> dict[str, Any]:
    """What decomp-permuter needs for the function `name` in `version`, from the toolchain rows the Makefile uses:
    `path` (the owning module), `source` (that module, or `text` standing for it, as the compiler reads it),
    `script` (compile.sh: the build's preprocess, compile and assemble of "$1" into the object "$3"), `cc` (the
    compile argv over {in} and {out}), `toolchain`, `cppflags`, `cflags`, `asm_path` and `asm` (the target assembly)
    and `assemble` (the argv that assembles it into {out}, as unbake assembles a target); `lines` holds per line of
    `source` the [file, line] it was written on, which a diagnostic that names a line of `source` is read through."""
    path = capture(config).owner(version, name)
    if path is None:
        raise Refusal(Finding("symbols.unknown", f"no module owns {name} in {version}", unit=name, versions=(version,)))
    recipe = recipes.of(config, path)
    root, target = config.project.root, _symbols.asm_path(config, version, name)
    cc = adapters.toolchain(config, recipe.toolchain).argv(
        "compile", codegen=recipe.cflags, name=["function"], source=["{in}"], out=["{out}"]
    )
    assemble = adapters.render(
        adapters.assemble_template(None)[0],
        {
            "as": [str(process.tool(config, "mips_as"))],
            "asflags": recipe.asflags,
            "includes": [f"-I{root / 'asm' / version / 'include'}", f"-I{root / 'include'}"],
            "source": [str(root / target)],
            "out": ["{out}"],
        },
    )
    return {
        "path": path,
        "source": tree.preprocess(config, path, version, text),
        "lines": [list(site) for site in tree.origins(config, path, version, text)],
        "script": adapters.script(config, recipe, version),
        "cc": cc,
        "toolchain": recipe.toolchain,
        "cppflags": list(recipe.cppflags),
        "cflags": list(recipe.cflags),
        "asm_path": target,
        "asm": (root / target).read_text(encoding="utf-8", errors="replace"),
        "assemble": assemble,
    }


def cached(config: Config, kind: str, key: str, produce: Callable[[], bytes]) -> bytes:
    return store.cached(config, kind, key, produce)


def memo[T](key: object, produce: Callable[[], T]) -> T:
    return effort.memo(key, produce)


def pool_map(
    config: Config,
    name: str,
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    key: Callable[[Any], str | None] | None = None,
) -> list[Any]:
    return pool.map(config, name, fn, items, key)


# ---------------------------------------------------------------- stage timing
def command(name: str, argv: Sequence[str]) -> AbstractContextManager[None]:
    """The span of one command: every stage opened inside it is recorded under it."""
    return effort.command(name, argv)


def bind(config: Config) -> None:
    """Record the stages of this process to the project's stage stream."""
    effort.bind(config)


def stage(name: str) -> AbstractContextManager[Span]:
    """One timed stage, a child of the one open around it."""
    return effort.stage(name)


def closed() -> tuple[StageRecord, ...]:
    """The stage records closed so far in this process."""
    return effort.closed()


def summary(records: Sequence[StageRecord]) -> list[str]:
    """The stage tree and the cache lines of closed records, as the human lines a command prints with --stages."""
    return human.summary(records)


def tree_lock(
    config: Config, *, exclusive: bool = False, who: str = "", say: Callable[[str], None] | None = None
) -> AbstractContextManager[float]:
    """Hold the work tree's reader/writer lock (see `unbake.treelock`); a reader waits for a ship instead of failing."""
    from unbake import treelock

    return treelock.hold(config.project.root, exclusive=exclusive, who=who, say=say or effort.say)
