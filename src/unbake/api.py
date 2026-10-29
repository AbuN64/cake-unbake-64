"""The read-only surface Recipe Maker imports: the source tree, the symbol facts, cached scores and the caches and
parallel map the tree's readers share. Nothing else of unbake is public."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from unbake import (
    adapters,
    effort,
    human,
    naming_baseline,
    pool,
    process,
    recipes,
    renameproof,
    store,
    symbol_names,
    tree,
)
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
    "guard_content",
    "guard_name",
    "layout",
    "load",
    "make",
    "memo",
    "name_bindings",
    "name_issues",
    "naming_census",
    "naming_debt",
    "naming_findings",
    "pool_map",
    "preprocess",
    "record_scope",
    "score",
    "snapshot",
    "stage",
    "summary",
    "symbols",
    "tree_epoch",
    "tree_lock",
]


def load(root: Path, host: Path) -> Config:
    return _config.load(root, host)


def layout(root: Path) -> Json:
    """The project's layout at `root` (repo.toml [layout] with its config.toml [layout] keys over it): source_roots,
    header_roots, library_source, library_header, subsystem_source, subsystem_header, shared_headers, module_cap."""
    return _config.layout_at(root)


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
    compile argv over {in} and {out}), `toolchain`, `cppflags`, `cflags`, `asm_path` and `asm` (the target assembly,
    None when that file is absent) and `assemble` (the argv that assembles it into {out}, as unbake assembles a
    target); `lines` holds per line of `source` the [file, line] it was written on, which a diagnostic that names a
    line of `source` is read through."""
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
            "includes": [
                f"-I{root / 'asm' / version}",
                f"-I{root / 'asm' / version / 'include'}",
                f"-I{root / 'include'}",
            ],
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
        # None once the function is C in this version (an exact match deletes its assembly; a name the version keeps
        # under another symbol's file has none): the module still builds, only a permuter target is missing
        "asm": (root / target).read_text(encoding="utf-8", errors="replace") if (root / target).is_file() else None,
        "assemble": assemble,
    }


def cached(config: Config, kind: str, key: str, produce: Callable[[], bytes]) -> bytes:
    return store.cached(config, kind, key, produce)


def memo[T](key: object, produce: Callable[[], T]) -> T:
    return effort.memo(key, produce)


def make(config: Config, targets: Sequence[str]) -> tuple[str, str]:
    """`make -k TARGETS` in the project under the host's one jobserver (`process.make`): what a tool builds runs with
    the host's slots and compiler cache, never a `-j` of its own. (stdout, stderr) as text."""
    ran = process.make(config, targets, len(targets))
    return ran.stdout.decode(errors="replace"), ran.stderr.decode(errors="replace")


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


def tree_epoch(config: Config) -> int:
    """How many exclusive holds of the work tree have ended: a change since a command began says a writer ran."""
    from unbake import treelock

    return treelock.epoch(config.project.root)


def name_issues(
    name: str, kind: str = "symbol", taken: Sequence[str] = (), implementation: bool = False, evidenced: bool = False
) -> tuple[str, ...]:
    """Portable naming policy, shared with source admission. Kinds: symbol, section, file. `implementation`: the SDK's
    own reserved names (`__osDisableInt`, `_G_*`) pass, exactly as source admission lets them. `evidenced`: the
    project's SDK evidence declares the name the original, so the reserved-identifier rule does not refuse it."""
    return symbol_names.issues(name, kind, taken, implementation=implementation, evidenced=evidenced)


def naming_findings(
    files: Mapping[str, str], cache: Path | None = None, evidenced: Collection[str] = ()
) -> tuple[Finding, ...]:
    """Portable source and path names, without a host, compiler or ROM. `cache`: a file that keeps each source file's
    census between runs, so a repeated call reads only the files that changed. `evidenced`: names the project's SDK
    evidence declares original, which the reserved-identifier rule does not report (see `name_issues`)."""
    return symbol_names.findings(files, cache, evidenced)


def naming_debt(root: Path) -> Json:
    """Per-file reviewed naming debt, refreshed by successful unbake audits."""
    return dict((naming_baseline.read(root) or {}).get("files", {}))


def guard_name(path: str) -> str:
    """Canonical generated include guard for a project header path."""
    from unbake import guards

    return guards.name(path)


def guard_content(text: str) -> str | None:
    """Exact guarded header content identity, excluding only the two wrapper tokens."""
    from unbake import guards

    return guards.content(text)


def name_bindings(text: str) -> tuple[symbol_names.Binding, ...]:
    """All lexical declarations, qualified by scope and source offset; no placeholder filtering."""
    return memo(("name.bindings", text), lambda: symbol_names.bindings(text))


def naming_census(config: Config) -> Json:
    """The tracked source/header declaration census and its exact hashed input universe."""
    from unbake import process

    with tree_lock(config, exclusive=False, who="naming census"):
        root = config.project.root
        paths = sorted(p for p in process.tracked(root) if _config.is_source(p) or _config.is_header(p))
        texts = {p: (root / p).read_text(errors="surrogateescape") for p in paths}
        return symbol_names.census(texts, _head(config))


def record_scope(texts: Mapping[str, str], tag: str, path: str, line: int = 0) -> tuple[list[str], str]:
    """The C files that one definition of the record TAG, in PATH (at LINE when given), owns: itself and the files whose
    uses bind to it alone, by include. Empty with the reason when PATH holds no such definition or some use of TAG could
    bind to it and to another definition (the rename proof applies the same rule)."""
    return renameproof.scope(texts, tag, path, line)
