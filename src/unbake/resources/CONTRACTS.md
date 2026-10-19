# Frozen contracts, revision 6

Frozen records and the public API of each module (revision 6). This file is shipped as `src/unbake/resources/CONTRACTS.md`; `tests/test_lint.py` reads its API table and its budgets: a module calls another only through the names in the other's row, and a public function is a row.

## Records (contracts.py)

`Json`, `digest(value) -> str`, `code_identity(*functions) -> str` (digest of the bytecode, constants and package callees they reach, once per process), `Origin`, `Finding`, `Refusal(*findings)`, `Host`, `Segment`, `VersionFacts`, `FileFacts`, `Project`, `Config`, `Version`, `UnitSpec`, `Snapshot` (config, commit, the captured `Tree`, the version records, overlays, digest), `Recipe`, `SourceView`, `NativeResult`, `Proof`, `Plan`, `Receipt`, `Submission`, `StageRecord`.

## Invariants

1. A `Refusal` always carries at least one `Finding`; every `Finding.key` is a key of `resources/data/keys.toml`.
2. `Finding.blocking=False` marks project debt. Debt never raises; it is returned and counted.
3. A `Proof` with `exact=False` has nonempty `missing` and `score < 1.0`; `exact=True` has `score == 1.0` and `symptoms == {}`. `land.not_exact` findings copy the `missing` strings and merge the proofs' symptoms.
4. Config values have no defaults. A missing or unknown key refuses with `config.schema` naming `file:dotted.path`. `"auto"` is an explicit host value for `resources.cores`/`resources.workers`; cores resolve to `len(os.sched_getaffinity(0))`; auto workers use the lesser of affinity and configured cores, bounded by the optional cap and available memory. System load does not reduce this capacity: the shared pool slots divide it between commands. `Origin.note` records the resolution. One exception: `resources.move_wait_s` is optional and reads 600 when absent, so existing host files keep loading.
Host tools used only by particular phases/commands are optional (`cpp`, `mips_as`, `mips_ld`, `mips_objcopy`, `mips_objdump`, `n64link`, `armips`). Config validates every supplied path as executable. A phase needing an omitted tool refuses with `native.missing_tool`, naming the tool and unit kind; no PATH fallback. Migration writes only supplied tools.

5. Every cache key is `contracts.digest(...)` of content (bytes digests, recipe digest, tool pins), never mtimes or reprs. A key that guards a producer holds `code_identity(producer)`, never a source file.
6. Unbake never writes Git state: no commit, no staging, no reset (read-only `rev-parse`, `ls-files`, `diff`, `show`, `log` only). Only `journal.apply` writes tracked project files, into the working tree, and only `land.drain`, `batch_data`, `blobs.apply_import` (`unbake blobs import`, under `store.exclusive(config, "structure")`, writing only `blobs.toml`) and `repo.setup` call it, `repo.setup` and  while they hold `store.exclusive(config, "structure")`, `batch_data` while it holds `store.exclusive(config, "land")` and then `store.shared(config, "structure")`, `land.drain` per claimed entry under `store.shared(config, "structure")` with the write alone serialised by `journal.lock`. `apply` stages every new content beside its target before taking `journal.lock`, then renames them all into place under it, and `layout.capture` reads under the shared side of that lock, so a capture sees the old file set or the new one. `apply` refuses a path whose current bytes differ from what the snapshot read (`Snapshot.reads`, seeded by the capture pins), journals the old bytes of every path (or absent) before writing, and returns the written paths for the caller to commit; the human output prints the exact `git add` and one-line `git commit -m`. Every other writer writes only untracked runtime files under `<project>/.unbake/` (inbox, ledger, stages, check.json), `local.mk`, view's content-keyed `build/views/<digest>/` materialisation (kept from revision 3), or the host cache, each by atomic create/rename or `O_APPEND`.
7. Only process.py starts processes, only pool.py creates worker processes, only effort.py, process.py and pool.py read rusage. Lint greps imports of `subprocess`, `multiprocessing`, `concurrent.futures`, `threading`, `os.fork`.
8. `view.all_versions` returns exactly the requested version keys.
9. Toolchain, assembler, linker, unit-kind and repository-file behaviour comes from resources; the strings `"gcc"`, `"ido"`, `"c"`, `"asm"`, `"data"`, `"resource"` appear only in resources and tests (lint).
10. Credit is state, not history: a name is matched in a version iff a file owns it as C (not behind a stand-in) and the last `make check` linked that version exact (`.unbake/check.json`, for this tree's digest). The landing gate guarantees exactness; `make check` re-verifies. The ledger is an operational log and is never read for credit.
11. Locks, all kernel `flock`, each justified, always taken in the order `land`, then `structure`, with `journal.lock` innermost: `store.exclusive("land")` (one drain or data batch at a time, never waited on: a drain that finds it held returns `running: true`; the write itself is serialised by `journal.lock`, so measuring takes no lock), `structure` (renames of paths and members:  and `repo.setup` take it exclusive and wait at most `host.move_wait_s` before refusing `setup.busy`; a drain takes it shared once per entry, once per fuzzy chunk and once around its generate, and a data batch shared for its run, so a move or setup lands between two entries of a running drain instead of excluding the drain), `structure.gate` (a writer waiting with a deadline holds it exclusive, and readers pass it shared before asking for `structure`, so arriving readers queue behind the waiting writer and cannot starve it), `store.slots` (host-wide native concurrency equals `host.workers` across all processes; blocking acquire is the signal), no cache lock (a miss produces unlocked and stores atomically, so production may fan out to the pool). There are no per-group claims.

## Layout is the source tree (schema 4)

`config.toml` is flat at schema 4 (name, toolchain, optional title, names_from, libraries, cppflags, cflags, asflags,
one `[version.<id>]` table per ROM, optional `[file."path or dir/"]` rows). There is no layout.toml, boards.toml,
types.toml or version_differences. A file owns the names it defines or keeps behind `INCLUDE_ASM(SEG, NAME)`,
`INCLUDE_DATA`, `INCLUDE_RODATA` and `INCLUDE_BSS(SEG, NAME, SIZE)`; `tree.capture(config, overlay)` reads it and
`Tree.owner(version, name)` answers who owns a name. Fuzzy C sits in `#ifdef NON_MATCHING`; a per-version difference sits
in `#if VERSION_X`. The commands are `init`, `setup`, `sync [--check]`, `check`, `audit`, `host`, `compare`, `score`, `submit`,
`land`, `report`, `map`, `symbols` and `names`. CI is `ci/run` (forge-neutral) with thin GitHub and GitLab wrappers. Policy keeps
the source rules and the `unclean` pool, and adds a header lint: each `extern` is declared at most once across `include/`
and a name has the same declaration everywhere (cached in the `header.names` kind by header text).

## Public API (each module implements exactly its row and calls only these)


| Module | Public functions |
|---|---|
| adapters | `render(template: Sequence[str], values: Mapping[str, Sequence[str]]) -> list[str]`; `assemble_template(row: Json \| None) -> tuple[tuple[str, ...], str]`; `host_tools(config: Config) -> dict[str, str]`; `row(id: str) -> Json`; `GnuAssembler(...)`; `Armips(...)`; `TemplateToolchain(...)`; `link_script(sections: Sequence[Placement], symbols: Path, trim: bool=False) -> str`; `link_commands(tools: Mapping[str, str], obj: Path, sections: Sequence[Placement], work: Path, trim: bool, *, score: bool=False) -> tuple[tuple[str, ...], ...]`; `N64Link(...)`; `chain_steps(config: Config, id: str, recipe: Recipe, include: Sequence[Path], stem: str) -> tuple[Step, ...]`; `tool_identity(config: Config, id: str) -> str`; `check_abucache() -> None`; `toolchain(config: Config, id: str) -> TemplateToolchain`; `assembler(config: Config, kind: str) -> Assembler`; `linker(config: Config) -> Linker`; `script(config: Config, recipe: Recipe, version: str) -> str` |
| alloc | `HARD: constant`; `priority(refs: int, length: int, size: int=1) -> float`; `flip(refs: int, length: int, beat: float) -> Json`; `report(function: str, greg: str, lreg: str, asm: str, votes: dict[str, Counter[str]]) -> list[Json]` |
| api | `load(root: Path, host: Path) -> Config`; `capture(config: Config, overlay: Mapping[str, bytes \| None]={}) -> Tree` (tree's); `Tree(...)`; `Module(...)`; `symbols(config: Config, v: str) -> dict[str, tuple[int, int \| None, int \| None, str]]`; `asm_path(config: Config, v: str, name: str) -> str`; `score(config: Config, name: str) -> Mapping[str, float]`; `preprocess(config: Config, path: str, version: str, text: bytes \| None=None, markers: bool=False) -> str`; `build_context(config: Config, name: str, version: str, text: bytes \| None=None) -> dict[str, Any]`; `snapshot(config: Config) -> Snapshot`; `cached(config: Config, kind: str, key: str, produce: Callable[[], bytes]) -> bytes`; `memo(key: object, produce: Callable[[], T]) -> T`; `pool_map(config: Config, name: str, fn: Callable[[Any], Any], items: Sequence[Any], key: Callable[[Any], str \| None] \| None=None) -> list[Any]`; `command(name: str, argv: Sequence[str]) -> AbstractContextManager[None]`; `bind(config: Config) -> None`; `stage(name: str) -> AbstractContextManager[Span]`; `closed() -> tuple[StageRecord, ...]`; `summary(records: Sequence[StageRecord]) -> list[str]` |
| batch_data | `run(config: Config, params: Json) -> Json` |
| blob_migration | `run(config: Config) -> Json` |
| blobs | `FILE: constant`; `sha(config: Config) -> str`; `parse(config: Config, data: bytes, file: str=FILE) -> Json`; `load(config: Config) -> Json`; `check(config: Config, document: Json, file: str=FILE) -> tuple[Finding, ...]`; `ranges(document: Json, version: str) -> tuple[Spans, Spans]`; `census(document: Json, version: str \| None=None) -> Json`; `rows(config: Config) -> list[Json]`; `merge(spans: Iterable[tuple[int, int]]) -> Spans`; `subtract(whole: Spans, cut: Spans) -> Spans`; `length(spans: Iterable[tuple[int, int]]) -> int`; `sweep(rom_size: int, code: Spans, data: Spans, platform: Spans, decoded: Spans, identified: Spans) -> Json`; `emit(document: Json) -> bytes`; `import_plan(config: Config, packet: Json) -> Plan`; `apply_import(config: Config, plan: Plan) -> Json`; `run(config: Config, params: Json) -> Json`; `shadows(config: Config, version: str, credit: Spans) -> tuple[Finding, ...]`; `Spans: alias` |
| build | `install_toolchains(config: Config, ids: Sequence[str]) -> list[Path]`; `install_toolchain(config: Config, id: str) -> Path`; `makefile(config: Config) -> bytes`; `linker(config: Config, held: Tree) -> bytes`; `splat_yaml(config: Config, v: str, held: Tree) -> bytes`; `make(config: Config, targets: Sequence[str]) -> NativeResult`; `text_sizes(config: Config, held: Tree, v: str) -> bytes`; `local_mk(config: Config) -> bytes`; `extract(config: Config, version: str \| None=None) -> NativeResult`; `inputs(config: Config) -> str`; `make_check(config: Config, held: Tree \| None=None) -> Json`; `project_inputs(project: Project) -> str`; `verify(params: Json) -> Json` |
| ci | `RUN: constant`; `wheel(version: str \| None=None) -> str`; `files() -> dict[str, bytes]`; `host(params: Json) -> Json` |
| cli | `main(argv: Sequence[str] \| None=None) -> int` |
| compare | `score_functions(config: Config, path: str, names: Sequence[str], version: str) -> dict[str, float]`; `gaps(name: str, held: Sequence[str], proofs: Sequence[Proof]) -> tuple[Finding, ...]`; `run(config: Config, params: Json) -> Json`; `submit_command(config: Config, params: Json) -> Json`; `score_command(config: Config, params: Json) -> Json` |
| config | `SCHEMA_OF: constant`; `toml(schema: str, data: bytes, file: str, key: str='config.schema', cache: Cached \| None=None) -> Json`; `validate(schema: str, value: Json, file: str) -> None`; `load_resource(name: str) -> Json`; `kinds() -> Json`; `library(project: Project, path: str) -> bool`; `is_assembly(path: str) -> bool`; `is_source(path: str) -> bool`; `is_header(path: str) -> bool`; `is_text(path: str) -> bool`; `source_stem(path: str) -> str`; `kind(name: str) -> Json`; `sections() -> Json`; `toolchains() -> Json`; `toolchain(name: str) -> Json`; `fuzzy() -> Json`; `template(name: str) -> str`; `sentence(key: str) -> str`; `load_host(path: Path) -> Host`; `load_project(root: Path, reader: Callable[[str], bytes] \| None=None, tracked: Callable[[], Iterable[str]] \| None=None) -> Project`; `load(root: Path, host: Path) -> Config`; `rom(project: Project, vid: str) -> Path`; `rom_meta(project: Project, vid: str) -> Json`; `Cached: alias` |
| delta | `DIR: constant`; `fingerprint(value: Any) -> str`; `map(config: Config, name: str, function: Job, items: Sequence[Any], key: Callable[[Any], str \| None] \| None, *, ident: Callable[[Any], str], stamp: Callable[[Any], str], pin: Callable[[str], str], context: Any=(), code: bool=True) -> list[Any]`; `Job: alias` |
| effort | `LOG_BYTES: constant`; `WAIT_NOTE: constant`; `wait_hook: constant`; `WALL_DEPTH: constant`; `expected(path: Sequence[str]) -> float \| None`; `memo(key: object, produce: Callable[[], T]) -> T`; `peek(key: object) -> Any`; `memo_valid(key: object, produce: Callable[[], T], valid: Callable[[T], bool]) -> T`; `forget(*kinds: str) -> None`; `epochs() -> dict[str, int]`; `follow(parent: Mapping[str, int]) -> None`; `release(digest_: str) -> int`; `trim_to_fork(keep: Collection[str]=(), digests: Collection[str]=()) -> int`; `held(kinds: Collection[str]) -> dict[object, Any]`; `adopt(entries: Mapping[object, Any], kinds: Collection[str]) -> None`; `mark_fork(data: int, budget: int=0) -> None`; `scoped(always: bool=False, keep: Collection[str]=()) -> Iterator[None]`; `listen(callback: Callable[[str, Json], None]) -> None`; `say(text: str) -> None`; `progress(name: str, done: int, total: int) -> None`; `closed() -> tuple[StageRecord, ...]`; `waited(seconds: float, what: str='lock', detail: str='') -> None`; `REASON_CHARS: constant`; `command(name: str, argv: Sequence[str]) -> Iterator[None]`; `bind(config: Config) -> None`; `stage(name: str) -> Iterator[Span]`; `record_native(name: str, result: NativeResult, start_ns: int) -> None`; `record_pool(name: str, items: int, jobs: int, admitted: int, start_ns: int, envelopes: Sequence[Json], dispatch_seconds: float=0.0, end_ns: int \| None=None, slot_seconds: float=0.0) -> None`; `fold(shipped: Sequence[tuple[StageRecord, bool]]) -> list[tuple[StageRecord, bool]]`; `count(kind: str, hit: bool) -> None`; `counters() -> dict[str, tuple[int, int]]`; `cache_total(stages: Json) -> Json`; `capture() -> Iterator[list[StageRecord]]`; `tree() -> Json`; `invocation() -> str` |
| human | `attach(stream: TextIO, tty: bool) -> None`; `finding(finding: Finding) -> str`; `summary(records: Sequence[StageRecord]) -> list[str]`; `finish(command: str, result: Json \| None, findings: Sequence[Finding]) -> None` |
| journal | `Reads(...)`; `held(config: Config) -> Iterator[None]`; `drop_inherited() -> None`; `holding() -> frozenset[Path]`; `adopt(roots: frozenset[Path]) -> None`; `reading(config: Config) -> Iterator[None]`; `changed(snapshot: Reads, paths: Sequence[str]) -> tuple[str, ...]`; `apply(config: Config, plan: Plan, snapshot: Reads) -> tuple[str, ...]`; `recover(config: Config) -> bool` |
| land | `submit(config: Config, request: Json, proofs: Sequence[Proof], source: bytes, origin: str, extras: Mapping[str, bytes] \| None=None) -> Submission \| None`; `submit_many(config: Config, calls: Sequence[Mapping[str, Any]]) -> list[Submission \| Refusal \| None]`; `inbox(config: Config, only: str \| None=None) -> list[Submission]`; `drain(config: Config, only: str \| None=None) -> Json`; `land_command(config: Config, params: Json) -> Json`; `check_command(config: Config, params: Json) -> Json` |
| libasm | `foreign(words: Sequence[int], vram: int, toolchain: Mapping[str, Any]) -> bool`; `render(name: str, words: Sequence[int], vram: int, symbols: Mapping[int, str]) -> str \| None`; `scan(text: str) -> dict[str, str]`; `plan(cfg: Config, held: Any) -> dict[str, bytes]`; `homes(cfg: Config, held: Any) -> dict[str, Any]` |
| native | `stamp(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe \| None=None, *, code: bool=True) -> str`; `objects(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path) -> tuple[Path, tuple[NativeResult, ...]]`; `recorded(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe) -> tuple[str, ...] \| None`; `warnings(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path) -> tuple[str, ...]`; `dumps(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, work: Path) -> tuple[str, str, str, dict[str, str]]`; `symbol_facts(cfg: Config, version: str, overlays: Mapping[str, bytes \| None] \| None=None) -> dict[str, symbols.Symbol]`; `version_records(cfg: Config, overlays: Mapping[str, bytes \| None] \| None=None) -> dict[str, Version]`; `extent(facts: Mapping[str, symbols.Symbol], name: str) -> int \| None`; `candidate(config_: Config, version: str, path: str, text: str, overrides: Json \| None=None, overlays: Mapping[str, bytes] \| None=None, names: Collection[str] \| None=None) -> tuple[Snapshot, UnitSpec, Recipe, list[str]]`; `measure(config_: Config, version: str, name: str, path: str, text: str, overrides: Json \| None=None, overlays: Mapping[str, bytes] \| None=None) -> Proof`; `measure_names(config_: Config, version: str, names: Sequence[str], path: str, text: str, overrides: Json \| None=None, overlays: Mapping[str, bytes] \| None=None) -> dict[str, Proof]` |
| policy | `scopes(text: str) -> dict[str, str]`; `header_lint(snapshot: Snapshot, only: Collection[str] \| None=None) -> tuple[Finding, ...]`; `text_findings(path: str, text: str, sdk_group: bool) -> tuple[Finding, ...]`; `evaluate(snapshot: Snapshot, path: str, text: str, view: SourceView \| None, sdk_group: bool, only: Collection[str] \| None=None) -> tuple[Finding, ...]`; `scope(before: Sequence[Finding], after: Sequence[Finding], writes: Collection[str]) -> tuple[tuple[Finding, ...], tuple[Finding, ...]]`; `fresh(snapshot: Snapshot, path: str, findings: Sequence[Finding], sdk_group: bool) -> tuple[Finding, ...]`; `census(snapshot: Snapshot, also: Sequence[tuple[Any, ...]]=(), changes: Mapping[str, tuple[str, str]] \| None=None, baseline: Snapshot \| None=None, only: Collection[str] \| None=None) -> tuple[tuple[Finding, ...], list[list[Any]]]`; `unclean(snapshot: Snapshot) -> frozenset[str]`; `unclean_rules(snapshot: Snapshot, only: Collection[str] \| None=None) -> dict[str, tuple[str, ...]]`; `credited(snapshot: Snapshot, path: str, text: str \| None=None) -> list[str]`; `pieces(snapshot: Snapshot, member: str, version: str) -> list[tuple[str, int]]`; `withheld_bytes(snapshot: Snapshot, rules: Collection[str], member: str, version: str) -> int`; `withheld(text: str, findings: Sequence[Finding], functions: Collection[str]) -> dict[str, tuple[str, ...]]`; `counts(findings: Sequence[Finding]) -> dict[str, int]`; `audit_command(params: Json) -> Json` |
| platform | `FILE: constant`; `load(config: Config) -> Json`; `read(config: Config, version: str, start: int, end: int) -> bytes`; `validate(config: Config, document: Json) -> None`; `ranges(config: Config, version: str, document: Json \| None=None) -> list[tuple[int, int]]`; `sha(config: Config) -> str` |
| pool | `KEEP: constant`; `WARM: constant`; `cores(host: Host, work: int=0) -> Iterator[int]`; `carry(name: str, table: dict[Any, Any]) -> None`; `map(config: Config, name: str, function: Callable[[Any], Any], items: Sequence[Any], key: Callable[[Any], str \| None] \| None=None) -> list[Any]`; `gather(config: Config, groups: Sequence[tuple[Any, ...]]) -> list[list[Any]]`; `jobserver(host: Host, work: int=0) -> Iterator[tuple[int, dict[str, str], tuple[int, ...]]]`; `main(argv: Sequence[str]) -> int`; `run() -> int` |
| progress | `locations(config: Any, tree: Any) -> dict[tuple[str, str], int]`; `document(config: Any, tree: Any, current: Json) -> Json`; `unmatched(config: Any, tree: Any) -> list[Json]` |
| present | `overview_path(version: str) -> str`; `OVERVIEW_WIDTH: constant`; `OVERVIEW_ROW: constant`; `OVERVIEW_LEFT: constant`; `OVERVIEW_BAR: constant`; `OVERVIEW_HEIGHT: constant`; `OVERVIEW_PITCH: constant`; `OVERVIEW_JOIN: constant`; `FUZZY_HIGH: constant`; `overview(credit_rows: Iterable[Any], version: str) -> bytes`; `files(config: Any, tree: Any) -> dict[str, bytes]` |
| process | `scratch(root: Path) -> Path`; `run(name: str, argv: Sequence[str \| Path], cwd: Path, *, tmp: Path, stdin: bytes=b'', stdout_path: Path \| None=None, timeout: float \| None=None, outputs: Mapping[str, Path]={}, env: Mapping[str, str]={}, pass_fds: Sequence[int]=()) -> NativeResult`; `tool(config: Config, name: str, *, kind: str \| None=None) -> Path`; `git(config: Config, *args: str) -> NativeResult`; `make(config: Config, targets: Sequence[str], jobs: int=0) -> NativeResult`; `tracked(root: Path) -> set[str]` |
| publish | `Item(...)`; `items(text: str) -> list[Item]`; `wrap(segment: str, name: str, body: str) -> str`; `candidate(source: str, name: str \| None) -> tuple[str, str]`; `splice(text: str, segment: str, name: str, source: str, fuzzy: bool, kind: str='function') -> str`; `splice_traced(text: str, segment: str, name: str, source: str, fuzzy: bool, kind: str='function') -> tuple[str, list[Origin]]`; `replace_traced(current: str, name: str, source: str) -> tuple[str, list[Origin]]`; `Reads(...)`; `Owner(...)`; `Origin: alias`; `Landing(...)`; `refuse_blocking(findings: Sequence[Finding]) -> None`; `make_plan(operation: str, base: str, writes: Mapping[str, bytes \| None], affected: tuple[str, ...], debt: tuple[Finding, ...], message: str) -> Plan`; `read_file(reads: Reads, path: str) -> bytes \| None`; `owners(captured: tree.Tree, name: str, version: str \| None=None) -> tuple[Owner, ...]`; `judged(config: Config, writes: Mapping[str, bytes \| None], old: Mapping[str, bytes \| None]) -> tuple[tuple[Finding, ...], tuple[Finding, ...]]`; `prepare(config: Config, request: Json, extras: Mapping[str, bytes], fuzzy: bool, message: str \| None=None) -> Landing`; `accept(config: Config, request: Json, extras: Mapping[str, bytes], fuzzy: bool) -> Landing`; `accept_each(config: Config, checks: Sequence[tuple[Json, Mapping[str, bytes], bool]]) -> list[tuple[Finding, ...]]`; `build(config: Config, versions: Sequence[str]) -> None`; `apply_checked(config: Config, landing: Landing) -> tuple[str, ...]`; `scores(config: Config, landing: Landing) -> dict[str, float]`; `before_score(config: Config, name: str) -> float`; `gained(config: Config, name: str, now: Mapping[str, float]) -> None`; `land(config: Config, submission: Submission) -> Receipt` |
| recipes | `unit(config: Config, module: Any) -> UnitSpec`; `of(config: Config, path: str) -> Recipe`; `resolve(config: Config, unit: UnitSpec, overrides: Json) -> Recipe` |
| repo | `validate_roms(roms: Mapping[str, Path], names_from: str) -> Json`; `HEADER: constant`; `init(params: Json) -> Json`; `stand_ins(config: Config, current: Any) -> dict[str, bytes]`; `sync(config: Config, params: Json) -> Json`; `symbols_command(config: Config, params: Json) -> Json`; `extract_job(job: tuple[Config, str]) -> bool`; `setup(config: Config, params: Json) -> Json` |
| report | `Symbol(...)`; `facts(config: Any) -> rescore.Facts`; `record(config: Any, tree: Any, **fields: Any) -> None`; `exact_versions(config: Any, tree: Any) -> frozenset[str]`;  `publish_hold(config: Any, tree: Any) -> Finding | None`; `progress_bytes(config: Any, tree: Any) -> bytes`; `Credit(...)`; `percent(credit: Credit) -> float`; `fuzzy(credit: Credit) -> bool`; `build_credits(versions: list[str], tree: Any, facts: rescore.Facts, exact: frozenset[str], scored: Mapping[str, Mapping[str, float]], broken: frozenset[str]) -> tuple[Credit, ...]`; `credits(config: Any, tree: Any) -> tuple[Credit, ...]`; `scores(config: Any, tree: Any) -> dict[str, dict[str, float]]`; `partition(config: Any, v: str, table: Mapping[str, Any], rom_size: int) -> Json`; `partitions(config: Any, tree: Any) -> dict[str, Json]`; `accounts(config: Any, tree: Any) -> dict[str, Json]`; `layers(config: Any, tree: Any, v: str) -> Json`; `progress(config: Any, tree: Any) -> Json`; `current(config: Any, tree: Any) -> Json`; `readme(text: str, config: Any, report: Json) -> str`; `files(config: Any, tree: Any) -> dict[str, bytes]`; `run(config: Any, params: Json) -> Json`; `map_command(config: Any, params: Json) -> Json`; `Buckets: alias`; `record_build(root: Path, inputs: str, exact: list[str], failed: str | None = None) -> None` |
| foreign | `KEY: constant`; `FLOOR: constant`; `findings(config: Config, modules: Sequence[Any], scores: Mapping[str, Mapping[str, float]]) -> tuple[Finding, ...]` |
| rescore | `KIND: constant`; `scores(config: Any, tree: _Tree, facts: Facts) -> dict[str, dict[str, float]]`; `rows(config: Any, tree: _Tree, facts: Facts) -> dict[str, dict[str, float]]`; `Fact: alias`; `Facts: alias` |
| sched | `memory_key(text: str) -> tuple[str, int, int, int] \| None`; `Pass(...)`; `hint(words: int, pick: Json, own: int) -> str`; `displacement(dump: Pass, uid: int) -> int`; `moved(everything: list[str]) -> list[tuple[str, str, str]]`; `report(function: str, passes: dict[str, str], listing: list[str], everything: list[str]) -> list[Json]` |
| store | `write(path: Path, content: bytes) -> bool`; `code() -> str`; `identity() -> str`; `installed(code: str) -> bool`; `finish(config: Config) -> None`; `finish_command(config: Config) -> None`; `loads(blob: bytes \| None) -> tuple[bool, Any]`; `Packed(...)`; `content(config: Config, named: bool=False) -> Packed`; `drop_inherited() -> None`; `cached(config: Config, kind: str, key: str, produce: Callable[[], bytes]) -> bytes`; `cached_object(config: Config, kind: str, key: str, produce: Callable[[], T]) -> T`; `get(config: Config, kind: str, key: str, named: bool=False) -> bytes \| None`; `put(config: Config, kind: str, key: str, value: bytes, named: bool=False) -> None`; `put_many(config: Config, kind: str, rows: Iterable[tuple[str, bytes]], named: bool=False) -> None`; `stem(member: str) -> str`; `listing(config: Config, folder: str) -> frozenset[str]`; `append(config: Config, stream: str, row: Json) -> None`; `rows(config: Config, stream: str) -> list[Json]`; `log(config: Config, kind: str, body: Json) -> None`; `exclusive(config: Config, name: str, *, wait: bool, wait_s: float \| None=None) -> Iterator[bool]`; `shared(config: Config, name: str) -> Iterator[None]`; `work(config: Config) -> Iterator[Path]` |
| symbols | `names_plan(document: Json, tables: dict, bindings: Json) -> Json`; `names_command(params: Json) -> Json` (generic input is validated by `schemas/symbols.schema.json`, the one schema of symbol facts, then runs through the existing symbol codec and Recipe moves); `KIND: constant`; `path(version: str) -> str`; `aliases_path(version: str) -> str`; `handwritten(config: Config, version: str, name: str) -> bool` (splat's `/* Handwritten function */` above the extracted assembly; foreign skips such a target's score); `identifier(name: str, where: str) -> None`; `parse(text: str, where: str='symbol_addrs.txt') -> dict[str, Symbol]`; `spans(config: Config, version: str) -> tuple[tuple[int, int, int], ...]`; `merged(text: str, aliases: str, held: Sequence[tuple[int, int, int]] = (), where: tuple[str, str] = ('symbol_addrs.txt', 'aliases.txt')) -> dict[str, Symbol]`; `load(config: Config, version: str) -> dict[str, Symbol]`; `inside_data(kind: str, vram: int) -> bool`; `segment_at(config: Config, version: str, rom: int) -> str \| None`; `asm_path(config: Config, version: str, name: str) -> str`; `tokens(text: str) -> set[str]`; `rewrite(text: str, old: str, new: str) -> str`; `rename_lines(text: str, renames: Mapping[str, str]) -> str`; `correspondences(config: Config) -> tuple[Correspondence, ...]`; `join(config: Config) -> tuple[Correspondence, ...]`; `referenced(text: str, table: Collection[str]) -> set[str]`; `Symbol: alias`; `Correspondence: alias` |
| symptoms | `measure(built: bytes, target: bytes, section: str) -> Json`; `score(built: bytes, target: bytes) -> float`; `literals(built: bytes, target: bytes, read_built: Callable[[int, int], bytes], read_target: Callable[[int, int], bytes]) -> bytes`; `shift(built: bytes, target: bytes) -> Json`; `register_votes(built: bytes, target: bytes) -> dict[str, Counter[str]]`; `from_missing(missing: Sequence[str]) -> Json`; `merge(facts: Sequence[Json]) -> Json`; `listing(data: bytes, vram: int) -> list[str]`; `diff(built: bytes, target: bytes, vram: int) -> str` |
| tree | `KIND: constant`; `LITERALS: constant`; `SOURCE_SUFFIXES: constant`; `Module(...)`; `Tree(...)`; `macro(version: str) -> str`; `preprocess(config: Config, path: str, version: str, text: bytes \| None=None, markers: bool=False) -> str`; `defined(text: str) -> list[str]`; `non_matching_blocks(lines: Sequence[str]) -> list[tuple[int, int, int]]`; `fuzzy_only(text: str, names: Collection[str]) -> str`; `origins(config: Config, path: str, version: str, text: bytes \| None=None) -> list[tuple[str, int]]`; `parse(config: Config, path: str, text: bytes, version: str, read: Callable[[str], bytes \| None] \| None=None, overlays: Mapping[str, bytes \| None] \| None=None) -> tuple[list[str], list[str], list[str]]`; `capture(config: Config, overlay: Mapping[str, bytes \| None]={}) -> Tree` |
| versions | `Row(...)`; `parse_symbol_addrs(data: bytes, file: str='symbol_addrs.txt') -> tuple[Row, ...]`; `symbol_file(version: str) -> str`; `symbol_addrs(config: Config, version: str, reader: Callable[[str], bytes] \| None=None) -> tuple[Row, ...]`; `segments(config: Config, version: str) -> tuple[tuple[str, int, int, int], ...]`; `pieces(config: Config, version: str) -> tuple[tuple[str, int, int, int], ...]`; `segment_of(config: Config, version: str, row: Row) -> str`; `asm_path(config: Config, version: str, name: str) -> Path`; `symbols_ld(config: Config, version: str, overlays: Mapping[str, bytes \| None] \| None=None) -> bytes`; `read(config: Config, reader: Callable[[str], bytes], only: Collection[str] \| None=None, reuse: Mapping[str, Version] \| None=None) -> dict[str, Version]`; `facts_digest(version: Version) -> str`; `idents(data: bytes \| None) -> frozenset[str]`; `delay_slot(word: int) -> bool`; `rom_view(version: Version) -> mmap.mmap`; `rom_bytes(version: Version, start: int, end: int) -> bytes`; `undefined(obj: Path) -> tuple[str, ...]`; `resolve(version: Version, names: Sequence[str], unit: str) -> tuple[Finding, ...]`; `defined(obj: Path) -> tuple[dict[str, tuple[str, int, int]], dict[str, int]]` |
| view | `pin(snapshot: Snapshot, path: str) -> str`; `directives(content: bytes) -> tuple[tuple[bytes, bytes], ...]`; `headers(snapshot: Snapshot, unit: UnitSpec, argv: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]`; `asked(snapshot: Snapshot, unit: UnitSpec, argv: Sequence[str]) -> dict[str, str]`; `closure(snapshot: Snapshot, unit: UnitSpec, version: str) -> tuple[tuple[str, str], ...]`; `search(snapshot: Snapshot, version: str) -> list[str]`; `get(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, *, lines: bool=True) -> SourceView`; `stored(config: Config, unit: str, version: str, key: str) -> SourceView \| None`; `all_versions(snapshot: Snapshot, unit: UnitSpec, recipe: Recipe, versions: Sequence[str]) -> dict[str, SourceView]`; `active_lines(view: SourceView, path: str) -> dict[int, str]`; `diagnostic(snapshot: Snapshot, unit: UnitSpec, version: str, recipe: Recipe, text: str) -> str` |

Removed public functions: `store.claim`, `store.ledger_append`, `store.ledger`, `publish.publish`, `publish.repair`, `queue.*`, `search.run`, `report.next_action`, `build.init`, `build.setup`.

## Shapes shared between modules


- `request` (admission) = `{"file": str, "function": str | None, "overrides": {"toolchain"?: str, "add": [str], "omit": [str]}, "note": str}`.
- CLI `params` = the click parameter dict with the names declared in `commands.toml` (`-` becomes `_`).
- Every CLI result validates against `schemas/result.<command>.schema.json`; the stdout line validates against `envelope`.
- DRAIN (`result.land`, embedded in submit/compare/check results) = `{running, landed: [{id, member, operation, files, message}], refused: [{id, member, findings}]}`.
- Built bytes: `native.measure` stores each proof's built material (sections in placement order) with `store.put(config, "bytes", proof.built_sha256, material)`; readers fetch it with `store.cached(config, "bytes", sha, produce)` where `produce` raises `Refusal(Finding("store.corrupt", ...))`.
- effort events passed to `listen` callbacks: `("open", {"path": [str]})`, `("close", {"path": [str], "wall": float, "status": str})`, `("progress", {"name": str, "done": int, "total": int})`, `("say", {"text": str})`.

## Runtime paths (project)

| Path | Writer | Format |
|---|---|---|
| `.unbake/inbox/<id>.json`, `<id>.c`, `<id>.xN`, `<id>.bN` | land.submit (create-new, then rename `.part`) | submission schema; source, candidate extra and measured base copies (`extra_bases`; null means absent) |
| `.unbake/inbox/done/<id>.json`, `.unbake/inbox/refused/<id>.json` | land.drain (rename) | same file moved |
| `.unbake/ledger.jsonl` | store.log | ledger schema |
| `.unbake/stages.jsonl` | effort | stage schema (the command's row carries `slot_wait_seconds` and `lock_wait_seconds`) |
| `$XDG_RUNTIME_DIR/unbake-pool-2/waits.jsonl` | pool (`effort.waited` past `effort.WAIT_NOTE`) | one line per wait of 60 s or more, any tree of the host: `{at, pid, project, what: slots\|lock, seconds, detail}`; the newest half is kept past 1 MiB |
| `.unbake/check.json` | land.check_command, build.make_check, publish.build (`report.record`) | `{digest, exact, failing, commit?, counts?}`: the verdict of the tree whose digest it names |
| `.unbake/extract.json` | repo.setup | `{version: digest(split yaml bytes, rom sha256)}` |
| `local.mk` | repo.setup | rendered `local.mk.in` |

Symbols: `versions/<v>/symbol_addrs.txt` (splat format, one line per symbol with `type:func|data rom:0x… size:0x…`) is the hand-kept fact file of a version; `unbake sync` appends a stand-in line for every symbol no file owns.

Blobs: `blobs.toml` at the project root (schema 1) is the hand-kept fact file of the ROM content that is neither code nor data: one `[blob."<id>"]` per graphics, model, audio, world or text item with its type, state (`decoded`, `identified`, `observed`), per-version physical `rom` ranges, a `parent` plus `decoded` offsets when it lives inside a compressed stream, and evidence cited by sha256. It stores no bytes. Only `blobs.apply_import` and the explicit `blobs migrate` cutover write it. Every ROM byte is classified once, in this precedence: code, data, platform (`rom.toml [platform]`, a platform fact), decoded blobs, identified blobs, unknown; `observed` blobs count in the census but their bytes stay unknown. `code + data + platform + blobs + unknown = rom_size` per version is the blocking check `report.partition`, and `unbake report` carries `rom_size`, `platform_total`, `blobs_total`, `blobs_decoded`, `blobs_identified`, `unknown_total` and a `blobs` census per version. `data` keeps its conventional name (with `INCLUDE_DATA`/`INCLUDE_RODATA`).

Tracked generated files (`unbake sync`): `Makefile`, `<name>.ld`, `versions/<v>/splat.yaml`, `include/include_asm.h` (absent only), the CI files; the README progress block between `repo.toml readme.begin/end` is written by `unbake report`. The README block, the overview image and an existing `progress.json` are written by `unbake sync`, and `progress.json` also by `unbake report`, only when `report.publish_hold` returns none, the one rule both use: a check recorded a verdict for the tree and that verdict is of a build that completed. With no verdict they stay byte for byte and the result carries a non-blocking `sync.unchecked` note under `notes` (run `make check`, then sync); after a failed build (make exited nonzero, recorded as `build_failed` in `.unbake/check.json`) the note is `sync.build_failed` naming the failure. `sync --check` refuses (`sync.changed`) when any of them differs from what sync would write given the verdict. `unbake audit --against REV` also refuses (`audit.credit_drop`) a progress.json whose matched functions per version fall by more than the function definitions the changed sources removed.

## Templates (exact placeholder sets; validate.py checks them)

| Template | Placeholders | Renderer |
|---|---|---|
| Makefile.in | title name versions tool_defaults version_vars stamps toolchain cppflags cflags asflags compile_recipes file_blocks toolchain_rules | build.makefile |
| link.ld.in | generated segments | build.linker |
| toolchain.mk.in | stamp dir url archive sha256 files pins stamp_name | build.makefile |
| local.mk.in | assignments | build.local_mk |
| host.toml | every host value | humans |
| config.toml | name title names_from toolchain version_blocks | repo.init |
| include_asm.h.in | none (copied verbatim) | repo.init, repo.sync (absent only) |
| run.in | wheel versions | ci.files |
| ci.yml.in | checkout python | ci.files |
| gitlab-ci.yml.in | image packages | ci.files |
| CONTRIBUTING.md.in | title roms | repo.init |
| README.md.in | title begin end | repo.init |

## Source budgets (physical lines; allocations are planning numbers; lint enforces only the total: a non-blocking warning above the soft cap of 6,000 and a failure above the hard cap of 7,200)

- contracts: 653
- adapters: 410
- alloc: 110
- api: 165
- batch_data: 90
- blob_migration: 150
- blobs: 330
- build: 492
- ci: 30
- cli: 174
- compare: 512
- config: 412
- delta: 98
- effort: 729
- human: 184
- journal: 224
- land: 330
- libasm: 230
- native: 474
- policy: 1235
- platform: 100
- pool: 1276
- progress: 110
- present: 117
- process: 239
- publish: 412
- recipes: 102
- repo: 452
- report: 418
- rescore: 99
- foreign: 70
- sched: 218
- store: 379
- symbols: 274
- symptoms: 261
- tree: 474
- versions: 326
- view: 423

## Published progress

`unbake report` atomically writes `progress.json`, schema 2, from its credit rows and checked ROM partition.
`carts` contains every version: cartridge `id`, `version`, `rom_size`, `code`, `data`, `functions`
(each has `matched`, `fuzzy`, `total`), `blobs` (`decoded`, `identified`, `total`, `by_type`), `platform`, `unknown`.
Code + data + blobs + platform + unknown equals ROM size. Fuzzy bytes are additional partial credit, not matched bytes.
The `map` is for `names_from`: `version`, `rom_size`, `fns` rows `[name, rom, size, percent, path, line]`,
`blobs` rows `[id, rom, size, state, type]`, and `spans` with `data` and `platform` half-open ranges.
A function without a C definition has null path/line. Zero-byte aliases are excluded. Observed blobs remain unknown.
`report --next` adds `next` rows (`name`, `version`, `size`, `percent`, `path`, `state`), sorted by size, name, version.
Queue `state` describes measured credit: `fuzzy` for a positive score, otherwise `unmatched`, using the same
predicate as `functions_fuzzy`. It does not describe source form: a NON_MATCHING C body scoring zero remains
unmatched. Source-body state stays internal for map links, including zero-scoring C bodies.
A NON_MATCHING body remains unmatched even at 100 percent until landed. No schema-1 reader or alternate metric is kept.

`make check` builds every ROM and calls `unbake verify STATUS`, including on build failure. Verify needs no host
config: it checks size/SHA-1 against config.toml and records `exact`, `failing`, `build_inputs` in `.unbake/check.json`.
The report grants that verdict only while all build inputs and outputs retain their identity. Source debt keeps its
own tree digest in that same document. A failed build clears exact credit; a missing or corrupt ROM cannot pass.

The README blobs bar uses `blobs_total + unknown_total` as its denominator: the ROM bytes outside code, data and
platform. Decoded bytes fill the solid part, identified bytes fill the shaded part, and unknown bytes stay empty.


## Blob types and platform cutover

The only live blob types are `graphics`, `model`, `audio`, `world`, `text`; schemas for both blobs.toml and import
packets reject retired types. There is no catch-all type. Unclassified ranges stay unknown. Microcode is platform
content, never a blob. The report exposes `platform_total`; progress.json schema 2 exposes cart `platform` and
map `spans.platform`. The former boot report fields are rejected, with no alias. Consumers must change together.

The packaged `rom.toml [platform]` defines `boot = [[0, 4096]]`. A project may add its own `rom.toml`:

```toml
schema = 1
[platform]
boot = [[0, 4096]]
[platform.rsp.sdk_code]
format = "microcode: reviewed SDK binary"
state = "identified"
rom.us = [[32768, 33024]]
sha256.us = ["<SHA-256 of exactly those 256 bytes>"]
```

Boot remains the packaged range; each RSP span needs one exact SHA-256 in the same version/order. A wrong hash,
unknown version or invalid range refuses before credit. This is a declaration of an identified SDK binary, not a
heuristic that treats arbitrary bytes as microcode. Its format/name/evidence retain the identity evidence.

`unbake blobs migrate` is the explicit one-time, journaled migration. It maps texture/sprite/font/layout to graphics,
sound/sequence to audio, level to world, strings/messages to text, and keeps model. The old type is prefixed to
format unless format already names it. Reviewed physical microcode rows move into project rom.toml beside boot,
with exact hashes of their ranges; their metadata/evidence survive. Observed or compressed-only microcode and
unclassifiable/catch-all rows refuse with names for review, without writes. Classify them or remove their declarations
to leave the ranges unknown, then rerun. Normal readers never accept those legacy types.

Before writing, the migration validates both proposed files and compares the exact union of known ROM ranges
(platform plus decoded/identified blobs) per version. The journal pins both input files and writes atomically. It
then rereads both files, rechecks hashes and requires identical ranges and per-version byte totals. Moving microcode
changes its layer from blobs to platform, while known coverage, unknown bytes and total ROM accounting stay fixed.
Comments of retained TOML tables and every metadata field are preserved; the command is idempotent.

Library assembly: setup/sync extracts foreign ROM routines owned under project.libraries into clean .s source units. The pre-cutover renderer is retained; differing version renderings remain stand-ins in a library file (a library routine that lost its stand-in is rendered again, or gets it back at the end of the nearest library file, never in a game file). The extracted `.s` files are tracked source: commit them, so a tree check needs no ROM and no setup output. Tree ownership, assembler builds and exact-ROM report credit include these units. Assembly outside libraries remains policy debt and earns no matched credit. Existing source is never overwritten.
