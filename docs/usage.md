# Usage

## Config

unbake requires a host configuration supplied with `--host FILE` or `$UNBAKE_HOST`.
There is no default path. Without either, it refuses with "no host config".
`unbake init` writes an example; fill in its machine limits and tools before `setup`.

`cores` and `workers` accept an integer or `"auto"`; workers are bounded by cores and
the parent/worker memory budgets, and make runs with `-j` set to `workers`.

## Start

```sh
unbake init NAME
unbake setup                      # installs toolchains, extracts and proves every ROM
```

## Work

The source tree is the layout. A file owns the functions and data it defines, in the order it defines them, and a
name still behind `INCLUDE_ASM(SEG, NAME)`, `INCLUDE_DATA(SEG, NAME)`, `INCLUDE_RODATA(SEG, NAME)` or
`INCLUDE_BSS(SEG, NAME, SIZE)` belongs to the file that holds the line. `SEG` is the segment name, so moving a line
between files never changes the path it assembles from. Fuzzy C sits in `#ifdef NON_MATCHING ... #else INCLUDE_ASM ...
#endif`, and a difference between versions sits in `#if VERSION_X`. Move code by editing the files; `unbake sync`
and `make check` prove the result.

```sh
unbake sync                                          # write the extraction yaml, linker script, Makefile and CI files
unbake sync --check                                  # fail if a committed generated file would change (no ROM needed)
unbake sync --place FUNCTION [--place ...]            # add the literals_ row that puts FUNCTION's own read-only data in its file, then sync
unbake report                                        # progress per version, kind and segment
unbake compare src/code_80200610.c --function NAME   # measure a source against every version (writes nothing)
unbake score src/code_80200610.c --function NAME     # one version and the instruction diff, nothing written
unbake submit src/code_80200610.c                    # put an exact source in the landing inbox
unbake land                                          # land every inbox submission, one commit each
unbake check                                         # census the source rules over the whole project
unbake audit --against REV                           # refuse (exact rules) or warn (fuzzy) the rows a change adds over REV: docs/audit-rules.md
unbake cheat --against REV                           # refuse what a change adds over REV that could fake a match
unbake symbols                                       # the symbol table of a version
unbake map                                           # write the layout map to .unbake/map.json
```

A function that is exact in every version lands at once with a ROM proof and a commit.
A C function compiles its own read-only data (jump tables, float and string literals), so that data belongs to the
function's file, not to the INCLUDE_RODATA stand-in or word array that held it while the body was assembly.
`unbake sync --place FUNCTION` writes the one row that says so per version (`literals_FUNCTION` in `aliases.txt`, at
the address of the function's data) and lets `sync` drop the superseded stand-ins and regenerate the link script and
yamls; what is committed is what `unbake sync` writes. The data is the rows the function's extracted assembly reads
plus the rows whose words are its jump table. It refuses, writing nothing, data any other function reads (it is not
the function's own), and a built object whose `.rodata` does not fit the ROM bytes it takes the place of (size,
alignment or content); with no built object the link proves the bytes. `--place` with `--check` lists the rows and
writes nothing.
`unbake cheat` treats `ci/`, `tools/`, `asm/`, `roms/`, `config/platform.toml`, `local.mk` and `include/include_asm.h` as build and
compare infrastructure and refuses any change to them (`cheat.infra`). A project's own data files (findings, notes,
feeds, rules tables) belong in `docs/`, which the rule does not guard.
The commands are `init`, `setup`, `sync`, `check`, `audit`, `cheat`, `compare`, `score`, `submit`, `land`, `report`, `map`,
`symbols` and `compilers`.

## Compilers

`unbake compilers` says which compiler built each unit and how few directory rows say so. It reads the build the project
already has (`build/<version>/` objects and preprocessed texts) and writes only `.unbake/compilers/`.

```sh
unbake compilers fingerprint                 # shapes of every function in the ROM-equal objects; no compiler runs
unbake compilers census --paths src/IO/      # recompile the C functions with every candidate; stops after its CPU budget
unbake compilers census                      # ... and run again to resume until `remaining` is 0
unbake compilers census --drafts             # also judge the NON_MATCHING bodies against the ROM-equal object
unbake compilers propose                     # the smallest rows, dead rows, shape disagreements, drafts a row makes exact
```

`fingerprint` classes a function by letters: the epilogue (`F` stack adjust in the delay slot, `U` adjust before a
`jr; nop`), how a global address is formed (`M` assembler temporary, `S` a split register) and `+f64` for single floats
in odd FPRs. The catalog row of each toolchain (`fingerprint` in `toolchains.toml`) says which shapes it can emit, so a
unit whose code has a shape its row cannot emit is flagged unless a census run shows it exact anyway.

`census` candidates are every installed catalog toolchain of the project's default family, each with its default
flags and each of its `probe_flags` sets, plus the unit's own recipe (`--toolchains a,b` names them). A function counts
the instruction lines that differ from the ROM-equal object, trailing alignment nops aside.

`propose` reads both files and prints rows as data: `kind` `dir` or `file`, the `toolchain`, flags to `add` and `omit`,
and how many units each covers. Rows land only when `make check` passes.

## Types

`unbake types` proposes record layouts from the loads and stores in `asm/<version>/**/*.s` (flat or nested under `nonmatchings/`) and never writes a header.
It needs no host config, only the extracted assembly (`unbake setup`), `versions/<v>/symbol_addrs.txt` and the C
headers under `include/` (function signatures are also read from `src/`).

```sh
unbake types                                   # default version us-rev1; named records only, JSON on stdout
unbake types --out types.json --text types.txt # full JSON (stable keys) and the human view of proposals and diffs
unbake types --requests REQUESTS.md --out t.json   # also settle each open request line
unbake types --functions boot.tsv --out t.json   # only records those functions touch (symbol per line, or TSV with a symbol column)
unbake types --record Dest --record sym:D_800CDD10 # only these records; --all adds per-function argument records
```

Each access is one load or store: offset, width and kind (`lb`/`lbu` signed or unsigned byte, `lh`/`lhu`, `lw`/`sw`
word, `lwc1`/`swc1` float, `ldc1`/`sdc1` double). The object is a global symbol (`%hi`/`%lo`, with or without an
addend), a function argument, a pointer loaded from another object's field, or an indexed element of an array. A
register is followed through moves, `addiu` offsets and `addu` indexing inside one function only (a forward dataflow;
a register that differs between two paths is dropped), and a call clobbers the caller-saved registers: a pointer
handed to a callee is not followed into it. So an access through a register the flow lost is simply not seen.

A record is a header struct when a prototype argument, a global's declaration or a pointer field's type names it,
otherwise a global symbol (`sym:NAME`) or an argument (`arg:function#argN`). One field is proposed per start offset
(the widest access there), with `s8/u8/s16/u16/s32/f32/f64` from width and opcode, `void *` when the loaded word is
dereferenced, called, passed where the callee's prototype takes a pointer, or a symbol's address is stored in it, and
`unk<OFFSET>` unless a header already names the offset. The size reported is `observed` (the furthest access), the
header's, and for a global the distance to the next data symbol, which bounds the record but is never its size.

The diff against a header names each access that has no field (`unknown-field`), lies in a reserved array
(`hidden-in-padding`), starts inside a field (`mid-field`), has another width (`width`), passes the struct's end
(`past-end`), proves a pointer where the header has an integer (`pointer-as-int`), uses float opcodes on an integer
field (`float-as-int`) or a signed/unsigned opcode against the other (`signedness`).

With `--requests`, an open request (a `- ` line not marked SETTLED) is matched to the record it names, or failing
that to the argument records of the functions it cites, and gets one status: `answered` (every `+0x` offset has an
access: the row gives type, width, pointer and the functions), `answered-via-function` (the same, found only through a
cited function's argument), `layout` (the offsets have evidence but the line also asks for a meaning, such as a
pointee or a name, that assembly cannot give), `partial`, `no-evidence`, `no-record` or `no-offset`; `missing` says
what is absent.

## Project

A project is `config.toml` at schema 4: `name`, `toolchain`, optional `title`, `names_from`, `libraries`, `cppflags`,
`cflags`, `asflags`, one `[version.<id>]` table per ROM (`sha1`, `segments`, optional README inputs) and optional
`[file."path or dir/"]` rows (`toolchain`, `cflags`, `omit`) where the most specific row wins. An optional `[non_matching]` table (`define`, `NON_MATCHING` when absent) adds the NON_MATCHING build below. Everything else is
derived: the baserom path, the `VERSION_<ID>` macro, `versions/<id>/` and the include directories. The facts on disk
are `versions/<id>/symbol_addrs.txt` (splat format, `type:func|data rom:0x... size:0x...`) and the sources.

Every function and data symbol of a version is owned by exactly one file; `unbake sync` appends `INCLUDE_ASM` or
`INCLUDE_DATA` for an unowned name to the file that owns the previous address.

Header lint: each `extern` is declared at most once across `include/`, and one name has one declaration everywhere.

For a data batch, the manifest is a nonempty JSON list of `{"member": "data/unresolved/ADDRESS", "file": "candidate.c"}`
records. File paths are relative to the manifest. Every member must be unpublished data. All holding versions are
proved in one worker dispatch and the batch lands in one journal commit; a failed proof leaves the project untouched.
`--withhold` explicitly permits each member to land only in its exact versions, retaining extracted data elsewhere.
Every command prints one JSON object on stdout. Human text goes to stderr. Its `cache` field holds the hits and
misses of the command's cache lookups, in total and per kind, in every result (`--short` too, which drops the stage
tree it was once read from). `setup` extracts a version again exactly when the facts it was extracted from
(`symbol_addrs.txt`, `splat.yaml`, the ROM, splat) differ from the stamp in that tree's `asm/<version>/`.

See [compiler ownership](compiler-families.md) and
[publication](publication.md) for the recipe and proof contracts.

## Build without unbake

```sh
make check
```

A clone builds with make, the pinned compilers the Makefile fetches, the MIPS binutils, python3 and splat (the generated
CONTRIBUTING names the pins); a version with no `asm/<version>/` is extracted with splat first (`make extract`
does it on demand). The repository never installs or calls unbake: unbake operates on it from outside. After a
hand-run build, `unbake verify STATUS` (make's exit) records the verdict the report and sync read.

On a host that several commands share, build through the host's worker slots instead of a raw `make -jN`, which the
slots cannot see or limit: `unbake-run make -j{j} check` (the same for `extract` and `CHECK=tree`). The build takes its
fair share as a make jobserver, grows into slots other commands free and gives them back to commands that wait. The
host file comes from `UNBAKE_HOST`; without one the command runs alone on one job. `unbake setup`, `check` and `score`
already build this way; only a hand-run `make` needs `unbake-run`.

A project with a `[non_matching]` table also builds every draft in: `make NON_MATCHING=1` (or `make nm`) compiles the
sources with `-D<define>` into `build/nmraw/<version>/` and links one ROM per version, `build/nm/<version>/<name>.z64`.
A unit whose NON_MATCHING object keeps the layout of its matching one (every section the same size, every symbol at
the same offset) takes its place; any other unit keeps its matching object, so every address and size in the ROM
stays and the raw tables that name code stay right. The ROM differs from the baserom in the drafts' code only and
matches no SHA-1, so nothing checks one. `build/nm/<version>/report.txt` lists the units left out and the section
that grew.
`make check CHECK=tree` checks only that the committed generated files are present, with no ROM;
`unbake sync --check` is the currency check, run from outside (a landing runs it every round).

## CI

Setup writes the CI of a project. `ci/run` is the only place CI logic lives and it is forge-neutral POSIX sh with one
stage, `check`: `make check CHECK=tree`. It installs nothing: a game repository does not depend on unbake, so no unbake
release ever changes it. `unbake sync --check` and `unbake audit --against BASE` run where unbake operates on the tree
(a landing runs both every round). The GitHub workflow and the GitLab pipeline are thin wrappers that only call
`sh ci/run check`.
## Development

## Development

Run `bin/dev-setup` once, then `bin/test`, `bin/lint` and `bin/hygiene`.
Unit tests mock external tools; use `bin/test -p 'test_config.py'` to select an affected module.
Real build proofs run against a project checkout (`unbake setup`, `make check`), outside the unit runner.

## Merge reviewed modules

`unbake layout mergecheck --modules reviewed.tsv` audits every module in the explicit TSV (`unit` and `module` columns).
The TSV selects membership; merge and mergecheck both use code link order, then verify that every version and section
can preserve that order. Contradictory orders, intervening bytes, section boundaries and compiler changes still refuse.
The scan reads each ELF once, counts named function/data sections individually, and respects linker minimum offsets,
input alignment overrides and defined alignment markers. Missing selected objects and units absent from every version's
linker inputs refuse: merging an unplaced unit could activate its data. An internal linker hole
cannot disappear merely because the source files are concatenated. These checks do not replace the final ROM proof.
`unbake layout merge --modules reviewed.tsv --prove` prepares all of those modules, reports every refusal before writing,
then stages the selected files, rewrites the map, syncs once and proves every ROM once. `--module NAME` restricts the
same operation to one module. The response lists each changed module under `modules`; `kept` counts already merged
modules. Repeating a completed map makes no writes. Duplicate unit ownership, missing units, occupied destinations,
compiler/library changes and hard merge findings refuse the whole request. With `--prove`, text conflicts also refuse
before writing; without it they remain explicit in each module result for review. Singleton moves use `layout reorg`.

Layout apply, reorg and merge carry the configured `module_map` TSV's active `unit`, `module` and `header` paths
through the exact mapping. Other columns, including historical `evidence`, retain their values. Moving the map itself
also updates its path in config.toml. The configured map must be readable before any source move; changed map paths
are included in `written`. The explicit merge membership map is still rewritten once through its existing lifecycle.

Rename proof distinguishes macro parameters, member/tag names and ordinary free identifiers. A field such as
`(entry)->text` in a macro body does not capture a local named `text`. Free identifiers on continued directives
remain guarded, and an object macro used as a call refuses when its argument expansion cannot be proved.


### Qualified declaration facts and generated guards

The read-only `unbake.api.name_bindings(text)` contract returns every lexical C declaration with kind,
source offset, enclosing scopes and owner. It retains meaningful names, shadowed declarations,
inactive branches, macro parameters and enum constants. `naming_census(config)` derives declarations,
hashed source/header inputs and both directory views from the same tracked path universe under a
shared tree lock. These are lexical facts, not linker identities or semantic acceptance: macro
expansion, activation, assembly declarations and C++ templates require additional evidence.
The `definition` flag distinguishes a function body from a prototype and an object definition
from an `extern` declaration without an initializer. Initializers belong to individual comma
declarators; an `extern` with an initializer remains a definition. An `extern "C"` block only
sets linkage. A non-extern object without an initializer remains a definition (including C's
tentative definitions); none of these lexical facts proves version-specific storage ownership.

`unbake layout header --file include/Review.h` normalizes the reviewed header's whole-file guard
using the same rule as header generation. For a batch use `--map reviewed.tsv` with one
`path<TAB>path` identity row per reviewed header. The whole batch is checked before writes; every independently detectable owner, destination and
consumer conflict is returned together with its path.
Shared guards, occupied destinations, partial guards, redefinitions, numeric guard uses and
ordinary identifier uses refuse. Only the owning empty definition and bound `ifdef`/`ifndef`/
`defined` tests change; comments and strings stay byte-identical. Compile flags and assembly
consumers participate in the refusal sweep. Repeating the operation makes no edits.
Ordinary layout moves carry a guard automatically only when it equals the generated spelling of
the original path. Other guards require explicit reviewed selection; vendor provenance is never
inferred from a spelling. Consumers outside the source tree must be regenerated and proved by their owner.
