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
unbake report                                        # progress per version, kind and segment
unbake compare src/code_80200610.c --function NAME   # measure a source against every version
unbake score src/code_80200610.c --function NAME     # one version and the instruction diff, nothing written
unbake submit src/code_80200610.c                    # put an exact source in the landing inbox
unbake land                                          # land every inbox submission, one commit each
unbake check                                         # census the source rules over the whole project
unbake audit --against REV                           # refuse the rule rows a change adds over REV
unbake symbols                                       # the symbol table of a version
unbake map                                           # write the layout map to .unbake/map.json
```

A function that is exact in every version lands at once with a ROM proof and a commit.
The commands are `init`, `setup`, `sync`, `check`, `audit`, `compare`, `score`, `submit`, `land`, `report`, `map`
and `symbols`.

## Project

A project is `config.toml` at schema 4: `name`, `toolchain`, optional `title`, `names_from`, `libraries`, `cppflags`,
`cflags`, `asflags`, one `[version.<id>]` table per ROM (`sha1`, `segments`, optional README inputs) and optional
`[file."path or dir/"]` rows (`toolchain`, `cflags`, `omit`) where the most specific row wins. Everything else is
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
make extract
make check
```

`make check CHECK=tree` checks only that the committed generated files are present and current, with no ROM.

## CI

Setup writes the CI of a project. `ci/run` is the only place CI logic lives and it is forge-neutral POSIX sh:
stage `check` runs `make check CHECK=tree`, stage `audit` installs the pinned wheel and runs
`unbake audit --against BASE`. The GitHub workflow and the GitLab pipeline are thin wrappers that only call
`sh ci/run STAGE`.
## Development

## Development

Run `bin/dev-setup` once, then `bin/test`, `bin/lint` and `bin/hygiene`.
Unit tests mock external tools; use `bin/test -p 'test_config.py'` to select an affected module.
Real build proofs run against a project checkout (`unbake setup`, `make check`), outside the unit runner.
