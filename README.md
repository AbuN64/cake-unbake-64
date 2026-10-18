# AbuCakeUnbake64

Tools to help unbake a cake!

## Install

### Linux

On Debian or Ubuntu install make, a C compiler and preprocessor, MIPS binutils, git and Python.

```sh
sudo apt install build-essential binutils-mips-linux-gnu git python3 python3-venv
```

Only toolchains that assemble with `n64link` need it. Download it from the
[mas-n64 releases](https://github.com/AbuN64/mas-n64/releases) and set `tools.n64link` in the config.
A project that needs it and lacks it is refused by name.

Install unbake with splat in a virtual environment.

```sh
python3 -m venv ~/.venvs/unbake
. ~/.venvs/unbake/bin/activate
pip install "splat64[mips]==0.50.0" spimdisasm==1.42.4 rabbitizer==1.16.2 \
  git+https://github.com/AbuN64/cache-64 \
  git+https://github.com/AbuN64/cake-unbake-64
```

Install another commit with the same `pip install` line.

### Windows

Install [WSL](https://learn.microsoft.com/windows/wsl/install) with Ubuntu and follow the Linux steps inside it.

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
unbake report --next --count 20                      # the ordered work list
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
Every command prints one JSON object on stdout. Human text goes to stderr.

See [compiler ownership](docs/compiler-families.md) and
[publication](docs/publication.md) for the recipe and proof contracts.

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

## AI Usage

### Workflow

AI is used in the decompilation process and every function is verified by compiling it against the original ROM. A match can still be a fakematch or use odd C semantics. I try to mark these in source as they are discovered.

I turned these efforts into a repeatable process with [AbuCakeUnbake64](https://github.com/AbuN64/cake-unbake-64). It is built so AI can drive it against any N64 ROM.

### Personal Thoughts

AI suits decompilation well. Every function is checked deterministically against the original ROM so correctness is always provable. Decompiled code is plain source built with the original toolchain so nothing opaque runs on the player's machine and the only attack surface is the build tooling. Readable source can be ported, fixed and preserved long after its tools are gone.

While recompilations can be a fine short-term way to play a favourite game, I believe AI-driven recomps are junk and should be avoided entirely. Used this way AI carries more security risk and is more likely to leave the scene littered with broken and abandoned ports. For games as compact as most N64 titles it makes more sense to point AI at a proper decompilation instead.

## Development

Run `bin/dev-setup` once, then `bin/test`, `bin/lint` and `bin/hygiene`.
Unit tests mock external tools; use `bin/test -p 'test_config.py'` to select an affected module.
Process and real build proofs belong in `bin/integration`, outside the unit runner.

Licensed under [GNU GPL version 3 or later](LICENSE).
