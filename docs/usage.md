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
unbake compare src/code_80200610.c --function NAME   # measure a source against every version (writes nothing)
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
