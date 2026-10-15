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

```sh
unbake report                                        # progress per version, kind and segment
unbake report --next --count 20                      # the ordered work list
unbake compare src/code_80200610.c --function NAME   # measure a source against every version
unbake score src/code_80200610.c --function NAME     # one version and the instruction diff, nothing written
unbake score src/code_80200610.c --alloc             # which value wins each register, and why a store sits where it does
unbake submit src/code_80200610.c                    # put an exact source in the landing inbox
unbake submit --batch-data data.json                 # atomically prove and publish independent extracted data
unbake submit --bulk A.c B.c H=h.new C.c            # a source plus its PATH=FILE headers is one candidate; groups are admitted alone
unbake land                                          # land every inbox submission, one commit each
unbake check                                         # census the source rules over the whole project
unbake move relayout                                 # once before moves: the units become the ones the modules lay out
unbake move member func_A code_B                     # one structure move: OP SUBJECT TARGET, proved byte-identical or not written
recipe structure | unbake move -                     # many moves from stdin, a JSON array or JSON lines
unbake map                                           # write the layout map and its evidence to .unbake/map.json
```

A function that is exact in every version lands at once with a ROM proof and a commit.
Run `unbake --help` for every verb: `init`, `setup`, `report`, `compare`, `score`, `submit`, `land`, `check`, `move` and `map`.

A move is `{"op": "symbol|member|module|source|header|file", "subject": "...", "target": "..."}`. Moves apply in order and one proof covers them all.

For a data batch, the manifest is a nonempty JSON list of `{"member": "data/unresolved/ADDRESS", "file": "candidate.c"}`
records. File paths are relative to the manifest. Every member must be unpublished data. All holding versions are
proved in one worker dispatch and the batch lands in one journal commit; a failed proof leaves the project untouched.
`--withhold` explicitly permits each member to land only in its exact versions, retaining extracted data elsewhere.
Every command prints one JSON object on stdout. Human text goes to stderr.

See [compiler ownership](docs/compiler-families.md) and
[publication](docs/publication.md) for the recipe and proof contracts.

## Build without unbake

```sh
make setup
make check
```

## CI

Setup writes the CI of a project. Every job calls `sh ci/run STAGE`.
A push checks the tree and the files a site reads, which are `progress.json`, `docs/memory-map-<v>.svg` and `versions/<v>/symbols.json`.
The boards workflow runs once a day and feeds each board only after `progress.json` changed.
Upload boards need a repository secret, for example `BYTEMATCH_TOKEN`.

## Development

Run `bin/dev-setup` once, then `bin/test`, `bin/lint` and `bin/hygiene`.
Unit tests mock external tools; use `bin/test -p 'test_config.py'` to select an affected module.
Process and real build proofs belong in `bin/integration`, outside the unit runner.

Licensed under [GNU GPL version 3 or later](LICENSE).

`unbake recover-structs --out DIR` reads landed C and ROM evidence using the
project's existing cache, reports families and conflicts as JSON on stdout,
and writes candidates to DIR outside the project. Use `--family ID` and
`--exclude PATH` to select work; `--submit` scores affected sources and header
consumers through the existing bulk submission and landing paths, returning
receipts with the files and commit message.
