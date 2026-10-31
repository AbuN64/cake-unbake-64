# cake-unbake-64

Tools to help unbake a cake: extract, build and prove N64 decompilations against the original ROM.

> This is a personal tool. It is not officially maintained and may change or break without notice.

## Install

On Windows use [WSL](https://learn.microsoft.com/windows/wsl/install) with Ubuntu.

```sh
sudo apt install build-essential binutils-mips-linux-gnu git python3 python3-venv
python3 -m venv ~/.venvs/unbake && . ~/.venvs/unbake/bin/activate
pip install "splat64[mips]==0.50.0" spimdisasm==1.42.4 rabbitizer==1.16.2 \
  git+https://github.com/AbuN64/cache-64 git+https://github.com/AbuN64/cake-unbake-64
```

Toolchains that assemble with `abumas` need it from [mas-n64](https://github.com/AbuN64/mas-n64/releases).

## Usage

```sh
unbake init NAME
unbake setup
unbake census --out CLASH.tsv   # names declared with more than one type
unbake unify                    # one type per name, every object byte-identical
```

`unbake --help` lists the rest. Config, project layout and CI are in [docs/usage.md](docs/usage.md).
Older-build address tables: [docs/address-tables.md](docs/address-tables.md).

## Development

Run `bin/dev-setup` once, then `bin/test`, `bin/lint` and `bin/hygiene`.

## License

Released under [GNU GPL version 3 or later](LICENSE).

## One owner per cache

AbuCache64 owns every cache, content store, worker slot, pool, executor and timing path (abucache.compile, store,
slots, pool, stages). `ci/abucache-audit`, run by the check entry point, refuses a new one here naming that owner; the
files it still lists move to abucache and leave the list, nothing joins it.
