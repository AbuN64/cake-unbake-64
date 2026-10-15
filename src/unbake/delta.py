"""A derivation over every unit runs again only for the items whose inputs moved since it last ran in this tree.

The Makefile rules, the unclean verdicts and the fuzzy keys map every unit of the project. Each keeps a record per
item: its result, a stamp of its inputs other than files, and the content pin of every file it read or probed. The
next run takes an item whose stamp and pins all hold from that record and dispatches only the rest, so a landing
derives what the landed units and the headers they reach changed, never the whole project again."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping, Sequence
from hashlib import blake2b
from typing import Any

from unbake import effort, pool, store
from unbake.contracts import Config, code_identity

_FORMAT = 1  # bump when a record's shape changes: an older record reads as empty
DIR = ".unbake/delta"

Job = Callable[[Any], tuple[Any, Mapping[str, str]]]  # an item's result and the pins of the files it read


def fingerprint(value: Any) -> str:
    """A fast digest of plain frozen values (dataclasses of str, int and tuples): their pickle is stable for them."""
    return blake2b(pickle.dumps(value, 5), digest_size=16).hexdigest()


def _load(path: Any, header: tuple[Any, ...]) -> dict[str, Any]:
    try:
        record = pickle.loads(path.read_bytes())
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError, ValueError):
        return {}
    return record if isinstance(record, dict) and record.get("header") == header else {}


def map(
    config: Config,
    name: str,
    function: Job,
    items: Sequence[Any],
    key: Callable[[Any], str | None] | None,
    *,
    ident: Callable[[Any], str],
    stamp: Callable[[Any], str],
    pin: Callable[[str], str],
    context: Any = (),
    code: bool = True,
) -> list[Any]:
    """pool.map of `function` over the items whose record moved; the others keep their recorded result. `ident` names
    an item across runs, `stamp` digests its inputs other than files (cheap: it runs in the parent for every item),
    `pin` pins a file as the item's job does, `context` is what every item reads. With `code`, a record made by other
    code of the job (the job and every package function it reaches) is not used, so an install that changed other
    code keeps it; without, a record is a fact of its inputs alone (a tool install re-derives nothing)."""
    with effort.stage("delta.map"):
        path = config.project.root / DIR / f"{name}.pickle"
        header = (_FORMAT, name, code_identity(function) if code else "", context)
        record = _load(path, header)
        pins: dict[str, str] = {}

        def current(file: str) -> str:
            if file not in pins:
                pins[file] = pin(file)
            return pins[file]

        old_pins: Mapping[str, str] = record.get("pins", {})
        old_sets: Sequence[tuple[str, ...]] = record.get("sets", ())
        old_items: Mapping[str, tuple[str, int, Any]] = record.get("items", {})
        moved = {file for file, value in old_pins.items() if current(file) != value}
        held_sets = [not moved.intersection(files) for files in old_sets]
        idents = [ident(item) for item in items]
        stamps = [stamp(item) for item in items]
        out: list[Any] = [None] * len(items)
        files_of: list[tuple[str, ...] | None] = [None] * len(items)
        todo: list[int] = []
        for n, (who, mark) in enumerate(zip(idents, stamps, strict=True)):
            row = old_items.get(who)
            held = row is not None and row[0] == mark and held_sets[row[1]]
            if row is not None and held:
                out[n], files_of[n] = row[2], old_sets[row[1]]
            else:
                todo.append(n)
            effort.count(f"{name}.delta", held)
        fresh = pool.map(config, name, function, [items[n] for n in todo], key) if todo else []
        for n, (result, read) in zip(todo, fresh, strict=True):
            out[n] = result
            # a file another command rewrote while the job read it is not recorded: the item derives again next run
            files_of[n] = tuple(sorted(read)) if all(current(f) == p for f, p in read.items()) else None
        if todo or len(old_items) != len(items):
            sets: dict[tuple[str, ...], int] = {}
            rows: dict[str, tuple[str, int, Any]] = {}
            for who, mark, files, result in zip(idents, stamps, files_of, out, strict=True):
                if files is not None:
                    rows[who] = (mark, sets.setdefault(files, len(sets)), result)
            kept = {file for files in sets for file in files}
            body = {"header": header, "pins": {f: current(f) for f in sorted(kept)}, "sets": list(sets), "items": rows}
            store.write(path, pickle.dumps(body, 5))
        return out
