"""unbake move: structure moves (symbol, type, member, module, source, header, file) applied as one proved change."""

from __future__ import annotations

import json
import os
import posixpath
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from typing import Any

from unbake import (
    build,
    compare,
    effort,
    headers,
    journal,
    layout,
    moved,
    ownership,
    policy,
    pool,
    publish,
    recipes,
    repo,
    store,
    symbols,
    view,
)
from unbake import config as configuration
from unbake import types as typedb
from unbake.contracts import Claim, Config, Finding, Group, Json, LayoutMap, Move, Refusal, Snapshot, UnitSpec, digest

_ROUNDS = 5  # partial applies a batch at most this many times
_SPLITS = 12  # rounds a command spends halving sets one cause blamed; then such sets refuse whole
_REAPPLY = 1  # proof-found boundaries re-apply a batch at most this many times per command: work lands early
_OPS = ("symbol", "type", "member", "module", "source", "header", "file", "options", "relayout")
_NAME = re.compile(r"[A-Za-z_]\w*")
_PATH = {  # any case: a project keeps the names its evidence gives (src/AI/ENEMIES/TURRETAI.CPP)
    "source": re.compile(r"src/[\w+.-]+(/[\w+.-]+)*\.(?i:c|cpp)"),
    "header": re.compile(r"include/[\w+.-]+(/[\w+.-]+)*\.(?i:h)"),
}


def _flag_set(text: str) -> Json | None:
    """The module flag set an `options` move names: JSON {"add": [...], "omit": [...]} of compiler options, each list
    optional; None when the text is no such set."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) - {"add", "omit"}:
        return None
    lists = [value.get(k, []) for k in ("add", "omit")]
    if any(not isinstance(v, list) or any(not isinstance(f, str) or not f.startswith("-") for f in v) for v in lists):
        return None
    return {"add": list(lists[0]), "omit": list(lists[1])}


def _placement(source: str, rows: Json) -> tuple[str, str]:
    """(subsystem, library) a module's source path places it in (config.layout_rows subsystem_source,
    library_source): src/sdk/libgultra/getsr.c is library libgultra, src/game/main/actor/x.c subsystem main/actor.
    A library folder is never a subsystem, also when the subsystem prefix holds it (src/{subsystem} over src/sdk)."""
    folder = source.rpartition("/")[0]
    library = rows["library_source"].split("{library}")[0]
    subsystem = rows["subsystem_source"].split("{subsystem}")[0]
    if folder.startswith(library):
        return "", folder.removeprefix(library).split("/")[0]
    return (folder.removeprefix(subsystem) if folder.startswith(subsystem) else "", "")


def _header_ok(group: Group, target: str, groups: dict[str, Group], rows: Json) -> bool:
    """Whether `target` may be the header of a module (paths from repo.toml [layout]): its subsystem's or library's
    header, a shared one, or a header no other module is bound to. A module a recipe has not placed yet (no subsystem
    or library, by field or by its source path) is limited to the last two, so it cannot take another module's
    header."""
    if not (group.subsystem or group.library):
        placed, library = _placement(group.source, rows)
        group = replace(group, subsystem=placed, library=library)
    own = {
        *([rows["subsystem_header"].format(subsystem=group.subsystem)] if group.subsystem else []),
        *([rows["library_header"].format(library=group.library)] if group.library else []),
    }
    return target in own | set(rows["shared_headers"]) or all(
        g.header != target for g in groups.values() if g is not group
    )


@dataclass
class Edited:
    layout: LayoutMap
    writes: dict[str, bytes | None]
    touched: dict[int, set[str]]
    events: list[tuple[str, str, str, frozenset[str]]]  # ("file"|"names", old header, new header, names)
    renames: dict[str, str] = field(default_factory=dict)
    bad: list[tuple[Move, Finding]] = field(default_factory=list)  # moves the map refuses
    types: dict[str, str] = field(default_factory=dict)  # type name (as the snapshot spells it) -> its new name
    folded: set[str] = field(default_factory=set)  # record tags (as the snapshot spells them) whose bodies go
    split: list[Finding] = field(default_factory=list)  # layout.decl_split debt: units physical keeps apart
    shaped: dict[str, list[tuple[str, list[str]]]] = field(default_factory=dict)  # record tag -> member edits
    # (sources, member renames, the type names those sources spell: a unit composed from their text spells one)
    accesses: list[tuple[tuple[str, ...], dict[str, str], frozenset[str]]] = field(default_factory=list)
    includes: dict[str, set[str]] = field(default_factory=dict)  # file -> headers a fold makes it include
    event_moves: list[int] = field(default_factory=list)  # the input move of each header event, before paths change


@dataclass
class _LayoutTypes:
    records: dict[str, str]
    typedefs: dict[str, str]


class _Types(_LayoutTypes):
    """The type names of a batch as its type moves leave them: per record tag and typedef name its statement, and per
    current name the name the snapshot spells it with."""

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot
        self.records, self.typedefs = headers.type_names(snapshot)
        self.was: dict[str, str] = {}
        self.renames: dict[str, str] = {}
        self.folded: set[str] = set()
        self.shaped: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
        self.accesses: list[tuple[tuple[str, ...], dict[str, str], frozenset[str]]] = []
        self.includes: dict[str, set[str]] = defaultdict(set)
        self.seen: dict[tuple[str, ...], Any] = {}  # what the snapshot answers (definers, reaches): fixed for the batch

    def ask(self, key: tuple[str, ...], produce: Callable[[], Any]) -> Any:
        """The snapshot's answer to key, produced once per batch: every fold asks it again for the same records."""
        if key not in self.seen:
            self.seen[key] = produce()
        return self.seen[key]

    def spelled(self, text: str) -> str:
        """The text with every name the batch renamed or folded spelled as it now is."""
        for old in sorted(symbols.tokens(text) & self.renames.keys()):
            text = symbols.rewrite(text, old, self.renames[old])
        return text

    def known(self, name: str) -> bool:
        if name in self.records or name in self.typedefs:
            return True
        if name in self.renames:
            return False  # renamed or folded away earlier in the batch
        records, typedefs = headers.source_types(self.snapshot, name)
        self.records.update({k: v for k, v in records.items() if k not in self.records})
        self.typedefs.update({k: v for k, v in typedefs.items() if k not in self.typedefs})
        return name in self.records or name in self.typedefs

    def tag(self, name: str) -> str | None:
        """The record tag a type name names: its own, or the one its typedef names."""
        if name in self.records and re.match(r"(struct|union)\b", self.records[name]):
            return name
        found = re.fullmatch(r"typedef (?:struct|union) (\w+) (\w+) ?;", self.typedefs.get(name, ""))
        return found[1] if found and found[2] == name and found[1] in self.records else None

    def _spell(self, old: str, new: str) -> None:
        """Every snapshot name spelled `old` now is spelled `new`."""
        origin = self.was.pop(old, old)
        self.renames.update({k: new for k, v in self.renames.items() if v == old})
        self.renames[origin] = new

    def rename(self, old: str, new: str) -> None:
        origin = self.was.get(old, old)
        self._spell(old, new)
        self.was[new] = origin
        for table in (self.records, self.typedefs):
            if old in table:
                table[new] = symbols.rewrite(table.pop(old), old, new)

    def fold(self, old: str, new: str) -> None:
        """`old` names `new` from now on; a record of its own loses its body."""
        mine, theirs = self.tag(old), self.tag(new)
        if mine and mine != theirs:
            self.folded.add(self.was.get(mine, mine))
            self.records.pop(mine, None)
            if mine != old and theirs:
                self._spell(mine, theirs)
        self.folded.add(self.was.get(old, old))  # its own definition and types.toml row go, even as a mere alias
        self._spell(old, new)
        self.records.pop(old, None)
        self.typedefs.pop(old, None)


def _layout_finding(old: str, new: str, mine: list[tuple[str, str]], theirs: list[tuple[str, str]]) -> Finding | None:
    """Why record `old` cannot fold into `new`: the first member whose type differs, or the member names that differ
    (accesses to them are not rewritten). None when both are the same record."""

    def said(row: tuple[str, str]) -> str:
        return f"{row[0] or 'nested'}: {row[1]}"

    for index, (a, b) in enumerate(zip(mine, theirs, strict=False)):
        if a[1] != b[1]:
            reason = f"{old} member {index} ({said(a)}) is not {new} member {index} ({said(b)})"
            return Finding("move.type_layout", reason, unit=old)
    if len(mine) != len(theirs):
        at = min(len(mine), len(theirs))
        longer, shorter, extra = (old, new, mine[at]) if len(mine) > len(theirs) else (new, old, theirs[at])
        return Finding("move.type_layout", f"{longer} member {at} ({said(extra)}) has no match in {shorter}", unit=old)
    pairs = [f"{a[0]}->{b[0]}" for a, b in zip(mine, theirs, strict=True) if a[0] != b[0]]
    if pairs:
        return Finding(
            "move.type_members",
            f"{old} and {new} lay out alike but name members otherwise: {', '.join(pairs)}",
            unit=old,
            action="rename the members of one of them first",
        )
    return None


def _json(text: str) -> Any:
    if not text.strip():
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = []
        for number, line in enumerate(text.splitlines(), 1):
            try:
                value += [json.loads(line)] if line.strip() else []
            except json.JSONDecodeError as error:
                raise Refusal(Finding("move.syntax", f"line {number} is not JSON: {error.msg}", line=number)) from error
    return value if isinstance(value, list) else [value]


def read(params: Json, stdin: bytes) -> tuple[Move, ...]:
    """The moves of the argument form, or with op `-` of a JSON array or JSON lines on stdin."""
    return _read_stamped(params, stdin)[0]


def _read_stamped(params: Json, stdin: bytes) -> tuple[tuple[Move, ...], list[str]]:
    """The moves and the layout digests the stream is stamped with (none for the argument form)."""
    stamped: list[str] = []
    if params["op"] == "relayout":  # the units become layout.physical's (CONTRACTS 2.2): one proved change, no move
        return (Move(0, "relayout", "", ""),), stamped
    if params["op"] != "-":
        if not (params.get("subject") and params.get("target")):
            raise Refusal(Finding("move.syntax", "a move needs OP SUBJECT TARGET", action="unbake move --help"))
        raw: Any = [{"op": params["op"], "subject": params["subject"], "target": params["target"]}]
    else:
        raw = _json(stdin.decode(errors="replace"))
        stamps = [r for r in raw if isinstance(r, dict) and set(r) == {"layout"}]
        raw = [r for r in raw if r not in stamps]
        stamped = [r["layout"] for r in stamps]
    try:
        configuration.validate("move", raw, "moves")
    except Refusal as refusal:
        first = refusal.findings[0]
        at = re.search(r"moves:(\d+)", first.path)
        raise Refusal(Finding("move.syntax", first.reason, line=int(at[1]) if at else 0)) from refusal
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw):
        if (item["op"], item["subject"]) in seen:
            raise Refusal(Finding("move.syntax", f"{item['subject']} is moved twice by {item['op']}", line=index))
        seen.add((item["op"], item["subject"]))
    return tuple(Move(i, m["op"], m["subject"], m["target"]) for i, m in enumerate(raw)), stamped


def edit(
    snapshot: Snapshot, moves: Sequence[Move], boundaries: Collection[str] = (), keep: Collection[str] = ()
) -> Edited:
    """The map after the moves in order, the text files that follow (symbols, split rows, config) and which move
    touched which member or module. Moves the map refuses are all collected in `bad`. `boundaries` are members a
    file starts at that this apply's proofs found, kept in the map beside its own; `keep` are old units whose proof
    failed this command: they stay as they are. A refused move changes nothing the moves after it see, so the moves
    the edit accepts are what it accepts without the refused ones (one pass judges a stream)."""
    with effort.stage("move.edit"):
        lay, project = snapshot.layout, snapshot.config.project
        rows = configuration.layout_rows(project)
        library = tuple(f.rstrip("/") + "/" for f in project.library_dirs)
        groups, table = dict(lay.groups), symbols.edit(snapshot)
        known = set(lay.members)
        renames: dict[str, str] = {}
        was: dict[str, str] = {}
        events: list[tuple[str, str, str, frozenset[str]]] = []
        event_moves: list[int] = []
        touched: defaultdict[int, set[str]] = defaultdict(set)
        bad: list[tuple[Move, Finding]] = []
        own = {m.subject for m in moves if m.op == "member"}
        types = _Types(snapshot) if any(m.op == "type" for m in moves) else None
        if types is not None:
            _warm_layouts(types, moves, table.keys() | known)

        def of(name: str) -> str:
            return next((g.name for g in groups.values() if name in g.members), "")

        def bind(group: Group) -> None:
            # sdk is what a load derives from library: the map the edit holds is the map its text reads back as
            groups[group.name] = replace(group, evidence="authored", sdk=bool(group.library))

        def reject(move: Move, key: str, reason: str, **extra: Any) -> None:
            bad.append((move, Finding(key, reason, **extra)))

        def locked(name: str) -> bool:
            unit = layout.unit_of(snapshot, was.get(name, name))
            return bool(unit and unit.path.startswith(library))

        def path_ok(move: Move, kind: str) -> bool:
            ok = bool(_PATH[kind].fullmatch(move.target)) and ".." not in move.target.split("/")
            if not ok:
                reject(move, "move.name", f"{move.target} is not a {kind} path", path=move.target)
            return ok

        for move in moves:
            op, subject, target, touch = move.op, move.subject, move.target, touched[move.index]
            if (op == "symbol" and subject not in known and subject not in table) or (
                op == "member" and subject not in known
            ):
                reject(move, "move.unknown", f"{subject} is not a symbol or member of the project", unit=subject)
            elif op in ("module", "source", "header", "options") and subject not in groups:
                reject(move, "move.unknown", f"{subject} is not a module", unit=subject)
            elif op == "file" and not any(g.header == subject for g in groups.values()) and not snapshot.peek(subject):
                reject(move, "move.unknown", f"{subject} is not a header", path=subject)
            elif op in ("symbol", "type", "member", "module") and not _NAME.fullmatch(target):
                reject(move, "move.name", f"{target} is not a C identifier", unit=target)
            elif op == "relayout":  # every module's units are laid again: each finding is this move's
                touch |= set(groups)
            elif op == "type" and types is not None:
                origin = types.was.get(subject, subject)  # as the snapshot spells it
                if found := _type_move(types, subject, target, table.keys() | known, groups):
                    bad.append((move, found))
                    continue
                touch |= {subject, target, *_type_users(snapshot, origin)}
            elif op == "symbol" and locked(subject):  # judged before anything is renamed: a refusal changes nothing
                reject(move, "move.library", f"{subject} belongs to a library unit", unit=subject)
            elif op == "symbol":
                for vid, record in snapshot.versions.items():  # a generated name (D_auto_...) is a fact of its version
                    if subject in record.symbols:
                        kind = getattr(lay.members.get(was.get(subject, subject)), "kind", "data")
                        table.setdefault(subject, {"kind": "function" if kind == "function" else "data"})[vid] = (
                            record.symbols[subject]
                        )
                try:
                    symbols.rename(table, subject, target)
                except Refusal as refusal:
                    found = refusal.findings[0]
                    reject(move, found.key.replace("symbols.", "move."), found.reason, unit=found.unit)
                    continue
                groups = {
                    k: replace(g, members=tuple(layout.renamed(n, {subject: target}) for n in g.members))
                    for k, g in groups.items()
                }
                known = {layout.renamed(n, {subject: target}) for n in known}
                origin = was.pop(subject, subject)
                renames[origin], was[target] = target, origin
                touch |= {target, of(target)}
            elif op == "member":
                src = of(subject)
                carried = [
                    n
                    for g in groups.values()
                    for n in g.members
                    if n == subject or (n.count("/") == 2 and n.split("/")[1] == subject and n not in own)
                ]
                if locked(subject):
                    reject(move, "move.library", f"{subject} belongs to a library unit", unit=subject)
                    continue
                if src == target:
                    continue
                first = lay.members[was.get(subject, subject)].placements[0]
                dest = groups.get(target) or Group(
                    target,
                    next(s[0] for s in snapshot.versions[first.version].segments if s[1] <= first.rom_start < s[2]),
                    (),
                    "authored",
                    (),
                    False,
                    f"src/{target}.c",
                    f"include/{target}.h",
                )
                if src:
                    old_header, left = groups[src].header, tuple(n for n in groups[src].members if n not in carried)
                    groups[src] = replace(groups[src], members=left)
                    if not left:
                        del groups[src]
                    if old_header != dest.header:
                        events.append(("names", old_header, dest.header, frozenset(carried)))
                        event_moves.append(move.index)
                bind(replace(dest, members=(*dest.members, *carried)))
                touch |= {*carried, src, target}
            elif op == "module":
                if subject == target:
                    continue
                group = groups.pop(subject)
                if target in groups:
                    into = groups[target]
                    if group.header != into.header:
                        events.append(("file", group.header, into.header, frozenset()))
                        event_moves.append(move.index)
                        for g in groups.values():
                            if g.header == group.header:
                                groups[g.name] = replace(g, header=into.header)
                    bind(replace(into, members=(*into.members, *group.members), sdk=into.sdk and group.sdk))
                else:
                    bind(replace(group, name=target))
                touch |= {subject, target}
            elif op == "options":  # one flag set per object: every unit of the module is laid and proved with it
                flags = _flag_set(target)
                if flags is None:
                    reject(move, "move.options", f'{target} is not a flag set {{"add": [...], "omit": [...]}}')
                elif any(locked(n) for n in groups[subject].members):
                    reject(move, "move.library", f"{subject} holds a library unit", unit=subject)
                else:
                    bind(replace(groups[subject], options=flags))
                    touch.add(subject)
            elif op == "source" and path_ok(move, "source"):
                clash = lay.units.get(target)
                if clash and clash.group != subject:
                    reject(move, "move.exists", f"{target} belongs to another module", path=target)
                    continue
                sub, lib = _placement(target, rows)  # where its source sits places the module
                bind(replace(groups[subject], source=target, subsystem=sub, library=lib))
                touch.add(subject)
            elif op in ("header", "file") and path_ok(move, "header"):
                named = groups.get(subject)
                old = named.header if named else subject
                # a shared header (repo.toml [layout]) is the base the other headers build on: a module bound to it
                # takes only its own names away, and the header itself moves into no other
                base = old in rows["shared_headers"]
                if base and not (named and named.members):
                    reject(move, "move.shared", f"{old} is a shared header: it moves into no other header", path=old)
                    continue
                shared = [g for g in groups.values() if g.header == old and g.name != subject]
                if old != target:
                    names = frozenset(named.members) if named and (shared or base) else frozenset()
                    events.append(("names" if names else "file", old, target, names))
                    event_moves.append(move.index)
                affected = [named] if named and (shared or base) else [g for g in groups.values() if g.header == old]
                if bad_header := [g for g in affected if not _header_ok(g, target, groups, rows)]:
                    reject(move, "move.name", f"{target} is not the header of {bad_header[0].name}", path=target)
                    continue
                for g in affected:
                    bind(replace(g, header=target))
                    touch.add(g.name)
        if bad:
            return Edited(lay, {}, {}, [], bad=bad)

        def rank(name: str) -> tuple[int, str]:
            item = lay.members.get(layout.renamed(name, was))
            return (item.placements[0].rom_start if item else 1 << 40), name

        for name, group in list(groups.items()):  # moved members keep ROM order within their module
            groups[name] = replace(group, members=tuple(sorted(group.members, key=rank)))
        writes = layout.rename_writes(snapshot, table, renames)
        fuzzy = {}
        for old, row in lay.fuzzy.items():
            new = layout.renamed(old, renames)
            fuzzy[new] = {**row, "path": row["path"].replace(f"/{old}.c", f"/{new}.c")}
        units = {
            p: replace(u, members=tuple(layout.renamed(n, renames) for n in u.members)) for p, u in lay.units.items()
        }
        bounds = tuple(sorted({*(layout.renamed(n, renames) for n in lay.boundaries), *boundaries}))
        shifted = replace(lay, groups=groups, units=units, fuzzy=fuzzy, digest="", boundaries=bounds)
        mid = layout.overlay(snapshot, {**writes, "layout.toml": layout.dump_map(shifted)})
        split: list[Finding] = []
        # a move re-lays only the modules it changes (a source move renames its module's files); every other module
        # keeps its units as they are, whatever their file names
        changed = {n for names in touched.values() for n in names if n in groups}
        fresh = set(boundaries) - set(lay.boundaries)  # a boundary this command's proofs found cuts its module again
        changed |= {u.group for u in lay.units.values() if fresh.intersection(u.members)}
        changed |= {g.name for g in groups.values() if fresh.intersection(g.members)}
        reflow = None if any(m.op == "relayout" for m in moves) else changed
        final = replace(mid.layout, units=layout.physical(mid, mid.layout, split, keep, reflow=reflow))
        if (final.groups, final.units, final.fuzzy, final.boundaries) != (
            lay.groups,
            lay.units,
            lay.fuzzy,
            lay.boundaries,
        ):
            writes["layout.toml"] = layout.dump_map(final)
        else:
            writes.pop("layout.toml", None)
        if types is not None:
            types.renames = {k: v for k, v in types.renames.items() if k != v}
        return Edited(
            final,
            writes,
            dict(touched),
            events,
            renames,
            types=types.renames if types else {},
            folded=types.folded if types else set(),
            split=split,
            shaped=dict(types.shaped) if types else {},
            accesses=types.accesses if types else [],
            includes=dict(types.includes) if types else {},
            event_moves=event_moves,
        )


def _type_move(
    types: _Types, subject: str, target: str, taken: Collection[str], groups: Mapping[str, Group]
) -> Finding | None:
    """Applies one type move to the batch's type names: a rename when no type is named `target`, else a fold of
    `subject` into it. The finding that refuses it, if any."""
    if not types.known(subject):
        return Finding(
            "move.unknown", f"{subject} is not a struct, union, enum or typedef the project declares", unit=subject
        )
    if subject in taken:
        return Finding(
            "move.exists", f"{subject} also names a symbol or member: a type move would rename it too", unit=subject
        )
    if subject == target:
        return None
    if not types.known(target):
        stems = {p.rpartition("/")[2].rpartition(".")[0] for g in groups.values() for p in (g.source, g.header)}
        if target in taken or target in stems or target in groups:
            return Finding("move.exists", f"{target} already names a symbol, member, module or file", unit=target)
        types.rename(subject, target)
        return None
    mine, theirs = types.tag(subject), types.tag(target)
    if theirs is None:
        return Finding("move.type_layout", f"{target} is not a complete struct or union to fold into", unit=target)
    if mine is None:
        return Finding("move.type_layout", f"{subject} is not a struct or union", unit=subject)
    if mine != theirs:
        kinds = (types.records[mine].split()[0], types.records[theirs].split()[0])
        if kinds[0] != kinds[1]:
            return Finding("move.type_layout", f"{subject} is a {kinds[0]} and {target} a {kinds[1]}", unit=subject)
        spelled = symbols.rewrite(symbols.rewrite(types.records[mine], subject, target), mine, theirs)
        found = _layout_finding(
            subject, target, headers.members(spelled) or [], headers.members(types.records[theirs]) or []
        )
        if found and (found := _view_fold(types, subject, target, mine, theirs)):
            return found
        _fold_includes(types, mine, theirs)
    types.fold(subject, target)
    return None


_PAD = re.compile(r"(?:pad|_pad|padding|unk_pad|reserved)", re.I)  # a member that only holds space: split it
_PLACEHOLDER = re.compile(r"(?:unk|field)_?(?:0x)?[0-9A-Fa-f]+|(?:pad|_pad|padding|unk_pad|reserved)\w*", re.I)
_PROBE = "__unbake_view"
_ACCESS = r"(->|\.)(\s*)"


def _body(statement: str, at: int = 0) -> tuple[int, int]:
    """(open, close) brace positions of the first record body at or after `at`."""
    start = statement.index("{", at)
    depth = 0
    for i in range(start, len(statement)):
        depth += (statement[i] == "{") - (statement[i] == "}")
        if not depth:
            return start, i
    return start, len(statement)


def _reshaped(text: str, tag: str, edits: Sequence[tuple[str, list[str]]]) -> str | None:
    """The text with record `tag`'s body edited: each named member declaration replaced by its declarations, a ""
    name appending them before the close. None when the record or a single-declarator member is not found."""
    found = re.search(rf"\b(?:struct|union)\s+{re.escape(tag)}\s*\{{", text)
    if not found:
        return None
    start, end = _body(text, found.end() - 1)
    body = text[start + 1 : end]
    for name, parts in edits:
        if not name:
            body = body.rstrip() + "\n" + "".join(f"    {p}\n" for p in parts)
            continue
        hit = re.search(
            rf"(?<=[;{{\n])([ \t]*)[A-Za-z_][\w \t*]*?\b{re.escape(name)}\s*(?:\[[^\]]*\]\s*)*;", "\n" + body
        )
        if not hit:
            return None
        indent = hit[1] or " "
        body = ("\n" + body)[: hit.start()] + indent + f"\n{indent}".join(parts) + ("\n" + body)[hit.end() :]
        body = body[1:]
    return text[: start + 1] + body + text[end:]


def _probe_layouts(types: _LayoutTypes, rows: dict[str, str]) -> dict[str, Any] | str:
    """The offset layouts of `rows` (key `struct TAG` -> its definition) parsed alone with the typedefs they spell
    and the record bodies they hold by value, as the batch's type names give them; else why they cannot be laid."""
    key, read, _ = _probe_reader(types, rows)
    laid: dict[str, Any] | str = effort.memo(key, read)
    return laid


def _probe_reader(
    types: _LayoutTypes, rows: dict[str, str]
) -> tuple[tuple[str, str], Callable[[], dict[str, Any] | str], _LayoutTypes]:
    """A text-dependent probe key and its lazy reader; preparation never parses or lays a record."""
    lines: list[str] = []
    done: set[str] = set()

    def need(text: str) -> None:
        for word in dict.fromkeys(re.findall(r"[A-Za-z_]\w*", text)):
            statement = types.typedefs.get(word)
            if word in done or statement is None:
                continue
            done.add(word)
            tagged = re.match(r"typedef\s+(struct|union)\s+(\w+)\s*\{", statement)
            statement = f"typedef {tagged[1]} {tagged[2]} {word};" if tagged else statement
            need(statement)
            lines.append(statement)

    rows = dict(rows)
    for row in list(rows.values()):
        need(row)
    # the result is a function of the text laid: every round of a partial move edits the same moves again, and many
    # folds lay one real record, so one text is parsed once per command (with the bodies a missing record takes)
    bodies: dict[str, str] = {}
    aliases: dict[str, str] = {}
    pending = re.findall(r"[A-Za-z_]\w*", "\n".join([*lines, *rows.values()]))
    while pending:  # include typedefs reached through a body that a later parse may need
        word = pending.pop()
        for source, found in ((types.records, bodies), (types.typedefs, aliases)):
            if word not in found and word in source:
                found[word] = source[word]
                pending += re.findall(r"[A-Za-z_]\w*", found[word])
    key = ("probe-layouts", digest((lines, rows, sorted(bodies.items()), sorted(aliases.items()))))
    return key, lambda: _laid(types, rows, lines, need), _LayoutTypes(bodies, aliases)


def _parsed_layouts(text: str, names: tuple[str, ...]) -> dict[str, Any] | str:
    """Parse and canonicalize one actual translation unit once, even across different dependency probes."""
    from pycparser import CParser

    def read() -> dict[str, Any] | str:
        try:
            ast = CParser().parse(text)
        except Exception as error:
            return f"its records do not parse alone ({error})"
        return typedb.record_layouts(ast, names)

    return effort.memo(("move.parsed_layouts", digest((text, names))), read)


def _laid(
    types: _LayoutTypes, rows: dict[str, str], lines: list[str], need: Callable[[str], None]
) -> dict[str, Any] | str:
    for _ in range(32):  # a record held by value is laid from its body: each one missing is added and parsed again
        laid = _parsed_layouts(headers.normal("\n".join(lines + list(rows.values()))), tuple(rows))
        if isinstance(laid, str):
            return laid
        causes = [laid.get(k, {}).get("conflict", "no layout") for k in rows if "fields" not in laid.get(k, {})]
        missing = [
            (key, tag)
            for key, tag in re.findall(r"incomplete record: ((?:struct|union) (\w+))", " ".join(causes))
            if key not in rows and tag in types.records and "{" in types.records[tag]
        ]
        if not causes:
            return laid
        if not missing:
            return str(causes[0])
        for key, tag in missing:
            start, end = _body(types.records[tag])
            rows[key] = f"{key} {types.records[tag][start : end + 1]};"
            need(rows[key])
    return "its records nest too deep to lay alone"


def _anonymous(field: Mapping[str, Any]) -> bool:
    """Whether a member is an unnamed record type laid in place (a union of views, or one view of it)."""
    return bool(re.match(r"(?:struct|union) \{", field["type"])) and bool(field.get("fields"))


def _same_kind(ours: str, theirs: str) -> bool:
    """Whether two member types are one type for a fold (a `void *` takes any pointer)."""
    return ours == theirs or ("*" in ours and "*" in theirs and "void *" in (ours, theirs))


def _leaves(fields: Sequence[Mapping[str, Any]], base: int = 0, path: str = "") -> list[tuple[str, int, int, str]]:
    """(dotted path, offset, width, type) of every member an access can name inside unnamed record members."""
    out: list[tuple[str, int, int, str]] = []
    for f in fields:
        at, here = base + f["offset"], f"{path}{f['name']}"
        out.append((here, at, f["width"], f["type"]))
        if _anonymous(f):
            out += _leaves(f["fields"], at, here + ".")
    return out


def _view_path(fields: Sequence[Mapping[str, Any]], name: str, at: int, width: int, kind: str) -> str | None:
    """The dotted path of the member of a view (an arm of an unnamed union member, at any depth) of `fields` that
    sits at `at` with `width` and `kind`: the one named `name` first, else the only one; None when there is none."""
    hits = [
        p
        for f in fields
        if _anonymous(f) and f["offset"] <= at and at + width <= f["offset"] + f["width"]
        for p, o, w, k in _leaves(f["fields"], f["offset"], f["name"] + ".")
        if (o, w) == (at, width) and _same_kind(kind, k) and not _PAD.match(p.rsplit(".", 1)[1])
    ]
    named = [p for p in hits if p.rsplit(".", 1)[1] == name]
    return (named or hits)[0] if len(named) == 1 or len(hits) == 1 else None


def _element(field: Mapping[str, Any]) -> tuple[str, int] | None:
    """(element type, element width) of a one-dimensional array member; None for any other member."""
    found = re.fullmatch(r"([^\[\]]*?) ?\[(\d+)\]", field["type"])
    return (found[1], field["width"] // int(found[2])) if found and int(found[2]) else None


def _indexes(texts: Sequence[str], name: str) -> tuple[dict[int, str], bool]:
    """How the texts index member `name`: per constant index one spelling of it, and whether any access is not one
    (a computed index, or the array itself)."""
    constant: dict[int, str] = {}
    other = False
    pattern = re.compile(_ACCESS + rf"{re.escape(name)}\b(\s*\[\s*([^\[\]]*?)\s*\])?")
    for text in texts:
        for found in pattern.finditer(text):
            index = found[4]
            if index is not None and re.fullmatch(r"0[xX][0-9A-Fa-f]+|\d+", index):
                constant.setdefault(int(index, 0), index)
            else:
                other = True
    return constant, other


def _access(key: str) -> str:
    """The pattern of one member access a fold rewrites, after `->` or `.`: a member `name`, one element
    `name[index]`, or the opening `name[` of any index."""
    if key.endswith("["):
        return rf"{re.escape(key[:-1])}\s*\["
    if "[" in key:
        name, index = key[:-1].split("[", 1)
        return rf"{re.escape(name)}\s*\[\s*{re.escape(index)}\s*\]"
    return rf"{re.escape(key)}\b"


def _renamed(text: str, renames: Mapping[str, str]) -> str:
    """The text with each access of `renames` rewritten in one pass (a member renamed onto another's old name is not
    renamed twice)."""
    if not renames:
        return text
    keys = sorted(renames, key=len, reverse=True)
    pattern = re.compile(_ACCESS + "(?:" + "|".join(f"(?P<k{i}>{_access(k)})" for i, k in enumerate(keys)) + ")")

    def swap(found: re.Match[str]) -> str:
        key = next(keys[i] for i in range(len(keys)) if found[f"k{i}"] is not None)
        return found[1] + found[2] + renames[key]

    return pattern.sub(swap, text)


def _view_rows(types: _Types, subject: str, target: str, mine: str, theirs: str) -> dict[str, str] | None:
    """The two record texts an offset fold lays, before any stateful access or member edit."""
    record = types.records[theirs]
    kind = re.search(rf"\b(struct|union)\s+{re.escape(theirs)}\s*\{{", record)
    if kind is None:
        return None
    spelled = types.spelled(symbols.rewrite(symbols.rewrite(types.records[mine], subject, target), mine, theirs))
    probe, real = f"{kind[1]} {_PROBE}", f"{kind[1]} {theirs}"
    (a, b), (c, d) = _body(spelled), _body(record)
    return {probe: f"{probe} {spelled[a : b + 1]};", real: f"{real} {types.spelled(record[c : d + 1])};"}


def _probe_job(item: tuple[_LayoutTypes, dict[str, str]]) -> dict[str, Any] | str:
    return _probe_layouts(*item)


def _warm_layouts(types: _Types, moves: Sequence[Move], taken: Collection[str]) -> None:
    """Lay independent initial record texts on the pool; moves still apply in order and look up their actual text.
    A preceding rename or reshape changes that lookup, so a speculative layout can never replace a later one.
    The private type catalogue keeps speculative source discovery out of the edit's mutable state."""
    candidates = [m for m in moves if m.op == "type" and m.subject not in taken and _NAME.fullmatch(m.target)]
    if len(candidates) < 2:
        return
    with effort.stage("move.type_layouts"):
        initial = _Types(types.snapshot)
        probes: list[dict[str, str]] = []
        for move in candidates:
            if not initial.known(move.subject) or not initial.known(move.target):
                continue
            mine, theirs = initial.tag(move.subject), initial.tag(move.target)
            if not mine or not theirs or mine == theirs or "{" not in initial.records[mine]:
                continue
            if initial.records[mine].split()[0] != initial.records[theirs].split()[0]:
                continue
            spelled = symbols.rewrite(symbols.rewrite(initial.records[mine], move.subject, move.target), mine, theirs)
            if not _layout_finding(
                move.subject,
                move.target,
                headers.members(spelled) or [],
                headers.members(initial.records[theirs]) or [],
            ):
                continue
            rows = _view_rows(initial, move.subject, move.target, mine, theirs)
            if rows is not None:
                probes.append(rows)
        jobs: dict[tuple[str, str], tuple[_LayoutTypes, dict[str, str]]] = {}
        for rows in probes:  # source discovery is complete before any key is computed
            key, _, dependencies = _probe_reader(initial, rows)
            if effort.peek(key) is None:
                jobs[key] = (dependencies, rows)
        if len(jobs) < 2:  # one unique probe is already computed once by the serial edit
            return
        for key, value in zip(
            jobs, pool.map(types.snapshot.config, "move.type_layouts", _probe_job, list(jobs.values())), strict=True
        ):
            effort.memo(key, partial(_given, value))


def _given(value: Any) -> Any:
    return value


def _view_fold(types: _Types, subject: str, target: str, mine: str, theirs: str) -> Finding | None:
    """Folds record `mine` into `theirs` as a partial view of it, by member offset. Each member the users of `mine`
    access is the member of `theirs` at its offset and width (the accesses renamed; a placeholder of `theirs` takes
    the real name instead, never the reverse), or an element of an array of `theirs` of its type (the access indexes
    it). A member `theirs` lacks is inserted at its offset where `theirs` holds padding or unknown bytes (a padding or
    placeholder byte array no user of `theirs` reads), or past its end. An array the users index only by constants
    folds element by element. A member at the offset of an unnamed union of views of `theirs` is the member of the
    view (at any depth) of its offset, width and type (the access takes its dotted path), and a union of views of
    `mine` folds onto the one of `theirs` at its offset when every member its users read sits at the same path. A
    `void *` on either side takes the other's pointee. A member whose type or pointee
    differs (a cast would be needed) or that overlaps named members refuses the fold, naming both."""

    def refuse(reason: str) -> Finding:
        return Finding("move.type_layout", f"{subject} cannot fold into {target} by offset: {reason}", unit=subject)

    rows = _view_rows(types, subject, target, mine, theirs)
    if rows is None:
        return refuse(f"{target} has no body")
    probe, real = rows
    record = types.records[theirs]
    laid = _probe_layouts(types, rows)
    if isinstance(laid, str):
        return refuse(laid)
    snapshot = types.snapshot

    def spell(*names: str) -> list[str]:  # the sources that spell any of the names, as the snapshot spells them
        return sorted({p for n in names for p in headers.spelled_in(snapshot, types.was.get(n, n))[0]})

    users = spell(subject, mine)
    texts = [snapshot.read(p).decode("latin-1") for p in users]
    olds = {f["name"]: f for f in laid[probe]["fields"]}
    used = {m for t in texts for m in re.findall(r"(?:->|\.)\s*([A-Za-z_]\w*)", t)}
    fields, size = laid[real]["fields"], laid[real]["size"]
    names = {f["name"] for f in fields}
    others = sorted(set(spell(target, theirs)) - set(users))
    read: set[str] | None = None  # the members the users of `theirs` access, read once when needed

    def unknown(g: Mapping[str, Any]) -> bool:
        """Whether a member of `theirs` only holds space: padding, or a placeholder byte array nobody reads."""
        nonlocal read
        if _PAD.match(g["name"]):
            return True
        element = _element(g)
        if not (_PLACEHOLDER.fullmatch(g["name"]) and element and element[1] == 1):
            return False
        if read is None:
            read = {
                m for p in others for m in re.findall(r"(?:->|\.)\s*([A-Za-z_]\w*)", snapshot.read(p).decode("latin-1"))
            }
        return g["name"] not in read

    renames: dict[str, str] = {}
    placeholders: dict[str, str] = {}
    edits: list[tuple[str, list[str]]] = []
    inserts: dict[str, list[tuple[int, int, str]]] = defaultdict(list)

    def overlapped(at: int, width: int) -> str:
        hit = [g for g in fields if g["offset"] < at + width and at < g["offset"] + g["width"] and not unknown(g)]
        return ", ".join(f"{g['name']} ({g['type']})" for g in hit)

    def place(name: str, key: str, at: int, width: int, kind: str, declaration: str, whole: bool) -> str | None:
        """Places one member (or one element of it, not `whole`) the users read at `at`, or says why it cannot be
        placed (and changes nothing)."""
        same = [
            g
            for g in fields
            if (g["offset"], g["width"]) == (at, width) and not _PAD.match(g["name"]) and not _anonymous(g)
        ]
        if same:
            g = next((g for g in same if g["type"] == kind), same[0])
            if g["type"] != kind:
                pointers = "*" in g["type"] and "*" in kind
                if not (pointers and whole and "void *" in (g["type"], kind)):
                    what = "pointee" if pointers else "type"
                    return f"its {key} ({kind}) and {g['name']} ({g['type']}) differ in {what}"
                if g["type"] == "void *":  # the unknown pointee of `theirs` takes the one its view reads
                    taken = _PLACEHOLDER.fullmatch(g["name"]) and not _PLACEHOLDER.fullmatch(name) and name not in names
                    final = name if taken else g["name"]
                    edits.append((g["name"], [symbols.rewrite(declaration, name, final)]))
                    if final != g["name"]:
                        placeholders[g["name"]] = final
                        names.add(final)
                    if key != final:
                        renames[key] = final
                    return None
            if g["name"] == key:
                return None
            # the placeholder takes the real name, unless another member of `theirs` holds it: then the access takes
            # the placeholder's
            if whole and _PLACEHOLDER.fullmatch(g["name"]) and not _PLACEHOLDER.fullmatch(name) and name not in names:
                placeholders[g["name"]] = name  # never a real name traded for a placeholder: it takes the real one
                names.add(name)
                edits.append((g["name"], [symbols.rewrite(g["declaration"], g["name"], name)]))
            else:
                renames[key] = g["name"]
            return None
        for g in fields:  # an element of an array of `theirs` holding its type
            element = _element(g)
            if element and element[0] == kind and g["offset"] <= at and at + width <= g["offset"] + g["width"]:
                step, rest = divmod(at - g["offset"], element[1])
                if not rest and width == element[1]:
                    renames[key] = f"{g['name']}[{step}]"
                    return None
        if (path := _view_path(fields, name, at, width, kind)) is not None:  # a member of one of its views
            if key != path:
                renames[key] = path
            return None
        if whole and name in names:
            return f"its {name} at 0x{at:X} is not {target}'s {name}"
        holds = [g for g in fields if g["offset"] <= at and at + width <= g["offset"] + g["width"] and unknown(g)]
        if not holds and at < size:
            hit = overlapped(at, width) or "other members"
            return f"its {key} at 0x{at:X} (width {width}) overlaps {hit} of {target}"
        final = name if whole else f"unk{at:X}"
        if not whole and final in names:
            return f"its {key} at 0x{at:X} would be {final}, which names another member of {target}"
        if not whole:  # one element of the array, declared as the array's declaration spells its element type
            declaration = re.sub(rf"\b{re.escape(name)}\s*\[[^\]]*\]", final, declaration, count=1)
        inserts[holds[0]["name"] if holds else ""].append((at, width, declaration))
        names.add(final)
        if key != final:
            renames[key] = final
        return None

    def shifted(name: str, f: Mapping[str, Any]) -> str | None:
        """An array whose elements are elements of an array of `theirs` of the same type: its indexes shift."""
        ours = _element(f)
        for g in fields:
            if not (ours and ours == _element(g) and g["offset"] <= f["offset"]):
                continue
            step, rest = divmod(f["offset"] - g["offset"], ours[1])
            if rest or f["offset"] + f["width"] > g["offset"] + g["width"]:
                continue
            if not step:
                renames[name] = g["name"]
                return None
            bare = re.compile(_ACCESS + rf"{re.escape(name)}\b(?!\s*\[)")
            if any(bare.search(t) for t in texts):
                return f"its {name} is read whole and sits at element {step} of {g['name']}"
            renames[name + "["] = f"{g['name']}[{step} + "
            return None
        return "no array"

    def viewed(f: Mapping[str, Any]) -> str | None:
        """Folds an unnamed record member of `mine` (a union of views) onto the one of `theirs` at its offset and
        width, of the same kind and holding every member the users read at the same path; or says why it cannot."""
        g = next((g for g in fields if (g["offset"], g["width"]) == (f["offset"], f["width"]) and _anonymous(g)), None)
        if g is None or g["type"].split()[0] != f["type"].split()[0]:
            return None
        theirs_at = {p.split(".", 1)[1]: (o, w, k) for p, o, w, k in _leaves(g["fields"], g["offset"], "_.")}
        for p, o, w, k in _leaves(f["fields"], f["offset"], "_."):
            p = p.split(".", 1)[1]
            hit = theirs_at.get(p)
            if p.rsplit(".", 1)[-1] not in used or (hit and hit[:2] == (o, w) and _same_kind(k, hit[2])):
                continue
            if hit is None:
                return f"its {f['name']}.{p} at 0x{o:X} is no member of {target}'s {g['name']}"
            return f"its {f['name']}.{p} ({k}) and {g['name']}.{p} ({hit[2]}) differ"
        if g["name"] != f["name"]:
            renames[f["name"]] = g["name"]
        return ""

    for name in sorted(used & olds.keys(), key=lambda n: (olds[n]["offset"], n)):
        f = olds[name]
        at, width = f["offset"], f["width"]
        if _anonymous(f) and (why := viewed(f)) is not None:
            if why:
                return refuse(why)
            continue
        why = place(name, name, at, width, f["type"], f["declaration"], True)
        if why is None:
            continue
        element = _element(f)
        if element is None:
            return refuse(why)
        if shifted(name, f) is None:
            continue
        constant, other = _indexes(texts, name)
        if other or not constant:
            return refuse(why)
        for index, spelling in sorted(constant.items()):  # read only element by element: each element on its own
            if index * element[1] >= width:
                return refuse(f"its {name}[{spelling}] is past the end of {name}")
            key, offset = f"{name}[{spelling}]", at + index * element[1]
            if why := place(name, key, offset, element[1], element[0], f["declaration"], False):
                return refuse(why)
    for pad, items in inserts.items():
        spans = next(((g["offset"], g["offset"] + g["width"]) for g in fields if g["name"] == pad), (size, 0))
        pos, parts = spans[0], []
        for at, width, declaration in sorted(items):
            if at < pos:
                return refuse(f"two of its members overlap at 0x{at:X}")
            filler = f"pad{pos:X}"
            if at > pos and filler in names - {pad}:
                return refuse(f"the filler {filler} names a member of {target}")
            parts += [f"char {filler}[0x{at - pos:X}];"] if at > pos else []
            parts.append(declaration)
            pos = at + width
        parts += [f"char pad{pos:X}[0x{spans[1] - pos:X}];"] if pos < spans[1] else []
        edits.append((pad, parts))
    shaped = _reshaped(record, theirs, edits)
    if shaped is None:
        return refuse(f"a member {target} declares is not one plain declaration")
    if edits:
        types.records[theirs] = shaped
        types.shaped[types.was.get(theirs, theirs)] += edits
    if renames:
        types.accesses.append((tuple(users), renames, frozenset(types.was.get(n, n) for n in (subject, mine))))
    if placeholders:
        types.accesses.append((tuple(others), placeholders, frozenset(types.was.get(n, n) for n in (target, theirs))))
    return None


def _fold_includes(types: _Types, mine: str, theirs: str) -> None:
    """Where the body of record `mine` goes, the header defining `theirs` comes in: in the file that defined it, or,
    when that header already reaches the file (an include there would loop), in its users."""
    snapshot = types.snapshot
    old, new = types.was.get(mine, mine), types.was.get(theirs, theirs)

    def definers(tag: str) -> list[str]:
        sources, heads = headers.spelled_in(snapshot, tag)
        body = re.compile(rf"\b(?:struct|union)\s+{re.escape(tag)}\s*\{{")
        return [p for p in (*heads, *sources) if body.search(snapshot.read(p).decode("latin-1"))]

    def reaches(start: str, goal: str) -> bool:
        return bool(types.ask(("reaches", start, goal), lambda: headers.reaches(snapshot, start, goal)))

    home = next((p for p in types.ask(("definers", new), lambda: definers(new)) if configuration.is_header(p)), None)
    for path in types.ask(("definers", old), lambda: definers(old)) if home else ():
        assert home is not None
        if path == home or reaches(path, home):
            continue
        for user in headers.spelled_in(snapshot, old)[0] if reaches(home, path) else [path]:
            types.includes[user].add(home.removeprefix("include/"))


def _shape(snapshot: Snapshot, files: Mapping[str, str | None], edited: Edited) -> dict[str, str]:
    """The texts the partial-view folds change before the names are spelled again: each record they widen or whose
    placeholders they name, each access to a renamed member, each include a dropped body needs."""
    out: dict[str, str] = {}

    def text(path: str) -> str | None:
        if path in out or path in files:
            return out.get(path, files.get(path))
        found = snapshot.peek(path)
        return None if found is None else found.decode(errors="replace")

    for tag, edits in edited.shaped.items():
        for path in (p for paths in headers.spelled_in(snapshot, tag) for p in paths):
            if (now := text(path)) is not None and (shaped := _reshaped(now, tag, edits)) is not None:
                out[path] = shaped
    for paths, renames, spelled in edited.accesses:
        # a relayout may compose a source's text into a new unit: every composed unit spelling the type is one too
        composed = (
            p for p, t in files.items() if configuration.is_source(p) and t and spelled & set(re.findall(r"\w+", t))
        )
        for path in dict.fromkeys((*paths, *composed)):
            if (now := text(path)) is not None:
                out[path] = _renamed(now, renames)
    for path, wanted in edited.includes.items():
        if (now := text(path)) is not None:
            out[path] = headers.included(now, wanted)
    return out


def _type_users(snapshot: Snapshot, name: str) -> set[str]:
    """The modules and members of the units whose own text spells a type name. A unit that only includes a header
    spelling it is not blamed: a shared header would make every unit's debt the move's."""
    with effort.stage("move.type_users"):
        spelled, _ = headers.spelled_in(snapshot, name)
        units = [snapshot.layout.units[p] for p in spelled if p in snapshot.layout.units]
        return {n for u in units for n in (u.group, *u.members) if n}


def _moves_for(edited: Edited, moves: Sequence[Move], unit: UnitSpec | None, every: bool = True) -> list[Move]:
    """The moves that touched the unit's module or members; every move when none did and `every` holds."""
    names = {unit.group, *unit.members, *(layout.renamed(n, edited.renames) for n in unit.members)} if unit else set()
    hit = [m for m in moves if edited.touched.get(m.index, set()) & names]
    return hit or (list(moves) if every else [])


def _cycle_moves(edited: Edited, moves: Sequence[Move], path: str) -> list[Move]:
    """Trace the refused header through its actual events before consulting final names. A renamed old path may
    name no final module; that must not turn its one causal move into blame for the whole stream."""
    live, causes = {path}, set()
    for (kind, old, new, _), index in zip(edited.events, edited.event_moves, strict=True):
        if old not in live:
            continue
        causes.add(index)
        if kind == "file":  # follow this header's body, not an unrelated file that later reuses its old name
            live.remove(old)
        live.add(new)  # a names part also travels through later moves of its new header
    if causes:
        return [m for m in moves if m.index in causes]
    named = [m for m in moves if path and path in (m.subject, m.target)]
    return named or (_header_moves(edited, moves, path, path) if path else []) or list(moves)


def _header_moves(edited: Edited, moves: Sequence[Move], old: str, new: str) -> list[Move]:
    """The moves that made one header event: those naming either header, else those that touched a module whose header
    either is."""
    named = [m for m in moves if {m.subject, m.target} & {old, new}]
    if named:
        return named
    groups = {g.name for g in edited.layout.groups.values() if g.header in (old, new)}
    return [m for m in moves if edited.touched.get(m.index, set()) & groups]


def _file_moves(before: Snapshot, after: Snapshot, edited: Edited, moves: Sequence[Move], path: str) -> list[Move]:
    """The moves that changed one file: those whose subject or target the lines it changed spell (a type or symbol
    move rewrites the lines naming it), else those naming it or a module bound to it, and only when none can be told
    every move."""
    old = set((before.peek(path) or b"").decode(errors="replace").splitlines())
    new = set((after.peek(path) or b"").decode(errors="replace").splitlines())
    words = symbols.tokens("\n".join(old ^ new))
    spelled = [m for m in moves if {m.subject, m.target} & words]
    return spelled or _header_moves(edited, moves, path, path) or list(moves)


_HEADER_AT = re.compile(r"([\w./+-]+\.(?i:h)):\d+")
_INCLUDED_AT = re.compile(r"(?:In file included|^\s+) from ([\w./+-]+\.(?i:h)):(\d+)", re.M)
_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')


def _edge_moves(
    before: Snapshot, after: Snapshot, moves: Sequence[Move], said: str, root: str, changed: Collection[str]
) -> list[Move]:
    """The moves that added an include edge the compiler's include chain runs through: a changed header whose chain
    line is an #include it did not have before now pulls in a header because a move spelled one of that header's
    names into it. Those moves (their names spelled both in the includer's new lines and in the included header) made
    the cycle or the order that breaks the unit, never every move touching a header on the chain (one kept unit's
    chain refused 741 of 751 rows on 2026-10-15)."""
    hit: dict[int, Move] = {}
    for at, line in sorted({(_VIEWS.sub("", p.removeprefix(root)), int(n)) for p, n in _INCLUDED_AT.findall(said)}):
        if at not in changed:
            continue
        lines = (after.peek(at) or b"").decode(errors="replace").splitlines()
        old = set((before.peek(at) or b"").decode(errors="replace").splitlines())
        added = _INCLUDE.match(lines[line - 1]) if 0 < line <= len(lines) and lines[line - 1] not in old else None
        if added is None:
            continue
        found = (posixpath.normpath(f"{b}/{added[1]}") for b in (posixpath.dirname(at), "include"))
        target = next((t for t in found if after.peek(t) is not None), None)
        if target is None:
            continue
        both = symbols.tokens("\n".join(set(lines) - old)) & symbols.tokens(
            (after.peek(target) or b"").decode(errors="replace")
        )
        hit.update((m.index, m) for m in moves if {m.subject, m.target} & both)
    return [hit[i] for i in sorted(hit)]


_VIEWS = re.compile(r"^(?:.*/)?build/views/[0-9a-f]+/")
_SAID = 1500  # chars of a compiler message a finding keeps: the head names the error and the include chain


def _probed(snapshot: Snapshot, paths: Sequence[str]) -> dict[str, list[str]]:
    """Per header its first error lines compiled alone (unbake headers --probe: an include of it and nothing else,
    with the flags of the project's first C unit outside the libraries), in one dispatch; [] when it compiles."""
    library = tuple(f.rstrip("/") + "/" for f in snapshot.config.project.library_dirs)
    units = [
        u
        for _, u in sorted(snapshot.layout.units.items())
        if {"preprocess", "compile"} <= set(configuration.kind(u.kind)["phases"]) and not u.path.startswith(library)
    ]
    if not units or not paths:
        return {}
    probes = {h: f"src/.probe/{digest(h)[:16]}.c" for h in paths}
    texts = {p: f'#include "{h.removeprefix("include/")}"\n'.encode() for h, p in probes.items()}
    overlay = layout.overlay(snapshot, texts)
    recipe = recipes.resolve(snapshot.config, units[0], units[0].options)
    version = next(iter(snapshot.config.project.versions))
    items = [(overlay, replace(units[0], path=probes[h]), version, h, recipe) for h in paths]
    rows = pool.map(snapshot.config, "move.headers", headers._probe_job, items)
    return {
        h: (list(r["first"]) or ["the header does not compile alone"]) if r["errors"] else []
        for h, r in zip(paths, rows, strict=True)
    }


def _broken_headers(before: Snapshot, after: Snapshot, changed: Sequence[str]) -> dict[str, Finding]:
    """Per changed header that no longer compiles alone, why: each changed header is compiled once on its own, never
    in every includer's proof. One that failed alone before the change too is not the change's (its includers are
    proved as usual); one that fails only because a broken changed header it includes does is that header's."""
    now = {h: e for h, e in _probed(after, changed).items() if e}
    was = _probed(before, [h for h in now if before.peek(h) is not None])
    now = {h: e for h, e in now.items() if not was.get(h)}
    root = str(after.config.project.root.resolve()) + "/"
    out: dict[str, Finding] = {}
    for path, errors in sorted(now.items()):
        named = {_VIEWS.sub("", h.removeprefix(root)) for h in _HEADER_AT.findall("\n".join(errors))} & now.keys()
        if named and path not in named:
            continue  # the header it includes is the broken one
        said = "; ".join(errors)
        said = said[:_SAID] + (f" ... ({len(said)} chars)" if len(said) > _SAID else "")
        out[path] = Finding(
            "move.header",
            f"{path} does not compile alone: {said}",
            path=path,
            unit=path,
            action="the moves that change it are refused; its includers are not proved",
        )
    return out


def _claims(claims: Sequence[Any]) -> set[tuple[str, str, int, int]]:
    """The ROM ranges the units claim, adjacent ranges of one section joined."""
    rows: defaultdict[tuple[str, str], list[list[int]]] = defaultdict(list)
    for c in sorted(claims, key=lambda c: (c.version, c.section, c.start)):
        run = rows[c.version, c.section]
        if run and run[-1][1] >= c.start:
            run[-1][1] = max(run[-1][1], c.end)
        else:
            run.append([c.start, c.end])
    return {(v, s, a, b) for (v, s), run in rows.items() for a, b in run}


def _unseamed(before: Sequence[Any], after: Sequence[Any], keep: Collection[str]) -> set[str]:
    """The units bordering a change in the claimed ROM ranges that no kept unit borders: the bytes decide, so when
    every rebuilt unit proved exact, a claim shifting between a kept unit and its new neighbour is bookkeeping."""
    old, new = _claims(before), _claims(after)
    left: set[str] = set()
    for v, s, a, b in old ^ new:  # the change spans every joined run on either side that it overlaps
        a, b = (
            min((x for w, t, x, y in old | new if (w, t) == (v, s) and x <= b and y >= a), default=a),
            max((y for w, t, x, y in old | new if (w, t) == (v, s) and x <= b and y >= a), default=b),
        )
        touching = {
            c.unit for c in (*before, *after) if (c.version, c.section) == (v, s) and c.start <= b and c.end >= a
        }
        if not touching & set(keep):
            left |= touching
    return left


def _blamed(edited: Edited, moves: Sequence[Move], units: Sequence[UnitSpec | None]) -> list[Move]:
    """Every move that touched any of the failing units, in batch order; all moves when none can be told."""
    hit = {m.index: m for unit in units for m in _moves_for(edited, moves, unit)}
    return [hit[i] for i in sorted(hit)]


def _failing_claims(
    old: Snapshot, new: Snapshot, edited: Edited, before: Sequence[Claim], after: Sequence[Claim]
) -> list[UnitSpec | None]:
    """The units whose ROM claims differ before and after (`before`, `after`: the claims of `old` and `new`, made once
    by the caller): those holding a range the other side does not claim."""

    def key(claim: Claim) -> tuple[str, str, int, int]:
        return claim.version, claim.section, claim.start, claim.end

    gone, fresh = {key(c) for c in before}, {key(c) for c in after}
    found: dict[str, UnitSpec | None] = {c.unit: new.layout.units.get(c.unit) for c in after if key(c) not in gone}
    for claim in before:
        unit = old.layout.units.get(claim.unit)
        if key(claim) not in fresh and unit and unit.path not in found:
            found[unit.path] = replace(unit, members=tuple(layout.renamed(n, edited.renames) for n in unit.members))
    return list(found.values())


@dataclass(frozen=True)
class Seen:
    """A unit's view as the parent keeps it: its store key and the files it read. The text (megabytes a unit, for
    every unit of a move) stays in the store, and the worker that folds the unit loads it from there."""

    unit: str
    version: str
    key: str
    read: frozenset[str]


def _view_job(item: tuple[Config, Snapshot, UnitSpec]) -> Seen | Refusal:
    config, overlay, unit = item
    try:
        recipe = recipes.resolve(config, unit, unit.options)
        # a fold reads a view's text and its reads, never its line table (2 MB a unit, held for every unit of a move)
        made = view.get(overlay, unit, compare.holders(overlay, unit)[0], recipe, lines=False)
        return Seen(made.unit, made.version, made.key, frozenset(_read(made)))
    except Refusal as refusal:
        return refusal


def _read(made: Any) -> set[str]:
    """The files a view read: its unit and every dependency."""
    if isinstance(made, Seen):
        return set(made.read)
    return {made.unit, *(p for p, _ in made.dependencies)}


Consulted = tuple[frozenset[str], frozenset[str]] | None  # (names, files) a fold read; None: everything


def _fold_job(
    item: tuple[Config, Snapshot, UnitSpec, Any],
) -> tuple[dict[str, bytes | None], tuple[Finding, ...], set[str], Consulted]:
    """One unit's fold: (files it writes, findings, files its view read, what it consulted). A unit without a view
    (None) is viewed again first, one the parent kept as its key is loaded from the store (viewed again when the store
    lost it); a refusal anywhere is its findings."""
    config, overlay, unit, made = item
    try:
        with effort.scoped():
            if isinstance(made, Seen):
                made = view.stored(config, made.unit, made.version, made.key)
            if made is None:
                recipe = recipes.resolve(config, unit, unit.options)
                made = view.get(overlay, unit, compare.holders(overlay, unit)[0], recipe, lines=False)
            if isinstance(made, Refusal):
                raise made
            headers.consulted()  # what an earlier fold read is not this one's
            folded, found = headers.fold(overlay, unit, made, strict=False)
            return folded, tuple(found), _read(made) | {unit.path}, headers.consulted()
    except Refusal as refusal:
        return {}, tuple(replace(f, unit=f.unit or unit.path) for f in refusal.findings), {unit.path}, None


@dataclass(frozen=True)
class Folded:
    """One unit's fold judged where it ran: the writes that differ from what it folded on, its findings, the files its
    view read, the moves its findings blame and whether its writes land."""

    path: str
    fresh: dict[str, bytes | None]
    found: tuple[Finding, ...]
    read: set[str]
    blamed: tuple[Move, ...]
    lands: bool
    consulted: Consulted = None


def _settle(
    path: str,
    done: tuple[dict[str, bytes | None], tuple[Finding, ...], set[str], Consulted],
    here: Snapshot,
    blamed: tuple[Move, ...],
    clean: bool,
) -> Folded:
    """A fold's result judged against the snapshot it folded on. Its writes land when they change a file and no
    finding of it blames a move; proved (`clean` is False after any blamed finding), nothing lands after one."""
    folded, found, read, consulted = done
    fresh = {p: b for p, b in folded.items() if here.peek(p) != b}
    return Folded(path, fresh, found, read, blamed, bool(fresh) and clean and not (found and blamed), consulted)


@dataclass
class Written:
    """What a wave's landed writes changed: the files, the names whose declarations differ in them, and the files whose
    includes differ (every reader of one is affected)."""

    paths: set[str]
    names: set[str]
    broad: set[str]

    def add(self, before: Snapshot, landed: Mapping[str, bytes | None]) -> None:
        for path, text in landed.items():
            self.paths.add(path)
            names = headers.changed(before.peek(path), text, path)
            if names is None:
                self.broad.add(path)
            else:
                self.names |= names

    def stale(self, r: Folded) -> bool:
        """Whether a fold read something a landed write changed: only then is it folded again. One whose view read
        none of the written files is not; nor is one that consulted none of the changed names or whole files."""
        if not r.read & self.paths:
            return False
        if r.consulted is None or r.read & self.broad:
            return True
        names, paths = r.consulted
        return bool(names & self.names or paths & self.paths)


def _fold_module(
    item: tuple[Config, Snapshot, list[tuple[UnitSpec, Any, tuple[Move, ...]]], bool, bool],
) -> list[Folded]:
    """One module's units folded in order in one job, each on the writes the earlier ones landed: every unit of a
    module writes its header, so folding them in one job folds each once. A view that read a file an earlier unit
    rewrote is made again."""
    config, overlay, rows, unproved, clean = item
    here, out = overlay, []
    local: dict[str, bytes | None] = {}
    headers.warm(overlay)  # the wave's indexes stay for the worker's later jobs
    with effort.scoped(
        always=True, keep=("spelled", "declarations", "words", "typed")
    ):  # the job's own overlays and their indexes end with it
        for unit, made, blamed in rows:
            if made is not None and not isinstance(made, Refusal) and _read(made) & local.keys():
                made = None
            result = _settle(unit.path, _fold_job((config, here, unit, made)), here, blamed, clean)
            clean = clean and (unproved or not (result.found and result.blamed))
            if result.lands:
                local.update(result.fresh)
                here = layout.overlay(overlay, local)
            out.append(result)
    return out


def _views(config: Config, overlay: Snapshot, units: Sequence[UnitSpec]) -> dict[str, Any]:
    """The preprocessed view of each unit, made in one pool dispatch; a Refusal stands for a unit that has none."""
    with effort.stage("move.views"):
        made = pool.map(config, "move.views", _view_job, [(config, overlay, u) for u in units])
        return {u.path: v for u, v in zip(units, made, strict=True)}


def newly_unclean(
    before: Snapshot, after: Snapshot, affected: Collection[str]
) -> tuple[list[tuple[str, Finding]], tuple[Finding, ...]]:
    """The affected units holding functions an unclean finding withholds after and not before: their bytes stay exact
    (the move is proved) yet their exact credit would fall, so the write fails naming them. Only the affected units are
    judged, on both sides. A source the rules cannot judge is the census's refusal to make: it comes back as
    non-blocking findings, not as a lost credit."""
    with effort.stage("move.unclean"):
        try:
            return _unclean_losses(before, after, affected), ()
        except Refusal as refusal:
            return [], tuple(replace(f, blocking=False) for f in refusal.findings)


def _unclean_losses(before: Snapshot, after: Snapshot, affected: Collection[str]) -> list[tuple[str, Finding]]:
    mine = {p: after.layout.units[p] for p in affected if p in after.layout.units}
    members = {n for u in mine.values() for n in u.members}
    olds = {p for p, u in before.layout.units.items() if members & set(u.members)}
    was = policy.unclean_rules(before, olds)
    credited = {n for p in olds for n in policy.credited(before, p)}  # a function new to the tree had no credit
    now = policy.unclean_rules(after, set(mine))
    ids = configuration.load_resource("rules.toml").get("unclean", ())
    found: list[tuple[str, Finding]] = []
    for path in sorted(mine):
        text = after.read(path).decode()
        functions = policy.credited(after, path, text)
        lost = [n for n in functions if n in now and n not in was and n in credited]
        if not lost:
            continue
        hits = policy.evaluate(after, path, text, None, False, ids)
        located = [f for f in hits if set(policy.withheld(text, [f], functions)) & set(lost)]
        for f in located[:3] or [Finding(now[lost[0]][0], "fires")]:
            found.append(
                (
                    path,
                    Finding(
                        "move.unclean",
                        f"{path}:{f.line} {f.key} ({f.reason}) would cost {', '.join(lost)} its exact credit",
                        path=path,
                        line=f.line,
                        unit=path,
                        action="keep the function out of this unit or retype it",
                    ),
                )
            )
    return found


def _policy_units(overlay: Snapshot, path: str) -> list[UnitSpec | None]:
    """The units a policy finding belongs to: the unit at its path, or the units that include the header."""
    if path in overlay.layout.units:
        return [overlay.layout.units[path]]
    if configuration.is_header(path):
        return [overlay.layout.units.get(p) for p in headers.consumers(overlay, [path])]
    return [None]


def _changes(snapshot: Snapshot, edited: Edited) -> list[tuple[str, str, str]]:
    """Every path and symbol a batch renamed, as (kind, old, new) rows in a stable order."""
    final = edited.layout
    paths = {old: new for kind, old, new, _ in edited.events if kind == "file" and old != new}
    owner = {n: p for p, u in final.units.items() for n in u.members}
    for path, unit in sorted(snapshot.layout.units.items()):
        first = layout.renamed(unit.members[0], edited.renames) if unit.members else ""
        new = owner.get(first)
        if path not in final.units and new and new != path and new not in snapshot.layout.units:
            paths.setdefault(path, new)
    return [
        *(("path", old, new) for old, new in sorted(paths.items())),
        *(("symbol", old, new) for old, new in sorted({**edited.types, **edited.renames}.items()) if old != new),
    ]


def _result(moves: Sequence[Move], snapshot: Snapshot, after: Snapshot | None = None, **extra: Any) -> Json:
    """The result document; `after` is the proved map when the moves were written."""
    done = after or snapshot
    return {
        "moves": len(moves),
        "applied": {op: sum(m.op == op for m in moves) for op in _OPS},
        "units": {"before": len(snapshot.layout.units), "after": len(done.layout.units)},
        "headers": {"before": len(headers.sources(snapshot)), "after": len(headers.sources(done))},
        "proved": {"units": 0, "proofs": 0, "exact": 0, "versions": []},
        "extracted": [],
        "files": [],
        "deleted": [],
        "refused": [],
        "message": f"move: {len(moves)} moves, nothing to write",
        **extra,
    }


def _refusal(moves: Sequence[Move], snapshot: Snapshot, rows: Sequence[tuple[Move, Finding]]) -> Json:
    by: defaultdict[int, list[Finding]] = defaultdict(list)
    rows = [(move, _capped(found)) for move, found in rows]
    for move, found in rows:
        by[move.index].append(found)
    refused = [
        {
            "index": i,
            "move": {"op": moves[i].op, "subject": moves[i].subject, "target": moves[i].target},
            "findings": [asdict(f) for f in fs],
        }
        for i, fs in sorted(by.items())
    ]
    distinct = Counter((f.key, f.reason) for _, f in rows)  # each cause once, however many moves it refuses
    conflicts = [{"key": k, "reason": r, "moves": n} for (k, r), n in distinct.most_common()]
    return _result(
        moves,
        snapshot,
        refused=refused,
        conflicts=conflicts,
        message=f"move: {len(refused)} of {len(moves)} moves refused by {len(conflicts)} distinct causes",
    )


def _causes(rows: Mapping[int, Json], blamed: Sequence[int]) -> list[list[int]]:
    """The blamed batch positions grouped by the cause their first finding names (its key, unit and reason), in batch
    order: one group is the set of moves one broken unit or rule refused together."""
    groups: dict[tuple[str, str, str], list[int]] = {}
    for k in sorted(blamed):
        found = rows[k]["findings"][0] if rows[k].get("findings") else {}
        cause = (found.get("key", ""), found.get("unit", ""), found.get("reason", "")) if found else ("", "", str(k))
        groups.setdefault(cause, []).append(k)  # a row naming no finding is a cause of its own
    return list(groups.values())


def _subjects(refused: Sequence[Json]) -> list[Json]:
    """Each refused subject once (a module's source and header moves refuse together), with its first finding and the
    indexes of all its refused moves."""
    by: dict[str, Json] = {}
    for row in refused:
        found = row["findings"][0] if row["findings"] else {}
        entry = by.setdefault(
            row["move"]["subject"],
            {
                "subject": row["move"]["subject"],
                "key": found.get("key", ""),
                "reason": found.get("reason", ""),
                "moves": [],
            },
        )
        entry["moves"].append(row["index"])
    return list(by.values())


_CONTENT_KEYED = ("declarations", "words", "spelled", "typed")  # memo kinds keyed by content: valid in any round


def _capped(found: Finding) -> Finding:
    """A finding with its reason and each missing line cut to their head: a compile error quoting whole headers is
    100 KB, and a round can fail thousands of units."""

    def cut(text: str) -> str:
        return text if len(text) <= _SAID else text[:_SAID] + f" ... ({len(text)} chars)"

    return replace(found, reason=cut(found.reason), missing=tuple(cut(m) for m in found.missing))


def run(config: Config, params: Json) -> Json:
    wait_s = config.host.move_wait_s
    with effort.stage("move.run"), store.exclusive(config, "structure", wait=True, wait_s=wait_s) as held:
        if not held:
            raise Refusal(Finding("setup.busy", configuration.sentence("setup.busy")))
        journal.recover(config)
        moves, stamped = _read_stamped(params, sys.stdin.buffer.read() if params["op"] == "-" else b"")
        if stamped and (current := layout.capture(config).layout.digest) not in stamped:
            raise Refusal(
                Finding(
                    "move.stale",
                    f"the move stream was made for layout {stamped[0]} and the layout is {current}",
                    action="run unbake map --moves again",
                )
            )
        refused: list[Json] = []
        original, rounds, chained, landed = moves, 0, 0, False
        unproved = bool(params.get("unproved"))
        partial = bool(params.get("partial")) or unproved
        files: dict[str, None] = {}
        deleted: dict[str, None] = {}
        # each round applies one batch of the stream's indexes. The moves a round's findings blame for one cause (one
        # unit that stops matching) are split: the first halves of every such set apply with the unblamed rest next
        # round and the second halves the round after, so a set of 17 with one breaker refuses that one in about
        # 2*log2(17) rounds while every split set of the command shares them. A cause blaming one move refuses it. A
        # move the plan refuses beside others is tried once alone, after the rest.
        queue: list[list[int]] = [list(range(len(moves)))]
        alone: set[int] = set()
        bounds: list[str] = []  # members a file starts at, found by this command's proofs
        kept: dict[str, list[Json]] = {}  # old units whose proof failed, with why: they stay as they are
        size: dict[str, int] = {}
        # every requeue below is bounded: boundary re-applies share _REAPPLY, kept units and boundaries only grow, and
        # a batch met again with the same boundaries and kept units would apply the same, so it is refused instead
        reapplied, seen, split = 0, set(), 0
        per_unit = partial or any(m.op == "relayout" for m in moves)  # a stream fails per unit; one move all or nothing
        # every round's outcome is journalled as it ends: a killed or crashed command resumes after its last round,
        # with its landed files, kept units, boundaries and refusals, and proves none of them again
        stream = digest(([(m.op, m.subject, m.target) for m in moves], partial, unproved))
        saved = _resumed(config, stream)
        if saved is None:
            (config.project.root / ".unbake/move/rounds.jsonl").unlink(missing_ok=True)  # this command's rounds only
        else:
            queue, refused, kept, size, bounds = (
                saved["queue"],
                saved["refused"],
                saved["kept"],
                saved["size"],
                saved["bounds"],
            )
            alone, seen = set(saved["alone"]), {(tuple(a), b, c) for a, b, c in saved["seen"]}
            rounds, chained, reapplied, landed = saved["rounds"], saved["chained"], saved["reapplied"], saved["landed"]
            split = saved.get("split", 0)
            files, deleted = dict.fromkeys(saved["files"]), dict.fromkeys(saved["deleted"])
            said = f"{len(kept)} units kept, {len(refused)} moves refused, {sum(map(len, queue))} moves left"
            effort.say(f"unbake: move resumes after round {rounds}: {said}")
        result: Json = _result(moves, layout.capture(config))
        while queue:
            _journal(
                config,
                stream,
                {
                    "queue": queue,
                    "refused": refused,
                    "kept": kept,
                    "size": size,
                    "bounds": bounds,
                    "alone": sorted(alone),
                    "seen": sorted(seen),
                    "rounds": rounds,
                    "chained": chained,
                    "reapplied": reapplied,
                    "split": split,
                    "landed": landed,
                    "files": list(files),
                    "deleted": list(deleted),
                },
            )
            ids = queue.pop(0)
            state = (tuple(ids), len(bounds), len(kept))
            if state in seen:
                refused += _rows(original, ids, Finding("move.rounds", "the same batch came back unchanged"))
                continue
            seen.add(state)
            batch = tuple(replace(original[x], index=k) for k, x in enumerate(ids))
            rounds += 1
            # a round keeps nothing for the next: that one captures the tree again, so what this one memoised (its
            # overlays, indexes and pool blobs) goes when it ends
            with effort.stage(f"move.round.{rounds}"), effort.scoped(always=True, keep=_CONTENT_KEYED):
                result = (
                    _apply(config, batch, unproved=True)
                    if unproved
                    else _apply(config, batch, False, bounds, kept, per_unit)
                )
            result = dict(result)
            if more := result.pop("keep", {}):  # units whose proof failed keep their old files: the same moves again
                kept.update(more)  # (the next pass proves only what changed: the kept units' neighbours)
                _log_round(config, rounds, {u: more[u] for u in sorted(more)}, ())
                size.update(result.pop("bytes", {}))
                bounds += [b for b in result.pop("boundaries", ()) if b not in bounds]
                queue.insert(0, ids)
                continue
            found = [b for b in result.pop("boundaries", ()) if b not in bounds]
            bounds += found  # file boundaries a proof found: later rounds apply with them
            _log_round(config, rounds, {}, result.get("refused", ()))
            if found and reapplied < _REAPPLY:  # the same batch again with them, _REAPPLY times per command at most;
                reapplied += 1  # then this round's refusals stand, so the rest applies (and is written) next round
                queue.insert(0, ids)
                continue
            planned = result.pop("planned", False)
            files.update(dict.fromkeys(result.get("files", ())))
            deleted.update(dict.fromkeys(result.get("deleted", ())))
            if not (partial and result["refused"]):
                landed = landed or not result["refused"]
                continue
            rows = {r["index"]: r for r in result["refused"]}
            final, blamed = [], []
            for k in rows:
                if planned and len(ids) > 1 and ids[k] not in alone:
                    alone.add(ids[k])
                    queue.append([ids[k]])
                else:
                    blamed.append(k)
            firsts, seconds = [], []
            for group in _causes(rows, blamed):
                if len(group) == 1 or split >= _SPLITS:
                    final += group
                else:  # one cause blamed several moves: which of them break it is found by halves
                    firsts += [ids[k] for k in group[: len(group) // 2]]
                    seconds += [ids[k] for k in group[len(group) // 2 :]]
            refused += [{**rows[k], "index": ids[k]} for k in final]
            rest = [x for k, x in enumerate(ids) if k not in rows]
            if seconds:  # halving has its own bound (_SPLITS), so it spends none of the chained rounds
                split += 1
                queue[:0] = [sorted(rest + firsts), sorted(seconds)]
                continue
            chained += bool(rest)
            if rest and chained >= _ROUNDS and not unproved:
                # chained refusals cost a full apply each: the rest are refused as one finding
                refused += _rows(original, rest, Finding("move.rounds", f"{_ROUNDS} partial rounds were used"))
            elif rest:
                queue.insert(0, rest)
        if refused:
            refused.sort(key=lambda r: r["index"])
            gone = {r["index"] for r in refused}
            done = "the rest applied" if landed else "all refused"
            subjects = _subjects(refused)
            result = {
                **result,
                "applied": {op: sum(m.op == op for m in original if m.index not in gone) for op in _OPS},
                "files": list(files),
                "deleted": list(deleted),
                "refused": refused,
                "subjects": subjects,
                "moves": len(original),
                "message": f"move: {len(refused)} of {len(original)} moves refused, {done} ({len(subjects)} subjects)",
            }
        result = {**result, "rounds": rounds, **({"boundaries": sorted(bounds)} if bounds else {})}
        if kept:  # the work list: every kept unit by class, written beside the project's other state
            census = _census(kept, size)
            result = {**result, "kept": kept, "census": census}
            store.write(config.project.root / ".unbake/move/census.json", json.dumps(census, indent=1).encode())
        if any(m.op == "relayout" for m in original) or partial:
            result["converged"] = True  # what is left is the work list, not a failure
        (config.project.root / _RESUME).unlink(missing_ok=True)  # done: a later run starts afresh
        return result


_RESUME = ".unbake/move/resume.json"


def _tree(config: Config, files: Collection[str]) -> str:
    """What a resumed move needs unchanged: the layout, every source and header as it stands (path, size, mtime) and
    the bytes of the files it wrote so far. A round killed after its write left a tree no journal names: it starts
    afresh."""
    root = config.project.root
    stats = []
    for place in ("src", "include"):
        for folder, _, names in os.walk(root / place):
            for name in names:
                st = os.stat(f"{folder}/{name}")
                stats.append((f"{folder}/{name}", st.st_size, st.st_mtime_ns))
    return digest(
        (
            layout.capture(config).layout.digest,
            sorted(stats),
            [(p, (root / p).read_bytes() if (root / p).is_file() else None) for p in sorted(files)],
        )
    )


def _journal(config: Config, stream: str, state: Json) -> None:
    """The move's state after its last round, written atomically before the next one starts."""
    path = config.project.root / _RESUME
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = _tree(config, state["files"])
    store.write(path, json.dumps({"stream": stream, "tree": tree, **state}, default=str).encode())


def _resumed(config: Config, stream: str) -> Json | None:
    """The journalled state of the same stream on the same tree, else None (a different stream, or a tree changed
    since: the move starts afresh)."""
    try:
        state = json.loads((config.project.root / _RESUME).read_bytes())
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or state.get("stream") != stream:
        return None
    return state if state.get("tree") == _tree(config, state.get("files", ())) else None


def _log_round(config: Config, round_: int, kept: Mapping[str, list[Json]], refused: Sequence[Json]) -> None:
    """One line per round, written as it ends (.unbake/move/rounds.jsonl) and said: the units it kept and why, the
    moves it refused, so a watcher sees progress and a killed command still names what failed."""
    line = {
        "round": round_,
        "kept": {
            u: [f.get("key", "") + ": " + str(f.get("reason", ""))[:200] for f in found] for u, found in kept.items()
        },
        "refused": [
            {"index": r.get("index"), "move": r.get("move"), "findings": r.get("findings", [])[:2]} for r in refused
        ],
    }
    path = config.project.root / ".unbake/move/rounds.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps(line, default=str) + "\n")
    effort.say(f"unbake: move round {round_}: {len(kept)} units kept, {len(refused)} moves refused")


def _rows(moves: Sequence[Move], ids: Sequence[int], finding: Finding) -> list[Json]:
    """Refused rows for the moves at these indexes, all for one finding the command found (move these again)."""
    found = asdict(replace(finding, action="move these again"))
    return [
        {
            "index": x,
            "move": {"op": moves[x].op, "subject": moves[x].subject, "target": moves[x].target},
            "findings": [found],
        }
        for x in ids
    ]


def _bytes_changed(gap: Finding) -> bool:
    """Whether a proof built the unit and its bytes differ (not a compile, link or symbol failure)."""
    return bool(gap.missing) and all(
        re.search(r"\bdiffer\b|\bsize\b", m) and not re.search(r"error|unresolved|link", m) for m in gap.missing
    )


def _boundaries(snapshot: Snapshot, unit: UnitSpec) -> set[str]:
    """The members of a new unit that start another old unit's run: where its old files began."""
    found, before = set(), None
    for n in unit.members:
        here = getattr(layout.unit_of(snapshot, n), "path", None)
        if before is not None and here != before:
            found.add(n)
        before = here
    return found


def _keep_together(snapshot: Snapshot, final: Any, why: dict[str, list[Finding]]) -> None:
    """Every old unit that shares a new unit with a kept one is kept too, until none is left: keeping a unit changes
    each new unit its members went into, so their other old units would only fail the next pass (a claims cascade).
    One pass keeps the whole group of old and new units the failure touched."""
    news = {n: path for path, u in final.units.items() for n in u.members}
    lay = snapshot.layout
    members: defaultdict[str, list[str]] = defaultdict(list)  # old module -> its old units
    for path, spec in lay.units.items():
        members[spec.group].append(path)
    done: set[str] = set()  # modules already kept whole
    todo = list(why)
    while todo:  # both relations to a fixpoint: one pass keeps the whole component
        old = todo.pop()
        if old not in lay.units:
            continue
        # a kept unit reflows the rest of its module (layout.physical joins around it): every unit of it stays
        if (group := lay.units[old].group) not in done:
            done.add(group)
            for path in members[group]:
                if path not in why:
                    why[path] = [Finding("move.not_exact", f"kept with its module {group}", unit=path)]
                    todo.append(path)
        for new in {news[n] for n in lay.units[old].members if n in news}:
            for n in final.units[new].members:
                if (unit := layout.unit_of(snapshot, n)) is not None and unit.path not in why:
                    why[unit.path] = [Finding("move.not_exact", f"kept with {old}: they share {new}", unit=unit.path)]
                    todo.append(unit.path)


def _kept(
    snapshot: Snapshot,
    final: Any,
    failing: Mapping[str, list[Finding]],
    keep: Collection[str],
    moves: Sequence[Move] = (),
    stuck: Any = None,
) -> Json:
    """The old units whose proof failed (a new unit's failure keeps every old unit it came from), why, how many bytes
    each holds and the boundaries that keep each apart in later layouts: its first member and the member after it.
    When every failing unit is kept already, what fails is what they read: `stuck(failing)` names the moves that
    changed it and those are refused, never the batch's first move."""
    lay = snapshot.layout
    why: dict[str, list[Finding]] = defaultdict(list)
    for path, found in failing.items():
        unit = final.units.get(path)
        olds = (
            {path}
            if path in lay.units and (unit is None or unit.members == lay.units[path].members)
            else {u.path for n in (unit.members if unit else ()) if (u := layout.unit_of(snapshot, n)) is not None}
        ) | ({path} if path in lay.units else set())  # an old unit that fails under its own path keeps itself too
        for old in olds:
            why[old] += found
    _keep_together(snapshot, final, why)
    if not why or set(why) <= set(keep):  # nothing new to keep: the failure is not any unit's
        if stuck is not None and moves and not any(m.op == "relayout" for m in moves):
            found = [_capped(f) for fs in failing.values() for f in fs][:1]
            rows = [(m, f) for m in stuck({u: fs for u, fs in failing.items() if u}) for f in found]
            return {**_refusal(moves, snapshot, rows), "unkept": True}
        rows = [(Move(0, "relayout", "", ""), _capped(f)) for found in failing.values() for f in found]
        return {**_refusal([Move(0, "relayout", "", "")], snapshot, rows), "unkept": True}
    bounds: set[str] = set()
    for old in why:
        members = lay.units[old].members
        order = lay.groups[lay.units[old].group].members if lay.units[old].group in lay.groups else ()
        bounds.add(members[0])
        at = max((order.index(n) for n in members if n in order), default=-1)
        bounds.update(n for n in order[at + 1 : at + 2] if n not in members)
    size = {
        old: sum(
            p.size or p.rom_end - p.rom_start
            for n in lay.units[old].members
            for p in lay.members[n].placements
            if p.version == lay.members[n].reference(snapshot.config.project.names_from)
        )
        for old in why
    }
    return {
        "keep": {old: [asdict(_capped(f)) for f in found] for old, found in sorted(why.items())},
        "boundaries": sorted(bounds),
        "bytes": size,
    }


_QUOTED = re.compile(r"`[^']*'|\b0x[0-9A-Fa-f]+\b|\b\d+\b")


def _class(finding: Json) -> str:
    """A finding's class for the census: its key and the first compiler message or gap it names, names and numbers
    blanked, so one class gathers every unit that fails one way."""
    said = [m for m in finding.get("missing", ()) if m] or [finding.get("reason", "")]
    text = said[0].split(": ", 1)[1] if said[0].startswith("version ") else said[0]
    lines = [x for x in text.splitlines() if re.search(r":\d+: ", x) and "In file included" not in x]
    message = re.sub(r"^.*?:\d+: ", "", lines[0]) if lines else text.splitlines()[0] if text else ""
    return f"{finding.get('key', '')}: {_QUOTED.sub('N', message)[:120]}"


def _census(kept: Mapping[str, list[Json]], size: Mapping[str, int]) -> list[Json]:
    """The kept units ranked as a work list: per class the units, their count and bytes, most bytes first."""
    by: defaultdict[str, set[str]] = defaultdict(set)
    for unit, found in kept.items():
        by[_class(found[0]) if found else "unknown"].add(unit)
    rows = [
        {"class": c, "units": sorted(u), "count": len(u), "bytes": sum(size.get(x, 0) for x in u)}
        for c, u in by.items()
    ]
    return sorted(rows, key=lambda r: (-r["bytes"], -r["count"], r["class"]))


def _apply(
    config: Config,
    moves: Sequence[Move],
    unproved: bool = False,
    boundaries: Collection[str] = (),
    keep: Collection[str] = (),
    per_unit: bool = False,
) -> Json:
    """One apply of the moves. Unproved, it runs every stage that writes text (edit, relocate, compose, respell, views,
    fold, generate) for real and skips the proof, the claims compare and the policy gates; every unit's fold runs and
    each failing unit's moves come back refused together, so the caller drops them and applies the rest."""
    with effort.stage("move.apply"):
        snapshot = layout.capture(config)
        if not moves:
            return _result(moves, snapshot)
        edited = edit(snapshot, moves, boundaries, keep)
        # a stream's (partial or relayout) items are its units: one whose proof fails at any stage keeps its old files
        # (the caller applies the moves again with it kept) and every stage still runs, so one apply names every unit
        # that fails and the rest land; a batch is never refused whole for one unit
        converge = not unproved and (per_unit or any(m.op == "relayout" for m in moves))
        failing: defaultdict[str, list[Finding]] = defaultdict(list)  # per unit path, old or new, why it fails
        dropped: list[tuple[Move, Finding]] = []  # unproved: every refusal of this round, all classes, then one return
        while unproved and edited.bad:  # the map refuses some: the rest are edited again, so their dependents show
            dropped += edited.bad
            gone = {m.index for m, _ in dropped}
            edited = edit(snapshot, [m for m in moves if m.index not in gone], boundaries, keep)
        if edited.bad:
            return {**_refusal(moves, snapshot, edited.bad), "planned": True}
        live = [m for m in moves if m.index not in {d.index for d, _ in dropped}]
        unresolved: list[Finding] = []

        def blame(found: Finding, hit: Sequence[Move]) -> None:
            """Unproved: the finding refuses the moves that made it; with none, it is left as debt."""
            dropped.extend((m, found) for m in hit)
            if not hit:
                unresolved.append(replace(found, blocking=False))

        final = edited.layout
        was = {new: old for old, new in edited.renames.items()}
        try:
            failed: list[tuple[str, str, tuple[Finding, ...]]] | None = [] if unproved else None
            with effort.stage("move.relocate"):
                header_texts, relocated, stay = headers.relocate(snapshot, edited.events, failed)
        except Refusal as refusal:
            rows = [(m, f) for f in refusal.findings for m in _cycle_moves(edited, moves, f.path)]
            return _refusal(moves, snapshot, rows)
        for source, target, reasons in failed or ():  # unproved: every merge that refuses, found in one pass
            for f in reasons:
                blame(f, _header_moves(edited, live, source, target))
        exists = lambda p: header_texts[p] is not None if p in header_texts else snapshot.peek(p) is not None  # noqa: E731
        with effort.stage("move.compose"):
            units, before, problems = layout.compose(snapshot, final, was, exists)
        if problems:
            blamed: dict[tuple[str, str], dict[int, Move]] = defaultdict(dict)  # per finding the moves of all its units
            for unit_path, f in problems:
                touching = _moves_for(edited, live, snapshot.layout.units[unit_path], every=not unproved)
                blamed[f.key, f.reason].update({m.index: m for m in touching})
            clash = {(f.key, f.reason): f for _, f in problems}
            for unit_path, f in problems if converge else ():
                failing[unit_path].append(f)
            if not unproved and not converge:
                return _refusal(moves, snapshot, [(m, clash[k]) for k, by in blamed.items() for m in by.values()])
            for k, by in blamed.items():  # a clash no move makes (the layout puts the units together) is debt
                blame(clash[k], list(by.values()))
        files: dict[str, str | None] = {**header_texts, **{p: b.decode() for p, b in units.items()}}
        files.update({p: None for p in snapshot.layout.units.keys() - final.units.keys()})
        with effort.stage("move.shape"):
            files.update(_shape(snapshot, files, edited))
        with effort.stage("move.respell"):
            spelled = headers.respell(
                snapshot, files, relocated, stay, {**edited.types, **edited.renames}, edited.folded
            )
        writes = {**edited.writes, **spelled}
        for old, new in edited.renames.items():  # a fuzzy source is named after its member
            path = snapshot.layout.fuzzy.get(old, {}).get("path")
            if path and (target := final.fuzzy[new]["path"]) != path:
                writes[target], writes[path] = writes.get(path) or snapshot.read(path), None
        overlay = layout.overlay(snapshot, writes)
        called = {m[1] for b in units.values() for m in re.finditer(rb"(?m)^extern\b[^;{}]*?\b(\w+)\s*\(", b)}
        owners = {p for p, b in units.items() if called & {n.encode() for n in headers.defined(b)}}
        frozen = headers._closure(snapshot, list(keep)) if keep else set()
        with effort.stage("move.fold"):
            # an owner folds first: its prototype is in its header before a caller's extern of it is folded; a unit
            # whose module header a kept unit reaches folds nothing (the kept unit must read today's headers)
            order = sorted(
                (p for p in units if final.groups[final.units[p].group].header not in frozen),
                key=lambda p: (p not in owners, p),
            )
            # every unit's view is made once, together, from the overlay of the moves; a fold that then changes a file
            # a later view read makes that view again
            with effort.stage("move.warm"):  # the workers fork holding the indexes; a later wave's overlay derives
                headers.warm(overlay)  # its own from them
            # the parent keeps each view's key and reads, never its text: the fold's worker loads that from the store
            views: dict[str, Any] = _views(config, overlay, [final.units[p] for p in order])
            touched: set[str] = set()
            bad: list[tuple[Move, Finding]] = []
            # waves over the pool, owners first, one job per module: a result whose view read a file an earlier job
            # of its wave wrote (or that writes a file one did) is folded again in a later wave, on the overlay that
            # holds those writes, and so is every later unit of its job
            blames = {p: tuple(_moves_for(edited, live, final.units[p], every=not unproved)) for p in order}
            waiting, first = list(order), set(owners)
            # a unit that wrote a file an earlier job of its wave wrote waits in one job per such file: the writers of
            # one file fold in turn in that job, each on the others' writes, so a chain of N writers costs one wave
            chain: dict[str, str] = {}

            def key(p: str) -> str:
                return final.units[p].group or p

            waves = 0
            while waiting:
                wave = [p for p in waiting if p in first] or waiting
                modules: dict[str, list[str]] = defaultdict(list)
                for p in wave:
                    made = views.get(p)
                    if made is not None and not isinstance(made, Refusal) and _read(made) & touched:
                        views[p] = None  # its view predates a fold of an earlier wave
                    modules[chain.get(p) or key(p)].append(p)
                jobs = [
                    (
                        config,
                        overlay,
                        [(final.units[p], views.get(p), blames[p]) for p in paths],
                        unproved,
                        converge or not bad,
                    )
                    for paths in modules.values()
                ]
                fresh_here: set[str] = set()
                landing = Written(set(), set(), set())
                later: list[str] = []
                biggest = sorted(range(len(jobs)), key=lambda i: -len(jobs[i][2]))  # the longest jobs start first
                waves += 1
                with effort.stage(f"move.fold.wave.{waves}"):  # each wave is its own stage: it closes once
                    folded = pool.map(config, "move.fold", _fold_module, [jobs[i] for i in biggest])
                ran = dict(zip(biggest, folded, strict=True))
                for results in (ran[i] for i in range(len(jobs))):  # merged in module order
                    landed: dict[str, bytes | None] = {}
                    blocked = ""  # a unit that met an earlier job's writes waits, and every later unit of its job
                    for r in results:
                        if blocked or landing.stale(r) or r.fresh.keys() & fresh_here:
                            if not blocked:
                                views[r.path] = None
                                blocked = min(r.fresh.keys() & fresh_here, default=chain.get(r.path) or key(r.path))
                            chain[r.path] = blocked
                            later.append(r.path)
                            continue
                        if unproved:
                            for f in r.found:
                                blame(f, r.blamed)
                        elif converge and r.found and r.blamed:
                            failing[r.path] += r.found
                        else:
                            bad += [(m, f) for f in r.found for m in r.blamed]
                        if r.lands and (unproved or converge or not bad):
                            landed.update(r.fresh)
                    landing.add(overlay, landed)  # the wave folded on overlay: no file lands twice in one wave
                    writes.update(landed)
                    fresh_here.update(landed)
                done = set(wave) - set(later)
                first -= done
                waiting = [p for p in waiting if p not in done]
                if fresh_here:
                    touched |= fresh_here
                    laid, overlay = overlay, layout.overlay(snapshot, writes)
                    effort.release(laid.digest)  # the wave before is never read again: its overlay and blobs go
            if bad:
                return _refusal(moves, snapshot, bad)
        if dropped:  # unproved: this round's every refusal at once; the caller applies the rest in the next round
            return _refusal(moves, snapshot, dropped)
        writes = {p: b for p, b in writes.items() if snapshot.peek(p) != b}
        if not writes:
            return _result(moves, snapshot)

        def stuck(failing: Mapping[str, Sequence[Finding]]) -> list[Move]:
            """Kept units that still fail read a changed header: the moves that added an include on the chain their
            compiler message names (`_edge_moves`), else the moves that changed the changed headers it names (the
            error's file and its include chain), else every changed header they reach."""
            after = layout.overlay(snapshot, writes)
            changed = {p for p, b in writes.items() if p.startswith("include/") and b is not None}
            root = str(config.project.root.resolve()) + "/"
            hit: dict[int, Move] = {}
            for unit, found in failing.items():
                reach = headers._closure(after, [unit]) & changed
                said = "\n".join(t for f in found for t in (f.reason, *f.missing))
                if edges := _edge_moves(snapshot, after, live, said, root, reach):
                    hit.update((m.index, m) for m in edges)  # the include a move added, not the whole chain
                    continue
                named = {_VIEWS.sub("", h.removeprefix(root)) for h in _HEADER_AT.findall(said)} & reach
                for p in sorted(named or reach):
                    hit.update((m.index, m) for m in _file_moves(snapshot, after, edited, live, p))
            units = [after.layout.units.get(u) for u in failing]
            return [hit[i] for i in sorted(hit)] or _blamed(edited, live, units)

        with effort.stage("move.generate"):
            try:
                generated = repo.files(layout.overlay(snapshot, writes))
            except Refusal as refusal:
                if converge and all(f.unit for f in refusal.findings):
                    for f in refusal.findings:
                        failing[f.unit].append(f)
                    return _kept(snapshot, final, failing, keep, moves, stuck)
                rows = [
                    (m, f)
                    for f in refusal.findings
                    for m in _moves_for(edited, live, final.units.get(f.unit), every=not unproved)
                ]
                return _refusal(moves, snapshot, rows or [(m, f) for f in refusal.findings for m in live])
        writes.update({p: b for p, b in generated.items() if snapshot.peek(p) != b})
        overlay = layout.overlay(snapshot, writes)
        changed = [p for p, b in writes.items() if p.startswith("include/") and b is not None]
        if changed and not unproved and not any(m.op == "relayout" for m in moves):
            with effort.stage("move.headers"):  # a broken header is found once, never in each includer's proof
                broken = _broken_headers(snapshot, overlay, changed)
            if broken:
                rows = [(m, f) for p, f in broken.items() for m in _file_moves(snapshot, overlay, edited, live, p)]
                return _refusal(moves, snapshot, rows)
        with effort.stage("move.readers"):
            readers = headers.readers(snapshot, overlay, changed)  # as a publish judges its consumers
        # a unit whose flag set changed builds other bytes from the same text: it is proved too
        prior = snapshot.layout.units
        flagged = {p for p, u in overlay.layout.units.items() if p in prior and prior[p].options != u.options}
        affected = sorted({*(p for p in writes if p in overlay.layout.units), *readers, *flagged})
        with effort.stage("move.proof"):
            proofs, gaps = publish._proofs(overlay, affected) if affected and not unproved else ((), ())
        refused = [
            (m, replace(g, key="move.not_exact", unit=holder.path))
            for g in gaps
            if (holder := layout.unit_of(overlay, g.unit))
            for m in _moves_for(edited, moves, holder)
        ]
        # a merge whose proof changes bytes is evidence of a file boundary: the map records where the merged unit's
        # old files began and the caller applies the moves again with it (layout.physical never joins them again)
        found = {
            n
            for g in gaps
            if _bytes_changed(g) and (holder := layout.unit_of(overlay, g.unit))
            for n in _boundaries(snapshot, holder)
        } - set(boundaries)
        if converge:
            for g in gaps:
                if holder := layout.unit_of(overlay, g.unit):
                    failing[holder.path].append(replace(g, key="move.not_exact", unit=holder.path))
            if failing:
                return _kept(snapshot, final, failing, keep, moves, stuck)
            refused = []
        if refused and found:
            return {**_refusal(moves, snapshot, refused), "boundaries": sorted(found)}
        claims = not refused and not unproved
        # the units the writes reach (the proved ones, and any includer of a written non-header file) are located
        # again; every other unit keeps the old pass's placements (headers.readers decides a header edit's reach)
        reached = {*affected, *headers.consumers(overlay, [p for p in writes if not configuration.is_header(p)])}
        old_claims, new_claims = (
            (ownership.claims(snapshot)[1], ownership.claims(overlay, reuse=(snapshot, reached))[1])
            if claims
            else ((), ())
        )
        if claims and _claims(new_claims) != _claims(old_claims):
            lost = Finding("move.not_exact", "the unit claims differ from the old claims", missing=("claims",))
            unseamed = _unseamed(old_claims, new_claims, keep)  # a change at a kept unit's seam is bookkeeping
            losing = [
                u
                for u in _failing_claims(snapshot, overlay, edited, old_claims, new_claims)
                if u is None or u.path in unseamed
            ]
        else:
            losing = []
        if losing:
            if converge:
                for u in losing:
                    failing[u.path if u else ""].append(lost)
                return _kept(snapshot, final, failing, keep, moves, stuck)
            refused = [(m, lost) for m in _blamed(edited, moves, losing)]
        if refused:
            return _refusal(moves, snapshot, refused)
        census: list[Finding] = []
        if not unproved:
            baseline = layout.overlay(snapshot, {**edited.writes, **{p: t.encode() for p, t in before.items()}})
            changes = {
                p: (
                    before.get(p) or (snapshot.peek(p) or b"").decode(errors="replace"),
                    (b or b"").decode(errors="replace"),
                )
                for p, b in writes.items()
                if configuration.is_text(p)
            }
            with effort.stage("move.policy"):
                try:
                    census = list(policy.census(overlay, changes=changes, baseline=baseline)[0])
                except Refusal as refusal:  # a rule that refuses outright is a blocking finding at its path all the
                    census = [replace(f, blocking=True) for f in refusal.findings]  # same: its units keep or refuse
            # an unclean rule costs a unit its credit, which move.unclean guards: here it is debt, never a refusal
            unclean = set(configuration.load_resource("rules.toml").get("unclean", ()))
            census = [replace(f, blocking=False) if f.key.removeprefix("source.") in unclean else f for f in census]
        if converge and (blocking := [replace(f, key="move.policy") for f in census if f.blocking]):
            for f in blocking:
                for u in _policy_units(overlay, f.path):
                    failing[u.path if u else ""].append(f)
            return _kept(snapshot, final, failing, keep, moves, stuck)
        if blocking := [replace(f, key="move.policy") for f in census if f.blocking]:
            rows = [(m, f) for f in blocking for m in _blamed(edited, moves, _policy_units(overlay, f.path))]
            return _refusal(moves, snapshot, rows)
        newly, unjudged = newly_unclean(snapshot, overlay, affected) if not unproved else ([], ())
        if converge and keep:  # a kept unit's text is unchanged: a rule its new headers fire is census debt
            unjudged = (*unjudged, *(replace(f, blocking=False) for p, f in newly if p in keep))
            newly = [(p, f) for p, f in newly if p not in keep]
        if newly and converge:
            for path, f in newly:  # a unit the generate step split (name_15) maps back to the old units of its members
                unit = overlay.layout.units.get(path)
                olds = {u.path for n in (unit.members if unit else ()) if (u := layout.unit_of(snapshot, n))}
                for old in olds | ({path} if path in snapshot.layout.units or not olds else set()):
                    failing[old].append(f)
            return _kept(snapshot, final, failing, keep, moves, stuck)
        if newly:
            rows = [(m, f) for path, f in newly for m in _blamed(edited, moves, [overlay.layout.units.get(path)])]
            return _refusal(moves, snapshot, rows)
        shape = f"{len(snapshot.layout.units)} -> {len(overlay.layout.units)} units"
        count = f"{len(headers.sources(snapshot))} -> {len(headers.sources(overlay))} headers"
        left = f", {len(unresolved)} unit clashes left for the layout" if unresolved else ""
        message = f"move: {len(moves)} moves{' (unproved)' if unproved else ''}, {shape}, {count}{left}"
        debt = (*(f for f in census if not f.blocking), *unjudged, *unresolved, *edited.split)
        plan = publish.make_plan("move", snapshot, writes, tuple(affected), debt, message)
        written = journal.apply(config, plan, snapshot)
        moved.record(config, _changes(snapshot, edited), overlay.layout.digest)
        extracted = sorted(v for v in config.project.versions if config.project.version_files[v].split in writes)
        with effort.stage("move.extract"):
            for version in extracted:
                build.extract(config, version)
        return _result(
            moves,
            snapshot,
            overlay,
            proved={
                "units": len(affected),
                "proofs": len(proofs),
                "exact": sum(p.exact for p in proofs),
                "versions": sorted({p.version for p in proofs}),
            },
            extracted=extracted,
            files=list(written),
            deleted=sorted(p for p in written if plan.writes[p] is None),
            message=message,
        )


def _record_types(snapshot: Snapshot) -> _Types:
    """The batch type names of the headers with every record and typedef the sources declare (fuzzy ones too), once
    per process and tree state."""

    def build() -> _Types:
        types = _Types(snapshot)
        for path in headers.sources(snapshot, "src", ".c"):
            for key, _, statement in headers._declared(snapshot.read(path).decode(errors="replace")):
                kind, _, tag = key.partition(" ")
                if tag and kind in ("struct", "union"):
                    types.records.setdefault(tag, statement)
                elif re.match(r"typedef\b", statement):
                    types.typedefs.setdefault(key, statement)
        return types

    return effort.memo(("move.record_types", snapshot.digest), build)


def _lay_job(item: tuple[Snapshot, str]) -> Any:
    """One record's laid layout, or {"conflict": why}."""
    snapshot, tag = item
    types = _record_types(snapshot)
    statement = types.records[tag]
    start, end = _body(statement)
    laid = _probe_layouts(types, {f"struct {tag}": f"struct {tag} {statement[start : end + 1]};"})
    return laid.get(f"struct {tag}", {"conflict": "no layout"}) if isinstance(laid, dict) else {"conflict": laid}


def source_records(snapshot: Snapshot) -> set[str]:
    """The record tags and typedef names only sources declare (no header does): a source's own records."""
    types = _record_types(snapshot)
    records, typedefs = headers.type_names(snapshot)
    return {n for n in [*types.records, *types.typedefs] if n not in records and n not in typedefs}


def record_layouts(snapshot: Snapshot) -> tuple[dict[str, Any], dict[str, str]]:
    """(per record tag and typedef name its laid layout, per typedef name the tag it names) of every record the
    headers and sources declare, each record laid on the pool (all cores)."""
    types = _record_types(snapshot)
    aliases: dict[str, str] = {}
    for name in types.typedefs:
        found = re.fullmatch(r"typedef (?:struct|union) (\w+) (\w+) ?;", types.typedefs[name])
        named = types.tag(name) or (found[1] if found and found[2] == name else None)
        if named and named != name:
            aliases[name] = named
    tags = [
        tag
        for tag, statement in types.records.items()
        if re.match(r"(?:typedef\s+)?struct\b", statement) and "{" in statement
    ]
    with effort.stage("move.record_layouts"):
        laid = pool.map(snapshot.config, "move.record_layouts", _lay_job, [(snapshot, t) for t in tags])
    out: dict[str, Any] = dict(zip(tags, laid, strict=True))
    for name, tag in aliases.items():
        if tag in out and name not in out:
            out[name] = out[tag]
    return out, aliases
