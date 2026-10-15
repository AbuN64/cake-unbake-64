"""The fold plan of every invented per-function record (rule fake-type): one `unbake move -` row per record, by class.

a  its laid members fit exactly one real (non-invented) record at the same offsets, or a typedef names it as one:
   fold into that record.
b  several invented records share one layout: fold them into one shared record named from the declarators that
   reach them (dispatch-group rule).
c  one data symbol is all that reaches it: rename it to the symbol's owner-local record (Data<addr>[Ptr] or
   Data<PascalWords>[Ptr], the recipe's scheme), or fold into that record when it exists.
e  flow: the pointer flows (call argument to parameter, assignment, member load, return value) between it and
   other records. A flow component holding exactly one real (or Data*) record folds every invented record into it
   when each fits it; a component with none gets ONE shared record named from evidence (the most common declarator
   word reaching it, else the role prefix with the module of its lowest function). A member conflict or several real
   records refuse the whole component. Records a source defines (fuzzy local records) take part as well: they fold
   into a real header record their flow or layout gives them.
d  none of these: no row, the reason is reported.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from unbake.contracts import Json
from unbake.families import typedebt

_PLACE = re.compile(r"^(?:pad\w*|unk_?\w*|field_?\w*|_?[0-9A-Fa-f]+|reserved\w*|filler\w*)$", re.I)
_NEUTRAL = re.compile(r"D_(?:auto_)?([0-9A-Fa-f]+)")


def invented(name: str) -> bool:
    """Whether a type name is a per-function invention the fake-type rule flags."""
    return typedebt.fake_name(name)


def data_name(symbol: str, pointee: bool) -> str:
    """A data symbol's owner-local record name: `D_[auto_]<hex>` -> Data<HEX>, a named symbol by its words in
    PascalCase (`g_menuState` -> DataMenuState), +Ptr for what a pointer symbol points at."""
    neutral = _NEUTRAL.fullmatch(symbol)
    if neutral:
        stem = f"{int(neutral[1], 16):08X}"
    else:
        words = [w for w in re.split(r"_+", re.sub(r"^(?:g|s|D)_", "", symbol)) if w]
        stem = "".join(w[:1].upper() + w[1:] for w in words if w.isalnum())
    return f"Data{stem}" + ("Ptr" if pointee else "") if stem else ""


def _pascal(word: str) -> str:
    parts = [p for p in re.split(r"_+|(?<=[a-z0-9])(?=[A-Z])", word) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


def reaches(texts: Mapping[str, str], names: Iterable[str]) -> dict[str, dict[str, Counter[str]]]:
    """Per type name, per kind (data: a file-scope object, member: a record member, local: a local or parameter)
    the declarator names that hold it or point at it, with a `*` prefix for pointers."""
    wanted = set(names)
    found: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    if not wanted:
        return found
    alternation = "|".join(sorted(map(re.escape, wanted), key=len, reverse=True))
    pattern = re.compile(
        rf"^([ \t]*)(?:(extern|static)\s+)?(?:const\s+|volatile\s+)*(?:struct\s+|union\s+)?\b({alternation})\b"
        rf"\s*((?:\*\s*(?:const\s*)?)*)(\w+)\s*(\[[^\]]*\])*\s*(=|;)",
        re.M,
    )
    params = re.compile(rf"[(,]\s*(?:const\s+)?(?:struct\s+|union\s+)?\b({alternation})\b\s*(\**)\s*(\w+)\s*(?=[,)])")
    for path, text in texts.items():
        for m in pattern.finditer(text):
            indent, storage, name, stars, declarator = m[1], m[2], m[3], m[4], m[5]
            if re.match(r"\s*typedef\b", text[text.rfind("\n", 0, m.start()) + 1 : m.start()]):
                continue
            kind = "data" if not indent or storage == "extern" else "member" if path.endswith(".h") else "local"
            found[name][kind]["*" * bool(stars.strip()) + declarator] += 1
        for m in params.finditer(text):
            found[m[1]]["local"]["*" * bool(m[2]) + m[3]] += 1
    return found


_KEYWORDS = frozenset({"return", "else", "case", "goto", "do", "if", "while", "for", "switch", "sizeof", "typedef"})
_TYPED = r"(?:const\s+|volatile\s+|register\s+|static\s+)*(?:struct\s+|union\s+)?([A-Za-z_]\w*)\s*\*\s*(?:const\s+)?"
_HEAD = re.compile(
    r"^[ \t]*(?:static\s+|inline\s+|extern\s+)*(?:const\s+)?(?:struct\s+|union\s+)?(\w+)\s*(\**)\s*\b(\w+)\s*"
    r"\(([^()]*)\)\s*([{;])",
    re.M,
)
_LOCAL = re.compile(rf"(?:^|[;{{}}])\s*{_TYPED}([A-Za-z_]\w*)\s*(?=[;=,\[])")
_ASSIGN = re.compile(r"(?<![=!<>+\-*/&|^%.>])\b([A-Za-z_]\w*)\s*=(?!=)\s*([^;]+);")


def _param_types(text: str) -> list[str | None]:
    """The record word of each parameter declared as one pointer to it, else None."""
    out: list[str | None] = []
    for param in text.split(","):
        m = re.fullmatch(rf"\s*{_TYPED}(\w+)?\s*", param)
        out.append(m[1] if m and m[1] not in ("void", "char") else None)
    return out


def facts(code: str) -> Json:
    """The flow facts of one comment-free source or header text: per function defined or declared its parameter and
    return record words, and per definition the (callee, argument index, record word) of each pointer variable passed
    and the (record word, kind, what) of each assignment to one: from another variable ("var", word), a member load
    ("member", base word, member) or a call ("call", callee)."""
    from unbake import retype  # the call-site parser is retype's

    functions: dict[str, dict[str, Any]] = {}
    for head in _HEAD.finditer(code):
        ret, stars, name, params, end = head[1], head[2], head[3], head[4], head[5]
        if name in _KEYWORDS or ret in _KEYWORDS:
            continue
        row = functions.setdefault(name, {"params": _param_types(params), "returns": None, "calls": [], "assigns": []})
        if len(stars) == 1 and ret not in ("void", "char"):
            row["returns"] = ret
        if end != "{" or row["calls"] or row["assigns"]:
            continue
        depth, close = 0, len(code)
        for index in range(head.end() - 1, len(code)):
            depth += (code[index] == "{") - (code[index] == "}")
            if depth == 0:
                close = index
                break
        body = code[head.end() - 1 : close]
        declared = [re.search(r"(\w+)\s*$", p) for p in params.split(",")]
        names: dict[str, str] = {m[1]: w for m, w in zip(declared, row["params"], strict=True) if m and w}
        names.update({m[2]: m[1] for m in _LOCAL.finditer(body) if m[1] not in _KEYWORDS})
        if not names:
            continue
        for call in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", body):
            if call[1] in _KEYWORDS:
                continue
            last = retype._balanced(body, call.end() - 1)
            if last < 0:
                continue
            for index, (_, arg) in enumerate(retype._split(body[call.end() : last])):
                if arg.strip() in names:
                    row["calls"].append([call[1], index, names[arg.strip()]])
        assigns = [(m[1], m[2]) for m in _ASSIGN.finditer(body)]
        assigns += [
            (m[2], init[1]) for m in _LOCAL.finditer(body) if (init := re.match(r"\s*=\s*([^;]+);", body[m.end() :]))
        ]
        for target, value in assigns:
            word, rhs = names.get(target), value.strip()
            if not word:
                continue
            if rhs in names:
                row["assigns"].append([word, "var", names[rhs]])
            elif (load := re.fullmatch(r"(\w+)\s*->\s*(\w+)", rhs)) and load[1] in names:
                row["assigns"].append([word, "member", names[load[1]], load[2]])
            elif called := re.fullmatch(r"(\w+)\s*\(.*\)", rhs, re.S):
                row["assigns"].append([word, "call", called[1]])
    return functions


def flow(
    found: Iterable[Json], layouts: Mapping[str, Mapping[str, Any]], aliases: Mapping[str, str]
) -> list[tuple[str, str]]:
    """The flow edges between record names (each as the record it names) that the facts of every file give, only
    where at least one side is invented or defined in a source: two real records never join by flow."""

    def own(name: str) -> str:
        return aliases.get(name, name)

    params: dict[str, list[str | None]] = {}
    returns: dict[str, str] = {}
    for functions in found:
        for name, row in functions.items():
            if row["calls"] or row["assigns"] or name not in params:  # a definition outweighs a prototype
                params[name] = row["params"]
            if row["returns"]:
                returns.setdefault(name, row["returns"])
    edges: set[tuple[str, str]] = set()

    def join(first: str | None, second: str | None) -> None:
        if not (first and second):
            return
        a, b = own(first), own(second)
        if a != b and a in layouts and b in layouts:
            edges.add((min(a, b), max(a, b)))

    for functions in found:
        for row in functions.values():
            for callee, index, word in row["calls"]:
                taken = params.get(callee) or []
                join(word, taken[index] if index < len(taken) else None)
            for word, kind, *what in row["assigns"]:
                if kind == "var":
                    join(word, what[0])
                elif kind == "call":
                    join(word, returns.get(what[0]))
                else:
                    member = next(
                        (f for f in layouts.get(own(what[0]), {}).get("fields", ()) if f["name"] == what[1]), None
                    )
                    pointee = re.fullmatch(r"(?:struct\s+|union\s+)?(\w+)\s*\*", member["type"]) if member else None
                    join(word, pointee[1] if pointee else None)
    return sorted(edges)


def _members(lays: Iterable[Mapping[str, Any]]) -> tuple[list[tuple[int, int, str, str]], str]:
    """The union of the named members of laid records, and the conflict that refuses it ("" when none): two members
    at one offset must agree in width and type (a `void *` takes any pointer), and members never overlap."""
    rows: dict[int, tuple[int, str, str]] = {}
    for lay in lays:
        for offset, size, kind, name in _fields(lay):
            if _pad(name, kind):
                continue
            seen = rows.get(offset)
            if seen is None:
                rows[offset] = (size, kind, name)
            elif seen[0] != size or (
                seen[1] != kind and not ("*" in kind and "*" in seen[1] and "void *" in (kind, seen[1]))
            ):
                return [], f"offset 0x{offset:X}: {seen[1]} {seen[2]} and {kind} {name}"
            elif seen[1] == "void *" and kind != seen[1]:
                rows[offset] = (size, kind, seen[2])
    ordered = sorted(rows.items())
    for (a, (sa, ka, na)), (b, (_, kb, nb)) in itertools.pairwise(ordered):
        if a + sa > b:
            return [], f"offset 0x{a:X} {ka} {na} overlaps 0x{b:X} {kb} {nb}"
    return [(o, s, k, n) for o, (s, k, n) in ordered], ""


def _fields(layout: Mapping[str, Any]) -> list[tuple[int, int, str, str]]:
    return [(f["offset"], f["width"], f["type"], f["name"]) for f in layout.get("fields", ())]


def _shape(layout: Mapping[str, Any]) -> tuple[Any, ...]:
    """The layout with member names left out: offsets, widths and types of the members a reader can name."""
    return (layout.get("size"), tuple((o, s, t) for o, s, t, n in _fields(layout) if not _pad(n, t)))


def _pad(name: str, kind: str) -> bool:
    return bool(_PLACE.match(name)) and "[" in kind


def fits(mine: Mapping[str, Any], theirs: Mapping[str, Any]) -> int:
    """How many of `mine`'s non-pad members `theirs` holds at the same offset, width and type (a `void *` takes any
    pointer), -1 when one of them overlaps a member of another shape or runs past `theirs`' end."""
    if mine.get("size", 0) > theirs.get("size", 0):
        return -1
    rows = _fields(theirs)
    hits = 0
    for offset, size, kind, name in _fields(mine):
        if _pad(name, kind):
            continue
        same = [r for r in rows if r[0] == offset and r[1] == size]
        if any(r[2] == kind or ("*" in kind and "*" in r[2] and "void *" in (kind, r[2])) for r in same):
            hits += 1
            continue
        if any(r[0] < offset + size and offset < r[0] + r[1] and not _pad(r[3], r[2]) for r in rows):
            return -1
    return hits


def _chosen(tops: list[str], role: str, size: Any, real: Mapping[str, Mapping[str, Any]]) -> str:
    """The one real record of equally fitting `tops`: the only one, else the one the invented name's role prefix
    names (`ALSynth_func_..` -> ALSynth) or starts (Menu -> MenuRecAt24), else the one of the same size; else none."""
    if len(tops) == 1:
        return tops[0]
    prefixed = [t for t in tops if role and t.startswith(role)]
    for pool in ([t for t in tops if t == role], prefixed):
        if len(pool) == 1:
            return pool[0]
    for pool in (prefixed, tops):
        exact = [t for t in pool if real[t].get("size") == size]
        if len(exact) == 1:
            return exact[0]
    return ""


def plan(
    layouts: Mapping[str, Mapping[str, Any]],
    aliases: Mapping[str, str],
    reached: Mapping[str, Mapping[str, Counter[str]]],
    taken: Iterable[str],
    weight: Mapping[str, int] | None = None,
    edges: Iterable[tuple[str, str]] = (),
    local: Iterable[str] = (),
    modules: Mapping[str, str] | None = None,
) -> tuple[list[Json], list[Json]]:
    """(move rows, census rows). `layouts` holds every record by name (laid, or {"conflict": why}); `aliases` maps a
    typedef name to the record it names; `reached` is `reaches()`; `taken` every name in use; `weight` the bytes each
    invented name withholds; `edges` the flow edges (`flow`); `local` the records only sources define (they fold
    into header records, by layout or flow); `modules` the module of each function symbol. Rows fold records first,
    then the typedef aliases of each into its record's new name."""
    weight = weight or {}
    taken = set(taken)
    local = set(local) - {n for n in layouts if invented(n)}
    fakes = sorted(n for n in layouts if invented(n) or n in local)
    real = {n: lay for n, lay in layouts.items() if not invented(n) and n not in local and "fields" in lay}
    target: dict[str, str] = {}
    census: list[Json] = []
    why: dict[str, tuple[str, str]] = {}

    def own(name: str) -> str:
        return aliases.get(name, name)

    shapes: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    joined: list[tuple[str, str]] = []
    for name in fakes:
        lay = layouts[name]
        alias = own(name)
        if alias != name and invented(alias):
            continue  # an alias of an invented record follows that record
        if alias != name and not invented(alias) and alias in real:
            target[name] = alias
            why[name] = ("a", f"a typedef names it as {alias}")
            continue
        if "fields" not in lay:
            why[name] = ("d", f"cannot be laid: {lay.get('conflict', 'no layout')}")
            continue
        scored = [(fits(lay, other), name2) for name2, other in real.items()]
        best = max((s for s, _ in scored), default=-1)
        named = sum(1 for o, s, k, n in _fields(lay) if not _pad(n, k))
        tops = sorted(n2 for s, n2 in scored if s == best and best > 0)
        pick = _chosen(tops, re.split(r"_?func_", name)[0], lay.get("size"), real)
        if best > 0 and best == named and pick:
            target[name] = pick
            why[name] = ("a", f"its {named} members fit {pick}")
            continue
        if name in local:  # a source's own record moves only into a header record (flow below)
            why[name] = ("d", "a source record no header record fits")
            continue
        holders = {d.lstrip("*"): d.startswith("*") for d in reached.get(name, {}).get("data", Counter())}
        members = reached.get(name, {}).get("member", Counter())
        if len(holders) == 1 and not members:
            symbol, pointer = next(iter(holders.items()))
            new = data_name(symbol, pointer)
            if new and not invented(new):
                target[name] = new
                why[name] = ("c", f"only {symbol} reaches it")
                continue
        shapes[_shape(lay)].append(name)
        reason = (
            f"{len(tops)} real records fit equally"
            if best > 0 and best == named
            else f"reached by {len(holders)} data symbols and {sum(members.values())} members; no real record fits"
        )
        why[name] = ("d", reason)
    for group in shapes.values():
        if len(group) < 2:
            continue
        words: Counter[str] = Counter()
        prefix = Counter(re.split(r"_?func_", n)[0] for n in group)
        for n in group:
            for kind in ("member", "data"):
                for d, count in reached.get(n, {}).get(kind, Counter()).items():
                    d = d.lstrip("*")
                    if not _PLACE.match(d) and not re.search(r"^(?:D_|temp|var|sp|arg|func_)", d) and len(d) > 2:
                        words[d] += count
        role = next((p for p, _ in prefix.most_common() if p and not invented(p) and p != "Shared"), "")
        names = [_pascal(role) + _pascal(w) for w, _ in words.most_common(3)]
        names += [role] if len(prefix) == 1 else []  # every record of the group plays this one role
        pick = next((n for n in names if n and n not in taken and not invented(n)), "")
        if not pick:
            for n in group:
                why[n] = ("d", f"shares its layout with {len(group) - 1} more, but no declarator names the group")
            joined += list(itertools.pairwise(group))  # one flow component: named there by role and module
            continue
        taken.add(pick)
        for n in group:
            target[n] = pick
            via = f"reached as {words.most_common(1)[0][0]}" if pick != role else f"all in role {role}"
            why[n] = ("b", f"one of {len(group)} same-layout records {via}")
    edges = [*edges, *joined]
    seeds, components = _flow_classes(layouts, real, local, edges, target, why, own, reached, taken, modules or {})
    for name in fakes:
        alias = own(name)
        if alias != name and (invented(alias) or alias in local):
            why[name] = why.get(alias, ("d", f"an alias of {alias}, which is not a record"))
            if alias in target:
                target[name] = target[alias]
            if alias in components:
                components[name] = components[alias]
    rows: list[Json] = [
        {"op": "type", "subject": name, "target": target[name]}
        for name in sorted(target, key=lambda n: (own(n) != n, n not in seeds, n))
    ]
    for name in fakes:
        cls, reason = why[name]
        census.append(
            {
                "record": name,
                "class": cls,
                "target": target.get(name),
                "bytes": weight.get(name, 0),
                "reason": reason,
                "component": components.get(name),
                "local": name in local,
            }
        )
    return rows, census


def _flow_classes(
    layouts: Mapping[str, Mapping[str, Any]],
    real: Mapping[str, Mapping[str, Any]],
    local: set[str],
    edges: Iterable[tuple[str, str]],
    target: dict[str, str],
    why: dict[str, tuple[str, str]],
    own: Callable[[str], str],
    reached: Mapping[str, Mapping[str, Counter[str]]],
    taken: set[str],
    modules: Mapping[str, str],
) -> tuple[set[str], dict[str, int]]:
    """Class e: union-find over the flow edges, each name standing as the record it already folds into. Moves the d
    records of each component (in `target` and `why`); returns the seeds a new shared record is renamed from and the
    component id of every name in one."""
    parent: dict[str, str] = {}

    def node(name: str) -> str:
        return target.get(own(name), own(name))

    def root(name: str) -> str:
        parent.setdefault(name, name)
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    near: dict[str, set[str]] = defaultdict(set)  # each record's direct flow neighbours
    for a, b in edges:
        a, b = node(a), node(b)
        if (a in real) and (b in real):
            continue  # two real records never join
        near[a].add(b)
        near[b].add(a)
        parent[root(a)] = root(b)
    for name, (cls, reason) in why.items():  # an invented record no flow reaches is a component of its own
        if cls == "d" and name not in local and reason.endswith("no real record fits") and own(name) == name:
            root(name)
    groups: dict[str, list[str]] = defaultdict(list)
    for name in list(parent):
        groups[root(name)].append(name)
    seeds: set[str] = set()
    components: dict[str, int] = {}
    for number, names in enumerate(sorted(sorted(g) for g in groups.values())):
        open_ = [n for n in names if why.get(n, ("", ""))[0] == "d" and "fields" in layouts.get(n, {})]
        open_ = [n for n in open_ if own(n) == n]
        if not open_:
            continue
        for n in names:
            components[n] = number
        reals = sorted({n for n in names if n in real})
        lays = [layouts[n] for n in open_]
        if len(reals) > 1:  # each record takes the one real record its own pointer flows into and fits whole
            for n in open_:
                named = sum(1 for f in _fields(layouts[n]) if not _pad(f[3], f[2]))
                fit = [r for r in sorted(near[n] & set(reals)) if fits(layouts[n], real[r]) == named]
                if len(fit) == 1:
                    target[n] = fit[0]
                    why[n] = ("e", f"its pointer flows into {fit[0]}, which it fits")
                else:
                    why[n] = ("d", f"its flow joins {len(reals)} real records: {', '.join(reals[:3])}")
            continue
        if reals:
            home = reals[0]
            _, conflict = _members([*lays, real[home]])
            bad = next((n for n in open_ if fits(layouts[n], real[home]) < 0), "")
            if conflict or bad:
                reason = conflict or f"{bad} does not fit {home}"
                for n in open_:
                    why[n] = ("d", f"flow component into {home} refused: {reason}")
                continue
            for n in open_:
                target[n] = home
                why[n] = ("e", f"its pointer flows into {home} ({len(names)} records)")
            continue
        fakes = [n for n in open_ if n not in local]
        if not fakes:
            for n in open_:
                why[n] = ("d", "a flow component of source records with no header record")
            continue
        _, conflict = _members(lays)
        if conflict:
            for n in open_:
                why[n] = ("d", f"flow component member conflict at {conflict}")
            continue
        pick = _flow_name(open_, reached, taken, modules)
        if not pick:
            for n in open_:
                why[n] = ("d", f"{why[n][1]}; no evidence names its flow component")
            continue
        taken.add(pick)
        seed = max(fakes, key=lambda n: (sum(1 for f in _fields(layouts[n]) if not _pad(f[3], f[2])), n))
        seeds.add(seed)
        for n in open_:
            target[n] = pick
            why[n] = ("e", f"one shared record of a flow component of {len(open_)} records")
        if len(open_) == 1:
            why[open_[0]] = ("e", "its own flow component: an evidence-named record")
    return seeds, components


def _flow_name(
    names: Sequence[str], reached: Mapping[str, Mapping[str, Counter[str]]], taken: set[str], modules: Mapping[str, str]
) -> str:
    """The evidence name of a flow component: its most common declarator word, else the role prefix of its invented
    names with the module of its lowest function; never a function symbol, a Data* name or a name in use."""
    words: Counter[str] = Counter()
    for n in names:
        for kind in ("member", "data", "local"):
            for d, count in reached.get(n, {}).get(kind, Counter()).items():
                d = d.lstrip("*")
                if (
                    not _PLACE.match(d)
                    and not re.search(r"^(?:D_|temp|var|sp|arg|func_|ptr|phi|unused)", d)
                    and len(d) > 2
                ):
                    words[d] += count
    lowest = sorted((m[0], n) for n in names if invented(n) and (m := re.search(r"func_[0-9A-Fa-f]{8}", n)))
    roles = Counter(re.split(r"_?func_", n)[0] for n in names if invented(n))
    first = [re.split(r"_?func_", n)[0] for _, n in lowest]  # a tie goes to the role of the lowest function
    ranked = sorted(roles, key=lambda r: (-roles[r], first.index(r) if r in first else len(first)))
    role = next((r for r in ranked if r and not invented(r) and r != "Shared"), "")
    symbols = sorted(m[0] for n in names if (m := re.search(r"func_[0-9A-Fa-f]{8}", n)))
    module = modules.get(next(iter(symbols), ""), "")
    picks = [_pascal(w) for w, _ in words.most_common(3)]
    stem = _pascal(role) or "Rec"  # a numbered view has no role: the module names it
    picks += [stem + _pascal(module)] if module else []
    if module and lowest:  # several components of one role in one module: the lowest function's address
        picks.append(stem + _pascal(module) + lowest[0][0].removeprefix("func_").upper())
    return str(
        next((p for p in picks if p and p not in taken and not invented(p) and not p.startswith(("Data", "Func"))), "")
    )
