"""The fold plan of every invented per-function record (rule fake-type): one `unbake move -` row per record, by class.

a  its laid members fit exactly one real (non-invented) record at the same offsets, or a typedef names it as one:
   fold into that record.
b  several invented records share one layout: fold them into one shared record named from the declarators that
   reach them (dispatch-group rule).
c  one data symbol is all that reaches it: rename it to the symbol's owner-local record (Data<addr>[Ptr] or
   Data<PascalWords>[Ptr], the recipe's scheme), or fold into that record when it exists.
d  none of these: no row, the reason is reported.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
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
) -> tuple[list[Json], list[Json]]:
    """(move rows, census rows). `layouts` holds every record by name (laid, or {"conflict": why}); `aliases` maps a
    typedef name to the record it names; `reached` is `reaches()`; `taken` every name in use; `weight` the bytes each
    invented name withholds. Rows fold records first, then the typedef aliases of each into its record's new name."""
    weight = weight or {}
    taken = set(taken)
    fakes = sorted(n for n in layouts if invented(n))
    real = {n: lay for n, lay in layouts.items() if not invented(n) and "fields" in lay}
    target: dict[str, str] = {}
    census: list[Json] = []
    why: dict[str, tuple[str, str]] = {}

    def own(name: str) -> str:
        return aliases.get(name, name)

    shapes: dict[tuple[Any, ...], list[str]] = defaultdict(list)
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
            continue
        taken.add(pick)
        for n in group:
            target[n] = pick
            via = f"reached as {words.most_common(1)[0][0]}" if pick != role else f"all in role {role}"
            why[n] = ("b", f"one of {len(group)} same-layout records {via}")
    for name in fakes:
        alias = own(name)
        if alias != name and invented(alias):
            why[name] = why.get(alias, ("d", f"an alias of {alias}, which is not a record"))
            if alias in target:
                target[name] = target[alias]
    rows: list[Json] = [
        {"op": "type", "subject": name, "target": target[name]}
        for name in sorted(target, key=lambda n: (own(n) != n, n))
    ]
    for name in fakes:
        cls, reason = why[name]
        census.append(
            {"record": name, "class": cls, "target": target.get(name), "bytes": weight.get(name, 0), "reason": reason}
        )
    return rows, census
