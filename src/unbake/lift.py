"""Lift the address words of an extracted data array into named symbols."""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Any

from unbake import effort, versions
from unbake.contracts import Member, Snapshot

_ARRAY = re.compile(
    r"^(?P<head>[ \t]*(?:const[ \t]+)?)(?:unsigned[ \t]+int|u32)(?P<mid>[ \t]+\w+[ \t]*\[[^\]]*\][ \t]*=[ \t]*\{)"
    r"(?P<body>[^{}]*)(?P<tail>\})",
    re.M,
)
_KSEG0 = 0x80000000  # the first MIPS virtual address a ROM symbol can have
_LITERAL = re.compile(r"\s*(0[xX][0-9A-Fa-f]{1,8}|\d+)[uU]?\s*")


def _index(snapshot: Snapshot, version: str) -> tuple[list[int], list[tuple[int, Member]]]:
    """The sorted vram starts of every placed member in a version, with (end, member) beside them, once per snapshot."""

    def build() -> tuple[list[int], list[tuple[int, Member]]]:
        rows = sorted(
            (
                (p.vram, p.vram + p.size, m.name, m)
                for m in snapshot.layout.members.values()
                for p in m.placements
                if p.version == version and p.size > 0
            ),
            key=lambda r: r[:3],
        )
        return [r[0] for r in rows], [(r[1], r[3]) for r in rows]

    return effort.memo(("lift-index", snapshot.digest, version), build)


def _target(index: tuple[list[int], list[tuple[int, Member]]], value: int) -> tuple[Member, int] | None:
    """The placed member that holds the address, and the offset inside it."""
    at = bisect_right(index[0], value) - 1
    if at >= 0 and value < index[1][at][0]:
        return index[1][at][1], value - index[0][at]
    return None


def _element(name: str, index: int) -> str:
    return name if index == 0 else f"&{name}[{index:#x}]"


def lift(snapshot: Snapshot, member: Member, text: str) -> tuple[str, dict[str, str]]:
    """The text with every word of its first u32 array that is a ROM address in each version holding the member
    spelled as the named symbol it reaches (the ROM word at that position names the same member and offset in every
    holder, so it is a pointer and not a coincidence), the array retyped to pointers, and the extern each name needs
    ({name: declaration}). Text with no such array or no such word comes back unchanged."""
    found = _ARRAY.search(text)
    if found is None:
        return text, {}
    holds = [p for p in member.placements if p.section != ".bss"]
    elements = found["body"].split(",")  # a u32 array: element i is the ROM word at 4*i
    if not holds or not any(_LITERAL.fullmatch(e) for e in elements):
        return text, {}
    reference = snapshot.config.project.names_from
    base = next((p for p in holds if p.version == reference), holds[0])
    declared = _declared(snapshot)
    vrams = _index(snapshot, base.version)
    lifted: dict[int, str] = {}
    declarations: dict[str, str] = {}
    for index, element in enumerate(elements):
        word = _LITERAL.fullmatch(element)
        value = int(word[1], 16 if word[1][:2] in ("0x", "0X") else 10) if word else 0
        hit = _target(vrams, value) if value >= _KSEG0 else None  # a smaller word is a count, mask or index
        if hit is None:
            continue
        target, offset = hit
        if target.name == member.name:  # a pointer into the array itself is an element address of the array
            if offset % 4 == 0 and all(_agrees(snapshot, p, target, offset, index) for p in holds):
                lifted[index] = _element(_symbol(member.name), offset // 4)
            continue
        if offset:
            continue  # only the exact start of another member has a name; the rest stay literals
        if _holds_all(target, holds) and all(_agrees(snapshot, p, target, offset, index) for p in holds):
            name = _symbol(target.name)
            lifted[index] = name
            if name not in declared:  # a header's own declaration stands; only an undeclared name gets an extern
                declarations[name] = f"extern void {name}();" if target.kind == "function" else f"extern char {name}[];"
    if not lifted:
        return text, {}
    out = []
    for index, element in enumerate(elements):
        word = _LITERAL.fullmatch(element)
        lead = element[: len(element) - len(element.lstrip())]
        lag = element[len(element.rstrip()) :]
        if index in lifted:
            out.append(f"{lead}{lifted[index]}{lag}")
        elif word:
            out.append(f"{lead}(void *){word[1]}{lag}")
        else:
            out.append(element)
    head = f"{found['head']}void *{found['mid'].lstrip()}"
    return text[: found.start()] + head + ",".join(out) + found["tail"] + text[found.end() :], declarations


def _declared(snapshot: Snapshot) -> set[str]:
    """The names the headers declare (their catalog): a comment or a struct field of that name declares nothing."""
    from unbake import headers as _headers

    return set(_headers.catalog(snapshot))


def _symbol(name: str) -> str:
    """The C name of a member: an unresolved row (`data/unresolved/80126768`) is spelled D_auto_80126768."""
    if versions.unowned(name):
        return f"D_auto_{name.rsplit('/', 1)[1]}"
    return re.sub(r"\W", "_", name)


def _holds_all(target: Member, holds: Any) -> bool:
    return {p.version for p in holds} <= {p.version for p in target.placements}


def _agrees(snapshot: Snapshot, place: Any, target: Member, offset: int, index: int) -> bool:
    """Whether the ROM word at the position of this holder equals the address of the same target and offset there."""
    at = place.rom_start + 4 * index
    if at + 4 > place.rom_end:
        return False
    there = [p for p in target.placements if p.version == place.version and p.size > 0]
    if len(there) != 1:
        return False
    word = versions.rom_bytes(snapshot.versions[place.version], at, at + 4)
    return len(word) == 4 and int.from_bytes(word, "big") == there[0].vram + offset
