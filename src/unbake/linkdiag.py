"""Where a link goes wrong: the first object whose bytes the linker starts off the ROM address its first name has."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from unbake import process, symbols, tree
from unbake.contracts import Config

_ENTRY = re.compile(r"^ (\.\S+)(?:[ \t]+|\n[ \t]+)0x([0-9a-f]+)[ \t]+0x([0-9a-f]+)[ \t]+(\S+\.o)$", re.M)
_KIND = {".rdata": ".rodata"}


def parse(map_text: str) -> list[tuple[str, str, int]]:
    """(object, section, address) of each nonempty input section of an ld map, in link order; .rdata is .rodata."""
    return [
        (m[4], _KIND.get(m[1], m[1]), int(m[2], 16))
        for m in _ENTRY.finditer(map_text)
        if int(m[3], 16) and m[1] != ".bss"
    ]


def walk(
    placed: Sequence[tuple[str, str, int]], expected: Mapping[tuple[str, str], int]
) -> tuple[str, int, int] | None:
    """The first (object, address its first name has, address the link gave it): an object that starts with bytes no
    name owns is off alone, so a shift is real when the next object with a name is off by the same amount."""
    seen, got = [], set()
    for obj, section, address in placed:
        if (obj, section) in expected and (obj, section) not in got:
            got.add((obj, section))
            seen.append((obj, expected[obj, section], address))
    for k, (obj, want, have) in enumerate(seen):
        later = seen[k + 1 :]
        if want != have and (not later or later[0][1] - later[0][2] == want - have):
            return obj, want, have
    return None


def first_off(config: Config, version: str) -> str:
    """One sentence naming the first object of `version`'s link that the linker places off the address of the first
    name it owns (an object before it too long or too short), or "" when every object is where its names say or when
    it cannot be told: a diagnostic never replaces the refusal it explains. The link runs again with a map."""
    try:
        found = _find(config, version)
    except Exception:
        return ""
    if found is None:
        return ""
    obj, want, have = found
    return (
        f"{version}: the first object off its ROM address is {obj}: its first name is at 0x{want:X} but the link "
        f"placed the object at 0x{have:X} ({have - want:+d} bytes): an object before it is {abs(have - want)} bytes "
        f"{'too long' if have > want else 'too short'}"
    )


def _find(config: Config, version: str) -> tuple[str, int, int] | None:
    root, held = config.project.root, tree.capture(config)
    table = symbols.load(config, version)
    expected: dict[tuple[str, str], int] = {}
    for module in held.modules:
        for name in module.owns.get(version, ()):
            if name in table:
                section = module.sections.get(version, {}).get(name) or ".text"
                key = (f"{module.path}.o", _KIND.get(section, section))
                expected[key] = min(table[name][0], expected.get(key, table[name][0]))
    scratch = process.scratch(root)
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / f"linkdiag-{version}.map"
    elf = scratch / f"linkdiag-{version}.elf"
    argv: list[str | Path] = [process.tool(config, "mips_ld"), "-T", "link.ld", f"-Map={out}", "-o", str(elf)]
    process.run("linkdiag", argv, root / "build" / version, tmp=scratch)
    text = out.read_text(errors="replace") if out.is_file() else ""
    out.unlink(missing_ok=True)
    elf.unlink(missing_ok=True)
    return walk(parse(text), expected)
