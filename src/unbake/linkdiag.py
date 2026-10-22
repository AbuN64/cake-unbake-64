"""Where a link goes wrong: the first object of a segment whose bytes start off the ROM address its first name has."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path

from elftools.elf.elffile import ELFFile

from unbake import symbols, tree
from unbake.contracts import Config

_SEGMENT = re.compile(r"^\s*(\.\w+)\s+(0x[0-9a-fA-F]+)\s*:\s*AT\((0x[0-9a-fA-F]+)\)", re.M)
_ROW = re.compile(r"^\s*(\S+)\.o\((\.\w+)\);\s*$", re.M)


def walk(
    link: str, size: Callable[[str, str], int | None], expected: Mapping[tuple[str, str], int]
) -> tuple[str, int, int] | None:
    """The first (object, expected address, address the link gives it) of a link script text: objects follow each
    other without padding (SUBALIGN(1)) from their segment's start, `size(object, section)` is the bytes an object
    holds there (None: not built) and `expected` the address of the first name each (object, section) owns."""
    segments = list(_SEGMENT.finditer(link))
    for n, found in enumerate(segments):
        cursor = int(found[2], 16)
        end = segments[n + 1].start() if n + 1 < len(segments) else len(link)
        for row in _ROW.finditer(link, found.end(), end):
            wanted = expected.get((row[1], row[2]))
            if wanted is not None and wanted != cursor and wanted >= int(found[2], 16):
                return row[1], wanted, cursor
            cursor += size(row[1], row[2]) or 0
    return None


def _sizes(root: Path, version: str) -> Callable[[str, str], int | None]:
    def size(obj: str, section: str) -> int | None:
        try:
            with (root / "build" / version / f"{obj}.o").open("rb") as handle:
                found = ELFFile(handle).get_section_by_name(section)
                return int(found["sh_size"]) if found is not None else 0
        except OSError:
            return None

    return size


def first_off(config: Config, version: str) -> str:
    """One sentence naming the first object of `version`'s link script that sits off the address of the first name it
    owns (an object of the wrong size, a pad, a missing object before it), or "" when every object is where its names
    say, or when it cannot be told: a diagnostic never replaces the refusal it explains."""
    try:
        found = _find(config, version)
    except Exception:
        return ""
    if found is None:
        return ""
    obj, want, got = found
    return (
        f"{version}: the first object off its ROM address is {obj}: its first name sits at 0x{want:X} but the link "
        f"places the object at 0x{got:X} ({got - want:+d} bytes): an object before it is {abs(got - want)} bytes "
        f"{'too long' if got > want else 'too short'}"
    )


def _find(config: Config, version: str) -> tuple[str, int, int] | None:
    root, held = config.project.root, tree.capture(config)
    table = symbols.load(config, version)
    expected: dict[tuple[str, str], int] = {}
    for module in held.modules:
        for name in module.owns.get(version, ()):
            if name in table:
                key = (module.path, module.sections.get(version, {}).get(name) or ".text")
                expected[key] = min(table[name][0], expected.get(key, table[name][0]))
    link = (root / "build" / version / "link.ld").read_text()
    return walk(link, _sizes(root, version), expected)
