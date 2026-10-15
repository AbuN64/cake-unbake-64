"""The addresses a version's own code gives the names an object leaves undefined."""

from __future__ import annotations

import io
from collections import defaultdict

from elftools.elf.elffile import ELFFile
from elftools.elf.enums import ENUM_RELOC_TYPE_MIPS as _TYPES
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection

_WORD = 0xFFFFFFFF


def _signed(imm: int) -> int:
    return (imm & 0xFFFF) - ((imm & 0x8000) << 1)


def read(blob: bytes, rom: bytes, vram: int) -> dict[str, tuple[set[int], bool]]:
    """Per undefined name the ROM addresses formed at the object's relocation sites, and whether one is a call."""
    elf = ELFFile(io.BytesIO(blob))
    sections = list(elf.iter_sections())
    text = next(s.data() for s in sections if s.name == ".text")

    def word(data: bytes, at: int) -> int:
        return int.from_bytes(data[at : at + 4], "big")

    sites: dict[str, list[tuple[int, int]]] = defaultdict(list)
    masks = [0xFFFFFFFF] * (len(text) // 4)
    for section in sections:
        if section.name != ".rel.text" or not isinstance(section, RelocationSection):
            continue
        symbols = sections[section["sh_link"]]
        assert isinstance(symbols, SymbolTableSection)
        for r in section.iter_relocations():
            symbol = symbols.get_symbol(r["r_info_sym"])
            masks[r["r_offset"] // 4] = 0xFC000000 if r["r_info_type"] == _TYPES["R_MIPS_26"] else 0xFFFF0000
            if symbol["st_shndx"] == "SHN_UNDEF" and symbol.name:
                sites[symbol.name].append((r["r_offset"], r["r_info_type"]))
    if len(rom) != len(text) or any(word(rom, 4 * i) & m != word(text, 4 * i) & m for i, m in enumerate(masks)):
        return {}  # the code is not the ROM's outside its relocations: what it forms there says nothing
    found: dict[str, tuple[set[int], bool]] = {}
    for name, rows in sites.items():
        addresses, call = set(), False
        lows = sorted(at for at, kind in rows if kind == _TYPES["R_MIPS_LO16"])
        for at, kind in sorted(rows):
            if kind == _TYPES["R_MIPS_26"]:
                target = ((vram + at + 4) & 0xF0000000) | ((word(rom, at) & 0x3FFFFFF) << 2)
                addresses.add((target - ((word(text, at) & 0x3FFFFFF) << 2)) & _WORD)
                call = True
            elif kind == _TYPES["R_MIPS_HI16"] and (low := next((o for o in lows if o > at), None)) is not None:
                had = ((word(text, at) & 0xFFFF) << 16) + _signed(word(text, low))
                addresses.add((((word(rom, at) & 0xFFFF) << 16) + _signed(word(rom, low)) - had) & _WORD)
        if addresses:
            found[name] = addresses, call
    return found
