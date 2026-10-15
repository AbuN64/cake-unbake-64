"""evidence: the addresses a version's own code gives the names an object leaves undefined."""

from __future__ import annotations

from types import SimpleNamespace

from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection

from unbake import evidence

CALL, HIGH, LOW = (evidence._TYPES[k] for k in ("R_MIPS_26", "R_MIPS_HI16", "R_MIPS_LO16"))


def word(*values: int) -> bytes:
    return b"".join(v.to_bytes(4, "big") for v in values)


class Section:
    def __init__(self, name, data=b"", relocs=(), symbols=()):
        self.name, self._data, self.relocs, self.symbols = name, data, relocs, symbols

    def __getitem__(self, key):
        return 1 if key == "sh_link" else None

    def data(self):
        return self._data

    def iter_relocations(self):
        return iter(self.relocs)

    def get_symbol(self, number):
        return self.symbols[number]


class RelSection(Section, RelocationSection):
    pass


class SymSection(Section, SymbolTableSection):
    pass


class Symbol(dict):
    def __init__(self, name, defined):
        super().__init__(st_shndx=1 if defined else "SHN_UNDEF")
        self.name = name


def fake_object(monkeypatch, text, relocs, symbols):
    sections = [Section(".text", text), SymSection(".symtab", symbols=symbols), RelSection(".rel.text", relocs=relocs)]
    monkeypatch.setattr(evidence, "ELFFile", lambda stream: SimpleNamespace(iter_sections=lambda: iter(sections)))


def test_a_call_and_a_high_low_pair_give_the_addresses_the_rom_forms(monkeypatch):
    # object: jal 0 ; lui $a0, 0 ; addiu $a0, $a0, 4 (the symbol plus 4)
    text = word(0x0C000000, 0x3C040000, 0x24840004)
    # ROM: jal 0x802BB550 ; lui $a0, 0x800E ; addiu $a0, $a0, 0x8A28 (0x800D8A28 = symbol + 4)
    rom = word(0x0C000000 | (0x2BB550 >> 2), 0x3C04800E, 0x24848A28)
    fake_object(
        monkeypatch,
        text,
        [
            {"r_offset": 0, "r_info_sym": 0, "r_info_type": CALL},
            {"r_offset": 4, "r_info_sym": 1, "r_info_type": HIGH},
            {"r_offset": 8, "r_info_sym": 1, "r_info_type": LOW},
        ],
        [Symbol("func", False), Symbol("data", False)],
    )
    assert evidence.read(b"", rom, 0x80200000) == {"func": ({0x802BB550}, True), "data": ({0x800D8A24}, False)}


def test_code_that_is_not_the_roms_outside_its_relocations_says_nothing(monkeypatch):
    text = word(0x0C000000, 0x03E00008)
    fake_object(monkeypatch, text, [{"r_offset": 0, "r_info_sym": 0, "r_info_type": CALL}], [Symbol("func", False)])
    assert evidence.read(b"", word(0x0C0000AB, 0x03E00009), 0x80200000) == {}  # jr $ra differs from the ROM's
    assert evidence.read(b"", word(0x0C0000AB), 0x80200000) == {}  # not the same length


def test_a_defined_symbol_is_not_a_reading(monkeypatch):
    fake_object(
        monkeypatch, word(0x0C000000), [{"r_offset": 0, "r_info_sym": 0, "r_info_type": CALL}], [Symbol("local", True)]
    )
    assert evidence.read(b"", word(0x0C0000AB), 0x80200000) == {}
