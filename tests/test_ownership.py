"""ownership: a C unit owns the rodata and data its object emits, located in the ROM; each mismatch class has a case."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection

from unbake import ownership
from unbake.contracts import Claim, Finding, LayoutMap, Member, Placement, Refusal, Snapshot, UnitSpec

TEXT_VRAM, TEXT_ROM = 0x80207738, 0x1000


def word(*values: int) -> bytes:
    return b"".join(v.to_bytes(4, "big") for v in values)


def index(rom: bytes, rows: list[tuple[int, int, str, str, int]]) -> ownership._Index:
    """The tuple ownership._index builds: rom, rows by ROM start, start -> row, first word -> starts, digest."""
    first = defaultdict(list)
    for start, *_ in rows:
        first[int.from_bytes(rom[start : start + 4], "big")].append(start)
    by_vram = sorted((r[4], r[4] + r[1] - r[0], r[0], r[1]) for r in rows)
    return ownership._Index(rom, rows, {r[0]: i for i, r in enumerate(rows)}, first, "rows", by_vram, ())


def emits(monkeypatch, **sections):
    monkeypatch.setattr(ownership, "emitted", lambda blob: sections)


# ------------------------------------------------------------------ what the object emits


class Section:
    def __init__(self, name, data=b"", kind="SHT_PROGBITS", align=1, link=0, info=0, relocs=(), symbols=()):
        self.name, self._data, self.relocs, self.symbols = name, data, relocs, symbols
        self.fields = {"sh_size": len(data), "sh_type": kind, "sh_link": link, "sh_info": info, "sh_addralign": align}

    def __getitem__(self, key):
        return self.fields[key]

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


def test_emitted_concatenates_in_file_order_with_alignment_relocations_and_low_references(monkeypatch):
    lo16 = ownership.ENUM_RELOC_TYPE_MIPS["R_MIPS_LO16"]
    code = word(0x3C040000, 0x24840008, 0, 0)  # lui, addiu $a0, $a0, 8 (the low half references .rdata + 8)
    sections = [
        Section(".text", code),
        Section(".rodata", word(1, 2), align=4),
        Section(".rdata", word(0x34), align=8),
        Section(".sdata", word(7)),
        RelSection(
            ".rel.text", kind="SHT_REL", link=6, info=0, relocs=[{"r_offset": 4, "r_info_sym": 0, "r_info_type": lo16}]
        ),
        RelSection(
            ".rel.rodata", kind="SHT_REL", link=6, info=1, relocs=[{"r_offset": 0, "r_info_sym": 1, "r_info_type": 2}]
        ),
        SymSection(
            ".symtab", kind="SHT_SYMTAB", symbols=[{"st_shndx": 2, "st_value": 0}, {"st_shndx": 0, "st_value": 0}]
        ),
    ]
    monkeypatch.setattr(ownership, "ELFFile", lambda stream: SimpleNamespace(iter_sections=lambda: iter(sections)))
    got = ownership.emitted(b"object")
    assert got[".rodata"] == (word(1, 2, 0x34), {0: True}, [(4, 8 + 8)])  # .rdata starts at 8; the addend is 8
    assert got[".data"] == (word(7), {}, [])
    assert set(got) == {".rodata", ".data"}


# ------------------------------------------------------------------ where the bytes live


def test_locate_a_unique_range(monkeypatch):
    rom = bytes(0x10) + word(0x3F800000, 0x40000000) + word(9)
    rows = [
        (0x10, 0x14, "a", ".rodata", 0x800C0000),
        (0x14, 0x18, "b", ".rodata", 0x800C0004),
        (0x18, 0x1C, "c", ".rodata", 0x800C0008),
    ]
    emits(monkeypatch, **{".rodata": (word(0x3F800000, 0x40000000), {}, [])})
    assert ownership.locate(index(rom, rows), b"", None) == {".rodata": [[0x10, 0x18]]}


def test_locate_finds_bytes_that_start_inside_a_row_and_explains_what_is_nowhere(monkeypatch):
    rom = word(5, 6, 7)
    emits(monkeypatch, **{".rodata": (word(6, 7), {}, [])})
    assert ownership.locate(index(rom, [(0, 12, "wide", ".rodata", 0x80000000)]), b"", None) == {".rodata": [[4, 12]]}
    emits(monkeypatch, **{".rodata": (word(9, 9), {}, [])})
    assert ownership.locate(index(rom, [(0, 12, "w", ".rodata", 0)]), b"", None) == {
        ".rodata": {"why": "bytes found nowhere in the ROM at the unit's text", "at": ""}
    }


def test_locate_a_jump_table_into_the_units_own_text_at_the_address_the_link_gives(monkeypatch):
    # func_80207738_de: five relocated words, the ROM holds the text address, the object only the offsets
    offsets = (0x34, 0x2C, 0x2C, 0x24, 0x84)
    rom = word(*(TEXT_VRAM + o for o in offsets))
    emits(monkeypatch, **{".rodata": (word(*offsets), {4 * i: True for i in range(5)}, [])})
    rows = [(0, 20, "table", ".rodata", 0x800C1B88)]
    assert ownership.locate(index(rom, rows), b"", (TEXT_VRAM, TEXT_ROM))[".rodata"] == [[0, 20]]


def test_locate_a_jump_table_that_points_elsewhere_is_unplaced(monkeypatch):
    rom = word(0x00207770, 0x00207768)  # eight bytes further than the object's offsets say
    emits(monkeypatch, **{".rodata": (word(0x34, 0x2C), {0: True, 4: True}, [])})
    reason = ownership.locate(index(rom, [(0, 8, "t", ".rodata", 0x80000000)]), b"", (TEXT_VRAM, TEXT_ROM))
    assert reason[".rodata"]["why"].startswith("label table absent from the ROM")


def test_locate_loose_places_a_table_that_is_not_yet_right_where_the_roms_code_reads_it(monkeypatch):
    rom = word(0x00207770, 0x00207768) + bytes(8)
    emits(monkeypatch, **{".rodata": (word(0x34, 0x2C), {0: True, 4: True}, [(0, 0)])})
    monkeypatch.setattr(ownership, "_loaded", lambda rom, base, offset: 0x80000000)  # the code forms the row's address
    rows = [(0, 8, "t", ".rodata", 0x80000000)]
    assert "why" in ownership.locate(index(rom, rows), b"", (TEXT_VRAM, TEXT_ROM))[".rodata"]
    assert ownership.locate(index(rom, rows), b"", (TEXT_VRAM, TEXT_ROM), loose=True)[".rodata"] == [[0, 8]]


def test_locate_narrows_equal_constants_by_the_addresses_the_code_forms(monkeypatch):
    one = word(0x3F800000)
    # the unit's code at ROM 8: lui $a0, 0x800C; addiu $a0, $a0, 4 - the constant lives at exactly 0x800C0004
    rom = one + one + word(0x3C04800C, 0x24840004)
    rows = [(0, 4, "first", ".rodata", 0x800C1000), (4, 8, "second", ".rodata", 0x800C0004)]
    emits(monkeypatch, **{".rodata": (one, {}, [(4, 0)])})
    assert ownership.locate(index(rom, rows), b"", (TEXT_VRAM, 8))[".rodata"] == [[4, 8]]
    emits(monkeypatch, **{".rodata": (one, {}, [])})  # without a reference the range nearest to the code stands
    assert ownership.locate(index(rom, rows), b"", (TEXT_VRAM, 8))[".rodata"] == [[4, 8]]
    rom = bytes(8) + one + bytes(12) + one + bytes(8)  # code at 16: equally near on both sides, both stay candidates
    rows = [(8, 12, "first", ".rodata", 0x800C1000), (24, 28, "second", ".rodata", 0x800C2000)]
    assert ownership.locate(index(rom, rows), b"", (TEXT_VRAM, 16))[".rodata"] == [[8, 12], [24, 28]]


def test_locate_trusts_the_address_the_code_forms_over_a_row_that_holds_the_same_bytes(monkeypatch):
    one = word(0x3F800000)
    rom = one + word(5) + one + word(6) + word(0x3C04800C, 0x24840004)  # the code forms 0x800C0004: inside a row
    rows = [(0, 4, "elsewhere", ".rodata", 0x800C1000), (4, 16, "wide", ".rodata", 0x800C0000)]
    emits(monkeypatch, **{".rodata": (one, {}, [(4, 0)])})
    assert ownership.locate(index(rom, rows), b"", (TEXT_VRAM, 16))[".rodata"] == [[8, 12]]


def test_locate_picks_the_equal_constant_nearest_to_the_code_when_nothing_else_tells(monkeypatch):
    one = word(0x3F800000)
    rom = one + word(5, 6, 7) + one + bytes(12)
    rows = [(0, 4, "far", ".rodata", 0x800C0000), (16, 20, "near", ".rodata", 0x800C0100)]
    emits(monkeypatch, **{".rodata": (one, {}, [])})
    assert ownership.locate(index(rom, rows), b"", (TEXT_VRAM, 24))[".rodata"] == [[16, 20]]


def test_locate_names_the_runs_when_the_code_places_the_bytes_in_several_places(monkeypatch):
    """func_802362E8_de: two functions' pools, 28 foreign bytes between them: no ROM holds the 64 bytes as one run."""
    rom = word(1, 2, 7, 7, 7, 3, 4) + word(0x3C04800C, 0x24840000, 0x3C04800C, 0x24840014)
    rows = [
        (0, 8, "first", ".rodata", 0x800C0000),
        (8, 20, "foreign", ".rodata", 0x800C0008),
        (20, 28, "second", ".rodata", 0x800C0014),
    ]
    emits(monkeypatch, **{".rodata": (word(1, 2, 3, 4), {}, [(4, 0), (12, 8)])})
    got = ownership.locate(index(rom, rows), b"", (TEXT_VRAM, 28))[".rodata"]
    assert got == {"why": "the code places the bytes in 2 separate runs of the ROM, not one", "at": ""}


# ------------------------------------------------------------------ what a unit then owns


def member(name, kind="data", section=".rodata", holders=("a",)):
    return Member(name, kind, "asm", "g", tuple(Placement(v, section, 0, 4, 0x100, 0) for v in holders))


MEMBERS = {
    "f": member("f", "function", ".text"),
    "bss": member("bss", section=".bss"),
    "old1": member("old1"),
    "old2": member("old2"),
    "row": Member("row", "data", "asm", "g", tuple(Placement(v, ".rodata", 4, 8, 0x104, 0) for v in ("a",))),
    "labels": member("labels"),
}
# rom 0..12 as three whole rows that follow each other in memory
ROWS = [(0, 4, "old1", ".rodata", 0x100), (4, 8, "row", ".rodata", 0x104), (8, 12, "old2", ".rodata", 0x108)]
INDEXES = {"a": index(bytes(12), ROWS), "b": index(bytes(12), ROWS)}


def unit(*names, **kw):
    return UnitSpec("src/f.c", "c", "g", names, "tc", {"add": [], "omit": []}, **kw)


def resolve(u, result, names_from="a"):
    return ownership._resolve(u, MEMBERS, result, INDEXES, names_from)


def test_resolve_keeps_code_and_bss_and_names_each_claim_by_section_unit_and_address():
    keep, resolved, claims, debt = resolve(unit("f", "bss", "old1", "old2"), {"a": {".rodata": [[4, 12]]}})
    assert (keep, resolved, debt) == (["f", "bss"], {"a"}, [])
    assert claims == [Claim("src/f.c", "a", ".rodata", 4, 12, ("rodata/f/00000104",))]


def test_resolve_names_a_claim_by_the_names_from_holders_address_else_the_first():
    both = {"a": {".rodata": [[0, 4]]}, "b": {".rodata": [[4, 8]]}}
    assert {c.rows for c in resolve(unit("f"), both)[2]} == {("rodata/f/00000100",)}
    assert {c.rows for c in resolve(unit("f"), both, "b")[2]} == {("rodata/f/00000104",)}
    assert {c.rows for c in resolve(unit("f"), both, "c")[2]} == {("rodata/f/00000100",)}


def test_resolve_names_a_claim_after_the_data_member_the_unit_owns_where_it_starts():
    claims = resolve(unit("f", "row"), {"a": {".rodata": [[4, 8]]}})[2]
    assert [c.rows for c in claims] == [("row",)]


def test_resolve_drops_what_the_object_no_longer_emits():
    """func_80202CA0_de: the layout owned 20 bytes of rodata, the C compiles none."""
    keep, resolved, claims, debt = resolve(unit("f", "old1"), {"a": {}})
    assert (keep, resolved, claims, debt) == (["f"], {"a"}, [], [])


def test_resolve_withholds_a_holder_whose_data_cannot_be_placed():
    reason = {"why": "bytes found nowhere in the ROM at the unit's text", "at": ""}
    _, resolved, _, debt = resolve(unit("f", "old1"), {"a": {".data": reason}, "b": {}})
    assert resolved == {"b"} and debt == ["bytes found nowhere in the ROM at the unit's text: src/f.c a .data"]


def test_resolve_claims_bytes_that_end_inside_a_row_because_the_rows_follow_the_claim():
    _, resolved, claims, debt = resolve(unit("f"), {"a": {".rodata": [[2, 8]]}})
    assert resolved == {"a"} and debt == [] and [(c.start, c.end) for c in claims] == [(2, 8)]


def test_resolve_names_a_claim_that_is_not_rows_following_each_other_in_memory():
    rows = [(0, 4, "old1", ".rodata", 0x100), (4, 8, "row", ".rodata", 0x900)]  # a gap in memory
    _, resolved, _, debt = ownership._resolve(
        unit("f"), MEMBERS, {"a": {".rodata": [[0, 8]]}}, {"a": index(bytes(8), rows)}, "a"
    )
    assert resolved == set() and debt == [
        "bytes are not whole rows that follow each other in memory: src/f.c a .rodata @0x0-0x8"
    ]


def test_resolve_picks_among_ambiguous_ranges_by_the_previous_owner_else_names_them():
    options = [[4, 8], [0, 4]]
    assert [(c.start, c.end) for c in resolve(unit("f", "old1"), {"a": {".rodata": options}})[2]] == [(0, 4)]
    _, resolved, _, debt = resolve(unit("f"), {"a": {".rodata": options}})
    assert resolved == set() and debt == ["ambiguous: src/f.c a .rodata among 2 ranges 0x4-0x8, 0x0-0x4"]


def test_a_range_that_is_exactly_an_owned_row_beats_ranges_that_overlap_it():
    hits = [[0, 8], [4, 12], [8, 16]]
    assert ownership._range(hits, [(4, 12)]) == ([4, 12], "")
    assert ownership._range(hits, [(2, 6)])[0] is None  # two overlap, none is exact


@pytest.mark.parametrize(
    "kind, withheld, hits, expected",
    [
        ("data", (), [[4, 8]], None),
        ("data", ("b",), [[4, 8]], None),
        ("data", ("b",), [[0, 4]], (0, 4)),
        ("data", (), [[0, 4], [4, 8]], (0, 4)),
        ("data", ("b",), [[0, 8]], None),
        ("data", (), [[1, 4]], (1, 4)),
        ("c", ("b",), [[4, 8]], (4, 8)),
    ],
)
def test_data_stays_on_its_rows_and_withheld_holders_recover_only_there(monkeypatch, kind, withheld, hits, expected):
    members = {"own": member("own", holders=("a", "b")), "f": member("f", "function", ".text", ("a", "b"))}
    rows = [(0, 4, "own", ".rodata", 0x100), (4, 8, "neighbor", ".rodata", 0x104)]
    indexes = {v: index(bytes(8), rows) for v in ("a", "b")}
    u = replace(unit(*(("own",) if kind == "data" else ("f", "own")), withheld=withheld), kind=kind)
    monkeypatch.setattr(ownership, "locate", Mock(return_value={".rodata": hits}))
    result = {"a": {".rodata": [[0, 4]]}, "b": ownership.locate(indexes["b"], b"", None)}
    keep, resolved, claims, debt = ownership._resolve(u, members, result, indexes, "a")
    owned, kept = ownership._own(u, members, keep, resolved, claims, defaultdict(list), debt)
    assert owned.withheld == (("b",) if expected is None else ())
    assert [(c.start, c.end) for c in kept if c.version == "b"] == ([] if expected is None else [expected])
    assert all(c.rows == ("own",) for c in kept)
    assert debt == (["bytes outside owned rows: src/f.c b .rodata"] if expected is None else [])


@pytest.mark.parametrize("here, expected", [([(0, 4), (4, 8)], [0, 8]), ([(0, 4), (5, 8)], None), ([], None)])
def test_data_claim_can_span_adjacent_owned_rows_but_cannot_cross_a_foreign_gap(here, expected):
    assert ownership._range([[0, 8]], here, True)[0] == expected


def test_content_search_cut_short_still_counts_the_row_starts():
    count = ownership._MANY + 4
    rom = word(*([7] * (count + 1)))
    rows = [(0, 4, "a", ".rodata", 0x80000000), (4 * count, 4 * count + 4, "r", ".rodata", 0x80000100)]
    assert 4 * count in ownership._content(index(rom, rows), word(7), {}, None)
    assert len(ownership._content(index(rom, rows), word(7), {}, None)) == ownership._MANY + 1


def test_resolve_withholds_a_holder_where_the_unit_does_not_compile():
    _, resolved, _, debt = resolve(unit("f", "old1"), {"a": "preprocess.error: boom"})
    assert resolved == set() and debt == ["compile failed (preprocess.error): src/f.c a boom"]


def test_own_names_the_rows_of_the_kept_claims_and_withholds_the_failing_holders():
    members = {"f": member("f", "function", ".text", ("a", "b", "c"))}
    found = [Claim("src/f.c", "a", ".rodata", 0, 4, ("r",)), Claim("src/f.c", "b", ".rodata", 8, 12, ("r",))]
    taken = defaultdict(list)
    result, kept = ownership._own(unit("f"), members, ["f"], {"a", "b"}, found, taken, [])  # holder c is withheld
    assert (result.members, result.withheld, kept) == (("f", "r"), ("c",), found)
    assert taken == {"a": [(0, 4, "src/f.c")], "b": [(8, 12, "src/f.c")]}


def test_own_withholds_the_holder_that_wants_bytes_another_unit_owns_in_that_version_only():
    members = {"f": member("f", "function", ".text", ("a", "b"))}
    taken, debt = defaultdict(list, {"a": [(2, 6, "src/other.c")]}), []
    found = [Claim("src/f.c", "a", ".rodata", 4, 8, ("r",)), Claim("src/f.c", "b", ".rodata", 4, 8, ("r",))]
    result, kept = ownership._own(unit("f"), members, ["f"], {"a", "b"}, found, taken, debt)
    assert result.withheld == ("a",) and kept == found[1:] and result.members == ("f", "r")
    assert debt == ["row owned by another unit: src/f.c a (claimed by src/other.c)"]


@pytest.mark.parametrize(
    "start, end, owner",
    [(0, 2, None), (0, 4, "x"), (3, 5, "x"), (6, 7, "y"), (4, 6, None), (0, 20, "x"), (7, 9, "y"), (8, 9, None)],
)
def test_clash_names_the_unit_that_holds_some_of_the_bytes(start, end, owner):
    assert ownership._clash([(2, 4, "x"), (6, 8, "y")], start, end) == owner


def test_claims_give_bytes_to_one_unit_only_and_aggregate_the_debt(monkeypatch):
    missing = {"why": "not in the ROM", "at": ""}
    monkeypatch.setattr(
        ownership.pool,
        "map",
        lambda cfg, name, fn, items, key=None: [
            {"a": {".rodata": [[4, 8]]}},
            {"a": {".rodata": [[4, 8]]}},
            {"a": {".rodata": missing}},
        ],
    )
    monkeypatch.setattr(ownership, "_index", lambda snapshot, v: INDEXES["a"])
    monkeypatch.setattr(
        ownership.config, "load_resource", lambda name: {"kind": {"c": {"credit": "code", "phases": ["compile"]}}}
    )
    found = {}

    @contextmanager
    def stage(name):
        yield SimpleNamespace(add=lambda **kw: found.update(kw))

    monkeypatch.setattr(ownership.effort, "stage", stage)
    units = {p: replace(unit("f"), path=p) for p in ("src/a.c", "src/b.c", "src/c.c")}
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
        layout=LayoutMap({}, MEMBERS, units, "d", (), {}),
        versions={"a": None},
        digest="claims_give_bytes_to",
    )
    out, claims, findings = ownership.claims(snapshot)
    assert findings == found["findings"]
    assert [out[p].members for p in units] == [("f", "rodata/a/00000104"), ("f",), ("f",)]  # b.c lost them to a.c
    assert [out[p].withheld for p in units] == [(), ("a",), ("a",)]
    assert claims == (Claim("src/a.c", "a", ".rodata", 4, 8, ("rodata/a/00000104",)),)
    assert [f.reason for f in found["findings"]] == [
        "1 unit holders are withheld: not in the ROM",
        "1 unit holders are withheld: row owned by another unit",
    ]
    assert not found["findings"][0].blocking and found["items"] == 3


def test_exact_withholds_the_holders_whose_bytes_do_not_reproduce_the_rom(monkeypatch):
    members = {"f": member("f", "function", ".text", ("a", "b"))}
    monkeypatch.setattr(
        ownership.config, "load_resource", lambda name: {"kind": {"c": {"credit": "code", "phases": ["compile"]}}}
    )
    seen = []
    gaps = ["", ".text is 0x28 bytes"]
    monkeypatch.setattr(ownership.pool, "map", lambda cfg, name, fn, items, key=None: seen.extend(items) or gaps)
    found = {}

    @contextmanager
    def stage(name):
        yield SimpleNamespace(add=lambda **kw: found.update(kw))

    monkeypatch.setattr(ownership.effort, "stage", stage)
    snapshot = Snapshot(None, "h", LayoutMap({}, members, {}, "d", (), {}), {}, {}, "d")
    out, findings = ownership.exact(snapshot, {"src/f.c": unit("f", withheld=("c",))})
    assert [(u.path, v) for _, u, v in seen] == [("src/f.c", "a"), ("src/f.c", "b")]  # a withheld holder is not proved
    assert out["src/f.c"].withheld == ("b", "c")
    assert "not exact" in found["findings"][0].reason and findings == found["findings"]


def test_landed_c_that_does_not_compile_refuses_while_the_ownership_classes_stay_debt():
    found = ownership._findings(
        {
            "compile failed (compile.error)": ["src/f.c a boom"],
            "compile failed (symbols.unknown)": ["src/g.c b x"],
            "label table absent from the ROM": ["src/h.c a .data"],
        },
        "fix it",
    )
    assert [(f.key, f.blocking) for f in found] == [
        ("compile.error", True),
        ("layout.ownership", False),
        ("layout.ownership", False),
    ]
    assert found[0].reason == "1 unit holders do not compile: compile failed (compile.error)"
    assert found[0].missing == ("src/f.c a boom",) and found[0].action == "fix it"


@pytest.fixture
def inline(monkeypatch):
    monkeypatch.setattr(ownership.pool, "map", lambda config, name, fn, items, key=None: [fn(i) for i in items])


def test_derive_gives_a_candidate_the_rows_its_source_emits(monkeypatch, inline):
    from unbake import layout

    members = {"f": member("f", "function", ".text"), "row": member("row")}
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
        layout=LayoutMap({}, members, {}, "d", (), {}),
        versions={"a": None},
    )
    monkeypatch.setattr(
        ownership.config, "load_resource", lambda name: {"kind": {"c": {"credit": "code", "phases": ["compile"]}}}
    )
    monkeypatch.setattr(ownership, "_index", lambda snap, v: INDEXES["a"])
    monkeypatch.setattr(ownership, "_job", lambda item: {"a": {".rodata": [[4, 8]]}})
    monkeypatch.setattr(layout, "claim_rows", lambda snap, claims: {})  # the rows are the claims already
    shown, got = ownership.derive(snapshot, unit("f"))
    assert shown is snapshot and got.members == ("f", "rodata/f/00000104") and got.withheld == ()
    monkeypatch.setattr(
        ownership.config, "load_resource", lambda name: {"kind": {"c": {"credit": "code", "phases": ["assemble"]}}}
    )
    assert ownership.derive(snapshot, unit("f")) == (snapshot, unit("f"))  # not compiled: as it is


def test_derive_returns_a_data_unit_as_it_is(monkeypatch, inline):
    snapshot = SimpleNamespace(layout=SimpleNamespace(members={"row": member("row")}, units={}))
    monkeypatch.setattr(
        ownership.config, "load_resource", lambda name: {"kind": {"c": {"credit": "code", "phases": ["compile"]}}}
    )
    assert ownership.derive(snapshot, unit("row")) == (snapshot, unit("row"))


def test_derive_generates_the_rows_in_a_private_copy_and_takes_other_units_rows_as_taken(monkeypatch, inline):
    from unbake import layout

    members = {"f": member("f", "function", ".text"), "mine": member("mine"), "theirs": member("theirs")}
    other = replace(unit("theirs"), path="src/g.c")
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
        layout=LayoutMap({}, members, {"src/g.c": other}, "d", (), {}),
        versions={"a": None},
    )
    copy = SimpleNamespace(marker="copy")
    seen = {}
    monkeypatch.setattr(
        ownership.config, "load_resource", lambda name: {"kind": {"c": {"credit": "code", "phases": ["compile"]}}}
    )
    monkeypatch.setattr(ownership, "_index", lambda snap, v: INDEXES["a"])
    monkeypatch.setattr(ownership, "_job", lambda item: {"a": {".rodata": [[8, 12], [0, 4]][:1]}})
    monkeypatch.setattr(layout, "claim_rows", lambda snap, claims: {"split.yaml": b"rows"})
    monkeypatch.setattr(layout, "dump_map", lambda value: seen.setdefault("units", value.units) and b"map")
    monkeypatch.setattr(layout, "overlay", lambda snap, writes: seen.setdefault("writes", writes) and copy)
    shown, got = ownership.derive(snapshot, unit("f"))
    assert shown is copy and seen["writes"] == {"split.yaml": b"rows", "layout.toml": b"map"}
    assert seen["units"]["src/f.c"] == got and "src/g.c" in seen["units"]
    assert got.members == ("f", "rodata/f/00000108") and got.withheld == ()


def test_derive_does_not_reserve_another_units_unclaimed_holder(monkeypatch, inline):
    from unbake import layout

    members = {
        "f": member("f", "function", ".text", ("b",)),
        "rodata/g/00000100": member("rodata/g/00000100", holders=("a",)),
    }
    other = replace(unit("rodata/g/00000100"), path="src/g.c")
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
        layout=LayoutMap({}, members, {other.path: other}, "d", (), {}),
        versions={"a": None, "b": None},
    )
    monkeypatch.setattr(
        ownership.config, "load_resource", lambda name: {"kind": {"c": {"credit": "code", "phases": ["compile"]}}}
    )
    monkeypatch.setattr(ownership, "_index", lambda snap, v: INDEXES[v])
    monkeypatch.setattr(ownership, "_job", lambda item: {"b": {".rodata": [[0, 4]]}})
    claims = []
    monkeypatch.setattr(layout, "claim_rows", lambda snap, found: claims.extend(found) or {})
    shown, got = ownership.derive(snapshot, unit("f"))
    assert shown is snapshot and got.withheld == ()
    assert claims == [Claim("src/f.c", "b", ".rodata", 0, 4, ("rodata/f/00000100",))]
    assert got.members == ("f", "rodata/f/00000100")


def test_derive_takes_the_rows_of_an_unowned_data_unit_for_the_function_that_claims_them(monkeypatch, inline):
    from unbake import layout

    name = "rodata/unresolved/00000100"
    members = {"f": member("f", "function", ".text"), name: member(name)}
    placeholder = replace(unit(name), path="src/data.c", kind="data")
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
        layout=LayoutMap({}, members, {placeholder.path: placeholder}, "d", (), {}),
        versions={"a": None},
    )
    seen = {}
    monkeypatch.setattr(
        ownership.config,
        "load_resource",
        lambda n: {
            "kind": {"c": {"credit": "code", "phases": ["compile"]}, "data": {"credit": "data", "phases": ["compile"]}}
        },
    )
    monkeypatch.setattr(ownership, "_index", lambda snap, v: INDEXES["a"])
    monkeypatch.setattr(ownership, "_job", lambda item: {"a": {".rodata": [[0, 4]]}})
    monkeypatch.setattr(layout, "claim_rows", lambda snap, claims: {"split.yaml": b"rows"})
    monkeypatch.setattr(layout, "dump_map", lambda value: seen.setdefault("units", value.units) and b"map")
    monkeypatch.setattr(layout, "overlay", lambda snap, writes: snap)
    _, got = ownership.derive(snapshot, unit("f"))
    assert got.withheld == () and got.members == ("f", "rodata/f/00000100")  # not "owned by another unit"
    assert set(seen["units"]) == {"src/f.c"}  # the placeholder unit is gone with its row


def test_claims_let_a_function_unit_take_bytes_before_an_unowned_data_unit(monkeypatch):
    name = "rodata/unresolved/00000100"
    members = {"f": member("f", "function", ".text"), name: member(name)}
    units = {"src/data.c": replace(unit(name), path="src/data.c"), "src/f.c": unit("f")}
    order = []
    monkeypatch.setattr(
        ownership.config, "load_resource", lambda n: {"kind": {"c": {"credit": "code", "phases": ["compile"]}}}
    )
    monkeypatch.setattr(
        ownership.pool,
        "map",
        lambda cfg, label, fn, items, key=None: order.extend(i[1].path for i in items) or [{"a": {}} for _ in items],
    )
    monkeypatch.setattr(ownership, "_index", lambda snap, v: INDEXES["a"])
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
        layout=LayoutMap({}, members, units, "d", (), {}),
        versions={"a": None},
        digest="claims_let_a_functio",
    )
    ownership.claims(snapshot)
    assert order == ["src/f.c", "src/data.c"]


def test_ownership_dispatches_one_keyed_job_per_unit_and_holder_so_versions_build_side_by_side(monkeypatch):
    members = {"f": member("f", "function", ".text", ("a", "b", "c"))}
    snapshot = SimpleNamespace(
        config=None, versions={"a": None, "b": None, "c": None}, layout=LayoutMap({}, members, {}, "d", (), {})
    )
    units = [unit("f"), replace(unit("f"), toolchain="other")]
    seen = []

    def run(cfg, label, function, jobs, key):
        seen.append((function, key, jobs))
        return [{job[2]: "x"} for job in jobs]

    monkeypatch.setattr(ownership.pool, "map", run)
    monkeypatch.setattr(ownership, "_derived", lambda snap, u, result: (snap, u))
    ownership.derive_many(snapshot, units, recipe="trial")
    ((function, key, jobs),) = seen
    assert (function, key) == (ownership._job, ownership._job_key) and len(jobs) == 3 * len(units)
    assert [job[2] for job in jobs] == ["a", "b", "c"] * 2  # each job is one holder of its unit
    assert [job[4] for job in jobs] == ["trial"] * 6  # the trial recipe is the one whose object is followed


def test_ownership_keys_name_the_pinned_stamp_of_every_holder_so_an_unchanged_unit_dispatches_nothing(monkeypatch):
    members = {"f": member("f", "function", ".text", ("a", "b"))}
    snapshot = SimpleNamespace(
        config=None, versions={"a": None, "b": None}, layout=LayoutMap({}, members, {}, "d", (), {})
    )
    stamps = []

    def stamp(snap, u, v, recipe=None, pinned=False, code=True):
        assert pinned  # the key never preprocesses the unit
        stamps.append((v, recipe))
        return f"stamp-{v}-{recipe}"

    monkeypatch.setattr(ownership.native, "stamp", stamp)
    monkeypatch.setattr(ownership, "_index", lambda snap, version: INDEXES["a"])
    assert ownership._job_key((snapshot, unit("f"), None, False, None)) == ownership._job_key(
        (snapshot, unit("f"), None, False, None)
    )
    assert [v for v, _ in stamps] == ["a", "b", "a", "b"]
    assert ownership._job_key((snapshot, unit("f"), None, False, "trial")) != ownership._job_key(
        (snapshot, unit("f"), None, False, None)
    )  # a trial recipe is another build of the same unit
    assert ("a", "trial") in stamps


@pytest.mark.parametrize("name", ["rodata/f/00000100", "src/old_pool.c"])
@pytest.mark.parametrize("coverage", ["whole", "partial", "missing_holder", "function_owner"])
def test_derive_reclaims_only_a_complete_independent_data_provider(monkeypatch, inline, name, coverage):
    from unbake import layout

    old = replace(unit(name), path="src/old.c", kind="c" if coverage == "function_owner" else "data")
    held = ("a", "b") if coverage == "missing_holder" else ("a",)
    row = member(name, holders=held)
    if coverage == "partial":
        row = replace(row, placements=(Placement("a", ".rodata", 0, 8, 0x100),))
    members = {"f": member("f", "function", ".text"), name: row}
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
        layout=LayoutMap({}, members, {old.path: old}, "d", (), {}),
        versions={"a": None, "b": None},
    )
    seen = {}
    monkeypatch.setattr(
        ownership.config,
        "load_resource",
        lambda n: {
            "kind": {"c": {"credit": "code", "phases": ["compile"]}, "data": {"credit": "data", "phases": ["compile"]}}
        },
    )
    monkeypatch.setattr(ownership, "_index", lambda snap, v: INDEXES[v])
    monkeypatch.setattr(ownership, "_job", lambda item: {"a": {".rodata": [[0, 4]]}})
    monkeypatch.setattr(layout, "claim_rows", lambda snap, claims: {"split.yaml": b"rows"} if claims else {})
    monkeypatch.setattr(layout, "dump_map", lambda value: seen.setdefault("units", value.units) and b"map")
    monkeypatch.setattr(layout, "overlay", lambda snap, writes: snap)
    _, got = ownership.derive(snapshot, unit("f"))
    # a provider holding the claimed bytes inside a longer row yields too: the claim splits the row
    assert got.withheld == (() if coverage in ("whole", "partial") else ("a",))
    if coverage in ("whole", "partial"):
        assert set(seen["units"]) == {"src/f.c"}


def test_a_derive_tests_only_the_units_near_its_claims_and_lists_every_units_rows_once_per_layout(monkeypatch, inline):
    """A bulk submit derives hundreds of consumers in one tree: each derive used to test every unit of the layout
    for yielding and re-sort every data row in the parent. Only units naming or touching a claim can yield."""
    from unbake import layout

    far = 300
    name = "rodata/f/00000100"
    provider = replace(unit(name), path="src/old.c", kind="data")
    members = {"f": member("f", "function", ".text"), name: member(name)}
    units = {provider.path: provider}
    for i in range(far):  # data rows elsewhere in the ROM, owned by other units
        row = f"d{i}"
        members[row] = Member(row, "data", "asm", "g", (Placement("a", ".rodata", 64 + 8 * i, 72 + 8 * i, 0x200, 0),))
        units[f"src/d{i}.c"] = replace(unit(row), path=f"src/d{i}.c", kind="data")
    mapping = LayoutMap({}, members, units, "d", (), {})
    trees = [
        SimpleNamespace(
            config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
            layout=mapping,
            versions={"a": None},
            digest=f"tree{n}",
        )
        for n in range(3)
    ]
    monkeypatch.setattr(
        ownership.config,
        "load_resource",
        lambda n: {
            "kind": {"c": {"credit": "code", "phases": ["compile"]}, "data": {"credit": "data", "phases": ["compile"]}}
        },
    )
    monkeypatch.setattr(ownership, "_index", lambda snap, v: INDEXES[v])
    monkeypatch.setattr(ownership, "_job", lambda item: {"a": {".rodata": [[0, 4]]}})
    kept = {}
    monkeypatch.setattr(layout, "claim_rows", lambda snap, claims: {"split.yaml": b"rows"} if claims else {})
    monkeypatch.setattr(layout, "dump_map", lambda value: kept.setdefault(len(kept), set(value.units)) and b"map")
    monkeypatch.setattr(layout, "overlay", lambda snap, writes: snap)
    tested = []
    covered = ownership._covered
    monkeypatch.setattr(
        ownership, "_covered", lambda snap, u, claims, held=(): tested.append(u.path) or covered(snap, u, claims, held)
    )
    built = []
    real = ownership.effort.memo
    monkeypatch.setattr(ownership.effort, "memo", lambda key, make: real(key, lambda: built.append(key[0]) or make()))
    got = ownership.derive_pairs([(t, unit("f")) for t in trees])
    assert all(name in u.members and not u.withheld for _, u in got)
    assert set(tested) == {provider.path}  # never one of the 300 far units
    assert len(tested) == 2 * len(trees)  # the yield test and the rows rewrite, per derive
    assert built.count("ownership.rows") == 1  # three tree states of one layout list its rows once
    assert all(units_ == {"src/f.c", *(f"src/d{i}.c" for i in range(far))} for units_ in kept.values())


def test_derive_a_provider_of_the_claimed_rows_yields_in_the_versions_the_claims_reach(monkeypatch, inline):
    """The jump table of a function exact in some versions: its provider holds the row in all, the claims name it in
    the versions where the compile placed it. The compiled unit takes the row (withheld elsewhere), not a clash."""
    from unbake import layout

    name = "rodata/f/00000100"
    old = replace(unit(name), path="src/old.c", kind="data")
    row = replace(
        member(name, holders=("a", "b")),
        placements=(Placement("a", ".rodata", 0, 4, 0x100), Placement("b", ".rodata", 0, 4, 0x100)),
    )
    members = {"f": member("f", "function", ".text", ("a", "b")), name: row}
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
        layout=LayoutMap({}, members, {old.path: old}, "d", (), {}),
        versions={"a": None, "b": None},
    )
    seen = {}
    monkeypatch.setattr(
        ownership.config,
        "load_resource",
        lambda n: {
            "kind": {"c": {"credit": "code", "phases": ["compile"]}, "data": {"credit": "data", "phases": ["compile"]}}
        },
    )
    monkeypatch.setattr(ownership, "_index", lambda snap, v: INDEXES[v])
    monkeypatch.setattr(ownership, "_job", lambda item: {"a": {".rodata": [[0, 4]]}, "b": "native.exit: no table"})
    monkeypatch.setattr(layout, "claim_rows", lambda snap, claims: {"split.yaml": b"rows"} if claims else {})
    monkeypatch.setattr(layout, "dump_map", lambda value: seen.setdefault("units", value.units) and b"map")
    monkeypatch.setattr(layout, "overlay", lambda snap, writes: snap)
    _, got = ownership.derive(snapshot, unit("f"))
    assert got.withheld == ("b",) and name in got.members
    assert set(seen["units"]) == {"src/f.c"}


@pytest.mark.parametrize("key", ["journal.changed", "compile.error"])
def test_job_propagates_stale_inputs_but_keeps_compile_debt(monkeypatch, tmp_path, key):
    spec = unit("f")
    snapshot = SimpleNamespace(config=object(), layout=LayoutMap({}, MEMBERS, {spec.path: spec}, "d", (), {}))
    refusal = Refusal(Finding(key, "source moved"))
    monkeypatch.setattr(ownership.store, "work", lambda cfg: nullcontext(tmp_path))
    monkeypatch.setattr(ownership.native, "objects", Mock(side_effect=refusal))
    monkeypatch.setattr(ownership, "_built_key", lambda snap, u, v, recipe: f"built-{v}")
    monkeypatch.setattr(ownership.store, "cached_object", lambda cfg, kind, key, produce: produce())
    item = (snapshot, spec, "a", False, object())
    if key == "journal.changed":
        with pytest.raises(Refusal) as caught:
            ownership._job(item)
        assert caught.value is refusal
    else:
        assert ownership._job(item) == {"a": "compile.error: source moved"}


def test_literal_low_relocation_includes_the_symbol_section_offset(monkeypatch):
    sections = [
        Section(".text", word(0x3C010000, 0xC4200000)),
        Section(".rodata", word(0, 0x4F000000)),
        RelSection(
            ".rel.text", kind="SHT_REL", link=3, info=0, relocs=[{"r_offset": 4, "r_info_sym": 0, "r_info_type": 6}]
        ),
        SymSection(".symtab", kind="SHT_SYMTAB", symbols=[{"st_shndx": 1, "st_value": 4}]),
    ]
    monkeypatch.setattr(ownership, "ELFFile", lambda _: SimpleNamespace(iter_sections=lambda: iter(sections)))
    assert ownership.emitted(b"object")[".rodata"][2] == [(4, 4)]


def test_initialized_data_provider_yields_to_a_proved_compiler_rodata_claim(monkeypatch):
    provider = UnitSpec("src/pool.c", "data", "g", ("pool",), "t", {})
    member = Member("pool", "data", "data", "g", (Placement("a", ".data", 8, 16, 0x80000008),))
    snapshot = SimpleNamespace(layout=SimpleNamespace(members={"pool": member}))
    monkeypatch.setattr(ownership.layout, "is_datum", lambda unit: True)
    assert ownership._covered(snapshot, provider, [Claim("src/f.c", "a", ".rodata", 12, 16, ("literal",))], {"a"})


# ------------------------------------------------------------------ borrowed pools


def _borrow(monkeypatch, rows, at, size, words=None):
    """What borrowed() returns for a pool the unit's code places at ROM offset `at`, over the given data rows."""
    rom = bytes(0x100)
    emits(monkeypatch, **{".rodata": (bytes(size), words or {}, [(0, 4)])})
    monkeypatch.setattr(ownership, "_index", lambda snapshot, version: index(rom, rows))
    monkeypatch.setattr(ownership, "_placed", lambda *args, **kw: at)
    text = Placement("us", ".text", 0x1000, 0x1040, TEXT_VRAM)
    return ownership.borrowed(Mock(), UnitSpec("src/u.c", "c", "g", (), "t", {}), "us", b"", [text])


def test_borrowed_places_jump_table_with_text_words(monkeypatch):
    rows = [(0x40, 0x4C, "D_a", ".rodata", 0x80300040)]
    (placed,) = _borrow(monkeypatch, rows, 0x40, 12, words={0: True, 4: True, 8: True})
    assert (placed.section, placed.rom_start, placed.rom_end) == (".rodata", 0x40, 0x4C)


def test_borrowed_pool_spanning_two_rows(monkeypatch):
    rows = [(0x40, 0x44, "D_a", ".rodata", 0x80300040), (0x44, 0x48, "D_b", ".rodata", 0x80300044)]
    (placed,) = _borrow(monkeypatch, rows, 0x40, 8)
    assert (placed.rom_start, placed.rom_end) == (0x40, 0x48)


def test_borrowed_refuses_gap_between_rows(monkeypatch):
    rows = [(0x40, 0x44, "D_a", ".rodata", 0x80300040), (0x48, 0x4C, "D_b", ".rodata", 0x80300048)]
    assert _borrow(monkeypatch, rows, 0x40, 12) == ()


def test_deriving_n_units_is_one_dispatch_and_writes_claim_rows_only_for_units_that_claim(monkeypatch):
    from unbake import layout

    count = 40
    members = {f"f{i}": member(f"f{i}", "function", ".text") for i in range(count)}
    units = {f"src/f{i}.c": replace(unit(f"f{i}"), path=f"src/f{i}.c") for i in range(count)}
    snapshot = SimpleNamespace(
        config=SimpleNamespace(project=SimpleNamespace(names_from="a")),
        layout=LayoutMap({}, members, units, "d", (), {}),
        versions={"a": None},
    )
    monkeypatch.setattr(
        ownership.config, "load_resource", lambda name: {"kind": {"c": {"credit": "code", "phases": ["compile"]}}}
    )
    monkeypatch.setattr(ownership, "_index", lambda snap, v: INDEXES["a"])
    dispatched, claimed = [], []
    monkeypatch.setattr(
        ownership.pool,
        "map",
        lambda cfg, name, fn, items, key=None: dispatched.append(len(items)) or [{"a": {}} for _ in items],
    )
    monkeypatch.setattr(layout, "claim_rows", lambda snap, claims: claimed.append(claims) or {})
    ownership.derive_many(snapshot, list(units.values()))
    assert dispatched == [count] and claimed == []  # one pool.map for all units; no claim, no claim_rows call


def test_a_tree_state_is_derived_once_per_command_its_output_is_a_fixed_point_and_a_changed_input_derives_again(
    monkeypatch,
):
    members = {"f": member("f", "function", ".text", ("a", "b")), "g": member("g", "function", ".text", ("a", "b"))}
    units = [unit("f"), replace(unit("g"), path="src/g.c")]
    tree = SimpleNamespace(
        config=None, versions={"a": None, "b": None}, layout=LayoutMap({}, members, {}, "d", (), {}), digest="t1"
    )
    owned = SimpleNamespace(**{**vars(tree), "digest": "t1-owned"})  # what deriving lays over the tree
    dispatched = []
    monkeypatch.setattr(ownership, "_emits_data", lambda snap, u: True)
    monkeypatch.setattr(
        ownership.pool,
        "map",
        lambda cfg, name, fn, jobs, key=None: dispatched.append(len(jobs)) or [{j[2]: {}} for j in jobs],
    )
    monkeypatch.setattr(ownership, "_derived", lambda snap, u, result: (owned, u))
    first = ownership.derive_many(tree, units, recipe="trial")
    assert ownership.derive_many(tree, units, recipe="trial") == first  # the same inputs: taken as derived
    assert [ownership.derive(owned, u, recipe="trial") for u in units] == first  # a derive's output holds its rows
    assert dispatched == [4]  # 2 units x 2 holders, once for the command
    changed = SimpleNamespace(**{**vars(tree), "digest": "t2"})
    ownership.derive(changed, units[0], recipe="trial")  # an input digest moved: derived again
    ownership.derive(tree, units[0], recipe="other")  # so does another trial recipe
    assert dispatched == [4, 2, 2]


def test_derive_pairs_places_every_tree_state_in_one_dispatch(monkeypatch):
    members = {"f": member("f", "function", ".text", ("a",))}
    trees = [
        SimpleNamespace(config=None, versions={"a": None}, layout=LayoutMap({}, members, {}, "d", (), {}), digest=d)
        for d in ("t1", "t2", "t3")
    ]
    dispatched = []
    monkeypatch.setattr(ownership, "_emits_data", lambda snap, u: True)
    monkeypatch.setattr(
        ownership.pool,
        "map",
        lambda cfg, name, fn, jobs, key=None: (
            dispatched.append([j[0].digest for j in jobs]) or [{"a": {}} for _ in jobs]
        ),
    )
    monkeypatch.setattr(ownership, "_derived", lambda snap, u, result: (snap, u))
    got = ownership.derive_pairs([(t, unit("f")) for t in trees])
    assert dispatched == [["t1", "t2", "t3"]] and got == [(t, unit("f")) for t in trees]


def test_borrowed_places_a_pool_where_the_code_forms_it_even_when_its_bytes_differ(monkeypatch):
    seen = []
    rows = [(0x40, 0x48, "D_a", ".rodata", 0x80300040)]
    rom = bytes(0x100)
    emits(monkeypatch, **{".rodata": (b"\1" * 8, {}, [(0, 4)])})
    monkeypatch.setattr(ownership, "_index", lambda snapshot, version: index(rom, rows))
    monkeypatch.setattr(ownership, "_placed", lambda *args, **kw: seen.append(kw) or 0x40)
    text = Placement("us", ".text", 0x1000, 0x1040, TEXT_VRAM)
    (placed,) = ownership.borrowed(Mock(), UnitSpec("src/u.c", "c", "g", (), "t", {}), "us", b"", [text])
    assert (placed.rom_start, placed.rom_end) == (0x40, 0x48) and seen == [{"loose": True}]


def test_a_low_half_ref_aligns_to_the_targets_own_instruction_when_the_candidate_is_a_word_longer():
    # target: lui at,0x800D; lwc1 f20,-0x7A14(at). The candidate has one more instruction first: its lo is at +8
    lui, lwc1 = 0x3C01800D, 0xC43485EC
    rom = b"".join(w.to_bytes(4, "big") for w in (0, lui, lwc1, 0))
    code = b"".join(w.to_bytes(4, "big") for w in (0, 0, 0x3C010000, 0xC4340000))
    assert ownership._loaded(rom, 0, 12) is None  # the word at the candidate's offset forms nothing
    assert ownership._aligned(rom, 0, code, [(12, 0)]) == [(8, 0)]
    assert ownership._loaded(rom, 0, 8) == 0x800C85EC
    assert ownership._aligned(rom, 0, rom, [(8, 0)]) == [(8, 0)]  # a candidate equal to the target: kept
    assert ownership._aligned(bytes(16), 0, code, [(12, 0)]) == []  # no lo near: the ref forms nothing


def test_a_move_that_renames_a_data_row_keeps_the_index_digest_every_ownership_key_reads(monkeypatch):
    # cutover slow step: one member move renamed or re-homed a data row and the index digest of the whole version
    # moved, so all 5,446 units' layout.ownership keys went cold (404 s for one move)
    monkeypatch.setattr(ownership.versions, "rom_view", lambda record: word(1, 2, 3, 4))

    def snapshot(name, digest):
        rows = [SimpleNamespace(version="a", section=".rodata", rom_start=0, rom_end=8, vram=0x80000000)]
        moved = [SimpleNamespace(version="a", section=".data", rom_start=8, rom_end=16, vram=0x80000008)]
        members = {
            "D_80000000": SimpleNamespace(kind="data", name="D_80000000", placements=rows),
            name: SimpleNamespace(kind="data", name=name, placements=moved),
        }
        record = SimpleNamespace(segments=())
        return SimpleNamespace(versions={"a": record}, layout=SimpleNamespace(members=members), digest=digest)

    before = ownership._index(snapshot("D_80000008", "one"), "a")
    after = ownership._index(snapshot("rodata/func_80000010/80000008", "two"), "a")
    assert before.rows != after.rows and before.digest == after.digest
    shifted = snapshot("D_80000008", "three")
    shifted.layout.members["D_80000008"].placements[0].rom_end = 12  # a row cut elsewhere: placements may differ
    assert ownership._index(shifted, "a").digest != before.digest


def test_an_overlay_pass_locates_only_the_units_the_writes_reach_and_keeps_the_rest(monkeypatch):
    located = {"src/a.c": {".rodata": [[4, 8]]}, "src/b.c": {".rodata": [[0, 4]]}, "src/c.c": {".rodata": [[8, 12]]}}
    seen: list[list[str]] = []

    def run(cfg, name, fn, items, key=None):
        seen.append([u.path for _, u, *_ in items])
        return [{"a": located[u.path]} for _, u, *_ in items]

    monkeypatch.setattr(ownership.pool, "map", run)
    digests = {"before": INDEXES["a"], "after": INDEXES["a"]}
    monkeypatch.setattr(ownership, "_index", lambda snap, v: digests[snap.digest.split("-")[0]])
    monkeypatch.setattr(
        ownership.config, "load_resource", lambda name: {"kind": {"c": {"credit": "code", "phases": ["compile"]}}}
    )
    units = {p: replace(unit("f"), path=p) for p in ("src/a.c", "src/b.c", "src/c.c")}
    cfg, versions = SimpleNamespace(project=SimpleNamespace(names_from="a")), {"a": None}

    def snap(name, layout_units, members=MEMBERS):
        return SimpleNamespace(
            config=cfg, layout=LayoutMap({}, members, layout_units, "d", (), {}), versions=versions, digest=name
        )

    before = snap("before-1", units)
    changed = {**units, "src/c.c": replace(units["src/c.c"], options={"add": ["-O1"], "omit": []})}
    after = snap("after-1", changed)
    full = ownership.claims(after)[1]  # what a pass of its own finds
    seen.clear()
    old = ownership.claims(before)[1]
    assert seen == [["src/a.c", "src/b.c", "src/c.c"]]
    new = ownership.claims(snap("after-2", changed), reuse=(before, {"src/b.c"}))[1]
    assert seen[1] == ["src/b.c", "src/c.c"]  # b is reached, c's spec changed, a keeps its placements
    assert new == full and old
    digests["after"] = INDEXES["b"]._replace(digest="moved rows")  # a data row moved: nothing is kept
    ownership.claims(snap("after-3", changed), reuse=(before, set()))
    assert seen[2] == ["src/a.c", "src/b.c", "src/c.c"]
    moved = {**MEMBERS, "f": replace(MEMBERS["f"], state="c")}  # a member's record changed: its units go again
    digests["after"] = INDEXES["a"]
    ownership.claims(snap("after-4", units, moved), reuse=(before, set()))
    assert seen[3] == ["src/a.c", "src/b.c", "src/c.c"]


def test_the_job_key_names_only_the_code_that_follows_an_object_to_the_rom(monkeypatch):
    asked = []
    monkeypatch.setattr(ownership.native, "stamp", lambda *a, **kw: asked.append(kw) or "build inputs")
    monkeypatch.setattr(ownership, "_index", lambda snap, v: INDEXES["a"])
    snapshot = SimpleNamespace(layout=LayoutMap({}, MEMBERS, {}, "d", (), {}), versions={"a": None})
    item = (snapshot, unit("f"), None, False, None)
    monkeypatch.setattr(ownership, "code_identity", lambda *f: "placement code one")
    first = ownership._job_key(item)
    monkeypatch.setattr(ownership, "code_identity", lambda *f: "placement code two")
    assert ownership._job_key(item) != first  # an edit of the placement code moves it
    assert asked[0]["code"] is False and ownership._job_key.coded  # the tool's build code and pool add nothing


def test_a_changed_data_row_relocates_every_unit_from_its_cached_build_without_building_again(monkeypatch, tmp_path):
    spec = unit("f")
    snapshot = SimpleNamespace(config=object(), layout=LayoutMap({}, MEMBERS, {spec.path: spec}, "d", (), {}))
    cache: dict[tuple[str, str], object] = {}

    def cached_object(cfg, kind, key, produce):
        if (kind, key) not in cache:
            cache[kind, key] = produce()
        return cache[kind, key]

    built = Mock(return_value=({".rodata": (word(7), {}, [])}, None))
    monkeypatch.setattr(ownership, "_built", built)
    monkeypatch.setattr(ownership, "_built_key", lambda snap, u, v, recipe: f"built-{v}")
    monkeypatch.setattr(ownership.store, "cached_object", cached_object)
    rows = [index(word(0, 7, 0, 7), [(4, 8, "rodata/a", ".rodata", 0x80000004)])]
    monkeypatch.setattr(ownership, "_index", lambda snap, version: rows[0])
    item = (snapshot, spec, "a", False, object())
    assert ownership._job(item) == {"a": {".rodata": [[4, 8]]}}
    before = ownership._job_key(item)
    rows[0] = index(word(0, 7, 0, 7), [(12, 16, "rodata/b", ".rodata", 0x8000000C)])._replace(digest="moved")
    assert ownership._job_key(item) != before  # the placement is located again on the new rows
    assert ownership._job(item) == {"a": {".rodata": [[12, 16]]}}
    assert built.call_count == 1  # from the object built before: the rows never reach the build


@pytest.mark.parametrize("withheld", [(), ("a",)])
def test_compile_failure_blocks_only_a_previously_built_holder(monkeypatch, inline, withheld):
    spec = unit("f", withheld=withheld)
    snapshot = Snapshot(
        SimpleNamespace(project=SimpleNamespace(names_from="a")),
        "h",
        LayoutMap({}, MEMBERS, {spec.path: spec}, "d", (), {}),
        {"a": None},
        {},
        "compile-debt",
    )
    monkeypatch.setattr(ownership, "_job", lambda item: {"a": "compile.error: undeclared constant"})
    monkeypatch.setattr(ownership, "_index", lambda snapshot, v: INDEXES[v])
    out, claims, findings = ownership.claims(snapshot, units={spec.path: spec})
    assert out[spec.path].withheld == ("a",) and not claims
    assert len(findings) == 1
    assert findings[0].blocking is (not withheld)
    assert findings[0].key == ("layout.ownership" if withheld else "compile.error")
    assert findings[0].missing == ("src/f.c a undeclared constant",)


@pytest.mark.parametrize("gap", ["", "text differs"])
def test_withheld_compile_recovery_still_needs_exactness(monkeypatch, inline, gap):
    spec = unit("f", withheld=("a",))
    snapshot = Snapshot(
        SimpleNamespace(project=SimpleNamespace(names_from="a")),
        "h",
        LayoutMap({}, MEMBERS, {spec.path: spec}, "d", (), {}),
        {"a": None},
        {},
        "recovered",
    )
    monkeypatch.setattr(ownership, "_job", lambda item: {"a": {}})
    monkeypatch.setattr(ownership, "_index", lambda snapshot, v: INDEXES[v])
    assigned, _, findings = ownership.claims(snapshot, units={spec.path: spec})
    assert not findings and assigned[spec.path].withheld == ()
    seen = []
    monkeypatch.setattr(ownership, "_exact_job", lambda item: seen.append(item) or gap)
    out, findings = ownership.exact(snapshot, assigned)
    assert [v for _, _, v in seen] == ["a"]
    assert out[spec.path].withheld == (("a",) if gap else ())
    assert bool(findings) == bool(gap)


@pytest.mark.parametrize(
    "module,name,changes",
    [
        (ownership, "_built", True),
        (ownership.native, "objects", True),
        (ownership.native, "sections", True),
        (ownership, "emitted", True),
        (ownership.pool, "map", False),
        (ownership.effort, "stage", False),
    ],
)
def test_holder_build_key_follows_its_producer_but_not_scheduling(monkeypatch, module, name, changes):
    from types import FunctionType

    monkeypatch.setattr(ownership.native, "stamp", lambda *args, **kwargs: "unchanged native inputs")
    ownership.code_identity.cache_clear()
    before = ownership._built_key(None, None, "a", None)
    original = getattr(module, name)
    while hasattr(original, "__wrapped__"):
        original = original.__wrapped__
    edited = FunctionType(
        original.__code__.replace(co_consts=(*original.__code__.co_consts, "producer edit")),
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    edited.__module__ = original.__module__
    edited.__qualname__ = original.__qualname__
    edited.__kwdefaults__ = original.__kwdefaults__
    monkeypatch.setattr(module, name, edited)
    ownership.code_identity.cache_clear()
    after = ownership._built_key(None, None, "a", None)
    assert (after != before) is changes
    ownership.code_identity.cache_clear()
