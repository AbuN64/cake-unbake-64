"""place.py: the rows that put a C function's jump tables and literal pools in the function's own file."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

from unbake import place, repo
from unbake.contracts import Refusal

pytestmark = pytest.mark.usefixtures("plain_roms")

F, G = "func_80400000_de", "func_80400100_de"
TABLE = (0x800E1500, 0x2000)  # vram, rom
WORDS = (0x00400010, 0x00400018, 0x00400020, 0x00400028)  # a jump table: every case inside F


def _facts(extra=None):
    facts = {
        F: (0x80400000, 0x1000, 0x100, "func"),
        G: (0x80400100, 0x1100, 0x100, "func"),
        "jtbl_800E1500": (TABLE[0], TABLE[1], None, "data"),
        "D_800E1510": (0x800E1510, 0x2010, None, "data"),
    }
    return {**facts, **(extra or {})}


def _tree(owned=(), compiled=(F,), literals=None):
    module = SimpleNamespace(
        path="src/m.c",
        owns={"a": (*compiled, *([literals] if literals else []))},
        stand_ins={"a": ()},
        sections={"a": {**{n: ".text" for n in compiled}, **({literals: ".rodata"} if literals else {})}},
    )
    names = {n: "src/m.c" for n in module.owns["a"]}
    return SimpleNamespace(modules=(module,), owned=lambda v, n: names.get(n))


def _world(monkeypatch, tmp_path, facts=None, words=WORDS, asm=None):
    root = tmp_path / "proj"
    (root / "asm/a/nonmatchings/seg").mkdir(parents=True)
    (root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text(
        "lui $at, %hi(jtbl_800E1500)\nlw $v0, %lo(jtbl_800E1500)($at)\n"
    )
    for name, text in (asm or {}).items():
        (root / "asm/a/nonmatchings/seg" / f"{name}.s").write_text(text)
    rom = bytearray(0x3000)
    rom[0x2000 : 0x2000 + 4 * len(words)] = b"".join(w.to_bytes(4, "big") for w in words)
    (tmp_path / "rom").write_bytes(bytes(rom))
    monkeypatch.setattr(place.api, "symbols", lambda config, v: facts or _facts())
    monkeypatch.setattr(place.api, "asm_path", lambda config, v, n: f"asm/a/nonmatchings/seg/{n}.s")
    monkeypatch.setattr(place.api, "preprocess", lambda config, path, v: "")
    monkeypatch.setattr(place.configuration, "rom", lambda project, v: tmp_path / "rom")
    return SimpleNamespace(project=SimpleNamespace(versions=("a",), root=root))


def test_a_c_functions_jump_table_becomes_its_literals_row(monkeypatch, tmp_path):
    config = _world(monkeypatch, tmp_path)
    (found,) = place.plan(config, _tree(), [F])
    assert found.row == f"literals_{F} = 0x800E1500; // rom:0x2000 size:0x10" and found.file == "versions/a/aliases.txt"
    assert found.names == ("jtbl_800E1500",) and found.path == "src/m.c"


def test_the_literal_pool_a_function_reads_is_placed_too(monkeypatch, tmp_path):
    # no jump table: the words are floats; the function's assembly names the pool, so it is the function's
    facts = _facts({"D_800E1520": (0x800E1520, 0x2020, None, "data")})
    config = _world(monkeypatch, tmp_path, facts=facts, words=(0x3F800000, 0x40000000))
    (root := config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text("lwc1 $f0, %lo(D_800E1520)($at)\n")
    (found,) = place.plan(config, _tree(), [F])
    assert found.row.startswith(f"literals_{F} = 0x800E1520;") and found.names == ("D_800E1520",)
    assert root.is_file()


def test_a_c_reader_of_the_table_keeps_its_extern_and_the_table_is_placed(monkeypatch, tmp_path):
    # G is C and reads the table through `extern char jtbl_800E1500;`: the symbol resolves by address after the move
    config = _world(monkeypatch, tmp_path, asm={G: "lw $v0, %lo(jtbl_800E1500)($at)\n"})
    (found,) = place.plan(config, _tree(compiled=(F, G)), [F])
    assert found.row.startswith(f"literals_{F} = 0x800E1500;") and f"{G} (C: extern)" in found.note


def test_a_table_an_assembly_function_also_reads_is_placed_with_its_first_owner(monkeypatch, tmp_path):
    # G stays assembly: its %lo(jtbl_800E1500) links by the symbol's address, so the table moves with F alone
    config = _world(monkeypatch, tmp_path, asm={G: "lw $v0, %lo(jtbl_800E1500)($at)\n"})
    (found,) = place.plan(config, _tree(), [F])
    assert found.row.startswith(f"literals_{F} = 0x800E1500;") and G in found.note


def test_the_assembly_an_earlier_name_left_behind_is_no_second_reader(monkeypatch, tmp_path):
    # F was old_name once: asm/.../old_name.s is F's own text under a name the facts dropped
    old = "/* 1000 80400000 27BDFE70 */  addiu $sp, $sp, -8\nlw $v0, %lo(jtbl_800E1500)($at)\n"
    config = _world(monkeypatch, tmp_path, asm={"old_name": old})
    (found,) = place.plan(config, _tree(), [F])
    assert found.names == ("jtbl_800E1500",) and "old_name" not in found.note


def test_a_second_name_of_the_same_bytes_counts_as_a_reader(monkeypatch, tmp_path):
    facts = _facts({"D_alias": (TABLE[0], TABLE[1], None, "data")})
    config = _world(monkeypatch, tmp_path, facts=facts, asm={G: "lw $v0, %lo(D_alias)($at)\n"})
    (found,) = place.plan(config, _tree(), [F])
    assert f"{G} (assembly)" in found.note


def test_a_file_that_already_places_its_data_gets_no_row_and_is_no_refusal(monkeypatch, tmp_path):
    facts = _facts({f"literals_{F}": (TABLE[0], TABLE[1], None, "data")})
    config = _world(monkeypatch, tmp_path, facts=facts)
    assert place.plan(config, _tree(literals=f"literals_{F}"), [F]) == []


def test_a_function_with_nothing_to_place_is_refused(monkeypatch, tmp_path):
    config = _world(monkeypatch, tmp_path, words=(0, 0, 0, 0))
    (config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text("jr $ra\n")
    with pytest.raises(Refusal) as refused:
        place.plan(config, _tree(), [F])
    assert refused.value.findings[0].key == "place.none"


def test_assembly_only_versions_are_left_alone(monkeypatch, tmp_path):
    config = _world(monkeypatch, tmp_path)
    with pytest.raises(Refusal) as refused:
        place.plan(config, _tree(compiled=()), [F])  # F is not C in this version: nothing to place anywhere
    assert refused.value.findings[0].key == "place.none"


def _object(path, data, align=4, reloc=()):
    """A big-endian ELF32 with a .rodata (`data`) and a .rel.rodata naming the offsets in `reloc`."""
    shstr = b"\0.rodata\0.rel.rodata\0.shstrtab\0"
    rel = b"".join(struct.pack(">II", off, 2) for off in reloc)
    body = bytearray(52)
    off_ro = len(body)
    body += data
    off_rel = len(body)
    body += rel
    off_str = len(body)
    body += shstr
    shoff = len(body)
    sh = [
        bytes(40),
        struct.pack(">10I", 1, 1, 2, 0, off_ro, len(data), 0, 0, align, 0),
        struct.pack(">10I", 9, 9, 0, 0, off_rel, len(rel), 0, 1, 4, 8),
        struct.pack(">10I", 19, 3, 0, 0, off_str, len(shstr), 0, 0, 1, 0),
    ]
    body += b"".join(sh)
    body[0:16] = b"\x7fELF\x01\x02\x01" + bytes(9)
    struct.pack_into(">HHIIII", body, 16, 1, 8, 1, 0, 0, shoff)
    struct.pack_into(">HHHHHH", body, 0x28, 52, 0, 0, 40, 4, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(body))


def _built(config, data, **kw):
    src = config.project.root / "src/m.c"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("x")
    _object(config.project.root / "build/a/src/m.c.o", data, **kw)


def test_a_built_object_whose_rodata_is_the_roms_proves_the_move(monkeypatch, tmp_path):
    config = _world(monkeypatch, tmp_path)
    data = b"".join(
        w.to_bytes(4, "big") for w in (0, 0x00400018, 0, 0x00400028)
    )  # words 0 and 2 are relocated, the rest is ROM's
    _built(config, data, reloc=(0, 8))
    (found,) = place.plan(config, _tree(), [F])
    assert found.note == "compiled .rodata == ROM bytes"


def test_an_object_whose_rodata_differs_or_is_too_long_refuses(monkeypatch, tmp_path):
    config = _world(monkeypatch, tmp_path)
    _built(config, b"".join(w.to_bytes(4, "big") for w in (0, 0x99, 0, 0x28)), reloc=(0, 8))
    with pytest.raises(Refusal) as refused:
        place.plan(config, _tree(), [F])
    assert (
        refused.value.findings[0].key == "place.object"
        and "differs from the ROM at 0x2004" in refused.value.findings[0].reason
    )
    _built(config, bytes(0x40))
    with pytest.raises(Refusal) as refused:
        place.plan(config, _tree(), [F])
    assert "0x40 bytes" in refused.value.findings[0].reason


def test_a_fresh_object_with_no_rodata_refuses_the_move(monkeypatch, tmp_path):
    config = _world(monkeypatch, tmp_path)
    _built(config, bytes(4))
    obj = config.project.root / "build/a/src/m.c.o"
    obj.write_bytes(b"\x7fELF no sections")  # the C reads the pool through an extern: the object has no .rodata at all
    with pytest.raises(Refusal) as refused:
        place.plan(config, _tree(), [F])
    assert refused.value.findings[0].key == "place.object" and "no .rodata" in refused.value.findings[0].reason


def test_an_eight_aligned_rodata_cannot_sit_on_a_four_aligned_address(monkeypatch, tmp_path):
    facts = _facts({"jtbl_800E1500": (0x800E1504, TABLE[1], None, "data")})
    config = _world(monkeypatch, tmp_path, facts=facts)
    _built(config, b"".join(w.to_bytes(4, "big") for w in (0, 0x00400018, 0, 0x00400028)), align=8, reloc=(0, 8))
    with pytest.raises(Refusal) as refused:
        place.plan(config, _tree(), [F])
    assert "8-aligned" in refused.value.findings[0].reason


def test_a_stale_object_is_not_trusted_the_link_proves_the_bytes(monkeypatch, tmp_path):
    config = _world(monkeypatch, tmp_path)
    _built(config, bytes(0x40))
    import os

    os.utime(config.project.root / "build/a/src/m.c.o", (1, 1))  # older than the source
    (found,) = place.plan(config, _tree(), [F])
    assert "link proves" in found.note


def test_rows_are_added_once_and_keep_the_files_other_lines():
    p = place.Placement("a", F, "src/m.c", TABLE[0], TABLE[1], 0x2010, ("jtbl_800E1500",))
    out = place.rows([p, p], {p.file: "x = 0x80000000;"})
    assert out == {p.file: f"x = 0x80000000;\n{p.row}\n".encode()}
    assert place.rows([p], {p.file: f"{p.row}\n"}) == {}  # already there: nothing to write


def test_sync_place_puts_the_aliases_back_when_the_sync_after_it_refuses(monkeypatch, tmp_path):
    root = tmp_path
    (root / "versions/a").mkdir(parents=True)
    (root / "versions/a/aliases.txt").write_text("x = 0x80000000;\n")
    p = place.Placement("a", F, "src/m.c", TABLE[0], TABLE[1], 0x2010, ("jtbl_800E1500",))
    monkeypatch.setattr(repo.place, "plan", lambda config, current, functions: [p])
    monkeypatch.setattr(repo.tree, "capture", lambda config: None)
    seen = []

    def failing(config, params):
        seen.append((root / p.file).read_text())
        raise Refusal(repo.Finding("sync.literals", "x"))

    monkeypatch.setattr(repo, "sync", failing)
    config = SimpleNamespace(project=SimpleNamespace(root=root))
    with pytest.raises(Refusal):
        repo._sync_placing(config, {"place": [F]})
    assert p.row in seen[0] and (root / p.file).read_text() == "x = 0x80000000;\n"


def test_sync_place_check_lists_the_rows_and_writes_nothing(monkeypatch, tmp_path):
    p = place.Placement("a", F, "src/m.c", TABLE[0], TABLE[1], 0x2010, ("jtbl_800E1500",), "n")
    monkeypatch.setattr(repo.place, "plan", lambda config, current, functions: [p])
    monkeypatch.setattr(repo.tree, "capture", lambda config: None)
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    result = repo._sync_placing(config, {"place": [F], "check": True})
    assert result["placed"] == [{"version": "a", "function": F, "row": p.row, "file": p.file, "note": "n"}]
    assert not (tmp_path / "versions").exists()


def test_bytes_after_the_table_stay_behind_a_stand_in_at_the_top_of_the_file_after_them(monkeypatch, tmp_path):
    # the compiled .rodata of a file is one section: bytes of the old table's piece that it does not hold (here a
    # foreign word) are a stand-in of their own, in the file that starts after them, never in the one before
    config = _world(monkeypatch, tmp_path, words=(*WORDS[:3], 0x00999999))
    (config.project.root / "versions/a").mkdir(parents=True)
    (config.project.root / "versions/a/splat.yaml").write_text("  - [0x2000, rodata, data_x/jtbl_800E1500]\n")
    later = SimpleNamespace(
        path="src/next.c",
        owns={"a": ("D_800E1510",)},
        stand_ins={"a": ("D_800E1510",)},
        sections={"a": {"D_800E1510": ".rodata"}},
    )
    current = _tree()
    current.modules = (*current.modules, later)
    (found,) = place.plan(config, current, [F])
    assert found.row.endswith("rom:0x2000 size:0xC")
    assert found.leftover == f"D_auto_rodata_{F}_800E150C = 0x800E150C; // rom:0x200C"
    assert (found.after, found.segment) == ("src/next.c", "data_x")


def test_the_stand_in_goes_at_the_top_after_the_includes_once():
    p = place.Placement(
        "a", F, "src/m.c", TABLE[0], TABLE[1], 0x200C, ("jtbl_800E1500",), "", 0x2010, "src/next.c", "data_x"
    )
    line = f'INCLUDE_RODATA("data_x", D_auto_rodata_{F}_800E150C);'
    top = f'#include "include_asm.h"\n#if defined(VERSION_A)\n{line}\n#endif\nint a;\n'.encode()
    assert place.sources([p], {"src/next.c": '#include "include_asm.h"\nint a;\n'}) == {"src/next.c": top}
    assert place.sources([p], {"src/next.c": top.decode()}) == {}  # already there
    bare = place.sources([p], {"src/next.c": "int a;\n"})["src/next.c"].decode()
    assert bare.startswith('#include "include_asm.h"\n#if defined(VERSION_A)\n') and bare.endswith("int a;\n")


def test_leftover_bytes_with_no_file_after_them_are_refused(monkeypatch, tmp_path):
    config = _world(monkeypatch, tmp_path, words=(*WORDS[:3], 0x00999999))
    (config.project.root / "versions/a").mkdir(parents=True)
    (config.project.root / "versions/a/splat.yaml").write_text("  - [0x2000, rodata, data_x/jtbl_800E1500]\n")
    with pytest.raises(Refusal) as refused:
        place.plan(config, _tree(), [F])
    assert refused.value.findings[0].key == "place.leftover"


def test_a_file_on_both_sides_of_the_table_is_split_and_the_new_file_takes_the_stand_in(monkeypatch, tmp_path):
    # one object's .rodata is listed at its first byte: a file owning data before the table and after the leftover
    # would pull the leftover in front of the table, so its names from the leftover on move to a sibling
    config = _world(
        monkeypatch,
        tmp_path,
        words=(*WORDS[:3], 0x00999999),
        facts=_facts({"D_800E14F0": (0x800E14F0, 0x1FF0, None, "data")}),
    )
    (config.project.root / "versions/a").mkdir(parents=True)
    (config.project.root / "versions/a/splat.yaml").write_text("  - [0x2000, rodata, data_x/jtbl_800E1500]\n")
    both = SimpleNamespace(
        path="src/d.c",
        owns={"a": ("D_800E14F0", "D_800E1510")},
        stand_ins={"a": ("D_800E14F0", "D_800E1510")},
        sections={"a": {"D_800E14F0": ".rodata", "D_800E1510": ".rodata"}},
    )
    current = _tree()
    current.modules = (*current.modules, both)
    (found,) = place.plan(config, current, [F])
    assert (found.split_from, found.after, found.split) == ("src/d.c", "src/d_2.c", ("D_800E1510",))
    text = (
        '#include "include_asm.h"\n'
        "#if defined(VERSION_A)\nconst unsigned int D_800E14F0[1] = {\n    0x00000000,\n};\n#endif\n"
        '#if defined(VERSION_A)\nINCLUDE_RODATA("data_x", D_800E1510);\n#endif\n'
    )
    out = place.sources([found], {"src/d.c": text})
    kept, new = out["src/d.c"].decode(), out["src/d_2.c"].decode()
    assert "D_800E14F0" in kept and "D_800E1510" not in kept and "D_800E14F0" not in new
    assert kept.count("#if") == kept.count("#endif") == 1  # no empty #if left where D_800E1510 was
    stand_in = f'INCLUDE_RODATA("data_x", D_auto_rodata_{F}_800E150C);'
    assert new.index(stand_in) < new.index("D_800E1510") and new.startswith('#include "include_asm.h"')


def test_a_split_moves_only_the_placed_versions_blobs_and_leaves_the_other_versions_in_place():
    # 10-24 System_RunPiManager: one name defined per version; the split wrapped and moved the wrong version block
    text = (
        '#include "include_asm.h"\n'
        "#ifdef VERSION_DE\nconst unsigned int D_blob[2] = {\n    1, 2,\n};\n#endif\n"
        "#ifdef VERSION_EU_X\nconst unsigned int D_blob[1] = {\n    3,\n};\n#endif\n"
        "#ifdef VERSION_US_REV1\nconst unsigned int D_blob[1] = {\n    4,\n};\n#endif\n"
    )
    ps = [
        place.Placement(
            v,
            F,
            "src/m.c",
            TABLE[0],
            TABLE[1],
            0x200C,
            ("jtbl",),
            "",
            0x2010,
            "src/d_2.c",
            "data_x",
            "src/d.c",
            ("D_blob",),
        )
        for v in ("eu-x", "us-rev1")
    ]
    out = place.sources(ps, {"src/d.c": text})
    kept, new = out["src/d.c"].decode(), out["src/d_2.c"].decode()
    assert "1, 2" in kept and "VERSION_DE" in kept and "3," not in kept and "4," not in kept
    assert "VERSION_DE" not in new and "1, 2" not in new
    assert "3," in new and "4," in new and new.count("#if") == new.count("#endif")


# --- bounds: a row holds only what the function owns (10-24 REV1-ONE-11 / ONE-8B) -----------------------------------
H = "func_80400200_de"  # another function, whose own table follows F's


def test_a_pool_stops_where_another_functions_jump_table_begins(monkeypatch, tmp_path):
    # no splat row for H's table: F's pool row would run on over it (us/eu: "took the next two tables")
    facts = {
        F: (0x80400000, 0x1000, 0x100, "func"),
        H: (0x80400200, 0x1200, 0x100, "func"),
        "D_800E1520": (0x800E1520, 0x2020, None, "data"),
        "D_far": (0x800E1600, 0x2100, None, "data"),
    }
    config = _world(monkeypatch, tmp_path, facts=facts)
    rom = bytearray((tmp_path / "rom").read_bytes())
    pool = (0x3F800000, 0x40000000, 0, 0, 0x00400210, 0x00400218, 0x00400220, 0x00400228)  # 4 own words, H's table
    rom[0x2020 : 0x2020 + 4 * len(pool)] = b"".join(w.to_bytes(4, "big") for w in pool)
    (tmp_path / "rom").write_bytes(bytes(rom))
    (config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text("lwc1 $f0, %lo(D_800E1520)($at)\n")
    (found,) = place.plan(config, _tree(), [F])
    assert found.row.endswith("size:0x10"), found.row  # the pool's 4 words, not H's table behind it
    assert found.end == 0x2030 and found.leftover == ""


def _gap_world(monkeypatch, tmp_path, middle_asm=None):
    # F reads two pools with an unread row (D_800E1530) between them: one literals_ row would hold its bytes
    facts = _facts(
        {
            "D_800E1520": (0x800E1520, 0x2020, None, "data"),
            "D_800E1530": (0x800E1530, 0x2030, None, "data"),
            "D_800E1540": (0x800E1540, 0x2040, None, "data"),
        }
    )
    config = _world(monkeypatch, tmp_path, facts=facts, words=(0x3F800000,) * 4, asm=middle_asm)
    (config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text(
        "lwc1 $f0, %lo(D_800E1520)($at)\nlwc1 $f2, %lo(D_800E1540)($at)\n"
    )
    return config


def test_an_unread_unowned_row_between_the_functions_pools_is_junk_to_remove(monkeypatch, tmp_path):
    config = _gap_world(monkeypatch, tmp_path)
    (found,) = place.plan(config, _tree(), [F])
    assert found.junk == (("D_800E1530", 0x2030),) and found.names[0] == "D_800E1520"
    texts = {
        "versions/a/symbol_addrs.txt": (
            "D_800E1520 = 0x800E1520;\nD_800E1530 = 0x800E1530; // junk\nD_800E1540 = 0x800E1540;\n"
        ),
        "versions/a/aliases.txt": "D_800E1530x = 1;\nD_800E1530 = 0x800E1530;\n",
    }
    out = {k: v.decode() for k, v in place.rows([found], texts).items()}
    assert out["versions/a/symbol_addrs.txt"] == "D_800E1520 = 0x800E1520;\nD_800E1540 = 0x800E1540;\n"
    assert (
        out["versions/a/aliases.txt"].startswith("D_800E1530x = 1;\n")
        and "D_800E1530 =" not in out["versions/a/aliases.txt"]
    )
    assert found.row in out["versions/a/aliases.txt"]


def test_a_row_between_the_pools_that_other_code_reads_stays_a_refusal(monkeypatch, tmp_path):
    config = _gap_world(monkeypatch, tmp_path, middle_asm={G: "lwc1 $f4, %lo(D_800E1530)($at)\n"})
    with pytest.raises(Refusal) as refused:
        place.plan(config, _tree(), [F])
    assert "place.gap" in {f.key for f in refused.value.findings} and "D_800E1530" in str(refused.value)


def test_a_zero_row_between_the_pools_that_a_file_defines_is_a_pad_to_drop_from_it(monkeypatch, tmp_path):
    # the compiled .rodata holds the pad's bytes: its zero array and its symbol row go (rodata_unresolved_* arrays)
    config = _gap_world(monkeypatch, tmp_path)
    tree = _tree()
    tree.owned = lambda v, n: "src/d.c" if n in (F, "D_800E1530") and n != F else ("src/m.c" if n == F else None)
    (found,) = place.plan(config, tree, [F])
    assert found.junk == (("D_800E1530", 0x2030),) and found.pads == (("D_800E1530", "src/d.c"),)
    text = (
        '#include "types.h"\n'
        "const unsigned int D_800E1520[4] = {1, 2, 3, 4};\n"
        "const unsigned int D_800E1530[4] = {\n    0,\n    0,\n    0,\n    0,\n};\n"
        "const unsigned int D_800E1540[4] = {5, 6, 7, 8};\n"
    )
    kept = place.sources([found], {"src/d.c": text})["src/d.c"].decode()
    assert "D_800E1530" not in kept and "D_800E1520" in kept and "D_800E1540" in kept
    again = place.sources([found, found], {"src/d.c": text})["src/d.c"].decode()  # a second version: already gone
    assert again == kept


def test_a_row_between_the_pools_that_a_file_owns_stays_a_refusal(monkeypatch, tmp_path):
    config = _gap_world(monkeypatch, tmp_path)
    rom = bytearray((tmp_path / "rom").read_bytes())
    rom[0x2030:0x2034] = b"\x00\x00\x00\x01"  # not a pad: it holds data
    (tmp_path / "rom").write_bytes(bytes(rom))
    tree = _tree()
    tree.owned = lambda v, n: "src/m.c" if n in (F, "D_800E1530") else None
    with pytest.raises(Refusal) as refused:
        place.plan(config, tree, [F])
    assert "place.gap" in {f.key for f in refused.value.findings}


def test_a_row_read_only_past_an_offset_starts_there_not_at_the_foreign_blob_before_it(monkeypatch, tmp_path):
    # de: the table is reached as `D_blob + 0xC`; the 12 bytes before it are not F's
    facts = {
        F: (0x80400000, 0x1000, 0x100, "func"),
        G: (0x80400100, 0x1100, 0x100, "func"),
        "D_blob": (0x800E14F4, 0x1FF4, None, "data"),
        "D_next": (0x800E1510, 0x2010, None, "data"),
    }
    config = _world(monkeypatch, tmp_path, facts=facts)
    words = (0xDEAD0000, 0xBEEF0000, 0x12345678, *WORDS)
    rom = bytearray((tmp_path / "rom").read_bytes())
    rom[0x1FF4:0x2010] = b"".join(w.to_bytes(4, "big") for w in words)
    (tmp_path / "rom").write_bytes(bytes(rom))
    (config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text("lw $v0, %lo(D_blob + 0xC)($at)\n")
    (found,) = place.plan(config, _tree(), [F])
    assert found.row == f"literals_{F} = 0x800E1500; // rom:0x2000 size:0x10"


def test_a_file_with_another_functions_row_still_places_this_function(monkeypatch, tmp_path):
    # DispatchCurrentScreenEvent: the file placed DispatchElementEvent's table; the second function got no row and rc 0
    facts = _facts({f"literals_{G}": (0x800E1580, 0x2080, None, "data")})
    config = _world(monkeypatch, tmp_path, facts=facts)
    (found,) = place.plan(config, _tree(compiled=(F, G), literals=f"literals_{G}"), [F])
    assert found.function == F and found.row.startswith(f"literals_{F} = 0x800E1500;")


def test_nothing_placed_in_any_version_is_a_refusal_not_a_silent_success(monkeypatch, tmp_path):
    config = _world(monkeypatch, tmp_path, words=(0, 0, 0, 0))
    (config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text("jr $ra\n")
    with pytest.raises(Refusal) as refused:
        place.plan(config, _tree(compiled=(F, G), literals=f"literals_{G}"), [F])
    assert refused.value.findings[0].key == "place.none"


def test_alignment_padding_before_a_jump_table_is_not_part_of_the_row(monkeypatch, tmp_path):
    """10-24 de: the row began at the 3 pad words before the 16-aligned table (0xC26E4 size 0x2C, wanted 0xC26F0 size 0x20)."""  # noqa: E501
    facts = _facts()
    facts.pop("jtbl_800E1500")
    facts["jtbl_800E14F4"] = (0x800E14F4, 0x1FF4, None, "data")
    config = _world(monkeypatch, tmp_path, facts=facts)
    (config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text("lw $v0, %lo(jtbl_800E14F4)($at)\n")
    (found,) = place.plan(config, _tree(), [F])
    assert found.row == f"literals_{F} = 0x800E1500; // rom:0x2000 size:0x10"


def test_a_data_file_on_both_sides_of_a_pool_with_nothing_left_over_is_split_at_the_pool(monkeypatch, tmp_path):
    # REV1-ONE-21: the pool sat mid-unit and the bytes after it start at the next row; the unit's names from there on
    # move to a sibling, so the link order puts the function's pool between the two halves
    config = _world(monkeypatch, tmp_path, facts=_facts({"D_800E14F0": (0x800E14F0, 0x1FF0, None, "data")}))
    both = SimpleNamespace(
        path="src/d.c",
        owns={"a": ("D_800E14F0", "D_800E1510")},
        stand_ins={"a": ("D_800E14F0", "D_800E1510")},
        sections={"a": {"D_800E14F0": ".rodata", "D_800E1510": ".rodata"}},
    )
    current = _tree()
    current.modules = (*current.modules, both)
    (found,) = place.plan(config, current, [F])
    assert found.leftover == "" and (found.split_from, found.after, found.split) == (
        "src/d.c",
        "src/d_2.c",
        ("D_800E1510",),
    )
    text = (
        '#include "include_asm.h"\n'
        "#if defined(VERSION_A)\nconst unsigned int D_800E14F0[1] = {\n    0x00000000,\n};\n#endif\n"
        '#if defined(VERSION_A)\nINCLUDE_RODATA("data_x", D_800E1510);\n#endif\n'
    )
    out = place.sources([found], {"src/d.c": text})
    kept, new = out["src/d.c"].decode(), out["src/d_2.c"].decode()
    assert "D_800E14F0" in kept and "D_800E1510" not in kept and "D_800E1510" in new
    assert "INCLUDE_RODATA" in new and "D_auto_rodata" not in new


def test_a_piece_shared_with_the_next_functions_pool_is_split_not_cut(monkeypatch, tmp_path):
    """REV1-ONE-40 01:3x: one 18-word array held Nav_PrepareProbeSegments' 10 words and func_80210EFC_de's 8-word pool;
    the 0x28 row superseded the whole piece and the 8 words vanished (every later address 0x20 off). No version
    splits it: the rest gets a row of its own named for the function and its address."""
    facts = _facts(
        {
            "D_800E1500": (0x800E1500, 0x2000, None, "data"),
            "D_800E1504": (0x800E1504, 0x2004, None, "data"),
            "D_800E1508": (0x800E1508, 0x2008, None, "data"),
            "D_800E150C": (0x800E150C, 0x200C, None, "data"),
        }
    )
    del facts["jtbl_800E1500"]
    floats = (0x3F800000, 0x40000000, 0x40400000, 0x40800000)  # literals, no jump targets
    config = _world(
        monkeypatch,
        tmp_path,
        facts=facts,
        words=floats,
        asm={G: "lui $at, %hi(D_800E1508)\nlwc1 $f0, %lo(D_800E1508)($at)\nlwc1 $f2, %lo(D_800E150C)($at)\n"},
    )
    (config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text(
        "lui $at, %hi(D_800E1500)\nlwc1 $f0, %lo(D_800E1500)($at)\nlwc1 $f2, %lo(D_800E1504)($at)\n"
    )
    (config.project.root / "versions/a").mkdir(parents=True)
    (config.project.root / "versions/a/splat.yaml").write_text(
        "  - [0x2000, rodata, data_x/D_blob_shared]\n  - [0x2010, rodata, data_x/D_800E1510]\n"
    )
    (found,) = place.plan(config, _tree(), [F])
    assert found.tail == f"D_auto_rodata_{F}_800E1508 = 0x800E1508; // rom:0x2008"


def test_a_piece_that_ends_with_the_functions_data_is_no_remainder(monkeypatch, tmp_path):
    assert place._piece_tail([(0x2000, "rodata", "d/x"), (0x2008, "rodata", "d/y")], 0x2000, 0x2008) is None
    assert place._piece_tail([(0x2000, "rodata", "d/literals_f"), (0x2010, "rodata", "d/y")], 0x2000, 0x2008) is None
    got = place._piece_tail([(0x2000, "rodata", "d/x"), (0x2010, "rodata", "d/y")], 0x2000, 0x2008)
    assert got == (0x2000, 0x2010, "d/x")


def test_a_shared_piece_is_split_at_the_functions_data_as_the_reference_version_names_it(monkeypatch, tmp_path):
    """REV1-ONE-40: one version already holds the next function's pool as its own piece
    (D_auto_src_rw_pool_<owner>_...);
    the versions that hold both in one blob get a row of that name for the rest instead of a refusal."""
    facts = _facts(
        {
            "D_800E1500": (0x800E1500, 0x2000, None, "data"),
            "D_800E1504": (0x800E1504, 0x2004, None, "data"),
            "D_800E1508": (0x800E1508, 0x2008, None, "data"),
            "D_800E150C": (0x800E150C, 0x200C, None, "data"),
        }
    )
    del facts["jtbl_800E1500"]
    floats = (0x3F800000, 0x40000000, 0x40400000, 0x40800000)
    config = _world(
        monkeypatch,
        tmp_path,
        facts=facts,
        words=floats,
        asm={G: "lui $at, %hi(D_800E1508)\nlwc1 $f0, %lo(D_800E1508)($at)\nlwc1 $f2, %lo(D_800E150C)($at)\n"},
    )
    (config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text(
        "lui $at, %hi(D_800E1500)\nlwc1 $f0, %lo(D_800E1500)($at)\nlwc1 $f2, %lo(D_800E1504)($at)\n"
    )
    config.project.versions = ("ref", "b-x")
    ref = (
        "  - [0x2000, rodata, d/D_auto_src_rw_pool_80400100_2000_ref]\n"
        "  - [0x2008, rodata, d/D_auto_src_rw_pool_80400100_2008_ref]\n  - [0x2010, rodata, d/D_800E1510]\n"
    )
    for v, yaml in (
        ("ref", ref),
        ("b-x", "  - [0x2000, rodata, d/D_blob_shared]\n  - [0x2010, rodata, d/D_800E1510]\n"),
    ):
        (config.project.root / f"versions/{v}").mkdir(parents=True)
        (config.project.root / f"versions/{v}/splat.yaml").write_text(yaml)
    current = _tree()
    module = current.modules[0]
    for v in config.project.versions:
        module.owns[v], module.stand_ins[v], module.sections[v] = (F,), (), {F: ".text"}
    found = place.plan(config, current, [F])
    tails = {p.version: p.tail for p in found}
    assert tails["ref"] == ""
    assert tails["b-x"] == "D_auto_src_rw_pool_80400100_2008_b_x = 0x800E1508;"
    written = place.rows(found, {})
    assert "D_auto_src_rw_pool_80400100_2008_b_x = 0x800E1508;" in written["versions/b-x/aliases.txt"].decode()
    assert "rw_pool" not in written["versions/ref/aliases.txt"].decode()


def test_a_far_constant_other_functions_read_stays_a_shared_row_not_part_of_the_data(monkeypatch, tmp_path):
    # F reads D_800E1520 and D_800E1540; G reads D_800E1530 and D_800E1540 too: the shared 1540 is no part of F
    config = _gap_world(
        monkeypatch, tmp_path, middle_asm={G: "lwc1 $f4, %lo(D_800E1530)($at)\nlwc1 $f6, %lo(D_800E1540)($at)\n"}
    )
    (found,) = place.plan(config, _tree(), [F])
    assert found.names == ("D_800E1520",) and found.end == 0x2030


def test_a_row_the_functions_c_declares_extern_is_read_not_placed(monkeypatch, tmp_path):
    # g_name[18] lives in another file; F's pool would take 16 of its 18 bytes and leave "k\0" to the next piece
    facts = _facts({"g_name": (0x800E1520, 0x2020, None, "data"), "D_800E1530": (0x800E1530, 0x2030, None, "data")})
    config = _world(monkeypatch, tmp_path, facts=facts, words=(0x3F800000, 0x40000000))
    (config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text("lw $a3, %lo(g_name)($at)\n")
    monkeypatch.setattr(place.api, "preprocess", lambda config, path, v: "extern u8 g_name[];\nvoid f(void);\n")
    with pytest.raises(Refusal) as refused:
        place.plan(config, _tree(), [F])
    assert {f.key for f in refused.value.findings} == {"place.none"}  # nothing of F's own: widen tolerates this


def test_a_row_the_functions_c_does_not_declare_extern_is_still_placed(monkeypatch, tmp_path):
    facts = _facts({"D_800E1520": (0x800E1520, 0x2020, None, "data")})
    config = _world(monkeypatch, tmp_path, facts=facts, words=(0x3F800000, 0x40000000))
    (config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text("lw $a3, %lo(D_800E1520)($at)\n")
    monkeypatch.setattr(place.api, "preprocess", lambda config, path, v: "extern u8 D_800E1530[];\n")
    (found,) = place.plan(config, _tree(), [F])
    assert found.names == ("D_800E1520",)


def test_a_jump_table_cut_by_facts_symbols_inside_it_is_one_row(monkeypatch, tmp_path):
    """J007 eu: symbols on words 2 and 4 of a 4-word table sized the pool 0x4; the table is the function's 0x10."""
    facts = _facts({"D_800E1504": (0x800E1504, 0x2004, None, "data"), "D_800E150C": (0x800E150C, 0x200C, None, "data")})
    config = _world(monkeypatch, tmp_path, facts=facts)
    (found,) = place.plan(config, _tree(), [F])
    assert found.row == f"literals_{F} = 0x800E1500; // rom:0x2000 size:0x10"
    assert {n for n, _ in found.junk} == {"D_800E1504", "D_800E150C"}


def test_a_function_reading_two_places_far_apart_is_refused_naming_both(monkeypatch, tmp_path):
    """J028 eu: a jump table and a double 0x5000 bytes after it were reported as one span with unread rows inside."""
    facts = _facts(
        {"D_800E1514": (0x800E1514, 0x2014, None, "data"), "far_800E6600": (0x800E6600, 0x7100, None, "data")}
    )
    config = _world(monkeypatch, tmp_path, facts=facts, asm={G: "lw $v0, %lo(D_800E1514)($at)\n"})
    (config.project.root / "asm/a/nonmatchings/seg" / f"{F}.s").write_text(
        "lw $v0, %lo(jtbl_800E1500)($at)\nldc1 $f0, %lo(far_800E6600)($at)\n"
    )
    (tmp_path / "rom").write_bytes((tmp_path / "rom").read_bytes().ljust(0x8000, b"\0"))
    with pytest.raises(Refusal) as refused:
        place.plan(config, _tree(), [F])
    assert refused.value.findings[0].key == "place.apart"
    assert "bytes apart: two places, not one pool" in refused.value.findings[0].reason
