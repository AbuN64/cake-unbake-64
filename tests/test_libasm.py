"""Library assembly survives source-tree discovery, setup extraction, binding and measured report credit."""

from types import SimpleNamespace as N

import fixture
import pytest

from unbake import adapters, libasm, native, policy, recipes, repo, report, symbols, tree, view
from unbake.contracts import Refusal, sha1

pytestmark = pytest.mark.usefixtures("toolchains")
WORDS = (0x40026000, 0x03E00008, 0, 0)


@pytest.fixture
def library(tmp_path, monkeypatch):
    facts = "f = 0x80000400; // type:func rom:0x1000 size:0x10\n"
    cfg = fixture.config(
        tmp_path, sources={"src/sdk/f.c": 'INCLUDE_ASM("main", f);\n'}, symbols={v: facts for v in ("a", "b")}
    )
    cfg.project.libraries = ("src/sdk",)
    for version in cfg.project.versions:
        path = cfg.project.root / f"roms/baserom.{version}.z64"
        data = bytearray(path.read_bytes())
        data[0x1000:0x1010] = b"".join(w.to_bytes(4, "big") for w in WORDS)
        path.write_bytes(data)
        cfg.project.version[version].sha1 = sha1(data)
    monkeypatch.setattr(repo, "_generated", lambda config, held: {})
    return cfg


def test_sync_extracts_library_assembly_and_binds_and_credits_it(library):
    result = repo.sync(library, {})
    path = library.project.root / "src/sdk/f.s"
    assert "src/sdk/f.s" in result["written"]
    assert "mfc0" in path.read_text() and "$12" in path.read_text()
    assert "INCLUDE_ASM" not in (library.project.root / "src/sdk/f.c").read_text()
    held = tree.capture(library)
    assert held.owner("a", "f") == held.owner("b", "f") == "src/sdk/f.s"
    module = next(m for m in held.modules if m.path.endswith(".s"))
    assert recipes.unit(library, module).kind == "asm"
    assert module.library and not module.stand_ins
    credits = report.build_credits(
        list(library.project.versions), held, held.symbols, frozenset({"a", "b"}), {}, frozenset()
    )
    assert all(r.exact and r.kept == 16 for r in credits)
    assert repo.sync(library, {})["written"] == []


def test_nonlibrary_source_stays_policy_debt_and_gets_no_matched_credit(library):
    repo.sync(library, {})
    library.project.libraries = ()
    library.project.digest = "outside-libraries"
    held = tree.capture(library)
    credits = report.build_credits(["a"], held, held.symbols, frozenset({"a"}), {}, frozenset())
    assert all(not r.exact and not r.kept for r in credits)
    path = "src/sdk/f.s"
    findings = policy.evaluate(N(config=library), path, (library.project.root / path).read_text(), None, False)
    assert [f.key for f in findings] == ["source.library-asm"]
    assert policy.text_findings(path, "", False)[0].key == "source.library-asm"


def _patch(cfg, version, at, words):
    path = cfg.project.root / f"roms/baserom.{version}.z64"
    data = bytearray(path.read_bytes())
    data[at : at + 4 * len(words)] = b"".join(w.to_bytes(4, "big") for w in words)
    path.write_bytes(data)
    cfg.project.version[version].sha1 = sha1(data)


def test_normal_c_instructions_are_not_extracted(library):
    assert not libasm.foreign([0x24020001, 0x03E00008, 0], 0, adapters.row("gcc-test"))
    _patch(library, "b", 0x1000, [0x24020001])  # version b holds compiler code there: not foreign in every version
    assert libasm.plan(library, tree.capture(library)) == {}


def test_version_split_is_one_source_with_version_conditionals(library):
    _patch(library, "b", 0x1000, [0x40026800])  # mfc0 $v0, $13 in b where a reads $12
    text = libasm.plan(library, tree.capture(library))["src/sdk/f.s"].decode()
    assert ".if VERSION_B\n    mfc0 $v0, $13\n.else\n    mfc0 $v0, $12\n.endif\n" in text
    renders = {v: libasm.render("f", w, 0x80000400, {}) for v, w in (("a", WORDS), ("b", (0x40026800, *WORDS[1:])))}
    assert {v: libasm.select(text, v, ["a", "b"]) for v in "ab"} == renders


def test_existing_source_is_not_overwritten(library):
    path = library.project.root / "src/sdk/f.s"
    path.write_text("# hand maintained\n")
    with pytest.raises(Refusal, match=r"library-asm\.source"):
        libasm.plan(library, tree.capture(library))
    assert path.read_text() == "# hand maintained\n"


def test_assembly_objects_use_the_assembler_not_the_c_compiler(library, monkeypatch, tmp_path):
    repo.sync(library, {})
    held = tree.capture(library)
    module = next(m for m in held.modules if m.path.endswith(".s"))
    unit = recipes.unit(library, module)
    recipe = recipes.resolve(library, unit, {})
    snapshot = N(config=library, read=lambda path: (library.project.root / path).read_bytes())
    calls = []

    flags = []

    def assemble(source, recipe, includes, out):
        calls.append(source.read_text())
        flags.append(recipe.asflags)
        out.write_bytes(b"object")
        return fixture.native_result(outputs={"object": out})

    monkeypatch.setattr(adapters, "GnuAssembler", lambda cfg: N(assemble=assemble))
    obj, results = native.objects(snapshot, unit, "a", recipe, tmp_path / "work")
    assert obj.read_bytes() == b"object" and len(results) == 1 and "mfc0" in calls[0]
    assert flags[0][-4:] == ("--defsym", "VERSION_A=1", "--defsym", "VERSION_B=0")  # what the Makefile's ASDEFS passes
    assert view.get(snapshot, unit, "a", recipe).text == calls[0]
    assert native.warnings(snapshot, unit, "a", recipe, tmp_path / "work") == ()
    assert native.recorded(snapshot, unit, "a", recipe) == ()


def test_renderer_uses_local_branch_labels_and_preserves_wide_registers():
    text = libasm.render("f", [0x1000FFFF, 0], 0x80001000, {})
    assert ".Lf_000" in text and libasm.scan(text) == {"f": ".text"}
    assert ".set gp=64" in libasm.render("f", [0xDFBF0000, 0x03E00008, 0], 0x80001000, {})


def _boot_entry(tmp_path, marked):
    """sdk/boot/entry.s of the RageWars data slice: a LEAF unit outside every configured library, in the asm home."""
    cfg = fixture.config(
        tmp_path,
        sources={"src/boot/entry.s": "#include <sys/asm.h>\nLEAF(f)\n    mfc0 $v0, $12\n    jr $ra\nEND(f)\n"},
        symbols={v: "f = 0x80000400; // type:func rom:0x1000 size:0x10\n" for v in ("a", "b")},
    )
    cfg.project.cheat = {"asm_home": ["src/boot/"]}
    for version in cfg.project.versions:
        path = cfg.project.root / symbols.asm_path(cfg, version, "f")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(("/* Handwritten function */\n" if marked else "") + "glabel f\n")
    return cfg


def test_handwritten_unit_in_the_asm_home_outside_every_library_earns_exact_credit(tmp_path):
    cfg = _boot_entry(tmp_path, marked=True)
    held = tree.capture(cfg)
    assert held.owner("a", "f") == "src/boot/entry.s" and not cfg.project.libraries
    credits = report.build_credits(
        list(cfg.project.versions), held, held.symbols, frozenset({"a", "b"}), {}, frozenset()
    )
    assert all(r.exact and r.kept == 16 for r in credits)


def test_unmarked_unit_in_the_asm_home_outside_every_library_stays_without_credit(tmp_path):
    cfg = _boot_entry(tmp_path, marked=False)
    held = tree.capture(cfg)
    credits = report.build_credits(
        list(cfg.project.versions), held, held.symbols, frozenset({"a", "b"}), {}, frozenset()
    )
    assert all(not r.exact and not r.kept for r in credits)


def test_scan_reads_the_entry_macros_of_sys_asm_h_as_global_definitions():
    text = (
        "#include <sys/asm.h>\n.set noreorder\n.section .text\n"
        "LEAF(Decompress_ExpandBackReferences)\n    addiu $t5, $zero, 0x0\n.Lloop:\n    b .Lloop\n"
        "END(Decompress_ExpandBackReferences)\nXLEAF(alias)\nNESTED(nested, 24, ra)\n"
        "# LEAF(commented)\n/* LEAF(c2) */\n"
    )
    assert libasm.scan(text) == {"Decompress_ExpandBackReferences": ".text", "alias": ".text", "nested": ".text"}


def test_renderer_reads_a_target_inside_another_routine_as_an_offset_from_it():
    jump, branch = 0x08000106, 0x0741FFFE  # j 0x80000418; bgez $k0 back to 0x800003FC (before every function)
    functions = {0x80000410: "g"}
    assert "j g+0x8" in libasm.render("f", [jump, 0], 0x80001000, {}, functions)
    assert libasm.render("f", [jump, 0], 0x80001000, {}) is None  # no function to name it from
    assert libasm.render("f", [branch, 0], 0x80000400, {}, functions) is None


def test_renderer_pairs_hi_and_lo_through_an_index_add():
    words = [0x3C01800C, 0x00280821, 0x94227A70, 0x03E00008, 0]  # lui $at; addu $at, $at, $t0; lhu $v0, lo($at)
    text = libasm.render("f", words, 0x80001000, {0x800C7A70: "table"})
    assert "%hi(table)" in text and "%lo(table)($at)" in text


def test_merge_keeps_shared_lines_once_and_gives_every_version_its_render():
    a = "x\ny\nz\n"
    texts = {"a": a, "b": "x\nY\nz\n", "c": "x\nz\n", "d": a}
    text = libasm.merge(texts)
    assert text == "x\n.if VERSION_A || VERSION_D\ny\n.elseif VERSION_B\nY\n.endif\nz\n"
    assert all(libasm.select(text, v, list(texts)) == t for v, t in texts.items())
    assert libasm.merge({"a": a, "b": a}) == a


def test_assembly_unit_yields_a_version_whose_bytes_at_its_address_another_source_owns(tmp_path):
    shared = "f = 0x80000400; // type:func rom:0x1000 size:0x10\n"
    cfg = fixture.config(
        tmp_path,
        sources={
            "src/sdk/f.s": ".globl f\nf:\n    jr $ra\n    nop\n",
            "src/x.c": '#if defined(VERSION_B)\nINCLUDE_DATA("main", d);\n#endif\n',
        },
        symbols={"a": shared, "b": shared + "d = 0x80000400; // rom:0x1000 size:0x10\n"},
    )
    cfg.project.libraries = ("src/sdk",)
    held = tree.capture(cfg)
    assert held.owner("a", "f") == "src/sdk/f.s"
    assert held.owned("b", "f") is None and held.owner("b", "d") == "src/x.c"


def test_versions_whose_stand_in_lives_in_another_file_share_one_source(library):
    other = (0x40026000, 0x03E00008, 0x24020001, 0)  # the same routine name with other bytes in version b
    path = library.project.root / "roms/baserom.b.z64"
    data = bytearray(path.read_bytes())
    data[0x1000:0x1010] = b"".join(w.to_bytes(4, "big") for w in other)
    path.write_bytes(data)
    library.project.version["b"].sha1 = sha1(data)
    sdk = library.project.root / "src/sdk"
    (sdk / "f.c").write_text('#if defined(VERSION_A)\nINCLUDE_ASM("main", f);\n#endif\n')
    (sdk / "g.c").write_text('#if defined(VERSION_B)\nINCLUDE_ASM("main", f);\n#endif\n')
    library.project.digest = "split-owners"
    text = libasm.plan(library, tree.capture(library))["src/sdk/f.s"].decode()
    assert ".if VERSION_B\n    addiu $v0, $zero, 0x1\n.else\n    nop\n.endif\n" in text


@pytest.fixture
def lost(tmp_path, monkeypatch):
    """The library's stand-in line for `f` is gone and a game file owns the address before it: nothing owns `f`."""
    facts = (
        "keep = 0x80000420; // type:func rom:0x1020 size:0x10\n"
        "f = 0x80000410; // type:func rom:0x1010 size:0x10\n"
        "g = 0x80000400; // type:func rom:0x1000 size:0x10\n"
    )
    cfg = fixture.config(
        tmp_path,
        sources={"src/sdk/keep.c": 'INCLUDE_ASM("main", keep);\n', "src/game.c": 'INCLUDE_ASM("main", g);\n'},
        symbols={v: facts for v in ("a", "b")},
    )
    cfg.project.libraries = ("src/sdk",)
    for version in cfg.project.versions:
        path = cfg.project.root / f"roms/baserom.{version}.z64"
        data = bytearray(path.read_bytes())
        data[0x1010:0x1020] = b"".join(w.to_bytes(4, "big") for w in WORDS)
        path.write_bytes(data)
        cfg.project.version[version].sha1 = sha1(data)
    monkeypatch.setattr(repo, "_generated", lambda config, held: {})
    return cfg


def test_lost_library_routine_is_rendered_again_and_never_placed_in_a_game_file(lost):
    result = repo.sync(lost, {})
    root = lost.project.root
    assert "src/sdk/f.s" in result["written"]
    assert '("main", f)' not in (root / "src/game.c").read_text()


def test_lost_library_routine_that_cannot_render_keeps_its_stand_in_in_the_library(lost):
    _patch(lost, "b", 0x1014, [0x08000000])  # b jumps below every symbol: nothing names the target
    repo.sync(lost, {})
    root = lost.project.root
    assert '("main", f)' not in (root / "src/game.c").read_text()
    assert 'INCLUDE_ASM("main", f);' in (root / "src/sdk/keep.c").read_text()


def _without_roms(cfg):
    for version in cfg.project.versions:
        (cfg.project.root / f"roms/baserom.{version}.z64").unlink()


def test_sync_check_of_an_extracted_tree_opens_no_rom(library):
    repo.sync(library, {})
    _without_roms(library)
    assert repo.sync(library, {"check": True})["written"] == []


def test_stand_in_nothing_shows_foreign_stays_without_roms(library):
    _without_roms(library)
    assert libasm.plan(library, tree.capture(library)) == {}


def test_lost_library_routine_without_roms_refuses_naming_it(lost):
    _without_roms(lost)
    with pytest.raises(Refusal, match=r"library-asm\.rom") as raised:
        libasm.homes(lost, tree.capture(lost))
    assert "`f`" in raised.value.findings[0].reason


def test_report_counts_matched_library_assembly(library, monkeypatch):
    repo.sync(library, {})
    held = tree.capture(library)
    rows = report.build_credits(
        list(library.project.versions), held, held.symbols, frozenset({"a", "b"}), {}, frozenset()
    )
    monkeypatch.setattr(report, "credits", lambda config, tree: rows)
    totals = report._tallies(library, held)[0]
    assert totals["a"]["code_asm"] == totals["a"]["code_matched"] == 16 and totals["a"]["functions_asm"] == 1
    assert report._line("code", totals["a"], 20, "code").endswith("16 of 16")


@pytest.mark.parametrize("toolchain", ["gcc-2.7.2-kmc", "gcc-2.8.1-sn64"])
def test_gcc_never_emits_trapping_arithmetic(toolchain):
    import tomllib
    from pathlib import Path

    real = (Path(libasm.__file__).parent / "resources/data/toolchains.toml").read_text()  # the fixture fakes the rows
    row = tomllib.loads(real)["toolchain"][toolchain]
    assert libasm.foreign([0x23BDFFD8, 0x03E00008, 0], 0, row)  # addi $sp, $sp, -0x28: hand-written frame
    assert libasm.foreign([0x01E97822, 0x03E00008, 0], 0, row)  # sub $t7, $t7, $t1
    assert not libasm.foreign([0x27BDFFD8, 0x03E00008, 0], 0, row)  # addiu: what GCC writes


def test_filled_load_delay_slot_is_foreign_only_where_loads_interlock():
    row = adapters.row("gcc-test")  # isa [3]
    pop = [0x8C820000, 0, 0x8C590000, 0xAC400000, 0x03E00008, 0xAC990000]  # lw v0; nop; lw t9,0(v0): MIPS I slot
    assert libasm.foreign(pop, 0x80001000, row)
    assert not libasm.foreign(pop, 0x80001000, {**row, "isa": [1, 2]})  # a MIPS I compiler fills it too
    interlocked = [0x8C820000, 0x8C430000, 0xAC400000, 0x03E00008, 0xAC830000]  # what GCC writes for the same pop
    assert not libasm.foreign(interlocked, 0x80001000, row)
    unrelated = [0x8C820000, 0, 0x8C830004, 0x03E00008, 0]  # the instruction after the nop does not read v0
    assert not libasm.foreign(unrelated, 0x80001000, row)
    hilo = [0x00001012, 0x806500C7, 0, 0x00A50018, 0x03E00008, 0]  # mflo v0; lb a1; nop; mult a1,a1: HI/LO hazard
    assert not libasm.foreign(hilo, 0x80001000, row)


def test_routines_the_versions_name_apart_share_one_source_named_without_the_suffix(tmp_path, monkeypatch):
    words = {"a": (0x40026000, 0x0C000140, 0, 0x03E00008, 0), "b": (0x40026000, 0x0C000150, 0, 0x03E00008, 0)}
    facts = {
        "a": "f_a = 0x80000400; // type:func rom:0x1000 size:0x14\n"
        "x_a = 0x80000500; // type:func rom:0x1100 size:0x4\n",
        "b": "f_b = 0x80000400; // type:func rom:0x1000 size:0x14\n"
        "x_b = 0x80000540; // type:func rom:0x1100 size:0x4\n",
    }
    cfg = fixture.config(
        tmp_path,
        sources={
            "src/sdk/f.c": '#if defined(VERSION_A)\nINCLUDE_ASM("main", f_a);\n#endif\n'
            '#if defined(VERSION_B)\nINCLUDE_ASM("main", f_b);\n#endif\n',
            "src/game.c": 'INCLUDE_ASM("main", x_a);\nINCLUDE_ASM("main", x_b);\n',
        },
        symbols=facts,
    )
    cfg.project.libraries = ("src/sdk",)
    for version in cfg.project.versions:
        _patch(cfg, version, 0x1000, words[version])
    monkeypatch.setattr(repo, "_generated", lambda config, held: {})
    text = libasm.plan(cfg, tree.capture(cfg))["src/sdk/f.s"].decode()
    assert ".if VERSION_B\n.globl f_b\nf_b:\n.else\n.globl f_a\nf_a:\n.endif\n" in text
    assert "jal x_a" in text and "jal x_b" in text
    assert [libasm.select(text, v, ["a", "b"]).count("jal") for v in "ab"] == [1, 1]
    result = repo.sync(cfg, {})
    held = tree.capture(cfg)
    assert "src/sdk/f.s" in result["written"] and "INCLUDE_ASM" not in (cfg.project.root / "src/sdk/f.c").read_text()
    assert held.owner("a", "f_a") == held.owner("b", "f_b") == "src/sdk/f.s"
    assert repo.sync(cfg, {})["written"] == []


def test_a_named_routine_gives_its_name_to_the_shared_source(tmp_path, monkeypatch):
    words = (0x40026000, 0x03E00008, 0)
    names = {"a": "func_80000400_a", "b": "Named"}
    facts = {v: f"{n} = 0x80000400; // type:func rom:0x1000 size:0xC\n" for v, n in names.items()}
    cfg = fixture.config(
        tmp_path,
        sources={
            "src/sdk/f.c": '#if defined(VERSION_A)\nINCLUDE_ASM("main", func_80000400_a);\n#endif\n'
            '#if defined(VERSION_B)\nINCLUDE_ASM("main", Named);\n#endif\n'
        },
        symbols=facts,
    )
    cfg.project.libraries = ("src/sdk",)
    for version in cfg.project.versions:
        _patch(cfg, version, 0x1000, words)
    monkeypatch.setattr(repo, "_generated", lambda config, held: {})
    assert list(libasm.plan(cfg, tree.capture(cfg))) == ["src/sdk/Named.s"]
