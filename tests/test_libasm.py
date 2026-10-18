"""Library assembly survives source-tree discovery, setup extraction, binding and measured report credit."""

from types import SimpleNamespace as N

import fixture
import pytest

from unbake import adapters, libasm, native, policy, recipes, repo, report, tree, view
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


def test_normal_c_instructions_are_not_extracted_and_version_split_is_left_alone(library):
    assert not libasm.foreign([0x24020001, 0x03E00008, 0], 0, adapters.row("gcc-test"))
    path = library.project.root / "roms/baserom.b.z64"
    data = bytearray(path.read_bytes())
    data[0x1000:0x1004] = (0x40026800).to_bytes(4, "big")
    path.write_bytes(data)
    library.project.version["b"].sha1 = sha1(data)
    assert libasm.plan(library, tree.capture(library)) == {}


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

    def assemble(source, recipe, includes, out):
        calls.append(source.read_text())
        out.write_bytes(b"object")
        return fixture.native_result(outputs={"object": out})

    monkeypatch.setattr(adapters, "GnuAssembler", lambda cfg: N(assemble=assemble))
    obj, results = native.objects(snapshot, unit, "a", recipe, tmp_path / "work")
    assert obj.read_bytes() == b"object" and len(results) == 1 and "mfc0" in calls[0]
    assert view.get(snapshot, unit, "a", recipe).text == calls[0]
    assert native.warnings(snapshot, unit, "a", recipe, tmp_path / "work") == ()
    assert native.recorded(snapshot, unit, "a", recipe) == ()


def test_renderer_uses_local_branch_labels_and_preserves_wide_registers():
    text = libasm.render("f", [0x1000FFFF, 0], 0x80001000, {})
    assert ".Lf_000" in text and libasm.scan(text) == {"f": ".text"}
    assert ".set gp=64" in libasm.render("f", [0xDFBF0000, 0x03E00008, 0], 0x80001000, {})


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
    assert held.owner("b", "f") is None and held.owner("b", "d") == "src/x.c"


def test_versions_whose_stand_in_lives_in_another_file_must_render_the_same_assembly(library):
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
    assert libasm.plan(library, tree.capture(library)) == {}


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


def test_lost_library_routine_that_renders_differently_keeps_its_stand_in_in_the_library(lost):
    path = lost.project.root / "roms/baserom.b.z64"
    data = bytearray(path.read_bytes())
    data[0x1010:0x1014] = (0x40026800).to_bytes(4, "big")
    path.write_bytes(data)
    lost.project.version["b"].sha1 = sha1(data)
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
