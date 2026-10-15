"""Hardware-only routines become library assembly: recognised by instruction shape, generated from the ROM words."""

from contextlib import nullcontext
from dataclasses import replace

import pytest

from unbake import config as configuration
from unbake import libasm, report
from unbake.contracts import (
    Config,
    Group,
    Host,
    LayoutMap,
    Member,
    Placement,
    Project,
    Snapshot,
    UnitSpec,
    Version,
    VersionFiles,
)

JR, NOP = 0x03E00008, 0
GET_SR = [0x40026000, JR, NOP]
CACHE_LOOP = [0x3C088000, 0x25091FF0, 0xBD010000, 0x1509FFFE, 0x25080010, JR, NOP]
TLB_WIRE = [0x40085000, 0x40840000, 0x3C098000, 0x40895000, 0x40801000, 0x40801800, NOP, 0x42000002, NOP, NOP, NOP, NOP]
TLB_WIRE += [0x40885000, JR, NOP]
SET_FPCSR = [0x44C4F800, JR, NOP]
MASKED = [0x3C0E8034, 0x8DCF1230, 0x40026000, JR, NOP]
LEAF_C = [0x00851021, JR, NOP]  # addu v0, a0, a1
FRAMED_CALL = [0x27BDFFE8, 0x0C000100, NOP, 0x40026000, 0x27BD0018, JR, NOP]  # jal 0x80000400
JUMP_OUT = [0x40026000, 0x08000100, NOP]  # j 0x80000400
TRUNC = [0x4600010D, 0x4442F800, 0x44C4F800, JR, NOP]  # trunc.w.s beside cfc1 and ctc1: compiler output
TOOLCHAIN = {
    "never_emits": ["mfc0", "mtc0", "dmfc0", "dmtc0", "cache", "sync", "eret", "tlbp", "tlbr", "tlbwi", "tlbwr"],
    "fpu_control_with": ["cvt.w.s", "cvt.w.d", "trunc.w.s", "trunc.w.d"],
}


@pytest.mark.parametrize(
    "words", [GET_SR, CACHE_LOOP, TLB_WIRE, SET_FPCSR, FRAMED_CALL, JUMP_OUT], ids=lambda w: hex(w[0])
)
def test_code_the_toolchain_never_emits_is_foreign(words):
    assert libasm.foreign(words, 0x80000400, TOOLCHAIN)


@pytest.mark.parametrize("words", [LEAF_C, TRUNC, []], ids=lambda w: hex(w[0]) if w else "empty")
def test_compiler_output_is_not_foreign(words):
    assert not libasm.foreign(words, 0x80000400, TOOLCHAIN)


def test_every_toolchain_names_what_it_never_emits_and_where_it_pairs_fpu_control():
    for row in configuration.toolchains().values():
        assert row["never_emits"] and row["fpu_control_with"]


def test_render_names_external_targets_and_refuses_unnamed_ones():
    text = libasm.render("f", JUMP_OUT, 0x80000000, {0x80000400: "osNext"})
    assert "    j osNext\n" in text and ".L" not in text
    assert "jal osNext" in libasm.render("f", FRAMED_CALL, 0x80000000, {0x80000400: "osNext"})
    assert libasm.render("f", JUMP_OUT, 0x80000000, {}) is None


def test_render_is_one_instruction_per_line_with_branch_labels():
    text = libasm.render("osCacheLoop", CACHE_LOOP, 0x80000400, {})
    assert text == (
        ".set noreorder\n.set noat\n.section .text\n.globl osCacheLoop\nosCacheLoop:\n"
        "    lui $t0, 0x8000\n    addiu $t1, $t0, 0x1FF0\n.LosCacheLoop_008:\n    cache 0x01, 0x0($t0)\n"
        "    bne $t0, $t1, .LosCacheLoop_008\n    addiu $t0, $t0, 0x10\n    jr $ra\n    nop\n"
    )
    assert libasm.render("osCacheLoop", CACHE_LOOP, 0x80000400, {}) == text


def test_render_names_coprocessor_registers_by_number_and_symbols_by_halves():
    assert "    mfc0 $v0, $12\n" in libasm.render("a", GET_SR, 0x80000400, {})
    assert "    ctc1 $a0, $31\n" in libasm.render("b", SET_FPCSR, 0x80000400, {})
    text = libasm.render("c", MASKED, 0x80000400, {0x80341230: "gMask"})
    assert "lui $t6, %hi(gMask)" in text and "lw $t7, %lo(gMask)($t6)" in text
    assert "0x8034" in libasm.render("c", MASKED, 0x80000400, {})  # no symbol there: the number stays


def test_render_keeps_every_word_and_each_loop_label_once():
    text = libasm.render("w", TLB_WIRE, 0x80000400, {})
    body = [line for line in text.splitlines() if line.startswith("    ")]
    assert len(body) == len(TLB_WIRE) and text.count(".L") == 0


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(libasm.effort, "stage", lambda name: nullcontext())
    monkeypatch.setattr(
        libasm.layout, "unit_of", lambda s, m: next((u for u in s.layout.units.values() if m in u.members), None)
    )
    monkeypatch.setattr(libasm.layout, "edit_split", lambda text, name, **kw: text + f"|{name}:{kw['state']}".encode())
    monkeypatch.setattr(libasm.layout, "dump_map", lambda m: b"map:" + ",".join(sorted(m.units)).encode())
    monkeypatch.setattr(libasm.adapters, "row", lambda id: TOOLCHAIN)
    rom = {}
    monkeypatch.setattr(libasm.versions, "rom_bytes", lambda v, lo, hi: rom[v.id][lo - 0x1000 : hi - 0x1000])

    def make(routines, *, library=("src/libultra",), units=(), versions=("a", "b"), symbols=None, state="asm"):
        vf = {v: VersionFiles(f"{v}.z64", "0" * 40, f"{v}.yaml", f"{v}.txt", {}) for v in versions}
        project = Project(
            tmp_path,
            "fixture",
            "fixture",
            "Fixture",
            versions,
            versions[0],
            "test",
            {},
            vf,
            {},
            {},
            {},
            "project",
            library_dirs=library,
        )
        host = Host(
            2,
            2,
            1024,
            1024,
            tmp_path / "cache",
            1024,
            tmp_path / "tools",
            {},
            None,
            2,
            2,
            0.8,
            2,
            ("test", "t@example.invalid"),
            {},
            "host",
        )
        members, start = {}, 0x1000
        for name, words in routines.items():
            places = []
            for v in versions:
                blob = b"".join(w.to_bytes(4, "big") for w in words)
                rom[v] = rom.get(v, b"") + blob
                places.append(Placement(v, ".text", start, start + len(blob), 0x80000000 + start))
            members[name] = Member(name, "function", state, "g", tuple(places))
            start += 4 * len(words)
        group = Group("g", "main", tuple(members), "authored", (), False, "src/libultra/g.c", "include/g.h")
        layout_map = LayoutMap({"g": group}, members, {u.path: u for u in units}, "layout", (), {})
        table = {
            v: Version(v, tmp_path / f"{v}.z64", "a" * 64, vf[v].split, vf[v].symbols, symbols or {}, ())
            for v in versions
        }
        contents = {vf[v].split: b"split:" + v.encode() for v in versions}
        return Snapshot(Config(project, host, "config"), "b" * 40, layout_map, table, contents, "snapshot")

    return make


def test_plan_puts_every_foreign_function_of_a_module_in_its_one_source(case):
    writes, findings = libasm.plan(case({"osGetSR": GET_SR, "osCache": CACHE_LOOP}))
    assert sorted(writes) == ["a.yaml", "b.yaml", "layout.toml", "src/libultra/g.s"] and findings == ()
    assert writes["layout.toml"] == b"map:src/libultra/g.s"
    assert writes["src/libultra/g.s"].decode() == (
        libasm.render("osGetSR", GET_SR, 0x80001000, {}) + libasm.render("osCache", CACHE_LOOP, 0x8000100C, {})
    )
    assert writes["a.yaml"] == b"split:a|osGetSR:c|osCache:c"


def test_plan_leaves_a_module_of_compiler_output_alone(case):
    assert libasm.plan(case({"osAdd": LEAF_C})) == ({}, ())


def test_plan_writes_nothing_for_an_unplaced_module_and_says_so(case):
    writes, findings = libasm.plan(case({"osGetSR": GET_SR}, library=("src/libgcc",)))
    assert writes == {} and [(f.key, f.unit, f.path, f.blocking) for f in findings] == [
        ("library-asm.unplaced", "osGetSR", "src/libultra/g.c", False)
    ]
    assert "unbake move" in findings[0].action
    assert libasm.plan(case({"osGetSR": GET_SR}, library=()))[0] == {}


def test_plan_writes_nothing_for_a_module_mixing_compiler_output_and_says_so(case):
    writes, findings = libasm.plan(case({"osGetSR": GET_SR, "osAdd": LEAF_C}))
    assert writes == {} and [(f.key, f.unit) for f in findings] == [("library-asm.mixed", "osGetSR")]


def test_plan_skips_functions_already_in_a_unit_or_not_assembly(case):
    unit = UnitSpec("src/libultra/g.s", "asm", "g", ("osGetSR",), "test", {})
    assert libasm.plan(case({"osGetSR": GET_SR}, units=[unit])) == ({}, ())
    assert libasm.plan(case({"osGetSR": GET_SR}, state="c")) == ({}, ())


def test_plan_renders_an_external_jump_by_its_symbol_and_flags_an_unnamed_one(case):
    named = case({"osJump": JUMP_OUT}, symbols={"osNext": 0x80000400})
    assert "    j osNext\n" in libasm.plan(named)[0]["src/libultra/g.s"].decode()
    writes, findings = libasm.plan(case({"osJump": JUMP_OUT}))
    assert writes == {} and [f.key for f in findings] == ["library-asm.unnamed-target"]


def test_plan_leaves_a_routine_whose_versions_read_differently(case):
    snapshot = case({"osMask": MASKED}, symbols={"gMask": 0x80341230})
    other = replace(snapshot.versions["b"], symbols={"gMask": 0x80341240})  # the address moved in b: the text differs
    snapshot = replace(snapshot, versions={**snapshot.versions, "b": other})
    writes, findings = libasm.plan(snapshot)
    assert writes == {} and [f.key for f in findings] == ["library-asm.version-split"]


def test_assembly_outside_a_library_directory_is_named(case):
    inside = UnitSpec("src/libultra/os/a.s", "asm", "g", ("osGetSR",), "test", {})
    outside = UnitSpec("src/game/b.s", "asm", "g", ("osAdd",), "test", {})
    compiled = UnitSpec("src/game/c.c", "c", "g", ("osCache",), "test", {})
    snapshot = case({"osGetSR": GET_SR, "osAdd": LEAF_C, "osCache": CACHE_LOOP}, units=[inside, outside, compiled])
    assert libasm.outside(snapshot) == ("src/game/b.s",)
    assert configuration.library(snapshot.config.project, "src/libultra/os/a.s")
    assert not configuration.library(snapshot.config.project, "src/libultra2/a.s")


def test_report_names_hardware_routines_library_asm_and_credits_library_units(case, monkeypatch):
    monkeypatch.setattr(report.layout, "unit_of", libasm.layout.unit_of)
    monkeypatch.setattr(report, "split_slot", lambda s, m: "system code C cannot emit" if m != "osAdd" else None)
    monkeypatch.setattr(report.policy, "unclean", lambda s: frozenset())
    monkeypatch.setattr(report.effort, "stage", lambda name: nullcontext())
    snapshot = case({"osGetSR": GET_SR, "osAdd": LEAF_C, "osCache": CACHE_LOOP})
    rows = report.items(snapshot, {})
    assert {r["member"]: r["kind"] for r in rows} == {
        "osGetSR": "library-asm",
        "osCache": "library-asm",
        "osAdd": "function",
    }
    unit = UnitSpec("src/libultra/os/osGetSR.s", "asm", "g", ("osGetSR",), "test", {})
    landed = replace(snapshot, layout=replace(snapshot.layout, units={unit.path: unit}))
    assert "osGetSR" not in {r["member"] for r in report.items(landed, {})}
    elsewhere = replace(unit, path="src/game/osGetSR.s")
    away = replace(snapshot, layout=replace(snapshot.layout, units={elsewhere.path: elsewhere}))
    assert "osGetSR" in {r["member"] for r in report.items(away, {})}


def test_plan_takes_hand_written_assembly_as_library_assembly_whatever_its_instructions(case):
    writes, findings = libasm.plan(case({"osAdd": LEAF_C}, state="hasm"))
    assert sorted(writes) == ["a.yaml", "b.yaml", "layout.toml", "src/libultra/g.s"] and findings == ()
    assert writes["a.yaml"] == b"split:a|osAdd:c"


def test_a_module_another_module_splits_becomes_one_file_per_contiguous_run(case):
    # real case: libasm_80200610 interleaved with other modules' code was refused as library-asm.scattered
    snapshot = case({"osGetSR": GET_SR, "osMid": LEAF_C, "osCache": CACHE_LOOP})
    members = dict(snapshot.layout.members)
    members["osMid"] = replace(members["osMid"], group="other")
    groups = {"g": replace(snapshot.layout.groups["g"], members=("osGetSR", "osCache"))}
    snapshot = replace(snapshot, layout=replace(snapshot.layout, members=members, groups=groups))
    writes, findings = libasm.plan(snapshot)
    assert findings == ()
    assert writes["src/libultra/g.s"].decode() == libasm.render("osGetSR", GET_SR, 0x80001000, {})
    assert writes["src/libultra/osCache.s"].decode() == libasm.render("osCache", CACHE_LOOP, 0x80001010, {})
    assert writes["layout.toml"] == b"map:src/libultra/g.s,src/libultra/osCache.s"


def test_a_module_whose_routines_different_versions_hold_gets_one_file_per_version_run(case):
    # real case: libasm_80200400 holds func_80200400_de/_eu/_us, each held by one version: library-asm.version-split
    snapshot = case({"osA": GET_SR, "osB": CACHE_LOOP})
    members = dict(snapshot.layout.members)
    for name, version in (("osA", "a"), ("osB", "b")):
        places = tuple(p for p in members[name].placements if p.version == version)
        members[name] = replace(members[name], placements=places)
    snapshot = replace(snapshot, layout=replace(snapshot.layout, members=members))
    writes, findings = libasm.plan(snapshot)
    assert findings == ()
    assert sorted(writes) == ["a.yaml", "b.yaml", "layout.toml", "src/libultra/g.s", "src/libultra/osB.s"]
    assert writes["a.yaml"] == b"split:a|osA:c" and writes["b.yaml"] == b"split:b|osB:c"


def test_a_run_ends_at_a_routine_that_cannot_be_rendered(case):
    snapshot = case({"osA": GET_SR, "osJump": JUMP_OUT, "osB": CACHE_LOOP})
    writes, findings = libasm.plan(snapshot)
    assert [f.key for f in findings] == ["library-asm.unnamed-target"]
    assert sorted(p for p in writes if p.endswith(".s")) == ["src/libultra/g.s", "src/libultra/osB.s"]


def test_plan_labels_are_local_to_each_routine_so_one_module_assembles(case):
    text = libasm.plan(case({"osA": CACHE_LOOP, "osB": CACHE_LOOP}))[0]["src/libultra/g.s"].decode()
    labels = [line for line in text.splitlines() if line.startswith(".L")]
    assert labels == [".LosA_008:", ".LosB_008:"]
