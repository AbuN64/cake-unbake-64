"""Native object and per-function proof tests with no tools, processes, builds or ROM bytes beyond the fixture's."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import fixture
import pytest
from abucache.compile import Step

from unbake import effort, native, store
from unbake.contracts import Finding, Placement, Recipe, Refusal, Snapshot, SourceView, UnitSpec, digest, sha256

ZEROS = bytes(0x20)


@pytest.fixture(autouse=True)
def invocation():
    with effort.command("test-native", ()):
        yield


@pytest.fixture
def case(tmp_path, monkeypatch, toolchains):
    """(snapshot, unit, recipe, object, work, linker): a unit over the fixture project with canned tools."""
    cfg = fixture.config(tmp_path)
    unit = UnitSpec("src/f.c", "c", ("f",), "gcc-test", {})
    snapshot = Snapshot(cfg, "head", None, {}, {unit.path: b"void f(void) {}\n"}, "snapshot")
    recipe = Recipe("gcc-test", (), ("-O2",), (), "recipe")
    obj = tmp_path / "input.o"
    obj.write_bytes(b"object")
    linker = Mock()
    monkeypatch.setattr(native.adapters, "linker", Mock(return_value=linker))
    return snapshot, unit, recipe, obj, tmp_path / "work", linker


def _outputs(case, data=None):
    """The linker answers with these section files (the default: 32 zero bytes of code)."""
    linker = case[-1]
    data = {".text": ZEROS} if data is None else data
    seen = linker.placements = []

    def link(obj, placed, symbols, work, trim):
        seen.append(tuple(placed))
        outputs = {}
        for section, blob in data.items():
            path = work / (section.removeprefix(".") + ".out")
            path.write_bytes(blob)
            outputs[section] = path
        return fixture.native_result(outputs=outputs)

    linker.link.side_effect = link
    return linker


def _tools(case, monkeypatch, toolchains, *, emits_asm=False):
    snapshot, unit, recipe, obj, work, linker = case
    row = toolchains["toolchain"]["gcc-test"]
    row["emits_asm"] = emits_asm
    calls = []
    toolchain = Mock()

    def step(phase, out):
        return Step(phase, out, (phase, "{in}", "{out}"))

    def run(phase, argv, cwd, **_):  # an abucache chain step: the files are named {in} and {out} in a private directory
        calls.append((argv[0], (cwd / argv[1]).read_bytes(), Path(argv[2]).suffix))
        (cwd / argv[2]).write_bytes(b"compiled" if argv[0] == "compile" else b"assembled")
        return fixture.native_result(argv=(argv[0],))

    def check(step, result):
        if step.phase == "compile" and (findings := toolchain.diagnose(result)):
            raise Refusal(*findings)

    toolchain.diagnose.return_value = ()
    toolchain.refused.return_value = []
    toolchain.check.side_effect = check
    chain = [step("compile", "source.s" if emits_asm else "source.o")]
    chain += [step("assemble", "source.o")] if emits_asm else []
    monkeypatch.setattr(native.adapters, "chain_steps", Mock(return_value=tuple(chain)))
    monkeypatch.setattr(native.adapters, "tool_identity", Mock(return_value="tools"))
    monkeypatch.setattr(native.process, "run", Mock(side_effect=run))
    monkeypatch.setattr(native.adapters, "toolchain", Mock(return_value=toolchain))
    source_view = SourceView(unit.path, "a", "view-key", "preprocessed\n", (), ())
    monkeypatch.setattr(native.view, "get", Mock(return_value=source_view))
    return (snapshot, unit, recipe, obj, work, linker), calls, toolchain


# ---------------------------------------------------------------- objects
@pytest.mark.parametrize(("emits", "phases"), [(False, ["compile"]), (True, ["compile", "assemble"])])
def test_objects_phase_walk(case, monkeypatch, toolchains, emits, phases):
    case, calls, _ = _tools(case, monkeypatch, toolchains, emits_asm=emits)
    snapshot, unit, recipe, _, work, _ = case
    obj, results = native.objects(snapshot, unit, "a", recipe, work)
    assert obj == work / "f.o"
    assert [call[0] for call in calls] == phases
    assert [result.argv[0] for result in results] == phases
    assert obj.read_bytes() == (b"compiled" if phases == ["compile"] else b"assembled")  # the last step's file
    assert calls[0][1:] == (b"preprocessed\n", ".s" if emits else ".o")
    cached_obj, cached_results = native.objects(snapshot, unit, "a", recipe, work / "second")
    assert cached_obj.read_bytes() == obj.read_bytes()
    assert cached_results == ()
    assert len(calls) == len(phases)


def test_every_version_with_the_same_preprocessed_bytes_shares_one_object(case, monkeypatch, toolchains):
    case, calls, _ = _tools(case, monkeypatch, toolchains, emits_asm=True)
    snapshot, unit, recipe, _, work, _ = case
    for version, directory in (("a", "first"), ("b", "second"), ("a", "third")):
        obj, _ = native.objects(snapshot, unit, version, recipe, work / directory)
        assert obj.read_bytes() == b"assembled"
    assert [call[0] for call in calls] == ["compile", "assemble"]  # one compile for three requests
    native.view.get.return_value = replace(
        native.view.get.return_value, key="other-key", text="preprocessed differently\n"
    )
    native.objects(snapshot, unit, "b", recipe, work / "fourth")
    assert [call[0] for call in calls] == ["compile", "assemble", "compile", "assemble"]


def test_a_failed_compile_is_cached_like_an_object(case, monkeypatch, toolchains):
    case, calls, toolchain = _tools(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    toolchain.diagnose.return_value = (Finding("compile.error", "parse error"),)
    for directory in ("first", "second"):
        with pytest.raises(Refusal) as refused:
            native.objects(snapshot, unit, "a", recipe, work / directory)
        assert refused.value.findings[0].key == "compile.error"
    assert len(calls) == 1  # the second refusal is replayed from the store


@pytest.mark.parametrize(
    ("text", "named"), [("int f;\n", False), ('__asm__(".include \\"nonmatchings/main/f.s\\"");\n', True)]
)
def test_include_directories_key_the_object_only_when_the_text_names_a_file(case, monkeypatch, toolchains, text, named):
    case, _, _ = _tools(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    native.view.get.return_value = replace(native.view.get.return_value, text=text)
    native.objects(snapshot, unit, "a", recipe, work)
    includes = native.adapters.chain_steps.call_args.args[3]
    assert bool(includes) is named


def test_stamp_names_the_text_the_recipe_and_the_tools_and_nothing_else(case, monkeypatch, toolchains):
    case, _, _ = _tools(case, monkeypatch, toolchains)
    snapshot, unit, recipe, *_ = case
    renamed = replace(unit, path="src/g.c", members=("g",))
    assert native.stamp(snapshot, unit, "a", recipe) == native.stamp(snapshot, renamed, "a", recipe)
    assert native.stamp(snapshot, unit, "a", recipe) != native.stamp(snapshot, unit, "a", replace(recipe, digest="o3"))
    before = native.stamp(snapshot, unit, "a", recipe)
    native.view.get.return_value = replace(native.view.get.return_value, text="other\n")
    assert native.stamp(snapshot, unit, "a", recipe) != before


def test_recorded_reads_the_builds_compile_err_and_never_compiles(tmp_path, monkeypatch):
    cfg = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    unit = UnitSpec("src/a.c", "c", ("f",), "gcc-test", {})
    kept = tmp_path / "build/a/src/a/@12@"
    kept.mkdir(parents=True)
    (kept / "compile.err").write_text("a.c:3: warning: makes pointer from integer\na.c:4: warning: unused\n")
    monkeypatch.setattr(
        native.adapters,
        "toolchain",
        lambda c, i: SimpleNamespace(refused=lambda text: [x for x in text.splitlines() if "pointer" in x]),
    )
    monkeypatch.setattr(native.process, "run", Mock(side_effect=AssertionError("never compiles")))
    snapshot = SimpleNamespace(config=cfg)
    recipe = SimpleNamespace(toolchain="gcc-test")
    assert native.recorded(snapshot, unit, "a", recipe) == ("a.c:3: warning: makes pointer from integer",)


def test_every_native_refusal_boundary_uses_diagnostic_query(case, monkeypatch):
    snapshot, unit, recipe, *_ = case
    calls = []
    monkeypatch.setattr(native.view, "diagnostic", lambda *args: calls.append(args[-1]) or "include/type.h:2: error")
    with pytest.raises(Refusal) as caught, native._tools(unit, snapshot, unit, "a", recipe):
        raise Refusal(Finding("compile.error", "unit.i:330: error"))
    assert caught.value.findings[0].reason == "include/type.h:2: error"
    assert calls == ["unit.i:330: error"]


@pytest.mark.parametrize(
    ("exit", "files", "key"),
    [
        (1, (), "score.compile"),
        (0, (), "land.request"),
        (0, ("greg", "lreg"), None),
        (0, ("greg", "lreg", "sched"), None),
    ],
)
def test_allocation_dumps_name_a_compile_error_as_one_and_a_dumpless_toolchain_as_one(
    case, monkeypatch, exit, files, key
):
    snapshot, unit, recipe, _, work, _ = case
    step = SimpleNamespace(argv=("cc1", "{in}", "-o", "{out}"))
    monkeypatch.setattr(native, "_compile_inputs", lambda *a: ("int f;", (step,), "", ""))
    monkeypatch.setattr(native, "_include", lambda *a: ())

    def run(name, argv, cwd, **_):
        assert name == "native.dumps" and cwd == work
        assert argv == ["cc1", "f.i", "-o", "f.s", "-g", "-dg", "-dl", "-dS", "-dR"]
        assert (cwd / "f.i").read_text() == "int f;"
        for kind in files:
            (cwd / f"f.i.{kind}").write_text(f";; Function f\n;; {kind}\n")
        (cwd / "f.s").write_text("f:\n")
        return fixture.native_result(argv=argv, exit=exit, stderr=b"f.c:3: structure has no member named `x'\n")

    monkeypatch.setattr(native.process, "run", run)
    monkeypatch.setattr(native.process, "scratch", lambda root: root)
    if key is None:
        assert native.dumps(snapshot, unit, "a", recipe, work) == (
            ";; Function f\n;; greg\n",
            ";; Function f\n;; lreg\n",
            "f:\n",
            {kind: f";; Function f\n;; {kind}\n" for kind in ("sched", "sched2") if kind in files},
        )
        return
    with pytest.raises(Refusal) as refused:
        native.dumps(snapshot, unit, "a", recipe, work)
    found = refused.value.findings[0]
    assert found.key == key and (key != "score.compile" or "no member named `x'" in found.reason)


# ---------------------------------------------------------------- the proof of one function
@pytest.fixture
def world(cfg, monkeypatch, tmp_path):
    """The fixture project (c_func at rom 0x1000, stand_in at 0x1010, ROM all zero) with the compile and the object
    reader replaced: the object holds c_func and stand_in in 32 bytes of .text."""
    obj = tmp_path / "unit.o"
    obj.write_bytes(b"object")
    state = SimpleNamespace(
        cfg=cfg,
        recipes=[],
        defined={"c_func": (".text", 0, 16), "stand_in": (".text", 0x10, 16)},
        sizes={".text": 32},
        linker=Mock(),
    )

    def objects(snapshot, unit, version, recipe, work):
        state.recipes.append(recipe)
        return obj, ()

    monkeypatch.setattr(native, "objects", objects)
    monkeypatch.setattr(native.versions, "defined", lambda path: (state.defined, state.sizes))
    monkeypatch.setattr(native.adapters, "linker", Mock(return_value=state.linker))
    monkeypatch.setattr(native.adapters, "tool_identity", Mock(return_value="tools"))
    monkeypatch.setattr(native.view, "diagnostic", lambda snapshot, unit, version, recipe, text: text)
    _outputs((None, None, None, None, None, state.linker))
    return state


def _text(cfg, path="src/main/c_func.c"):
    return (cfg.project.root / path).read_text()


def test_exact_proof_score_one_empty_symptoms(world):
    proof = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    assert (proof.exact, proof.missing, proof.score, proof.symptoms) == (True, (), 1.0, {})
    assert proof.built_sha256 == proof.target_sha256 == sha256(bytes(0x10))
    assert (proof.unit, proof.member, proof.version) == ("src/main/c_func.c", "c_func", "a")
    (placed,) = world.linker.placements
    assert placed == (Placement("a", ".text", 0, 32, 0x400),)  # the code sits at the masked address of c_func
    assert world.linker.link.call_args.args[4] is False


def test_the_measured_module_defines_non_matching(world):
    native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    (recipe,) = world.recipes
    assert "-DNON_MATCHING" in recipe.cppflags


def test_a_second_function_of_the_object_is_read_at_its_own_offset(world):
    _outputs((None, None, None, None, None, world.linker), {".text": bytes(0x10) + b"\x01" + bytes(0x0F)})
    path = "src/main/stand_in.c"
    proof = native.measure(world.cfg, "a", "stand_in", path, "void stand_in(void) {}")
    assert not proof.exact and "1 bytes differ at +0x0 .text" in proof.missing[0]
    assert native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg)).exact  # +0x10 is not c_func


def test_inexact_proof_has_score_and_symptoms(world):
    _outputs((None, None, None, None, None, world.linker), {".text": bytes(3) + b"\x01" + bytes(28)})
    proof = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    assert not proof.exact and proof.missing
    assert 0 <= proof.score < 1
    assert proof.symptoms["bytes_differ"] is True and proof.symptoms["score"] == proof.score
    assert "1 bytes differ at +0x3 .text" in proof.missing[0]


def test_a_short_output_is_a_size_gap(world):
    _outputs((None, None, None, None, None, world.linker), {".text": bytes(4)})
    proof = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    assert not proof.exact and any(".text size 4 != 16" in m for m in proof.missing)


def test_built_bytes_are_stored_for_the_diff(world):
    proof = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    producer = Mock(side_effect=AssertionError("built bytes must already be cached"))
    assert store.cached(world.cfg, "bytes", proof.built_sha256, producer) == bytes(0x10)


def test_a_warm_proof_links_nothing(world):
    first = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    second = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    assert second == first
    world.linker.link.assert_called_once()


def test_a_name_the_module_does_not_own_is_a_gap(world):
    proof = native.measure(world.cfg, "a", "fuzzy", "src/main/c_func.c", _text(world.cfg))
    assert not proof.exact and "fuzzy is not owned by src/main/c_func.c" in proof.missing[0]
    assert proof.symptoms.get("no_measurement") is None and proof.object_sha256 == ""
    world.linker.link.assert_not_called()


def test_a_compile_failure_is_a_gap_with_the_compiler_lines(world, monkeypatch):
    def failing(*args):
        raise Refusal(Finding("compile.error", "c_func.c:1: parse error"))

    monkeypatch.setattr(native, "objects", failing)
    proof = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    assert not proof.exact and proof.score == 0.0
    assert "compile.error: c_func.c:1: parse error" in proof.missing[0]
    assert proof.symptoms["compile_failed"] is True


def test_a_section_no_named_symbol_places_follows_the_code(world):
    world.sizes = {".text": 32, ".rodata": 8}
    _outputs((None, None, None, None, None, world.linker), {".text": bytes(32), ".rodata": bytes(8)})
    native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    (placed,) = world.linker.placements
    assert {p.section: p.vram for p in placed} == {".text": 0x400, ".rodata": 0x420}


def test_a_symbol_the_object_does_not_define_is_unproved(world):
    world.defined = {"stand_in": (".text", 0x10, 16)}
    proof = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    assert not proof.exact and any("unproved" in m for m in proof.missing)


def test_the_proof_key_holds_the_target_bytes(world, monkeypatch):
    native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    monkeypatch.setattr(native.versions, "rom_bytes", lambda record, start, end: b"\x01" * (end - start))
    proof = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    assert not proof.exact  # the ROM moved under the same object: not a cache hit
    assert world.linker.link.call_count == 2


def test_digest_of_the_overlay_names_every_file_it_replaces(world):
    snapshot, *_ = native.candidate(world.cfg, "c_func" and "a", "src/main/c_func.c", "int x;\n")
    other, *_ = native.candidate(world.cfg, "a", "src/main/c_func.c", "int y;\n")
    assert snapshot.digest != other.digest and digest(snapshot.digest)


def test_assembly_stand_in_cannot_score_as_candidate_c(world):
    path = "src/main/stand_in.c"
    proof = native.measure(world.cfg, "a", "stand_in", path, _text(world.cfg, path))
    assert not proof.exact and proof.score == 0
    assert "is not owned" in proof.missing[0]
    assert not world.recipes


def test_nonmatching_body_is_measured_instead_of_its_assembly(world):
    path = "src/main/stand_in.c"
    source = "#ifdef NON_MATCHING\nvoid stand_in(void) {}\n#else\n" + _text(world.cfg, path) + "\n#endif\n"
    proof = native.measure(world.cfg, "a", "stand_in", path, source)
    assert proof.exact and proof.score == 1
    assert "-DNON_MATCHING" in world.recipes[0].cppflags


def test_several_functions_of_one_module_compile_once(world):
    path = "src/main/c_func.c"
    proofs = native.measure_names(world.cfg, "a", ["c_func", "fuzzy", "c_func"], path, _text(world.cfg))
    assert sorted(proofs) == ["c_func", "fuzzy"] and proofs["c_func"].exact and not proofs["fuzzy"].exact
    assert len(world.recipes) == 1 and world.linker.link.call_count == 1  # one compile, one link per owned name


def test_a_symbol_file_in_the_overlay_is_the_one_the_link_resolves_against(world):
    seen = []
    plain = world.linker.link.side_effect

    def link(obj, placed, symbols, work, trim):
        seen.append(symbols.read_text())
        return plain(obj, placed, symbols, work, trim)

    world.linker.link.side_effect = link
    path = "versions/a/symbol_addrs.txt"
    overlay = (world.cfg.project.root / path).read_bytes() + b"gOverlaid = 0x80158A80; // type:data\n"
    native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg), None, {path: overlay})
    assert "gOverlaid" not in seen[0]
    assert "PROVIDE(gOverlaid = 0x80158A80);" in seen[1]
    snapshot, *_ = native.candidate(world.cfg, "a", "src/main/c_func.c", "int x;\n", None, {path: overlay})
    assert snapshot.versions["a"].symbols["gOverlaid"] == 0x80158A80  # undefined-symbol checks see it too


def test_extent_is_the_size_in_the_facts_else_the_distance_to_the_next_symbol():
    facts = {
        "sized": (0x80000000, 0x1000, 8, "func"),
        "bare": (0x80000010, 0x1010, None, "func"),
        "inner": (0x80000014, 0x1014, None, "label"),
        "next": (0x80000028, 0x1028, None, "func"),
        "last": (0x80000040, 0x1040, None, "func"),
        "romless": (0x80000050, None, None, "func"),
    }
    assert [native.extent(facts, n) for n in ("sized", "bare", "next", "last", "romless", "absent")] == [
        8, 0x18, 0x18, None, None, None,
    ]  # fmt: skip


def test_a_shorter_candidate_is_compared_against_the_targets_whole_extent(world, monkeypatch):
    facts = native.symbol_facts(world.cfg, "a")
    base = facts["c_func"]
    monkeypatch.setattr(
        native,
        "symbol_facts",
        lambda *a, **k: {
            **facts,
            "c_func": (base[0], base[1], None, base[3]),
            "stand_in": (base[0] + 0x18, base[1] + 0x18, None, "func"),
        },
    )
    _outputs((None, None, None, None, None, world.linker), {".text": bytes(8)})
    proof = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    assert not proof.exact and proof.target_sha256 == sha256(bytes(0x18))
    assert any(".text size 8 != 24" in m for m in proof.missing)


def test_the_selected_function_is_read_at_its_own_object_offset_whatever_an_earlier_member_measures(world):
    world.defined = {"stand_in": (".text", 0, 16), "c_func": (".text", 0x20, 16)}  # the member before it is longer
    world.sizes = {".text": 0x30}
    _outputs((None, None, None, None, None, world.linker), {".text": b"\x01" * 0x20 + bytes(0x10)})
    proof = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    assert proof.exact, proof.missing
    (placed,) = world.linker.placements[-1:]
    assert placed[0].vram == 0x400 - 0x20  # the text starts where the selected function's address less its offset says


def test_a_measured_module_compiles_only_the_non_matching_bodies_of_the_names_measured(world):
    block = "#ifdef NON_MATCHING\nint {}(void) {{ return 1; }}\n#endif\n"
    text = block.format("sibling") + block.format("c_func")
    snapshot, *_ = native.candidate(world.cfg, "a", "src/main/c_func.c", text, None, None, ["c_func"])
    assert snapshot.overlays["src/main/c_func.c"].decode().splitlines()[0] == "#if 0"
    assert snapshot.overlays["src/main/c_func.c"].decode().splitlines()[3] == "#ifdef NON_MATCHING"
    whole, *_ = native.candidate(world.cfg, "a", "src/main/c_func.c", text)
    assert whole.overlays["src/main/c_func.c"].decode() == text  # no names: the module as it is


# ---------------------------------------------------------------- the unit's own constant pool
LUI_AT, LWC1 = 0x3C010000, 0xC4200000  # lui at,0 and lwc1 f0,0(at): the candidate's object, relocated later


def _word(*words):
    return b"".join(w.to_bytes(4, "big") for w in words)


def test_the_pool_starts_where_the_roms_code_forms_the_address_less_the_offset_it_reads(monkeypatch):
    monkeypatch.setattr(native, "_pool_refs", lambda blob, table: (_word(LUI_AT, LWC1), [(4, 8)], set()))
    target = _word(0x3C01800C, 0xC4206E44)  # lui at,0x800C; lwc1 f0,0x6E44(at)
    pool = native._pool(b"", {}, (".text", 0, 8), 0, target, 8)
    assert pool == native._Pool(0x800C6E44 - 8, ((8, 4),))


def test_a_reference_the_roms_code_has_a_few_words_off_still_votes(monkeypatch):
    monkeypatch.setattr(native, "_pool_refs", lambda blob, table: (_word(LUI_AT, LWC1), [(4, 0)], set()))
    target = _word(0x3C01800C, 0, 0, 0xC4206E44)  # three words more between the lui and the load
    assert native._pool(b"", {}, (".text", 0, 8), 0, target, 16).base == 0x800C6E44


def test_a_jump_table_is_read_whole_and_a_string_to_its_nul(monkeypatch):
    monkeypatch.setattr(native, "_pool_refs", lambda blob, table: (_word(LUI_AT, 0x24210000), [(4, 4)], {4, 8, 12}))
    pool = native._pool(b"", {}, (".text", 0, 8), 0, _word(0x3C01800C, 0x24216E44), 8)
    assert pool.reads == ((4, 12),)
    monkeypatch.setattr(native, "_pool_refs", lambda blob, table: (_word(LUI_AT, 0x24210000), [(4, 4)], set()))
    assert native._pool(b"", {}, (".text", 0, 8), 0, _word(0x3C01800C, 0x24216E44), 8).reads == ((4, 0),)


_SECTIONS = {"text": {"inputs": [".text"]}, "rodata": {"inputs": [".rodata", ".rdata"]}}
_JTBL = {"f": (".text", 0, 0x100), "jtbl_800DD4D0": (".rdata", 0, 0)}  # a stand-in's jump table at the object's start
_FACTS = {"f": (0x8041D550, 0x18B550, None, "func"), "jtbl_800DD4D0": (0x800E1500, None, None, "data")}


def test_a_rodata_stand_in_the_c_body_replaces_does_not_hold_its_pool_off():
    # SPEED-4 func_8041D4E0_de: the stand-in's old 0x38-byte table first, the body's own table after it; the ROM's
    # code reads the table where the stand-in lies, so the pool links there and not 0x38 later
    pool = native._Pool(0x800E1500 - 0x38, ((0x38, 0x38),))
    assert native._superseded(pool, _JTBL, _FACTS, _SECTIONS, {"f", "s32"}, "f")


def test_c_data_and_a_stand_in_the_body_reads_keep_the_rodata_anchor():
    pool = native._Pool(0x800E1500 - 0x38, ((0x38, 0x38),))
    assert not native._superseded(pool, _JTBL, _FACTS, _SECTIONS, {"f", "jtbl_800DD4D0"}, "f")  # C defines it
    read = native._Pool(0x800E1500, ((0, 4), (0x38, 4)))
    assert not native._superseded(read, _JTBL, _FACTS, _SECTIONS, {"f"}, "f")  # the body reads the stand-in
    elsewhere = native._Pool(0x800F0000, ((0x38, 0x38),))
    assert not native._superseded(elsewhere, _JTBL, _FACTS, _SECTIONS, {"f"}, "f")  # its pool is not where it lies
    assert not native._superseded(native._Pool(), _JTBL, _FACTS, _SECTIONS, {"f"}, "f")


def test_a_candidate_with_no_pool_references_has_no_base(monkeypatch):
    monkeypatch.setattr(native, "_pool_refs", lambda blob, table: (b"", [], set()))
    assert native._pool(b"", {}, (".text", 0, 8), 0, bytes(8), 8) == native._Pool()
    assert native._pool_refs(b"object", {}) == (b"", [], set())  # not an ELF object


def test_the_unit_rodata_links_at_the_pool_base_and_its_bytes_are_the_roms(world, monkeypatch):
    world.sizes = {".text": 32, ".rodata": 8}
    monkeypatch.setattr(native, "_pool", lambda *a: native._Pool(0x800C6E00, ((0, 4),)))
    monkeypatch.setattr(native, "_rom_at", lambda cfg, version, address: 0)
    _outputs((None, None, None, None, None, world.linker), {".text": bytes(32), ".rodata": bytes(8)})
    proof = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    assert proof.exact, proof.missing
    (placed,) = world.linker.placements
    assert {p.section: p.vram for p in placed}[".rodata"] == 0x800C6E00
    _outputs((None, None, None, None, None, world.linker), {".text": bytes(32), ".rodata": b"\x01" + bytes(7)})
    monkeypatch.setattr(native, "_pool", lambda *a: native._Pool(0x800C6E00, ((0, 4), (4, 4))))
    other = native.measure(world.cfg, "a", "c_func", "src/main/c_func.c", _text(world.cfg))
    assert not other.exact and "1 of 2 reads of the unit's .rodata differ" in other.missing[0]
    assert other.score < 1.0


def test_a_pool_the_rom_has_nowhere_is_a_difference(monkeypatch):
    snapshot = SimpleNamespace(config=None, versions={"a": object()})
    monkeypatch.setattr(native, "_rom_at", lambda cfg, version, address: None)
    gaps = native._pool_gaps(snapshot, "a", native._Pool(0x800C0000, ((0, 4),)), bytes(4))
    assert gaps and "0x800C0000+0x0" in gaps[0]


def test_an_unnamed_type_added_to_the_text_shares_the_compile(case, monkeypatch, toolchains):
    case, calls, _ = _tools(case, monkeypatch, toolchains, emits_asm=True)
    snapshot, unit, recipe, _, work, _ = case
    for name, text in (("first", "int v;\n"), ("second", "typedef int T;\nstruct S { T a; };\nint v;\n")):
        native.view.get.return_value = replace(native.view.get.return_value, text=text)
        obj, _ = native.objects(snapshot, unit, "a", recipe, work / name)
        assert obj.read_bytes() == b"assembled"
    assert [call[0] for call in calls] == ["compile", "assemble"]  # the second text added only unnamed types
    native.view.get.return_value = replace(native.view.get.return_value, text="typedef int T;\nT v;\n")
    native.objects(snapshot, unit, "a", recipe, work / "third")
    assert [call[0] for call in calls] == ["compile", "assemble", "compile", "assemble"]  # now the type is named


def test_a_view_directory_named_in_the_text_does_not_split_the_shared_compile(case, monkeypatch, toolchains):
    case, calls, _ = _tools(case, monkeypatch, toolchains, emits_asm=True)
    snapshot, unit, recipe, _, work, _ = case
    for name, key in (("first", "6119c08cc36202f4"), ("second", "3648cfc1c10e3962")):
        text = f'const char *file = "build/views/{key}/src/f.c";\ntypedef int T_{key[:4]};\n'
        native.view.get.return_value = replace(native.view.get.return_value, text=text)
        native.objects(snapshot, unit, "a", recipe, work / name)
    assert [call[0] for call in calls] == ["compile", "assemble"]
