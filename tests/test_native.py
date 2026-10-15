"""Native phase and proof tests with no tools, processes, builds or ROM files."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import fixture
import pytest
from abucache.compile import Step

from unbake import effort, native, ownership, store
from unbake.contracts import (
    Finding,
    LayoutMap,
    Member,
    Placement,
    Recipe,
    Refusal,
    Resident,
    Snapshot,
    SourceView,
    UnitSpec,
    Version,
    digest,
    sha256,
)

TARGET = bytes.fromhex("03e0000800000000")


@pytest.fixture(autouse=True)
def invocation():
    with effort.command("test-native", ()):
        yield


@pytest.fixture
def case(tmp_path, monkeypatch, toolchains):
    root = tmp_path / "project"
    cfg = fixture.bare_config(
        root, memory=1 << 30, toolchain_root=tmp_path / "toolchains", author=("test", "test"), macros={"a": (), "b": ()}
    )
    unit = UnitSpec("src/f.c", "c", "group", ("f",), "gcc-test", {})
    placement = Placement("a", ".text", 0x1000, 0x1008, 0x80000400)
    member = Member("f", "function", "c", "group", (placement,))
    layout = LayoutMap({}, {"f": member}, {unit.path: unit}, "layout", (), {})
    record = Version("a", root / "unused.rom", "rom-digest", "split", "symbols", {"f": placement.vram}, ())
    snapshot = Snapshot(cfg, "head", layout, {"a": record}, {unit.path: b"void f(void) {}\n"}, "snapshot")
    recipe = Recipe("gcc-test", (), ("-O2",), (), "recipe")
    obj = tmp_path / "input.o"
    obj.write_bytes(b"object")
    monkeypatch.setattr(native.versions, "undefined", Mock(return_value=("f",)))
    monkeypatch.setattr(native.build, "symbols_ld", Mock(return_value=b"f = 0x80000400;\n"))
    monkeypatch.setattr(native.adapters, "host_tools", Mock(return_value={"mips_ld": "pinned-linker"}))
    monkeypatch.setattr(
        native.versions,
        "rom_bytes",
        Mock(side_effect=lambda version, start, end: TARGET[start - 0x1000 : end - 0x1000]),
    )
    linker = Mock()
    monkeypatch.setattr(native.adapters, "linker", Mock(return_value=linker))
    return snapshot, unit, recipe, obj, tmp_path / "work", linker


def _outputs(case, data=None):
    linker = case[-1]
    data = {".text": TARGET} if data is None else data

    def link(obj, ranges, symbols, work, trim):
        outputs = {}
        for section, blob in data.items():
            path = work / (section.removeprefix(".") + ".out")
            path.write_bytes(blob)
            outputs[section] = path
        return fixture.native_result(outputs=outputs)

    linker.link.side_effect = link
    return linker


@pytest.fixture(autouse=True)
def _no_emitted_sections(monkeypatch):
    """The tests' object files are stand-in bytes: they emit no pool a unit could borrow."""
    monkeypatch.setattr(ownership, "emitted", lambda blob: {})


def _measure(case):
    snapshot, unit, recipe, obj, work, _ = case
    return native.measure(snapshot, unit, "a", recipe, obj, work)


def _placements(case, rows, *, members=None):
    snapshot, unit, recipe, obj, work, linker = case
    members = members or {"f": replace(snapshot.layout.members["f"], placements=tuple(rows))}
    unit = replace(unit, members=tuple(members))
    snapshot = replace(snapshot, layout=replace(snapshot.layout, members=members, units={unit.path: unit}))
    return snapshot, unit, recipe, obj, work, linker


@pytest.mark.parametrize("vram_delta,rom_delta", [(0, 0), (4, 0), (-4, 0), (0, 4), (0, -4)])
def test_sections_contiguity_gaps_and_overlaps(case, vram_delta, rom_delta):
    first = Placement("a", ".text", 0x1000, 0x1004, 0x80000400)
    second = Placement("a", ".text", 0x1004 + rom_delta, 0x1008 + rom_delta, 0x80000404 + vram_delta)
    snapshot, unit, *_ = _placements(case, (second, first))
    if vram_delta or rom_delta:
        with pytest.raises(Refusal) as refused:
            native.sections(snapshot, unit, "a")
        assert refused.value.findings[0].key == "link.error"
        assert refused.value.findings[0].unit == unit.path
    else:
        assert native.sections(snapshot, unit, "a") == (replace(first, rom_end=0x1008),)


def test_sections_bss_memory_extent_and_version_filter(case):
    rows = (
        Placement("a", ".bss", 0x1010, 0x1010, 0x80000500, 4),
        Placement("a", ".bss", 0x1020, 0x1020, 0x80000504, 12),
        Placement("b", ".text", 0, 8, 0),
    )
    snapshot, unit, *_ = _placements(case, rows)
    assert native.sections(snapshot, unit, "a") == (Placement("a", ".bss", 0x1010, 0x1020, 0x80000500, 16),)
    assert native.sections(snapshot, unit, "absent") == ()


def test_exact_proof_score_one_empty_symptoms(case):
    linker = _outputs(case)
    (proof,) = _measure(case)
    assert (proof.exact, proof.missing, proof.score, proof.symptoms) == (True, (), 1.0, {})
    assert proof.built_sha256 == proof.target_sha256 == sha256(TARGET)
    assert proof.source_sha256 == sha256(case[0].read(case[1].path))
    assert proof.object_sha256 == sha256(b"object")
    args = linker.link.call_args.args
    assert args[0] == case[3]
    assert args[2].read_bytes() == b"f = 0x80000400;\n"  # no ROM and no resident windows reach the link
    assert args[3:] == (case[4], False)


def test_inexact_proof_has_score_and_symptoms(case):
    _outputs(case, {".text": bytes.fromhex("03e0000900000000")})
    (proof,) = _measure(case)
    assert not proof.exact and proof.missing
    assert 0 <= proof.score < 1
    assert proof.symptoms["bytes_differ"] is True
    assert proof.symptoms["score"] == proof.score
    assert "1 bytes differ at +0x3 .text" in proof.missing[0]


@pytest.mark.parametrize(
    "data,missing",
    [
        ({}, "unproved .text"),
        ({".text": TARGET[:4]}, ".text size 4 != 8"),
        ({".text": TARGET + b"padding"}, ".text size 15 != 8"),
    ],
)
def test_missing_and_wrongly_sized_outputs(case, data, missing):
    _outputs(case, data)
    (proof,) = _measure(case)
    assert not proof.exact
    assert any(missing in item for item in proof.missing)
    assert isinstance(proof.missing, tuple)
    assert proof.symptoms["score"] == proof.score


def test_the_link_reads_neither_the_rom_nor_the_resident_windows(case):
    linker = _outputs(case)
    _measure(case)
    first = linker.link.call_args
    snapshot, unit, recipe, obj, work, _ = case
    project = replace(snapshot.config.project, resident={"a": (Resident(1, 2, 3, 0),)})
    other = replace(
        snapshot,
        config=replace(snapshot.config, project=project),
        versions={"a": replace(snapshot.versions["a"], rom_sha256="another rom")},
    )
    native.measure(other, unit, "a", recipe, obj, work)
    assert linker.link.call_count == 1  # the proof key holds neither, so the second measure is a hit
    assert first.args[0] == obj


def test_built_bytes_stored(case):
    _outputs(case)
    (proof,) = _measure(case)
    producer = Mock(side_effect=AssertionError("built material must already be cached"))
    assert store.cached(case[0].config, "bytes", proof.built_sha256, producer) == TARGET
    producer.assert_not_called()


def test_proof_cache_hit_skips_link(case):
    linker = _outputs(case)
    first = _measure(case)
    second = _measure(case)
    assert second == first
    linker.link.assert_called_once()
    assert effort.counters()["proof"] == (1, 1)


def test_keep_links_again_on_a_warm_proof_and_leaves_the_cache_entry_in_place(case):
    linker = _outputs(case)
    first = _measure(case)
    snapshot, unit, recipe, obj, work, _ = case
    kept = native.measure(snapshot, unit, "a", recipe, obj, work, True)  # an export needs the linker's outputs
    assert kept == first and linker.link.call_count == 2
    assert _measure(case) == first and linker.link.call_count == 2  # and the cache still answers the next plain call


@pytest.mark.parametrize("change", ["object", "source", "recipe", "symbols", "pins", "trim", "names", "slice"])
def test_proof_cache_inputs_invalidate(case, monkeypatch, change):
    linker = _outputs(case)
    _measure(case)
    snapshot, unit, recipe, obj, work, _ = case
    if change == "object":
        obj.write_bytes(b"changed object")
    elif change == "source":
        snapshot = replace(snapshot, overlays={unit.path: b"changed source"})
    elif change == "recipe":
        recipe = replace(recipe, digest="changed recipe")
    elif change == "symbols":
        native.build.symbols_ld.return_value = b"changed symbols"
    elif change == "pins":
        native.adapters.host_tools.return_value = {"mips_ld": "changed pin"}
    elif change == "names":  # the same bytes under another member name are another proof
        member = replace(snapshot.layout.members["f"], name="g")
        unit = replace(unit, members=("g",))
        snapshot = replace(snapshot, layout=replace(snapshot.layout, members={"g": member}))
    elif change == "slice":
        short = Placement("a", ".text", 0x1000, 0x1004, 0x80000400)
        member = replace(snapshot.layout.members["f"], placements=(short,))
        snapshot = replace(snapshot, layout=replace(snapshot.layout, members={"f": member}))
    else:
        member = replace(snapshot.layout.members["f"], state="hasm")
        snapshot = replace(snapshot, layout=replace(snapshot.layout, members={"f": member}))
    native.measure(snapshot, unit, "a", recipe, obj, work)
    assert linker.link.call_count == 2


@pytest.mark.parametrize("witness", [b"16", b"15"])
def test_bss_witness_stored_but_excluded_from_score(case, monkeypatch, witness):
    text = case[0].layout.members["f"].placements[0]
    case = _placements(case, (text, Placement("a", ".bss", 0x1008, 0x1008, 0x80000500, 16)))
    built = bytes.fromhex("03e0000900000000")
    _outputs(case, {".text": built, ".bss": witness})
    real_score = native.symptoms.score
    score = Mock(wraps=real_score)
    monkeypatch.setattr(native.symptoms, "score", score)
    (proof,) = _measure(case)
    score.assert_called_once_with(built, TARGET)
    assert proof.built_sha256 == sha256(built + witness)
    assert proof.target_sha256 == sha256(TARGET + b"16")
    assert any("unproved .bss size" in item for item in proof.missing) == (witness != b"16")
    producer = Mock(side_effect=AssertionError("missing material"))
    assert store.cached(case[0].config, "bytes", proof.built_sha256, producer) == built + witness


def _tools(case, monkeypatch, toolchains, *, kind="c", emits_asm=False, custom_assemble=False):
    snapshot, unit, recipe, obj, work, linker = case
    unit = replace(unit, kind=kind)
    row = toolchains["toolchain"]["gcc-test"]
    row["emits_asm"] = emits_asm
    if custom_assemble:
        row["assemble"] = ["custom"]
    calls = []
    toolchain, assembler, armips = Mock(), Mock(), Mock()

    def compile(source, recipe, out):
        calls.append(("compile", source.read_bytes(), out.suffix))
        out.write_bytes(b"compiled")
        return fixture.native_result(argv=("compile",))

    def assemble(label):
        def run(source, recipe, include, out):
            calls.append((label, source.read_bytes(), tuple(include)))
            out.write_bytes(b"assembled")
            return fixture.native_result(argv=(label,))

        return run

    def step(phase, out):
        return Step(phase, out, (phase, "{in}", "{out}"))

    def run(phase, argv, cwd, **_):  # an abucache chain step: the files are named {in} and {out} in a private directory
        calls.append((argv[0], (cwd / argv[1]).read_bytes(), Path(argv[2]).suffix))
        (cwd / argv[2]).write_bytes(b"compiled" if argv[0] == "compile" else b"assembled")
        return fixture.native_result(argv=(argv[0],))

    def check(step, result):
        if step.phase == "compile" and (findings := toolchain.diagnose(result)):
            raise Refusal(*findings)

    toolchain.compile.side_effect = compile
    toolchain.diagnose.return_value = ()
    toolchain.refused.return_value = []
    toolchain.check.side_effect = check
    chain = [step("compile", "source.s" if emits_asm else "source.o")]
    chain += [step("assemble", "source.o")] if emits_asm else []
    monkeypatch.setattr(native.adapters, "chain_steps", Mock(return_value=tuple(chain)))
    monkeypatch.setattr(native.adapters, "tool_identity", Mock(return_value="tools"))
    monkeypatch.setattr(native.process, "run", Mock(side_effect=run))
    toolchain.assemble.side_effect = assemble("paired")
    assembler.assemble.side_effect = assemble("gnu")
    armips.assemble.side_effect = assemble("armips")
    monkeypatch.setattr(native.adapters, "toolchain", Mock(return_value=toolchain))
    monkeypatch.setattr(
        native.adapters, "assembler", Mock(side_effect=lambda cfg, kind: {"gnu": assembler, "armips": armips}[kind])
    )
    source_view = SourceView(unit.path, "a", "view-key", "preprocessed\n", (), ())
    monkeypatch.setattr(native.view, "get", Mock(return_value=source_view))
    return (snapshot, unit, recipe, obj, work, linker), calls, toolchain


@pytest.mark.parametrize(
    "kind,emits,paired,phases",
    [
        ("c", False, False, ["compile"]),
        ("c", True, False, ["compile", "assemble"]),
        ("asm", False, False, ["gnu"]),
        ("data", False, False, ["compile"]),
        ("resource", False, False, ["armips"]),
    ],
)
def test_objects_phase_walk(case, monkeypatch, toolchains, kind, emits, paired, phases):
    case, calls, _ = _tools(case, monkeypatch, toolchains, kind=kind, emits_asm=emits, custom_assemble=paired)
    snapshot, unit, recipe, _, work, _ = case
    obj, results = native.objects(snapshot, unit, "a", recipe, work)
    assert obj == work / "f.o"
    assert [call[0] for call in calls] == phases
    assert [result.argv[0] for result in results] == phases
    assert obj.read_bytes() == (b"compiled" if phases == ["compile"] else b"assembled")  # the last step's file
    if kind in ("c", "data"):
        assert calls[0][1:] == (b"preprocessed\n", ".s" if emits else ".o")
    else:
        assert calls[0][1] == snapshot.read(unit.path)
        native.view.get.assert_not_called()
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


def _consumers(header):
    """Preprocessed views of four units that include one header: only `user` names what the header's B declares."""
    bodies = {
        "src/user.c": "int f(B *p) { return p->x; }",
        "src/other.c": "int g(void) { return 0; }",
        "src/more.c": "int h(int v) { return v; }",
        "src/last.c": "int k;",
    }
    return {
        path: SourceView(
            path, "a", digest((path, header)), f'# 1 "include/h.h"\n{header}\n# 1 "/r/{path}"\n{body}\n', (), ()
        )
        for path, body in bodies.items()
    }


HEADER = "typedef struct B { int x; } B;\ntypedef struct C { int y; } C;"


def test_a_stamp_moves_with_any_included_header_byte_and_repeats_from_the_memo(case, monkeypatch):
    snapshot, unit, recipe, *_ = case
    project = replace(snapshot.config.project, version_macros={"a": ()})
    snapshot = replace(snapshot, config=replace(snapshot.config, project=project))
    monkeypatch.setattr(native.recipes, "resolve", Mock(return_value=recipe))
    monkeypatch.setattr(native.adapters, "tool_identity", Mock(return_value="tools"))
    views = {}
    monkeypatch.setattr(native.view, "get", lambda snap, u, *a, **k: views[u.path])
    # a view is named by the bytes it reads: here the canned text stands for them
    monkeypatch.setattr(native.view, "_key", lambda snap, u, *a: (digest(views[u.path].text), (), False))
    live = Mock(side_effect=native.view.declarations.live)
    monkeypatch.setattr(native.view.declarations, "live", live)

    def stamps(header):
        views.update(_consumers(header))
        return {path: native.stamp(snapshot, replace(unit, path=path), "a") for path in views}

    before = stamps(HEADER)
    work = live.call_count
    assert (
        stamps(HEADER) == before and live.call_count == work
    )  # same bytes: the second call is a hit, no digest redone
    unused = stamps(HEADER.replace("int y;", "long y, w;") + "\nextern int more;")
    used = stamps(HEADER.replace("int x;", "long x;"))
    # a cached compile failure replays under the stamp, so an edit of an unreached declaration must move it too
    assert all(unused[path] != before[path] and used[path] != before[path] for path in before)


def test_an_unused_declaration_edit_reuses_the_object_and_a_used_one_compiles_again(case, monkeypatch, toolchains):
    case, calls, _ = _tools(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    now = {}
    monkeypatch.setattr(native.view, "get", lambda *a, **k: now["view"])

    def objects(header, name):
        now["view"] = _consumers(header)["src/user.c"]
        return native.objects(snapshot, unit, "a", recipe, work / name)

    objects(HEADER, "first")
    objects(HEADER.replace("int y;", "long y;") + "\nextern int more;", "unused")
    assert [call[0] for call in calls] == ["compile", "compile"]
    assert "int f" not in calls[1][1].decode()  # the second is the one proof of the header text, not a consumer compile
    objects(HEADER.replace("int x;", "long x;"), "used")
    assert [call[0] for call in calls] == ["compile"] * 3
    objects(HEADER.replace("int x;", "long x;"), "again")  # nothing changed: nothing compiles
    assert len(calls) == 3


def test_an_exact_run_compiles_every_unit_whose_header_text_changed_even_when_the_edit_is_unreached(
    case, monkeypatch, toolchains
):
    case, calls, _ = _tools(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    now = {}
    monkeypatch.setattr(native.view, "get", lambda *a, **k: now["view"])

    def objects(header, name):
        now["view"] = _consumers(header)["src/user.c"]
        return native.objects(snapshot, unit, "a", recipe, work / name)

    objects(HEADER, "first")
    with native.exactly():
        objects(HEADER + "\nextern int more;", "unused")
        objects(HEADER + "\nextern int more;", "again")  # the same whole text: its object is found
    assert [call[0] for call in calls] == ["compile", "compile"]  # the unit again, not a header proof alone
    assert "int f" in calls[1][1].decode()  # and the second repeat finds its object


def test_header_proof_shared_across_marker_only_difference(case, monkeypatch, toolchains):
    case, calls, _ = _tools(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    now = {}
    monkeypatch.setattr(native.view, "get", lambda *a, **k: now["view"])

    def objects(header, name, marker='# 1 "include/h.h"'):
        view = _consumers(header)["src/user.c"]
        now["view"] = replace(view, text=view.text.replace('# 1 "include/h.h"', marker))
        return native.objects(snapshot, unit, "a", recipe, work / name)

    objects(HEADER, "first")
    objects(HEADER + "\nextern int more;", "unused")
    assert [call[0] for call in calls] == ["compile", "compile"]
    objects(HEADER + "\nextern int more;", "moved", '# 9 "/other/include/h.h" 1')
    assert len(calls) == 2  # the same header set under other markers is one proof


GFX = "extern void draw(Gfx *g);"
OWN = "int f(void) { return 1; }"


def _gfx_case(case, monkeypatch, toolchains):
    """A unit that never names `draw`, over a header whose declaration of it may be broken; the compiler stub refuses
    any text that still has a `Gfx` it was not given a type for, as the real parse error of func_80203848_de did."""
    case, calls, toolchain = _tools(case, monkeypatch, toolchains)
    now = {}

    def run(phase, argv, cwd, **_):
        text = (cwd / argv[1]).read_text()
        calls.append((argv[0], text, Path(argv[2]).suffix))
        broken = "Gfx" in text and "typedef struct Gfx" not in text
        (cwd / argv[2]).write_bytes(b"compiled")
        return fixture.native_result(
            argv=(argv[0],), exit=1 if broken else 0, stderr=b"parse error before `Gfx'" if broken else b""
        )

    monkeypatch.setattr(native.process, "run", Mock(side_effect=run))
    toolchain.diagnose.side_effect = lambda result: (
        (Finding("compile.error", "parse error before `Gfx'"),) if result.exit else ()
    )
    monkeypatch.setattr(
        native.view,
        "get",
        lambda *a, **k: SourceView(
            "src/f.c",
            "a",
            digest(now["header"]),
            f'# 1 "include/h.h"\n{now["header"]}\n# 1 "/r/src/f.c"\n{OWN}\n',
            (),
            (),
        ),
    )
    return case, calls, now


def test_a_failure_from_an_unreached_declaration_is_not_replayed_once_the_declaration_is_fixed(
    case, monkeypatch, toolchains
):
    case, calls, now = _gfx_case(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    now["header"] = f"typedef int T;\n{GFX}"
    with pytest.raises(Refusal):
        native.objects(snapshot, unit, "a", recipe, work / "first")
    now["header"] = "typedef int T;\ntypedef struct Gfx Gfx;\n" + GFX  # same live part, the header now compiles
    assert native.objects(snapshot, unit, "a", recipe, work / "second")[0].read_bytes() == b"compiled"
    now["header"] = f"typedef int T;\n{GFX}"  # the same broken text still refuses, from the cache
    before = len(calls)
    with pytest.raises(Refusal):
        native.objects(snapshot, unit, "a", recipe, work / "third")
    assert len(calls) == before


def test_an_object_of_the_live_part_is_not_reused_when_the_rest_of_the_text_no_longer_compiles(
    case, monkeypatch, toolchains
):
    case, _calls, now = _gfx_case(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    now["header"] = "typedef int T;\ntypedef struct Gfx Gfx;"
    native.objects(snapshot, unit, "a", recipe, work / "first")
    now["header"] = f"typedef int T;\n{GFX}"  # an unreached declaration breaks: the live key is unchanged
    with pytest.raises(Refusal):
        native.objects(snapshot, unit, "a", recipe, work / "second")


@pytest.mark.parametrize(("kind", "cacheable"), [("c", True), ("data", True), ("asm", False), ("resource", False)])
def test_kinds_that_cannot_be_cached_say_why(kind, cacheable):
    row = native.config.load_resource("units.toml")["kind"][kind]
    assert row["cacheable"] is cacheable
    assert bool(row.get("uncacheable")) is (not cacheable)  # a reason for every kind that is not cached


def test_a_cacheable_kind_without_a_preprocess_phase_refuses(case, monkeypatch, toolchains):
    case, *_ = _tools(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    monkeypatch.setattr(native, "_phases", lambda unit: ("compile", "link"))
    with pytest.raises(Refusal) as caught:
        native.objects(snapshot, unit, "a", recipe, work)
    assert caught.value.findings[0].reason == "unit kind c is cacheable but has no preprocess phase"


def test_asm_include_has_macro_dir(case, monkeypatch, toolchains):
    case, calls, _ = _tools(case, monkeypatch, toolchains, kind="asm")
    snapshot, unit, recipe, _, work, _ = case
    native.objects(snapshot, unit, "a", recipe, work)
    macro = native.config.load_resource("repo.toml")["splat"]["options"]["generated_asm_macros_directory"]
    root = snapshot.config.project.root
    assert calls[0][2] == (
        root / "include",
        (root / unit.path).parent,
        root / macro.format(version="a", name=snapshot.config.project.name),
    )


def test_gaps_proof_score_zero(case, monkeypatch, toolchains):
    case, _, toolchain = _tools(case, monkeypatch, toolchains)
    toolchain.diagnose.return_value = (Finding("compile.error", "bad source"),)
    snapshot, unit, recipe, _, work, linker = case
    (proof,) = native.prove(snapshot, unit, "a", recipe, work)
    assert not proof.exact and proof.score == 0.0
    assert proof.symptoms["compile_failed"] is True
    assert proof.missing == ("version a: compile.error: bad source",)
    assert proof.object_sha256 == ""
    linker.link.assert_not_called()


@pytest.mark.parametrize("unresolved", [False, True])
def test_prove_resolves_undefined_symbols(case, monkeypatch, unresolved):
    snapshot, unit, recipe, obj, work, linker = case
    monkeypatch.setattr(native, "objects", Mock(return_value=(obj, ())))
    monkeypatch.setattr(native.versions, "undefined", Mock(return_value=("f", "external") if unresolved else ("f",)))
    resolve = Mock(return_value=(Finding("link.error", "undefined external"),) if unresolved else ())
    monkeypatch.setattr(native.versions, "resolve", resolve)
    _outputs(case)
    (proof,) = native.prove(snapshot, unit, "a", recipe, work)
    resolve.assert_called_once_with(snapshot.versions["a"], native.versions.undefined.return_value, "src/f.c")
    if unresolved:
        assert proof.missing == ("version a: unresolved external",)
        assert proof.score == 0.0
        assert proof.object_sha256 == sha256(b"object")
        linker.link.assert_not_called()
    else:
        assert proof.exact
        linker.link.assert_called_once()


def test_prove_link_refusal_becomes_gaps(case, monkeypatch):
    snapshot, unit, recipe, obj, work, linker = case
    monkeypatch.setattr(native, "objects", Mock(return_value=(obj, ())))
    monkeypatch.setattr(native.versions, "undefined", Mock(return_value=()))
    monkeypatch.setattr(native.versions, "resolve", Mock(return_value=()))
    linker.link.side_effect = Refusal(Finding("link.error", "missing linked output"))
    (proof,) = native.prove(snapshot, unit, "a", recipe, work)
    assert proof.missing == ("version a: link.error: missing linked output",)
    assert proof.score == 0.0 and not proof.exact


def test_measure_no_sections_skips_link(case):
    case = _placements(case, ())
    assert _measure(case) == ()
    case[-1].link.assert_not_called()


def test_member_proofs_slice_owned_bytes(case):
    snapshot = case[0]
    f = replace(snapshot.layout.members["f"], placements=(Placement("a", ".text", 0x1000, 0x1004, 100),))
    g = replace(f, name="g", placements=(Placement("a", ".text", 0x1004, 0x1008, 104),))
    case = _placements(case, (), members={"f": f, "g": g})
    _outputs(case, {".text": TARGET[:4] + b"\x01\x00\x00\x00"})
    first, second = _measure(case)
    assert (first.member, first.exact, first.built_sha256) == ("f", True, sha256(TARGET[:4]))
    assert second.member == "g" and not second.exact
    assert second.built_sha256 == sha256(b"\x01\x00\x00\x00")


@pytest.mark.parametrize("bad_sections", [False, True])
def test_resource_prove_without_link_or_symbol_scan(case, monkeypatch, bad_sections):
    snapshot, unit, recipe, obj, work, linker = case
    unit = replace(unit, kind="resource")
    obj.write_bytes(TARGET)
    if bad_sections:
        case = _placements(
            (snapshot, unit, recipe, obj, work, linker), (Placement("a", ".bss", 0x1000, 0x1000, 100, 8),)
        )
        snapshot, unit, recipe, obj, work, linker = case
    monkeypatch.setattr(native, "objects", Mock(return_value=(obj, ())))
    undefined = Mock(side_effect=AssertionError("resource must not scan ELF symbols"))
    monkeypatch.setattr(native.versions, "undefined", undefined)
    monkeypatch.setattr(native.versions, "resolve", Mock(return_value=()))
    (proof,) = native.prove(snapshot, unit, "a", recipe, work)
    assert proof.exact == (not bad_sections)
    if bad_sections:
        assert proof.score == 0.0
        assert proof.missing == ("version a: unproved resource sections",)
    undefined.assert_not_called()
    linker.link.assert_not_called()


def test_missing_tool_is_refusal_with_unit_kind(case, monkeypatch):
    snapshot, unit, recipe, _obj, work, _linker = case
    monkeypatch.setattr(
        native,
        "objects",
        Mock(side_effect=Refusal(Finding("native.missing_tool", "host tools.armips is not configured"))),
    )
    with pytest.raises(Refusal, match=r"native\.missing_tool"):
        native.prove(snapshot, unit, "a", recipe, work)
    with pytest.raises(Refusal) as caught, native._tools(unit):
        raise Refusal(Finding("native.missing_tool", "host tools.cpp is not configured"))
    assert unit.kind in caught.value.findings[0].reason


def test_proof_cache_ignores_unreferenced_symbols_and_unused_tools(case, monkeypatch):
    snapshot, unit, recipe, obj, work, linker = case
    _outputs(case)

    def symbols(s, v):
        return repr(sorted(s.versions[v].symbols.items())).encode()

    monkeypatch.setattr(native.build, "symbols_ld", symbols)
    first = native.measure(snapshot, unit, "a", recipe, obj, work)
    record = replace(snapshot.versions["a"], symbols={**snapshot.versions["a"].symbols, "unrelated": 123})
    updated = replace(snapshot, versions={"a": record})
    monkeypatch.setattr(
        native.adapters, "host_tools", lambda cfg: {"mips_ld": "pinned-linker", "armips": "unrelated change"}
    )
    assert native.measure(updated, unit, "a", recipe, obj, work) == first
    assert linker.link.call_count == 1
    record = replace(record, symbols={"f": 123})
    native.measure(replace(snapshot, versions={"a": record}), unit, "a", recipe, obj, work)
    assert linker.link.call_count == 2


def test_a_failed_compile_is_cached_like_an_object(case, monkeypatch, toolchains):
    case, calls, toolchain = _tools(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    toolchain.diagnose.return_value = (Finding("compile.error", "conflicting types for `x'"),)
    for attempt in ("first", "second"):
        with pytest.raises(Refusal) as error:
            native.objects(snapshot, unit, "a", recipe, work / attempt)
        assert error.value.findings[0].reason == "conflicting types for `x'"
    assert len(calls) == 1  # the second refusal came from the cache, not from the compiler


def test_sections_link_the_code_at_the_masked_address_the_project_gives(case):
    snapshot, unit, *_ = _placements(
        case,
        (Placement("a", ".text", 0x1000, 0x1004, 0x80000400), Placement("a", ".rodata", 0x1100, 0x1104, 0x80000500)),
    )
    project = replace(snapshot.config.project, build={**snapshot.config.project.build, "text_mask": 0x1FFFFFFF})
    masked = replace(snapshot, config=replace(snapshot.config, project=project))
    got = {p.section: p.vram for p in native.sections(masked, unit, "a")}
    assert got == {".text": 0x400, ".rodata": 0x80000500}  # only the code moves
    assert {p.section: p.vram for p in native.sections(snapshot, unit, "a")}[".text"] == 0x80000400


def test_proof_cache_key_names_the_members_so_a_renamed_member_is_not_served_a_stale_name(case, monkeypatch):
    snapshot, unit, recipe, obj, work, _ = case
    monkeypatch.setattr(native, "objects", Mock(return_value=(obj, ())))
    _outputs(case)
    cache = {}

    def cached(cfg, kind, key, produce):
        if (kind, key) not in cache:
            cache[kind, key] = produce()
        return cache[kind, key]

    monkeypatch.setattr(native.store, "cached", cached)
    (first,) = native.prove(snapshot, unit, "a", recipe, work)
    member = replace(snapshot.layout.members["f"], name="g")
    renamed_unit = replace(unit, members=("g",))
    renamed = replace(snapshot, layout=replace(snapshot.layout, members={"g": member}, units={unit.path: renamed_unit}))
    (second,) = native.prove(renamed, renamed_unit, "a", recipe, work)
    assert (first.member, second.member) == ("f", "g")


def test_recorded_reads_the_builds_compile_err_and_never_compiles(tmp_path, monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    unit = UnitSpec("src/a.c", "c", "group", ("f",), "gcc-test", {})
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


def test_a_stamp_names_an_overlaid_snapshot_by_the_content_it_reads(case, monkeypatch):
    snapshot, unit, recipe, *_ = case
    project = replace(snapshot.config.project, version_macros={"a": ()})
    snapshot = replace(snapshot, config=replace(snapshot.config, project=project))
    monkeypatch.setattr(native.recipes, "resolve", Mock(return_value=recipe))
    monkeypatch.setattr(native.adapters, "tool_identity", Mock(return_value="tools"))
    monkeypatch.setattr(
        native.view,
        "closure",
        lambda snap, *_: (("src/f.c", "pin"), ("include/h.h", sha256(snap.overlays.get("include/h.h", b"disk")))),
    )
    on_disk = native.stamp(snapshot, unit, "a")
    same_bytes = replace(snapshot, overlays={"include/h.h": b"disk"}, digest="another snapshot")
    other_bytes = replace(snapshot, overlays={"include/h.h": b"edited"}, digest="third snapshot")
    assert native.stamp(same_bytes, unit, "a") == on_disk  # warm: the ownership and proof jobs it keys are not redone
    assert native.stamp(other_bytes, unit, "a") != on_disk  # only a unit that reads the edited file is redone


@pytest.mark.parametrize(
    ("missing", "found"),
    [
        ((), ()),
        (("version us: 4 bytes differ at +0x0 .text",), ()),  # fuzzy bytes still build
        (
            ("version us: .text size 8 != 12", "version us: 4 bytes differ at +0x0 .text"),
            ("version us: .text size 8 != 12",),
        ),
        (("version us: unproved .data",), ("version us: unproved .data",)),  # make's owned-size assert fails too
    ],
)
def test_compiles_refuses_what_the_make_link_asserts(tmp_path, monkeypatch, missing, found):
    unit = Mock(path="src/a.c", withheld=("eu",))
    snapshot = Mock(config=Mock())
    proof = Mock(missing=missing)
    monkeypatch.setattr(native.recipes, "resolve", lambda *a: Mock())
    monkeypatch.setattr(native.store, "work", lambda cfg: _Here(tmp_path))
    monkeypatch.setattr(native, "objects", lambda *a: (tmp_path / "a.o", ()))
    measure = Mock(return_value=(proof,))
    monkeypatch.setattr(native, "measure", measure)
    result = native.compiles((snapshot, unit, "us"))
    assert [f.reason.removeprefix("src/a.c: ") for f in result] == list(found)
    assert {f.key for f in result} <= {"link.error"} and measure.call_count == 1
    assert native.compiles((snapshot, unit, "eu")) == () and measure.call_count == 1  # withheld: make never links it


class _Here:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self.path

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(
    "path,state,changed,pinned,advisory",
    [
        ("src/rodata.func_80409448_de.800E0DF0.c", "missing", "layout.toml", True, True),
        ("src/rw_pool_804243E0_E22B0_us_rev1.c", "missing", "split", True, True),
        ("src/f.c", "missing", "deleted-layout", True, True),
        ("src/f.c", "missing", None, True, False),  # absent at capture, unchanged layout: real fault
        ("src/f.c", "missing", "symbols", True, False),  # unrelated input is not publication evidence
        ("src/f.c", "missing", "layout.toml", False, False),  # no captured evidence
        ("src/f.c", "overlay", "layout.toml", True, False),
        ("src/f.c", "present", "layout.toml", True, False),
        ("src/f.c", "permission", "layout.toml", True, False),
        ("src/f.c", "compiler-missing", "layout.toml", True, False),
    ],
)
def test_objects_missing_source_requires_changed_capture_inputs(
    case, monkeypatch, path, state, changed, pinned, advisory
):
    snapshot, unit, recipe, obj, work, _ = case
    root = snapshot.config.project.root
    root.mkdir(parents=True)
    for name in ("layout.toml", "split", "symbols"):
        (root / name).write_bytes(b"captured")
    pins = {name: sha256(b"captured") for name in ("layout.toml", "split", "symbols")} if pinned else {}
    unit = replace(unit, path=path, kind="data", withheld=("a", "b"))
    snapshot = replace(snapshot, overlays={path: None} if state == "overlay" else {}, reads=pins)
    if state in ("present", "compiler-missing"):
        (root / path).parent.mkdir(parents=True)
        (root / path).write_bytes(b"int data;")
    if changed == "deleted-layout":
        (root / "layout.toml").unlink()
    elif changed:
        (root / changed).write_bytes(b"published")
    if state == "permission":
        monkeypatch.setattr(Snapshot, "read", Mock(side_effect=PermissionError("source unreadable")))
    compiled = Mock(
        return_value=(obj, ()), side_effect=FileNotFoundError("compiler input") if state == "compiler-missing" else None
    )
    monkeypatch.setattr(native, "_compiled", compiled)
    monkeypatch.setattr(native.recipes, "resolve", Mock(return_value=recipe))
    materialize = Mock(side_effect=AssertionError("a deferred unit must not materialize"))
    monkeypatch.setattr(native.build, "_unit_key", materialize)
    if advisory:
        with pytest.raises(Refusal) as stale:
            native.objects(snapshot, unit, "a", recipe, work)
        assert [(f.key, f.path, f.blocking) for f in stale.value.findings] == [("journal.changed", path, True)]
        for version in ("a", "b"):  # pool chunks share the snapshot; a second version must also defer
            found = native.build.compile_unit((snapshot, unit, version))
            assert found == (
                Finding(
                    "check.moved-during-run",
                    "the source changed while check ran; it is checked next run",
                    path=path,
                    unit=path,
                    versions=(version,),
                    blocking=False,
                ),
            )
        assert snapshot.reads[path] is None  # retain the existing read-and-missing freshness record
        materialize.assert_not_called()
    elif state == "present":
        assert native.compiles((snapshot, unit, "a"), work) == ()
    else:
        with pytest.raises(PermissionError if state == "permission" else FileNotFoundError):
            native.compiles((snapshot, unit, "a"), work)
    assert compiled.call_count == int(state in ("present", "compiler-missing"))


@pytest.mark.parametrize(
    ("value", "referenced", "bound"), [(0x4F000000, True, True), (0x4E000000, True, False), (0x4F000000, False, False)]
)
def test_measure_links_only_equal_referenced_owned_literals(case, monkeypatch, value, referenced, bound):
    snapshot, unit, recipe, obj, work, linker = case
    address = 0x800C34FC
    datum = Member("constant", "data", "data", "g", (Placement("a", ".data", 0x2000, 0x2004, address),))
    provider = UnitSpec("src/data.c", "data", "g", ("constant",), "gcc-test", {})
    snapshot = replace(
        snapshot,
        layout=replace(
            snapshot.layout,
            members={**snapshot.layout.members, "constant": datum},
            units={**snapshot.layout.units, provider.path: provider},
        ),
    )
    rom = bytearray(0x2004)
    rom[0x1000:0x1008] = bytes.fromhex("3c01800cc43434fc")
    rom[0x2000:] = bytes.fromhex("4f000000")
    index = ownership._Index(
        rom,
        [(0x2000, 0x2004, "constant", ".data", address)],
        {},
        {},
        "rows",
        [(address, address + 4, 0x2000, 0x2004)],
        (),
    )
    monkeypatch.setattr(ownership, "_index", lambda *a: index)
    monkeypatch.setattr(
        ownership, "emitted", lambda blob: {".rodata": (value.to_bytes(4, "big"), {}, [(4, 0)] if referenced else [])}
    )
    _outputs(case)
    (proof,) = native.measure(snapshot, unit, "a", recipe, obj, work)
    assert proof.exact and proof.member == "f"
    placed = [p for p in linker.link.call_args.args[1] if p.section == ".rodata"]
    assert placed == ([Placement("a", ".rodata", 0x2000, 0x2004, address)] if bound else [])
    assert snapshot.layout.units[provider.path] == provider


@pytest.mark.parametrize(
    "text,named",
    [("preprocessed\n", False), ('__asm__(".include \\"m.inc\\"");\n', True), ('__asm__(".incbin \\"x\\"");\n', True)],
)
def test_include_directories_key_the_object_only_when_the_text_names_a_file(case, monkeypatch, toolchains, text, named):
    case, _, _ = _tools(case, monkeypatch, toolchains, emits_asm=True)
    snapshot, unit, recipe, _, work, _ = case
    monkeypatch.setattr(native.view, "get", Mock(return_value=SourceView(unit.path, "a", "view-key", text, (), ())))
    native.objects(snapshot, unit, "a", recipe, work)
    include = native.adapters.chain_steps.call_args.args[3]
    assert bool(include) is named


def test_a_stored_unit_counts_no_live_object_miss(case, monkeypatch, toolchains):
    case, _calls, now = _gfx_case(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    counted = Mock()
    monkeypatch.setattr(native.effort, "count", counted)
    now["header"] = "typedef int T;\ntypedef struct Gfx Gfx;"
    native.objects(snapshot, unit, "a", recipe, work / "first")  # new text: one miss of the shortcut
    native.objects(snapshot, unit, "a", recipe, work / "second")  # stored: it never tries the shortcut
    assert [c.args for c in counted.call_args_list if c.args[0] == "live-object"] == [("live-object", False)]
