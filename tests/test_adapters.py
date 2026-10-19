"""Adapter contracts, with every native invocation replaced by a mock."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import fixture
import pytest

from unbake import adapters, process
from unbake import config as configuration
from unbake.contracts import Config, Finding, Host, Placement, Recipe, Refusal


@pytest.fixture(autouse=True)
def isolated_toolchains(toolchains):
    # conftest reuses TOOLCHAIN_ROW; keep all mutations local to this test.
    toolchains["toolchain"] = deepcopy(toolchains["toolchain"])


@pytest.fixture
def adapter_config(tmp_path):
    tools = {name: tmp_path / "host-bin" / name for name in fixture.TOOLS}
    host = Host(
        2, 2, 1024, 1024, tmp_path / "cache", 1024, tmp_path / "chains", tools, None, 2, 2, 0.8, 2,
        ("test", "test@invalid"), {}, "host",
    )  # fmt: skip
    project = SimpleNamespace(
        root=tmp_path, name="fixture", title="Fixture", toolchain="gcc-test", versions=("a", "b"),
        version={"a": SimpleNamespace(), "b": SimpleNamespace()}, cppflags=(), cflags=(), asflags=(), files={},
        libraries=(), digest="project",
    )  # fmt: skip
    return Config(project, host, "config")  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def native_mock(monkeypatch):
    run = Mock(return_value=fixture.native_result())
    monkeypatch.setattr(process, "run", run)
    monkeypatch.setattr(process, "tool", lambda config, name, **kw: config.host.tools[name])
    return run


@pytest.fixture
def recipe():
    return Recipe("gcc-test", ("-E",), ("-O2",), ("-EB",), "recipe")


@pytest.mark.parametrize(
    "template,values,expected",
    [
        (
            ("{flags}", "{empty}", "-I{dirs}", "{source}"),
            {"flags": ["-O2", "-g"], "empty": [], "dirs": ["one", "two"], "source": ["a b.c"]},
            ["-O2", "-g", "-Ione two", "a b.c"],
        ),
        (("literal", "{{brace}}"), {}, ["literal", "{brace}"]),
        ((), {}, []),
    ],
)
def test_render(template, values, expected):
    assert adapters.render(template, values) == expected


@pytest.mark.parametrize("case", ["none", "ido", "gcc", "no-template", "non-emitting"])
def test_assemble_template_choice(toolchains, case):
    row = None
    if case == "ido":
        row = toolchains["toolchain"]["ido-7.1"]
    elif case != "none":
        row = {"emits_asm": case != "non-emitting"}
        if case != "no-template":
            row["assemble"] = ["{as}", "custom", "{source}"]
    template, tool = adapters.assemble_template(row)
    if case == "gcc":
        assert (template, tool) == (("{as}", "custom", "{source}"), "as")
    else:
        assert (template, tool) == (("{as}", "{asflags}", "{includes}", "{source}", "-o", "{out}"), "mips_as")


@pytest.mark.parametrize("invalid", ["missing", "both"])
def test_toolchain_as_host_required(toolchains, native_mock, invalid):
    row = dict(toolchains["toolchain"]["gcc-test"])
    if invalid == "missing":
        del row["as_host"]
    else:
        row["as"] = "as1"
    with pytest.raises(Refusal) as caught:
        configuration.validate("toolchains", {"schema": 1, "toolchain": {"invalid": row}}, "toolchains.toml")
    assert all(f.key == "config.schema" for f in caught.value.findings)
    assert all(f.path.startswith("toolchains.toml:toolchain.invalid") for f in caught.value.findings)
    native_mock.assert_not_called()


def test_host_tools(adapter_config):
    expected = ("cpp", "mips_as", "mips_ld", "mips_objcopy", "n64link", "armips")
    assert adapters.host_tools(adapter_config) == {name: str(adapter_config.host.tools[name]) for name in expected}


@pytest.mark.parametrize("id,host_as", [("gcc-test", True), ("ido-7.1", False)])
def test_toolchain_tools(adapter_config, toolchains, id, host_as):
    chain = adapters.toolchain(adapter_config, id)
    row = toolchains["toolchain"][id]
    expected = adapter_config.host.tools["mips_as"] if host_as else adapter_config.host.toolchain_root / id / row["as"]
    assert chain.values()["as"] == [str(expected)]
    assert chain.values()["cc"] == [str(adapter_config.host.toolchain_root / id / row["cc"])]


@pytest.mark.parametrize("operation", ["toolchain", "assembler", "version"])
def test_unknown_refuses(adapter_config, recipe, toolchains, operation):
    with pytest.raises(Refusal) as caught:
        if operation == "toolchain":
            adapters.toolchain(adapter_config, "missing")
        elif operation == "assembler":
            adapters.assembler(adapter_config, "missing")
        else:
            adapters.script(adapter_config, recipe, "missing")
    assert caught.value.findings[0].key == "adapter.unknown"


def test_gnu_assembler_argv(adapter_config, native_mock, recipe, tmp_path):
    source, out = tmp_path / "input.s", tmp_path / "out.o"
    result = adapters.assembler(adapter_config, "gnu").assemble(source, recipe, [tmp_path / "include"], out)
    assert result is native_mock.return_value
    native_mock.assert_called_once_with(
        "assemble",
        [str(adapter_config.host.tools["mips_as"]), "-EB", f"-I{tmp_path / 'include'}", str(source), "-o", str(out)],
        adapter_config.project.root,
        outputs={"object": out},
        tmp=adapter_config.project.root / ".unbake" / "tmp",
    )


@pytest.mark.parametrize("exit,signal", [(1, None), (0, 9), (None, 15)])
def test_assemble_failures(adapter_config, native_mock, recipe, tmp_path, exit, signal):
    native_mock.return_value = replace(fixture.native_result(exit=exit, stderr=b"assembler failed"), signal=signal)
    with pytest.raises(Refusal) as caught:
        adapters.assembler(adapter_config, "gnu").assemble(tmp_path / "in.s", recipe, [], tmp_path / "out.o")
    assert caught.value.findings[0].key == "assemble.error"
    assert caught.value.findings[0].reason == "assembler failed"


def test_armips_wrapper_controls_output(adapter_config, native_mock, recipe, tmp_path):
    source, out = tmp_path / "resource.s", tmp_path / "resource.bin"
    source.write_text('.create "stale.bin", 0\n.word 1\n.close\n')
    adapters.assembler(adapter_config, "armips").assemble(source, recipe, [], out)
    wrapper = out.with_suffix(".armips.s")
    assert wrapper.read_text().count(".create") == 1
    assert "stale.bin" not in wrapper.read_text()
    assert f'.create "{out}", 0' in wrapper.read_text()
    assert wrapper.read_text().count(".close") == 1
    native_mock.assert_called_once_with(
        "armips",
        [str(adapter_config.host.tools["armips"]), str(wrapper), "-root", str(tmp_path)],
        tmp_path,
        outputs={"object": out},
        tmp=adapter_config.project.root / ".unbake" / "tmp",
    )


@pytest.mark.parametrize("phase", ["preprocess", "compile", "assemble"])
def test_toolchain_native_calls(adapter_config, recipe, toolchains, native_mock, tmp_path, phase):
    chain = adapters.toolchain(adapter_config, "gcc-test")
    source, out = tmp_path / "input.c", tmp_path / "output.o"
    if phase == "preprocess":
        chain.preprocess(source, recipe, "a", [tmp_path / "inc"], out)
        argv = native_mock.call_args.args[1]
        assert argv == [
            str(adapter_config.host.tools["cpp"]),
            "-E",
            "-E",
            "-DVERSION_A",
            f"-I{tmp_path / 'inc'}",
            str(source),
        ]
        scratch = adapter_config.project.root / ".unbake" / "tmp"
        assert native_mock.call_args.kwargs == {"stdout_path": out, "outputs": {"preprocessed": out}, "tmp": scratch}
    elif phase == "compile":
        chain.compile(source, recipe, out)
        assert native_mock.call_args.args[1] == [
            str(adapter_config.host.toolchain_root / "gcc-test" / "cc"),
            "-O2",
            "-c",
            str(source),
            "-o",
            str(out),
        ]
        scratch = adapter_config.project.root / ".unbake" / "tmp"
        assert native_mock.call_args.kwargs == {"outputs": {"object": out}, "tmp": scratch}
    else:
        chain.assemble(source, recipe, [], out)
        assert native_mock.call_args.args[1][0] == str(adapter_config.host.tools["mips_as"])


@pytest.mark.parametrize(
    "exit,outputs,key", [(2, {}, "compile.error"), (2, {"preprocessed": Path("out.i")}, "preprocess.error")]
)
def test_diagnose_native_error(adapter_config, toolchains, exit, outputs, key):
    result = fixture.native_result(
        exit=exit, outputs=outputs, stderr="\n".join(f"line {i}" for i in range(25)).encode()
    )
    findings = adapters.toolchain(adapter_config, "gcc-test").diagnose(result)
    assert findings[0].key == key
    assert findings[0].reason.splitlines() == [f"line {i}" for i in range(5, 25)]


def test_diagnose_ido_error_pattern(adapter_config, toolchains):
    chain = adapters.toolchain(adapter_config, "ido-7.1")
    message = "cfe: Warning 605: input.c: 4: #error stop"
    findings = chain.diagnose(fixture.native_result(stderr=f"before\n{message}\nafter\n".encode()))
    assert findings[0].key == "preprocess.error"
    assert findings[0].reason == message
    assert chain.diagnose(fixture.native_result(stderr=b"ordinary warning")) == ()


def test_refused_names_the_warning_lines_and_diagnose_never_fails_on_them(adapter_config, toolchains):
    chain = adapters.toolchain(adapter_config, "gcc-test")
    warning = "t.c:3: warning: passing arg 1 of `take' makes pointer from integer without a cast"
    stderr = f"t.c: In function `g':\n{warning}\nt.c:4: warning: unused variable `x'\n"
    assert chain.refused(stderr) == [warning]
    assert chain.diagnose(fixture.native_result(stderr=stderr.encode())) == ()


def test_script_quotes_words_and_preserves_variables(adapter_config, recipe, toolchains):
    recipe = replace(recipe, cflags=("-DVALUE=a b",))
    script = adapters.script(adapter_config, recipe, "a")
    assert script.startswith("#!/bin/sh\nset -e\n")
    assert """trap 'rm -f "$3.i" "$3.s"' EXIT""" in script  # the intermediates never outlive the compile
    assert '"$1" > "$3.i"' in script
    assert '"$3.i" -o "$3"' in script
    assert "'-DVALUE=a b'" in script


def test_script_emitted_assembly(adapter_config, recipe, toolchains):
    row = toolchains["toolchain"]["gcc-test"]
    row.update(emits_asm=True, assemble=["{as}", "{asflags}", "{source}", "-o", "{out}"])
    script = adapters.script(adapter_config, recipe, "a")
    assert len(script.splitlines()) == 6
    assert '"$3.i" -o "$3.s"' in script
    assert '"$3.s" -o "$3"' in script


@pytest.mark.parametrize("trim,score", [(False, False), (True, True)])
def test_link_commands_are_ld_and_one_objcopy_per_section(trim, score):
    tools = {"mips_objcopy": "$(OBJCOPY)", "mips_ld": "$(LD)"}
    sections = [
        Placement("a", ".text", 0x1000, 0x1008, 0x80000400),
        Placement("a", ".rodata", 0x1100, 0x1108, 0x80000500),
        Placement("a", ".bss", 0, 0, 0x80001000, 16),
    ]
    commands = adapters.link_commands(tools, Path("source.o"), sections, Path("work"), trim, score=score)
    ld, text, *rest = commands
    trimmed, (rodata,) = [c for c in rest if c[0] == "sh"], [c for c in rest if c[0] == "$(OBJCOPY)"]
    assert ld[0] == "$(LD)" and ld[-1] == "source.o" and ("--noinhibit-exec" in ld) == score
    assert text[:3] == ("$(OBJCOPY)", "-O", "binary") and "--set-section-flags" not in text
    assert rodata[rodata.index("--set-section-flags") + 1] == ".rodata=alloc,load,contents"
    assert len(trimmed) == (1 if trim else 0) and all(c[0] == "sh" for c in trimmed)
    assert not any("place" in w or "normal.o" in w or "placed.o" in w for c in commands for w in c)
    assert not any(".bss" in c for c in commands)


def test_link_script_asserts_ownership():
    script = adapters.link_script([Placement("a", ".text", 0x1000, 0x1008, 0x80000400)], Path("symbols.ld"))
    assert ".text 0x80000400  : SUBALIGN(4) {" in script  # a wider input alignment would move the code up
    assert 'INCLUDE "symbols.ld"' in script
    assert ".text 0x80000400" in script
    assert 'ASSERT(SIZEOF(.text) == 0x8, "owned .text size")' in script
    assert 'ASSERT(SIZEOF(.data) == 0x0, "owned .data size")' in script
    assert 'ASSERT(SIZEOF(.unowned) == 0, "unowned section")' in script


def test_link_script_discards_the_standard_mips_sections_from_abucache_alone():
    script = adapters.link_script([Placement("a", ".text", 0x1000, 0x1008, 0x80000400)], Path("symbols.ld"))
    assert (
        "/DISCARD/ : { *(.MIPS.abiflags .reginfo .pdr .mdebug* .options .debug* .comment .note* .gnu.attributes) }"
        in script.splitlines()
    )


def _link_inputs(tmp_path):
    obj, symbols = tmp_path / "source.o", tmp_path / "symbols.ld"
    obj.write_bytes(fixture.elf_object(["f"]))
    symbols.write_text("f = 0x80000400;\nunused = 0x80000500;\n")
    return obj, symbols


@pytest.mark.parametrize("mode", ["exact", "failed", "missing"])
def test_linker_outputs(adapter_config, native_mock, tmp_path, mode):
    sections = (Placement("a", ".text", 0x1000, 0x1008, 0x80000400), Placement("a", ".bss", 0, 0, 0x80001000, 16))
    work, (obj, symbols) = tmp_path / "work", _link_inputs(tmp_path)

    def run(name, argv, cwd, **kw):  # the whole chain is one shell
        assert argv[:2] == ["/bin/sh", "-c"]
        if mode == "failed":
            return fixture.native_result(exit=2, stderr=b"link failed")
        (work / "linked.elf").write_bytes(b"elf")
        if mode != "missing":
            (work / "text.bin").write_bytes(b"x" * 8)
        return fixture.native_result(argv=argv)

    native_mock.side_effect = run
    link = adapters.linker(adapter_config).link
    if mode != "exact":
        with pytest.raises(Refusal) as caught:
            link(obj, sections, symbols, work, False)
        assert caught.value.findings[0].key == "link.error"
        return
    result = link(obj, sections, symbols, work, False)
    assert set(result.outputs) == {"elf", ".text", ".bss"} and result.outputs[".bss"].read_text() == "16"
    chain = native_mock.call_args.args[1][2]
    assert native_mock.call_count == 1 and chain.index("--noinhibit-exec") < chain.index("objcopy")  # a draft is scored
    assert (work / "needed.ld").read_text() == "f = 0x80000400;\n"  # ld never sees "unused"


@pytest.fixture
def chains(toolchains):
    """gcc-test compiles to an object. fake-9 compiles to assembly, assembles it, and names a tool no row names."""
    toolchains["toolchain"]["fake-9"] = {
        **deepcopy(toolchains["toolchain"]["gcc-test"]),
        "emits_asm": True,
        "compile": ["{cc}", "{armips}", "{source}", "-o", "{out}"],
        "assemble": ["{n64link}", "asn64", "--as", "{as}", "{asflags}", "{source}", "-o", "{out}"],
        "pins": {"cc": "d" * 64},
    }
    return toolchains


@pytest.mark.parametrize("id,names", [("gcc-test", ["source.o"]), ("fake-9", ["source.s", "source.o"])])
def test_chain_steps_are_neutral(adapter_config, chains, recipe, id, names):
    steps = adapters.chain_steps(adapter_config, id, recipe, [Path("/project/include")], "unit")
    assert [s.out for s in steps] == names
    assert [s.phase for s in steps] == ["compile", "assemble"][: len(names)]
    assert steps[0].argv[-3:] == ("{in}", "-o", "{out}") or "{in}" in steps[0].argv
    assert all("{in}" in s.argv and "{out}" in s.argv for s in steps)
    assert not any("VERSION" in w for s in steps for w in s.argv)  # nothing of a version reaches a cached step


def _tool_files(config, names):
    for name in names:
        path = config.host.toolchain_root / name if "/" in name else config.host.tools[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())


@pytest.mark.parametrize("id,extra", [("gcc-test", False), ("fake-9", True)])
def test_tool_identity_follows_the_bytes_of_the_tools_the_commands_name(adapter_config, chains, id, extra):
    _tool_files(adapter_config, [f"{id}/cc", "n64link", "mips_as", "armips", "mips_ld"])
    first = adapters.tool_identity(adapter_config, id)
    assert adapters.tool_identity(adapter_config, id) == first
    (adapter_config.host.toolchain_root / id / "cc").write_bytes(b"another compiler")
    changed = adapters.tool_identity(adapter_config, id)
    assert changed != first
    adapter_config.host.tools["mips_ld"].write_bytes(b"another linker")  # not named by any command
    assert adapters.tool_identity(adapter_config, id) == changed
    adapter_config.host.tools["armips"].write_bytes(b"another armips")  # named by fake-9's compile command only
    assert (adapters.tool_identity(adapter_config, id) != changed) is extra


def test_pins_change_the_tool_identity(adapter_config, chains):
    _tool_files(adapter_config, ["gcc-test/cc", "mips_as"])
    first = adapters.tool_identity(adapter_config, "gcc-test")
    chains["toolchain"]["gcc-test"]["pins"] = {"cc": "e" * 64}
    assert adapters.tool_identity(adapter_config, "gcc-test") != first


def test_a_failed_chain_step_is_refused_as_the_direct_calls_refuse(adapter_config, chains, recipe, tmp_path):
    chain = adapters.toolchain(adapter_config, "fake-9")
    steps = adapters.chain_steps(adapter_config, "fake-9", recipe, [], "unit")
    chain.check(steps[0], fixture.native_result())
    with pytest.raises(Refusal) as caught:
        chain.check(steps[0], fixture.native_result(exit=1, stderr=b"compile failed"))
    assert caught.value.findings[0].key == "compile.error"
    with pytest.raises(Refusal) as caught:
        chain.check(steps[1], fixture.native_result(exit=1, stderr=b"assembler failed"))
    assert caught.value.findings[0].key == "assemble.error"


_PIN = adapters._pin  # the real reading, before the autouse fixture pins it to the installed abucache


@pytest.mark.parametrize(("installed", "refused"), [("0.3.3", False), ("0.1.0", True)])
def test_abucache_must_be_the_version_this_unbake_requires(monkeypatch, installed, refused):
    """The pin comes from the requirement unbake was installed with (canned here: the runner's venv may predate a
    pin bump), and setup refuses any other installed abucache."""
    wheel = "abucache @ https://example.invalid/releases/download/v0.3.3/abucache-0.3.3-py3-none-any.whl"
    requires = {"unbake": ["click==8.5.0", wheel]}
    monkeypatch.setattr(adapters, "_pin", _PIN)
    monkeypatch.setattr(adapters.metadata, "requires", lambda name: requires[name])
    monkeypatch.setattr(adapters.abucache, "__version__", installed)
    if not refused:
        adapters.check_abucache()
        return
    with pytest.raises(Refusal) as caught:
        adapters.check_abucache()
    (finding,) = caught.value.findings
    assert (finding.key, finding.reason) == ("tool.version", "abucache version mismatch: expected 0.3.3, found 0.1.0")


def test_unused_tools_not_resolved(adapter_config, toolchains):
    cfg = replace(adapter_config, host=replace(adapter_config.host, tools={}))
    chain = adapters.toolchain(cfg, "gcc-test")
    assert chain.argv("compile", codegen=(), source=("in.i",), out=("out.o",))[0].endswith("/cc")
    assert adapters.host_tools(cfg) == {}


def test_n64link_required_only_when_the_row_names_it(adapter_config, chains, monkeypatch):
    without = replace(
        adapter_config,
        host=replace(adapter_config.host, tools={k: v for k, v in adapter_config.host.tools.items() if k != "n64link"}),
    )
    assert adapters.toolchain(without, "gcc-test").argv("compile", codegen=(), source=("a",), out=("b",))

    def real(config, name, **kw):
        if name not in config.host.tools:
            raise Refusal(Finding("native.missing_tool", reason=f"host tools.{name} is not configured"))
        return config.host.tools[name]

    monkeypatch.setattr(process, "tool", real)
    parts = {"asflags": (), "includes": (), "source": ("a",), "out": ("b",)}
    with pytest.raises(Refusal) as caught:
        adapters.toolchain(without, "fake-9").argv("assemble", **parts)
    (finding,) = caught.value.findings
    assert finding.key == "native.missing_tool" and "n64link" in finding.reason
    assert adapters.toolchain(adapter_config, "fake-9").argv("assemble", **parts)[0].endswith("n64link")


def test_armips_missing_names_resource(adapter_config, native_mock, tmp_path, monkeypatch):
    cfg = replace(adapter_config, host=replace(adapter_config.host, tools={}))
    monkeypatch.undo()
    source = tmp_path / "input.s"
    source.write_text(".word 0")
    with pytest.raises(Refusal) as caught:
        adapters.assembler(cfg, "armips").assemble(source, None, (), tmp_path / "out.bin")
    (finding,) = caught.value.findings
    assert finding.key == "native.missing_tool"
    assert "armips" in finding.reason and "resource" in finding.reason


def test_an_assemble_template_without_includes_still_gets_the_include_directories(adapter_config, chains, recipe):
    """The shipped gcc rows spell no {includes} (the Makefile adds $(ASINC) to $(ASFLAGS)): a stand-in's .include of
    nonmatchings/... must still find the extracted assembly, so the directories follow the flags."""
    steps = adapters.chain_steps(adapter_config, "fake-9", recipe, [Path("/project/asm/a")], "unit")
    assert steps[1].argv[-4:] == ("-I/project/asm/a", "{in}", "-o", "{out}")


def test_a_row_that_leaves_rdata_unallocated_gets_the_makefiles_objcopy_step(adapter_config, chains, recipe):
    plain = adapters.chain_steps(adapter_config, "fake-9", recipe, [], "unit")
    assert [s.phase for s in plain] == ["compile", "assemble"]
    chains["toolchain"]["fake-9"]["rdata_alloc"] = True
    steps = adapters.chain_steps(adapter_config, "fake-9", recipe, [], "unit")
    assert [s.phase for s in steps] == ["compile", "assemble", "objcopy"]
    flags = steps[2].argv
    assert flags[1:] == ("--set-section-flags", ".rdata=alloc,load,readonly,data,contents", "{in}", "{out}")
    assert flags[0] == str(adapter_config.host.tools["mips_objcopy"])
    assert adapter_config.host.tools["mips_objcopy"] in adapters.toolchain(adapter_config, "fake-9").tool_files()
