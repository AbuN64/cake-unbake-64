"""Target naming policy through public API, source admission and the existing audit gate; no native builds."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from unbake import api, config, policy
from unbake.contracts import Refusal


@pytest.mark.parametrize(
    "name",
    [
        "class",
        "new",
        "delete",
        "restrict",
        "co_await",
        "__declspec",
        "__thread",
        "__block",
        "__private",
        "_Global",
        "_global",
        "foo__bar",
        "a$b",
        "éclair",
        "1name",
        "a-b",
        "A" * 248,
    ],
)
def test_every_target_must_accept_a_proposed_symbol(name):
    assert api.name_issues(name)


@pytest.mark.parametrize("name", ["Menu_EvaluateCameraSpline", "Menu_EvaluateCameraSpline_value", "foo", "A" * 247])
def test_portable_identifiers(name):
    assert api.name_issues(name) == ()


def test_target_table_is_valid_generic_and_records_abi_distinctions():
    table = config.load_resource("symbol-names.toml")
    assert set(table["target"]) == {"elf-gcc", "coff-mingw", "macho-clang", "android-clang", "msvc"}
    for row in table["target"].values():
        assert row["symbol_case_sensitive"]
        assert row["cxx_linkage"] == 'extern "C"'
        assert row["explicit_asm_labels"] is False
    assert table["target"]["coff-mingw"]["symbol_prefix"] == {"i386": "_", "x86_64": "", "arm64": ""}
    assert table["target"]["macho-clang"]["section_max"] == 16
    assert not any(game in str(table).lower() for game in ("ragewars", "battletanx", "turok"))


@pytest.mark.parametrize(
    "path", ["src/CON.c", "src/aux/header.h", "src/a?.c", "src/../foo.c", "/src/foo.c", "src/" + "a" * 256, "src/ends."]
)
def test_file_components_and_windows_devices(path):
    assert api.name_issues(path, "file")


def test_collision_targets_and_section_limits():
    why = api.name_issues("Draw", taken=("draw",))
    assert len(why) == 1 and "coff-mingw" in why[0] and "msvc" in why[0] and "elf-gcc" not in why[0]
    assert api.name_issues(".text", "section") == ()
    assert "coff-mingw" in api.name_issues(".longtext", "section")[0]
    assert any("macho-clang" in why for why in api.name_issues(".section_name_too_long", "section"))
    assert api.name_issues("bad section", "section")


@pytest.mark.parametrize(
    "text",
    [
        'extern int Function(void) __asm__("Other");',
        'extern int Function(void) asm /* gap */ ("Other");',
        'extern int value __asm ("Other");',
        'extern int value[3]\n__asm__\n("Other");',
        '#define LABEL "Other"\nextern int Function(void) __asm__(LABEL);',
        "#define BIND(label) __asm__(label)\n",
        'int Function(void) __attribute__((alias("Other")));',
        "#pragma redefine_extname Function Other\n",
        '#pragma comment(linker, "/alternatename:Function=Other")\n',
        'extern "C++" int Function(void);',
        'int data __attribute__((section(".section_name_too_long")));',
        'int data;\n#pragma data_seg("section_too_long")\n',
    ],
)
def test_text_gate_refuses_labels_aliases_sections_and_mangled_linkage(text):
    assert api.naming_findings({"src/game.c": text})
    assert any(f.key == "source.symbol-name" for f in policy.text_findings("src/game.c", text, False))


@pytest.mark.parametrize(
    "text",
    [
        '/* extern int Function(void) __asm__("Other"); */\nint Function(void);',
        'const char *message = "class new alias(\\"x\\")";\n',
        "#define Function Function_value\nextern int Function(void);\n"
        "#undef Function\nint Function(void) { return Function_value(); }",
        'extern "C" { int Function(void); }\n',
        "void Function(unsigned long, int *, void (*callback)(int));",
        "void Function(void) { int _local; _local = 0; }",
        'void Function(void) { __asm__ volatile ("move $v0,$zero"); }',
        'void Function(void) { asm("nop"); }',
    ],
)
def test_labels_are_distinct_from_comments_strings_c_aliases_and_inline_asm(text):
    assert api.naming_findings({"src/game.c": text}) == ()


@pytest.mark.parametrize(
    "text",
    [
        "int class;",
        "void Function(int new);",
        "struct Widget { int delete; };",
        "int __private(void);",
        "int _reserved;",
        "int dollar$name;",
        "int café;",
        "#define class Draw\n",
        "int good, class;",
        "typedef struct { int member; } class;",
        "enum Value { new = 0, valid = 1 };",
        "void Function(void) { for (int class = 0; class < 3; ++class) {} }",
    ],
)
def test_declarations_parameters_fields_and_macros(text):
    assert api.naming_findings({"src/game.c": text})


def test_cross_file_collisions_and_static_namespaces():
    assert api.naming_findings({"src/A.c": "", "src/a.c": ""})
    assert api.naming_findings({"src/Actors/a.c": "", "src/actors/b.c": ""})
    assert api.naming_findings({"src/a.c": "int Draw(void);", "src/b.c": "int draw(void);"})
    assert api.naming_findings({"src/a.c": "int Draw(void);", "src/b.c": "int Draw(void);"}) == ()
    assert api.naming_findings({"src/a.c": "static int Draw(void);", "src/b.c": "static int draw(void);"}) == ()
    assert (
        api.naming_findings({"src/a.c": "void A(void) { int value; }", "src/b.c": "void B(void) { int Value; }"}) == ()
    )


def test_audit_fails_unchanged_tracked_sources_without_a_host(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.c").write_text('extern int Function(void) __asm__("Other");\n')
    monkeypatch.setattr(policy.shutil, "which", lambda _: "git")
    monkeypatch.setattr(policy.process, "scratch", lambda root: tmp_path)

    def run(name, argv, root, **kwargs):
        return SimpleNamespace(exit=0, stdout=b"src/a.c\0" if argv[1] == "ls-files" else b"")

    runner = Mock(side_effect=run)
    monkeypatch.setattr(policy.process, "run", runner)
    with pytest.raises(Refusal) as error:
        policy.audit_command({"project": str(tmp_path), "against": "HEAD"})
    assert any("explicit asm symbol label" in f.reason for f in error.value.findings)
    assert {call.args[1][1] for call in runner.call_args_list} == {"diff", "ls-files"}


def test_audit_respects_configured_library_roots(tmp_path: Path, monkeypatch):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor/a.c").write_text("int __vendor(void);\n")
    (tmp_path / "config.toml").write_text('[library."vendor/"]\n')
    monkeypatch.setattr(policy.shutil, "which", lambda _: "git")
    monkeypatch.setattr(policy.process, "scratch", lambda root: tmp_path)
    monkeypatch.setattr(
        policy.process,
        "run",
        lambda name, argv, root, **kw: SimpleNamespace(
            exit=0, stdout=b"vendor/a.c\0" if argv[1] == "ls-files" else b""
        ),
    )
    monkeypatch.setattr(policy, "_offset_audit", lambda *a: ([], []))
    monkeypatch.setattr(policy, "_removed_functions", lambda *a: 0)
    monkeypatch.setattr(policy, "_credit_drops", lambda *a: {})
    assert policy.audit_command({"project": str(tmp_path), "against": "HEAD"})["counts_after"] == {}


@pytest.mark.parametrize(
    "text",
    [
        "typedef struct { int member; } Record; typedef union { int bits; float value; } Value;",
        "void Function(void) { while (1) { if (1) {} break; } if (1) {} return; }",
        "void Function(void) { while (1) { if (1) {} continue; } }",
        "Effect *effect = Allocate(1, 0x24, sizeof(Effect));",
        "int values[] = {1, 2, 3};",
        "int a = Call(1, 2), b = Call(3, 4);",
    ],
)
def test_declaration_scanner_does_not_treat_syntax_or_initializer_arguments_as_names(text):
    assert api.naming_findings({"src/game.c": text}) == ()


def test_comma_declarator_after_initializer_is_still_checked():
    findings = api.naming_findings({"src/game.c": "int a = Call(1, 2), class = 0;"})
    assert any("class: compiler keyword" in f.reason for f in findings)


@pytest.mark.parametrize(
    "text",
    [
        "void Function(int value) { switch(value) { case 1: int class; break; } }",
        "int * const private;",
        '_Pragma("redefine_extname Function Other")',
        "static int Draw(void); static int draw(void);",
    ],
)
def test_source_naming_boundaries(text):
    assert api.naming_findings({"src/game.c": text})


def test_sdk_reserved_names_pass_as_implementation_names_only():
    assert api.name_issues("__osDisableInt")
    assert api.name_issues("__osDisableInt", implementation=True) == ()
    assert api.name_issues("__secret", implementation=True)


def test_the_whole_tree_census_reads_an_unchanged_file_once():
    from unbake import symbol_names

    files = {"src/a.c": "int Foo_bar;\n", "src/b.c": "int Baz(void) { return 1; }\n"}
    symbol_names._per_file.cache_clear()
    first = symbol_names.findings(files)
    again = symbol_names.findings({**files, "src/b.c": files["src/b.c"] + "int Qux;\n"})
    info = symbol_names._per_file.cache_info()
    assert info.hits >= 1 and info.misses == 3  # a.c and b.c once, the edited b.c again
    assert symbol_names.findings(files) == first and isinstance(again, tuple)
