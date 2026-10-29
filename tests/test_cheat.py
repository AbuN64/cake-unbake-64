"""The no-cheat check on tiny synthetic git histories: facts and the generated Makefile are mocked, no ROM or build."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from unbake import api, build, cheat, symbols, tree
from unbake import config as configuration
from unbake.contracts import Finding, Refusal

STAND_IN = 'INCLUDE_RODATA("data_800C", D_auto_rodata_func_8040BB40_de_800DD07C);\n'
FACTS = {
    "D_auto_rodata_func_8040BB40_de_800DD07C": (0x800DD07C, 0xDDC7C, 4, ""),
    "func_80000000": (0x80000000, 0x1000, 8, "func"),
}


def _git(root: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": "/usr/bin:/bin",
    }
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True, env=env
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.c").write_text(
        '#include "include_asm.h"\nint f(void) { return 1; }\nINCLUDE_ASM("seg", func_80000000);\n'
    )
    (tmp_path / "Makefile").write_text("all:\n$(foreach v,$(VERSIONS),build/$(v)/src/a.o): TC := gcc\n")
    (tmp_path / "config.toml").write_text("x\n")
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / ".gitignore").write_text(".unbake/\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path, versions=("de",)), host=None)
    monkeypatch.setattr(api, "symbols", lambda cfg, v: FACTS)
    monkeypatch.setattr(symbols, "segment_at", lambda cfg, v, rom: "data_800C")
    monkeypatch.setattr(cheat, "_tree_findings", lambda cfg: ())
    monkeypatch.setattr(cheat, "_makefile_generated", lambda cfg, git, head, rows: False)
    base = _git(tmp_path, "rev-parse", "HEAD")

    def check(**files: str | None) -> dict[str, Any]:
        for name, text in files.items():
            path = tmp_path / PATHS.get(name, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text or "")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-qm", "change")
        return cheat.cheat_command(config, {"against": base})

    return SimpleNamespace(check=check, root=tmp_path, base=base, config=config)


def _refused(repo: SimpleNamespace, **files: str | None) -> set[str]:
    with pytest.raises(Refusal) as refused:
        repo.check(**files)
    return {f.key for f in refused.value.findings}


PATHS = {"blobs_toml": "blobs.toml", "src__a_c": "src/a.c", "src__x_s": "src/x.s", "local_mk": "local.mk"}
A = '#include "include_asm.h"\nint f(void) { return 1; }\nINCLUDE_ASM("seg", func_80000000);\n'


def test_plain_c_passes(repo: SimpleNamespace) -> None:
    assert repo.check(src__a_c="int f(void) { return 2; }\n")["stand_ins"] == []


def test_a_stand_in_the_tool_would_write_passes(repo: SimpleNamespace) -> None:
    """The INCLUDE_RODATA line `sync --place` writes for a symbol the facts name and no C defined."""
    result = repo.check(src__a_c=A + "#if defined(VERSION_DE)\n" + STAND_IN + "#endif\n")
    assert result["stand_ins"] == ["D_auto_rodata_func_8040BB40_de_800DD07C"]


def test_a_stand_in_for_a_symbol_no_fact_names_fails(repo: SimpleNamespace) -> None:
    assert _refused(repo, src__a_c=A + 'INCLUDE_RODATA("data_800C", D_made_up);\n') == {"cheat.added"}


def test_a_stand_in_in_the_wrong_segment_fails(repo: SimpleNamespace) -> None:
    assert _refused(repo, src__a_c=A + STAND_IN.replace("data_800C", "elsewhere")) == {"cheat.added"}


def test_a_stand_in_hiding_a_c_definition_fails(repo: SimpleNamespace) -> None:
    """The name was C at the base: turning it into a stand-in is a hidden body, whatever the facts say."""
    (repo.root / "src/b.c").write_text("const int D_auto_rodata_func_8040BB40_de_800DD07C[1] = {1};\n")
    _git(repo.root, "add", "-A")
    _git(repo.root, "commit", "-qm", "c")
    repo.base = _git(repo.root, "rev-parse", "HEAD")
    (repo.root / "src/b.c").write_text(STAND_IN)
    _git(repo.root, "add", "-A")
    _git(repo.root, "commit", "-qm", "hide")
    with pytest.raises(Refusal) as hidden:
        cheat.cheat_command(repo.config, {"against": repo.base})
    assert {f.key for f in hidden.value.findings} == {"cheat.added"}


def test_a_hand_added_include_asm_fails(repo: SimpleNamespace) -> None:
    assert _refused(repo, src__a_c=A + 'INCLUDE_ASM("seg", func_80000004);\n') == {"cheat.include-asm"}


@pytest.mark.parametrize(
    "line",
    [
        ".word 0x1234",
        'asm("nop");',
        '__asm__ ("nop");',
        "#pragma GCC optimize 0",
        'int x __attribute__((section(".x")));',
        '__asm("nop");',
    ],
)
def test_assembly_words_and_pragmas_fail(repo: SimpleNamespace, line: str) -> None:
    assert _refused(repo, src__a_c=A + line + "\n") == {"cheat.added"}


def test_a_non_matching_only_branch_is_exempt_from_the_scan(repo: SimpleNamespace) -> None:
    body = A + '#ifdef NON_MATCHING\nint g(void) { asm("nop"); }\n#endif\n'
    assert repo.check(src__a_c=body)["stand_ins"] == []


def test_an_unreadable_condition_keeps_lines_in_scope(repo: SimpleNamespace) -> None:
    body = A + "#if defined(NON_MATCHING) +\n.word 1\n#endif\n"
    assert _refused(repo, src__a_c=body) == {"cheat.added"}


def test_a_non_c_file_under_src_fails(repo: SimpleNamespace) -> None:
    assert _refused(repo, src__x_s="nop\n") == {"cheat.non-c"}


def _handwritten(monkeypatch: pytest.MonkeyPatch, names: set[str]) -> None:
    monkeypatch.setattr(symbols, "handwritten", lambda config, version, name: name in names)


ASM = ".globl func_a_de\nfunc_a_de:\n    .word 0x40845800\n    jr $ra\n"


def test_a_handwritten_only_new_asm_file_in_an_asm_home_passes(repo: SimpleNamespace, monkeypatch) -> None:
    _handwritten(monkeypatch, {"func_a_de"})
    assert repo.check(**{"src/sdk/x.s": ASM})["against"] == repo.base


def test_a_new_asm_file_with_a_compiled_function_fails(repo: SimpleNamespace, monkeypatch) -> None:
    _handwritten(monkeypatch, {"func_a_de"})
    both = ASM + ".globl func_b_de\nfunc_b_de:\n    jr $ra\n"
    assert "cheat.non-c" in _refused(repo, **{"src/sdk/x.s": both})


def test_a_new_asm_file_in_the_game_dir_fails(repo: SimpleNamespace, monkeypatch) -> None:
    _handwritten(monkeypatch, {"func_a_de"})
    assert "cheat.non-c" in _refused(repo, **{"src/game/x.s": ASM})


def test_a_new_asm_file_defining_no_global_fails(repo: SimpleNamespace, monkeypatch) -> None:
    _handwritten(monkeypatch, set())
    assert _refused(repo, **{"src/sdk/x.s": "nop\n"}) == {"cheat.non-c"}


def test_infrastructure_edits_fail(repo: SimpleNamespace) -> None:
    assert _refused(repo, local_mk="X=1\n") == {"cheat.infra"}


HEADER = (
    '#ifndef INCLUDE_ASM_H\n#define INCLUDE_ASM_H\n#include "TYPES.H"\n'
    "#define INCLUDE_ASM(SEG, NAME) \\\n    __asm__(NAME)\n#endif\n"
    "extern void func_80289970_de(int a);\nextern int D_800C0000;\n"
)


def _header(repo: SimpleNamespace, text: str) -> dict[str, Any]:
    """Check HEADER -> TEXT as include/include_asm.h, an infrastructure path, over its own base."""
    path = repo.root / "include/include_asm.h"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER)
    _git(repo.root, "add", "-A")
    _git(repo.root, "commit", "-qm", "header")
    base = _git(repo.root, "rev-parse", "HEAD")
    path.write_text(text)
    _git(repo.root, "add", "-A")
    _git(repo.root, "commit", "-qm", "edit")
    return cheat.cheat_command(repo.config, {"against": base})


def test_a_declaration_removed_from_a_mixed_header_is_not_infrastructure(repo: SimpleNamespace) -> None:
    assert _header(repo, HEADER.replace("extern void func_80289970_de(int a);\n", ""))["against"]


def test_a_declaration_added_or_edited_in_a_mixed_header_is_not_infrastructure(repo: SimpleNamespace) -> None:
    assert _header(repo, HEADER.replace("(int a)", "(int a, int b)") + "extern void g(void);\n")["against"]


@pytest.mark.parametrize(
    "edit",
    [
        ("#define INCLUDE_ASM(SEG, NAME) \\\n    __asm__(NAME)", "#define INCLUDE_ASM(SEG, NAME) \\\n    __asm__(SEG)"),
        ("#endif\nextern void func", "#endif\n#define X 1\nextern void func"),
        ('#include "TYPES.H"\n', ""),
        ("extern int D_800C0000;", "int D_800C0000 = 1;"),
    ],
)
def test_a_macro_include_or_definition_edit_of_a_mixed_header_stays_infrastructure(repo, edit) -> None:
    with pytest.raises(Refusal) as refused:
        _header(repo, HEADER.replace(*edit))
    assert "cheat.infra" in {f.key for f in refused.value.findings}


def test_a_declaration_edit_cannot_carry_an_asm_line_in(repo: SimpleNamespace) -> None:
    with pytest.raises(Refusal) as refused:
        _header(repo, HEADER + "#pragma weak x\nextern void g(void);\n")
    assert {f.key for f in refused.value.findings} >= {"cheat.infra", "cheat.added"}


def test_makefile_rows_pass_other_lines_fail_unless_generated(
    repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = "$(foreach v,$(VERSIONS),build/$(v)/src/a.o): TC := ido\n"
    assert repo.check(Makefile="all:\n" + row)["makefile"] == "behind"  # row-only edit, unstamped Makefile
    assert _refused(repo, Makefile="all:\n" + row + "evil:\n\tcp a b\n") == {"cheat.makefile"}
    monkeypatch.setattr(cheat, "_makefile_generated", lambda cfg, git, head, rows: True)
    assert cheat.cheat_command(repo.config, {"against": repo.base})["makefile"] == "generated"


def test_a_dirty_fact_tree_is_refused(repo: SimpleNamespace) -> None:
    (repo.root / "config.toml").write_text("changed\n")
    with pytest.raises(Refusal) as refused:
        cheat.cheat_command(repo.config, {"against": repo.base})
    assert refused.value.findings[0].key == "cheat.dirty"


def test_a_renamed_file_adds_only_its_changed_lines(repo: SimpleNamespace) -> None:
    _git(repo.root, "mv", "src/a.c", "src/b.c")
    assert repo.check()["stand_ins"] == []


def test_a_name_added_twice_to_the_facts_fails(repo: SimpleNamespace) -> None:
    rows = "lit_x = 0x800E16C8; // rom:0xE22C8\nlit_x = 0x800E173C; // rom:0xE233C\n"
    assert _refused(repo, **{"versions/de/aliases.txt": rows}) == {"cheat.duplicate-name"}


def test_a_name_in_both_fact_files_fails(repo: SimpleNamespace) -> None:
    keys = _refused(
        repo, **{"versions/de/symbol_addrs.txt": "lit_y = 0x1;\n", "versions/de/aliases.txt": "lit_y = 0x2;\n"}
    )
    assert keys == {"cheat.duplicate-name"}


def test_unique_names_pass(repo: SimpleNamespace) -> None:
    assert repo.check(**{"versions/de/aliases.txt": "lit_a = 0x1;\nlit_b = 0x2;\n"})["head"] == "HEAD"


def test_a_duplicate_already_at_base_is_not_new(repo: SimpleNamespace) -> None:
    (repo.root / "versions/de").mkdir(parents=True)
    (repo.root / "versions/de/aliases.txt").write_text("dup = 0x1;\ndup = 0x2;\n")
    _git(repo.root, "add", "-A")
    _git(repo.root, "commit", "-qm", "old")
    base = _git(repo.root, "rev-parse", "HEAD")
    (repo.root / "src/a.c").write_text(A + "int g;\n")
    _git(repo.root, "commit", "-qam", "c")
    assert cheat.cheat_command(repo.config, {"against": base})["head"] == "HEAD"


LITERALS = Finding("tree.literals", "literals_f_2 in de places literals for f_2, which no file owns")


def test_a_tree_that_cannot_be_captured_fails(repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `literals_` row for a function no file owns: tree capture refuses it, so the gate must too."""
    monkeypatch.setattr(cheat, "_tree_findings", lambda cfg: (LITERALS,))
    assert _refused(repo, src__a_c="int f(void) { return 2; }\n") == {"tree.literals"}


def test_tree_findings_are_the_capture_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(cfg: object) -> None:
        raise Refusal(LITERALS)

    monkeypatch.setattr(tree, "capture", refuse)
    assert cheat._tree_findings(SimpleNamespace()) == (LITERALS,)
    monkeypatch.setattr(tree, "capture", lambda cfg: None)
    assert cheat._tree_findings(SimpleNamespace()) == ()


def test_blobs_toml_is_a_fact_not_infrastructure_and_is_verified_when_it_changes(
    repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unbake import blobs

    seen: list[int] = []
    monkeypatch.setattr(blobs, "verify", lambda cfg, base=None: seen.append(1) or ())
    repo.check(blobs_toml="schema = 1\n")
    assert seen == [1]
    base = _git(repo.root, "rev-parse", "HEAD")
    (repo.root / "src/a.c").write_text("int f(void) { return 3; }\n")
    _git(repo.root, "add", "-A")
    _git(repo.root, "commit", "-qm", "c only")
    cheat.cheat_command(repo.config, {"against": base})
    assert seen == [1]  # blobs.toml did not differ: no verification


def test_a_blobs_toml_that_fails_verification_is_refused(
    repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unbake import blobs

    monkeypatch.setattr(blobs, "verify", lambda cfg, base=None: (Finding("blobs.evidence", "bad sha"),))
    assert _refused(repo, blobs_toml="schema = 1\n") == {"blobs.evidence"}


def test_the_uncommitted_tree_is_checked_without_a_commit(
    repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cheat, "_makefile_generated", lambda cfg, git, head, rows: True)
    head = _git(repo.root, "rev-parse", "HEAD")
    (repo.root / "src/a.c").write_text(A + "int g;\n")  # tracked change, not committed
    (repo.root / "src/new.s").write_text("nop\n")  # untracked file under the source roots
    with pytest.raises(Refusal) as refused:
        cheat.cheat_command(repo.config, {"against": repo.base, "head": "WORKTREE"})
    assert {f.key for f in refused.value.findings} == {"cheat.non-c"}
    (repo.root / "src/new.s").unlink()
    done = cheat.cheat_command(repo.config, {"against": repo.base, "head": "WORKTREE"})
    assert done["head"] == "WORKTREE"
    assert _git(repo.root, "rev-parse", "HEAD") == head and not _git(repo.root, "diff", "--cached", "--name-only")
    assert _git(repo.root, "status", "--porcelain") == "M src/a.c"


def test_a_makefile_behind_the_generator_is_mains_lag_not_a_refusal(
    repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo.root / "src/a.c").write_text(A + "int g;\n")
    run = lambda: cheat.cheat_command(repo.config, {"against": repo.base, "head": "WORKTREE"})["makefile"]  # noqa: E731
    monkeypatch.setattr(build, "generator_version", lambda: "abc123def456")
    assert run() == "behind"  # the committed Makefile has no generator stamp
    (repo.root / "Makefile").write_text("# t\n# generator: 000000000000\nall:\n")
    _git(repo.root, "add", "-A")
    _git(repo.root, "commit", "-qm", "other generator")
    repo.base = _git(repo.root, "rev-parse", "HEAD")
    assert run() == "behind"  # another generator version
    (repo.root / "Makefile").write_text("# t\n# generator: abc123def456\nall:\n")
    _git(repo.root, "add", "-A")
    _git(repo.root, "commit", "-qm", "current generator")
    repo.base = _git(repo.root, "rev-parse", "HEAD")
    assert run() == "unchanged"  # the deployed generator wrote it: nothing behind


def test_the_generator_version_follows_the_template_and_stamps_the_makefile() -> None:
    first = build.generator_version()
    assert re.fullmatch(r"[0-9a-f]{12}", first) and build.generator_version() == first
    assert build.generator_of(f"# t: matching build.\n# generator: {first}\nNAME := x\n") == first
    assert "# generator: ${generator}" in configuration.template("Makefile.in")


def test_the_uncommitted_tree_still_refuses_added_assembly(
    repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cheat, "_makefile_generated", lambda cfg, git, head, rows: True)
    (repo.root / "src/a.c").write_text(A + 'void h(void) { __asm__("nop"); }\n')
    with pytest.raises(Refusal) as refused:
        cheat.cheat_command(repo.config, {"against": repo.base, "head": "WORKTREE"})
    assert {f.key for f in refused.value.findings} == {"cheat.added"}


def test_a_base_that_moved_on_is_not_charged_to_the_head(repo: SimpleNamespace) -> None:
    """Main matched a function after the lane branched (its INCLUDE_ASM went away): the lane is judged against the
    merge-base, not charged with the INCLUDE_ASM line main removed."""
    _git(repo.root, "checkout", "-q", "-b", "ahead")
    (repo.root / "src/a.c").write_text('#include "include_asm.h"\nint f(void) { return 1; }\n')
    _git(repo.root, "commit", "-qam", "main matched func_80000000")
    ahead = _git(repo.root, "rev-parse", "HEAD")
    _git(repo.root, "checkout", "-q", repo.base)
    (repo.root / "src/b.c").write_text("int g(void) { return 3; }\n")
    _git(repo.root, "add", "-A")
    _git(repo.root, "commit", "-qm", "lane work")
    assert cheat.cheat_command(repo.config, {"against": ahead})["stand_ins"] == []


def test_an_added_sdk_trap_is_recorded_not_refused_and_game_code_is_refused(repo: SimpleNamespace) -> None:
    """10-25 REV1-ONE-72: the narrow SDK exception (fakematch.sdk_line); the owner still confirms the form."""
    repo.config.project.libraries = (Path("src/sdk/libsn"),)
    line = '    __asm__ volatile ("break 1");\n'
    got = repo.check(**{"src/sdk/libsn/s.c": "int S(int a) {\n    if (a) {\n" + line + "    }\n    return a;\n}\n"})
    assert got["sdk_asm"] == ['src/sdk/libsn/s.c: "break 1"']
    assert _refused(repo, src__a_c=A + "int G(int a) {\n" + line + "    return a;\n}\n") == {"cheat.added"}


MOVED = {
    "cheat": {
        "source": ["src", "include", "sdk"],
        "asm_home": ["sdk/"],
        "makefile_row": r"^\$\(foreach v,\$\(VERSIONS\),build/\$\(v\)/(src|sdk)/[^ ]+\.o\): (TC|CPPFLAGS|CFLAGS) :=",
    }
}


def test_a_project_cheat_table_replaces_the_packaged_asm_home(repo: SimpleNamespace, monkeypatch) -> None:
    _handwritten(monkeypatch, {"func_a_de"})
    repo.config.project.cheat = MOVED["cheat"]
    assert repo.check(**{"sdk/x.s": ASM})["against"] == repo.base


def test_a_project_cheat_table_leaves_the_old_asm_home_to_fail(repo: SimpleNamespace, monkeypatch) -> None:
    _handwritten(monkeypatch, {"func_a_de"})
    repo.config.project.cheat = MOVED["cheat"]
    assert "cheat.non-c" in _refused(repo, **{"src/sdk/x.s": ASM})


def test_a_project_cheat_table_replaces_the_makefile_row(repo: SimpleNamespace) -> None:
    repo.config.project.cheat = MOVED["cheat"]
    row = "$(foreach v,$(VERSIONS),build/$(v)/sdk/a.o): TC := ido\n"
    assert repo.check(Makefile="all:\n" + row)["makefile"] == "behind"


def test_the_layout_is_the_packaged_one_with_the_projects_keys_over_it(tmp_path: Path) -> None:
    packaged = configuration.load_resource("repo.toml")["layout"]
    assert configuration.layout(SimpleNamespace()) == packaged
    assert packaged["library_source"] == "src/sdk/{library}" and packaged["source_roots"] == ["src"]
    moved = {"source_roots": ["src", "sdk"], "library_source": "sdk/{library}"}
    assert configuration.layout(SimpleNamespace(layout=moved)) == {**packaged, **moved}
    (tmp_path / "config.toml").write_text('[layout]\nheader_roots = ["include", "sdk/include"]\n')
    assert api.layout(tmp_path)["header_roots"] == ["include", "sdk/include"]
    assert api.layout(tmp_path)["library_header"] == packaged["library_header"]
    assert api.layout(tmp_path / "none") == packaged  # no config.toml: the packaged layout


def test_a_layout_key_the_schema_does_not_know_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('[layout]\nsdk_root = "sdk"\n')
    with pytest.raises(Refusal):
        api.layout(tmp_path)
