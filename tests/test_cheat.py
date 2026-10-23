"""The no-cheat check on tiny synthetic git histories: facts and the generated Makefile are mocked, no ROM or build."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from unbake import api, cheat, symbols, tree
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


def test_makefile_rows_pass_other_lines_fail_unless_generated(
    repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = "$(foreach v,$(VERSIONS),build/$(v)/src/a.o): TC := ido\n"
    assert repo.check(Makefile="all:\n" + row)["makefile"] == "unchanged"
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
