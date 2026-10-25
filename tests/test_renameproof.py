"""The rename-map proof: a pure rename passes, everything else refuses so the build proves it."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from unbake import renameproof
from unbake.contracts import Refusal

ASM = (
    '#define INCLUDE_ASM(SEG, NAME) \\\n    __asm__(".include \\"nonmatchings/" SEG "/" #NAME ".s\\"\\n")\n'
    "#define STR(x) #x\n#define CAT(a, b) a##b\n"
)
FILES = {
    "include/api.h": "extern int func_80100000_de(int);\n/* func_80100000_de is the init */\n",
    "include/include_asm.h": ASM,
    "src/a.c": (
        '#include "api.h"\n#include "include_asm.h"\n'
        "int Run(void) { return func_80100000_de(1) + func_80100100_de(2); }\n"
        "INCLUDE_ASM(CODE, func_80100100_de);\n"
    ),
    "versions/de/symbol_addrs.txt": "func_80100000_de = 0x80100000; // type:func\nfunc_80100100_de = 0x80100100;\n",
}


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    root = tmp_path / "repo"
    for name, text in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _rename(root: Path, old: str, new: str, only: tuple[str, ...] = ()) -> None:
    for path in _git(root, "ls-files").split():
        if only and path not in only:
            continue
        text = (root / path).read_text()
        (root / path).write_text(text.replace(old, new))


def _prove(root: Path, **extra: object) -> dict:
    return renameproof.rename_command({"project": str(root), "against": "HEAD", **extra})


def _refused(root: Path, **extra: object) -> str:
    with pytest.raises(Refusal) as refused:
        _prove(root, **extra)
    assert {f.key for f in refused.value.findings} == {"renameproof.not_rename"}
    return " | ".join(f.reason for f in refused.value.findings)


def test_a_rename_in_headers_sources_symbols_and_assembly_text_is_proved(repo: Path) -> None:
    _rename(repo, "func_80100000_de", "Game_Init")
    _rename(repo, "func_80100100_de", "Game_Step")
    result = _prove(repo)
    assert result["map"] == [["func_80100000_de", "Game_Init"], ["func_80100100_de", "Game_Step"]]
    assert result["files"] == 3 and result["renames"] == 2


def test_a_patch_is_proved_in_a_private_index_without_touching_the_tree(repo: Path) -> None:
    _rename(repo, "func_80100000_de", "Game_Init")
    patch = repo / "rename.patch"
    patch.write_text(_git(repo, "diff"))
    _git(repo, "checkout", "-q", "--", ".")
    result = _prove(repo, patch=[str(patch)])
    assert result["map"] == [["func_80100000_de", "Game_Init"]]
    assert _git(repo, "status", "--short").strip() == "?? rename.patch"
    assert (repo / "src/a.c").read_text() == FILES["src/a.c"]


def test_comments_and_spacing_are_free_but_a_changed_token_is_not(repo: Path) -> None:
    (repo / "src/a.c").write_text(FILES["src/a.c"].replace("return", "/* sum */  return") + "\n")
    assert _prove(repo)["renames"] == 0
    (repo / "src/a.c").write_text(FILES["src/a.c"].replace("(2)", "(3)"))
    assert "not an identifier" in _refused(repo)


def test_a_name_left_behind_in_one_place_is_not_a_rename(repo: Path) -> None:
    _rename(repo, "func_80100000_de", "Game_Init", only=("include/api.h", "src/a.c"))
    assert "left in another" in _refused(repo) or "still a code token" in _refused(repo)


def test_a_name_left_behind_in_an_unchanged_file_is_refused(repo: Path) -> None:
    _rename(repo, "func_80100000_de", "Game_Init", only=("include/api.h",))
    assert "still a code token" in _refused(repo)


def test_a_new_name_that_already_exists_would_capture(repo: Path) -> None:
    (repo / "src/b.c").write_text("int Game_Init;\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "b")
    _rename(repo, "func_80100000_de", "Game_Init", only=("include/api.h", "src/a.c", "versions/de/symbol_addrs.txt"))
    assert "already a code token" in _refused(repo)


def test_two_names_renamed_to_one_are_refused(repo: Path) -> None:
    _rename(repo, "func_80100000_de", "Game_Init")
    _rename(repo, "func_80100100_de", "Game_Init")
    assert "both renamed to" in _refused(repo)


def test_a_swap_is_a_rename(repo: Path) -> None:
    for path in ("include/api.h", "src/a.c", "versions/de/symbol_addrs.txt", "include/include_asm.h"):
        text = (
            (repo / path).read_text().replace("func_80100000_de", "@A").replace("func_80100100_de", "func_80100000_de")
        )
        (repo / path).write_text(text.replace("@A", "func_80100100_de"))
    assert _prove(repo)["renames"] == 2


def test_a_name_stringified_outside_assembly_is_refused(repo: Path) -> None:
    (repo / "src/c.c").write_text("const char *Name(void) { return STR(func_80100000_de); }\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")
    _rename(repo, "func_80100000_de", "Game_Init")
    assert "stringifies it outside inline assembly" in _refused(repo)


def test_a_name_pasted_is_refused(repo: Path) -> None:
    (repo / "src/c.c").write_text("int v = CAT(func_80100000_de,);\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")
    _rename(repo, "func_80100000_de", "Game_Init")
    assert "pasted" in _refused(repo) or "stringif" in _refused(repo)


def test_a_string_that_differs_outside_assembly_is_refused(repo: Path) -> None:
    (repo / "src/c.c").write_text('const char *Tag = "func_80100000_de";\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")
    _rename(repo, "func_80100000_de", "Game_Init")
    assert "not an identifier" in _refused(repo)


def test_an_added_file_refuses_unless_declared_inert(repo: Path) -> None:
    _rename(repo, "func_80100000_de", "Game_Init")
    (repo / "versions/de/names.tsv").write_text("addr name\n")
    _git(repo, "add", "-N", "versions/de/names.tsv")
    patch = repo / "rename.patch"
    patch.write_text(_git(repo, "diff"))
    _git(repo, "reset", "-q")
    _git(repo, "checkout", "-q", "--", ".")
    assert "added or removed" in _refused(repo, patch=[str(patch)])
    assert _prove(repo, patch=[str(patch)], inert=["versions/*/names.tsv"])["inert"] == ["versions/de/names.tsv"]


def test_a_line_that_moves_where_the_tree_reads_line_is_refused(repo: Path) -> None:
    (repo / "src/c.c").write_text("int l = __LINE__;\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")
    _rename(repo, "func_80100000_de", "Game_Init")
    text = (repo / "src/a.c").read_text().replace("int Run", "\nint Run")
    (repo / "src/a.c").write_text(text)
    assert "another line" in _refused(repo)


def test_a_change_to_no_file_is_not_a_proof(repo: Path) -> None:
    assert "touches no file" in _refused(repo)


def test_text_files_keep_their_indent_and_directives_their_line_ends() -> None:
    assert renameproof._tokens("a:\n  b\n", False) == ["a", ":", "\n", "   ", "b", "\n"]
    assert renameproof._tokens("#define X 1\nint y;\n", True) == ["#", "define", "X", "1", "\n", "int", "y", ";"]
    assert renameproof._tokens('__asm__("a");\n"b"', True)[2] == '\0"a"'


def test_a_rename_to_or_from_a_compiler_builtin_is_refused(repo: Path) -> None:
    _rename(repo, "func_80100000_de", "sqrtf")
    assert "expands itself" in _refused(repo)


def test_a_builtin_name_is_fine_when_the_project_turns_builtins_off(repo: Path) -> None:
    (repo / "config.toml").write_text('cflags = ["-O2", "-fno-builtin"]\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "flags")
    _rename(repo, "func_80100000_de", "sqrtf")
    assert _prove(repo)["map"] == [["func_80100000_de", "sqrtf"]]


def test_steps_judge_each_patch_over_the_state_the_earlier_ones_leave(repo: Path) -> None:
    patches = []
    for n, (old, new) in enumerate([("func_80100000_de", "Game_Init"), ("func_80100100_de", "sqrtf")]):
        _rename(repo, old, new)
        _git(repo, "commit", "-q", "-am", f"step {n}")
        patches.append(repo.parent / f"{n}.patch")
        patches[-1].write_text(_git(repo, "diff", "HEAD~1", "HEAD"))
    _git(repo, "reset", "-q", "--hard", "HEAD~2")
    result = _prove(repo, patch=[str(p) for p in patches], steps=True)
    assert [row["pure"] for row in result["steps"]] == [True, False]
    assert result["map"] == [["func_80100000_de", "Game_Init"]]


LOCALS = (
    "int Grow(int step) {\n    int range = step + 1;\n    return range * 2;\n}\n"
    "int Other(int step) {\n    int range = 3;\n    return step + range + Util;\n}\n"
)


def _locals(repo: Path, extra: str = "") -> None:
    (repo / "src/loc.c").write_text(LOCALS + extra)
    (repo / "include/g.h").write_text("extern int Util;\n#define SCALE(v) ((v) * 2)\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "locals")


def _grow(repo: Path, old: str, new: str) -> None:
    text = (repo / "src/loc.c").read_text()
    head, tail = text.split("int Other", 1)
    (repo / "src/loc.c").write_text(re.sub(rf"\b{old}\b", new, head) + "int Other" + tail)


def test_a_local_rename_to_a_name_another_function_uses_is_proved(repo: Path) -> None:
    _locals(repo)
    _grow(repo, "step", "growStep")
    _grow(repo, "range", "influenceRange")
    result = _prove(repo)
    assert result["map"] == [["range", "influenceRange"], ["step", "growStep"]]


def test_a_local_may_take_a_builtin_spelling_a_global_may_not(repo: Path) -> None:
    _locals(repo)
    _grow(repo, "step", "index")  # a loop index: shadows libc index() inside this function only
    assert _prove(repo)["map"] == [["step", "index"]]


def test_a_local_rename_to_a_name_the_function_already_uses_is_refused(repo: Path) -> None:
    _locals(repo)
    _grow(repo, "step", "range")
    assert "already used in the function" in _refused(repo) or "both renamed" in _refused(repo)


def test_a_local_rename_shadowing_a_global_the_function_uses_is_refused(repo: Path) -> None:
    _locals(repo)
    text = (repo / "src/loc.c").read_text().replace("return range * 2;", "return range * Util;", 1)
    (repo / "src/loc.c").write_text(text)
    _git(repo, "commit", "-q", "-am", "Grow uses Util")
    _grow(repo, "step", "Util")
    assert "already used in the function" in _refused(repo)


def test_a_local_rename_meeting_a_macro_free_name_is_refused(repo: Path) -> None:
    _locals(repo)
    (repo / "include/g.h").write_text("extern int Util;\n#define SCALE(v) ((v) * growStep)\n")
    _git(repo, "commit", "-q", "-am", "macro")
    _grow(repo, "step", "growStep")
    assert "preprocessor directive" in _refused(repo)


def test_a_local_rename_left_half_done_in_its_function_is_refused(repo: Path) -> None:
    _locals(repo)
    (repo / "src/loc.c").write_text(
        (repo / "src/loc.c").read_text().replace("int Grow(int step)", "int Grow(int growStep)", 1)
    )
    assert "left in another" in _refused(repo)


def test_a_global_rename_to_a_name_that_is_another_functions_local_is_proved(repo: Path) -> None:
    (repo / "src/loc.c").write_text("int Other(int step) { return step; }\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-am", "loc")
    _rename(repo, "func_80100000_de", "step", only=("include/api.h", "src/a.c", "versions/de/symbol_addrs.txt"))
    assert _prove(repo)["map"] == [["func_80100000_de", "step"]]


def test_per_version_copies_of_one_function_may_share_a_new_name(repo: Path) -> None:
    (repo / "versions/eu").mkdir()
    (repo / "versions/eu/symbol_addrs.txt").write_text("func_80100200_eu = 0x80100200; // type:func\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "eu")
    _rename(repo, "func_80100000_de", "Game_Init")
    _rename(repo, "func_80100200_eu", "Game_Init")
    assert ["func_80100200_eu", "Game_Init"] in _prove(repo)["map"]


def test_a_static_without_a_symbol_row_and_another_versions_extern_may_share_a_new_name(repo: Path) -> None:
    (repo / "versions/us-rev1").mkdir()
    (repo / "versions/de/symbol_addrs.txt").write_text(
        (repo / "versions/de/symbol_addrs.txt").read_text() + "D_80100300_de = 0x80100300; // type:data\n"
    )
    (repo / "src/s.c").write_text("static int D_80100300_us_rev1;\nint Rd(void) { return D_80100300_us_rev1; }\n")
    (repo / "src/d.c").write_text("extern int D_80100300_de;\nint Rd2(void) { return D_80100300_de; }\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "statics")
    _rename(repo, "D_80100300_us_rev1", "g_value")
    _rename(repo, "D_80100300_de", "g_value")
    assert ["D_80100300_us_rev1", "g_value"] in _prove(repo)["map"]


def test_a_fuzzy_alias_dropped_with_its_undef_is_a_rename(repo: Path) -> None:
    text = (repo / "src/a.c").read_text()
    shim = "#ifdef NON_MATCHING\n#define func_80100000_de fz_func_80100000_de\n#endif\n"
    undef = "#ifdef NON_MATCHING\n#undef func_80100000_de\n#endif\n"
    (repo / "src/a.c").write_text(shim + text + undef)
    _git(repo, "commit", "-q", "-am", "shim")
    _rename(repo, "func_80100000_de", "Game_Init")
    (repo / "src/a.c").write_text(
        (repo / "src/a.c").read_text().replace("#define Game_Init fz_Game_Init\n", "").replace("#undef Game_Init\n", "")
    )
    assert ["func_80100000_de", "Game_Init"] in _prove(repo)["map"]


def test_a_prototype_parameter_named_like_a_renamed_global_is_no_use_of_it(repo: Path) -> None:
    (repo / "include/rate.h").write_text("extern int rate;\nint Scale(int rate, int other);\n")
    (repo / "src/r.c").write_text('#include "rate.h"\nint rate;\nint Use(void) { return rate; }\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "rate")
    (repo / "include/rate.h").write_text("extern int g_rate;\nint Scale(int rate, int other);\n")
    (repo / "src/r.c").write_text('#include "rate.h"\nint g_rate;\nint Use(void) { return g_rate; }\n')
    assert ["rate", "g_rate"] in _prove(repo)["map"]


def test_two_copies_in_one_version_still_share_no_name(repo: Path) -> None:
    _rename(repo, "func_80100000_de", "Game_Init")
    _rename(repo, "func_80100100_de", "Game_Init")
    assert "both renamed to" in _refused(repo)


def test_a_later_parameter_with_a_typed_name_is_a_local_of_its_function(repo: Path) -> None:
    (repo / "src/loc.c").write_text(
        "typedef struct Box Box;\n"
        "int One(Box *arg0, Box *arg1, int arg2) {\n    return arg1->w + arg2;\n}\n"
        "int Two(Box *arg0, Box *arg1) {\n    return arg1->h;\n}\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "params")
    text = (repo / "src/loc.c").read_text()
    one, two = text.split("int Two")
    (repo / "src/loc.c").write_text(re.sub(r"\barg1\b", "entry", one) + "int Two" + re.sub(r"\barg1\b", "box", two))
    result = _prove(repo)
    assert result["map"] == [["arg1", "box"], ["arg1", "entry"]]


def test_a_tag_and_a_typedef_of_one_name_may_take_different_names(repo: Path) -> None:
    (repo / "include/m.h").write_text(
        "struct Message;\nstruct Message { int a; };\ntypedef struct Message Message;\nextern Message *g;\n"
        "extern struct Message *h;\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "tag")
    (repo / "include/m.h").write_text(
        "struct CMsg_t;\nstruct CMsg_t { int a; };\ntypedef struct CMsg_t CMsg;\nextern CMsg *g;\n"
        "extern struct CMsg_t *h;\n"
    )
    assert _prove(repo)["map"] == [["Message", "CMsg"], ["Message", "CMsg_t"]]


def test_a_tag_renamed_in_one_place_and_left_in_another_is_refused(repo: Path) -> None:
    (repo / "include/m.h").write_text("struct Message { int a; };\nextern struct Message *g;\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "tag")
    (repo / "include/m.h").write_text("struct CMsg_t { int a; };\nextern struct Message *g;\n")
    assert _refused(repo)


STRUCTS = {
    "include/s.h": "typedef struct A { int unk58; int unk60; } A;\ntypedef struct B { int unk58; } B;\n"
    "typedef struct C { int unk58; } C;\nenum E { unk58x };\n",
    "src/m.c": "int Run(A *a, B *b, C *c) { a->unk58 = 1; b->unk58 = 2; return a->unk60 + c->unk58; }\n",
}


@pytest.fixture
def structs(repo: Path) -> Path:
    for name, text in STRUCTS.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "structs")
    return repo


def _edit(root: Path, path: str, *pairs: tuple[str, str]) -> None:
    text = (root / path).read_text()
    for old, new in pairs:
        assert old in text
        text = text.replace(old, new)
    (root / path).write_text(text)


def test_one_member_name_renamed_differently_in_two_structs_is_a_rename(structs: Path) -> None:
    _edit(
        structs,
        "include/s.h",
        ("struct A { int unk58;", "struct A { int modeState;"),
        ("struct B { int unk58;", "struct B { int phase;"),
    )
    _edit(structs, "src/m.c", ("a->unk58", "a->modeState"), ("b->unk58", "b->phase"))
    result = _prove(structs)
    assert ["unk58", "modeState"] in result["map"] and ["unk58", "phase"] in result["map"]


def test_a_member_access_no_declaration_renames_that_way_is_refused(structs: Path) -> None:
    _edit(structs, "include/s.h", ("struct A { int unk58;", "struct A { int modeState;"))
    _edit(structs, "src/m.c", ("a->unk58", "a->modeState"), ("b->unk58", "b->other"))
    _edit(structs, "include/s.h", ("struct B { int unk58;", "struct B { int phase;"))
    assert "no struct declaration renames it so" in _refused(structs)


def test_two_members_of_one_struct_renamed_to_one_name_are_refused(structs: Path) -> None:
    _edit(
        structs,
        "include/s.h",
        ("int unk58; int unk60; }", "int same; int same; }"),
        ("struct B { int unk58;", "struct B { int phase;"),
    )
    _edit(structs, "src/m.c", ("a->unk58", "a->same"), ("a->unk60", "a->same"), ("b->unk58", "b->phase"))
    assert "both renamed to same" in _refused(structs)


def test_a_typed_member_name_already_used_in_the_tree_is_refused(structs: Path) -> None:
    (structs / "src/z.c").write_text("typedef struct Z { int phase; } Z;\n")
    _git(structs, "add", "-A")
    _git(structs, "commit", "-q", "-m", "z")
    _edit(
        structs,
        "include/s.h",
        ("struct A { int unk58;", "struct A { int modeState;"),
        ("struct B { int unk58;", "struct B { int phase;"),
    )
    _edit(structs, "src/m.c", ("a->unk58", "a->modeState"), ("b->unk58", "b->phase"))
    assert "already used in the old tree" in _refused(structs)


def _twin(repo: Path, alias_address: str) -> None:
    (repo / "versions/us-rev1").mkdir()
    (repo / "versions/us-rev1/symbol_addrs.txt").write_text("D_80100300_de = 0x80101300; // type:data\n")
    (repo / "versions/us-rev1/aliases.txt").write_text(f"D_auto_blob_us_rev1 = {alias_address};\n")
    (repo / "src/s.c").write_text("extern int D_80100300_de;\nint Rd(void) { return D_80100300_de; }\n")
    (repo / "src/t.c").write_text("extern int D_auto_blob_us_rev1;\nint Rt(void) { return D_auto_blob_us_rev1; }\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "twin")
    _rename(repo, "D_80100300_de", "g_value")
    _rename(repo, "D_auto_blob_us_rev1", "g_value")


def test_two_names_of_one_address_in_one_version_may_share_a_new_name(repo: Path) -> None:
    _twin(repo, "0x80101300;")
    assert ["D_auto_blob_us_rev1", "g_value"] in _prove(repo)["map"]


def test_two_names_of_different_addresses_in_one_version_may_not(repo: Path) -> None:
    _twin(repo, "0x80101340;")
    with pytest.raises(Refusal, match="both renamed to g_value"):
        _prove(repo)
