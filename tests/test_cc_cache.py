"""The compiler front: a text compiled once is copied, never compiled twice, wherever the tree is."""

from __future__ import annotations

import importlib.resources
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from unbake import build
from unbake.resources import cc_cache


def fake_cc(tmp_path: Path) -> Path:
    """A compiler that counts its runs and writes the text it was given, upper-cased, as the assembly."""
    cc = tmp_path / "cc1"
    cc.write_text(
        '#!/bin/sh\necho run >> "$0.runs"\n'
        'while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift;; *.i) src=$1;; esac; shift; done\n'
        'tr a-z A-Z < "$src" > "$out"\n'
    )
    cc.chmod(0o755)
    return cc


def runs(cc: Path) -> int:
    log = Path(f"{cc}.runs")
    return len(log.read_text().splitlines()) if log.exists() else 0


def tree(tmp_path: Path, name: str, text: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "x.i").write_text(text)
    return root


def compile_in(root: Path, cache: Path, cc: Path) -> str:
    cwd = os.getcwd()
    os.chdir(root)
    try:
        assert cc_cache.run(cache, str(cc), ["-quiet", "-O2", "x.i", "-o", "x.s"]) == 0
    finally:
        os.chdir(cwd)
    return (root / "x.s").read_text()


def test_a_second_tree_copies_the_assembly_of_the_same_text(tmp_path):
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    assert compile_in(tree(tmp_path, "a", "int a;"), cache, cc) == "INT A;"
    assert compile_in(tree(tmp_path, "b", "int a;"), cache, cc) == "INT A;"
    assert runs(cc) == 1


def test_a_changed_text_compiles_again(tmp_path):
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    compile_in(tree(tmp_path, "a", "int a;"), cache, cc)
    assert compile_in(tree(tmp_path, "b", "int a, b;"), cache, cc) == "INT A, B;"
    assert runs(cc) == 2


def test_other_flags_are_another_entry(tmp_path):
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    root = tree(tmp_path, "a", "int a;")
    os.chdir(root)
    try:
        cc_cache.run(cache, str(cc), ["-O1", "x.i", "-o", "x.s"])
        cc_cache.run(cache, str(cc), ["-O2", "x.i", "-o", "x.s"])
    finally:
        os.chdir(tmp_path)
    assert runs(cc) == 2


def test_a_failed_compile_is_not_stored(tmp_path):
    cc = tmp_path / "bad"
    cc.write_text("#!/bin/sh\necho run >> $0.runs\nexit 3\n")
    cc.chmod(0o755)
    root, cache = tree(tmp_path, "a", "x"), tmp_path / "cc"
    os.chdir(root)
    try:
        assert cc_cache.run(cache, str(cc), ["x.i", "-o", "x.s"]) == 3
        assert cc_cache.run(cache, str(cc), ["x.i", "-o", "x.s"]) == 3
    finally:
        os.chdir(tmp_path)
    assert runs(cc) == 2


def test_a_command_without_output_runs_as_given(tmp_path):
    assert cc_cache.parse(["-quiet", "x.i"]) is None
    assert cc_cache.parse(["a.i", "b.i", "-o", "x.s"]) is None
    assert cc_cache.parse(["-O2", "x.i", "-o", "x.s"]) == ("x.i", "x.s")
    assert cc_cache.run(tmp_path / "cc", "/bin/sh", ["-c", "exit 4"]) == 4
    assert not (tmp_path / "cc").exists()


def test_a_hit_survives_a_corrupt_entry(tmp_path):
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    compile_in(tree(tmp_path, "a", "int a;"), cache, cc)
    for entry in cache.glob("??/*"):
        entry.write_bytes(b"not zlib")
    assert compile_in(tree(tmp_path, "b", "int a;"), cache, cc) == "INT A;"
    assert runs(cc) == 2


def test_prune_drops_only_what_aged_out(tmp_path):
    cache = tmp_path / "cc"
    (cache / "ab").mkdir(parents=True)
    old, new = cache / "ab" / "old", cache / "ab" / "new"
    old.write_bytes(b"1")
    new.write_bytes(b"2")
    aged = time.time() - cc_cache.MAX_AGE - 10
    os.utime(old, (aged, aged))
    assert cc_cache.prune(cache, time.time()) == 1
    assert new.exists() and not old.exists()


def test_the_script_runs_standalone_from_the_package_file(tmp_path):
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    root = tree(tmp_path, "a", "int a;")
    script = Path(cc_cache.__file__)
    done = subprocess.run(
        [sys.executable, "-I", str(script), str(cache), str(cc), "x.i", "-o", "x.s"], cwd=root, capture_output=True
    )
    assert done.returncode == 0 and (root / "x.s").read_text() == "INT A;"


def test_local_mk_puts_each_compiler_behind_the_front(tmp_path):
    host = SimpleNamespace(
        workers=4, tools={}, toolchain_root=Path("/chains"), cache_dir=Path("/home/x/.cache/unbake/content")
    )
    text = build.local_mk(SimpleNamespace(host=host)).decode()
    line = next(row for row in text.splitlines() if row.startswith("CC_gcc_2_8_1_sn64 :="))
    assert "cc_cache.py /home/x/.cache/unbake/content.cc /chains/gcc-2.8.1-sn64/cc1" in line
    assert "python3 -I " in line
    assert importlib.resources.files("unbake.resources").joinpath("cc_cache.py").is_file()


def test_renamed_accepts_only_a_consistent_one_for_one_rename():
    old = b"struct S { int unk10; }; int f(struct S *s) { return s->unk10 + 1; }\n"
    asm = b"f:\n\tlw\t$2,16($4)\n\tj\t$31\n"
    new = old.replace(b"unk10", b"fogNear")
    assert cc_cache.renamed(old, new, asm)
    assert not cc_cache.renamed(old, new.replace(b"+ 1", b"+ 2"), asm)  # a value moved
    assert not cc_cache.renamed(old, old.replace(b"unk10", b"fogNear", 1), asm)  # one use renamed, one not
    assert not cc_cache.renamed(old, old.replace(b"int", b"long", 1), asm)  # a keyword is not a name
    assert not cc_cache.renamed(old, old.replace(b"unk10", b"S"), asm)  # the new name is already in use
    assert not cc_cache.renamed(old, old.replace(b" ", b"  ", 1), asm)  # whitespace differs
    assert not cc_cache.renamed(old, new, asm + b".comm unk10,4\n")  # an old name spelled in the assembly
    assert not cc_cache.renamed(old, new, b"static.unk10.3:\n")  # as a part of a word
    assert not cc_cache.renamed(old, old, asm)  # nothing renamed


def test_a_rename_run_skips_the_compiler_and_stores_the_entry(tmp_path):
    cc = tmp_path / "cc1"
    cc.write_text(
        '#!/bin/sh\necho run >> "$0.runs"\n'
        'while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift;; *.i) src=$1;; esac; shift; done\n'
        'sed -n \'s/.*return \\(.*\\);.*/\\1/p\' "$src" > "$out"\n'
    )
    cc.chmod(0o755)
    cache = tmp_path / "cc"
    old = "struct S { int unk10; }; int f(struct S *s) { return 7; }\n"
    assert compile_in(tree(tmp_path, "a", old), cache, cc) == "7\n"
    new = old.replace("unk10", "fogNear")
    assert compile_in(tree(tmp_path, "b", new), cache, cc) == "7\n"
    assert runs(cc) == 1
    assert compile_in(tree(tmp_path, "c", new), cache, cc) == "7\n"  # now an exact hit
    assert runs(cc) == 1
    assert compile_in(tree(tmp_path, "d", new.replace("return 7", "return 8")), cache, cc) == "8\n"
    assert runs(cc) == 2


def test_pruned_drops_only_type_declarations_nothing_names():
    base = "typedef int s32;\nstruct A { s32 a; };\nstruct A g;\nint f(void) { return 1; }\n"
    extra = "typedef struct Fresh { int q; } FreshT;\nenum E { E1, E2 };\nstruct Fwd;\n"
    assert cc_cache.pruned((extra + base).encode()) == cc_cache.pruned(base.encode())
    # a used type, a variable, an extern, a function and a variable with a struct body all stay
    assert b"struct A" in cc_cache.pruned(base.encode()) and b"int f" in cc_cache.pruned(base.encode())
    assert cc_cache.pruned(b"struct B { int b; } v;\n") != cc_cache.pruned(b"")
    assert cc_cache.pruned(b"struct B v;\n") != cc_cache.pruned(b"")
    assert cc_cache.pruned(b"extern int x;\n") != cc_cache.pruned(b"")
    # an enumerator used by a kept chunk keeps its enum
    used = "enum E { E1, E2 };\nint g(void) { return E2; }\n"
    assert cc_cache.pruned(used.encode()) != cc_cache.pruned(b"int g(void) { return E2; }\n")
    # a field change of a type nothing names is invisible; of a named one it is not
    assert cc_cache.pruned(b"struct U { int a; };\nint h;\n") == cc_cache.pruned(b"struct U { int a; int b; };\nint h;\n")
    assert cc_cache.pruned(b"struct U { int a; };\nstruct U h;\n") != cc_cache.pruned(
        b"struct U { int a; int b; };\nstruct U h;\n"
    )


def test_a_header_edit_adding_unused_types_skips_the_compiler(tmp_path):
    cc = tmp_path / "cc1"
    cc.write_text(
        '#!/bin/sh\necho run >> "$0.runs"\n'
        'while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift;; *.i) src=$1;; esac; shift; done\n'
        'sed -n \'s/.*return \\(.*\\);.*/\\1/p\' "$src" > "$out"\n'
    )
    cc.chmod(0o755)
    cache = tmp_path / "cc"
    old = "typedef int s32;\nint f(void) { return 7; }\n"
    assert compile_in(tree(tmp_path, "a", old), cache, cc) == "7\n"
    grown = "typedef struct Fresh { s32 q; } FreshT;\n" + old
    assert compile_in(tree(tmp_path, "b", grown), cache, cc) == "7\n"
    assert runs(cc) == 1
    assert compile_in(tree(tmp_path, "c", old.replace("return 7", "return 8")), cache, cc) == "8\n"
    assert runs(cc) == 2
