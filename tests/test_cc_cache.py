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
    assert line.startswith("CC_gcc_2_8_1_sn64 := sh ") and "/resources/cc_front.sh " in line
    assert line.endswith(" /home/x/.cache/unbake/content.cc /chains/gcc-2.8.1-sn64/cc1")
    assert importlib.resources.files("unbake.resources").joinpath("cc_front.sh").is_file()


FRONT = Path(cc_cache.__file__).with_name("cc_front.sh")


def front(root: Path, cache: Path, cc: Path, *, collect: Path | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "ABU_CC_COLLECT"}
    if collect:
        env["ABU_CC_COLLECT"] = str(collect)
    args = ["sh", str(FRONT), str(cache), str(cc), "-quiet", "-O2", "x.i", "-o", "x.s"]
    return subprocess.run(args, cwd=root, capture_output=True, env=env, check=False)


def batch_of(root: Path, cache: Path, manifest: Path, workers: int = 2) -> int:
    return sum(cc_cache.batch(str(cache), str(root), str(manifest), i, workers) for i in range(workers))


def test_front_records_a_compile_and_fails_it_in_collect_mode(tmp_path):
    cc, cache, manifest = fake_cc(tmp_path), tmp_path / "cc", tmp_path / "commands"
    root = tree(tmp_path, "a", "int a;")
    done = front(root, cache, cc, collect=manifest)
    assert done.returncode == 125 and not (root / "x.s").exists() and runs(cc) == 0
    assert cc_cache.commands(str(manifest)) == [(str(cc), ["-quiet", "-O2", "x.i", "-o", "x.s"])]


def test_a_batch_answers_hits_and_leaves_misses_to_the_front(tmp_path):
    cc, cache, manifest = fake_cc(tmp_path), tmp_path / "cc", tmp_path / "commands"
    root = tmp_path / "a"
    root.mkdir()
    for name, text in (("hit", "int a;"), ("miss", "int c;")):
        (root / f"{name}.i").write_text(text)

    def command(name: str) -> list[str]:
        return ["-quiet", "-O2", f"{name}.i", "-o", f"{name}.s"]

    cwd = os.getcwd()
    os.chdir(root)
    try:
        assert cc_cache.run(str(cache), str(cc), command("hit")) == 0 and runs(cc) == 1  # seeds the cache
    finally:
        os.chdir(cwd)
    manifest.write_text("".join(f"{cc}\x1f" + "\x1f".join(command(n)) + "\n" for n in ("hit", "miss")))
    assert batch_of(root, cache, manifest) == 1 and runs(cc) == 1
    assert (root / "hit.s.cc").read_text() == "INT A;" and not (root / "miss.s.cc").exists()
    taken = subprocess.run(["sh", str(FRONT), str(cache), str(cc), *command("hit")], cwd=root, check=False)
    assert taken.returncode == 0 and (root / "hit.s").read_text() == "INT A;" and not (root / "hit.s.cc").exists()
    assert runs(cc) == 1  # an answered unit ran no compiler
    made = subprocess.run(["sh", str(FRONT), str(cache), str(cc), *command("miss")], cwd=root, check=False)
    assert made.returncode == 0 and (root / "miss.s").read_text() == "INT C;" and runs(cc) == 2


def test_a_batch_answer_equals_the_single_front_answer_for_hit_and_reuse(tmp_path):
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    base = "struct S { int unk10; }; int f(struct S *s) { return s->unk10 + 1; }\n"
    renamed = base.replace("unk10", "fogNear")
    seed = tree(tmp_path, "seed", base)
    compile_in(seed, cache, cc)
    args = ["-quiet", "-O2", "x.i", "-o", "x.s"]
    for number, text in enumerate((base, renamed)):
        single, batched = tree(tmp_path, f"s{number}", text), tree(tmp_path, f"b{number}", text)
        manifest = tmp_path / f"m{number}"
        manifest.write_text(f"{cc}\x1f" + "\x1f".join(args) + "\n")
        assert cc_cache.batch(str(cache), str(batched), str(manifest), 0, 1) == 1
        os.chdir(single)
        try:
            assert cc_cache.run(str(cache), str(cc), args) == 0
        finally:
            os.chdir(tmp_path)
        assert (batched / "x.s.cc").read_bytes() == (single / "x.s").read_bytes()
        assert runs(cc) == 1  # neither compiled: the renamed text reused the seed's assembly
        assert (batched / "x.s.cc").stat().st_mtime >= (batched / "x.i").stat().st_mtime + 1


def test_the_front_ignores_an_answer_older_than_its_input(tmp_path):
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    root = tree(tmp_path, "a", "int a;")
    (root / "x.s.cc").write_text("STALE")
    os.utime(root / "x.s.cc", (1, 1))
    assert front(root, cache, cc).returncode == 0 and (root / "x.s").read_text() == "INT A;" and runs(cc) == 1


def test_a_failing_compile_fails_the_front_after_a_batch_left_it(tmp_path):
    cc, cache, manifest = tmp_path / "cc1", tmp_path / "cc", tmp_path / "commands"
    cc.write_text("#!/bin/sh\necho 'x.i:1: error' >&2\nexit 3\n")
    cc.chmod(0o755)
    root = tree(tmp_path, "a", "int a;")
    front(root, cache, cc, collect=manifest)
    assert batch_of(root, cache, manifest) == 0
    done = front(root, cache, cc)
    assert done.returncode == 3 and b"x.i:1: error" in done.stderr and not (root / "x.s.cc").exists()


def test_a_batch_skips_a_command_it_cannot_key_and_an_unreadable_source(tmp_path):
    cc, cache, manifest = fake_cc(tmp_path), tmp_path / "cc", tmp_path / "commands"
    root = tree(tmp_path, "a", "int a;")
    manifest.write_text(f"{cc}\x1f-E\x1fy.c\n{cc}\x1f-quiet\x1fgone.i\x1f-o\x1fgone.s\n")
    assert cc_cache.batch(str(cache), str(root), str(manifest), 0, 1) == 0
    assert not list(root.glob("*.cc"))


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


def test_only_unnamed_type_declarations_may_differ():
    base = "typedef int s32;\nstruct A { s32 a; };\nstruct A g;\nint f(void) { return 1; }\n"
    extra = "typedef struct Fresh { int q; } FreshT;\nenum E { E1, E2 };\nstruct Fwd;\n"
    same = cc_cache.types_only_differ
    assert same((extra + base).encode(), base.encode())
    assert same(base.encode(), (base + extra).encode())  # at the end
    mid = base.replace("struct A g;", extra + "struct A g;")  # in the middle
    assert same(mid.encode(), base.encode())
    # a variable, an extern, a function, a used type or a used enumerator changes the object
    assert not same(b"struct B { int b; } v;\n" + base.encode(), base.encode())
    assert not same(b"struct B v;\n" + base.encode(), base.encode())
    assert not same(b"extern int x;\n" + base.encode(), base.encode())
    assert not same(base.encode(), base.replace("return 1", "return 2").encode())
    assert not same(base.encode(), base.replace("s32 a;", "s32 a; s32 b;").encode())
    used = "enum E { E1, E2 };\nint g(void) { return E2; }\n"
    assert not same(used.encode(), b"int g(void) { return E2; }\n")
    # a field change of a type nothing names is invisible; of a named one it is not
    assert same(b"struct U { int a; };\nint h;\n", b"struct U { int a; int b; };\nint h;\n")
    assert not same(b"struct U { int a; };\nstruct U h;\n", b"struct U { int a; int b; };\nstruct U h;\n")
    # two edits far apart, both unnamed types
    assert same((extra + base + "int z;\n").encode(), (base + "int z;\n" + extra.replace("Fresh", "Other")).encode())


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


def test_a_static_inline_function_nothing_calls_may_differ():
    same = cc_cache.types_only_differ
    base = "typedef int s32;\nint f(void) { return 1; }\nint g;\n"
    old = "static __inline__ s32 pick(s32 *p) { return *p; }\n"
    new = "static __inline__ s32 pick(const s32 *p) { return p[0]; }\nstatic __inline__ s32 other(void) { return 2; }\n"
    assert same((old + base).encode(), (new + base).encode())
    assert same((base + old).encode(), base.encode())
    # one a surviving chunk calls, a plain static (the compiler writes it) and a static local are not droppable
    called = base + "int h(s32 *p) { return pick(p); }\n"
    assert not same((old + called).encode(), (new + called).encode())
    assert not same(b"static s32 pick(s32 *p) { return *p; }\n" + base.encode(), base.encode())
    counted = "static __inline__ s32 pick(void) { static s32 n; return n; }\n"
    assert not same((counted + base).encode(), base.encode())
    # a type the dropped function names is dropped with it
    typed = "typedef struct P { s32 a; } P;\nstatic __inline__ s32 pick(P *p) { return p->a; }\n"
    assert same((typed + base).encode(), base.encode())


def test_hunks_split_two_edits_far_apart_and_the_words_are_whole():
    lines = [f"int v{i};" for i in range(2000)]
    other = list(lines)
    other.insert(10, "typedef int A;")
    other.insert(1501, "typedef long B;")
    spans = cc_cache.hunks("\n".join(lines).encode(), "\n".join(other).encode())
    assert len(spans) == 2
    assert all(
        lo == hi and new_hi - new_lo <= 16 for lo, hi, new_lo, new_hi in spans
    )  # two insertions, nothing between
    assert cc_cache.mentioned({b"abc", b"xyz"}, b"abcd _xyz abc") == {b"abc"}
    far = "\n".join(lines).encode()
    assert cc_cache.types_only_differ(far, "\n".join(other).encode())


def test_canonical_is_the_declarations_that_reach_the_object():
    base = "int f(void) {\n return 1;\n}\nint g;\n"
    extra = "typedef int T;\nenum E { E1 };\nstatic __inline__ int unused(void) { return 2; }\n"
    assert cc_cache.canonical(base.encode()) == cc_cache.canonical((extra + base).encode())
    assert cc_cache.canonical(base.encode()) == cc_cache.canonical(base.replace("\n ", " ").encode())  # layout
    assert cc_cache.canonical(base.encode()) != cc_cache.canonical(base.replace("return 1", "return 2").encode())
    named = "typedef int T;\nT g;\n"
    assert b"typedef" in cc_cache.canonical(named.encode())
    calls = "static __inline__ int pick(void) { return 2; }\nint h(void) { return pick(); }\n"
    assert b"pick" in cc_cache.canonical(calls.encode())


def test_another_trees_body_does_not_hide_the_text_a_header_edit_only_adds_types_to(tmp_path):
    """Lanes compile different bodies of one unit: a third tree adding a type must still find the first."""
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    mine = "struct S { int a; };\nint f(struct S *s) { return s->a + 1; }\n"
    theirs = "struct S { int a; };\nint f(struct S *s) { return s->a + 2; }\n"
    edited = "struct S { int a; };\ntypedef int unused_t;\nint f(struct S *s) { return s->a + 1; }\n"
    for number, text in enumerate((mine, theirs)):
        compile_in(tree(tmp_path, f"t{number}", text), cache, cc)
    assert runs(cc) == 2
    third = tree(tmp_path, "third", edited)
    manifest = tmp_path / "m"
    manifest.write_text(f"{cc}\x1f" + "\x1f".join(["-quiet", "-O2", "x.i", "-o", "x.s"]) + "\n")
    assert cc_cache.batch(str(cache), str(third), str(manifest), 0, 1) == 1  # answered from `mine`, not compiled
    assert runs(cc) == 2
