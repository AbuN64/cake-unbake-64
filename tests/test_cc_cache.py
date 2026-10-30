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


def test_a_batch_answer_equals_the_single_front_answer(tmp_path):
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    text = "struct S { int unk10; }; int f(struct S *s) { return s->unk10 + 1; }\n"
    compile_in(tree(tmp_path, "seed", text), cache, cc)
    args = ["-quiet", "-O2", "x.i", "-o", "x.s"]
    single, batched = tree(tmp_path, "s", text), tree(tmp_path, "b", text)
    manifest = tmp_path / "m"
    manifest.write_text(f"{cc}\x1f" + "\x1f".join(args) + "\n")
    assert cc_cache.batch(str(cache), str(batched), str(manifest), 0, 1) == 1
    os.chdir(single)
    try:
        assert cc_cache.run(str(cache), str(cc), args) == 0
    finally:
        os.chdir(tmp_path)
    assert (batched / "x.s.cc").read_bytes() == (single / "x.s").read_bytes()
    assert runs(cc) == 1


REORDERED = (
    "int g(int);\nint a, b;\nvoid f(int x) {\n    a = x;\n    g(x);\n    b = x;\n}\n",
    "int g(int);\nint a, b;\nvoid f(int x) {\n    a = x;\n    b = x;\n    g(x);\n}\n",  # a store moved ahead of a call
    "int g(int);\nint a, b;\nvoid f(int x) {\n    b = x;\n    a = x;\n    g(x);\n}\n",  # two stores swapped
)


def test_a_text_that_only_reorders_statements_misses_the_cache(tmp_path):
    """10-22 OPUS-WT-6: texts that only moved two stores (one ahead of a call) were answered with the previous text's
    assembly. Every distinct text is compiled; the same text again is an exact hit."""
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    for number, text in enumerate(REORDERED):
        assert compile_in(tree(tmp_path, f"t{number}", text), cache, cc) == text.upper()
        assert runs(cc) == number + 1
    assert compile_in(tree(tmp_path, "again", REORDERED[1]), cache, cc) == REORDERED[1].upper()
    assert runs(cc) == len(REORDERED)


def test_renames_layout_and_unused_types_are_other_texts(tmp_path):
    """No similar text is ever answered: a rename, a layout change or an added unused type compiles again."""
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    text = "struct S { int unk10; };\nint f(struct S *s) { return s->unk10 + 1; }\n"
    others = (text.replace("unk10", "fogNear"), text.replace("+ 1", "+1"), "typedef int unused_t;\n" + text)
    for number, other in enumerate((text, *others)):
        assert compile_in(tree(tmp_path, f"t{number}", other), cache, cc) == other.upper()
    assert runs(cc) == 4


def test_a_batch_never_answers_a_reordered_text_from_another_trees_entry(tmp_path):
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    compile_in(tree(tmp_path, "seed", REORDERED[0]), cache, cc)
    other = tree(tmp_path, "other", REORDERED[1])
    manifest = tmp_path / "m"
    manifest.write_text(f"{cc}\x1f" + "\x1f".join(["-quiet", "-O2", "x.i", "-o", "x.s"]) + "\n")
    assert cc_cache.batch(str(cache), str(other), str(manifest), 0, 1) == 0
    assert front(other, cache, cc).returncode == 0 and (other / "x.s").read_text() == REORDERED[1].upper()
    assert runs(cc) == 2


def test_the_front_takes_a_batch_answer_only_for_its_exact_command_and_input(tmp_path):
    """An answer's key names its command and text: a leftover for another text (the .i rewritten within a second,
    whatever the dates) or other flags is dropped, never moved into place."""
    cc, cache = fake_cc(tmp_path), tmp_path / "cc"
    root = tree(tmp_path, "a", REORDERED[0])
    args = ["-quiet", "-O2", "x.i", "-o", "x.s"]
    (root / "x.s.cc").write_text("STALE")
    (root / "x.s.cc.key").write_text(cc_cache.answer_key(str(cc), args, REORDERED[1].encode()) + "\n")
    later = time.time() + 100  # newer than its input: a date check would take it
    os.utime(root / "x.s.cc", (later, later))
    assert front(root, cache, cc).returncode == 0 and (root / "x.s").read_text() == REORDERED[0].upper()
    assert not (root / "x.s.cc").exists() and not (root / "x.s.cc.key").exists() and runs(cc) == 1
    (root / "x.s.cc").write_text("ANSWER")
    (root / "x.s.cc.key").write_text(cc_cache.answer_key(str(cc), ["-O1", *args[1:]], REORDERED[0].encode()) + "\n")
    assert front(root, cache, cc).returncode == 0 and (root / "x.s").read_text() == REORDERED[0].upper()
    (root / "x.s.cc").write_text("ANSWER")
    (root / "x.s.cc.key").write_text(cc_cache.answer_key(str(cc), args, REORDERED[0].encode()) + "\n")
    assert front(root, cache, cc).returncode == 0 and (root / "x.s").read_text() == "ANSWER"


def test_a_changed_compiler_or_a_file_beside_it_is_another_key(tmp_path):
    cc = fake_cc(tmp_path)
    args = ["-O2", "x.i", "-o", "x.s"]
    before = cc_cache.key(str(cc), args, b"int a;")
    cc_cache.IDENTITIES.clear()
    (tmp_path / "as1").write_text("another pass")  # a driver runs the passes beside it
    assert cc_cache.key(str(cc), args, b"int a;") != before
    cc_cache.IDENTITIES.clear()


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


def fake_cpp(tmp_path: Path) -> Path:
    """A preprocessor that counts its runs, answers `-print-prog-name=cc1`, and writes the source followed by each
    `-I`-less `#include "x"` file's text to standard output with a dependency file naming them (-MP)."""
    cpp = tmp_path / "cpp"
    cpp.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = -print-prog-name=cc1 ]; then echo cc1; exit 0; fi\n'
        'set -e\necho run >> "$0.runs"\n'
        'while [ $# -gt 0 ]; do case "$1" in -MF) mf=$2; shift;; -M*|-I*|-D*) ;; *) src=$1;; esac; shift; done\n'
        'cat "$src"; echo "x.o: $src h.h \\\\" > "$mf"; echo " " >> "$mf"; echo "h.h:" >> "$mf"; cat h.h\n'
    )
    cpp.chmod(0o755)
    return cpp


def tools(tmp_path: Path) -> tuple[Path, Path]:
    """(hdrkey, stamprec) stand-ins: the key is the text less its COMMENT lines; the record only says it was asked."""
    hdrkey, stamprec = tmp_path / "hdrkey", tmp_path / "stamprec"
    hdrkey.write_text('#!/bin/sh\ngrep -v COMMENT "$1"\n')
    stamprec.write_text('#!/bin/sh\necho "$@" >> "$0.log"\n')
    for tool in (hdrkey, stamprec):
        tool.chmod(0o755)
    return hdrkey, stamprec


def unit_argv(cache: Path, cpp: Path, tmp_path: Path, *flags: str) -> list[str]:
    hdrkey, stamprec = tools(tmp_path)
    return [
        "python3", "-I", "-S", str(Path(cc_cache.__file__)), "--unit", str(cache),
        "build/x.i.ok", "FLAGS " + " ".join(flags), str(hdrkey), str(stamprec), str(cpp),
        "-MMD", "-MP", "-MF", "x.d", *flags, "x.c",
    ]  # fmt: skip


def pp(cache: Path, cpp: Path, root: Path, *flags: str) -> SimpleNamespace:
    """The `--unit` process of a stamp; `stdout` is the text it left in `build/x.i`."""
    done = subprocess.run(unit_argv(cache, cpp, root.parent, *flags), cwd=root, capture_output=True, check=False)
    text = root / "build" / "x.i"
    return SimpleNamespace(
        returncode=done.returncode, stdout=text.read_bytes() if text.exists() else b"", stderr=done.stderr
    )


def source_tree(tmp_path: Path, name: str, header: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "x.c").write_text("int x;\n")
    (root / "h.h").write_text(header)
    return root


def test_a_second_tree_copies_the_text_of_a_source_whose_headers_did_not_move(tmp_path):
    cpp, cache = fake_cpp(tmp_path), tmp_path / "cache"
    first, second = source_tree(tmp_path, "a", "#define A 1\n"), source_tree(tmp_path, "b", "#define A 1\n")
    one, two = pp(cache, cpp, first), pp(cache, cpp, second)
    assert one.stdout == two.stdout == b"int x;\n#define A 1\n" and one.returncode == two.returncode == 0
    assert runs(cpp) == 1
    assert (second / "x.d").read_bytes() == (first / "x.d").read_bytes() and b"h.h:" in (second / "x.d").read_bytes()


def test_a_changed_header_or_source_or_flag_preprocesses_again(tmp_path):
    cpp, cache = fake_cpp(tmp_path), tmp_path / "cache"
    root = source_tree(tmp_path, "a", "#define A 1\n")
    pp(cache, cpp, root)
    (root / "h.h").write_text("#define A 2\n")
    assert pp(cache, cpp, root).stdout == b"int x;\n#define A 2\n"
    (root / "x.c").write_text("int y;\n")
    assert pp(cache, cpp, root).stdout == b"int y;\n#define A 2\n"
    assert pp(cache, cpp, root, "-DV=1").returncode == 0
    assert runs(cpp) == 4
    assert pp(cache, cpp, root, "-DV=1").returncode == 0 and runs(cpp) == 4


def test_a_deleted_header_is_a_miss(tmp_path):
    cpp, cache = fake_cpp(tmp_path), tmp_path / "cache"
    root = source_tree(tmp_path, "a", "#define A 1\n")
    pp(cache, cpp, root)
    (root / "h.h").unlink()
    assert pp(cache, cpp, root).returncode != 0 and runs(cpp) == 2


def test_a_failed_preprocess_is_not_stored_and_a_command_without_a_dependency_file_runs_as_given(tmp_path):
    cpp, cache = fake_cpp(tmp_path), tmp_path / "cache"
    root = source_tree(tmp_path, "a", "#define A 1\n")
    (root / "x.c").unlink()
    assert pp(cache, cpp, root).returncode != 0 and not cache.exists()
    (root / "x.c").write_text("int x;\n")
    bare = unit_argv(cache, cpp, tmp_path)
    del bare[bare.index("-MMD") : bare.index("-MF")]  # no -MP: the cache cannot name the headers
    done = subprocess.run(bare, cwd=root, capture_output=True, check=False)
    assert done.returncode == 0 and runs(cpp) == 2 and not cache.exists()


def test_local_mk_puts_the_stamp_recipe_in_one_process(tmp_path):
    host = SimpleNamespace(
        workers=4, tools={}, toolchain_root=Path("/chains"), cache_dir=Path("/home/x/.cache/unbake/content")
    )
    text = build.local_mk(SimpleNamespace(host=host)).decode()
    line = next(row for row in text.splitlines() if row.startswith("PPUNIT :="))
    assert line.startswith("PPUNIT := python3 -I -S ") and line.endswith(
        "/resources/cc_cache.py --unit /home/x/.cache/unbake/content.cc"
    )
    recipe = build._recipe("gcc-2.8.1-sn64")
    assert (
        "$(PPUNIT) $@ $(call shq,$(PP_FLAGS)) $(HDRKEY) $(STAMPREC) $(CPP)" in recipe
        and ">" not in recipe.split("endef")[0]
    )
    rule = Path(build.__file__).parent / "resources" / "templates" / "Makefile.in"
    lines = rule.read_text().split("\nbuild/%.i.ok:")[1].split("\n# The extracted files")[0].splitlines()
    assert lines[1:] == ["\t$(PREPROCESS_$(TC))"]  # the stamp's recipe is the one process, no shell lines around it


def test_a_unit_leaves_the_text_the_flags_a_dated_stamp_and_a_record(tmp_path):
    cpp, cache = fake_cpp(tmp_path), tmp_path / "cache"
    root = source_tree(tmp_path, "a", "#define A 1\n")
    assert pp(cache, cpp, root).returncode == 0
    assert (root / "build/x.i.ok").exists() and (root / "build/x.i.ok.flags").read_text() == "FLAGS \n"
    assert (root / "build/x.i.key").read_text() == "int x;\n#define A 1\n" and not (root / "build/x.i.new").exists()
    assert (tmp_path / "stamprec.log").read_text() == "record build/x.i.ok\n"


def test_a_text_with_the_same_compile_key_keeps_the_old_text_and_its_date(tmp_path):
    cpp, cache = fake_cpp(tmp_path), tmp_path / "cache"
    root = source_tree(tmp_path, "a", "#define A 1\n")
    pp(cache, cpp, root)
    old = (root / "build/x.i").stat().st_mtime_ns
    (root / "h.h").write_text("#define A 1\nCOMMENT\n")  # another text, the same key
    assert pp(cache, cpp, root).returncode == 0
    assert (root / "build/x.i").read_bytes() == b"int x;\n#define A 1\n" and (
        root / "build/x.i"
    ).stat().st_mtime_ns == old
    assert not (root / "build/x.i.new").exists()
    (root / "x.c").write_text("int y;\n")  # another key: the new text goes in
    assert pp(cache, cpp, root).stdout == b"int y;\n#define A 1\nCOMMENT\n"


def test_a_failed_preprocess_leaves_no_stamp_and_says_why(tmp_path):
    cpp, cache = fake_cpp(tmp_path), tmp_path / "cache"
    root = source_tree(tmp_path, "a", "#define A 1\n")
    (root / "x.c").unlink()
    done = pp(cache, cpp, root)
    assert done.returncode != 0 and not (root / "build/x.i.ok").exists() and done.stderr


def test_a_unit_without_a_cache_still_preprocesses(tmp_path):
    cpp = fake_cpp(tmp_path)
    root = source_tree(tmp_path, "a", "#define A 1\n")
    assert pp(Path(""), cpp, root).stdout == b"int x;\n#define A 1\n"


def test_a_unit_starts_two_programs_beyond_itself_when_the_cache_answers(tmp_path, monkeypatch):
    """The regression: a stamp used to cost ~14 processes; a hit costs the compile key and the record."""
    cpp, cache = fake_cpp(tmp_path), tmp_path / "cache"
    pp(cache, cpp, source_tree(tmp_path, "a", "#define A 1\n"))
    root = source_tree(tmp_path, "b", "#define A 1\n")
    started = []
    real = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: started.append(Path(argv[0]).name) or real(argv, **kw))
    cc_cache.preprocessor_identity.cache_clear()
    hdrkey, stamprec = tools(tmp_path)
    monkeypatch.chdir(root)
    args = ["-MMD", "-MP", "-MF", "x.d", "x.c"]
    assert cc_cache.unit(str(cache), "build/x.i.ok", "F", str(hdrkey), str(stamprec), str(cpp), args) == 0
    assert started == ["cpp", "hdrkey", "stamprec"]  # cpp: only `-print-prog-name=cc1`, once per process


def test_collect_mode_makes_the_first_units_and_records_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(cc_cache, "INLINE_UNITS", 2)
    manifest = str(tmp_path / "pp")
    rows = [[f"build/{n}.i.ok", "F", "hdrkey", "stamprec", "cpp", "-E", f"{n}.c"] for n in range(4)]
    assert [cc_cache.defer(manifest, row) for row in rows] == [False, False, True, True]
    assert [row for _, row in cc_cache.commands(manifest)] == [row[1:] for row in rows[2:]]
    assert [stamp for stamp, _ in cc_cache.commands(manifest)] == ["build/2.i.ok", "build/3.i.ok"]


def test_a_batch_makes_every_recorded_stamp_and_leaves_a_failing_one_to_make(tmp_path):
    cpp, cache = fake_cpp(tmp_path), tmp_path / "cache"
    root = source_tree(tmp_path, "a", "#define A 1\n")
    hdrkey, stamprec = tools(tmp_path)
    manifest = tmp_path / "pp"
    good = ["build/x.i.ok", "F", str(hdrkey), str(stamprec), str(cpp), "-MMD", "-MP", "-MF", "x.d", "x.c"]
    bad = ["build/y.i.ok", "F", str(hdrkey), str(stamprec), str(cpp), "-MMD", "-MP", "-MF", "y.d", "y.c"]
    manifest.write_text("".join(cc_cache.SEPARATOR.join(row) + "\n" for row in (good, bad)))
    done = subprocess.run(
        ["python3", "-I", str(Path(cc_cache.__file__)), "--units", str(cache), str(manifest), "0", "1"],
        cwd=root, capture_output=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0 and done.stderr == b""  # the failure is make's to name, not the batch's
    assert (root / "build/x.i.ok").exists() and (root / "build/x.i").exists() and not (root / "build/y.i.ok").exists()


def test_the_front_batches_the_stamps_before_it_records_the_compiles(tmp_path, monkeypatch):
    from unbake import front, process

    calls, batches = [], []
    monkeypatch.setattr(process, "scratch", lambda root: tmp_path)
    monkeypatch.setattr(process, "batch", lambda store, root, manifest, workers, mode="--batch": batches.append(mode))

    def start(record, preprocess):
        calls.append((record is not None, preprocess))
        if preprocess:
            Path(record + ".pp").write_text("x\n")  # stale stamps beyond the first few
        elif record:
            Path(record).write_text("")  # the compile commands, found once the stamps stand
        return 0

    assert front.build(start, tmp_path, tmp_path / "store", 3) == 0
    assert calls == [(True, True), (True, False), (False, False)] and batches == ["--units", "--batch"]
    calls.clear()
    batches.clear()
    front.build(
        lambda record, preprocess: calls.append((record is not None, preprocess)) or 0, tmp_path, tmp_path / "store", 3
    )
    assert calls == [(True, True)] and batches == []  # nothing stale: the first run was the build


# The `.i.ok` command of one real unit, as make expanded it in a 5-version tree (paths shortened): before the unit
# process this stamp took `mkdir`, the preprocessor behind a python front, then a shell of cmp/hdrkey/mv/rm, printf,
# touch and stamprec (~10 forks a unit, ~35k units in a full check).
REAL_STAMP = (
    "build/us/src/game/DATA/OVLDAT3/800CC690/800CC690.CPP.i.ok|gcc_2_8_1_sn64 -P -undef -nostdinc |"
    "-P -undef -nostdinc -MMD -MP -MF build/us/src/game/DATA/OVLDAT3/800CC690/800CC690.CPP.d "
    "-MT build/us/src/game/DATA/OVLDAT3/800CC690/800CC690.CPP.i.ok src/game/DATA/OVLDAT3/800CC690/800CC690.CPP"
)


def test_a_real_units_stamp_command_makes_the_whole_stamp_with_one_preprocessor_run_and_one_record(tmp_path):
    stamp, flags, cpp_args = REAL_STAMP.split("|")
    cpp, cache = fake_cpp(tmp_path), tmp_path / "cache"
    root = tmp_path / "tree"
    source = root / "src/game/DATA/OVLDAT3/800CC690/800CC690.CPP"
    source.parent.mkdir(parents=True)
    source.write_text("int x;\n")
    (root / "h.h").write_text("#define A 1\n")
    hdrkey, stamprec = tools(tmp_path)
    argv = ["python3", "-I", "-S", str(Path(cc_cache.__file__)), "--unit", str(cache), stamp, flags]
    argv += [str(hdrkey), str(stamprec), str(cpp), *cpp_args.split()]
    for _ in range(2):  # the second answers from the cache: still the whole stamp, no second preprocessor run
        assert subprocess.run(argv, cwd=root, check=False).returncode == 0
    text = root / stamp.removesuffix(".ok")
    assert text.read_bytes() == b"int x;\n#define A 1\n" and (root / (stamp + ".flags")).read_text() == flags + "\n"
    assert (root / stamp).exists() and not text.with_suffix(".new").exists()
    assert runs(cpp) == 1
    assert (tmp_path / "stamprec.log").read_text() == f"record {stamp}\n" * 2
