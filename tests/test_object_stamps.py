"""The generated object rules, run by real make with stub tools: a flags row change remakes the objects it reaches, and
a failed compile never lets the next good one keep a stale object."""

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from test_build import make_config

from unbake import build


class Tree:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        (tmp_path / "Makefile").write_bytes(build.makefile(make_config(tmp_path)))
        (tmp_path / "src").mkdir()
        (tmp_path / "asm/us").mkdir(parents=True)
        (tmp_path / "asm/us/.extracted").touch()
        (tmp_path / "versions/us").mkdir(parents=True, exist_ok=True)
        (tmp_path / "versions/us/text_sizes.mk").write_text("")
        self.cc = tmp_path / "cc.py"
        self.cc.write_text(
            "from pathlib import Path\nimport sys\n"
            "text = Path(sys.argv[-3]).read_text()\n"
            "if 'FAIL' in text:\n    Path(sys.argv[-1]).write_text('partial')\n    sys.exit(1)\n"
            "Path(sys.argv[-1]).write_text(text)\n"
            "with Path('compiled').open('a') as log: log.write(sys.argv[-1] + '\\n')\n"
        )
        self.asm = tmp_path / "asm.py"
        self.asm.write_text(
            "from pathlib import Path\nimport sys\n"
            "Path(sys.argv[-1]).write_bytes(Path(sys.argv[-3]).read_bytes())\n"
            "with Path('assembled').open('a') as log: log.write(sys.argv[-1] + '\\n')\n"
        )
        self.trim = tmp_path / "trim.py"
        self.trim.write_text("")
        self.target = "build/us/src/a.c.o"

    def make(
        self,
        cflags: str = "-O2",
        cppflags: str = "-DX",
        expect: int = 0,
        deps: bool = False,
        level: int = 0,
        extra: tuple[str, ...] = (),
    ) -> tuple[int, int]:
        for log in ("compiled", "assembled", "preprocessed"):
            (self.root / log).write_text("")
        result = subprocess.run(
            [
                "make",
                "--no-print-directory",
                "-j1",
                self.target,
                "STAMPS=",
                "TC=gcc_2_8_1_sn64",
                f"CFLAGS={cflags}",
                f"CPPFLAGS={cppflags}",
                "PREPROCESS_gcc_2_8_1_sn64=@cp $< $(@:.ok=.new)"
                + (" && echo p >> preprocessed && printf '%s: %s src/h.h\\n' $@ $< > $(@:.i.ok=.d)" if deps else ""),
                f"CC_gcc_2_8_1_sn64={sys.executable} {self.cc}",
                f"N64LINK={sys.executable} {self.asm}",
                "OBJCOPY=true",
                f"TRIM_TEXT={sys.executable} {self.trim}",
                "TEXTSIZE=0x10",
                *extra,
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            env={**os.environ, "MAKELEVEL": str(level)},
        )
        assert result.returncode == expect, result.stderr
        return (
            len((self.root / "compiled").read_text().splitlines()),
            len((self.root / "assembled").read_text().splitlines()),
        )

    def age(self) -> None:
        """Everything built is older than anything written next."""
        now = time.time_ns()
        for path in self.root.rglob("*"):
            if "build" in path.relative_to(self.root).parts:
                age = 10**10 + (10**9 if path.suffix == ".i" else 0)
                os.utime(path, ns=(now - age, now - age))
        for path in (self.root / "asm/us/.extracted", self.root / "src/a.c"):
            os.utime(path, ns=(now - 2 * 10**10, now - 2 * 10**10))

    def write(self, text: str) -> None:
        (self.root / "src/a.c").write_text(text)


def test_a_flags_change_remakes_the_object_and_an_unchanged_row_does_not(tmp_path):
    tree = Tree(tmp_path)
    tree.write("int a;\n")
    assert tree.make() == (1, 1)
    tree.age()
    assert tree.make() == (0, 0)
    assert tree.make(cflags="-O2") == (0, 0)
    assert tree.make(cflags="-O1") == (1, 1)  # CFLAGS row changed: compiled again
    tree.age()
    assert tree.make(cflags="-O1") == (0, 0)
    assert tree.make(cflags="-O1", cppflags="-DY") == (0, 0)  # the same text: preprocessed again, nothing compiled
    assert "-DY" in (tmp_path / "build/us/src/a.c.i.ok.flags").read_text()


def test_an_object_without_a_flags_stamp_is_trusted(tmp_path):
    tree = Tree(tmp_path)
    tree.write("int a;\n")
    assert tree.make() == (1, 1)
    for stamp in list(tmp_path.rglob("*.flags")):
        stamp.unlink()
    tree.age()
    assert tree.make(cflags="-O3") == (0, 0)


def test_a_failed_compile_does_not_leave_a_stale_object_to_the_next_good_one(tmp_path):
    tree = Tree(tmp_path)
    tree.write("int a;\n")
    assert tree.make() == (1, 1)
    tree.age()
    tree.write("int a; // FAIL\n")
    tree.make(expect=2)
    assert (tmp_path / "build/us/src/a.c.s.prev").read_text() == "int a;\n"  # the assembly the object holds
    tree.age()
    tree.write("int b;\n")  # a different text: the object follows it
    assert tree.make() == (1, 1)
    assert (tmp_path / "build/us/src/a.c.o").read_bytes() == b"int b;\n"
    assert not (tmp_path / "build/us/src/a.c.s.prev").exists()


def test_a_failed_compile_never_makes_a_partial_assembly_the_last_one(tmp_path):
    tree = Tree(tmp_path)
    tree.write("int a;\n")
    assert tree.make() == (1, 1)
    tree.age()
    tree.write("FAIL\n")
    tree.make(expect=2)
    tree.make(expect=2)  # a second failure still keeps the assembly the object was made from
    assert (tmp_path / "build/us/src/a.c.s.prev").read_text() == "int a;\n"
    tree.age()
    tree.write("int a;\n")  # back to the text the object holds: nothing to assemble
    assert tree.make() == (1, 0)


def test_local_mk_names_the_venv_link_not_its_versioned_directory(tmp_path):
    venv = tmp_path / "venv-decomp.0123456789ab"
    (venv / "bin").mkdir(parents=True)
    (tmp_path / "venv-decomp").symlink_to(venv)
    assert build._stable(venv / "bin/splat") == tmp_path / "venv-decomp/bin/splat"
    other = tmp_path / "plain.dir"
    other.mkdir()
    assert build._stable(other / "x") == other / "x"
    host = SimpleNamespace(workers=1, tools={"splat": venv / "bin/splat"}, toolchain_root=Path("/c"), cache_dir=None)
    text = build.local_mk(SimpleNamespace(host=host)).decode()
    assert str(venv) not in text


def _old(path: Path) -> None:
    os.utime(path, ns=(10**18, 10**18))


def test_a_header_changed_under_a_newer_stamp_remakes_the_stamp(tmp_path):
    """Dates lie after a copy or a fast-forward: a stamp is newer than a header whose bytes it never saw."""
    tree = Tree(tmp_path)
    (tmp_path / "src/h.h").write_text("one\n")
    tree.write("int a;\n")
    tree.make(deps=True)
    tree.age()
    tree.make(deps=True)
    assert (tmp_path / "preprocessed").read_text() == ""
    (tmp_path / "src/h.h").write_text("two\n")
    _old(tmp_path / "src/h.h")
    tree.make(deps=True)
    assert (tmp_path / "preprocessed").read_text() == "p\n"
    tree.age()
    tree.make(deps=True)  # the new bytes are recorded: nothing runs again
    assert (tmp_path / "preprocessed").read_text() == ""


def test_an_extracted_file_changed_under_a_newer_object_remakes_the_object(tmp_path):
    tree = Tree(tmp_path)
    asm = tmp_path / "asm/us/nonmatchings/seg/f.s"
    asm.parent.mkdir(parents=True)
    asm.write_text("glabel f\n")
    tree.write('INCLUDE_ASM("seg", f);\n')
    assert tree.make(deps=True) == (1, 1)
    tree.age()
    tree.make(deps=True)
    assert tree.make(deps=True) == (0, 0)
    asm.write_text("glabel g\n")
    _old(asm)
    assert tree.make(deps=True) == (1, 1)


def test_an_included_asm_naming_a_renamed_callee_remakes_the_object_under_an_older_date(tmp_path):
    """After a rebase the extracted .s of a renamed callee is older than the object built before it: the object that
    includes it is keyed on the file's bytes, so it is assembled again and no link keeps the old name."""
    tree = Tree(tmp_path)
    for name in ("f", "g"):
        asm = tmp_path / f"asm/us/nonmatchings/seg/{name}.s"
        asm.parent.mkdir(parents=True, exist_ok=True)
        asm.write_text(f"glabel {name}\n  jal func_802A037C_de\n")
    tree.write('INCLUDE_ASM("seg", f);\nINCLUDE_ASM("seg", g);\n')
    assert tree.make(deps=True) == (1, 1)
    tree.age()
    tree.make(deps=True)
    assert tree.make(deps=True) == (0, 0)
    renamed = tmp_path / "asm/us/nonmatchings/seg/g.s"
    renamed.write_text("glabel g\n  jal Resmgr_CopyString\n")
    _old(renamed)  # extraction wrote it with a date older than the object
    assert tree.make(deps=True) == (1, 1)
    assert tree.make(deps=True) == (0, 0)


def test_an_extraction_inside_the_make_that_renamed_an_included_asm_remakes_the_object(tmp_path):
    """A make that extracts while it builds has read the file's old date (no sweep runs in a sub-make, level 1), and the
    object's other prerequisite is a newer .i: the shortcuts would keep the object that names the old callee."""
    tree = Tree(tmp_path)
    asm = tmp_path / "asm/us/nonmatchings/seg/f.s"
    asm.parent.mkdir(parents=True)
    asm.write_text("glabel f\n  jal func_802A037C_de\n")
    tree.write('INCLUDE_ASM("seg", f);\n')
    assert tree.make(deps=True, level=1) == (1, 1)
    tree.age()
    tree.make(deps=True, level=1)
    assert tree.make(deps=True, level=1) == (0, 0)
    asm.write_text("glabel f\n  jal Resmgr_CopyString\n")
    _old(asm)
    os.utime(tmp_path / "build/us/src/a.c.i.ok")  # only the stamp is newer: the shortcut is what decides
    assert tree.make(deps=True, level=1)[1] == 1
    assert tree.make(deps=True, level=1) == (0, 0)


def test_a_tree_without_recorded_inputs_is_trusted(tmp_path):
    tree = Tree(tmp_path)
    (tmp_path / "src/h.h").write_text("one\n")
    tree.write("int a;\n")
    tree.make(deps=True)
    (tmp_path / "build/.inputs.json").unlink()
    (tmp_path / "build/.inputs.log").unlink(missing_ok=True)
    tree.age()
    (tmp_path / "src/h.h").write_text("two\n")
    _old(tmp_path / "src/h.h")
    tree.make(deps=True)
    assert (tmp_path / "preprocessed").read_text() == ""


def test_records_append_without_a_lock_and_the_sweep_folds_them_in(tmp_path):
    """A record was a global flock + rewrite of the whole state (6.5 MB on a full tree): a cold make -j12 of all
    versions ran one record at a time (POOL-SRC refresh 21+ min at 11 s make CPU). A record now appends one line."""
    import json

    tree = Tree(tmp_path)
    (tmp_path / "src/h.h").write_text("one\n")
    tree.write("int a;\n")
    tree.make(deps=True)
    log = tmp_path / "build/.inputs.log"
    assert not (tmp_path / "build/.inputs.json.lock").exists()
    assert log.exists() and any(".i.ok" in line for line in log.read_text().splitlines())
    tree.age()
    tree.make(deps=True)  # the sweep folds the log into the state; this run's records start a new log
    state = json.loads((tmp_path / "build/.inputs.json").read_text())
    assert any(k.endswith(".i.ok") for k in state)


def test_the_c_record_writes_the_python_record_and_a_tree_without_a_compiler_still_records(tmp_path):
    """A python record per output was 20 ms (two per unit: 70% of a no-op header rebuild). build/stamprec writes the
    same line; without a host compiler the python record runs."""
    import json
    import shutil

    tree = Tree(tmp_path)
    (tmp_path / "src/h.h").write_text("one\n")
    tree.write("int a;\n")
    tree.make(deps=True)
    rec = tmp_path / "build/stamprec"
    if shutil.which("cc") is None:
        return
    assert rec.is_file()
    stamp = "build/us/src/a.c.i.ok"
    d = tmp_path / "build/us/src/a.c.d"
    d.write_text(f"{stamp}: src/a.c \\\n src/h.h src/gone.h build\r\nsrc/h.h:\n\nsrc/a.c: src/h.h\n")
    (tmp_path / "asm/us/b.s").write_text("b")
    (tmp_path / "asm/us/a.s").write_text("a")
    log = tmp_path / "build/.inputs.log"
    obj = ["build/us/src/a.c.o", "asm/us/b.s", "asm/us/a.s", "asm/us/b.s", "asm/us/none.s"]
    lines = []
    for argv in ([stamp], obj, ["build/us/src/nodeps.i.ok"]):
        log.write_text("")
        subprocess.run([sys.executable, "-I", "build/stamp-sweep.py", "record", *argv], cwd=tmp_path, check=False)
        subprocess.run([str(rec), "record", *argv], cwd=tmp_path, check=True)
        lines.append([json.loads(x) for x in log.read_text().splitlines()])
    assert len(lines[0]) == 2 and lines[0][0] == lines[0][1]
    assert len(lines[1]) == 2 and lines[1][0] == lines[1][1]
    assert lines[2] == []  # no .d: no key, no record, exit 0
    assert subprocess.run([str(rec), "record", 'build/q"uote.i.ok'], cwd=tmp_path).returncode == 2  # JSON escapes it

    bare = Tree(tmp_path / "bare") if (tmp_path / "bare").mkdir() is None else None
    (bare.root / "src/h.h").write_text("one\n")
    bare.write("int a;\n")
    env = os.environ.get("HOSTCC")
    os.environ["HOSTCC"] = "false"
    try:
        bare.make(deps=True)
    finally:
        os.environ.pop("HOSTCC") if env is None else os.environ.__setitem__("HOSTCC", env)
    assert not (bare.root / "build/stamprec").exists()
    assert any(".i.ok" in x for x in (bare.root / "build/.inputs.log").read_text().splitlines())


def test_asm_deps_is_replaced_whole_and_the_scripts_are_not_rewritten_unchanged(tmp_path):
    """REV1-ONE-40: `build/asm.deps:157: missing separator`: a make -included asm.deps another make of the same tree was
    rewriting in place (shell redirect). It is now written to a temporary file and replaced, and the generated scripts
    are written only when their text changed (a parallel make never runs a half-written one)."""
    import os

    tree = Tree(tmp_path)
    asm = tmp_path / "asm/us/nonmatchings/seg/f.s"
    asm.parent.mkdir(parents=True)
    asm.write_text("glabel f\n")
    tree.write('INCLUDE_ASM("seg", f);\n')
    tree.make(deps=True)
    deps, script, sweep = (tmp_path / "build" / n for n in ("asm.deps", "asm-deps.py", "stamp-sweep.py"))
    assert "asm/us/nonmatchings/seg/f.s" in deps.read_text()
    ino, when, swept = deps.stat().st_ino, script.stat().st_mtime_ns, sweep.stat().st_mtime_ns
    os.utime(script, ns=(10**9, 10**9))
    os.utime(sweep, ns=(10**9, 10**9))
    tree.make(deps=True)
    assert deps.stat().st_ino != ino  # a new file took the name: a reader sees the old one or the new one, whole
    assert script.stat().st_mtime_ns == 10**9 and sweep.stat().st_mtime_ns == 10**9  # unchanged text: not rewritten
    assert not list((tmp_path / "build").glob("asm.deps.*"))
    assert when and swept


def test_a_sweep_with_nothing_moved_does_not_walk_or_rewrite_the_state(tmp_path):
    """WATCH-0153: every make ran a full walk and rehash (two cores held across trees). With every input and output as
    the
    last sweep saw it, the sweep reads the small cache and leaves the state alone; a later record still joins it."""
    tree = Tree(tmp_path)
    (tmp_path / "src/h.h").write_text("one\n")
    tree.write("int a;\n")
    tree.make(deps=True)
    tree.age()
    tree.make(deps=True)  # the first sweep after the ageing walks and writes the cache
    state = tmp_path / "build/.inputs.json"
    assert (tmp_path / "build/.dep-stat.json").is_file()
    before = state.stat().st_mtime_ns
    tree.make(deps=True)
    tree.make(deps=True)
    assert state.stat().st_mtime_ns == before  # nothing was recorded or moved: no walk, no rewrite
    tree.write("int a; int b;\n")  # a new record joins the cache without a walk
    tree.make(deps=True)
    (tmp_path / "src/h.h").write_text("two\n")
    _old(tmp_path / "src/h.h")  # a header changed under a newer stamp is still found
    tree.make(deps=True)
    assert (tmp_path / "preprocessed").read_text() == "p\n"


def test_a_recursive_make_never_sweeps_the_objects_its_parent_is_about_to_link(tmp_path):
    tree = Tree(tmp_path)
    asm = tmp_path / "asm/us/nonmatchings/seg/f.s"
    asm.parent.mkdir(parents=True)
    asm.write_text("glabel f\n")
    tree.write('INCLUDE_ASM("seg", f);\n')
    assert tree.make(deps=True) == (1, 1)
    tree.age()
    tree.make(deps=True)
    asm.write_text("glabel g\n")
    _old(asm)
    assert tree.make(deps=True, level=1) == (0, 0)  # a sub-make leaves the object where the parent's link expects it
    assert tree.make(deps=True) == (1, 1)  # the top-level make's sweep still drops it


def test_a_stamp_signed_under_the_new_header_bytes_survives_the_sweep(tmp_path):
    """hdrnarrow.sign: a unit the narrowing proved unreachable keeps its stamp and object when a later make skips the
    header by date (`make -o`); unsigned, the sweep drops it (same date, other bytes) and it is preprocessed again."""
    import json

    from unbake import hdrnarrow

    tree = Tree(tmp_path)
    (tmp_path / "src/h.h").write_text("one\n")
    tree.write("int a;\n")
    tree.make(deps=True)
    tree.age()
    tree.make(deps=True)
    stamp = "build/us/src/a.c.i.ok"
    when = (tmp_path / stamp).stat().st_mtime_ns
    (tmp_path / "src/h.h").write_text("two\n")
    _old(tmp_path / "src/h.h")  # make skips the header by date, as `make -o src/h.h` does
    assert hdrnarrow.sign(tmp_path, [stamp, "build/us/src/none.c.i.ok"]) == 1
    last = (tmp_path / "build/.inputs.log").read_text().splitlines()[-1]
    assert json.loads(last)[:2] == [stamp, when]
    assert tree.make(deps=True) == (0, 0)
    assert (tmp_path / "preprocessed").read_text() == ""
    assert (tmp_path / stamp).stat().st_mtime_ns == when  # still pristine for the next narrowing
    (tmp_path / "src/h.h").write_text("three\n")  # unsigned: the sweep drops the stamp
    _old(tmp_path / "src/h.h")
    tree.make(deps=True)
    assert (tmp_path / "preprocessed").read_text() == "p\n"


def test_narrow_apply_lets_make_skip_the_header_and_rebuild_only_what_the_edit_reaches(tmp_path):
    """`unbake narrow --against REV --apply` (the train's check): a unit whose key names nothing the header edit
    changed is signed, and `make -o HEADER` leaves it; an edit it names dates it for a rebuild."""
    import shutil

    from unbake import hdrnarrow

    if shutil.which("cc") is None:
        return
    tree = Tree(tmp_path)
    (tmp_path / "src/h.h").write_text("extern int D_1;\nextern int D_2;\n")
    tree.write("int f(void) { return D_1; }\n")
    (tmp_path / "versions/us/symbol_addrs.txt").write_text("")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "base"]):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True)
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    tree.make(deps=True)
    tree.age()
    tree.make(deps=True)
    (tmp_path / "src/h.h").write_text("extern int D_1;\nextern short D_2;\n")  # D_2: no name the unit's key spells
    got = hdrnarrow.narrow_command({"project": str(tmp_path), "against": base, "apply": True, "version": ["us"]})
    assert got["skip_headers"] == ["src/h.h"] and got["versions"]["us"]["skipped"] == 1 and got["signed"] == 1
    assert tree.make(deps=True, extra=("-o", "src/h.h")) == (0, 0)
    assert (tmp_path / "preprocessed").read_text() == ""
    (tmp_path / "src/h.h").write_text("extern short D_1;\nextern short D_2;\n")  # D_1: the unit's key names it
    got = hdrnarrow.narrow_command({"project": str(tmp_path), "against": base, "apply": True, "version": ["us"]})
    assert got["versions"]["us"]["skipped"] == 0 and got["dropped"] == 1
    tree.make(deps=True, extra=("-o", "src/h.h"))
    assert (tmp_path / "preprocessed").read_text() == "p\n"


def test_a_process_of_an_older_venv_still_names_the_link(tmp_path):
    old, live = tmp_path / "venv-decomp.0123456789ab", tmp_path / "venv-decomp.ba9876543210"
    for v in (old, live):
        (v / "lib").mkdir(parents=True)
    (tmp_path / "venv-decomp").symlink_to(live)  # a deploy moved the link while a keeper ran from OLD
    assert build._stable(old / "lib/cc_front.sh") == tmp_path / "venv-decomp/lib/cc_front.sh"
    stranger = tmp_path / "venv-other.0123456789ab"
    stranger.mkdir()
    assert build._stable(stranger / "x") == stranger / "x"  # no link of its family: its own spelling


def test_narrow_apply_that_cannot_narrow_refuses(tmp_path, monkeypatch):
    import pytest

    from unbake import hdrnarrow, process
    from unbake.contracts import Refusal

    monkeypatch.setattr(process, "changed", lambda root, rev: ["include/w.h"])
    monkeypatch.setattr(hdrnarrow, "includers", lambda root, v, headers: ["src/a.c"])
    why = {"includers": 4305, "skipped": 0, "why": "4305 includers unsafe"}
    monkeypatch.setattr(hdrnarrow, "narrow_tree", lambda *a, **k: (["src/a.c"], why))
    params = {"project": str(tmp_path), "against": "HEAD", "version": ["us"]}
    assert hdrnarrow.narrow_command(params)["applied"] is False  # a report: nothing was asked to apply
    with pytest.raises(Refusal) as refused:
        hdrnarrow.narrow_command({**params, "apply": True})
    assert refused.value.findings[0].key == "narrow.unsafe"
