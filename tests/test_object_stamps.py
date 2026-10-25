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

    def make(self, cflags: str = "-O2", cppflags: str = "-DX", expect: int = 0, deps: bool = False) -> tuple[int, int]:
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
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
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
    """A record was a global flock + rewrite of the whole state (6.5 MB on a full tree): a cold make -j12 of all versions
    ran one record at a time (POOL-SRC refresh 21+ min at 11 s make CPU). A record now appends one line."""
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
