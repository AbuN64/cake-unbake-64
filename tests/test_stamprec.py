"""stamprec's `object` (C, built from the Makefile's own text) does what cc_cache.py's python does, byte for byte."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from test_build import make_config

from unbake import build
from unbake.resources import cc_cache

CC = Path(cc_cache.__file__)

# The argument vector of a real object (`build/us/.../802A8A94_2.CPP.o` of a RageWars tree, captured from `make -n`):
# the extracted facts it was made from, the flags it was made with and its text size. Its compile chain is not run
# here (no compiler): an object that stands never starts it, and the chain that does run is a stub.
OUT = "build/us/src/game/ONSCRN_2/802A8A94_2/802A8A94_2.CPP.o"
FACTS = "41327499592ee592e97f31afe3ad49f99b5963a0"
FLAGS = (
    "gcc_2_8_1_sn64 sh /v/unbake/resources/cc_front.sh /c/content.cc /t/gcc-2.8.1-sn64/cc1 -G0 -mips3 -O2 -mgas -meb "
    "-mcpu=VR4300 -mhard-float -mgp32 -mfp64 -mno-fix4300 -march=vr4300 -mabi=32 -EB --no-pad-sections "
    "-mips3 -DVERSION_US  "
)
S = OUT[:-1] + "s"
FRONT = [
    "sh",
    "/v/unbake/resources/cc_front.sh",
    "/c/content.cc",
    "/t/gcc-2.8.1-sn64/cc1",
    "-quiet",
    OUT[:-1] + "i",
    "-o",
    S,
]
REAL_CHAIN = ["@rotate", S, S + ".prev", "&&", "@compile", *FRONT, "&&", "@nonempty", S]


@pytest.fixture(scope="module")
def stamprec(tmp_path_factory) -> str:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    text = build.makefile(make_config(tmp_path_factory.mktemp("cfg"))).decode()
    source = re.search(r"define STAMPREC_C\n(.*?)\nendef\n", text, re.S)[1].replace("$$", "$")  # make's own unescape
    folder = tmp_path_factory.mktemp("stamprec")
    (folder / "stamprec.c").write_text(source + "\n")
    done = subprocess.run(
        [cc, "-O2", "-Wall", "-Werror", "-o", str(folder / "stamprec"), str(folder / "stamprec.c")],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return str(folder / "stamprec")


def elf(text: bytes) -> bytes:
    names = b"\0.text\0.shstrtab\0"
    body = text + names
    head = (
        b"\x7fELF\x01\x02\x01"
        + bytes(9)
        + struct.pack(">HHIIIIIHHHHHH", 1, 8, 1, 0, 0, 52 + len(body), 0, 52, 0, 0, 40, 3, 2)
    )
    text_h = struct.pack(">IIIIIIIIII", 1, 1, 6, 0, 52, len(text), 0, 0, 16, 0)
    str_h = struct.pack(">IIIIIIIIII", 7, 3, 0, 0, 52 + len(text), len(names), 0, 0, 1, 0)
    return head + body + bytes(40) + text_h + str_h


def tree(root: Path, *, state: str) -> None:
    """A tree holding an object of `state`: 'stands' (its text is older than it, size and sum recorded) or 'stale'."""
    (root / "asm/us").mkdir(parents=True)
    (root / "asm/us/.extracted").write_text(FACTS + "\n")
    (root / "asm/us/a.s").write_text("one\n")
    (root / "asm/us/b.s").write_text("two\n")
    (root / "build").mkdir()
    (root / "make-object.py").write_text(
        f"import sys\nopen(sys.argv[1], 'wb').write(bytes.fromhex({elf(b'\x03\xe0\x00\x08' + bytes(12)).hex()!r}))\n"
    )
    (root / "build/.inputs.log").write_text("")
    out = root / OUT
    out.parent.mkdir(parents=True)
    source = root / (OUT[:-1] + "i")
    source.write_text("text\n")
    out.write_bytes(elf(b"\x03\xe0\x00\x08" + bytes(12)))
    (root / (OUT + ".textsize")).write_text("0x638\n" if state == "stands" else "0x10\n")
    os.utime(source, ns=(10**9, 10**9))
    os.utime(out, ns=(2 * 10**9, 2 * 10**9))
    total = hashlib.sha1(b"one\ntwo\n").hexdigest()
    (root / (OUT + ".asm")).write_text(f"{total}  -\n")


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file by name; the record's date (when the object was made) is not part of what must agree."""
    files = {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}
    files["build/.inputs.log"] = re.sub(rb", \d+, ", b", DATE, ", files["build/.inputs.log"])
    return files


def run_object(
    root: Path, how: str, stamprec: str, changed: str, textsize: str, program: list[str], env=None
) -> subprocess.CompletedProcess:
    args = [OUT, "us", FACTS, changed, textsize, FLAGS, stamprec, "asm/us/a.s", "asm/us/b.s", "--", *program]
    command = [stamprec, "object", *args] if how == "c" else [sys.executable, "-I", "-S", str(CC), "--object", *args]
    return subprocess.run(command, cwd=root, capture_output=True, text=True, env={**os.environ, **(env or {})})


def both(
    tmp_path: Path, stamprec: str, state: str, changed: str, textsize: str, program: list[str], extra=None, env=None
):
    """The same object made by the C tool and by the python: the same files, bytes and record afterwards."""
    done = {}
    for how in ("c", "py"):
        root = tmp_path / how
        tree(root, state=state)
        if extra:
            extra(root)
        done[how] = (run_object(root, how, stamprec, changed, textsize, program, env), snapshot(root))
    (c, c_files), (py, py_files) = done["c"], done["py"]
    assert (c.returncode, c.stderr.replace("stamprec:", "cc_cache:")) == (py.returncode, py.stderr)
    assert c_files.keys() == py_files.keys()
    for name in c_files:
        assert c_files[name] == py_files[name], name
    return c, c_files


def test_a_real_objects_vector_leaves_a_standing_object_alone_without_starting_its_compile(tmp_path, stamprec):
    done, files = both(tmp_path, stamprec, "stands", "", "0x638", REAL_CHAIN)
    assert done.returncode == 0, done.stderr
    assert files[OUT + ".flags"] == (FLAGS + "\n").encode()
    assert files[OUT + ".asm"].endswith(b"  -\n")
    assert files["build/.inputs.log"].startswith(f'["{OUT}", '.encode())  # recorded once


MAKE = [
    sys.executable,
    "make-object.py",
    OUT,
]  # the stub chain: the assembler's object, 16 bytes of text, 12 of padding


def text_size(data: bytes) -> int:
    return struct.unpack_from(">I", data, struct.unpack_from(">I", data, 32)[0] + 40 + 20)[0]


@pytest.mark.parametrize("state,changed", [("stands", "x.i"), ("stands", ""), ("stale", "")])
def test_an_object_that_does_not_stand_compiles_then_trims_and_records_what_it_was_made_from(
    tmp_path, stamprec, state, changed
):
    """An input moved, or the recorded size is not the wanted one (0x8 against 0x638 or 0x10)."""
    done, files = both(tmp_path, stamprec, state, changed, "0x8", MAKE)
    assert done.returncode == 0, done.stderr
    assert (files[OUT + ".textsize"], files[OUT + ".flags"]) == (b"0x8\n", (FLAGS + "\n").encode())
    assert text_size(files[OUT]) == 8  # the zero tail past the wanted size is cut
    assert files["build/.inputs.log"].count(b"\n") == 1


def test_a_compile_that_reports_the_same_assembly_keeps_its_trim(tmp_path, stamprec):
    def same(root: Path) -> None:
        (root / (OUT + ".same")).write_text("")

    done, files = both(tmp_path, stamprec, "stands", "x.i", "0x8", MAKE, same)
    assert done.returncode == 0 and OUT + ".same" not in files
    assert text_size(files[OUT]) == 16 and files[OUT + ".textsize"] == b"0x638\n"  # trim and size file as they were


def test_an_object_of_other_extracted_facts_is_refused_and_one_of_other_bytes_is_made_again(tmp_path, stamprec):
    def other(root: Path) -> None:
        (root / "asm/us/.extracted").write_text("0" * 40 + "\n")

    done, _ = both(tmp_path / "facts", stamprec, "stands", "", "0x638", REAL_CHAIN, other)
    assert done.returncode == 1 and "run make extract-us" in done.stderr

    def moved(root: Path) -> None:
        (root / "asm/us/b.s").write_text("other\n")

    done, files = both(tmp_path / "bytes", stamprec, "stands", "", "0x638", MAKE, moved)
    assert done.returncode == 0 and text_size(files[OUT]) == 16  # the object stood, but the sum moved: compiled again


def test_a_failed_compile_fails_the_object_with_its_code_and_leaves_no_bookkeeping(tmp_path, stamprec):
    done, files = both(tmp_path, stamprec, "stale", "", "0x8", [sys.executable, "-c", "raise SystemExit(7)"])
    assert done.returncode == 7
    assert files[OUT + ".textsize"] == b"0x10\n" and OUT + ".flags" not in files


def front(args: list[str]) -> list[str]:
    return ["@compile", "sh", "/v/unbake/resources/cc_front.sh", "/c/content.cc", "cc1", *args]


@pytest.mark.parametrize("size", [0, 1, 63, 64, 65, 70_000])
@pytest.mark.parametrize("answer", ["right", "wrong", "none"])
def test_the_compile_step_moves_the_answer_only_for_its_own_command_and_text(tmp_path, stamprec, size, answer):
    """The key is the python's answer_key (SHA-256 over the command joined by \\037, a newline and the text); a unit
    without its answer goes to the front itself (here a front that is not there: exit 127)."""
    text = bytes((i * 7) % 251 for i in range(size))
    args = ["-quiet", "-G0", OUT[:-1] + "i", "-o", S]

    def answered(root: Path) -> None:
        (root / (OUT[:-1] + "i")).write_bytes(text)
        if answer != "none":
            (root / (S + ".cc")).write_bytes(b"assembly")
            key = cc_cache.answer_key("cc1", args, text if answer == "right" else text + b"\n")
            (root / (S + ".cc.key")).write_text(key + "\n")

    done, files = both(tmp_path, stamprec, "stale", "x.i", "0x8", front(args), answered)
    assert (done.returncode == 0) == (answer == "right")
    assert (S in files) == (answer == "right")
    assert S + ".cc" not in files and S + ".cc.key" not in files


def test_the_compile_step_records_a_collected_command_and_fails_it(tmp_path, stamprec):
    args = ["-quiet", OUT[:-1] + "i", "-o", S]
    done, files = both(tmp_path, stamprec, "stale", "x.i", "0x8", front(args), env={"ABU_CC_COLLECT": "collected.txt"})
    assert done.returncode == 125
    assert files["collected.txt"] == ("cc1\x1f" + "\x1f".join(args) + "\n").encode()
    assert OUT + ".flags" not in files  # a collecting run makes nothing


def test_the_compile_step_starts_a_command_that_is_not_the_front(tmp_path, stamprec):
    done, files = both(tmp_path, stamprec, "stale", "x.i", "0x8", ["@compile", *MAKE])
    assert done.returncode == 0 and text_size(files[OUT]) == 8


ASM = (
    "\t.word\t.L12\n\t.word\t$L7  \n.word .L9\t\n\t.word\t.L1+4\n\t.word\tfoo\n"
    "x: .word .L2\n  .word\t.L33\n\t.word.L3\n\t.word\t.L"
)


def write_asm(text: str) -> list[str]:
    return [sys.executable, "-c", "import sys;open(sys.argv[1], 'w').write(sys.argv[2])", S, text]


@pytest.mark.parametrize("ending", ["\n", ""])
def test_the_jumptable_step_biases_the_labels_a_table_holds_as_sed_does(tmp_path, stamprec, ending):
    program = [
        *write_asm(ASM + ending),
        "&&",
        "@jumptable",
        S,
        "80000000",
        "&&",
        "@nonempty",
        S,
        "&&",
        "sh",
        "-c",
        "exit 0",
    ]
    done, files = both(tmp_path, stamprec, "stale", "", "0x8", [*program[:-4], "&&", *MAKE])
    assert done.returncode == 0, done.stderr
    sed = subprocess.run(
        ["sed", "-E", r"s/^([ \t]*\.word[ \t]+[.$]L[0-9]+)[ \t]*$/\1-0x80000000/"],
        input=(ASM + ending).encode(),
        capture_output=True,
    )
    assert files[S] == sed.stdout and b"-0x80000000" in sed.stdout


@pytest.mark.parametrize("kept,stops", [("same\n", True), ("other\n", False)])
def test_the_settle_step_keeps_an_object_whose_assembly_did_not_change_and_drops_the_kept_one(
    tmp_path, stamprec, kept, stops
):
    program = [
        "@rotate",
        S,
        S + ".prev",
        "&&",
        *write_asm("same\n"),
        "&&",
        "@settle",
        S,
        S + ".prev",
        "&&",
        "sh",
        "-c",
        'echo >> assembled && exec "$@"',
        "sh",
        *MAKE,
    ]

    def tree_with_kept(root: Path) -> None:
        (root / (S + ".prev")).write_text(kept)
        os.utime(root / (OUT[:-1] + "i"), ns=(3 * 10**9, 3 * 10**9))  # the text moved (a new .i), nothing else did

    done, files = both(tmp_path, stamprec, "stands", "", "0x638", program, tree_with_kept)
    assert done.returncode == 0, done.stderr
    assert S + ".prev" not in files and OUT + ".same" not in files
    assert ("assembled" not in files) == stops  # stopped: the object is as it was; else the assembler ran again
    assert files[OUT + ".textsize"] == b"0x638\n"


def test_a_rotate_keeps_the_last_assembly_and_a_kept_one_still_there_says_the_last_run_failed(tmp_path, stamprec):
    for how, kept in (("moved", False), ("failed run", True)):

        def tree_with(root: Path, kept: bool = kept) -> None:
            (root / S).write_text("assembly\n")
            if kept:
                (root / (S + ".prev")).write_text("older\n")

        done, files = both(
            tmp_path / how, stamprec, "stale", "", "0x8", ["@rotate", S, S + ".prev", "&&", "@nonempty", S], tree_with
        )
        assert done.returncode == 1 and S not in files  # the program stopped at the empty/missing assembly
        assert (files.get(S + ".prev") == b"assembly\n") if not kept else (files[S + ".prev"] == b"older\n")


def test_the_out_and_py_steps_redirect_a_command_and_run_an_exported_pass(tmp_path, stamprec):
    program = ["@out", S, "echo", "text", "&&", "@py", "PASS", S, "&&", "@nonempty", S, "&&", *MAKE]
    done, files = both(
        tmp_path,
        stamprec,
        "stale",
        "",
        "0x8",
        program,
        env={"PASS": "import sys;open(sys.argv[1], 'a').write('more\\n')"},
    )
    assert done.returncode == 0, done.stderr
    assert files[S] == b"text\nmore\n"


def test_a_step_that_does_not_exist_or_a_program_that_is_missing_fails_the_object(tmp_path, stamprec):
    done, _ = both(tmp_path / "a", stamprec, "stale", "", "0x8", ["@nothing", "x"])
    assert done.returncode == 64 and "no step @nothing" in done.stderr
    done, _ = both(tmp_path / "b", stamprec, "stale", "", "0x8", ["no-such-program-here"])
    assert done.returncode == 127
