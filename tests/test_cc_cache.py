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
    assert compile_in(tree(tmp_path, "b", "int b;"), cache, cc) == "INT B;"
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
