"""The object cache: an object made once under a key (toolchain, flags row, `.i` bytes) is copied, never made twice."""

from __future__ import annotations

import os
from pathlib import Path

from unbake.resources import cc_cache


class Tree:
    """A tree with one source text; its program is mocked: it counts its compiler runs, no process starts."""

    def __init__(self, root: Path, compiler: Path, cache: Path, text: str = "int a;") -> None:
        self.root, self.compiler, self.cache, self.text = root, compiler, cache, text
        self.runs = 0

    def key(self, flags: str = "-O2", textsize: str = "", program: str = "cc x.i", extracted: str = "") -> list[str]:
        return [
            str(self.cache),
            "x.o",
            str(self.compiler),
            flags,
            textsize,
            program,
            *([extracted] if extracted else []),
        ]

    def build(self, **key: str) -> str:
        """`@probe --objget`, else the compiler and `--objput` (the object rule's program); the object's text."""
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            (self.root / "x.i").write_text(self.text)
            args = self.key(**key)
            if cc_cache.object_get(*args[:6], args[6:]) != 0:  # type: ignore[arg-type]
                self.runs += 1
                (self.root / "x.o").write_text(self.text.upper() + key.get("flags", "-O2"))
                assert cc_cache.object_put(*args[:6], args[6:]) == 0  # type: ignore[arg-type]
            return (self.root / "x.o").read_text()
        finally:
            os.chdir(cwd)


def trees(tmp_path: Path) -> tuple[Tree, Tree]:
    compiler = tmp_path / "tc" / "cc"
    compiler.parent.mkdir()
    compiler.write_text("cc")
    cache = tmp_path / "host.obj"
    for name in "ab":
        (tmp_path / name).mkdir()
    return Tree(tmp_path / "a", compiler, cache), Tree(tmp_path / "b", compiler, cache)


def test_an_identical_key_hits_in_another_tree(tmp_path):
    a, b = trees(tmp_path)
    assert a.build() == "INT A;-O2"
    assert b.build() == "INT A;-O2"
    assert (a.runs, b.runs) == (1, 0)


def test_a_flag_change_misses(tmp_path):
    a, b = trees(tmp_path)
    a.build()
    assert b.build(flags="-O1") == "INT A;-O1"
    assert b.runs == 1


def test_an_i_change_misses(tmp_path):
    a, b = trees(tmp_path)
    a.build()
    b.text = "int b;"
    assert b.build() == "INT B;-O2"
    assert b.runs == 1


def test_the_text_size_the_program_and_the_extracted_files_decide_the_key(tmp_path):
    a, b = trees(tmp_path)
    (tmp_path / "b" / "d.s").write_text("one")
    a.build()
    for change in ({"textsize": "0x40"}, {"program": "cc -g x.i"}, {"extracted": "d.s"}):
        before = b.runs
        b.build(**change)
        assert b.runs == before + 1, change
    (tmp_path / "b" / "d.s").write_text("two")
    b.build(extracted="d.s")
    assert b.runs == 4


def test_a_changed_compiler_misses(tmp_path):
    a, b = trees(tmp_path)
    a.build()
    cc_cache.IDENTITIES.clear()
    b.compiler.write_text("another cc")
    b.build()
    assert b.runs == 1


def test_the_hit_is_a_copy_the_trim_cannot_reach_the_cache_through(tmp_path):
    a, b = trees(tmp_path)
    a.build()
    b.build()
    (tmp_path / "b" / "x.o").write_text("trimmed")  # trim_text rewrites the object in place
    assert a.build() == "INT A;-O2"
    assert b.runs == 0
    assert Tree(tmp_path / "a", a.compiler, a.cache).build() == "INT A;-O2"


def test_a_cache_that_cannot_be_read_is_a_miss_and_costs_only_the_saving(tmp_path):
    a, _ = trees(tmp_path)
    a.cache = tmp_path / "x.o"  # a file where the directory should be
    a.cache.write_text("")
    assert a.build() == "INT A;-O2"
    assert a.runs == 1


def test_the_assembly_comes_back_with_the_object(tmp_path):
    a, b = trees(tmp_path)
    (tmp_path / "a" / "x.s").write_text("asm")
    a.build()
    (tmp_path / "b" / "x.s").unlink(missing_ok=True)
    b.build()
    assert (tmp_path / "b" / "x.s").read_text() == "asm"


def test_probe_skips_the_program_on_a_hit_and_runs_it_on_a_miss(tmp_path):
    state: dict[str, object] = {"stop": False}
    assert cc_cache.object_step(["@probe", "true"], state) == 0 and state["stop"] is True
    assert cc_cache.object_step(["@probe", "sh", "-c", f"exit {cc_cache.OBJECT_MISS}"], state) == 0
    assert state["stop"] is False
    assert cc_cache.object_step(["@probe", "sh", "-c", "exit 1"], state) == 1


def test_local_mk_declares_the_object_cache_beside_the_host_content_cache():
    from types import SimpleNamespace

    from unbake import build

    host = SimpleNamespace(workers=4, tools={}, toolchain_root=Path("/chains"), cache_dir=Path("/c/unbake/content"))
    assert "OBJCACHE := /c/unbake/content.obj\n" in build.local_mk(SimpleNamespace(host=host)).decode()
    bare = SimpleNamespace(workers=4, tools={}, toolchain_root=Path("/chains"))
    assert "OBJCACHE := \n" in build.local_mk(SimpleNamespace(host=bare)).decode()


def test_every_recipe_probes_the_cache_then_stores_what_its_program_made():
    from unbake import build, config

    for id in config.toolchains():
        text = build._recipe(id)
        key = build._key(id)
        assert f"COMPILE_{key} = $(if $(OBJCACHE),@probe $(CCPY) --objget " in text
        assert text.count("--objget") == text.count("--objput") == 1
        assert f"'&&' $(PROGRAM_{key}) '&&' $(CCPY) --objput" in text
        assert text.endswith(f",$(PROGRAM_{key}))")


def test_the_object_rule_program_skips_to_the_trim_on_a_hit(tmp_path, monkeypatch):
    """object_file with the cache's own `--objget`/`--objput` as steps: the second tree runs no program step."""
    import sys

    script = Path(cc_cache.__file__)
    compiler = tmp_path / "cc"
    compiler.write_text("cc")
    cache = tmp_path / "host.obj"
    for name in "ab":
        (tmp_path / name).mkdir()
        (tmp_path / name / "x.i").write_text("int a;")
        (tmp_path / name / "build").mkdir()
    cmd = [sys.executable, "-I", str(script)]
    key = [str(cache), "x.o", str(compiler), "-O2", "", "program"]
    program = [
        "@probe", *cmd, "--objget", *key, "&&",
        "sh", "-c", "echo run >> ../runs; echo obj > x.o", "&&",
        *cmd, "--objput", *key,
    ]  # fmt: skip
    for name in "ab":
        monkeypatch.chdir(tmp_path / name)
        assert cc_cache.object_file("x.o", "us", "", "", "", "-O2", "", ["--", *program]) == 0
        assert (tmp_path / name / "x.o").read_text() == "obj\n"
    assert (tmp_path / "runs").read_text().count("run") == 1
