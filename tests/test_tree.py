"""tree: ownership, link order and the file cache, over the schema-4 fixture and its fake cpp."""

from __future__ import annotations

from pathlib import Path

import fixture
import pytest

from unbake import api, pool, tree
from unbake.contracts import Config, Refusal


@pytest.fixture(autouse=True)
def _inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pool jobs run in this process, so spies see them."""
    monkeypatch.setattr(pool, "_INLINE_AT", 10**6)


def edit(config: Config, path: str, text: str) -> None:
    (config.project.root / path).write_text(text)


def test_ownership(tmp_path: Path) -> None:
    held = tree.capture(fixture.config(tmp_path))
    assert [m.path for m in held.modules] == ["src/main/c_func.c", "src/main/stand_in.c", "src/ovl/fuzzy.c"]
    assert held.owner("a", "c_func") == "src/main/c_func.c"
    assert held.owner("b", "stand_in") == "src/main/stand_in.c"
    assert held.owner("a", "nope") is None
    by_path = {m.path: m for m in held.modules}
    assert by_path["src/main/c_func.c"].owns == {"a": ("c_func",), "b": ("c_func",)}
    assert by_path["src/main/c_func.c"].stand_ins == {}
    assert by_path["src/main/stand_in.c"].stand_ins == {"a": ("stand_in",), "b": ("stand_in",)}
    assert [m.segment for m in held.modules] == ["main", "main", "ovl"]


def test_non_matching_detected(tmp_path: Path) -> None:
    held = tree.capture(fixture.config(tmp_path))
    fuzzy = {m.path: m for m in held.modules}["src/ovl/fuzzy.c"]
    assert fuzzy.fuzzy == ("fuzzy",) and fuzzy.stand_ins == {"a": ("fuzzy",), "b": ("fuzzy",)}
    assert not any(m.fuzzy for m in held.modules if m.path != "src/ovl/fuzzy.c")


def test_unowned_and_duplicate(tmp_path: Path) -> None:
    config = fixture.config(tmp_path, sources={"src/main/c_func.c": "s32 c_func(void) {\n    return 1;\n}\n"})
    assert tree.capture(config).unowned("a") == ("stand_in", "fuzzy")
    twice = {
        "src/main/c_func.c": "s32 c_func(void) { return 1; }\n",
        "src/main/other.c": "s32 c_func(void) { return 2; }\n",
    }
    with pytest.raises(Refusal) as error:
        tree.capture(fixture.config(tmp_path / "twice", sources=twice))
    finding = error.value.findings[0]
    assert finding.key == "tree.duplicate" and finding.unit == "c_func" and finding.versions == ("a",)


def test_statics_and_unlisted_names_are_not_owned(tmp_path: Path) -> None:
    text = "static s32 helper(void) { return 0; }\nextern s32 c_func(void);\ns32 local[2] = {1, 2};\n"
    text += "s32 c_func(void) { return 1; }\n"
    held = tree.capture(
        fixture.config(tmp_path, sources={"src/main/a.c": text, "src/main/b.c": "s32 helper(void) { return 3; }\n"})
    )
    assert [m.path for m in held.modules] == ["src/main/a.c"] and held.modules[0].owns["a"] == ("c_func",)


def test_derived_order(tmp_path: Path) -> None:
    sources = {
        "src/main/z_first.c": "s32 c_func(void) { return 1; }\n",
        "src/main/a_second.c": "s32 stand_in(void) { return 1; }\n",
        "src/ovl/fuzzy.c": "s32 fuzzy(void) { return 1; }\n",
    }
    held = tree.capture(fixture.config(tmp_path, sources=sources))
    assert held.order("a", "main") == ("src/main/z_first.c", "src/main/a_second.c")  # by address, not by name
    assert held.order("a", "ovl") == ("src/ovl/fuzzy.c",) and held.order("a", "gone") == ()


def test_a_module_may_own_names_of_several_segments_and_belongs_to_its_first_functions(tmp_path: Path) -> None:
    both = "s32 fuzzy(void) { return 1; }\ns32 c_func(void) { return 1; }\n"
    held = tree.capture(fixture.config(tmp_path, sources={"src/main/both.c": both}))
    (module,) = held.modules
    assert module.segment == "main"  # c_func has the lower address; fuzzy sits in ovl
    assert module.owns["a"] == ("c_func", "fuzzy")
    assert held.order("a", "ovl") == () and held.order("a", "main") == ("src/main/both.c",)


def test_a_symbol_in_no_segment_is_refused(tmp_path: Path) -> None:
    far = "far = 0x80000100; // type:func rom:0x10 size:0x10\n"
    symbols = {"a": fixture.symbol_file("a") + far, "b": fixture.symbol_file("b")}
    sources = {"src/main/far.c": "s32 far(void) { return 1; }\n"}
    with pytest.raises(Refusal) as error:
        tree.capture(fixture.config(tmp_path, sources=sources, symbols=symbols))
    assert error.value.findings[0].key == "tree.segments" and error.value.findings[0].unit == "far"


def test_per_version_if_file(tmp_path: Path) -> None:
    sources = {
        "src/main/c_func.c": "s32 c_func(void) { return 1; }\n",
        "src/main/only_b.c": "#if VERSION_B\ns32 stand_in(void) { return 2; }\n#endif\n",
    }
    held = tree.capture(fixture.config(tmp_path, sources=sources))
    assert held.owner("b", "stand_in") == "src/main/only_b.c" and held.owner("a", "stand_in") is None
    assert held.unowned("a") == ("stand_in", "fuzzy") and held.unowned("b") == ("fuzzy",)
    only = {m.path: m for m in held.modules}["src/main/only_b.c"]
    assert dict(only.owns) == {"b": ("stand_in",)}


def test_scanner_reads_depth_zero_only() -> None:
    text = """
    typedef struct { int x; } T;
    extern int ghost;
    int proto(int);
    static int (*hook)(int) = 0;
    struct S { int inner; } placed = {1};
    const char *label = "int fake(void) { }";   /* int comment(void) { } */
    int arr[] = {1, 2};
    int one, two = 2;
    int run(int a)
    {
        int not_global = 1; if (a) { return 1; }
        return 0;
    }
    __asm__(".include \\"nonmatchings/" "main" "/" "later" ".s\\"");
    """
    defined, stand_ins = tree._scan(text)
    assert defined == ["hook", "placed", "label", "arr", "one", "two", "run"] and stand_ins == ["later"]



def test_scanner_reads_functions_returning_aggregates() -> None:
    text = """
    struct Point *at(unsigned id) { return 0; }
    union U make(void) { union U u; return u; }
    enum E pick(void) { return 0; }
    void after(void) { }
    """
    assert tree._scan(text) == (["at", "make", "pick", "after"], [])

def test_second_capture_parses_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = fixture.config(tmp_path)
    calls = []
    real = tree._parsed

    def spy(*args, **kw):
        calls.append(args[1])
        return real(*args, **kw)

    monkeypatch.setattr(tree, "_parsed", spy)
    first = tree.capture(config)
    assert len(calls) == 6  # three files, two versions
    runs = fixture.cpp_runs(tmp_path)
    assert tree.capture(config) is first and len(calls) == 6 and fixture.cpp_runs(tmp_path) == runs


def test_edit_reparses_only_the_edited_file(tmp_path: Path) -> None:
    config = fixture.config(tmp_path)
    tree.capture(config)
    before = fixture.cpp_runs(tmp_path)
    edit(config, "src/main/c_func.c", "s32 c_func(void) { return 7; }\n")
    tree.capture(config)
    assert fixture.cpp_runs(tmp_path) - before == 4  # one file, two versions, two runs each


def test_rename_hits_file_cache(tmp_path: Path) -> None:
    config = fixture.config(tmp_path)
    tree.capture(config)
    before = fixture.cpp_runs(tmp_path)
    root = config.project.root
    (root / "src/main/renamed.c").write_text((root / "src/main/c_func.c").read_text())
    (root / "src/main/c_func.c").unlink()
    held = tree.capture(config)
    assert fixture.cpp_runs(tmp_path) == before and held.owner("a", "c_func") == "src/main/renamed.c"


def test_overlay_replaces_a_file_without_touching_disk(tmp_path: Path) -> None:
    config = fixture.config(tmp_path)
    base = tree.capture(config)
    over = tree.capture(
        config, {"src/main/c_func.c": b"s32 c_func(void) { return 9; }\ns32 extra(void) { return 0; }\n"}
    )
    assert over.digest != base.digest and over.owner("a", "c_func") == "src/main/c_func.c"
    assert (config.project.root / "src/main/c_func.c").read_text().count("9") == 0
    gone = tree.capture(config, {"src/main/c_func.c": None})
    assert gone.owner("a", "c_func") is None


def test_api_capture_is_tree_capture(tmp_path: Path) -> None:
    assert api.capture is tree.capture


def test_every_owned_name_has_the_linker_section_its_bytes_are_in(tmp_path: Path) -> None:
    sources = {
        "src/main/mixed.c": (
            '#include "include_asm.h"\n\nconst s32 table[2] = {1, 2};\ns32 counter = 3;\ns32 scratch;\n'
            "s32 c_func(void) { return 1; }\nINCLUDE_RODATA(main, jtbl);\nINCLUDE_DATA(main, blob);\n"
            "INCLUDE_ASM(main, stand_in);\n"
        )
    }
    symbols = {
        v: "".join(
            f"{n} = 0x{0x80000400 + i * 0x10:08X}; // type:{k} rom:0x{0x1000 + i * 0x10:X} size:0x10\n"
            for i, (n, k) in enumerate(
                [("c_func", "func"), ("table", "data"), ("counter", "data"), ("scratch", "data"), ("jtbl", "data"),
                 ("blob", "data"), ("stand_in", "func")]
            )
        )
        for v in ("a", "b")
    }  # fmt: skip
    held = tree.capture(fixture.config(tmp_path, sources=sources, symbols=symbols))
    (module,) = held.modules
    assert dict(module.sections["a"]) == {
        "c_func": ".text",
        "table": ".rodata",
        "counter": ".data",
        "scratch": ".bss",
        "jtbl": ".rodata",
        "blob": ".data",
        "stand_in": ".text",
    }
    assert module.placed["a"]["table"] == "main"
