"""tree: ownership, link order and the file cache, over the schema-4 fixture and its fake cpp."""

from __future__ import annotations

import inspect
from pathlib import Path

import fixture
import pytest

from unbake import api, effort, pool, store, tree
from unbake.contracts import Config, Refusal

pytestmark = pytest.mark.usefixtures("toolchains")  # the parse runs the toolchain row's preprocess template


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


def test_a_stand_in_without_the_header_still_owns_its_name(tmp_path: Path) -> None:
    """The header is written on top by sync; until then the file is read as it will be, not left unowned."""
    sources = {
        "src/main/c_func.c": "s32 c_func(void) { return 1; }\n",
        "src/main/stand_in.c": "INCLUDE_ASM(main, stand_in);\n",
    }
    held = tree.capture(fixture.config(tmp_path, sources=sources))
    assert held.owner("a", "stand_in") == "src/main/stand_in.c"


def test_labels_inside_data_need_no_owner(tmp_path: Path) -> None:
    """A data name after data in one mapping is a label of that piece; a name at an owned address is a second name of
    those bytes; a function after data, and data after a function or in another mapping, still need an owner."""
    rows = {
        "c_func": ("main", 0x0, "func"),
        "table": ("main", 0x20, "data"),
        "inside": ("main", 0x24, "data"),
        "deeper": ("main", 0x28, "data"),
        "after": ("main", 0x30, "func"),
        "tail": ("main", 0x40, "data"),
        "fuzzy": ("ovl", 0x0, "func"),
        "first": ("ovl", 0x20, "data"),
    }
    text = fixture.symbol_file("a", rows) + "same_c_func = 0x80000400; // type:func rom:0x1000\n"
    sources = {"src/main/c.c": "s32 c_func(void) { return 1; }\ns32 table[1] = {1};\n"}
    held = tree.capture(fixture.config(tmp_path, versions=("a",), sources=sources, symbols={"a": text}))
    assert held.unowned("a") == ("after", "tail", "fuzzy", "first")


def test_literals_fact_is_rodata_of_the_file_owning_its_function(tmp_path: Path) -> None:
    """`literals_<fn>` is owned by the file that owns <fn>, as .rodata, and is never offered as unowned; a function
    no file owns makes it a finding."""
    rows = {"c_func": ("main", 0x0, "func"), "literals_c_func": ("main", 0x20, "data")}
    text = fixture.symbol_file("a", rows)
    sources = {"src/main/c.c": "s32 c_func(void) { return 1; }\n"}
    held = tree.capture(fixture.config(tmp_path, versions=("a",), sources=sources, symbols={"a": text}))
    assert held.owner("a", "literals_c_func") == "src/main/c.c" and held.unowned("a") == ()
    assert held.modules[0].sections["a"]["literals_c_func"] == ".rodata"
    bad = fixture.symbol_file("a", {"literals_gone": ("main", 0x20, "data")})
    with pytest.raises(Refusal) as error:
        tree.capture(fixture.config(tmp_path / "bad", versions=("a",), sources=sources, symbols={"a": bad}))
    assert error.value.findings[0].key == "tree.literals"


def test_literals_row_supersedes_a_data_stand_in_at_its_address(tmp_path: Path) -> None:
    """The bytes a `literals_<fn>` row hands to a function leave the stand-in of another file that names them at that
    address: it is stale (the sync drops it), not a second owner; a stand-in elsewhere stays."""
    rows = {
        "c_func": ("main", 0x0, "func"),
        "literals_c_func": ("main", 0x20, "data"),
        "chunk": ("main", 0x20, "data"),
        "pool_float": ("main", 0x28, "data"),
        "pool_pad": ("main", 0x2C, "data"),
        "other": ("main", 0x60, "data"),
    }
    text = fixture.symbol_file("a", rows)
    sources = {
        "src/main/c.c": "s32 c_func(void) { return 1; }\n",
        "src/main/d.c": "const unsigned int chunk[2] = {\n    0x1, 0x2,\n};\nconst unsigned int pool_float[1] = {0};\n"
        "unsigned int pool_pad[1] = {0};\nconst unsigned int other[1] = {0};\n",
    }
    held = tree.capture(fixture.config(tmp_path, versions=("a",), sources=sources, symbols={"a": text}))
    data = next(m for m in held.modules if m.path == "src/main/d.c")
    # a name inside the row's size (0x10 here) is the function's too: the old pool named its float on its own
    assert data.owns["a"] == ("other",) and data.stale["a"] == ("chunk", "pool_float", "pool_pad")
    assert held.owner("a", "literals_c_func") == "src/main/c.c"


def test_a_label_typed_name_inside_a_function_needs_no_owner(tmp_path: Path) -> None:
    rows = {"c_func": ("main", 0x0, "func"), "inside_c_func": ("main", 0x10, "label"), "after": ("main", 0x30, "func")}
    sources = {"src/main/c.c": "s32 c_func(void) { return 1; }\n"}
    text = fixture.symbol_file("a", rows)
    held = tree.capture(fixture.config(tmp_path, versions=("a",), sources=sources, symbols={"a": text}))
    assert held.unowned("a") == ("after",)


def test_aliases_are_facts_with_a_rom_from_their_segment(tmp_path: Path) -> None:
    """versions/<v>/aliases.txt rows join the facts; a row without a rom takes its segment's."""
    extra = {"versions/a/aliases.txt": "other = 0x80000400;\nnowhere = 0x90000000;\n"}
    sources = {"src/main/c.c": "s32 other(void) { return 1; }\n"}
    config = fixture.config(tmp_path, versions=("a",), sources=sources, extra=extra)
    held = tree.capture(config)
    assert held.symbols["a"]["other"][1] == 0x1000 and held.symbols["a"]["nowhere"][1] is None
    assert held.owner("a", "other") == "src/main/c.c" and "c_func" not in held.unowned("a")
    assert api.symbols(config, "a")["other"] == held.symbols["a"]["other"]


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


def test_scanner_names_a_variable_of_an_anonymous_aggregate() -> None:
    text = (
        "struct { int a; int b; } pair = {1, 2};\n"
        "const union { int i; float f; } both = {1};\n"
        "struct Tag { int x; } tagged = {3};\n"
    )
    assert tree._scan(text) == (["pair", "both", "tagged"], [])


def test_scanner_names_a_variable_with_an_attribute_group() -> None:
    text = (
        "struct { short h; int w[2]; } __attribute__((packed)) blob = {0, {1, 2}};\n"
        "int x __attribute__((aligned(8))) = 1;\n"
    )
    assert tree._scan(text) == (["blob", "x"], [])


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
    assert fixture.cpp_runs(tmp_path) - before == 2  # one file, two versions, one run each


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


def test_a_c_body_supersedes_a_stand_in_in_its_own_file_or_another(tmp_path: Path) -> None:
    sources = {
        "src/main/c_func.c": '#include "include_asm.h"\ns32 c_func(void) { return 1; }\nINCLUDE_ASM(main, c_func);\n',
        "src/main/stand_in.c": '#include "include_asm.h"\ns32 stand_in(void) { return 2; }\n',
        "src/main/old.c": '#include "include_asm.h"\nINCLUDE_ASM(main, stand_in);\n',
    }
    held = tree.capture(fixture.config(tmp_path, sources=sources))
    by_path = {m.path: m for m in held.modules}
    assert held.owner("a", "stand_in") == "src/main/stand_in.c" and held.owner("a", "c_func") == "src/main/c_func.c"
    assert by_path["src/main/c_func.c"].stale == {"a": ("c_func",), "b": ("c_func",)}
    assert by_path["src/main/c_func.c"].stand_ins == {}
    assert by_path["src/main/old.c"].stale == {"a": ("stand_in",), "b": ("stand_in",)}
    assert by_path["src/main/old.c"].owns == {} and by_path["src/main/stand_in.c"].stale == {}


def test_a_header_found_through_a_project_include_flag_keys_the_file() -> None:
    files = {"src/lib/inc.c": b"int x;\n", "src/a/f.c": b'#include "inc.c"\n'}
    plain = tree._closure(files.get, "src/a", "us", files["src/a/f.c"])
    flagged = tree._closure(files.get, "src/a", "us", files["src/a/f.c"], ["-DX", "-Isrc/lib/"])
    assert plain == [("inc.c", "")] and flagged == [("inc.c", tree.sha256(b"int x;\n"))]


def test_another_tool_edit_keeps_the_parse_cache_and_a_scan_edit_rescans_without_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = fixture.config(tmp_path)
    first = tree.capture(config)
    runs = fixture.cpp_runs(tmp_path)
    monkeypatch.setattr(store, "code", lambda: "another tool build")
    monkeypatch.setattr(effort, "_memo", {})
    tree.capture(config)
    assert fixture.cpp_runs(tmp_path) == runs  # every file parse is a hit
    scans = []
    real = tree._scan_expanded
    monkeypatch.setattr(tree, "_scan_expanded", lambda *a: scans.append(1) or real(*a))
    monkeypatch.setattr(tree, "_code", lambda: "another scanner")
    monkeypatch.setattr(effort, "_memo", {})
    again = tree.capture(config)
    assert scans and fixture.cpp_runs(tmp_path) == runs  # the rows are made again from the kept expansions
    assert again == first and [m.owns for m in again.modules] == [m.owns for m in first.modules]
    monkeypatch.setattr(tree, "_expansion_code", lambda: "another preprocessor step")
    monkeypatch.setattr(effort, "_memo", {})
    tree.capture(config)
    assert fixture.cpp_runs(tmp_path) > runs


def test_the_scan_code_stops_where_the_tree_section_starts() -> None:
    """An edit below the section keeps every file's parse; so the parse functions all sit above it and the tree's
    construction below."""
    marker = Path(tree.__file__).read_bytes().partition(tree._TREE_SECTION)
    assert marker[1], "tree.py lost the line that opens its tree section"
    above = marker[0].count(b"\n") + 1
    for parse in (tree._parsed, tree._expand_both, tree._scanned, tree._scan_expanded, tree._key, tree._closure):
        assert inspect.getsourcelines(parse)[1] < above, parse.__name__
    for build in (tree._build, tree._module, tree._yield_to_source):
        assert inspect.getsourcelines(build)[1] > above, build.__name__
    assert tree._code() == tree.sha256(marker[0])


def test_versions_that_expand_alike_share_one_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scans: list[str] = []
    real = tree._scan_expanded
    monkeypatch.setattr(tree, "_scan_expanded", lambda text, *a: scans.append(text) or real(text, *a))
    held = tree.capture(fixture.config(tmp_path))
    jobs = [m for m in held.modules for _ in m.owns]
    assert scans and len(scans) < 2 * len(jobs)  # two versions per file, the same expansion for most
    assert len({s for s in scans}) == len(scans)  # no text scanned twice


def test_a_command_keeps_every_expansion_reachable_though_all_its_parses_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = fixture.config(tmp_path)
    tree.capture(config)
    cache = store.content(config)._inner
    cache._touched.clear()
    monkeypatch.setattr(store, "code", lambda: "another tool build")  # the tree is built again from hits alone
    monkeypatch.setattr(effort, "_memo", {})
    tree.capture(config)
    assert fixture.cpp_runs(tmp_path) > 0
    touched = {t for t in cache._touched if t.startswith(tree.EXPANDED + ":")}
    assert len(touched) == len(tree._jobs(config, tree._files(config, {})))
    assert any(t.startswith(tree.TEXT + ":") for t in cache._touched)  # and the texts those point to


def test_a_cold_tree_and_a_tree_made_from_kept_expansions_are_the_same_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = fixture.config(tmp_path)
    cold = tree.capture(config)
    cache = store.content(config)._inner._cache
    for name in [k for k in cache.iterkeys() if str(k).startswith((tree.KIND, tree.SCANNED))]:
        cache.delete(name)  # the rows and scans are gone, the expansions stay
    runs = fixture.cpp_runs(tmp_path)
    monkeypatch.setattr(store, "code", lambda: "another tool build")
    monkeypatch.setattr(effort, "_memo", {})
    warm = tree.capture(config)
    assert fixture.cpp_runs(tmp_path) == runs and warm == cold
    assert [(m.path, m.owns, m.fuzzy, m.stand_ins, m.sections) for m in warm.modules] == [
        (m.path, m.owns, m.fuzzy, m.stand_ins, m.sections) for m in cold.modules
    ]


def test_a_static_inline_copy_does_not_own_the_name_its_real_definition_owns() -> None:
    text = "static inline s32 f(void) { return 1; }\ns32 g(void) { return f(); }\n"
    assert tree._scan(text)[0] == ["g"]
    assert tree._scan("static s32 f(void) { return 1; }\n")[0] == ["f"]


def test_a_function_is_code_though_a_version_knows_it_only_from_an_untyped_alias(tmp_path: Path) -> None:
    sources = {"src/main/f.c": "s32 c_func(void) { return 1; }\n"}
    symbols = {"a": "c_func = 0x80000400; // type:func rom:0x1000\n", "b": "c_func = 0x80000400;\n"}
    held = tree.capture(fixture.config(tmp_path, sources=sources, symbols=symbols))
    (module,) = held.modules
    assert module.sections["b"]["c_func"] == ".text"


def test_a_later_command_rebuilds_nothing_of_an_unchanged_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = fixture.config(tmp_path)
    first = tree.capture(config)
    runs, built = fixture.cpp_runs(tmp_path), []
    real = tree._build
    monkeypatch.setattr(tree, "_build", lambda *a: built.append(1) or real(*a))
    monkeypatch.setattr(effort, "_memo", {})  # a new command: only the durable cache is left
    again = tree.capture(config)
    assert again == first and again.owner("a", "c_func") == "src/main/c_func.c"
    assert not built and fixture.cpp_runs(tmp_path) == runs


def test_a_later_command_after_one_edit_parses_only_that_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = fixture.config(tmp_path)
    tree.capture(config)
    edit(config, "src/main/c_func.c", "s32 c_func(void) { return 7; }\n")
    runs, parsed = fixture.cpp_runs(tmp_path), []
    real = tree._parsed
    monkeypatch.setattr(tree, "_parsed", lambda *a, **kw: parsed.append(a[1]) or real(*a, **kw))
    monkeypatch.setattr(effort, "_memo", {})
    tree.capture(config)
    assert set(parsed) == {"src/main/c_func.c"} and len(parsed) == 2  # one file in its two versions
    assert fixture.cpp_runs(tmp_path) - runs == 2


def test_parse_keys_are_one_function_for_a_header_read_off_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A header outside src/ and include/ is read from disk; the key that stores a parse and the key that looks it
    up must agree, or that file is parsed again by every command."""
    config = fixture.config(tmp_path)
    root = config.project.root
    (root / "asm/a/include").mkdir(parents=True)
    (root / "asm/a/include/gen.h").write_text("#define GEN 1\n")
    edit(config, "src/main/c_func.c", '#include "gen.h"\ns32 c_func(void) { return 1; }\n')
    tree.capture(config)
    runs = fixture.cpp_runs(tmp_path)
    monkeypatch.setattr(effort, "_memo", {})
    monkeypatch.setattr(store, "cached_object", lambda config, kind, key, produce: produce())  # no tree entry
    tree.capture(config)
    assert fixture.cpp_runs(tmp_path) == runs  # every file parse is a hit
    (root / "asm/a/include/gen.h").write_text("#define GEN 2\n")
    monkeypatch.setattr(effort, "_memo", {})
    tree.capture(config)
    assert fixture.cpp_runs(tmp_path) > runs  # the header is an input of the parse


def test_a_comment_edit_of_a_shared_header_parses_nothing_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A header's comment (a field note naming a function) changes no preprocessed token: every file that includes it
    keeps its parse key, so no file is preprocessed again. A code edit of the same header reparses its includers."""
    config = fixture.config(tmp_path)
    root = config.project.root
    (root / "include").mkdir(exist_ok=True)
    (root / "include/shared.h").write_text("typedef struct S {\n    int arena; /* +0x0: func_1 */\n} S;\n")
    edit(config, "src/main/c_func.c", '#include "shared.h"\ns32 c_func(void) { return 1; }\n')
    tree.capture(config)
    runs = fixture.cpp_runs(tmp_path)
    (root / "include/shared.h").write_text("typedef struct S {\n    int arena; /* +0x0: Render_StartRound */\n} S;\n")
    monkeypatch.setattr(effort, "_memo", {})
    tree.capture(config)
    assert fixture.cpp_runs(tmp_path) == runs
    (root / "include/shared.h").write_text("typedef struct S {\n    int arena;\n} S;\nextern int round;\n")
    monkeypatch.setattr(effort, "_memo", {})
    tree.capture(config)
    assert fixture.cpp_runs(tmp_path) > runs


@pytest.mark.parametrize(
    "before,after,same",
    [
        (b"int a; /* x */\n", b"int a; /* longer note */\n", True),
        (b"int a; // x\n", b"int a; // y\n", True),
        (b"/* one\n   two */\nint a;\n", b"/* three\n   four */\nint a;\n", True),
        (b"/* one */\nint a;\n", b"/* one\n */\nint a;\n", False),  # a line more: __LINE__ moves
        (b"#define X 1 /* a\n b */ + 2\n", b"#define X 1 /* c\n d */ + 2\n", False),  # inside a directive
        (b"/* a\n */ int x;\n", b"/* b\n */ int x;\n", False),  # the comment's last line goes on
        (b"int a; // x \\\nint b;\n", b"int a; // y \\\nint b;\n", False),  # the backslash continues the comment
        (b'char *s = "/* x */";\n', b'char *s = "/* y */";\n', False),  # a string is no comment
        (b"char c = '\"'; /* x */\n", b"char c = '\"'; /* y */\n", True),
        (b"int a;\n", b"int b;\n", False),
    ],
)
def test_meaning_is_the_text_less_its_comments(before: bytes, after: bytes, same: bool) -> None:
    assert (tree._meaning(before) == tree._meaning(after)) is same


def test_the_tool_does_not_lower_its_own_priority() -> None:
    import inspect

    from unbake import process

    assert "nice" not in inspect.getsource(pool).replace("niceness", "") and "nice(" not in inspect.getsource(process)


def test_a_header_is_read_and_walked_once_for_all_the_files_that_reach_it(tmp_path: Path) -> None:
    config = fixture.config(tmp_path)
    files = {"src/x/a.h": b'#include "b.h"\n', "src/x/b.h": b'#include "a.h"\n'}  # a cycle is walked once too
    reads: dict[str, int] = {}

    def read(path: str) -> bytes | None:
        reads[path] = reads.get(path, 0) + 1
        return files.get(path)

    texts = [b'#include "a.h"\nint f(void);\n'] * 50
    headers = tree._Headers(read)
    shared = [tree._key(config, f"src/x/f{i}.c", t, "a", headers) for i, t in enumerate(texts)]
    after_fifty = dict(reads)
    tree._key(config, "src/x/f50.c", texts[0], "a", headers)
    assert reads == after_fifty  # a fifty-first file asks the disk for nothing it had already
    assert max(after_fifty.values()) <= 3 and len(set(shared)) == 1
    alone = [tree._key(config, f"src/x/f{i}.c", t, "a", read) for i, t in enumerate(texts[:3])]
    assert alone == shared[:3]  # the shared walk makes the very key an unshared one makes


def test_a_preprocessor_failure_keeps_the_fatal_error_and_its_site_not_only_the_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = fixture.config(tmp_path)
    stderr = (
        b"<stdin>:2:10: fatal error: MISSING.h: No such file or directory\n #include <MISSING.h>\n\n"
        b"compilation terminated.\n"
    )
    monkeypatch.setattr(tree.process, "run", lambda *a, **k: fixture.native_result(exit=1, stderr=stderr))
    with pytest.raises(Refusal) as caught:
        tree._preprocess(config, "src/main/c_func.c", b"#include <MISSING.h>\n", "a", ())
    (finding,) = caught.value.findings
    assert "MISSING.h: No such file or directory" in finding.reason and "compilation terminated" not in finding.reason
    assert (finding.path, finding.line, finding.versions) == ("src/main/c_func.c", 2, ("a",))


def test_a_new_header_only_an_overlay_holds_is_in_the_preprocessors_search_ahead_of_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = fixture.config(tmp_path)
    seen: list[tuple[list[str], bool]] = []

    def run(phase, argv, cwd, **kw):
        flags = [a for a in argv if str(a).startswith("-I")]
        home = Path(
            next(f for f in flags if f.endswith("/include"))[2:]
            if any(f.endswith("/include") for f in flags)
            else "/none"
        )
        seen.append((flags, (home / "COMMON/NEW.H").is_file()))
        return fixture.native_result(stdout=b"int f(void) { return 0; }\n")

    monkeypatch.setattr(tree.process, "run", run)
    text = b'#include "COMMON/NEW.H"\nint f(void) { return 0; }\n'
    names = tree.parse(config, "src/main/c_func.c", text, "a", None, {"include/COMMON/NEW.H": b"typedef int t;\n"})
    assert names[0] == ["f"]
    assert seen and all(found for _, found in seen)  # the header existed while cpp ran, in both parses
    assert all(
        flags.index(next(f for f in flags if f.endswith("/include"))) < flags.index("-Iinclude") for flags, _ in seen
    )


def test_each_line_of_the_build_output_is_said_to_be_written_where_it_was(monkeypatch):
    """The output the build compiles has no line markers; its lines are read through the marked output, whose blank
    lines the build output collapses."""
    marked = (
        '# 1 "<stdin>"\n# 1 "<built-in>"\n# 1 "<stdin>"\n# 1 "include/a.h" 1\nint a;\n\n\nint b;\n'
        '# 3 "<stdin>" 2\n\nint first;\n'
    )
    plain = "int a;\nint b;\n\nint first;\n"
    outputs = {True: marked, False: plain}
    monkeypatch.setattr(tree, "preprocess", lambda config, path, version, text=None, markers=False: outputs[markers])
    sites = tree.origins(None, "src/x.c", "us")
    assert sites[0] == ("include/a.h", 1) and sites[1] == ("include/a.h", 4)
    assert sites[3] == ("src/x.c", 4)  # `int first;` is on line 4 of the file whatever the blank lines before it
    assert len(sites) == 4


FUZZY_MODULE = (
    'extern int a;\n#ifdef NON_MATCHING\n#include "sibling.h"\nint sibling(void) { return 1; }\n#else\n'
    'INCLUDE_ASM("main", sibling);\n#endif\n#ifdef NON_MATCHING\nint wanted(void) { return 2; }\n#else\n'
    'INCLUDE_ASM("main", wanted);\n#endif\n#ifdef NON_MATCHING\nint third(void) { return 3; }\n#endif\n'
)


def test_only_the_non_matching_bodies_of_the_measured_names_stay_on():
    text = tree.fuzzy_only(FUZZY_MODULE, ["wanted"])
    lines = text.splitlines()
    assert len(lines) == len(FUZZY_MODULE.splitlines())  # every diagnostic keeps its line
    assert lines[1] == "#if 0" and lines[7] == "#ifdef NON_MATCHING" and lines[12] == "#if 0"
    assert 'INCLUDE_ASM("main", sibling);' in text  # the switched-off block keeps its stand-in side
    assert tree.fuzzy_only(FUZZY_MODULE, ["wanted", "third"]).splitlines()[12] == "#ifdef NON_MATCHING"
    assert tree.fuzzy_only(FUZZY_MODULE, []).count("#if 0") == 3


def test_a_block_sharing_its_draft_with_siblings_is_off_once_the_candidate_stands_in_the_stand_ins_place():
    """The candidate replaced the stand-in of `wanted`, so the block that holds an older draft of it, with the drafts
    (and the declarations) of the other functions it shares the block with, stays off: only the candidate is live."""
    shared = (
        "int wanted(void) { return 2; }\n#ifdef NON_MATCHING\nextern int broken;\nint wanted(void) { return 1; }\n"
        "int other(void) { return 3; }\n#endif\n"
    )
    assert tree.fuzzy_only(shared, ["wanted"]).splitlines()[1] == "#if 0"
    assert tree.fuzzy_only(shared, ["other"]).splitlines()[1] == "#ifdef NON_MATCHING"  # no plain definition of it
    drafted = shared.replace("int wanted(void) { return 2; }\n", 'INCLUDE_ASM("main", wanted);\n', 1)
    assert tree.fuzzy_only(drafted, ["wanted"]).splitlines()[1] == "#ifdef NON_MATCHING"  # a stand-in defines no C


def test_a_conditional_inside_a_body_does_not_end_the_block():
    text = "#ifdef NON_MATCHING\n#if X\nint a;\n#endif\nint wanted(void) { return 1; }\n#endif\n"
    assert tree.fuzzy_only(text, ["wanted"]) == text
    assert tree.fuzzy_only(text, ["other"]).startswith("#if 0\n#if X\n")


def test_a_non_matching_body_naming_no_symbol_is_recorded_as_unnamed(tmp_path: Path) -> None:
    text = (
        '#include "include_asm.h"\n\n#ifdef NON_MATCHING\ns32 fuzzy_us(s32 a) {\n    return a;\n}\n#else\n'
        "INCLUDE_ASM(ovl, fuzzy);\n#endif\n"
    )
    held = tree.capture(fixture.config(tmp_path, sources={"src/ovl/fuzzy.c": text}))
    module = {m.path: m for m in held.modules}["src/ovl/fuzzy.c"]
    assert module.unnamed == {"a": ("fuzzy_us",), "b": ("fuzzy_us",)} and module.fuzzy == ()
    assert not any(m.unnamed for m in held.modules if m is not module)


def test_a_non_matching_body_a_symbol_of_another_version_names_is_not_unnamed(tmp_path: Path) -> None:
    text = (
        '#include "include_asm.h"\n\n#ifdef NON_MATCHING\ns32 fuzzy(s32 a) {\n    return a;\n}\n#else\n'
        "INCLUDE_ASM(ovl, fuzzy);\n#endif\n"
    )
    held = tree.capture(fixture.config(tmp_path, sources={"src/ovl/fuzzy.c": text}))
    assert not any(m.unnamed for m in held.modules)


def test_a_static_helper_in_a_non_matching_block_is_the_files_own_and_not_unnamed(tmp_path: Path) -> None:
    text = (
        '#include "include_asm.h"\n\n#ifdef NON_MATCHING\nstatic s32 helper(s32 a) {\n    return a;\n}\n\n'
        "s32 fuzzy_us(s32 a) {\n    return helper(a);\n}\n#else\nINCLUDE_ASM(ovl, fuzzy);\n#endif\n"
    )
    held = tree.capture(fixture.config(tmp_path, sources={"src/ovl/fuzzy.c": text}))
    module = {m.path: m for m in held.modules}["src/ovl/fuzzy.c"]
    assert module.unnamed == {"a": ("fuzzy_us",), "b": ("fuzzy_us",)}  # helper is not named; the extern stray still is
    assert tree._scan("static s32 f(void) { return 1; }\n", kinds := {})[0] == ["f"] and kinds == {"f": tree.LOCAL_TEXT}
    assert tree._section(("x", 0, 0, "data"), tree.LOCAL_TEXT) == ".text"


def _fuzzy_runs(config: Config, monkeypatch: pytest.MonkeyPatch, opened: list[str], path: str) -> tuple[int, int]:
    """(second expansions made, scans made) parsing one file of the fixture when the preprocessor opened `opened`."""
    real = tree._expand
    seconds, scans = [], []
    monkeypatch.setattr(tree, "_expand", lambda *a, **k: (real(*a, **k)[0], opened))
    real_pre, real_scan = tree._preprocess, tree._scan_expanded
    monkeypatch.setattr(tree, "_preprocess", lambda *a, **k: seconds.append(1) or real_pre(*a, **k))
    monkeypatch.setattr(tree, "_scan_expanded", lambda *a, **k: scans.append(1) or real_scan(*a, **k))
    name = f"k-{path}-{opened}"
    tree._parsed(config, path, (config.project.root / path).read_bytes(), "a", key=name, inputs=name)
    return len(seconds), len(scans)


def test_a_file_nothing_of_which_can_tell_the_macro_is_expanded_and_scanned_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _fuzzy_runs(fixture.config(tmp_path), monkeypatch, [], "src/main/c_func.c") == (0, 1)


def test_a_header_that_holds_the_word_brings_the_second_expansion_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = fixture.config(tmp_path)
    (config.project.root / "include" / "plain.h").write_text("int plain;\n")
    (config.project.root / "include" / "fuzzy.h").write_text("#ifdef NON_MATCHING\nint x;\n#endif\n")
    assert _fuzzy_runs(config, monkeypatch, ["include/plain.h"], "src/main/c_func.c") == (0, 1)
    # the header holds the word but the expansion does not change: the second expansion is made, no scan is (the first
    # call's scan of the same text is kept)
    assert _fuzzy_runs(config, monkeypatch, ["include/plain.h", "include/fuzzy.h"], "src/main/c_func.c") == (1, 0)
    assert _fuzzy_runs(config, monkeypatch, ["include/missing.h"], "src/main/c_func.c") == (1, 0)


def test_a_file_holding_the_word_is_expanded_both_ways_and_scanned_for_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _fuzzy_runs(fixture.config(tmp_path), monkeypatch, [], "src/ovl/fuzzy.c") == (1, 2)


def test_a_compound_guard_is_a_block_but_its_sibling_body_stays_on():
    text = (
        "#if defined(A) && defined(NON_MATCHING)\nint other(void) { return 1; }\n"
        '#else\nINCLUDE_ASM("m", other);\n#endif\n'
    )
    assert tree.non_matching_blocks(text.splitlines(keepends=True)) == [(0, 2, 4)]
    assert tree.fuzzy_only(text, ["wanted"]) == text
    assert tree.fuzzy_only(text.replace("defined(A) && ", ""), ["wanted"]).startswith("#if 0\n")


def test_the_expanded_scan_reads_runs_once_and_gives_the_whole_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preprocessed text read in runs of lines with the runs remembered gives `_scan` exactly, also where a run ends
    inside a statement or a body; the same header text met again is not folded again."""
    header = "".join(f"extern s32 h{i}(s32 a,\n    s32 b);\nstruct S{i} {{\n    s32 x;\n}} s{i};\n" for i in range(200))
    unit = (
        header + "s32 f(void) {\n    return 0;\n}\nconst s32 table[] = {\n 1, 2 };\nstatic s32 g(void) { return 1; }\n"
    )
    unit += 'asm(".include \\"nonmatchings/main/h.s\\"");\n'
    monkeypatch.setattr(tree, "_CHUNKS", {})
    folds = []
    real = tree._fold
    monkeypatch.setattr(tree, "_fold", lambda *a: folds.append(1) or real(*a))
    whole_kinds: dict[str, str] = {}
    whole = tree._scan(unit, whole_kinds)
    folds.clear()
    kinds: dict[str, str] = {}
    assert tree._scan_expanded(unit, kinds) == whole and kinds == whole_kinds
    first = len(folds)
    other: dict[str, str] = {}
    assert tree._scan_expanded(header + "s32 k(void) { return 2; }\n", other)[0][-1] == "k"
    assert len(folds) - first < first / 4  # the header's runs were read before


def test_skeleton_drops_only_declarations_that_cannot_define_a_name() -> None:
    header = (
        b"#define M 1\n"
        b"typedef struct P { int a; } P;\nstruct Q { int b; };\nenum E { E1 };\n"
        b"static __inline__ int pick(P *p) { return p->a; }\n"
        b"extern int ext;\nint table[4] = {1, 2};\nint fn(void) { return 1; }\n"
        b"#if M\nstruct Split {\n#endif\n int c;\n};\n"
    )
    kept = tree._skeleton(header)
    assert b"typedef" not in kept and b"enum" not in kept and b"pick" not in kept
    assert b"#define M 1" in kept and b"extern int ext ;" in kept and b"table" in kept and b"fn" in kept
    assert b"struct Split {" in kept  # a run cut by a conditional is kept as written
    grown = header.replace(b"struct Q { int b; };", b"struct Q { int b; int d; };\ntypedef long L;")
    assert tree._skeleton(grown) == kept


def test_a_header_edit_of_unnamed_types_parses_nothing_again(tmp_path: Path) -> None:
    sources = {"src/main/c_func.c": '#include "t.h"\ns32 c_func(void) { return 1; }\n'}
    config = fixture.config(tmp_path, sources=sources, extra={"include/t.h": "typedef int T;\n"})
    tree.capture(config)
    before = fixture.cpp_runs(tmp_path)
    edit(config, "include/t.h", "typedef int T;\nstruct Fresh { int q; };\n")
    held = tree.capture(config)
    assert fixture.cpp_runs(tmp_path) == before and held.owner("a", "c_func") == "src/main/c_func.c"
    edit(config, "include/t.h", "typedef int T;\nstruct Fresh { int q; };\nint g(void) { return 2; }\n")
    tree.capture(config)
    assert fixture.cpp_runs(tmp_path) > before  # a function in the header can define a name: the source parses again


def test_an_address_named_file_binds_its_unlisted_definition_to_the_fact_at_that_address(tmp_path: Path) -> None:
    """DATA_<ADDR> defines a renamed table no fact lists: it owns the stand-in fact at its address, so sync keeps the
    piece (it dropped the fact from splat.yaml and the link script, 95901 bytes of us-rev1 fell out of the ROM)."""
    rows = {"c_func": ("main", 0x0, "func"), "D_auto_data_unresolved_80000420": ("main", 0x20, "data")}
    text = fixture.symbol_file("a", rows)
    sources = {
        "src/main/c.c": "s32 c_func(void) { return 1; }\n",
        "src/main/DATA_80000420.CPP": "unsigned short g_renamed[2] = {0, 1};\n",
        "src/main/other.c": "unsigned short g_other[2] = {0, 1};\n",
    }
    held = tree.capture(fixture.config(tmp_path, versions=("a",), sources=sources, symbols={"a": text}))
    assert held.owner("a", "D_auto_data_unresolved_80000420") == "src/main/DATA_80000420.CPP"
    assert "D_auto_data_unresolved_80000420" not in held.unowned("a")
