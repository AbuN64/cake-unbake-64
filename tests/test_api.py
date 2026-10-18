"""api: the one surface Recipe Maker imports."""

from __future__ import annotations

from pathlib import Path

import fixture
import pytest

from unbake import api, pool, process, store, tree
from unbake.contracts import Refusal

pytestmark = pytest.mark.usefixtures("toolchains")  # the parse runs the toolchain row's preprocess template

EXPECTED = [
    "Config",
    "Finding",
    "Json",
    "Module",
    "Refusal",
    "Tree",
    "asm_path",
    "bind",
    "build_context",
    "cached",
    "capture",
    "closed",
    "command",
    "load",
    "memo",
    "pool_map",
    "preprocess",
    "score",
    "snapshot",
    "stage",
    "summary",
    "symbols",
]


def test_all_is_exactly_the_surface() -> None:
    assert sorted(api.__all__) == EXPECTED
    assert all(hasattr(api, name) for name in EXPECTED)
    assert api.Tree is tree.Tree and api.Module is tree.Module and api.Refusal is Refusal


def test_symbols_and_asm_path(tmp_path: Path) -> None:
    config = fixture.config(tmp_path)
    assert api.symbols(config, "a")["c_func"] == (0x80000400, 0x1000, 0x10, "func")
    assert api.asm_path(config, "b", "fuzzy") == "asm/b/nonmatchings/ovl/fuzzy.s"


def test_cached_memo_and_pool_map_are_the_shared_ones(tmp_path: Path) -> None:
    config = fixture.config(tmp_path)
    runs = []
    produce = lambda: runs.append(1) or b"v"  # noqa: E731
    assert api.cached(config, "tmp", "k", produce) == b"v" == store.cached(config, "tmp", "k", produce)
    assert runs == [1]
    count = []
    assert api.memo("key", lambda: count.append(1) or 5) == 5 == api.memo("key", lambda: 6) and count == [1]
    assert api.pool_map(config, "api.test", str, [1, 2]) == pool.map(config, "api.test", str, [1, 2]) == ["1", "2"]


def test_score_reads_report_lazily(tmp_path: Path, monkeypatch) -> None:
    from unbake import report

    monkeypatch.setattr(report, "scores", lambda config, held: {"fuzzy": {"a": 0.5}}, raising=False)
    assert api.score(fixture.config(tmp_path), "fuzzy") == {"a": 0.5}


def test_stage_timing_is_the_shared_one(tmp_path: Path) -> None:
    """Recipe prints its --stages from these: a stage opened through the api closes into the same record list."""
    config = fixture.config(tmp_path)
    api.bind(config)
    with api.command("recipe test", ["x"]), api.stage("recipe.work"):
        pass
    records = api.closed()
    assert [r.span for r in records if r.kind == "stage"] and any(r.kind == "command" for r in records)
    assert any("recipe.work" in line for line in api.summary(records))


def _argvs(monkeypatch) -> list[list[str]]:
    """The argv of every cpp run, the real runs still made."""
    seen: list[list[str]] = []
    real = process.run

    def spy(name, argv, *args, **kw):
        seen.append([str(a) for a in argv])
        return real(name, argv, *args, **kw)

    monkeypatch.setattr(process, "run", spy)
    return seen


def test_preprocess_is_the_makefile_step_of_the_file_on_disk(tmp_path: Path, monkeypatch, toolchains) -> None:
    config = fixture.config(tmp_path)
    seen = _argvs(monkeypatch)
    text = api.preprocess(config, "src/ovl/fuzzy.c", "a")
    assert '.include \\"nonmatchings/ovl/fuzzy.s\\"' in text and "return a + 1" not in text
    cpp = str(config.host.tools["cpp"])
    # the toolchain row's template: {cpp} -E {cppflags} {defines} {includes} {source}, as the Makefile renders it
    assert seen[-1] == [cpp, "-E", "-DGAME", "-DVERSION_A", "-Iinclude", "-Isrc/ovl/", "src/ovl/fuzzy.c"]


def test_preprocess_of_text_and_a_file_rows_toolchain_and_flags(tmp_path: Path, monkeypatch, toolchains) -> None:
    from unbake.contracts import FileFacts

    toolchains["toolchain"]["gcc-p"] = {**toolchains["toolchain"]["gcc-test"], "cppflags": ["-P", "-DROW"]}
    config = fixture.config(tmp_path)
    config.project.files = {"src/ovl/fuzzy.c": FileFacts(toolchain="gcc-p", cflags=("-DFILE",))}
    seen = _argvs(monkeypatch)
    assert api.preprocess(config, "src/ovl/fuzzy.c", "b", b"#ifdef FILE\nyes\n#endif\n").strip() == "yes"
    assert seen[-1][1:] == ["-E", "-P", "-DROW", "-DGAME", "-DFILE", "-DVERSION_B", "-Iinclude", "-Isrc/ovl/", "-"]
    api.preprocess(config, "src/ovl/fuzzy.c", "b", markers=True)
    assert "-P" not in seen[-1] and "-DROW" in seen[-1]  # markers drop only the row's -P


def test_the_tree_parses_what_the_build_preprocesses_plus_p(tmp_path: Path, monkeypatch, toolchains) -> None:
    config = fixture.config(tmp_path)
    seen = _argvs(monkeypatch)
    tree.parse(config, "src/ovl/fuzzy.c", (config.project.root / "src/ovl/fuzzy.c").read_bytes(), "a")
    runs = [a for a in seen if a[-1] == "-"]
    assert runs and all(a[:4] == [str(config.host.tools["cpp"]), "-E", "-DGAME", "-P"] for a in runs)


def test_build_context_holds_what_the_permuter_needs(tmp_path: Path, toolchains) -> None:
    config = fixture.config(tmp_path)
    asm = config.project.root / "asm/a/nonmatchings/ovl/fuzzy.s"
    asm.parent.mkdir(parents=True, exist_ok=True)
    asm.write_text("glabel fuzzy\n")
    found = api.build_context(config, "fuzzy", "a")
    root = config.project.root
    assert found["path"] == "src/ovl/fuzzy.c" and found["toolchain"] == "gcc-test"
    assert found["source"] == api.preprocess(config, "src/ovl/fuzzy.c", "a")
    assert found["asm_path"] == "asm/a/nonmatchings/ovl/fuzzy.s" and found["asm"] == "glabel fuzzy\n"
    assert found["cc"][1:] == ["-O2", "-c", "{in}", "-o", "{out}"]  # the row's compile template over {in}, {out}
    assert found["cflags"] == ["-O2"] and found["cppflags"] == ["-DGAME"]
    assert found["script"].startswith("#!/bin/sh") and '"$1"' in found["script"] and '"$3"' in found["script"]
    assert found["assemble"][0] == str(config.host.tools["mips_as"])
    assert found["assemble"][-3:] == [str(root / "asm/a/nonmatchings/ovl/fuzzy.s"), "-o", "{out}"]
    assert f"-I{root / 'asm/a/include'}" in found["assemble"]
    candidate = api.build_context(config, "fuzzy", "a", b"#ifdef GAME\nint fuzzy(void) { return 2; }\n#endif\n")
    assert candidate["source"].strip() == "int fuzzy(void) { return 2; }"  # a candidate preprocessed as the module


def test_build_context_refuses_a_name_no_module_owns(tmp_path: Path, toolchains) -> None:
    with pytest.raises(Refusal, match="no module owns"):
        api.build_context(fixture.config(tmp_path), "nobody", "a")
