"""The what-if verb: the change parsing, the candidate universe, the search and the gdb run, with gdb and the
compiler mocked (the real cc1 run is the end-to-end proof, not a unit test)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from unbake import config, whatif
from unbake.contracts import Proof, Refusal

KINDS = {"anti": 14, "output": 15, "true": 0}


def _proof(score: float, sha: str = "x", exact: bool = False) -> Proof:
    return SimpleNamespace(score=score, built_sha256=sha, exact=exact)  # type: ignore[return-value]


def test_parse_deps_reads_a_gt_b_with_an_optional_kind() -> None:
    assert whatif.parse_deps(["41>31", "5>6:output"], KINDS) == [(41, 31, "anti"), (5, 6, "output")]


@pytest.mark.parametrize("bad", ["41-31", "a>b", "1>2:nope", "1>"])
def test_parse_deps_refuses_a_malformed_or_unknown_kind(bad: str) -> None:
    with pytest.raises(Refusal):
        whatif.parse_deps([bad], KINDS)


def test_the_marks_a_gdb_run_printed_are_grouped_by_word() -> None:
    out = b"noise\nWHATIF dep 41 31\nWHATIF dep 1 2\nWHATIF exit 0\n"
    assert whatif._marks(out) == {"dep": ["41 31", "1 2"], "exit": ["0"]}


def _pass() -> Any:
    block = {u: 0 for u in (10, 11, 12, 13)} | {99: 1}
    return SimpleNamespace(block=block, luid={10: 0, 11: 1, 12: 2, 13: 3, 99: 4}, deps={12: [(11, "true")]})


def test_universe_lists_backward_pairs_nearest_first_without_held_dependences() -> None:
    pairs = whatif.universe(_pass(), 0, "deps", 2, [])
    assert pairs == [(11, 10), (13, 12), (12, 10), (13, 11)]  # 12 on 11 is held; 99 is another block
    assert (10, 11) in whatif.universe(_pass(), 0, "deps", 2, [], anywhere=True)
    assert whatif.universe(_pass(), 0, "deps", 9, [12, 13]) == [(13, 12)]


def test_universe_of_top_priority_is_the_block_insns() -> None:
    assert whatif.universe(_pass(), 0, "top-priority", 8, []) == [10, 11, 12, 13]


def _many(run: Any) -> Any:
    return lambda changes: [run(c) for c in changes]


def _evaluate(good: set[tuple[Any, ...]]) -> Any:
    def run(change: tuple[Any, ...]) -> whatif.Outcome:
        hit = {c for c in change}
        score = 0.5 + 0.1 * len(hit & set().union(*good))
        exact = hit >= set().union(*good)
        return whatif.Outcome(
            _proof(1.0 if exact else score, sha="same" if not hit & set().union(*good) else "moved", exact=exact),
            cpu=0.5,
        )

    return run


def test_search_pairs_finds_the_smallest_exact_pair() -> None:
    found = whatif.search(_many(_evaluate({(1, 2)})), [1, 2, 3, 4], "pairs", 4, 2, 100, _proof(0.5, "same"))
    assert [sorted(s) for s in found["exact_sets"]] == [[1, 2]]
    assert found["exhausted"] is True and found["tried"] == 4 + 6


def test_search_stops_at_the_first_exact_size() -> None:
    found = whatif.search(_many(_evaluate({(3,)})), [1, 2, 3], "pairs", 4, 1, 100, _proof(0.5, "same"))
    assert [list(s) for s in found["exact_sets"]] == [[3]] and "2" not in found["sizes"]


def test_search_reports_impossible_inputs_as_exhausted_with_the_best_score() -> None:
    def never(change: tuple[Any, ...]) -> whatif.Outcome:
        return whatif.Outcome(_proof(0.6 + 0.01 * change[0], "m"), cpu=0.1)

    found = whatif.search(_many(never), [1, 2, 3], "pairs", 3, 1, 100, _proof(0.5, "b"))
    assert found["exact_sets"] == [] and found["exhausted"] is True and found["best"]["score"] == 0.6 + 0.01 * 3


def test_search_marks_itself_incomplete_when_the_cpu_budget_runs_out() -> None:
    found = whatif.search(_many(_evaluate({(9,)})), list(range(50)), "singles", 3, 1, 1.0, _proof(0.5, "same"))
    assert found["exhausted"] is False and found["tried"] < 50


def test_greedy_grows_a_set_then_drops_the_members_it_can_do_without() -> None:
    need = {2, 5}

    def run(change: tuple[Any, ...]) -> whatif.Outcome:
        have = set(change)
        exact = need <= have
        return whatif.Outcome(_proof(1.0 if exact else 0.5 + 0.1 * len(have & need), "m", exact), cpu=0.1)

    found = whatif.search(_many(run), [1, 2, 3, 5, 7], "greedy", 6, 2, 100, _proof(0.5, "b"))
    assert [sorted(s) for s in found["exact_sets"]] == [[2, 5]]


def test_the_gdb_programs_name_the_function_and_compile_as_python() -> None:
    compile(whatif._DEPS_PROGRAM, "program.py", "exec")
    script = whatif._TOP_PROGRAM.format(
        adjust="adjust_priority", birthing="birthing_insn_p", function="F", uids="$u == 3"
    )
    assert '"F"' in script and "return (int) 0" in script and "break adjust_priority" in script


def _lab(tmp_path: Path, steps: list[Any], kinds: dict[str, int] | None = KINDS) -> whatif.Lab:
    lab = object.__new__(whatif.Lab)
    lab.cfg = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    lab.version, lab.name, lab.block, lab.work = "v", "F", 0, tmp_path / "runs"
    lab.kinds, lab.gdb, lab.input, lab.steps = dict(kinds or {}), "/usr/bin/gdb", "int x;", steps
    lab.snapshot = lab.unit = lab.recipe = lab.facts = None
    lab.text = "t"
    import itertools

    lab._count = itertools.count()
    lab.work.mkdir(parents=True)
    return lab


def _run_with(monkeypatch: pytest.MonkeyPatch, stdout: bytes, exit: int = 0) -> list[list[str]]:
    seen: list[list[str]] = []

    def fake(name: str, argv: Any, cwd: Path, **kw: Any) -> Any:
        seen.append([str(a) for a in argv])
        return SimpleNamespace(
            exit=exit, stdout=stdout if name.endswith("compile") else b"", stderr=b"boom", cpu_seconds=0.25
        )

    monkeypatch.setattr(whatif.process, "run", fake)
    monkeypatch.setattr(whatif.process, "scratch", lambda root: root)
    monkeypatch.setattr(whatif.native, "link_object", lambda *a, **k: _proof(0.9, "built"))
    return seen


def _steps() -> list[Any]:
    return [
        SimpleNamespace(phase="compile", out="source.s", argv=("cc1", "-O2", "{in}", "-o", "{out}")),
        SimpleNamespace(phase="assemble", out="source.o", argv=("as", "{in}", "-o", "{out}")),
    ]


def test_a_change_runs_the_compile_under_gdb_and_reports_what_the_compiler_took(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _run_with(monkeypatch, b"WHATIF dep 41 31\nWHATIF exit 0\n")
    out = _lab(tmp_path, _steps()).run([(41, 31, "anti"), (7, 8, "anti")])
    assert seen[0][:4] == ["/usr/bin/gdb", "-q", "-batch", "-x"] and seen[0][-6:-3] == ["--args", "cc1", "-O2"]
    assert out.taken == ("41>31",) and out.missing == ("7>8",) and out.cpu == 0.5 and out.proof.score == 0.9  # type: ignore[union-attr]


def test_no_change_runs_the_plain_chain_without_gdb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _run_with(monkeypatch, b"")
    out = _lab(tmp_path, _steps()).run()
    assert seen[0][0] == "cc1" and seen[1][0] == "as" and out.taken == ()


def test_a_compile_the_changes_break_is_an_outcome_without_a_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_with(monkeypatch, b"WHATIF dep 1 2\nWHATIF exit 1\n")
    out = _lab(tmp_path, _steps()).run([(1, 2, "anti")])
    assert out.proof is None and out.why == "boom"


def test_an_error_the_gdb_program_names_is_a_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_with(monkeypatch, b"WHATIF error no symbol priority in the compiler\n", exit=3)
    with pytest.raises(Refusal) as refused:
        _lab(tmp_path, _steps()).run([(1, 2, "anti")])
    assert "no symbol priority" in refused.value.findings[0].reason


def test_a_toolchain_without_a_whatif_row_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(whatif.native, "candidate", lambda *a, **k: (None, None, SimpleNamespace(toolchain="t"), []))
    monkeypatch.setattr(whatif.config, "toolchain", lambda name: {})
    with pytest.raises(Refusal) as refused:
        whatif.Lab(SimpleNamespace(), "v", "F", "p", "t", {}, {}, 0, Path("."))
    assert "no whatif row" in refused.value.findings[0].reason


def test_the_toolchain_table_describes_the_gcc_281_scheduler() -> None:
    assert config.toolchain("gcc-2.8.1-sn64")["whatif"]["kinds"]["anti"] == 14


def test_the_toolchain_table_describes_the_gcc_272_schedulers() -> None:
    for name in ("gcc-2.7.2-sn64", "gcc-2.7.2-kmc"):
        assert config.toolchain(name)["whatif"]["kinds"] == {"anti": 14, "output": 15, "true": 0}


def test_a_cc1_without_the_adjust_priority_symbol_refuses_top_priority_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    lab = whatif.Lab.__new__(whatif.Lab)
    lab.gdb, lab.steps, lab.work = "gdb", [SimpleNamespace(argv=["cc1"])], Path(".")
    lab.recipe, lab.cfg = SimpleNamespace(toolchain="t"), SimpleNamespace(project=SimpleNamespace(root=Path(".")))
    monkeypatch.setattr(whatif.process, "scratch", lambda root: Path("."))
    said = SimpleNamespace(stdout=b'No symbol "adjust_priority" in current context.', stderr=b"")
    monkeypatch.setattr(whatif.process, "run", lambda *a, **k: said)
    with pytest.raises(Refusal) as refused:
        lab._need_symbols(("adjust_priority",))
    assert (
        "no symbol adjust_priority" in refused.value.findings[0].reason
        and "--kind deps" in refused.value.findings[0].action
    )
