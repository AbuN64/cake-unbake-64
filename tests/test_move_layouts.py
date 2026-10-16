"""Type-layout preparation: work counts and ordered edits, without timing assertions."""

from types import SimpleNamespace
from typing import Any

import pytest
from pycparser import CParser

from unbake import effort, headers, move
from unbake.contracts import Move


def catalogue(monkeypatch: pytest.MonkeyPatch, records: dict[str, str], aliases: dict[str, str] | None = None) -> Any:
    monkeypatch.setattr(headers, "type_names", lambda snapshot: (dict(records), dict(aliases or {})))
    monkeypatch.setattr(headers, "source_types", lambda snapshot, name: ({}, {}))
    return move._Types(SimpleNamespace(config=object()))


def parse_count(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    original = CParser.parse

    def parse(self: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(text)
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(CParser, "parse", parse)
    return calls


def isolated_pool(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Workers do not populate the parent's memo: the dispatch must return and install its answers."""
    calls: list[tuple[str, int]] = []

    def mapped(config: Any, name: str, function: Any, items: Any) -> list[Any]:
        calls.append((name, len(items)))
        results = []
        for item in items:
            with monkeypatch.context() as worker:
                worker.setattr(effort, "_memo", {})
                results.append(function(item))
        return results

    monkeypatch.setattr(move.pool, "map", mapped)
    return calls


def test_same_parser_text_reuses_layout_when_a_pointer_body_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    types = catalogue(monkeypatch, {"Child": "struct Child { int value; };"})
    calls = parse_count(monkeypatch)
    canonical = []
    original = headers.canonical

    def counted(*args: Any) -> str:
        canonical.append(None)
        return original(*args)

    monkeypatch.setattr(headers, "canonical", counted)
    rows = {"struct Root": "struct Root { struct Child *child; };"}
    first = move._probe_layouts(types, rows)
    types.records["Child"] = "struct Child { double value; };"
    second = move._probe_layouts(types, rows)
    assert first == second and first["struct Root"]["size"] == 4
    assert len(calls) == 1 and len(canonical) == 1


def test_body_typedef_changes_invalidate_the_probe_but_reuse_its_incomplete_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    types = catalogue(monkeypatch, {"Child": "struct Child { Elem value; };"}, {"Elem": "typedef int Elem;"})
    calls = parse_count(monkeypatch)
    rows = {"struct Root": "struct Root { struct Child child; };"}
    first = move._probe_layouts(types, rows)
    types.typedefs["Elem"] = "typedef double Elem;"
    second = move._probe_layouts(types, rows)
    assert first["struct Root"]["size"] == 4 and second["struct Root"]["size"] == 8
    assert len(calls) == 3  # one shared incomplete text, then each distinct complete text


def test_requested_record_names_are_part_of_the_parsed_layout_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = parse_count(monkeypatch)
    text = "struct A { int a; }; struct B { double b; };"
    a = move._parsed_layouts(text, ("struct A",))
    b = move._parsed_layouts(text, ("struct B",))
    assert a["struct A"]["size"] == 4 and b["struct B"]["size"] == 8
    assert len(calls) == 2


def test_parse_failures_are_memoized_by_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = parse_count(monkeypatch)
    first = move._parsed_layouts("struct A { @ };", ("struct A",))
    assert isinstance(first, str) and "do not parse" in first
    assert move._parsed_layouts("struct A { @ };", ("struct A",)) == first and len(calls) == 1


def test_independent_probes_dispatch_once_and_parent_reuses_each_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    types = catalogue(
        monkeypatch,
        {
            "Old": "struct Old { int a; };",
            "Real": "struct Real { int b; };",
            "Other": "struct Other { short c; };",
            "OtherReal": "struct OtherReal { short d; };",
        },
    )
    calls = parse_count(monkeypatch)
    dispatches = isolated_pool(monkeypatch)
    moves = [Move(0, "type", "Old", "Real"), Move(1, "type", "Other", "OtherReal")]
    move._warm_layouts(types, moves, ())
    assert dispatches == [("move.type_layouts", 2)] and len(calls) == 2
    for m in moves:
        rows = move._view_rows(types, m.subject, m.target, m.subject, m.target)
        assert rows is not None and isinstance(move._probe_layouts(types, rows), dict)
    assert len(calls) == 2
    types.records["Real"] = "struct Real { double b; };"
    rows = move._view_rows(types, "Old", "Real", "Old", "Real")
    assert move._probe_layouts(types, rows)["struct Real"]["size"] == 8
    assert len(calls) == 3


def test_duplicate_probe_texts_need_one_parse_and_no_parallel_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    types = catalogue(
        monkeypatch,
        {"Old": "struct Old { int a; };", "Other": "struct Other { int a; };", "Real": "struct Real { int b; };"},
    )
    calls = parse_count(monkeypatch)
    dispatches = isolated_pool(monkeypatch)
    moves = [Move(0, "type", "Old", "Real"), Move(1, "type", "Other", "Real")]
    move._warm_layouts(types, moves, ())
    assert dispatches == []
    for m in moves:
        rows = move._view_rows(types, m.subject, m.target, m.subject, m.target)
        assert rows is not None and isinstance(move._probe_layouts(types, rows), dict)
    assert len(calls) == 1


def test_a_prepared_fold_observes_the_previous_folds_member_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    records = {
        "Old": "struct Old { int a; };",
        "Other": "struct Other { int b; };",
        "Real": "struct Real { int unk0; };",
    }
    types = catalogue(monkeypatch, records)
    dispatches = isolated_pool(monkeypatch)
    calls = parse_count(monkeypatch)
    texts = {"Old": b"int f(Old *p) { return p->a; }", "Other": b"int g(Other *p) { return p->b; }"}
    types.snapshot.read = lambda path: texts[path]
    monkeypatch.setattr(headers, "spelled_in", lambda snapshot, name: ([name] if name in texts else [], []))
    moves = [Move(0, "type", "Old", "Real"), Move(1, "type", "Other", "Real")]
    move._warm_layouts(types, moves, ())
    assert dispatches == [("move.type_layouts", 2)]
    for m in moves:
        assert move._view_fold(types, m.subject, m.target, m.subject, m.target) is None
    assert "int a;" in types.records["Real"] and "unk0" not in types.records["Real"]
    assert (("Other",), {"b": "a"}, frozenset({"Other"})) in types.accesses
    assert len(calls) == 3  # the second fold must lay the changed target, not use its speculative answer


def test_single_move_does_not_launch_speculative_work(monkeypatch: pytest.MonkeyPatch) -> None:
    types = catalogue(monkeypatch, {"Old": "struct Old { int a; };", "Real": "struct Real { int b; };"})
    calls = parse_count(monkeypatch)
    dispatches = isolated_pool(monkeypatch)
    move._warm_layouts(types, [Move(0, "type", "Old", "Real")], ())
    assert calls == [] and dispatches == []
