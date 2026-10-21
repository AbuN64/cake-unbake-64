"""A warm command reads, resolves and parses each thing once: the tests count work, never time."""

from __future__ import annotations

import os
import tomllib
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from test_native import _outputs, _tools, case, invocation  # noqa: F401  (fixtures)

from unbake import config as configuration
from unbake import effort, native, pool, process, store, view
from unbake.contracts import digest


def test_digest_keeps_its_canonical_text() -> None:
    assert digest({"b": (1, 2), "a": [b"x"]}) == digest({"a": [b"x"], "b": [1, 2]})
    assert digest({"s": {3, 1, 2}}) == digest({"s": [1, 2, 3]})


def test_a_header_shared_by_two_units_is_read_and_parsed_once(tmp_path, monkeypatch) -> None:
    (tmp_path / "include").mkdir()
    (tmp_path / "include/shared.h").write_text("int x;\n")
    for name in ("a.c", "b.c"):
        (tmp_path / name).write_text(f'#include "shared.h"\nint {name[0]};\n')
    snapshot = Mock(
        digest="s",
        tree=False,
        overlays={},
        config=Mock(project=Mock(root=tmp_path)),
        peek=lambda p: (tmp_path / p).read_bytes() if (tmp_path / p).is_file() else None,
    )
    parsed = []
    real = view._DIRECTIVE.findall
    monkeypatch.setattr(view, "_DIRECTIVE", Mock(findall=lambda data: parsed.append(data) or real(data)))
    first, second = (view.headers(snapshot, Mock(path=n), ["-Iinclude"])[0] for n in ("a.c", "b.c"))
    assert first == second == ("include/shared.h",)
    assert len(parsed) == 3  # a.c, b.c and shared.h, each once


def test_a_pin_tracks_external_replacements_as_well_as_store_writes(cfg) -> None:
    path = cfg.project.root / "include/p.h"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"one")
    snapshot = Mock(overlays={}, tree=False, config=cfg)
    before = view.pin(snapshot, "include/p.h")
    path.write_bytes(b"two")
    assert view.pin(snapshot, "include/p.h") != before  # external writers invalidate the content pin
    store.write(path, b"three")
    assert view.pin(snapshot, "include/p.h") != before


def test_a_validated_table_is_parsed_once_across_commands(cfg, monkeypatch) -> None:
    parses = []
    real = tomllib.loads
    monkeypatch.setattr(configuration.tomllib, "loads", lambda text: parses.append(1) or real(text))
    data = b"schema = 1\n\n[symbol]\n"
    for _ in range(2):
        effort._memo.clear()
        configuration.toml("symbols", data, "symbols.toml", "symbols.table", store.content(cfg).cached)
    assert len(parses) == 1


def test_a_group_warm_item_by_item_is_sealed_for_the_next_run(cfg, monkeypatch) -> None:
    items = [1, 2, 3]
    for index in (str, str):
        pool.map(cfg, "units", abs, items, key=index)
    reads = []
    real = pool.store.get
    monkeypatch.setattr(pool.store, "get", lambda c, kind, key, **kw: reads.append(kind) or real(c, kind, key, **kw))
    pool.map(cfg, "units", abs, items, key=str)
    assert reads == ["units.group"]


def test_a_listing_is_read_once_until_a_stream_is_appended(cfg, monkeypatch) -> None:
    (cfg.project.root / ".unbake/attempts").mkdir(parents=True)
    (cfg.project.root / ".unbake/attempts/m.jsonl").write_text("")
    listed = []
    real = os.listdir
    monkeypatch.setattr(os, "listdir", lambda p: listed.append(p) or real(p))
    assert "m.jsonl" in store.listing(cfg, "attempts") and "m.jsonl" in store.listing(cfg, "attempts")
    assert len(listed) == 1
    store.append(cfg, "attempts/n", {"a": 1})
    assert "n.jsonl" in store.listing(cfg, "attempts") and len(listed) == 2


def _text(monkeypatch, texts):
    monkeypatch.setattr(native.view, "get", lambda snap, u, *a, **k: Mock(text=texts[u.path], unit=u.path))


def test_rename_keeps_object_key(case, monkeypatch, toolchains) -> None:  # noqa: F811
    case, calls, _ = _tools(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    texts = {"src/f.c": "int f;\n", "src/renamed.c": "int f;\n"}
    _text(monkeypatch, texts)
    monkeypatch.setattr(native.versions, "rom_bytes", lambda *a: b"")
    other = replace(unit, path="src/renamed.c")
    snapshot = replace(snapshot, overlays={**snapshot.overlays, other.path: b"int f;\n"})
    assert native.stamp(snapshot, unit, "a", recipe) == native.stamp(snapshot, other, "a", recipe)
    native.objects(snapshot, unit, "a", recipe, work / "first")
    native.objects(snapshot, other, "a", recipe, work / "renamed")
    assert len(calls) == 1  # the renamed file is the same text: one compile


def test_flag_change_misses(case, monkeypatch, toolchains) -> None:  # noqa: F811
    case, calls, _ = _tools(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    _text(monkeypatch, {unit.path: "int f;\n"})
    monkeypatch.setattr(native.versions, "rom_bytes", lambda *a: b"")
    other = replace(recipe, digest="recipe-O3")
    assert native.stamp(snapshot, unit, "a", recipe) != native.stamp(snapshot, unit, "a", other)
    native.objects(snapshot, unit, "a", recipe, work / "first")
    monkeypatch.setattr(
        native.adapters,
        "chain_steps",
        Mock(return_value=(native.compile.Step("compile", "source.o", ("compile", "{in}", "{out}", "-O3")),)),
    )
    native.objects(snapshot, unit, "a", other, work / "flags")
    assert len(calls) == 2  # other flags: a miss


def test_warm_measure_compiles_nothing(case, monkeypatch, toolchains) -> None:  # noqa: F811
    case, calls, _ = _tools(case, monkeypatch, toolchains)
    snapshot, unit, recipe, _, work, _ = case
    _text(monkeypatch, {unit.path: "int f;\n"})
    for name in ("cold", "warm"):
        native.objects(snapshot, unit, "a", recipe, work / name)
    assert len(calls) == 1  # the second object came from the store: 0 compiles


def test_workers_honour_host() -> None:
    host = Mock(workers=7, slots=3)
    assert pool._fan(host) == pool._budget(host) == 7


def _make_run(cfg, monkeypatch, record=None):
    """process.make with the pool, make and the batch replaced: the runs it started, the batches it asked for."""
    ran, batches = [], []
    monkeypatch.setattr(
        pool, "jobserver", lambda host, work=0: __import__("contextlib").nullcontext((4, {"MAKEFLAGS": "-j4"}, ()))
    )
    monkeypatch.setattr(process, "tool", lambda config, name, **k: cfg.project.root / "make")

    def run(*a, **k):
        ran.append((a, k))
        if record and "ABU_CC_COLLECT" in k["env"]:
            Path(k["env"]["ABU_CC_COLLECT"]).write_text(record)
        return f"result{len(ran)}"

    monkeypatch.setattr(process, "run", run)
    monkeypatch.setattr(process, "batch", lambda *a: batches.append(a))
    return ran, batches


def test_make_is_one_process_when_the_front_records_nothing(cfg, monkeypatch) -> None:
    ran, batches = _make_run(cfg, monkeypatch)
    assert process.make(cfg, ["check"], 4) == "result1"
    assert len(ran) == 1 and ran[0][0][1][-1] == "check" and ran[0][1]["env"]["MAKEFLAGS"] == "-j4"
    assert "ABU_CC_COLLECT" in ran[0][1]["env"] and not batches


def test_make_without_a_front_cache_is_one_plain_process(cfg, monkeypatch) -> None:
    cfg = replace(cfg, host=replace(cfg.host, cache_dir=None))
    ran, batches = _make_run(cfg, monkeypatch)
    assert process.make(cfg, ["check"], 4) == "result1"
    assert len(ran) == 1 and ran[0][1]["env"] == {"MAKEFLAGS": "-j4"} and not batches


def test_make_answers_the_recorded_compiles_in_one_batch_between_two_runs(cfg, monkeypatch) -> None:
    root = cfg.project.root
    (root / "build").mkdir(exist_ok=True)
    (root / "build/a.s.cc").write_text("left over")
    record = "/cc1\x1f-quiet\x1fbuild/a.i\x1f-o\x1fbuild/a.s\n/cc1\x1f-E\x1fb.c\n"
    ran, batches = _make_run(cfg, monkeypatch, record)
    assert process.make(cfg, ["check"], 4) == "result2"  # the second run is the build
    assert [("ABU_CC_COLLECT" in k["env"]) for _, k in ran] == [True, False]
    assert len(batches) == 1 and batches[0][3] == 4  # across the granted slots
    assert not (root / "build/a.s.cc").exists()  # an answer nobody claimed does not outlive the build
    assert not list((root / ".unbake/tmp").glob("cc-commands.*"))
