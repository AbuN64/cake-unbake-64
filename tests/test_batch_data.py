"""Batch data replaces INCLUDE_DATA / INCLUDE_RODATA lines in their owners and builds once."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_publish import World

from unbake import batch_data
from unbake.contracts import Refusal


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    (tmp_path / "src/a.c").write_text(
        'INCLUDE_DATA("main", D_80001000);\nINCLUDE_RODATA("main", D_80002000);\nint f(void) { return 0; }\n'
    )
    w.tree.modules[0].owns = {v: ("D_80001000", "D_80002000", "f") for v in ("us", "eu")}
    (tmp_path / "d1.c").write_text("int D_80001000[2] = {1, 2};\n")
    (tmp_path / "d2.c").write_text('const char D_80002000[] = "hi";\n')
    (tmp_path / "rows.json").write_text(
        json.dumps([{"member": "D_80001000", "file": "d1.c"}, {"member": "D_80002000", "file": "d2.c"}])
    )
    monkeypatch.setattr(batch_data, "tree", SimpleNamespace(capture=lambda config: w.tree))
    return w


def _params(world):
    return {"files": [], "function": None, "batch_data": str(world.root / "rows.json")}


def test_manifest_rejects_duplicates(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps([{"member": "a", "file": "x"}, {"member": "a", "file": "y"}]))
    with pytest.raises(Refusal):
        batch_data._manifest(tmp_path / "m.json")


def test_a_batch_replaces_each_line_and_builds_once(world, monkeypatch):
    logged = []
    monkeypatch.setattr(batch_data.store, "log", lambda c, kind, body: logged.append(body["member"]))
    out = batch_data.run(world.config, _params(world))
    text = world.owner
    assert "INCLUDE_DATA" not in text and "INCLUDE_RODATA" not in text
    assert "int D_80001000[2] = {1, 2};" in text and 'const char D_80002000[] = "hi";' in text
    assert world.makes == [["build/eu/game.z64", "build/us/game.z64"]]
    assert logged == ["D_80001000", "D_80002000"] and len(out["submissions"]) == 2


def test_a_batch_that_does_not_build_exact_changes_nothing(world):
    before = world.owner
    world.broken = {"us"}
    with pytest.raises(Refusal):
        batch_data.run(world.config, _params(world))
    assert world.owner == before
