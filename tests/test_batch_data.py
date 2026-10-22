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


def _git(world, shown):
    """A fake HEAD: `show` answers the file as committed, `diff` the patch text."""

    def git(config, *args):
        if args[0] == "ls-tree":
            return SimpleNamespace(exit=0, stdout=b"100644 blob x\t" + args[-1].encode() if args[-1] in shown else b"")
        if args[0] == "show":
            return SimpleNamespace(exit=0, stdout=shown[args[1].removeprefix("HEAD:")])
        return SimpleNamespace(exit=0, stdout=b"PATCH of " + b" ".join(a.encode() for a in args[4:]))

    return SimpleNamespace(git=git)


def _params(world):
    return {"files": [], "function": None, "batch_data": str(world.root / "rows.json")}


def test_manifest_rejects_duplicates(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps([{"member": "a", "file": "x"}, {"member": "a", "file": "y"}]))
    with pytest.raises(Refusal):
        batch_data._manifest(tmp_path / "m.json")


def test_a_batch_replaces_each_line_and_builds_once(world, monkeypatch):
    before = world.owner
    monkeypatch.setattr(batch_data, "process", _git(world, {"src/a.c": before.encode()}))
    seen = []
    real = batch_data.publish.apply_checked

    def applied(config, landing):
        files = real(config, landing)
        seen.append(world.owner)
        return files

    monkeypatch.setattr(batch_data.publish, "apply_checked", applied)
    out = batch_data.run(world.config, _params(world))
    assert "INCLUDE_DATA" not in seen[0] and 'const char D_80002000[] = "hi";' in seen[0]  # proved written, built whole
    assert world.makes == [["build/eu/game.z64", "build/us/game.z64"]]
    assert len(out["submissions"]) == 2 and out["drain"]["landed"] == []  # nothing landed: a patch was written
    assert world.owner == before  # the tracked tree is back as it was
    assert (world.root / out["patch"]["path"]).read_bytes().startswith(b"PATCH of")


def test_a_batch_that_does_not_build_exact_changes_nothing(world):
    before = world.owner
    world.broken = {"us"}
    with pytest.raises(Refusal):
        batch_data.run(world.config, _params(world))
    assert world.owner == before


def test_a_batch_retypes_data_the_owner_already_defines_in_c_and_builds_it(world, monkeypatch):
    world.root.joinpath("src/a.c").write_text(
        'INCLUDE_DATA("main", D_80001000);\n/* font */\nunsigned int D_80002000[1] = {\n    0,\n};\n'
        "int f(void) { return 0; }\n"
    )
    world.root.joinpath("d2.c").write_text("unsigned char *D_80002000 = 0;\n")
    written = world.root.joinpath("src/a.c").read_text()
    monkeypatch.setattr(batch_data, "process", _git(world, {"src/a.c": written.encode()}))
    seen = []
    real = batch_data.publish.apply_checked

    def applied(config, landing):
        files = real(config, landing)
        seen.append(world.owner)
        return files

    monkeypatch.setattr(batch_data.publish, "apply_checked", applied)
    batch_data.run(world.config, _params(world))
    text = seen[0]
    assert "unsigned char *D_80002000 = 0;" in text and "unsigned int D_80002000[1]" not in text
    assert "/* font */" in text and "INCLUDE_DATA" not in text
    assert world.makes == [["build/eu/game.z64", "build/us/game.z64"]]  # the bytes are proven by the exact build
