"""rescore: NON_MATCHING bodies measured once per content, through the `score` kind."""

import json
from types import SimpleNamespace

import pytest

from unbake import rescore

pytestmark = pytest.mark.usefixtures("plain_roms")


@pytest.fixture
def world(tmp_path, monkeypatch):
    (tmp_path / "roms").mkdir()
    (tmp_path / "roms/baserom.a.z64").write_bytes(bytes(range(64)))
    (tmp_path / "roms/baserom.b.z64").write_bytes(bytes(range(64, 128)))
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.c").write_text("#ifdef NON_MATCHING\nint f(void) { return 1; }\n#endif\n")
    (tmp_path / "src/b.c").write_text("int g(void) { return 1; }\n")
    stored, measured = {}, []
    monkeypatch.setattr(rescore, "_lookup", lambda config, key: stored.get(key))
    monkeypatch.setattr(rescore, "_record", lambda config, key, value: stored.__setitem__(key, value))
    monkeypatch.setattr(
        rescore,
        "_measure",
        lambda config, path, names, version: {n: measured.append((n, version)) or 0.25 for n in names},
    )
    monkeypatch.setattr(rescore.pool, "map", lambda config, name, fn, items, key=None: [fn(i) for i in items])
    config = SimpleNamespace(project=SimpleNamespace(root=tmp_path))
    tree = SimpleNamespace(
        digest="r",
        modules=(
            SimpleNamespace(path="src/a.c", owns={"a": ("f",), "b": ("f",)}, fuzzy=("f",)),
            SimpleNamespace(path="src/b.c", owns={"a": ("g",)}, fuzzy=()),
        ),
    )
    facts = {"a": {"f": (0x80000000, 0x10, 16, "func"), "g": (0, 0, 8, "func")}, "b": {"f": (0, 0x20, 16, "func")}}
    return config, tree, facts, stored, measured, tmp_path


def test_only_non_matching_names_are_measured_one_per_version(world):
    config, tree, facts, _stored, measured, _ = world
    assert rescore.rows(config, tree, facts) == {"f": {"a": 0.25, "b": 0.25}}
    assert sorted(measured) == [("f", "a"), ("f", "b")]


def test_a_changed_target_rekeys_only_that_version(world):
    config, tree, facts, _stored, measured, root = world
    rescore.rows(config, tree, facts)
    (root / "roms/baserom.b.z64").write_bytes(bytes(range(65, 129)))
    rescore.rows(config, tree, facts)
    assert measured[2:] == [("f", "b")]


def test_a_name_without_a_fact_in_a_version_is_skipped(world):
    config, tree, facts, _stored, measured, _ = world
    assert rescore.rows(config, tree, {"a": facts["a"]}) == {"f": {"a": 0.25}} and measured == [("f", "a")]


def test_the_key_names_content_never_a_path(world):
    config, tree, facts, stored, _measured, root = world
    rescore.rows(config, tree, facts)
    keys = set(stored)
    (root / "src/c.c").write_bytes((root / "src/a.c").read_bytes())  # the same text under another path
    moved = SimpleNamespace(
        digest="r2", modules=(SimpleNamespace(path="src/c.c", owns=tree.modules[0].owns, fuzzy=("f",)),)
    )
    rescore.rows(config, moved, facts)
    assert set(stored) == keys


def test_a_score_is_stored_as_json_in_the_score_kind(monkeypatch):
    puts = []
    monkeypatch.setattr(rescore.store, "put", lambda config, kind, key, value: puts.append((kind, key, value)))
    monkeypatch.setattr(rescore.store, "get", lambda config, kind, key: json.dumps(0.5).encode())
    rescore._record(None, "k", 0.5)
    assert puts == [("score", "k", b"0.5")] and rescore._lookup(None, "k") == 0.5


def test_the_missing_functions_of_a_module_are_measured_together_once_per_version(world):
    config, tree, facts, _stored, _measured, root = world
    (root / "src/a.c").write_text("#ifdef NON_MATCHING\nint f(void) { return 1; }\nint h(void) { return 2; }\n#endif\n")
    tree.modules[0].owns["a"] = ("f", "h")
    tree.modules[0].owns["b"] = ("f", "h")
    tree.modules[0].fuzzy = ("f", "h")
    facts["a"]["h"] = (0x80000010, 0x20, 8, "func")
    facts["b"]["h"] = (0x80000010, 0x30, 8, "func")
    calls = []
    original = rescore._measure
    rescore._measure = lambda config, path, names, version: (
        calls.append((path, tuple(names), version)) or original(config, path, names, version)
    )
    try:
        assert rescore.rows(config, tree, facts) == {"f": {"a": 0.25, "b": 0.25}, "h": {"a": 0.25, "b": 0.25}}
        assert calls == [("src/a.c", ("f", "h"), "a"), ("src/a.c", ("f", "h"), "b")]
        calls.clear()
        rescore.rows(config, tree, facts)
        assert calls == []  # a warm rerun measures nothing
    finally:
        rescore._measure = original


def test_only_names_given_are_looked_up_and_measured(world):
    config, tree, facts, _stored, measured, _ = world
    assert rescore.rows(config, tree, facts, ("g",)) == {} and measured == []
    assert rescore.rows(config, tree, facts, ("f",)) == {"f": {"a": 0.25, "b": 0.25}}
    assert sorted(measured) == [("f", "a"), ("f", "b")]
