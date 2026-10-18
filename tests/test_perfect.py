"""The perfect metric on mocked modules: no stand-in in any version that links exact."""

from types import SimpleNamespace

from unbake import perfect


def _module(path, owns, stand_ins=None):
    stand_ins = stand_ins or {}
    return SimpleNamespace(path=path, owns=owns, stand_ins=stand_ins)


def _row(version, group, code=16, data=0):
    return SimpleNamespace(version=version, code=code, data=data, member=SimpleNamespace(group=group))


def test_a_module_is_perfect_when_no_exact_version_holds_a_stand_in():
    tree = SimpleNamespace(
        digest="perfect-1",
        modules=(
            _module("src/c.c", {"a": ("f",), "b": ("f",)}),
            _module("src/s.c", {"a": ("g",), "b": ("g",)}, {"a": ("g",)}),
            _module("src/b.c", {"a": ("h",), "b": ("h",)}, {"b": ("h",)}),  # stand-in only where the link is not exact
        ),
    )
    rows = [_row(v, p) for v in ("a", "b") for p in ("src/c.c", "src/s.c", "src/b.c")]
    census, modules = perfect.census(tree, rows, {"a"})
    assert modules == {"src/c.c", "src/b.c"}
    assert census["a"] == {"bytes": 32, "total": 48, "modules": 3, "perfect_modules": 2}
    assert census["b"]["bytes"] == 32 and census["b"]["total"] == 48


def test_the_census_is_memoised_by_the_tree_digest():
    tree = SimpleNamespace(digest="perfect-2", modules=(_module("src/c.c", {"a": ("f",)}),))
    first = perfect.census(tree, [_row("a", "src/c.c")], {"a"})
    rows = [_row("a", "src/c.c", code=999)]  # same digest: the rows are not read again
    assert perfect.census(tree, rows, {"a"}) is first
    assert (
        perfect.census(SimpleNamespace(digest="perfect-3", modules=tree.modules), rows, {"a"})[0]["a"]["total"] == 999
    )


def test_an_empty_tree_has_an_empty_census():
    assert perfect.census(SimpleNamespace(digest="perfect-4", modules=()), [], set()) == ({}, frozenset())
