"""rescore: the fuzzy rows scored again, keyed per row."""

import pickle

from unbake import rescore


def test_the_row_key_is_a_module_function_the_workers_can_pickle() -> None:
    assert pickle.loads(pickle.dumps(rescore._row_key)) is rescore._row_key


class _Tree:
    """A snapshot stand-in: the files a row key reads, and nothing else."""

    def __init__(self, files, layout="layout-1"):
        from types import SimpleNamespace

        self.files, self.digest = files, f"tree-{layout}"
        self.layout = SimpleNamespace(members={"f": ("f", "placements")}, units={"other": layout})
        self.versions = {}
        self.config = SimpleNamespace(digest="config")

    def peek(self, path):
        return self.files.get(path)


FILES = {
    "src/fuzzy/f.c": b'#include "a.h"\nint f(void) { return 1; }\n',
    "include/a.h": b'#include "b.h"\n',
    "include/b.h": b"typedef int B;\n",
    "include/other.h": b"typedef int O;\n",
}


def _key(tree):
    row = {"path": "src/fuzzy/f.c"}
    return rescore._row_key((tree, "f", row, rescore.reach(tree, row["path"])))


def test_a_row_key_moves_with_what_its_score_reads_never_with_a_landing_elsewhere() -> None:
    """Every drain that landed re-keyed all 267 fuzzy rows (~6 s each) as the key held the tree digest."""
    before = _key(_Tree(dict(FILES)))
    assert _key(_Tree({**FILES, "include/other.h": b"typedef int P;\n"}, layout="layout-2")) == before
    assert _key(_Tree({**FILES, "include/b.h": b"typedef short B;\n"})) != before  # reached through a.h
    assert _key(_Tree({**FILES, "src/fuzzy/f.c": FILES["src/fuzzy/f.c"] + b"\n"})) != before
