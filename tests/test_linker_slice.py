"""The link script of a slice of real RageWars data (400 ROM rows of the `eu` main segment, three pieces of one tag,
14 files built with a section per function) and the lookups it relies on. The slice is the regression of the
quadratic scan: the script was once rebuilt by a pass over every module for every piece, 683,000 times on the full
tree (35 minutes); it now reads each version's rows once."""

import json
from pathlib import Path
from types import SimpleNamespace

from unbake import build
from unbake.contracts import Segment
from unbake.tree import Module, Tree

DATA = Path(__file__).parent / "data"


def _slice(tmp_path: Path) -> tuple[SimpleNamespace, Tree]:
    data = json.loads((DATA / "linker_slice.json").read_text())
    v = data["version"]
    segments = tuple(
        Segment(data["segment"], rom, data["vram"] + rom - data["cuts"][0], "code") for rom in data["cuts"]
    )
    project = SimpleNamespace(
        root=tmp_path,
        name="RageWars",
        toolchain=data["toolchain"],
        versions=(v,),
        version={v: SimpleNamespace(sha1="a" * 40, size=data["end"], segments=segments)},
    )
    if data["fills"]:
        (tmp_path / f"versions/{v}").mkdir(parents=True)
        (tmp_path / f"versions/{v}/linker_fill.toml").write_text(data["fills"])
    modules = tuple(
        Module(m["path"], data["segment"], {v: tuple(m["owns"])}, (), {}, {v: m["sections"]}) for m in data["modules"]
    )
    return SimpleNamespace(project=project), Tree(
        "slice", modules, {v: {n: tuple(f) for n, f in data["facts"].items()}}
    )


def test_the_link_script_of_real_data_is_what_the_per_piece_rescan_wrote(tmp_path):
    config, tree = _slice(tmp_path)
    assert build.linker(config, tree).decode() == (DATA / "linker_slice.ld").read_text()


def test_a_range_of_placed_rows_is_the_sorted_rows_that_lie_in_it():
    _, tree = _slice(Path("/nonexistent"))
    table = tree.symbols["eu"]
    everything = tree.placed("eu", 0, 1 << 40)
    assert everything == sorted(
        (table[n][1], m.path, m.sections["eu"].get(n, ".text"), n) for m in tree.modules for n in m.owns["eu"]
    )
    low, high = everything[100][0], everything[250][0]
    assert tree.placed("eu", low, high) == [r for r in everything if low <= r[0] < high]
    assert tree.placed("eu", high, low) == []
    assert tree.placed("us", 0, 1 << 40) == []


def test_the_linker_reads_the_modules_once_per_version_whatever_the_pieces(tmp_path):
    config, tree = _slice(tmp_path)

    class Counted(tuple):
        passes = 0

        def __iter__(self):
            Counted.passes += 1
            return super().__iter__()

    counted = Tree("slice", Counted(tree.modules), tree.symbols)
    Counted.passes = 0
    build.linker(config, counted)
    assert len(build.versions.pieces(config, "eu")) == 3
    assert Counted.passes <= 2  # the rows and the literal table, once each: not a pass per piece and helper


def test_fills_of_a_piece_stand_between_the_objects_they_lie_between_and_after_the_last(tmp_path):
    from test_build import both, make_config, make_tree

    from unbake import linkfill

    fills = [
        linkfill.Fill("src/a.c", ".text", rom, 0x80200000 + rom, b"\x01\x02", "x") for rom in (0x1020, 0x1030, 0x1100)
    ]
    (tmp_path / "versions/us").mkdir(parents=True, exist_ok=True)
    (tmp_path / "versions/us/linker_fill.toml").write_text(linkfill.render("us", fills, "test"))
    tree = make_tree(both("src/a.c", "f_1"), both("src/b.c", "f_4"))
    lines = [ln.strip() for ln in build.linker(make_config(tmp_path), tree).decode().splitlines()]
    at = {ln: i for i, ln in enumerate(lines)}
    first, second, last = (linkfill.statement(f, 0x1000) + ";" for f in fills)
    assert at["src/a.c.o(.text);"] < at[first] < at[second] < at["src/b.c.o(.text);"]
    assert at["src/b.c.o(.text);"] < at[last]
