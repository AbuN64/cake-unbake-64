"""Placement of modules: library regions, subsystem runs, data owners and the move stream."""

from __future__ import annotations

from unbake.layout import subsystems
from unbake.layout.subsystems import Node


def node(i: int, segment: str = "main", **kw) -> Node:
    return Node(f"m{i}", segment, 0x80000000 + 0x100 * i, f"m{i}.c", **kw)


def test_a_small_segment_is_one_subsystem_without_a_segment_level() -> None:
    placed = subsystems.place([node(i) for i in range(5)])
    assert {p.source for p in placed.values()} == {f"src/game/sys_80000000/m{i}.c" for i in range(5)}
    assert placed["m3"].header == "include/game/sys_80000000.h" and placed["m3"].subsystem == "sys_80000000"


def test_two_game_segments_add_the_segment_level() -> None:
    placed = subsystems.place([node(0), node(1, "ovl")])
    assert placed["m0"].source == "src/game/main/sys_80000000/m0.c" and placed["m1"].subsystem == "ovl/sys_80000100"


def test_a_leaf_closed_region_becomes_a_library() -> None:
    far = tuple(f"m{i}" for i in range(9, 14))
    game = [node(i, calls=("m9", *far)) for i in range(3)]  # game code calls into the region and out into game code
    region = [node(i, calls=(f"m{i + 1}",)) for i in range(3, 9)] + [node(9)]
    tail = [node(i, calls=("m0",)) for i in range(10, 20)]
    placed = subsystems.place(game + region + tail)
    assert {placed[f"m{i}"].library for i in range(3, 10)} == {"lib_80000300"}
    assert placed["m4"].source == "src/sdk/lib_80000300/m4.c" and placed["m4"].header == "include/sdk/lib_80000300.h"
    assert placed["m0"].library == "" and placed["m12"].library == ""


def test_the_catalog_library_names_a_module_whatever_its_calls() -> None:
    placed = subsystems.place([node(0, library="libx"), node(1)])
    assert placed["m0"].source == "src/sdk/libx/m0.c" and placed["m1"].library == ""


def test_regions_need_calls_that_stay_inside_and_callers_outside() -> None:
    order = [f"m{i}" for i in range(12)]
    chain = {f"m{i}": {f"m{i + 1}": 1} for i in range(7)}
    outside = {f"m{i}": {f"m{i + 1}": 1, "m0": 1} for i in range(8, 11)} | {"m11": {"m0": 1}}
    assert subsystems.regions(order, chain | outside, 40) == [(0, 8)]
    assert subsystems.regions(order, chain | outside, 12) == []  # not under half the segment
    assert subsystems.regions(order, chain | outside | {"m3": {"m4": 1, "m10": 9}}, 40) == []  # too many leave


def test_runs_cut_between_two_dense_blocks_within_the_size_bounds() -> None:
    weight = {(a, b): 1.0 for lo in (0, 20) for a in range(lo, lo + 20) for b in range(a + 1, lo + 20)}
    weight[19, 20] = 1.0
    assert subsystems.runs(40, weight) == [(0, 20), (20, 40)]
    assert subsystems.runs(7, {}) == [(0, 7)] and subsystems.runs(0, {}) == []
    assert sum(b - a for a, b in subsystems.runs(70, {})) == 70


def test_data_goes_beside_its_main_loader_or_into_shared() -> None:
    owned = [node(0, loads=("d0",)), node(1, loads=("d0",)), node(2, loads=("d1",)), node(3, loads=("d1",))]
    owned += [node(4, loads=("d1",)), node(5, loads=("d0",))]
    data = [Node("d0", "main", 0x80001000, "d0.c", True), Node("d1", "main", 0x80001100, "d1.c", True)]
    placed = subsystems.place(owned + data)
    assert placed["d0"].source == "src/game/sys_80000000/d0.c" and placed["d0"].header == ""
    lone = subsystems.place([*owned[:2], Node("d2", "main", 1, "d2.c", True), node(8, loads=("d2",))])
    assert lone["d2"].source.startswith("src/game/sys_")
    split = [node(0, loads=("d",)), node(1, loads=("d",))]
    same = subsystems.place([*split, Node("d", "main", 1, "d.c", True)])
    assert same["d"].subsystem == "sys_80000000"


def test_cohorts_label_nothing_without_anchors() -> None:
    assert set(subsystems.cohorts([node(0, calls=("m1",)), node(1)]).values()) == {""}
