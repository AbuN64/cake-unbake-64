"""The perfect metric per axis on a mocked ROM: boundary (B), flags (F), members (M) and name (N)."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

from unbake import edges, perfect
from unbake.contracts import Group, LayoutMap, Member, Placement, SdkHit, UnitSpec

JR, NOP, OP = 0x03E00008, 0, 0x27BDFFE8
# f1 | f2 f3 | f4: pad before f2 (a padding word), unal between f2 and f3, pad before f4 (12 zero bytes)
FUNCTIONS = (
    ("f1", 0x1000, [OP, JR, NOP, NOP]),
    ("f2", 0x1010, [OP, JR, NOP]),
    ("f3", 0x101C, [OP, OP, JR, NOP, OP, OP]),
    ("f4", 0x1040, [OP, JR, NOP]),
)


def _snapshot(tmp_path, groups, units, *, boundaries=(), library="", evidence=(), functions=FUNCTIONS):
    tmp_path.mkdir(exist_ok=True)
    rom = bytearray(0x2000)
    for _, start, words in functions:
        rom[start : start + 4 * len(words)] = struct.pack(f">{len(words)}I", *words)
    (tmp_path / "a.z64").write_bytes(bytes(rom))
    group_of = {n: g for g, names in groups.items() for n in names}
    members = {
        n: Member(n, "function", "c", group_of[n], (Placement("a", ".text", s, s + 4 * len(w), 0x80000000 + s),))
        for n, s, w in functions
    }
    lay = LayoutMap(
        {
            g: Group(g, "main", names, "authored", (), bool(library), f"src/{g}.c", f"include/{g}.h", "", library)
            for g, names in groups.items()
        },
        members,
        {u.path: u for u in units},
        f"lay-{tmp_path.name}-{sorted(groups.items())}-{[u.path for u in units]}-{boundaries}-{library}-{evidence}",
        (),
        {},
        tuple(boundaries),
    )
    version = SimpleNamespace(rom=tmp_path / "a.z64", rom_sha256="sha", segments=(("main", 0x1000, 0x2000, 0),))
    project = SimpleNamespace(versions=("a",), names_from="a", evidence_objects=evidence, library_dirs=())
    return SimpleNamespace(
        digest=lay.digest, config=SimpleNamespace(project=project), layout=lay, versions={"a": version}
    )


def _unit(path, names, kind="c", options=None):
    return UnitSpec(path, kind, "", tuple(names), "test", options or {})


@pytest.fixture(autouse=True)
def _inline(monkeypatch):
    monkeypatch.setattr(edges, "pool", SimpleNamespace(map=lambda c, n, f, items, key=None: [*map(f, items)]))
    monkeypatch.setattr(perfect.infer, "sdk", lambda s: {})


def test_a_module_between_two_pad_cuts_laid_as_one_unit_is_a_proven_boundary(tmp_path):
    snapshot = _snapshot(
        tmp_path,
        {"a": ("f1",), "m": ("f2", "f3"), "z": ("f4",)},
        [_unit("src/a.c", ["f1"]), _unit("src/m.c", ["f2", "f3"]), _unit("src/z.c", ["f4"])],
    )
    record = perfect.objects(snapshot)["m"]
    assert record == {
        "start": "pad",
        "end": "pad",
        "inside": "joined",
        "units": 1,
        "flags": "default",
        "name": "assigned",
        "boundary": True,
        "split_debt": 0,
    }
    assert perfect.objects(snapshot)["a"]["start"] == "segment"


def test_a_module_edge_at_a_proven_join_is_a_defect(tmp_path):
    snapshot = _snapshot(
        tmp_path,
        {"m": ("f1", "f2"), "n": ("f3", "f4")},
        [_unit("src/m.c", ["f1", "f2"]), _unit("src/n.c", ["f3", "f4"])],
    )
    records = perfect.objects(snapshot)
    assert (records["m"]["end"], records["n"]["start"]) == ("join", "join")
    assert records["m"]["inside"] == records["n"]["inside"] == "cut"  # each holds a pad cut inside
    assert not records["m"]["boundary"] and not records["n"]["boundary"]


def test_two_units_or_a_split_proof_inside_break_the_boundary(tmp_path):
    groups = {"a": ("f1",), "m": ("f2", "f3"), "z": ("f4",)}
    two = _snapshot(tmp_path, groups, [_unit("src/m.c", ["f2"]), _unit("src/m_2.c", ["f3"])])
    assert perfect.objects(two)["m"]["units"] == 2 and not perfect.objects(two)["m"]["boundary"]
    debt = _snapshot(tmp_path, groups, [_unit("src/m.c", ["f2", "f3"])], boundaries=["f3"])
    assert perfect.objects(debt)["m"]["split_debt"] == 1 and not perfect.objects(debt)["m"]["boundary"]


@pytest.mark.parametrize(
    ("units", "flags"),
    [
        ([_unit("src/m.c", ["f2", "f3"], options={"add": [], "omit": []})], "default"),
        ([_unit("src/m.c", ["f2", "f3"], options={"add": ["-ffast-math"], "omit": []})], "open"),
        ([_unit("asm/m.s", ["f2", "f3"], kind="asm")], "open"),  # not every member compiles from C yet
    ],
)
def test_flags_are_default_only_when_every_function_compiles_without_options(tmp_path, units, flags):
    snapshot = _snapshot(tmp_path, {"a": ("f1",), "m": ("f2", "f3"), "z": ("f4",)}, units)
    assert perfect.objects(snapshot)["m"]["flags"] == flags


def test_names_come_from_sdk_library_or_an_evidence_row(tmp_path, monkeypatch):
    groups = {"a": ("f1",), "m": ("f2", "f3"), "z": ("f4",)}
    units = [_unit("src/m.c", ["f2", "f3"])]
    assert perfect.objects(_snapshot(tmp_path, groups, units))["m"]["name"] == "assigned"
    assert perfect.objects(_snapshot(tmp_path, groups, units, library="libgcc"))["m"]["name"] == "library"
    row = {"first": "f2", "last": "f3", "source": "AI/ENEMIES/TURRETAI.CPP", "origin": "debug stream"}
    assert perfect.objects(_snapshot(tmp_path, groups, units, evidence=(row,)))["m"]["name"] == "evidence"
    monkeypatch.setattr(perfect.infer, "sdk", lambda s: {n: SdkHit(n, "sqrtf", "libm") for n in ("f2", "f3")})
    assert perfect.objects(_snapshot(tmp_path / "sdk", groups, units))["m"]["name"] == "sdk"


def test_an_sdk_module_change_proves_an_open_edge(tmp_path, monkeypatch):
    hits = {"f3": SdkHit("x", "one", "libultra"), "f4": SdkHit("y", "two", "libultra")}
    monkeypatch.setattr(perfect.infer, "sdk", lambda s: hits)
    functions = (*FUNCTIONS[:2], ("f3", 0x101C, [OP, OP, JR, NOP, OP]), ("f4", 0x1030, [OP, JR, NOP]))
    snapshot = _snapshot(tmp_path, {"m": ("f1", "f2", "f3"), "z": ("f4",)}, [], functions=functions)
    assert perfect.objects(snapshot)["z"]["start"] == "sdk"  # al0: open by the ROM alone


def _credit(name, group, version, exact, code=16, data=0):
    return SimpleNamespace(
        name=name,
        member=SimpleNamespace(group=group),
        version=version,
        exact=exact,
        code=code,
        data=data,
        segment="main",
    )


def test_census_counts_axes_and_perfect_bytes_per_version(tmp_path):
    snapshot = _snapshot(
        tmp_path,
        {"a": ("f1",), "m": ("f2", "f3"), "z": ("f4",)},
        [_unit("src/a.c", ["f1", "f4"]), _unit("src/m.c", ["f2", "f3"])],
    )
    credits = [
        _credit("f1", "a", "a", True),
        _credit("f2", "m", "a", True, code=12, data=8),
        _credit("f3", "m", "a", True, code=20),
        _credit("f4", "z", "a", False),
    ]
    census, whole = perfect.census(snapshot, credits)
    assert whole == frozenset({"a", "m"})  # z misses an exact member
    row = census["a"]
    assert (row["bytes"], row["total"], row["modules"], row["perfect_modules"]) == (56, 72, 3, 2)
    assert (row["boundary"], row["flags"], row["members"], row["named"]) == (3, 3, 2, 0)
    assert row["module_starts"] == {"pad": 2, "segment": 1} and row["open_edges"] == 0 and row["contradictions"] == 0
