"""infer: group partition, SDK identification, cross-version correspondence and the plan."""

from __future__ import annotations

import json
import tomllib
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import fixture
import pytest

from unbake import config as configuration
from unbake import infer, layout, pool
from unbake.contracts import Group, Member, Refusal, SdkHit, Snapshot

JR_RA, NOP = 0x03E00008, 0
VRAM = fixture.VRAM_BASE


def blob(*words: int) -> bytes:
    return b"".join(w.to_bytes(4, "big") for w in words)


def jal(address: int) -> int:
    return 0x0C000000 | ((address >> 2) & 0x03FFFFFF)


# The fixture writes no `unit` table for an empty list and replaces empty groups by an authored one,
# so every project carries one asm unit and one empty inferred group (infer drops both).
UNIT = {
    "path": "asm/a/f0.s",
    "kind": "asm",
    "group": "u",
    "members": ["f0"],
    "toolchain": "gcc-test",
    "options": {"add": [], "omit": []},
}
EMPTY = {
    "name": "empty",
    "segment": "main",
    "members": [],
    "evidence": "inferred",
    "signals": [],
    "subsystem": "",
    "library": "",
}
LEAF = (0x24020001, 0x24030002, JR_RA, NOP)
TAIL = (0x24040004, 0x24040004, JR_RA, NOP)  # the last member runs to the end of the ROM page


def vram(offset: int) -> int:
    return VRAM + offset - fixture.ROM_BASE


def both(offset: int, code: bytes) -> dict[str, tuple[int, bytes]]:
    return {v: (offset, code) for v in ("a", "b")}


def snap(tmp: Path, **kw: Any) -> Snapshot:
    kw.setdefault("groups", [EMPTY])
    units = kw.pop("units", [UNIT])
    cfg = fixture.config(tmp, units=units, **kw)
    for name, placed in kw.get("functions", {}).items():
        for version in placed:
            fixture._write(cfg.project.root / "asm" / version / f"{name}.s", f"glabel {name}\n")
    return layout.capture(cfg)


def chain(tmp: Path, bodies: list[tuple[int, ...]], **kw: Any) -> Snapshot:
    """Functions f0.. packed from 0x1000, each body written as given (four words keeps them 16 bytes)."""
    functions, offset = {}, 0x1000
    for i, words in enumerate(bodies):
        functions[f"f{i}"] = both(offset, blob(*words))
        offset += 4 * len(words)
    return snap(tmp, functions=functions, **kw)


def calls(i: int) -> tuple[int, ...]:
    return (jal(vram(0x1000 + 0x10 * i)), NOP, JR_RA, NOP)


def names(result: Any) -> list[tuple[str, ...]]:
    return [g.members for g in result]


def module(**kw: Any) -> dict[str, Any]:
    return {
        "name": "mine",
        "segment": "main",
        "members": ["f0"],
        "evidence": "authored",
        "signals": [],
        "subsystem": "",
        "library": "",
    } | kw


@pytest.mark.usefixtures("toolchains")
def test_unsignalled_functions_pack_up_to_the_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = chain(tmp_path, [LEAF, LEAF, LEAF])
    whole = infer.groups(snapshot)
    assert names(whole) == [("f0", "f1", "f2")] and whole[0].signals == ()
    assert whole[0].evidence == "inferred" and not whole[0].sdk
    name = f"code_{vram(0x1000):08X}"
    assert (whole[0].name, whole[0].source, whole[0].header) == (name, f"src/{name}.c", f"include/main/{name}.h")
    monkeypatch.setattr(infer, "_cap", lambda: 2)
    capped = infer.groups(snapshot)
    assert names(capped) == [("f0", "f1"), ("f2",)] and "cap" in capped[0].signals


@pytest.mark.usefixtures("toolchains")
def test_callee_join_is_a_signal(tmp_path: Path) -> None:
    result = infer.groups(chain(tmp_path, [calls(1), LEAF, LEAF]))
    assert names(result) == [("f0", "f1", "f2")] and result[0].signals == ("callee",)


@pytest.mark.usefixtures("toolchains")
def test_padding_cut_splits_groups_and_blocks_a_join(tmp_path: Path) -> None:
    far = (jal(vram(0x1020)), NOP, JR_RA, NOP)  # f0 calls f2 across f1
    joined = infer.groups(chain(tmp_path / "j", [far, LEAF, LEAF]))
    assert names(joined) == [("f0", "f1", "f2")] and joined[0].signals == ("callee",)
    padded = (jal(vram(0x1020)), JR_RA, NOP, NOP)  # trailing zero word beyond the delay slot
    cut = infer.groups(chain(tmp_path / "c", [padded, LEAF, LEAF]))
    assert names(cut) == [("f0",), ("f1", "f2")]
    assert cut[0].signals == ("padding",) and "callee" not in cut[1].signals


@pytest.mark.usefixtures("toolchains")
def test_authored_group_kept_and_excluded_from_inference(tmp_path: Path) -> None:
    result = infer.groups(chain(tmp_path, [LEAF, LEAF], groups=[module()]))
    assert result[0].name == "mine" and result[0].evidence == "authored"
    assert names(result) == [("f0",), ("f1",)]
    assert result[1].evidence == "inferred" and result[1].members == ("f1",)


@pytest.mark.usefixtures("toolchains")
def test_an_authored_group_drops_a_row_that_no_longer_exists_and_the_plan_runs(tmp_path: Path) -> None:
    # real case: setup's second map pass reads the ROM symbol table, which renamed the data row
    # src/ragewars_u64_remainder_bitwidth_us_rev1.c; its authored module still listed it and _plan raised KeyError
    snapshot = chain(tmp_path, [LEAF, LEAF], groups=[module(members=["f0", "renamed_row"])])
    result = infer.groups(snapshot)
    assert result[0].name == "mine" and result[0].members == ("f0",)
    infer.plan(snapshot)  # no KeyError for the vanished name


@pytest.mark.usefixtures("toolchains")
def test_an_sdk_function_between_game_functions_is_cut_into_its_own_module(tmp_path: Path) -> None:
    snapshot = chain(tmp_path, [LEAF, LEAF, LEAF])
    assert names(infer.groups(snapshot, frozenset({"f1"}))) == [("f0",), ("f1",), ("f2",)]
    assert [g.signals for g in infer.groups(snapshot, frozenset({"f1"}))][:2] == [("sdk",), ("sdk",)]
    assert names(infer.groups(snapshot, frozenset())) == [("f0", "f1", "f2")]


@pytest.mark.usefixtures("toolchains")
def test_a_toolchain_change_between_units_is_a_module_edge(tmp_path: Path) -> None:
    other = {**UNIT, "path": "asm/a/f1.s", "members": ["f1"], "toolchain": "other"}
    snapshot = chain(tmp_path, [LEAF, LEAF, LEAF], units=[UNIT, other])
    assert names(infer.groups(snapshot, frozenset())) == [("f0",), ("f1", "f2")]


@pytest.mark.usefixtures("toolchains")
def test_group_order_walk_does_no_member_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compares = [0]
    real = Member.__eq__

    def counted(self: Member, other: object) -> bool:
        compares[0] += 1
        return real(self, other)

    monkeypatch.setattr(Member, "__eq__", counted)
    infer.groups(chain(tmp_path, [LEAF] * 12), frozenset())
    assert compares[0] == 0


@pytest.mark.usefixtures("toolchains")
def test_cap_limits_group_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = chain(tmp_path, [calls(1), calls(2), calls(3), LEAF])
    assert len(infer.groups(snapshot)) == 1
    monkeypatch.setattr(infer, "_cap", lambda: 2)
    result = infer.groups(snapshot)
    assert [m for g in result for m in g.members] == ["f0", "f1", "f2", "f3"]
    assert max(len(g.members) for g in result) <= 2


@pytest.mark.usefixtures("toolchains")
def test_every_function_gets_a_group(tmp_path: Path) -> None:
    snapshot = chain(tmp_path, [LEAF, LEAF, LEAF], groups=[])
    result = infer.groups(snapshot)
    assert sorted(m for g in result for m in g.members) == ["f0", "f1", "f2"]
    assert len({g.name for g in result}) == len(result)
    planned = infer.plan(snapshot)
    value = layout.load_map(snapshot.config, snapshot.versions, planned.writes["layout.toml"], snapshot.read)
    assert [n for n, m in value.members.items() if m.kind == "function" and not m.group] == []


@pytest.mark.usefixtures("toolchains")
def test_version_only_member_is_placed_by_its_version_order(tmp_path: Path) -> None:
    functions = {
        "f0": both(0x1000, blob(*LEAF)),
        "extra": {"a": (0x1010, blob(*LEAF))},
        "f1": both(0x1020, blob(*LEAF)),
    }
    snapshot = snap(tmp_path, functions=functions)
    members = [m for g in infer.groups(snapshot) for m in g.members]
    assert members.index("extra") == members.index("f0") + 1 or members.index("extra") < members.index("f1")
    assert sorted(members) == ["extra", "f0", "f1"]


# ------------------------------------------------------------------ sdk


@pytest.fixture(autouse=True)
def no_shipped_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """The packaged catalog is judged in test_resources; here only the rows a test writes identify anything."""
    monkeypatch.setattr(infer, "_shipped_catalog", lambda: ({"source": "none", "signatures": []}, "empty"))


def catalog(tmp: Path, snapshot: Snapshot, signatures: list[dict[str, Any]]) -> Snapshot:
    path = tmp / "sdk.json"
    path.write_text(json.dumps({"source": "test", "signatures": signatures}))
    host = replace(snapshot.config.host, sdk_catalog=path)
    return replace(snapshot, config=replace(snapshot.config, host=host))


def words_row(name: str, words: tuple[int, ...], masks: tuple[int, ...] | None = None, **kw: Any) -> dict[str, Any]:
    masks = masks or (0,) * len(words)
    return {"name": name, "words": [f"{w:08x}" for w in words], "masks": [f"{m:08x}" for m in masks]} | kw


@pytest.mark.usefixtures("toolchains")
def test_sdk_without_catalog_is_empty(tmp_path: Path) -> None:
    assert infer.sdk(chain(tmp_path, [LEAF, TAIL])) == {}


@pytest.mark.usefixtures("toolchains")
def test_sdk_word_and_crc_signatures(tmp_path: Path) -> None:
    lui = (0x3C040001, 0x24030002, JR_RA, NOP)
    snapshot = chain(tmp_path, [LEAF, (0x24020001, JR_RA, NOP, NOP), lui, TAIL])
    masked = blob(0x3C040000, 0x24030000, JR_RA, NOP)
    rows = [
        words_row("leaf", LEAF),
        {"name": "lui", "size": 16, "crc_head": zlib.crc32(masked[:8]), "crc_body": zlib.crc32(masked)},
    ]
    assert set(infer.sdk(catalog(tmp_path, snapshot, rows))) == {"f0", "f2"}


@pytest.mark.usefixtures("toolchains")
def test_sdk_hit_carries_name_module_and_library(tmp_path: Path) -> None:
    snapshot = chain(tmp_path, [LEAF, TAIL])
    rows = [words_row("osLeaf", LEAF, module="leaf", library="libgultra", versions=["2.0L"])]
    assert infer.sdk(catalog(tmp_path, snapshot, rows)) == {"f0": SdkHit("osLeaf", "leaf", "libgultra")}


@pytest.mark.usefixtures("toolchains")
def test_sdk_ambiguous_signature_identifies_nothing(tmp_path: Path) -> None:
    snapshot = chain(tmp_path, [LEAF, TAIL])
    row_ = words_row("leaf", LEAF)
    assert infer.sdk(catalog(tmp_path, snapshot, [row_, {**row_, "name": "twin"}])) == {}


@pytest.mark.usefixtures("toolchains")
def test_sdk_mask_ignores_masked_bits(tmp_path: Path) -> None:
    snapshot = chain(tmp_path, [LEAF, TAIL])
    row_ = words_row("m", (0x24020000, *LEAF[1:]), (0x0000FFFF, 0, 0, 0))
    assert set(infer.sdk(catalog(tmp_path, snapshot, [row_]))) == {"f0"}


@pytest.mark.usefixtures("toolchains")
def test_sdk_compares_only_rows_of_the_members_size_and_the_second_call_none(tmp_path: Path) -> None:
    snapshot = chain(tmp_path, [LEAF, (0x24040004, 0x24040005, JR_RA, NOP), TAIL])
    rows = [words_row("a", LEAF), words_row("b", (*LEAF, NOP)), words_row("c", (0x24020001,) * 4)]
    with_rows = catalog(tmp_path, snapshot, rows)
    infer.SDK_COMPARES[0] = 0
    infer.sdk(with_rows)
    assert (
        infer.SDK_COMPARES[0] == 2 * 2
    )  # two four-word members x the two four-word rows; the five-word row is skipped
    infer.sdk(with_rows)
    assert infer.SDK_COMPARES[0] == 4


@pytest.mark.usefixtures("toolchains")
def test_sdk_host_catalog_adds_to_the_shipped_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shipped = {"source": "s", "signatures": [words_row("shipped", LEAF, module="m", library="libm")]}
    monkeypatch.setattr(infer, "_shipped_catalog", lambda: (shipped, "s"))
    snapshot = chain(tmp_path, [LEAF, (0x24040004, 0x24040005, JR_RA, NOP), TAIL])
    assert infer.sdk(snapshot) == {"f0": SdkHit("shipped", "m", "libm")}
    added = catalog(tmp_path, snapshot, [words_row("own", (0x24040004, 0x24040005, JR_RA, NOP))])
    assert {k: v.name for k, v in infer.sdk(added).items()} == {"f0": "shipped", "f1": "own"}


@pytest.mark.usefixtures("toolchains")
@pytest.mark.usefixtures("toolchains")
def test_proven_group_sdk_is_recomputed_in_the_plan(tmp_path: Path) -> None:
    mid = (0x24040004, 0x24040005, JR_RA, NOP)
    snapshot = chain(tmp_path, [LEAF, mid, TAIL], groups=[module(members=["f0", "f1", "f2"], evidence="proven")])
    tail = tuple(w for w, _ in infer._code(snapshot, infer._placement(snapshot, snapshot.layout.members["f2"])))
    partial = [words_row("leaf", LEAF), words_row("mid", mid)]
    assert "0 sdk units" in infer.plan(catalog(tmp_path, snapshot, partial)).message
    everything = [*partial, words_row("t", tail)]
    assert "1 sdk units" in infer.plan(catalog(tmp_path, snapshot, everything)).message


# ------------------------------------------------------------------ correspondence and plan


def corresponding(tmp: Path, pairs: list[tuple[str, str]], known_b: tuple[str, ...] = ()) -> Snapshot:
    """One function per (name_a, name_b) pair: version a calls name_a, version b calls name_b."""
    functions, symbols = {}, {"a": {}, "b": {}}
    for i, (_, y) in enumerate(pairs):
        offset = 0x1000 + 0x10 * i
        functions[f"f{i}"] = {
            "a": (offset, blob(jal(0x80001000 + 0x10 * i), NOP, JR_RA, NOP)),
            "b": (offset, blob(jal(0x80001100 + 0x10 * i), NOP, JR_RA, NOP)),
        }
        symbols["a"][f"f{i}"] = symbols["b"][f"f{i}"] = vram(offset)
        symbols["b"].setdefault(y, 0x80001100 + 0x10 * i)
    for name in known_b:
        symbols["b"][name] = 0x80002000
    root = fixture.project(tmp, functions=functions, symbols=symbols, groups=[EMPTY], units=[UNIT])
    for i, (x, y) in enumerate(pairs):
        for v, target in (("a", x), ("b", y)):
            (root / "asm" / v).mkdir(parents=True, exist_ok=True)
            (root / "asm" / v / f"f{i}.s").write_text(f"glabel f{i}\n    jal {target}\n    nop\n    jr $ra\n    nop\n")
    return layout.capture(configuration.load(root, fixture.host(tmp)))


@pytest.mark.usefixtures("toolchains")
def test_correspondences_name_the_two_symbols_of_one_thing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    snapshot = corresponding(tmp_path, [("foo_a", "foo_b")])
    assert infer.correspondences(snapshot) == (("foo_a", "a", "foo_b", "b"),)


@pytest.mark.usefixtures("toolchains")
def test_correspondences_dispatch_no_job_the_second_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real, dispatched = pool.map, []
    monkeypatch.setattr(
        pool,
        "map",
        lambda cfg, name, fn, items, key=None: real(cfg, name, lambda i: dispatched.append(name) or fn(i), items, key),
    )
    monkeypatch.setattr(pool, "_in_worker", True)
    snapshot = corresponding(tmp_path, [("foo_a", "foo_b")])
    infer.correspondences(snapshot)
    dispatched.clear()
    infer.correspondences(snapshot)
    assert "infer.votes" not in dispatched


@pytest.mark.usefixtures("toolchains")
def test_plan_writes_the_layout_and_no_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    snapshot = corresponding(tmp_path, [("foo_a", "foo_b")])
    plan = infer.plan(snapshot)
    assert plan.operation == "layout" and plan.base == snapshot.digest
    assert list(plan.writes) == ["layout.toml"]  # symbols are rows of symbols.toml, never planned here
    assert plan.message == "infer: 1 groups, 0 sdk units, 0 files cut"


@pytest.mark.usefixtures("toolchains")
def test_plan_keeps_a_data_module_and_its_live_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    snapshot = corresponding(tmp_path, [("foo_a", "foo_b")])
    row = Member("rodata/unresolved/80001200", "data", ".rodata", "data_80001200", ())
    module = Group(
        "data_80001200",
        "main",
        (row.name, "claimed/away"),
        "inferred",
        ("adjacent",),
        False,
        "src/data_80001200.c",
        "include/main/data_80001200.h",
    )
    members = {**snapshot.layout.members, row.name: row}
    snapshot = replace(
        snapshot,
        layout=replace(snapshot.layout, members=members, groups={**snapshot.layout.groups, module.name: module}),
    )
    kept = tomllib.loads(infer.plan(snapshot).writes["layout.toml"].decode())["group"]
    assert [g["members"] for g in kept if g["name"] == module.name] == [[row.name]]


@pytest.mark.usefixtures("toolchains")
@pytest.mark.parametrize("rounds", [0, 1, 2])
def test_plan_writes_the_rows_the_claims_give_until_they_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rounds: int
) -> None:
    monkeypatch.setattr(pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    snapshot = corresponding(tmp_path, [("foo_a", "foo_b")])
    calls, seen = [], []
    monkeypatch.setattr(
        infer.layout,
        "claim_rows",
        lambda snap, claims: calls.append(1) or ({"versions/a.yaml": b"rows"} if len(calls) <= rounds else {}),
    )
    monkeypatch.setattr(infer.layout, "overlay", lambda snap, writes: seen.append(sorted(writes)) or snap)
    plan = infer.plan(snapshot)
    assert seen == [["layout.toml", "versions/a.yaml"]] * rounds and len(calls) == rounds + 1
    assert ("versions/a.yaml" in plan.writes) == bool(rounds) and plan.message.endswith(f"{min(rounds, 1)} files cut")


@pytest.mark.usefixtures("toolchains")
def test_plan_refuses_when_the_rows_the_claims_give_move_the_claims_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    snapshot = corresponding(tmp_path, [("foo_a", "foo_b")])
    monkeypatch.setattr(infer.layout, "claim_rows", lambda snap, claims: {"versions/a.yaml": b"rows"})
    monkeypatch.setattr(infer.layout, "overlay", lambda snap, writes: snap)
    with pytest.raises(Refusal) as refused:
        infer.plan(snapshot)
    assert refused.value.findings[0].key == "layout.nonconvergent"
    assert refused.value.findings[0].missing == ("versions/a.yaml",)


def masks(*functions: tuple[int, ...]) -> tuple[list[str], list[bytes]]:
    return [f"f{i}" for i in range(len(functions))], [blob(*words) for words in functions]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        # unique equal code anchors, the function between them pairs by the words they share
        (
            ((1, 2, 3, 4), (10, 11, 12, 13, 14, 15), (5, 6, 7, 8)),
            ((1, 2, 3, 4), (10, 11, 12, 13, 14, 99), (5, 6, 7, 8)),
            {"f0": "f0", "f1": "f1", "f2": "f2"},
        ),
        # identical code that repeats pairs in order between the anchors
        (
            ((1, 2, 3, 4), (9, 9, 9, 9), (9, 9, 9, 9), (5, 6, 7, 8)),
            ((1, 2, 3, 4), (9, 9, 9, 9), (9, 9, 9, 9), (5, 6, 7, 8)),
            {"f0": "f0", "f1": "f1", "f2": "f2", "f3": "f3"},
        ),
        # too little in common: no pairing
        (
            ((1, 2, 3, 4), (10, 11, 12, 13), (5, 6, 7, 8)),
            ((1, 2, 3, 4), (20, 21, 22, 23), (5, 6, 7, 8)),
            {"f0": "f0", "f2": "f2"},
        ),
        # two candidates equally close: no pairing
        (
            ((1, 2, 3, 4), (10, 11, 12, 13), (5, 6, 7, 8)),
            ((1, 2, 3, 4), (10, 11, 12, 90), (10, 11, 12, 91), (5, 6, 7, 8)),
            {"f0": "f0", "f2": "f3"},
        ),
    ],
)
def test_align_pairs_by_anchors_order_and_shared_words(left: tuple, right: tuple, expected: dict) -> None:
    assert infer._align((masks(*left), masks(*right))) == expected


@pytest.mark.usefixtures("toolchains")
def test_scan_records_accesses_calls_argument_accesses_and_steps(tmp_path: Path) -> None:
    body = (
        0x8CAB0004,
        0x24A5000C,  # lw $t3, 4($a1); addiu $a1, $a1, 0xC
        0x3C018010,
        0x24300100,
        0x8E080010,  # lui/addiu $s0 = 0x80100100; lw $t0, 0x10($s0)
        0x3C048011,
        0x24840040,  # $a0 = 0x80110040
        jal(vram(0x1000 + 4 * 15)),
        0x24060100,  # jal f1 with $a2 = 0x100 in the delay slot
        0x2610000C,  # addiu $s0, $s0, 0xC: a pointer stepped by 0xC
        0x3C018012,
        0x8C290020,  # lui $at; lw $t1, 0x20($at): an access with %lo on itself
        JR_RA,
        NOP,
    )
    snapshot = chain(tmp_path, [body, LEAF])
    placement = infer._placement(snapshot, snapshot.layout.members["f0"])
    ((_, loads, callees, _, use),) = infer._scan_job(
        (snapshot, "a", [("f0", placement)], {vram(0x1000 + 60): "f1"}, [])
    )
    assert callees == {"f1"} and not loads
    assert use["accesses"] == [
        (0x80100110, 4, "lw", "based", 0x80100100),
        (0x80110040, 0, "jal", "taken", None),
        (0x80120020, 4, "lw", "abs", None),
    ]
    assert use["calls"] == [("f1", ((0, 0x80110040), (2, 0x100)))]
    assert use["args"] == [(1, 4, 4)]
    assert sorted(use["steps"], key=repr) == [("a1", 12), (0x80100100, 12)]


@pytest.mark.usefixtures("toolchains")
def test_correspondence_votes_survive_unrelated_landing_and_recompute_one_pair(tmp_path, monkeypatch):
    snapshot = corresponding(tmp_path, [("x_a", "x_b"), ("y_a", "y_b")])
    real, ran = pool.map, []
    monkeypatch.setattr(pool, "_in_worker", True)

    def mapped(cfg, name, fn, items, key=None):
        def counted(item):
            if name == "infer.votes":
                ran.append(item[3])
            return fn(item)

        return real(cfg, name, counted, items, key)

    monkeypatch.setattr(pool, "map", mapped)
    assert infer.correspondences(snapshot) == (("x_a", "a", "x_b", "b"), ("y_a", "a", "y_b", "b"))
    assert len(ran) == 2 and all(len(rows) == 2 for rows in ran)  # one chunk of pairs per direction
    ran.clear()
    assert infer.correspondences(replace(snapshot, digest="other landing", commit="other commit"))
    assert ran == []
    path = snapshot.config.project.root / "asm/a/f0.s"
    path.write_text(path.read_text().replace("x_a", "z_a"))
    updated = replace(snapshot, digest="changed assembly")
    assert ("z_a", "a", "x_b", "b") in infer.correspondences(updated)
    assert len(ran) == 2 and all(("f0", "f0") in rows for rows in ran)  # only the chunks holding the changed file


@pytest.mark.usefixtures("toolchains")
def test_correspondence_query_submits_only_member_pairs(tmp_path, monkeypatch):
    snapshot = corresponding(tmp_path, [("x_a", "x_b"), ("y_a", "y_b")])
    seen = []

    def mapped(cfg, name, fn, items, key=None):
        if name == "infer.votes":
            seen.extend(items)
        return [fn(item) for item in items]

    monkeypatch.setattr(pool, "map", mapped)
    assert infer.correspondence(snapshot, "f0") == (("x_a", "a", "x_b", "b"),)
    assert len(seen) == 2 and all(job[3] == [("f0", "f0")] for job in seen)


def test_shape_matches_mnemonics_calls_reference_order_and_literal_immediates():
    text = "/* 0 0 0 */ lui $a0, %hi(first)\n/* 4 4 4 */ lw $a0, %lo(first)($a0)\n"
    text += "/* 8 8 8 */ jal callee\n/* C C C */ addiu $a1, $zero, 8\n"
    shape = infer._shape(text)
    assert shape == infer._shape(text.replace("first", "other"))
    assert shape != infer._shape(text.replace("callee", "different"))
    assert shape != infer._shape(text.replace(", 8", ", 12"))
    assert shape != infer._shape(text.replace(" lw ", " sw "))
    assert infer._shape("/* 0 0 0 */ nop\n") == ""


@pytest.mark.usefixtures("toolchains")
def test_scan_retains_mask_and_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    snapshot = corresponding(tmp_path, [("x_a", "x_b")])
    for version, rows in infer.scan(snapshot).items():
        assert rows[0][4]["shape"] == infer._shape(infer._asm(snapshot, version, "f0").decode())
        assert isinstance(rows[0][4]["mask"], bytes)


@pytest.mark.parametrize(
    "kind, section, holder", [("data", ".data", "a"), ("function", ".text", "b"), ("function", ".text", "a")]
)
def test_address_inference_skips_units_without_text_in_the_holder(cfg, monkeypatch, kind, section, holder):
    from unittest.mock import Mock

    from unbake.contracts import Finding, LayoutMap, Placement, UnitSpec

    unit = UnitSpec("src/data.c", "data", "g", ("datum",), "gcc-test", {})
    member = Member("datum", kind, "data", "g", (Placement(holder, section, 0, 4, 0x80000000),))
    snapshot = Snapshot(
        cfg, "head", LayoutMap({}, {"datum": member}, {unit.path: unit}, "map", (), {}), {"a": Mock()}, {}, "snapshot"
    )
    objects = Mock(side_effect=RuntimeError("broken compiler"))
    monkeypatch.setattr(infer.native, "objects", objects)
    monkeypatch.setattr(infer.recipes, "resolve", lambda *a: None)
    monkeypatch.setattr(infer.symbols, "load", lambda *a: {})
    jobs = []

    def dispatch(config, stage, function, items, key):
        jobs.extend(items)
        return [function(item) for item in items]

    monkeypatch.setattr(infer.pool, "map", dispatch)
    debt = [Finding("symbols.unknown", "symbols.unknown", missing=(unit.path + " a unresolved",))]
    if kind == "function" and holder == "a":
        with pytest.raises(RuntimeError, match="broken compiler"):
            infer.addresses(snapshot, debt)
        assert len(jobs) == 1
    else:
        assert infer.addresses(snapshot, debt) == ({}, [])
        assert jobs == []
        objects.assert_not_called()


def _voting(cfg, monkeypatch, forms, owner_of_85d0):
    """A unit in two versions whose eu code forms 0x85D0 for D_800C85D0 where the table says 0x3790."""
    from unbake.contracts import Finding, LayoutMap, Placement, UnitSpec

    unit = UnitSpec("src/u.c", "c", "g", ("f",), "gcc-test", {})
    member = Member("f", "function", "c", "g", tuple(Placement(v, ".text", 0, 4, 0x80000000) for v in ("de", "eu")))
    snapshot = Snapshot(
        cfg,
        "head",
        LayoutMap({}, {"f": member}, {unit.path: unit}, "map", (), {}),
        {},
        {unit.path: b"int f(void) { return D_800C85D0; }"},
        "snapshot",
    )
    table = {"D_800C85D0": {"de": 0x34E0, "eu": 0x3790, "kind": "data"}}
    if owner_of_85d0:
        table["D_800CCC30"] = {"eu": 0x85D0, "kind": "data"}
    monkeypatch.setattr(infer.symbols, "load", lambda *a: table)
    seen = {"de": forms["de"], "eu": forms["eu"]}
    monkeypatch.setattr(
        infer.pool,
        "map",
        lambda config, stage, function, items, key: [{"D_800C85D0": ({seen[j[2]]}, False)} for j in items],
    )
    return snapshot, [Finding("symbols.unknown", "symbols.unknown", missing=(unit.path + " eu unresolved",))]


def test_addresses_drop_vote_contradicting_other_version(cfg, monkeypatch):
    snapshot, debt = _voting(cfg, monkeypatch, {"de": 0x79E0, "eu": 0x85D0}, True)
    found, findings = infer.addresses(snapshot, debt)
    assert found == {} and [f.key for f in findings] == ["symbols.misnamed"]
    assert "D_800CCC30" in findings[0].reason


def test_addresses_accept_consistent_vote(cfg, monkeypatch):
    snapshot, debt = _voting(cfg, monkeypatch, {"de": 0x34E0, "eu": 0x85D0}, False)
    assert infer.addresses(snapshot, debt) == ({("D_800C85D0", "eu"): (0x85D0, False)}, [])


def test_plan_orphan_unit_takes_predecessor_group() -> None:
    from types import SimpleNamespace

    from unbake.contracts import Placement, UnitSpec

    def member(name, at):
        return Member(name, "function", "c", "gone", (Placement("us", ".text", at, at + 4, 0x80000000 + at),))

    members = {"before": member("before", 0x100), "pad": member("pad", 0x110), "after": member("after", 0x120)}
    layout = SimpleNamespace(members=members, held=lambda names: {"us"})
    snapshot = SimpleNamespace(layout=layout)
    unit = UnitSpec("src/pad.c", "c", "gone", ("pad",), "t", {})
    groups = {
        "g1": Group("g1", "main", ("before",), "inferred", (), False, "src/g1.c", "include/main/g1.h"),
        "g2": Group("g2", "main", ("after",), "inferred", (), False, "src/g2.c", "include/main/g2.h"),
    }
    memberships = {"before": "g1", "after": "g2"}
    assert infer._neighbour_group(snapshot, unit, memberships, groups) == "g1"
    assert infer._neighbour_group(snapshot, unit, {"after": "g2"}, groups) == "g2"
    assert infer._neighbour_group(snapshot, replace(unit, members=("before",)), memberships, groups) == "g1"


@pytest.mark.usefixtures("toolchains")
def test_compile_regression_refuses_before_another_units_claim_rows_are_recaptured(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from unbake.contracts import Claim, Finding

    monkeypatch.setattr(pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    snapshot = corresponding(tmp_path, [("foo_a", "foo_b")])
    failed = Finding("compile.error", "active holder stopped compiling", missing=("src/broken.c a",))
    assigned = {p: replace(u, withheld=("a",)) for p, u in snapshot.layout.units.items()}
    claimed = (Claim("src/other.c", "a", ".rodata", 4, 8, ("new-row",)),)
    claims = Mock(side_effect=[(assigned, claimed, [failed]), (assigned, claimed, [])])
    monkeypatch.setattr(infer.ownership, "claims", claims)
    cuts = Mock(side_effect=[{"versions/a.yaml": b"split another unit data row"}, {}])
    monkeypatch.setattr(infer.layout, "claim_rows", cuts)
    monkeypatch.setattr(infer.layout, "overlay", lambda snap, writes: snap)
    plan = infer._plan(snapshot)
    assert plan.blocking == (failed,) and not plan.writes
    assert claims.call_count == 1
    cuts.assert_not_called()


@pytest.mark.usefixtures("toolchains")
@pytest.mark.parametrize("separate_group", [False, True])
@pytest.mark.parametrize("partial", [None, (0x1010, 0x1014), (0x1018, 0x101C), (0x1FFC, 0x2000)])
def test_infer_reconciles_displaced_provider_in_same_plan_and_is_stable(tmp_path, monkeypatch, separate_group, partial):
    from unbake import ownership

    code = {**UNIT, "path": "src/f0.c", "kind": "c"}
    provider = {**UNIT, "path": "src/provider.c", "kind": "data", "members": ["row"]}
    groups = [module(name="u", members=["f0", "row"])]
    if separate_group:
        provider["group"] = "d"
        groups = [module(name="u", members=["f0"]), module(name="d", members=["row"])]
    snapshot = snap(
        tmp_path,
        functions={"f0": {"a": (0x1000, blob(*LEAF))}},
        data={provider["members"][0]: {"a": (0x1010, bytes(0xFF0))}},
        versions=("a",),
        units=[code, provider],
        groups=groups,
        sources={"f0.c": "void f0(void) {}", "provider.c": "unsigned int data[1020] = {0};"},
    )
    snapshot = layout.overlay(
        snapshot,
        {
            "versions/a/Game.yaml": b"""name: a
segments:
  - name: main
    type: code
    start: 0x1000
    vram: 0x80000400
    subsegments:
      - [0x1000, asm, f0]
      - [0x1010, data, row]
      - [0x2000]
"""
        },
    )
    monkeypatch.setattr(pool, "map", lambda cfg, name, fn, items, key=None: [fn(i) for i in items])
    monkeypatch.setattr(infer, "sdk", lambda s: {})
    monkeypatch.setattr(infer, "groups", lambda s, identified: tuple(s.layout.groups.values()))
    monkeypatch.setattr(
        ownership,
        "_job",
        lambda item: {
            "a": {".data": [list(partial) if partial and item[1].path == code["path"] else [0x1010, 0x2000]]}
        },
    )
    monkeypatch.setattr(ownership, "exact", lambda s, units: (units, []))
    plan = infer._plan(snapshot)
    assert not plan.blocking
    assert "layout.toml" in plan.writes and "versions/a/Game.yaml" in plan.writes
    after = layout.overlay(snapshot, plan.writes)
    if partial:
        assert provider["path"] not in plan.writes
        assert after.read(provider["path"]) == snapshot.read(provider["path"])
        start, end = partial
        residuals = (("row",) if start > 0x1010 else ()) + (
            (f"data/unresolved/{0x80000400 + end - 0x1000:08X}",) if end < 0x2000 else ()
        )
        assert after.layout.units[provider["path"]].members == residuals
        assert after.layout.units[provider["path"]].withheld == ("a",)
        assert after.layout.units[code["path"]].members == ("f0", f"data/f0/{0x80000400 + start - 0x1000:08X}")
    else:
        assert plan.writes[provider["path"]] is None
        assert provider["path"] not in after.layout.units
        assert after.layout.units[code["path"]].members == ("f0", "data/f0/80000410")
    assert not infer._plan(after).writes


@pytest.mark.usefixtures("toolchains")
@pytest.mark.parametrize(
    "change", ["unchanged", "source", "header", "new-header", "source-overlay", "header-overlay", "deleted-header"]
)
def test_plan_reuses_only_the_captured_source_tree(tmp_path, monkeypatch, change):
    from unittest.mock import Mock

    snapshot = snap(
        tmp_path,
        functions={"f0": {"a": (0x1000, blob(*LEAF))}},
        versions=("a",),
        sources={"f0.c": "int f0(void) { return 1; }"},
        headers={"reached.h": "typedef int Value;"},
    )
    fingerprint = infer._tree(snapshot)
    after = snapshot
    root = snapshot.config.project.root
    if change == "source":
        (root / "src/f0.c").write_text("int f0(void) { return 2; }")
    elif change == "header":
        (root / "include/reached.h").write_text("typedef long Value;")
    elif change == "new-header":
        (root / "include/new.h").write_text("typedef int Added;")
    elif change == "deleted-header":
        after = layout.overlay(snapshot, {"include/reached.h": None})
    elif change.endswith("overlay"):
        path = "src/f0.c" if change == "source-overlay" else "include/reached.h"
        after = layout.overlay(snapshot, {path: b"changed overlay bytes"})
    planned = Mock(return_value="plan")
    monkeypatch.setattr(infer, "_plan", planned)
    monkeypatch.setattr(infer.store, "cached_object", lambda cfg, kind, key, produce: produce())
    prior = infer.ownership.LocatedBatch(snapshot, None, fingerprint, (), infer.effort.invocation())
    expected_tree = infer._tree(after)
    tree = Mock(wraps=infer._tree)
    monkeypatch.setattr(infer, "_tree", tree)
    assert infer.plan(after, reuse=prior) == "plan"
    assert planned.call_args.args[:2] == (after, prior if change == "unchanged" else None)
    assert planned.call_args.args[2] == expected_tree
    tree.assert_called_once_with(after)


@pytest.mark.usefixtures("toolchains")
@pytest.mark.parametrize("spelling", ["empty-fuzzy", "comment"])
@pytest.mark.parametrize("first", ["noncanonical", "canonical"])
def test_plan_cache_binds_raw_layout_delta(tmp_path, monkeypatch, spelling, first):
    snapshot = snap(tmp_path, functions={"f0": both(0x1000, blob(*LEAF))}, groups=[module()])
    initial = infer._plan(snapshot)
    snapshot = layout.overlay(snapshot, initial.writes)
    canonical = layout.dump_map(snapshot.layout)
    snapshot = replace(snapshot, overlays={**snapshot.overlays, "layout.toml": canonical})
    assert not infer._plan(snapshot).writes
    raw = (
        canonical.replace(b"schema = 5\n", b"schema = 5\nfuzzy = []\n", 1)
        if spelling == "empty-fuzzy"
        else b"# Equivalent map spelling\n" + canonical
    )
    # These inputs parse to one map, but one still needs a file normalization delta.
    parsed = layout.load_map(snapshot.config, snapshot.versions, raw, snapshot.read)
    assert parsed.digest == layout.load_map(snapshot.config, snapshot.versions, canonical, snapshot.read).digest
    noncanonical = replace(snapshot, overlays={**snapshot.overlays, "layout.toml": raw})
    real = infer.store.cached_object
    cached, produced, requested = {}, [], []

    def cache(cfg, kind, key, produce):
        if kind != "infer":
            return real(cfg, kind, key, produce)
        requested.append(key)
        if key not in cached:
            produced.append(key)
            cached[key] = produce()
        return cached[key]

    monkeypatch.setattr(infer.store, "cached_object", cache)
    order = [noncanonical, snapshot] if first == "noncanonical" else [snapshot, noncanonical]
    for current in [*order, *reversed(order), noncanonical, snapshot]:
        result = infer.plan(current)
        assert result.writes == ({"layout.toml": canonical} if current is noncanonical else {})
    assert len(produced) == 2
    assert len(set(requested)) == 2
    # A semantic edit must also invalidate the plan; replay of that changed input is warm.
    changed_text = canonical.replace(b"schema = 5\n", b'schema = 5\nboundaries = ["f0"]\n', 1)
    changed = layout.overlay(snapshot, {"layout.toml": changed_text})
    assert changed.digest != snapshot.digest
    result = infer.plan(changed)
    assert result.base == changed.digest
    assert infer.plan(changed) == result
    assert len(produced) == 3
