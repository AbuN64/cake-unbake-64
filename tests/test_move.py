"""move: JSON input, map edits, unit and header text, the all-or-nothing proof; every boundary mocked and counted."""

from __future__ import annotations

import dataclasses
import fcntl
import io
import json
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fixture
import pytest
from pycparser import CParser

from unbake import config as configuration
from unbake import effort, headers, layout, move, moved, store
from unbake.contracts import Claim, Finding, Move, Refusal, SourceView

_FOLD = headers.fold  # the real fold, kept before a fixture replaces it

CODE = bytes.fromhex("240200012403000203e0000800000000")
Patch = pytest.MonkeyPatch


def both(offset: int, version: tuple[str, ...] = ("a", "b")) -> dict[str, tuple[int, bytes]]:
    return {v: (offset, CODE) for v in version}


def group(name: str, *members: str, **kw: Any) -> dict[str, Any]:
    row = {
        "name": name,
        "segment": "main",
        "members": list(members),
        "evidence": "authored",
        "signals": [],
        "subsystem": "",
        "library": "",
    }
    return row | {"source": f"src/{name}.c", "header": f"include/main/{name}.h"} | kw


def unit(path: str, owner: str, *members: str, toolchain: str = "gcc-test") -> dict[str, Any]:
    return {
        "path": path,
        "kind": "c",
        "group": owner,
        "members": list(members),
        "toolchain": toolchain,
        "options": {"add": [], "omit": []},
    }


def body(name: str, extra: str = "") -> str:
    return f'#include "main/g{"abc"[int(name[1]) - 1]}.h"\n{extra}int {name}(void) {{ return 1; }}\n'


def world(tmp: Path, extra: str = "", **kw: Any) -> Any:
    """A project of three one-function modules ga, gb, gc (f1, f2, f3 side by side), loaded for real."""
    kw.setdefault("functions", {"f1": both(0x1000), "f2": both(0x1010), "f3": both(0x1020)})
    kw.setdefault(
        "groups",
        [
            group("ga", "f1", source="src/f1.c"),
            group("gb", "f2", source="src/f2.c"),
            group("gc", "f3", source="src/f3.c"),
        ],
    )
    kw.setdefault("units", [unit("src/f1.c", "ga", "f1"), unit("src/f2.c", "gb", "f2"), unit("src/f3.c", "gc", "f3")])
    kw.setdefault("sources", {f"{n}.c": body(n) for n in ("f1", "f2", "f3")})
    kw.setdefault(
        "headers",
        {f"main/g{g}.h": f"#ifndef G{g}\n#define G{g}\nvoid f{i}(void);\n#endif\n" for i, g in enumerate("abc", 1)},
    )
    root = fixture.project(tmp, **kw)
    if extra:
        path = root / "config.toml"
        path.write_text(path.read_text().replace("[project]\n", "[project]\n" + extra, 1))
    return configuration.load(root, fixture.host(tmp))


def splat_rows(cfg: Any) -> None:
    """Rewrite the fixture's JSON split files in the flow-row form splat writes, which the row editor reads."""
    for version in cfg.project.versions:
        path = cfg.project.root / cfg.project.version_files[version].split
        doc = json.loads(path.read_text())
        segment = doc["segments"][0]
        rows = "".join(
            f"      - [0x{r[0]:X}, {r[1]}, {r[2]}]\n" if len(r) == 3 else f"      - [0x{r[0]:X}]\n"
            for r in segment["subsegments"]
        )
        head = f"name: {doc['name']}\nsha1: {doc['sha1']}\nsegments:\n  - name: main\n    type: code\n"
        start = f"    start: 0x{segment['start']:X}\n    vram: 0x{segment['vram']:X}\n    subsegments:\n"
        path.write_text(head + start + rows)


@pytest.fixture
def seen(monkeypatch: Patch, toolchains: Any, warm: Any) -> SimpleNamespace:
    """Every boundary the engine crosses, replaced and counted; its pool dispatches run on warm workers."""
    w = SimpleNamespace(measured=[], plans=[], extracted=[], files=0, claims=0, exact=True, folded=[])

    def proofs(snapshot: Any, affected: Any) -> Any:
        w.measured.append(sorted(affected))
        units = [snapshot.layout.units[p] for p in affected]
        made = tuple(
            fixture.proof(u.path, m, v, w.exact, () if w.exact else ("size",))
            for u in units
            for m in u.members
            for v in snapshot.layout.members[m].holders()
        )
        return made, tuple(g for u in units for g in move.compare.gaps(snapshot, u, made))

    def generated(snapshot: Any) -> dict[str, bytes]:
        w.files += 1
        return {f"versions/{v}/symbols.ld": b"ld" for v in snapshot.config.project.versions}

    def claims(snapshot: Any, *args: Any, **kw: Any) -> Any:
        w.claims += 1
        return {}, (), []

    def apply(config: Any, plan: Any, snapshot: Any) -> tuple[str, ...]:
        w.plans.append(plan)
        return tuple(sorted(plan.writes))

    def fold(snapshot: Any, u: Any, view: Any, strict: bool = True) -> Any:
        w.folded.append(u.path)
        return {u.path: snapshot.read(u.path)}, ()

    monkeypatch.setattr(move.publish, "_proofs", proofs)
    monkeypatch.setattr(move.repo, "files", generated)
    monkeypatch.setattr(move.ownership, "claims", claims)
    monkeypatch.setattr(move.policy, "census", lambda s, **kw: ((), []))
    monkeypatch.setattr(move.journal, "apply", apply)
    monkeypatch.setattr(move.build, "extract", lambda config, version: w.extracted.append(version))
    monkeypatch.setattr(move.headers, "fold", fold)
    w.views = []
    monkeypatch.setattr(move.view, "get", lambda *a, **k: (w.views.append(a[1].path), _view())[1])
    monkeypatch.setattr(move.view, "stored", lambda config, unit, version, key: _view())  # a view get kept
    monkeypatch.setattr(move.recipes, "resolve", lambda *a: None)
    w.probed, w.broken = [], lambda snapshot, header: []  # a header compiled alone: its error lines

    def probe(item: Any) -> dict[str, Any]:
        snapshot, _, _, header, _ = item
        w.probed.append(header)
        first = w.broken(snapshot, header)
        return {"header": header, "errors": len(first), "first": first}

    _STUBS["probe"] = probe
    monkeypatch.setattr(move.headers, "_probe_job", _probe_stub)  # module level: the warm workers unpickle it
    return w


_STUBS: dict[str, Any] = {}  # the fakes a `seen` test's jobs call: its workers run in this process


def _probe_stub(item: Any) -> dict[str, Any]:
    return _STUBS["probe"](item)


def _view(dependencies: tuple[tuple[str, str], ...] = ()) -> SourceView:
    return SourceView("src/f1.c", "a", "key", "", dependencies, ())


def go(cfg: Any, op: str, subject: str = "", target: str = "") -> Any:
    return move.run(cfg, {"op": op, "subject": subject, "target": target})


def stdin(monkeypatch: Patch, text: str) -> None:
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(text.encode())))


def text(plan: Any, path: str) -> str:
    return plan.writes[path].decode()


# ------------------------------------------------------------------ read


def test_the_argument_form_a_json_array_and_json_lines_give_equal_moves() -> None:
    moves = [{"op": "member", "subject": "f1", "target": "ga"}, {"op": "module", "subject": "ga", "target": "gz"}]
    one = move.read({"op": "member", "subject": "f1", "target": "ga"}, b"")
    array = move.read({"op": "-"}, json.dumps(moves).encode())
    lines = move.read({"op": "-"}, "\n".join(json.dumps(m) for m in moves).encode())
    assert array == lines == (one[0], Move(1, "module", "ga", "gz"))
    assert move.read({"op": "-"}, b"") == ()


@pytest.mark.parametrize(
    ("params", "data", "line"),
    [
        ({"op": "member", "subject": "f1"}, b"", 0),
        ({"op": "-"}, b"{not json", 1),
        ({"op": "-"}, b'[{"op": "rename", "subject": "a", "target": "b"}]', 0),
        (
            {"op": "-"},
            b'[{"op": "file", "subject": "a", "target": "b"}, {"op": "file", "subject": "a", "target": "c", "x": 1}]',
            1,
        ),
        ({"op": "-"}, b'{"op": "file", "subject": "a"}', 0),
        (
            {"op": "-"},
            b'{"op": "file", "subject": "a", "target": "b"}\n{"op": "file", "subject": "a", "target": "c"}',
            1,
        ),
    ],
)
def test_malformed_input_refuses_whole_with_the_index(params: Any, data: bytes, line: int) -> None:
    with pytest.raises(Refusal) as error:
        move.read(params, data)
    assert [(f.key, f.line) for f in error.value.findings] == [("move.syntax", line)]


# ------------------------------------------------------------------ symbol


def test_a_symbol_move_writes_every_place_the_name_is_spelled_and_extracts_each_changed_version(
    tmp_path: Path, seen: Any, warm: Any
) -> None:
    extra = '[[project.version_differences]]\nfunction = "f1"\nsides = [["a"], ["b"]]\n'
    cfg = world(tmp_path, extra="", data={"f1_padding_1": both(0x1110)})
    config = cfg.project.root / "config.toml"
    config.write_text(config.read_text() + "\n" + extra)
    (cfg.project.root / "types.toml").write_text("schema = 1\n\n[function]\nf1 = 1\n\n[global]\n\n[struct]\n")
    splat_rows(cfg)
    cfg = configuration.load(cfg.project.root, fixture.host(tmp_path))
    result = go(cfg, "symbol", "f1", "g1")
    plan = seen.plans[0]
    assert set(plan.writes) == {
        "symbols.toml", "versions/a/symbol_addrs.txt", "versions/b/symbol_addrs.txt", "versions/a/symbols.ld",
        "versions/b/symbols.ld", "src/f1.c", "include/main/ga.h", "types.toml", "layout.toml", "versions/a/Game.yaml",
        "versions/b/Game.yaml", "config.toml", "README.md", "progress.json", "docs/memory-map-a.svg",
        "docs/memory-map-b.svg", "docs/overview-a.svg", "docs/overview-b.svg",
        "versions/a/symbols.json", "versions/b/symbols.json",
    }  # fmt: skip
    assert " bytes</code>" in text(plan, "README.md")  # the progress block of the moved tree, committed with the move
    assert '"g1":' in text(plan, "versions/a/symbols.json")  # the site links the new name to its line
    assert "g1]" in text(plan, "versions/a/Game.yaml") and "f1_padding_1" in text(plan, "versions/a/Game.yaml")
    assert 'function = "g1"' in text(plan, "config.toml") and "int g1(void)" in text(plan, "src/f1.c")
    assert '"g1"' in text(plan, "layout.toml") and sorted(seen.extracted) == ["a", "b"]
    # one warm pool ran the views, folds and probes: forked once, it laid the move's overlays as they changed
    assert warm.forks == 1 and warm.dispatches >= 2 and len(warm.laid) >= 2
    assert result["applied"]["symbol"] == 1 and result["extracted"] == ["a", "b"] and seen.measured == [["src/f1.c"]]


def test_a_symbol_rename_follows_a_functions_data_rows_and_leaves_its_padding_alone() -> None:
    renames = {"f1": "g1"}
    assert layout.renamed("rodata/f1/80001000", renames) == "rodata/g1/80001000"
    assert layout.renamed("f1_padding_1", renames) == "f1_padding_1" and layout.renamed("f1", renames) == "g1"


def test_a_symbol_that_exists_or_is_no_identifier_is_refused_with_nothing_written(tmp_path: Path, seen: Any) -> None:
    cfg = world(tmp_path)
    for target, key in (("f2", "move.exists"), ("1bad", "move.name")):
        refused = go(cfg, "symbol", "f1", target)["refused"]
        assert [f["key"] for f in refused[0]["findings"]] == [key] and refused[0]["index"] == 0
    assert go(cfg, "symbol", "nope", "x")["refused"][0]["findings"][0]["key"] == "move.unknown"
    assert seen.plans == [] and seen.measured == []


# ------------------------------------------------------------------ member and units


def test_a_member_move_joins_two_contiguous_units_into_one_at_the_module_source(tmp_path: Path, seen: Any) -> None:
    sources = {
        "f1.c": body("f1", '#include "zed.h"\n#include "shared.h"\n'),
        "f2.c": body("f2", '#include "other.h"\n#include "shared.h"\n'),
        "f3.c": body("f3"),
    }
    cfg = world(tmp_path, sources=sources)
    result = go(cfg, "member", "f2", "ga")
    plan = seen.plans[0]
    assert plan.writes["src/f2.c"] is None and "src/f2.c" in result["deleted"]
    merged = text(plan, "src/f1.c").splitlines()
    assert merged[:4] == [
        '#include "main/ga.h"',
        '#include "other.h"',
        '#include "shared.h"',
        '#include "zed.h"',
    ]  # own first, sorted
    assert merged.index("int f1(void) { return 1; }") < merged.index("int f2(void) { return 1; }")
    assert result["units"] == {"before": 3, "after": 2} and seen.measured == [["src/f1.c"]]
    assert "void f2(void);" in text(plan, "include/main/ga.h") and plan.writes["include/main/gb.h"] is None
    assert seen.folded == ["src/f1.c"] and seen.files == 1 and seen.claims == 2


def test_a_merged_unit_writes_a_forward_typedef_both_old_units_make_once(tmp_path: Path, seen: Any) -> None:
    forward = "typedef struct Owner Owner;\n"
    sources = {"f1.c": body("f1", forward), "f2.c": body("f2", forward), "f3.c": body("f3")}
    go(world(tmp_path, sources=sources), "member", "f2", "ga")
    assert text(seen.plans[0], "src/f1.c").count("typedef struct Owner Owner;") == 1


def test_a_merged_unit_keeps_each_non_member_definition_beside_the_member_it_followed(
    tmp_path: Path, seen: Any
) -> None:
    sources = {
        "f1.c": body("f1", "static inline int twice(int x) { return x + x; }\n") + "int tail(void) { return 2; }\n",
        "f2.c": body("f2", "int lead = 4;\n"),
        "f3.c": body("f3"),
    }
    go(world(tmp_path, sources=sources), "member", "f2", "ga")
    merged = text(seen.plans[0], "src/f1.c")
    order = ["static inline int twice", "int f1(void)", "int tail(void)", "int lead = 4;", "int f2(void)"]
    assert [merged.index(s) for s in order] == sorted(merged.index(s) for s in order)  # code and data keep their order


def test_a_split_unit_gives_an_unread_data_definition_to_the_unit_holding_its_data_rows(
    tmp_path: Path, seen: Any
) -> None:
    pad = "const int pad[1] = { 7 };\n"  # retained words beside the data rows: no function reads them
    sources = {"f1.c": body("f1", pad) + "int f2(void) { return 1; }\n", "a3.c": body("f3")}
    cfg = world(
        tmp_path,
        sources=sources,
        functions={"f3": both(0x1000), "f1": both(0x1010), "f2": both(0x1020)},
        data={"d2": both(0x1100)},
        groups=[group("ga", "f1", "f2", "d2", source="src/f1.c"), group("gc", "f3", source="src/a3.c")],
        units=[unit("src/a3.c", "gc", "f3"), unit("src/f1.c", "ga", "f1", "f2", "d2")],
        headers={"main/ga.h": "void f1(void);\nvoid f2(void);\n", "main/gc.h": "void f3(void);\n"},
    )
    snapshot = layout.capture(cfg)
    old = snapshot.layout.units
    final = dataclasses.replace(
        snapshot.layout,
        units={  # the unit without the data rows comes first: the one an unread block fell to
            "src/a3.c": dataclasses.replace(old["src/a3.c"], members=("f3", "f1")),
            "src/f1.c": dataclasses.replace(old["src/f1.c"], members=("f2", "d2")),
        },
    )
    texts, _, problems = layout.compose(snapshot, final, {}, lambda p: True)
    assert problems == [] and "int f1(void)" in texts["src/a3.c"].decode() and pad not in texts["src/a3.c"].decode()
    assert pad in texts["src/f1.c"].decode()  # its bytes sit among the data rows: they go where the rows go


@pytest.mark.parametrize(
    ("target", "refused"),
    [
        ("include/game/main/actor.h", False),
        ("include/types.h", False),
        ("include/main/free.h", False),  # bound to no module
        ("include/main/gb.h", True),  # another module's header
    ],
)
def test_a_header_outside_the_subsystem_set_is_refused(tmp_path: Path, seen: Any, target: str, refused: bool) -> None:
    groups = [
        group("ga", "f1", source="src/f1.c", subsystem="main/actor"),
        group("gb", "f2", source="src/f2.c"),
        group("gc", "f3", source="src/f3.c"),
    ]
    cfg = world(tmp_path, groups=groups)
    result = go(cfg, "header", "ga", target)
    keys = [f["key"] for r in result["refused"] for f in r["findings"]]
    assert keys == (["move.name"] if refused else []) and len(seen.plans) == (not refused)


@pytest.mark.parametrize(
    ("target", "refused"), [("include/types.h", False), ("include/main/free.h", False), ("include/main/gb.h", True)]
)
def test_an_unplaced_module_takes_only_a_shared_or_unbound_header(
    tmp_path: Path, seen: Any, target: str, refused: bool
) -> None:
    cfg = world(tmp_path)  # no module has a subsystem or library
    result = go(cfg, "header", "ga", target)
    keys = [f["key"] for r in result["refused"] for f in r["findings"]]
    assert keys == (["move.name"] if refused else []) and len(seen.plans) == (not refused)


@pytest.mark.parametrize(
    ("source", "placed"),
    [
        ("src/sdk/libgultra/getsr.c", ("", "libgultra")),
        ("src/game/main/actor/x.c", ("main/actor", "")),
        ("src/f1.c", ("", "")),
    ],
)
def test_a_source_path_places_its_module(source: str, placed: tuple[str, str]) -> None:
    assert move._placement(source) == placed


def test_every_module_of_a_library_takes_the_library_header_not_only_the_first(tmp_path: Path, seen: Any) -> None:
    groups = [  # both sit in src/sdk/libx by source; neither row names the library, as after `source` moves
        group("ga", "f1", source="src/sdk/libx/f1.c", header="include/sdk/libx.h"),
        group("gb", "f2", source="src/sdk/libx/f2.c"),
        group("gc", "f3", source="src/f3.c"),
    ]
    units = [unit("src/sdk/libx/f1.c", "ga", "f1"), unit("src/sdk/libx/f2.c", "gb", "f2"), unit("src/f3.c", "gc", "f3")]
    sources = {"sdk/libx/f1.c": body("f1"), "sdk/libx/f2.c": body("f2"), "f3.c": body("f3")}
    cfg = world(tmp_path, groups=groups, units=units, sources=sources)
    result = go(cfg, "header", "gb", "include/sdk/libx.h")
    assert result["refused"] == [] and len(seen.plans) == 1
    result = go(world(tmp_path / "other"), "header", "gb", "include/main/ga.h")  # unplaced: still refused
    assert [f["key"] for r in result["refused"] for f in r["findings"]] == ["move.name"]


def test_a_source_move_records_the_library_its_path_names(tmp_path: Path, seen: Any) -> None:
    result = go(world(tmp_path), "source", "ga", "src/sdk/libx/f1.c")
    assert result["refused"] == [] and len(seen.plans) == 1
    rows = text(seen.plans[0], "layout.toml")
    assert 'library = "libx"' in rows


def test_an_edited_map_lays_again_in_a_worker_to_the_overlay_the_parent_made(tmp_path: Path, seen: Any) -> None:
    cfg = world(tmp_path)
    snap = move.layout.capture(cfg)
    edited = move.edit(snap, [Move(0, "source", "ga", "src/sdk/libx/f1.c")])  # places ga in library libx
    parent = move.layout.overlay(snap, edited.writes)  # the parent reads the map it dumped from memory
    move.layout._dumped.clear()  # a worker forked before the edit parses the map's text instead
    move.effort.forget("overlay", "load-map")
    again = move.layout.overlay(snap, edited.writes)
    assert again.digest == parent.digest  # not "an overlay laid again in a worker has digest ..."
    assert parent.layout.groups["ga"].sdk and again.layout.groups["ga"].sdk  # library code is sdk, as a load says


def test_a_shared_header_moves_into_no_other_and_a_module_bound_to_it_takes_only_its_names(
    tmp_path: Path, seen: Any
) -> None:
    base = "#ifndef TYPES_H\n#define TYPES_H\ntypedef int s32;\nint f1(void);\n#endif\n"
    groups = [group("ga", "f1", source="src/f1.c", header="include/types.h"), group("gb", "f2", source="src/f2.c")]
    units = [unit("src/f1.c", "ga", "f1"), unit("src/f2.c", "gb", "f2")]
    sources = {"f1.c": '#include "types.h"\nint f1(void) { return 1; }\n', "f2.c": body("f2")}
    cfg = world(tmp_path, groups=groups, units=units, headers={"types.h": base}, sources=sources)
    result = go(cfg, "file", "include/types.h", "include/game/common.h")
    assert [f["key"] for r in result["refused"] for f in r["findings"]] == ["move.shared"]
    snap = move.layout.capture(cfg)
    edited = move.edit(snap, [Move(0, "header", "ga", "include/game/common.h")])
    assert [(k, o, n) for k, o, n, _ in edited.events] == [("names", "include/types.h", "include/game/common.h")]


def test_relayout_sets_the_units_the_groups_lay_out_once_and_a_move_after_it_leaves_other_modules_units(
    tmp_path: Path, seen: Any
) -> None:
    groups = [group("ga", "f1", "f2", source="src/f1.c"), group("gc", "f3", source="src/f3.c")]
    units = [unit("src/f1.c", "ga", "f1"), unit("src/f2.c", "ga", "f2"), unit("src/f3.c", "gc", "f3")]
    sources = {"f1.c": body("f1"), "f2.c": body("f1").replace("f1", "f2"), "f3.c": body("f3")}
    cfg = world(tmp_path, groups=groups, units=units, sources=sources)
    result = go(cfg, "relayout")
    assert result["refused"] == [] and result["applied"]["relayout"] == 1
    assert result["units"] == {"before": 3, "after": 2}
    plan = seen.plans[-1]
    assert plan.writes["src/f2.c"] is None and "int f2(void)" in text(plan, "src/f1.c")
    after = layout.overlay(move.layout.capture(cfg), plan.writes)
    assert {p: u.members for p, u in after.layout.units.items()} == {"src/f1.c": ("f1", "f2"), "src/f3.c": ("f3",)}
    again = move.edit(after, [Move(0, "relayout", "", "")])
    assert again.layout.units == after.layout.units and "layout.toml" not in again.writes  # a fixed point
    moved = move.edit(after, [Move(0, "source", "gc", "src/game/f3.c")])  # touches gc only: ga keeps its unit
    assert {p: u.members for p, u in moved.layout.units.items()} == {"src/f1.c": ("f1", "f2"), "src/game/f3.c": ("f3",)}


def test_a_new_module_header_stands_alone_with_the_old_header_includes(tmp_path: Path, seen: Any) -> None:
    types = "#ifndef T\n#define T\ntypedef int s32;\n#endif\n"
    old = '#ifndef GA\n#define GA\n#include "shared/t.h"\n#include "near.h"\ns32 f1(void);\ns32 f2(void);\n#endif\n'
    headers = {"shared/t.h": types, "main/near.h": "#define NEAR 1\n", "main/ga.h": old}
    groups = [group("ga", "f1", "f2", source="src/f1.c"), group("gc", "f3", source="src/f3.c")]
    units = [unit("src/f1.c", "ga", "f1", "f2"), unit("src/f3.c", "gc", "f3")]
    two = '#include "main/ga.h"\ns32 f1(void) { return 1; }\ns32 f2(void) { return 2; }\n'
    cfg = world(tmp_path, groups=groups, units=units, headers=headers, sources={"f1.c": two, "f3.c": body("f3")})
    result = go(cfg, "member", "f2", "gnew")
    assert result["refused"] == [] and len(seen.plans) == 1
    new = text(seen.plans[0], "include/gnew.h")
    assert new.startswith("#ifndef GNEW_H\n#define GNEW_H\n") and "s32 f2(void);" in new
    assert '#include "shared/t.h"' in new and '#include "main/near.h"' in new  # a sibling is spelled from include/


def test_a_new_module_whose_header_moves_into_a_library_header_includes_no_vacated_header(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    types = "#ifndef T\n#define T\ntypedef int s32;\n#endif\n"
    old = '#ifndef GA\n#define GA\n#include "shared/t.h"\ns32 f1(void);\ns32 f2(void);\n#endif\n'
    lib = '#ifndef LIBX\n#define LIBX\n#include "main/ga.h"\ns32 f3(void);\n#endif\n'
    headers = {"shared/t.h": types, "main/ga.h": old, "sdk/libx.h": lib}
    groups = [
        group("ga", "f1", "f2", source="src/f1.c"),
        group("gc", "f3", source="src/sdk/libx/f3.c", header="include/sdk/libx.h"),
    ]
    units = [unit("src/f1.c", "ga", "f1", "f2"), unit("src/sdk/libx/f3.c", "gc", "f3")]
    two = '#include "main/ga.h"\ns32 f1(void) { return 1; }\ns32 f2(void) { return f3(); }\n'
    f3 = '#include "sdk/libx.h"\ns32 f3(void) { return f2(); }\n'
    cfg = world(tmp_path, groups=groups, units=units, headers=headers, sources={"f1.c": two, "sdk/libx/f3.c": f3})
    moves = [
        {"op": "member", "subject": "f2", "target": "gnew"},  # makes include/gnew.h
        {"op": "source", "subject": "gnew", "target": "src/sdk/libx/gnew.c"},
        {"op": "header", "subject": "gnew", "target": "include/sdk/libx.h"},  # and vacates it again
    ]
    stdin(monkeypatch, "\n".join(json.dumps(m) for m in moves))
    result = move.run(cfg, {"op": "-"})
    assert result["refused"] == [] and len(seen.plans) == 1
    writes = {p: d.decode() for p, d in seen.plans[0].writes.items() if d and p.endswith((".c", ".h"))}
    assert "include/gnew.h" not in seen.plans[0].writes or seen.plans[0].writes["include/gnew.h"] is None
    assert not any("gnew.h" in t for t in writes.values())
    assert '#include "sdk/libx.h"' not in writes.get("include/sdk/libx.h", "")  # never itself
    assert "s32 f2(void);" in writes["include/sdk/libx.h"]


def test_a_foreign_row_or_a_difference_makes_more_units_and_never_refuses(tmp_path: Path, seen: Any) -> None:
    holes = {"f1": both(0x1000), "hole": both(0x1010), "f2": both(0x1020), "f3": both(0x1030)}
    cfg = world(tmp_path, functions=holes)
    result = go(cfg, "member", "f2", "ga")  # an asm row sits between f1 and f2
    # f1.c names nothing the header change touched (f2's prototype joins ga.h): only the new unit is proved
    assert result["units"] == {"before": 3, "after": 3} and seen.measured == [["src/f1_2.c"]]
    assert text(seen.plans[0], "layout.toml").count('path = "src/f1') == 2
    seen.plans.clear()
    held = world(tmp_path / "h", functions={"f1": both(0x1000), "f2": both(0x1010, ("a",)), "f3": both(0x1020)})
    assert go(held, "member", "f2", "ga")["units"]["after"] == 3 and not seen.plans[0].blocking  # other holders


def test_a_toolchain_difference_splits_the_unit_it_would_join(tmp_path: Path, seen: Any) -> None:
    other = [
        unit("src/f1.c", "ga", "f1"),
        unit("src/f2.c", "gb", "f2", toolchain="other"),
        unit("src/f3.c", "gc", "f3"),
    ]
    assert go(world(tmp_path, units=other), "member", "f2", "ga")["units"]["after"] == 3


def test_a_new_module_is_made_with_its_source_and_header(tmp_path: Path, seen: Any) -> None:
    result = go(world(tmp_path), "member", "f3", "gnew")
    plan = seen.plans[0]
    assert "src/gnew.c" in plan.writes and "void f3(void);" in text(plan, "include/gnew.h")
    assert 'name = "gnew"' in text(plan, "layout.toml") and result["units"]["after"] == 3


def test_source_and_module_moves_rename_the_unit_file_and_merge_modules(tmp_path: Path, seen: Any) -> None:
    cfg = world(tmp_path)
    go(cfg, "source", "ga", "src/lib/a.c")
    assert seen.plans[0].writes["src/f1.c"] is None and b"int f1" in seen.plans[0].writes["src/lib/a.c"]
    go(cfg, "module", "gb", "ga")
    merged = seen.plans[1]
    assert 'name = "gb"' not in text(merged, "layout.toml") and merged.writes["include/main/gb.h"] is None
    assert "void f2(void);" in text(merged, "include/main/ga.h")


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize("what", ["static", "directive", "library"])
def test_refusals_of_the_text_and_the_library_write_nothing(tmp_path: Path, seen: Any, what: str) -> None:
    include = '#include "main/ga.h"\n'
    shared = {
        "static": include + "static int s;\nint f1(void) { return s; }\nint f2(void) { return s; }\n",
        "directive": include + "#ifdef X\nint f1(void) { return 1; }\nint f2(void) { return 1; }\n#endif\n",
    }
    folder = "lib/" if what == "library" else ""
    cfg = world(
        tmp_path,
        'library_dirs = ["src/lib"]\n' if what == "library" else "",
        groups=[group("ga", "f1", "f2", source=f"src/{folder}f1.c"), group("gc", "f3", source="src/f3.c")],
        units=[unit(f"src/{folder}f1.c", "ga", "f1", "f2"), unit("src/f3.c", "gc", "f3")],
        sources={f"{folder}f1.c": shared.get(what, body("f1")), "f3.c": body("f3")},
    )
    result = go(cfg, "member", "f2", "gc")
    assert [f["key"] for f in result["refused"][0]["findings"]] == [f"move.{what}"]
    assert seen.plans == [] and seen.measured == [] and result["files"] == []


def _split_with(tmp_path: Path, helper: str, members: tuple[str, ...] = ("f1", "f2")) -> Any:
    """ga holds f1 and f2 that both call helper; moving f2 to gc splits the file in two."""
    shared = (
        '#include "main/ga.h"\n' + helper + "int f1(void) { return helper(1); }\nint f2(void) { return helper(2); }\n"
    )
    return world(
        tmp_path,
        groups=[group("ga", *members, source="src/f1.c"), group("gc", "f3", source="src/f3.c")],
        units=[unit("src/f1.c", "ga", *members), unit("src/f3.c", "gc", "f3")],
        sources={"f1.c": shared, "f3.c": body("f3")},
    )


def _texts(plan: Any) -> dict[str, str]:
    return {p: d.decode() for p, d in plan.writes.items() if p.endswith(".c") and d and b"helper" in d}


def test_a_static_inline_function_used_by_both_halves_is_copied_to_each(tmp_path: Path, seen: Any) -> None:
    result = go(_split_with(tmp_path, "static inline int helper(int x) { return x + 1; }\n"), "member", "f2", "gc")
    assert result["refused"] == [] and len(seen.plans) == 1
    texts = _texts(seen.plans[0])
    assert sorted(t.count("static inline int helper(int x) {") for t in texts.values()) == [1, 1]
    assert not any("extern" in t for t in texts.values())


def test_a_plain_static_function_used_by_both_halves_keeps_one_copy_the_other_calls_it(
    tmp_path: Path, seen: Any
) -> None:
    helper = "static int helper(int x);\nstatic int helper(int x) { return x + 1; }\n"
    result = go(_split_with(tmp_path, helper), "member", "f2", "gc")
    assert result["refused"] == [] and len(seen.plans) == 1
    texts = _texts(seen.plans[0])
    owner = [p for p, t in texts.items() if "{ return x + 1; }" in t]
    assert owner == ["src/f1.c"]  # no member before the bytes: they sit before f1, the member after them
    assert not any("static" in t for t in texts.values())  # the forward declaration loses it too
    assert "int helper(int x);" in texts["src/f1.c"]
    (caller,) = set(texts) - {"src/f1.c"}
    assert "extern int helper(int x);" in texts[caller] and "x + 1" not in texts[caller]
    assert seen.folded[0] == "src/f1.c"  # the owner folds its prototype into its header before the caller


def test_a_member_static_function_another_half_calls_becomes_external(tmp_path: Path, seen: Any) -> None:
    funcs = {"f1": both(0x1000), "helper": both(0x1010), "f2": both(0x1020), "f3": both(0x1030)}
    shared = (
        '#include "main/ga.h"\nint f1(void) { return helper(1); }\nstatic int helper(int x) { return x + 1; }\n'
        "int f2(void) { return helper(2); }\n"
    )
    cfg = world(
        tmp_path,
        functions=funcs,
        groups=[group("ga", "f1", "helper", "f2", source="src/f1.c"), group("gc", "f3", source="src/f3.c")],
        units=[unit("src/f1.c", "ga", "f1", "helper", "f2"), unit("src/f3.c", "gc", "f3")],
        sources={"f1.c": shared, "f3.c": body("f3")},
    )
    result = go(cfg, "member", "f2", "gc")
    assert result["refused"] == [] and len(seen.plans) == 1
    texts = _texts(seen.plans[0])
    assert [p for p, t in texts.items() if "x + 1" in t] == ["src/f1.c"]
    assert "static" not in texts["src/f1.c"]
    (caller,) = set(texts) - {"src/f1.c"}
    assert "extern int helper(int x);" in texts[caller]


def test_a_function_made_external_that_another_source_defines_is_refused_as_static(tmp_path: Path, seen: Any) -> None:
    cfg = _split_with(tmp_path, "static int helper(int x) { return x + 1; }\n")
    f3 = cfg.project.root / "src/f3.c"
    f3.write_text(f3.read_text() + "static int helper(int x) { return x; }\n")  # would meet the prototype
    result = go(cfg, "member", "f2", "gc")
    (finding,) = result["refused"][0]["findings"]
    assert finding["key"] == "move.static" and "helper" in finding["reason"] and "src/f3.c" in finding["reason"]
    assert seen.plans == [] and seen.measured == []


def test_an_inline_member_another_half_calls_is_refused_as_static(tmp_path: Path, seen: Any) -> None:
    funcs = {"f1": both(0x1000), "helper": both(0x1010), "f2": both(0x1020), "f3": both(0x1030)}
    shared = (
        '#include "main/ga.h"\nint f1(void) { return helper(1); }\n'
        "static inline int helper(int x) { return x + 1; }\nint f2(void) { return helper(2); }\n"
    )
    cfg = world(
        tmp_path,
        functions=funcs,
        groups=[group("ga", "f1", "helper", "f2", source="src/f1.c"), group("gc", "f3", source="src/f3.c")],
        units=[unit("src/f1.c", "ga", "f1", "helper", "f2"), unit("src/f3.c", "gc", "f3")],
        sources={"f1.c": shared, "f3.c": body("f3")},
    )
    result = go(cfg, "member", "f2", "gc")
    assert [f["key"] for f in result["refused"][0]["findings"]] == ["move.static"] and seen.plans == []


# ------------------------------------------------------------------ headers


def test_a_header_move_rewrites_the_include_in_every_includer_and_the_guard(tmp_path: Path, seen: Any) -> None:
    sources = {
        "f1.c": body("f1"),
        "f2.c": body("f2", '#include "main/gz.h"\n'),
        "f3.c": body("f3"),
        "fuzzy/q.c": body("f1"),
    }
    headers = {
        "main/ga.h": "#ifndef GA\n#define GA\nvoid f1(void);\n#endif\n",
        "main/gb.h": "#ifndef GB\n#define GB\nvoid f2(void);\n#endif\n",
        "main/gc.h": "#ifndef GC\n#define GC\nvoid f3(void);\n#endif\n",
        "main/gz.h": '#ifndef GZ\n#define GZ\n#include "main/ga.h"\n#endif\n',
    }
    cfg = world(tmp_path, sources=sources, headers=headers)
    result = go(cfg, "header", "ga", "include/x/ga.h")
    plan = seen.plans[0]
    rewritten = [p for p, data in plan.writes.items() if data and b'#include "x/ga.h"' in data]
    assert sorted(rewritten) == ["include/main/gz.h", "src/f1.c", "src/fuzzy/q.c"]  # one write per includer, counted
    assert plan.writes["include/main/ga.h"] is None and "#ifndef X_GA_H" in text(plan, "include/x/ga.h")
    assert "void f1(void);" in text(plan, "include/x/ga.h")
    assert seen.measured == [["src/f1.c", "src/f2.c"]] and result["applied"]["header"] == 1  # consumers, f3 untouched


def _placed(tmp_path: Path, headers: dict[str, str]) -> Any:
    """Three modules placed in subsystems main/ga..gc, each header at its subsystem path."""
    groups = [
        group(f"g{x}", f"f{i}", source=f"src/f{i}.c", header=f"include/game/main/g{x}.h", subsystem=f"main/g{x}")
        for i, x in enumerate("abc", 1)
    ]
    sources = {
        f"f{i}.c": f'#include "game/main/g{x}.h"\nint f{i}(void) {{ return 1; }}\n' for i, x in enumerate("abc", 1)
    }
    return world(tmp_path, groups=groups, sources=sources, headers={f"game/main/{k}": v for k, v in headers.items()})


def test_a_file_move_onto_an_existing_header_keeps_identical_statements_once(tmp_path: Path, seen: Any) -> None:
    headers = {
        "ga.h": "#ifndef GA\n#define GA\ntypedef int s32;\nvoid f1(void);\n#endif\n",
        "gb.h": "#ifndef GB\n#define GB\ntypedef int s32;\nvoid f2(void);\n#endif\n",
        "gc.h": "#ifndef GC\n#define GC\nvoid f3(void);\n#endif\n",
    }
    cfg = _placed(tmp_path, headers)
    go(cfg, "module", "gb", "ga")  # gb joins ga: its header merges into ga's
    merged = text(seen.plans[0], "include/game/main/ga.h")
    assert merged.count("typedef int s32;") == 1 and "void f1(void);" in merged and "void f2(void);" in merged
    assert seen.plans[0].writes["include/game/main/gb.h"] is None
    assert "game/main/ga.h" in text(seen.plans[0], "src/f1.c")


def test_a_conflicting_definition_in_a_merge_is_refused(tmp_path: Path, seen: Any) -> None:
    headers = {
        "ga.h": "#ifndef GA\n#define GA\nstruct P { int a; };\n#endif\n",
        "gb.h": "#ifndef GB\n#define GB\nstruct P { int b; };\n#endif\n",
        "gc.h": "#ifndef GC\n#define GC\nvoid f3(void);\n#endif\n",
    }
    result = go(_placed(tmp_path, headers), "module", "gb", "ga")
    assert result["refused"][0]["findings"][0]["key"] == "move.declaration"
    assert seen.plans == [] and seen.measured == []


# ------------------------------------------------------------------ proof, lock, no-ops


def test_a_proof_that_is_not_exact_refuses_every_move_that_touched_the_unit(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    seen.exact = False
    cfg = world(tmp_path)
    stdin(monkeypatch, "\n".join(json.dumps({"op": "member", "subject": s, "target": "ga"}) for s in ("f2", "f3")))
    result = move.run(cfg, {"op": "-"})
    assert [(r["index"], r["move"]["subject"]) for r in result["refused"]] == [(0, "f2"), (1, "f3")]
    assert {f["key"] for r in result["refused"] for f in r["findings"]} == {"move.not_exact"}
    assert seen.plans == [] and result["files"] == []
    # the merge's size change marked f2 and f3 as file starts; apart they still differ, so the refusal names f1's unit
    assert result["refused"][0]["findings"][0]["unit"].startswith("src/f1") and result["boundaries"] == ["f2", "f3"]


def test_a_structure_lock_held_past_the_wait_refuses_before_any_proof(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    cfg = world(tmp_path)
    (cfg.project.root / ".unbake").mkdir(exist_ok=True)
    now, naps = [0.0], []
    monkeypatch.setattr(move.store, "_CLOCK", lambda: now[0])
    monkeypatch.setattr(move.store, "_PAUSE", lambda s: (naps.append(s), now.__setitem__(0, now[0] + s))[0])
    with (cfg.project.root / ".unbake/structure.lock").open("a+b") as other:  # a landing in flight
        fcntl.flock(other.fileno(), fcntl.LOCK_SH)
        with pytest.raises(Refusal) as error:
            go(cfg, "member", "f2", "ga")
    assert error.value.findings[0].key == "setup.busy" and seen.measured == [] and seen.plans == []
    assert now[0] >= cfg.host.move_wait_s and len(naps) == round(cfg.host.move_wait_s / 0.2)  # polled, never slept


def test_a_batch_records_one_moved_row_per_path_and_symbol_move(tmp_path: Path, seen: Any) -> None:
    cfg = world(tmp_path)
    stdin_rows = [("path", "include/main/gb.h", "include/main/ga.h"), ("symbol", "f2", "g2")]
    assert moved.record(cfg, stdin_rows, "d1") == 2
    assert [(r["kind"], r["old"], r["new"], r["digest"]) for r in store.rows(cfg, "moved")] == [
        ("path", "include/main/gb.h", "include/main/ga.h", "d1"),
        ("symbol", "f2", "g2", "d1"),
    ]


def test_the_changes_of_a_batch_count_each_renamed_path_and_symbol(tmp_path: Path, seen: Any) -> None:
    cfg = world(tmp_path)
    snapshot = layout.capture(cfg)
    edited = move.edit(
        snapshot, move.read({"op": "file", "subject": "include/main/gb.h", "target": "include/x/gb.h"}, b"")
    )
    assert move._changes(snapshot, edited) == [("path", "include/main/gb.h", "include/x/gb.h")]
    edited = move.Edited(snapshot.layout, {}, {}, [], {"f1": "g1", "f2": "f2"})
    assert move._changes(snapshot, edited) == [("symbol", "f1", "g1")]


BATCH = [
    {"op": "source", "subject": "ga", "target": "src/ga2.c"},
    {"op": "header", "subject": "ga", "target": "include/main/ga2.h"},
    {"op": "source", "subject": "gb", "target": "src/gb2.c"},
    {"op": "source", "subject": "gc", "target": "src/gc2.c"},
    {"op": "header", "subject": "gc", "target": "include/main/gc2.h"},
]


def test_a_claims_mismatch_refuses_only_the_moves_that_touched_the_unit(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    cfg = world(tmp_path)

    made: list[str] = []

    def claims(snapshot: Any, *args: Any, **kw: Any) -> Any:  # the unit of ga claims four bytes more after the batch
        grown = "src/ga2.c" in snapshot.layout.units
        made.append(snapshot.digest)
        rows = [
            Claim(p, "a", ".text", 32 * i, 32 * i + 16 + (4 if grown and u.group == "ga" else 0))
            for i, (p, u) in enumerate(sorted(snapshot.layout.units.items()))
        ]
        return {}, tuple(rows), []

    monkeypatch.setattr(move.ownership, "claims", claims)
    stdin(monkeypatch, "\n".join(json.dumps(m) for m in BATCH))
    result = move.run(cfg, {"op": "-"})
    assert [r["index"] for r in result["refused"]] == [0, 1] and result["files"] == []
    assert {f["key"] for r in result["refused"] for f in r["findings"]} == {"move.not_exact"}
    assert seen.plans == []
    # each tree is claimed once per check: RW move round 3 claimed both trees twice (4 x ~9 s, half of it repeated)
    assert len(made) == len(set(made))


def test_a_policy_finding_refuses_only_the_moves_that_touched_its_unit(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    cfg = world(tmp_path)
    bad = Finding("policy.bad", "a rule", path="src/gc2.c")
    monkeypatch.setattr(move.policy, "census", lambda s, **kw: ((bad,), []))
    stdin(monkeypatch, "\n".join(json.dumps(m) for m in BATCH))
    result = move.run(cfg, {"op": "-"})
    assert [r["index"] for r in result["refused"]] == [3, 4] and seen.plans == []
    assert {f["key"] for r in result["refused"] for f in r["findings"]} == {"move.policy"}


def test_a_policy_rule_that_raises_refuses_only_the_moves_that_touched_its_unit(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    cfg = world(tmp_path)
    bad = Finding("source.file-version-guard", "not exact in every holding version", path="src/gc2.c", unit="gc2")

    def census(s: Any, **kw: Any) -> Any:
        raise Refusal(bad)

    monkeypatch.setattr(move.policy, "census", census)
    stdin(monkeypatch, "\n".join(json.dumps(m) for m in BATCH))
    result = move.run(cfg, {"op": "-"})
    assert [r["index"] for r in result["refused"]] == [3, 4] and seen.plans == []
    assert {f["key"] for r in result["refused"] for f in r["findings"]} == {"move.policy"}


def test_a_relayout_keeps_a_unit_whose_policy_rule_raises(tmp_path: Path, seen: Any, monkeypatch: Patch) -> None:
    def census(s: Any, **kw: Any) -> Any:
        if len(s.layout.units["src/f1.c"].members) > 1:  # the merged unit trips a rule that raises
            raise Refusal(Finding("source.file-version-guard", "not exact", path="src/f1.c", unit="f2"))
        return ((), [])

    monkeypatch.setattr(move.policy, "census", census)
    cfg = world(
        tmp_path,
        groups=[group("ga", "f1", "f2", source="src/f1.c")],
        units=[unit("src/f1.c", "ga", "f1"), unit("src/f2.c", "ga", "f2")],
    )
    result = go(cfg, "relayout")
    assert result["refused"] == [] and result["converged"] and sorted(result["kept"]) == ["src/f1.c", "src/f2.c"]
    assert result["census"][0]["class"].startswith("move.policy: ")


def test_an_unclean_rule_in_the_census_is_debt_not_a_refusal(tmp_path: Path, seen: Any, monkeypatch: Patch) -> None:
    cfg = world(tmp_path)
    debt = Finding("source.pad-view", "a padding view", path="src/gc2.c")  # move.unclean guards the credit instead
    monkeypatch.setattr(move.policy, "census", lambda s, **kw: ((debt,), []))
    stdin(monkeypatch, "\n".join(json.dumps(m) for m in BATCH))
    result = move.run(cfg, {"op": "-"})
    assert result["refused"] == [] and len(seen.plans) == 1


def test_each_refused_subject_is_reported_once_with_all_its_moves() -> None:
    finding = {"key": "move.policy", "reason": "r"}
    rows = [
        {"index": 0, "move": {"op": "source", "subject": "ga", "target": "src/x.c"}, "findings": [finding]},
        {"index": 1, "move": {"op": "header", "subject": "ga", "target": "include/x.h"}, "findings": [finding]},
        {"index": 2, "move": {"op": "source", "subject": "gb", "target": "src/y.c"}, "findings": []},
    ]
    assert move._subjects(rows) == [
        {"subject": "ga", "key": "move.policy", "reason": "r", "moves": [0, 1]},
        {"subject": "gb", "key": "", "reason": "", "moves": [2]},
    ]


def test_a_module_folds_each_unit_once_in_one_job_on_the_writes_of_the_units_before(
    tmp_path: Path, monkeypatch: Patch, toolchains: Any
) -> None:
    cfg = world(tmp_path)
    folded: list[tuple[str, bytes]] = []

    def fold(snapshot: Any, u: Any, view: Any, strict: bool = True) -> Any:
        head = snapshot.read("include/main/ga.h")
        folded.append((u.path, head))
        return {"include/main/ga.h": head + f"/* {u.path} */\n".encode()}, ()

    monkeypatch.setattr(move.headers, "fold", fold)
    monkeypatch.setattr(move.view, "get", lambda *a, **k: _view((("include/main/ga.h", "x"),)))
    monkeypatch.setattr(move.recipes, "resolve", lambda *a: None)
    with move.effort.command("move", []):
        snapshot = layout.capture(cfg)
        units = [snapshot.layout.units[p] for p in ("src/f1.c", "src/f2.c")]  # one module's units: both write ga.h
        results = move._fold_module(
            (cfg, snapshot, [(u, _view((("include/main/ga.h", "x"),)), ()) for u in units], True, True)
        )
    assert [p for p, _ in folded] == ["src/f1.c", "src/f2.c"]  # each unit folded once: fold jobs equal units
    assert b"src/f1.c" in folded[1][1]  # the second folds on the header the first wrote, its view made again
    assert all(r.lands for r in results) and b"src/f2.c" in results[1].fresh["include/main/ga.h"]


def test_an_overlay_lists_no_files_from_disk_and_scans_only_the_texts_it_changes(
    tmp_path: Path, monkeypatch: Patch, toolchains: Any
) -> None:
    cfg = world(tmp_path)
    with move.effort.command("move", []):
        snapshot = layout.capture(cfg)
        headers.spellings(snapshot)
        listed, scanned = [], []
        real_sources, real_map = headers.sources, headers.pool.map
        monkeypatch.setattr(headers, "sources", lambda *a: listed.append(a[1:]) or real_sources(*a))
        monkeypatch.setattr(
            headers.pool,
            "map",
            lambda c, name, fn, items, *k: scanned.extend(items) or real_map(c, name, fn, items, *k),
        )
        text = body("f1", "extern short wide;\n").encode()
        over = layout.overlay(snapshot, {"src/f1.c": text, "src/new.c": b"extern long wide;\n"})
        paths, _ = headers.landed_sources(over)
        assert paths == sorted({*headers.landed_sources(snapshot)[0], "src/new.c"}) and listed == []
        assert {s for _, s in headers.spellings(over)["wide"]} == {"short @", "long @"}
        assert sorted(t for t, _ in scanned) == sorted([text, b"extern long wide;\n"])  # only the changed texts


def test_no_moves_and_moves_that_change_nothing_write_and_prove_nothing(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    cfg = world(tmp_path)
    stdin(monkeypatch, "[]")
    assert move.run(cfg, {"op": "-"})["moves"] == 0
    result = go(cfg, "module", "ga", "ga")
    assert result["files"] == [] and result["refused"] == []
    assert seen.measured == [] and seen.plans == [] and seen.claims == 0


def test_only_the_rewritten_units_and_the_consumers_of_changed_headers_are_proved(tmp_path: Path, seen: Any) -> None:
    cfg = world(tmp_path)
    go(cfg, "member", "f3", "gb")  # f3's text moves into f2's unit: f1 stays out of the proof
    assert seen.measured == [["src/f2.c"]]
    assert layout.capture(cfg).layout.units.keys() == {"src/f1.c", "src/f2.c", "src/f3.c"}  # nothing was applied


# ------------------------------------------------------------------ atoms


def test_split_atoms_keeps_each_definition_with_its_comment_and_refuses_a_shared_region() -> None:
    source = '#include "a.h"\n#include <b.h>\nint shared;\n/* the first */\n'
    source += "int f1(void) { return 1; }\nint f2(void) { return 2; }\n"
    includes, blocks = headers.split_atoms(source, {"f1", "f2"})
    assert includes == ['#include "a.h"', "#include <b.h>"]
    assert (
        blocks["f1"] == "/* the first */\nint f1(void) { return 1; }" and blocks["f2"] == "int f2(void) { return 2; }"
    )
    assert [v for k, v in blocks.items() if k.startswith("#")] == ["int shared;"]
    assert (
        headers.render(includes, ["int shared;"], [blocks["f1"], blocks["f2"]])
        .decode()
        .startswith('#include "a.h"\n#include <b.h>\n\nint shared;')
    )
    switch = "int f1(int c) {\n    switch (c) {\n#if A\n    case 1:\n#endif\n        g();\n"
    switch += "#if B\n    case 2:\n#endif\n    }\n}\n"
    assert headers.split_atoms(switch, {"f1"})[1] == {"f1": switch.strip("\n")}  # directives in a body stay in it
    with pytest.raises(Refusal) as error:
        headers.split_atoms("#if X\nint f1(void) { return 1; }\nint f2(void) { return 2; }\n#endif\n", {"f1", "f2"})
    assert error.value.findings[0].key == "move.directive"


def test_a_refusal_names_each_distinct_cause_once_beside_the_per_move_rows(monkeypatch) -> None:
    moves = [Move(i, "symbol", f"s{i}", f"t{i}") for i in range(3)]
    same = Finding("headers.conflict", "f: local 'a' vs 'b'")
    other = Finding("move.unknown", "nope")
    rows = [(moves[0], same), (moves[1], same), (moves[2], other)]
    monkeypatch.setattr(move, "_result", lambda m, s, **extra: extra)
    result = move._refusal(moves, None, rows)
    assert result["conflicts"] == [
        {"key": "headers.conflict", "reason": "f: local 'a' vs 'b'", "moves": 2},
        {"key": "move.unknown", "reason": "nope", "moves": 1},
    ]
    assert len(result["refused"]) == 3 and result["message"].endswith("by 2 distinct causes")


def _fold_fails(monkeypatch: Patch, bad: str, seen: Any) -> None:
    def fold(snapshot: Any, u: Any, view: Any, strict: bool = True) -> Any:
        seen.folded.append(u.path)
        if u.path == bad:
            raise Refusal(Finding("headers.parse", "Invalid specifier list"))
        return {}, ()

    monkeypatch.setattr(move.headers, "fold", fold)


def test_the_fold_views_of_all_units_are_made_in_one_dispatch(tmp_path: Path, seen: Any, monkeypatch: Patch) -> None:
    dispatched = []
    real = move.pool.map
    monkeypatch.setattr(
        move.pool, "map", lambda c, name, f, items, *k: (dispatched.append(name), real(c, name, f, items, *k))[1]
    )
    result = go(_split_with(tmp_path, "static inline int helper(int x) { return x + 1; }\n"), "member", "f2", "gc")
    assert result["refused"] == [] and dispatched.count("move.views") == 1
    assert sorted(seen.views) == sorted(seen.folded)  # one view per folded unit, none made again


def test_the_parent_keeps_each_views_key_and_reads_and_its_worker_loads_the_text(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    big = "x" * 100_000  # a view's text: megabytes a unit, for every unit a move folds
    monkeypatch.setattr(move.view, "get", lambda s, u, *a, **k: seen.views.append(u.path) or _big(u.path, big))
    loaded: list[str] = []
    monkeypatch.setattr(move.view, "stored", lambda c, unit, v, key: loaded.append(unit) or _big(unit, big))
    real, held = move.pool.map, []

    def mapped(config: Any, name: str, fn: Any, items: Any, key: Any = None) -> Any:
        out = real(config, name, fn, items)
        held.append(len(pickle.dumps((items, out) if name == "move.views" else items)))
        return out

    monkeypatch.setattr(move.pool, "map", mapped)
    result = go(world(tmp_path), "member", "f2", "ga")
    assert result["refused"] == [] and sorted(seen.views) == sorted(seen.folded) == sorted(loaded)
    assert held and max(held) < len(big)  # the parent's views and fold jobs carry no view text


def test_a_fold_whose_kept_view_the_store_lost_views_the_unit_again(
    tmp_path: Path, monkeypatch: Patch, toolchains: Any
) -> None:
    cfg = world(tmp_path)
    made: list[str] = []
    monkeypatch.setattr(move.view, "get", lambda s, u, *a, **k: made.append(u.path) or _big(u.path, "int x;"))
    monkeypatch.setattr(move.recipes, "resolve", lambda *a: None)
    monkeypatch.setattr(move.headers, "fold", lambda s, u, v, strict=True: ({}, ()))
    with move.effort.command("move", []):
        snapshot = layout.capture(cfg)
        unit = snapshot.layout.units["src/f1.c"]
        seen_ = move.Seen(unit.path, "a", "lost", frozenset({unit.path}))
        move._fold_job((cfg, snapshot, unit, seen_))
    assert made == ["src/f1.c"]


def test_a_stored_view_loads_by_its_key_with_its_reads_and_no_line_table(tmp_path: Path, toolchains: Any) -> None:
    cfg = world(tmp_path)
    with move.effort.command("move", []):
        move.view._keep(cfg, "k" * 64, False, "int x;\n", [("src/f1.c", "pin")])
        loaded = move.view.stored(cfg, "src/f1.c", "a", "k" * 64)
        assert move.view.stored(cfg, "src/f1.c", "a", "m" * 64) is None
    assert loaded == SourceView("src/f1.c", "a", "k" * 64, "int x;\n", (("src/f1.c", "pin"),), ())


def _big(unit: str, text: str) -> SourceView:
    return SourceView(unit, "a", "key-" + unit, text, ((unit, "pin"),), ())


def test_a_unit_whose_fold_refuses_names_the_unit_and_its_moves_and_writes_nothing(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    _fold_fails(monkeypatch, "src/f1.c", seen)
    result = go(world(tmp_path), "member", "f2", "ga")
    (row,) = result["refused"]
    assert row["findings"][0]["key"] == "headers.parse" and row["findings"][0]["unit"] == "src/f1.c"
    assert seen.plans == []


def test_partial_keeps_each_failing_unit_and_applies_the_rest(tmp_path: Path, seen: Any, monkeypatch: Patch) -> None:
    _fold_fails(monkeypatch, "src/f1.c", seen)
    stdin(
        monkeypatch,
        "\n".join(
            json.dumps(m)
            for m in (
                {"op": "member", "subject": "f2", "target": "ga"},
                {"op": "member", "subject": "f3", "target": "gb"},
            )
        ),
    )
    result = move.run(world(tmp_path), {"op": "-", "partial": "1"})
    # the unit whose fold fails (and the old unit it came with) keeps its old files as a census row; the rest lands
    assert result["refused"] == [] and sorted(result["kept"]) == ["src/f1.c", "src/f2.c"] and result["rounds"] == 2
    assert len(seen.plans) == 1 and result["applied"]["member"] == 2 and result["converged"]
    assert result["census"][0]["class"].startswith("headers.parse")


def test_a_link_error_while_generating_blames_the_moves_of_its_unit(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    def files(snapshot: Any) -> Any:
        raise Refusal(Finding("link.error", "owned .rodata run is not contiguous", unit="src/f1.c"))

    monkeypatch.setattr(move.repo, "files", files)
    result = go(world(tmp_path), "member", "f2", "ga")
    assert result["refused"][0]["findings"][0]["key"] == "link.error" and seen.plans == []


def test_a_fold_that_changes_a_header_a_later_view_read_makes_that_view_again(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    def fold(snapshot: Any, u: Any, view: Any, strict: bool = True) -> Any:
        seen.folded.append(u.path)
        return ({"include/main/gc.h": b"void f3(void);\nvoid extra(void);\n"} if u.path == "src/f1.c" else {}), ()

    monkeypatch.setattr(move.headers, "fold", fold)
    monkeypatch.setattr(
        move.view, "get", lambda s, u, *a, **k: (seen.views.append(u.path), _view((("include/main/gc.h", "x"),)))[1]
    )
    monkeypatch.setattr(move.view, "stored", lambda c, unit, v, key: _view((("include/main/gc.h", "x"),)))
    stdin(monkeypatch, json.dumps([{"op": "member", "subject": "f2", "target": "ga"}]))
    go(_split_with(tmp_path, "static inline int helper(int x) { return x + 1; }\n"), "member", "f2", "gc")
    # f1 and f3 fold together; f3 read the header f1's fold wrote, so it is viewed and folded again after it
    assert seen.folded == ["src/f1.c", "src/f3.c", "src/f3.c"] and seen.views == ["src/f1.c", "src/f3.c", "src/f3.c"]


def test_partial_stops_after_its_rounds_and_refuses_the_rest_as_one_finding(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    rounds = []

    def apply(config: Any, moves: Any, *_: Any) -> Any:
        rounds.append(len(moves))
        m = moves[0]
        row = {"index": 0, "move": {"op": m.op, "subject": m.subject, "target": m.target}, "findings": []}
        return {"refused": [row], "moves": len(moves), "message": "x"}

    monkeypatch.setattr(move, "_apply", apply)
    batch = [{"op": "symbol", "subject": f"s{i}", "target": f"t{i}"} for i in range(9)]
    stdin(monkeypatch, json.dumps(batch))
    result = move.run(world(tmp_path), {"op": "-", "partial": "1"})
    assert rounds == [9, 8, 7, 6, 5] and result["rounds"] == 5
    assert sorted(r["index"] for r in result["refused"]) == list(range(9))
    assert sum(f["key"] == "move.rounds" for r in result["refused"] for f in r["findings"]) == 4


def test_a_round_keeps_nothing_it_memoised_for_the_next_round(tmp_path: Path, seen: Any, monkeypatch: Patch) -> None:
    held = []

    def apply(config: Any, moves: Any, *_: Any) -> Any:
        held.append(sorted(k for k in effort._memo if k[0] == "round"))  # what the round before left behind
        effort.memo(("round", len(moves)), lambda: [0] * 1000)  # its overlays, indexes and blobs
        m = moves[0]
        row = {"index": 0, "move": {"op": m.op, "subject": m.subject, "target": m.target}, "findings": []}
        return {"refused": [row], "moves": len(moves), "message": "x"}

    monkeypatch.setattr(move, "_apply", apply)
    stdin(monkeypatch, json.dumps([{"op": "symbol", "subject": f"s{i}", "target": f"t{i}"} for i in range(3)]))
    move.run(world(tmp_path), {"op": "-", "partial": "1"})
    assert held == [[], [], []] and not [k for k in effort._memo if k[0] == "round"]


def test_a_view_made_by_the_real_view_constructor_has_the_field_the_fold_reads() -> None:
    assert {"dependencies"} <= {f.name for f in dataclasses.fields(SourceView)}
    assert _view((("a.h", "x"),)).dependencies == (("a.h", "x"),)


def test_a_stream_stamped_for_another_layout_is_refused_once_naming_both_digests(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    cfg = world(tmp_path)
    batch = [{"layout": "other"}, *({"op": "member", "subject": s, "target": "ga"} for s in ("f2", "f3"))]
    stdin(monkeypatch, "\n".join(json.dumps(r) for r in batch))
    with pytest.raises(Refusal) as error:
        move.run(cfg, {"op": "-"})
    (found,) = error.value.findings
    digest = move.layout.capture(cfg).layout.digest
    assert found.key == "move.stale" and "other" in found.reason and digest in found.reason and seen.plans == []


def test_a_stream_stamped_for_this_layout_applies(tmp_path: Path, seen: Any, monkeypatch: Patch) -> None:
    cfg = world(tmp_path)
    digest = move.layout.capture(cfg).layout.digest
    batch = [{"layout": digest}, {"op": "member", "subject": "f2", "target": "ga"}]
    stdin(monkeypatch, "\n".join(json.dumps(r) for r in batch))
    assert move.run(cfg, {"op": "-"})["refused"] == [] and len(seen.plans) == 1


def test_a_move_that_makes_a_clean_unit_unclean_is_refused_naming_the_rule_and_partial_drops_it(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    # f2 joins f1: after the move src/f1.c fires the rule although neither unit did before
    monkeypatch.setattr(
        move.policy,
        "unclean_rules",
        lambda s, only=None: {"f1": ("source.address-int",)} if "src/f2.c" not in s.layout.units else {},
    )
    monkeypatch.setattr(
        move.policy,
        "evaluate",
        lambda *a, **k: (Finding("source.address-int", "(s32) &D_auto_1", path="src/f1.c", line=7),),
    )
    result = go(world(tmp_path), "member", "f2", "ga")
    (row,) = result["refused"]
    finding = row["findings"][0]
    assert (
        finding["key"] == "move.unclean"
        and "source.address-int" in finding["reason"]
        and "src/f1.c:7" in finding["reason"]
    )
    assert seen.plans == []


def test_a_stream_keeps_the_old_units_of_a_newly_unclean_unit_and_refuses_no_unrelated_move(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    # round 9 of inv 849e77f0: the unclean check named src/func_8025D928_de_15.c, a unit only the new layout has;
    # nothing mapped it to old units, so the stream refused its first move each round instead of keeping the unit
    calls = []
    real = move.newly_unclean

    def newly(before: Any, after: Any, affected: Any) -> Any:
        calls.append(sorted(after.layout.units))
        if len(calls) > 1:
            return real(before, after, affected)
        return [("src/f1.c", Finding("move.unclean", "source.fake-type", path="src/f1.c", unit="src/f1.c"))], ()

    monkeypatch.setattr(move, "newly_unclean", newly)
    stdin(monkeypatch, json.dumps([{"op": "member", "subject": "f2", "target": "ga"}]))
    result = move.run(world(tmp_path), {"op": "-", "partial": "1"})
    assert result["refused"] == [] and "src/f1.c" in result["kept"] and result["converged"]


def test_a_newly_unclean_old_unit_is_kept_even_when_the_new_unit_at_its_path_holds_only_kept_members(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    # round 9 of the 06:15 race: src/func_802BA210_de_33.c (old, not kept) fired an unclean rule; the new unit at
    # that path held members whose old units were all kept, so mapping by members alone kept nothing new
    found = Finding("move.unclean", "src/f1.c:4 source.fake-type", path="src/f1.c", unit="src/f1.c")
    monkeypatch.setattr(move, "newly_unclean", lambda *a: ([("src/f1.c", found)], ()))
    real = move.layout.unit_of
    monkeypatch.setattr(move.layout, "unit_of", lambda s, n: real(s, n) if n != "f1" else real(s, "f3"))
    result = move._apply(world(tmp_path), [move.Move(0, "member", "f2", "ga")], False, (), ["src/f3.c"], True)
    assert "src/f1.c" in result["keep"] and not result.get("unkept")


def test_a_kept_unit_that_turns_unclean_is_debt_and_the_stream_still_lands(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    # round 9 of inv 849e77f0: src/func_8025D928_de_15.c, kept with its module, fired source.param-cast once the
    # headers it reads changed; already kept, nothing new could be kept, so a fake move 0 was refused every round
    found = Finding("move.unclean", "src/f3.c:17 source.param-cast", path="src/f3.c", unit="src/f3.c")
    monkeypatch.setattr(move, "newly_unclean", lambda *a: ([("src/f3.c", found)], ()))
    cfg = world(tmp_path)
    moves = [move.Move(0, "member", "f2", "ga")]
    result = move._apply(cfg, moves, False, (), ["src/f3.c"], True)
    assert not result.get("refused") and not result.get("keep") and len(seen.plans) == 1
    assert not move._apply(cfg, moves, False, (), [], True).get("refused")  # nothing kept: f3 is new to keep


@pytest.mark.parametrize(
    "after", [{"f1": ("source.raw-offset",)}, {"f1": ("source.raw-offset",), "f2": ("source.raw-offset",)}]
)
def test_a_move_into_an_unclean_unit_fails_only_when_a_moved_exact_function_loses_its_credit(
    tmp_path: Path, seen: Any, monkeypatch: Patch, after: Any
) -> None:
    # f1's unit is unclean before; f2 joins it with its bytes exact: credit falls only if f2 is withheld after
    before = {"f1": ("source.raw-offset",)}
    monkeypatch.setattr(
        move.policy, "unclean_rules", lambda s, only=None: before if "src/f2.c" in s.layout.units else after
    )
    monkeypatch.setattr(
        move.policy,
        "evaluate",
        lambda *a, **k: (Finding("source.raw-offset", "raw: (u8 *)p + 4", path="src/f1.c", line=1),),
    )
    monkeypatch.setattr(move.publish, "progress_block", lambda snapshot, writes: {})
    result = go(world(tmp_path), "member", "f2", "ga")
    if "f2" not in after:
        assert result["refused"] == [] and len(seen.plans) == 1
        return
    (row,) = result["refused"]
    finding = row["findings"][0]
    assert finding["key"] == "move.unclean" and "cost f2 its exact credit" in finding["reason"]
    assert seen.plans == []


def test_the_unclean_check_judges_only_the_units_the_batch_touched(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    asked = []
    monkeypatch.setattr(move.policy, "unclean_rules", lambda s, only=None: asked.append(set(only)) or {})
    monkeypatch.setattr(move.publish, "progress_block", lambda snapshot, writes: {})  # the report reads the store
    result = go(world(tmp_path), "member", "f2", "ga")
    assert result["refused"] == [] and asked and all(a <= {"src/f1.c", "src/f2.c"} for a in asked)
    assert not any("src/f3.c" in a for a in asked)  # the untouched unit is never judged


def test_a_source_the_rules_cannot_judge_is_a_non_blocking_finding_not_a_refusal(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    def judge(s: Any, only: Any = None) -> Any:
        raise Refusal(Finding("source.file-version-guard", "cannot judge"))

    monkeypatch.setattr(move.policy, "unclean_rules", judge)
    monkeypatch.setattr(move.publish, "progress_block", lambda snapshot, writes: {})
    result = go(world(tmp_path), "member", "f2", "ga")
    assert result["refused"] == [] and any(
        f.key == "source.file-version-guard" and not f.blocking for f in seen.plans[0].debt
    )


# ------------------------------------------------------------------ merged units: lost bodies, member conflicts


def test_merging_a_unit_whose_signature_names_a_struct_keeps_both_definitions(tmp_path: Path, seen: Any) -> None:
    f2 = '#include "main/gb.h"\nstruct R { int x; };\nint f2(struct R *r) { return r->x; }\n'
    cfg = world(tmp_path, sources={"f1.c": body("f1"), "f2.c": f2, "f3.c": body("f3")})
    result = go(cfg, "member", "f2", "ga")
    assert result["refused"] == []
    merged = text(seen.plans[0], "src/f1.c")
    assert "int f1(void) { return 1; }" in merged and "int f2(struct R *r) { return r->x; }" in merged


def _disagreeing(tmp_path: Path, header: str = "") -> Any:
    f1 = '#include "main/ga.h"\nextern struct A *X;\nint f1(void) { return X != 0; }\n'
    f2 = '#include "main/gb.h"\nextern struct B *X;\nint f2(void) { return X == 0; }\n'
    headers = {
        f"main/g{g}.h": f"#ifndef G{g}\n#define G{g}\nvoid f{i}(void);\n#endif\n" for i, g in enumerate("abc", 1)
    }
    headers["main/ga.h"] = headers["main/ga.h"].replace("void f1", header + "void f1")
    return world(tmp_path, sources={"f1.c": f1, "f2.c": f2, "f3.c": body("f3")}, headers=headers)


def test_member_externs_that_disagree_move_to_block_scope_in_the_merged_unit(tmp_path: Path, seen: Any) -> None:
    result = go(_disagreeing(tmp_path), "member", "f2", "ga")
    assert result["refused"] == []
    merged = text(seen.plans[0], "src/f1.c")
    assert "int f1(void) {\n    extern struct A *X; return X != 0; }" in merged
    assert "int f2(void) {\n    extern struct B *X; return X == 0; }" in merged
    assert merged.count("extern struct") == 2  # none left at file scope


def test_an_extern_of_an_unnamed_record_that_disagrees_moves_to_block_scope_too(tmp_path: Path, seen: Any) -> None:
    cfg = _disagreeing(tmp_path)
    f2 = '#include "main/gb.h"\nextern struct { int value; } X;\nint f2(void) { return X.value; }\n'
    (cfg.project.root / "src/f2.c").write_text(f2)
    assert go(cfg, "member", "f2", "ga")["refused"] == []
    merged = text(seen.plans[0], "src/f1.c")
    assert "int f2(void) {\n    extern struct { int value; } X; return X.value; }" in merged
    assert "\nextern struct" not in merged  # none left at file scope


def test_a_disagreeing_extern_a_shared_inline_function_uses_moves_into_that_function(tmp_path: Path, seen: Any) -> None:
    cfg = _disagreeing(tmp_path)
    (cfg.project.root / "src/f2.c").write_text(
        '#include "main/gb.h"\nextern struct B *X;\nstatic inline int h(void) { return X == 0; }\n'
        "int f2(void) { return h(); }\n"
    )
    result = go(cfg, "member", "f2", "ga")
    assert result["refused"] == []
    assert "static inline int h(void) {\n    extern struct B *X; return X == 0; }" in text(seen.plans[0], "src/f1.c")


def test_member_externs_that_disagree_with_a_header_declaring_the_name_keep_their_units_apart_as_debt(
    tmp_path: Path, seen: Any
) -> None:
    result = go(_disagreeing(tmp_path, "extern void *X;\n"), "member", "f2", "ga")
    assert result["refused"] == []
    debt = [f for f in seen.plans[0].debt if f.key == "layout.decl_split"]
    assert [f.reason.split()[0] for f in debt] == ["X"] and not debt[0].blocking
    assert b"int f2(void)" not in seen.plans[0].writes.get("src/f1.c", b"")  # f2 keeps its own file


def test_compose_of_several_units_tokenizes_each_old_unit_once(tmp_path: Path, seen: Any, monkeypatch: Patch) -> None:
    calls: dict[str, list[int]] = {"split": [], "scope": []}
    split, scope = headers.split_atoms, layout._scope_rows
    monkeypatch.setattr(layout.headers, "split_atoms", lambda t, m: (calls["split"].append(1), split(t, m))[1])
    monkeypatch.setattr(layout, "_scope_rows", lambda b: (calls["scope"].append(1), scope(b))[1])
    stdin(monkeypatch, "\n".join(json.dumps({"op": "member", "subject": s, "target": "ga"}) for s in ("f2", "f3")))
    result = move.run(world(tmp_path), {"op": "-"})
    assert result["refused"] == [] and len(calls["split"]) == 3 and len(calls["scope"]) == 3


# ------------------------------------------------------------------ unproved


def test_unproved_drops_the_moves_of_a_unit_whose_fold_fails_and_applies_the_rest(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    def fold(snapshot: Any, u: Any, view: Any, strict: bool = True) -> Any:
        seen.folded.append(u.path)
        if u.path == "src/x/b.c":
            raise Refusal(Finding("headers.parse", "include/x.h: Invalid specifier list", path="include/x.h"))
        return {u.path: snapshot.read(u.path)}, ()

    monkeypatch.setattr(move.headers, "fold", fold)
    moves = [("source", "ga", "src/x/a.c"), ("source", "gb", "src/x/b.c")]
    stdin(monkeypatch, "\n".join(json.dumps({"op": o, "subject": s, "target": t}) for o, s, t in moves))
    result = move.run(world(tmp_path), {"op": "-", "unproved": True})
    assert [(r["index"], r["findings"][0]["key"]) for r in result["refused"]] == [(1, "headers.parse")]
    assert {"src/x/a.c", "src/x/b.c"} <= set(seen.folded)  # every unit folded in the round that failed
    (plan,) = seen.plans
    assert b"int f1" in plan.writes["src/x/a.c"] and "src/x/b.c" not in plan.writes
    assert seen.measured == [] and seen.claims == 0  # unproved: no proof, no claims compare


def test_independent_units_fold_in_one_pool_dispatch(tmp_path: Path, seen: Any, monkeypatch: Patch) -> None:
    dispatched: list[tuple[str, int]] = []

    def inline(config: Any, name: str, fn: Any, items: Any, key: Any = None) -> list[Any]:
        dispatched.append((name, len(items)))
        return [fn(i) for i in items]

    monkeypatch.setattr(move.pool, "map", inline)
    moves = [("source", g, f"src/x/{g}.c") for g in ("ga", "gb", "gc")]
    stdin(monkeypatch, "\n".join(json.dumps({"op": o, "subject": s, "target": t}) for o, s, t in moves))
    result = move.run(world(tmp_path), {"op": "-"})
    assert result["refused"] == [] and [d for d in dispatched if d[0] == "move.fold"] == [("move.fold", 3)]
    assert sorted(seen.folded) == ["src/x/ga.c", "src/x/gb.c", "src/x/gc.c"]  # each folded once, inside the job


def _cpp(snapshot: Any, path: str, guards: set[str]) -> tuple[list[str], set[str]]:
    """What a preprocessor makes of a file: quoted includes inlined once each between line markers, directives blank."""
    out, read = [f'# 1 "{path}"' + (" 1" if guards else "")], {path}
    guards.add(path)
    for number, line in enumerate((snapshot.peek(path) or b"").decode().splitlines(), 1):
        if (inc := re.match(r'\s*#\s*include\s*"([^"]+)"', line)) and f"include/{inc[1]}" not in guards:
            lines, more = _cpp(snapshot, f"include/{inc[1]}", guards)
            out += [*lines, f'# {number + 1} "{path}" 2']
            read |= more
        else:
            out.append("" if line.lstrip().startswith("#") else line)
    return out, read


def _preprocessed(snapshot: Any, unit: Any, *args: Any, **kw: Any) -> SourceView:
    lines, read = _cpp(snapshot, unit.path, set())
    deps = tuple((p, headers.sha256(snapshot.peek(p) or b"")) for p in [unit.path, *sorted(read - {unit.path})])
    return SourceView(unit.path, "a", "key", "\n".join(lines) + "\n", deps, ())


def _folded_world(tmp_path: Path) -> Any:
    """ga holds f1 and f3 in two units (f2 of gb sits between them), so one module job folds both."""
    sources = {
        "f1.c": '#include "main/ga.h"\nint f2(void);\nvoid fresh1(int a);\nint f1(void) { return 1; }\n',
        "f2.c": '#include "main/gb.h"\ntypedef int s32;\nint f3(void);\nextern s32 gTwo;\nint f2(void) { return 2; }\n',
        "f3.c": '#include "main/ga.h"\nint f2(void);\nvoid fresh3(long b);\nint f3(void) { return 3; }\n',
    }
    found = {"ga": "int f1(void);\nint f3(void);\n", "gb": "int f2(void);\n", "gc": ""}
    return world(
        tmp_path,
        sources=sources,
        groups=[group("ga", "f1", "f3", source="src/f1.c"), group("gb", "f2", source="src/f2.c"), group("gc")],
        units=[unit("src/f1.c", "ga", "f1"), unit("src/f2.c", "gb", "f2"), unit("src/f3.c", "ga", "f3")],
        headers={f"main/{g}.h": f"#ifndef {g}\n#define {g}\n{t}#endif\n" for g, t in found.items()},
    )


def _apply_all(cfg: Any, moves: list[dict[str, str]], seen: Any) -> tuple[Any, ...]:
    seen.plans.clear()
    headers.effort.forget("declarations", "words", "typed")
    rows = tuple(Move(i, m["op"], m["subject"], m["target"]) for i, m in enumerate(moves))
    result = move._apply(cfg, rows, unproved=True)
    writes = {p: (b or b"").decode() for p, b in (seen.plans[-1].writes.items() if seen.plans else ())}
    return writes, result["refused"]


def test_a_moves_views_carry_no_line_table(tmp_path: Path, seen: Any, monkeypatch: Patch) -> None:
    asked: list[bool] = []

    def get(snapshot: Any, unit: Any, *args: Any, lines: bool = True, **kw: Any) -> SourceView:
        asked.append(lines)  # a line table is 2 MB a unit: a move holds a view of every unit it folds
        return _preprocessed(snapshot, unit, *args, **kw)

    monkeypatch.setattr(move.view, "get", get)
    _apply_all(_folded_world(tmp_path), [{"op": "module", "subject": "ga", "target": "gc"}], seen)
    assert asked and not any(asked)


def test_module_jobs_over_the_index_write_and_find_what_one_whole_parse_per_unit_job_does(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    monkeypatch.setattr(move.view, "get", _preprocessed)
    monkeypatch.setattr(move.view, "stored", lambda *a: None)  # the store lost them: each fold views its unit
    moves = [{"op": "module", "subject": "ga", "target": "gc"}]
    folded: list[str] = []
    monkeypatch.setattr(move.headers, "fold", lambda s, u, *a, **k: folded.append(u.path) or _FOLD(s, u, *a, **k))
    new = _apply_all(_folded_world(tmp_path / "new"), moves, seen)
    assert sorted(folded) == ["src/gc.c", "src/gc_2.c"]  # one module, two units: one job folds both
    grouped = move.defaultdict

    class PerUnit(dict):  # the old path: one fold job per unit, each on its wave's overlay
        def __getitem__(self, key: str) -> list[str]:
            return self.setdefault((key, len(self)), [])

    monkeypatch.setattr(move, "defaultdict", lambda kind: PerUnit() if kind is list else grouped(kind))

    def whole(text: str, path: str, config: Any = None) -> Any:  # the old path: one parse of the whole unit
        runs: list[list[Any]] = []
        for node in CParser().parse(text, filename=path).ext:
            if not runs or runs[-1][0].coord.file != node.coord.file:
                runs.append([])
            runs[-1].append(node)
        return [headers._entry(nodes, {}, headers._words(text)) for nodes in runs]

    def every_typedef(self: Any, texts: Any) -> str:
        return "\n".join(self._typedef(n)[0] for n in self.typedefs)

    monkeypatch.setattr(headers, "_regions", whole)
    monkeypatch.setattr(headers._Shapes, "_body", every_typedef)
    old = _apply_all(_folded_world(tmp_path / "old"), moves, seen)
    assert folded[2:] == ["src/gc.c", "src/gc_2.c", "src/gc_2.c"]  # the old path folds gc_2 again in a wave
    assert new == old  # the same writes and refusals, byte for byte
    assert (
        "void fresh1(int a);" in new[0]["include/main/gc.h"] and "void fresh3(long b);" in new[0]["include/main/gc.h"]
    )


def test_a_wave_folds_again_only_the_units_that_consulted_a_name_its_writes_changed() -> None:
    base = SimpleNamespace(peek=lambda p: b'#include "a.h"\nint f1(void);\n')
    landing = move.Written(set(), set(), set())
    landing.add(base, {"include/main/ga.h": b'#include "a.h"\nint f1(void);\nvoid fresh(int a);\n'})

    def result(path: str, consulted: Any, read: tuple[str, ...] = ("include/main/ga.h",)) -> move.Folded:
        return move.Folded(path, {}, (), {path, *read}, (), True, consulted)

    rows = [
        result("src/caller.c", (frozenset({"fresh", "f1"}), frozenset())),  # it looked fresh up: folded again
        result("src/other.c", (frozenset({"f1", "other"}), frozenset())),  # read the header, not the new name
        result("src/whole.c", (frozenset(), frozenset({"include/main/ga.h"}))),  # read the header whole
        result("src/unknown.c", None),  # no record of what it read: every read counts
        result("src/away.c", None, read=("include/main/gb.h",)),  # read nothing written
    ]
    assert [r.path for r in rows if landing.stale(r)] == ["src/caller.c", "src/whole.c", "src/unknown.c"]
    landing.add(base, {"include/main/gb.h": b'#include "b.h"\n'})  # its includes changed: every reader is affected
    assert landing.stale(result("src/other.c", (frozenset(), frozenset()), read=("include/main/gb.h",)))


def test_the_writers_of_one_file_fold_in_turn_in_one_job_of_the_next_wave(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    dispatched: list[tuple[str, int]] = []

    def inline(config: Any, name: str, fn: Any, items: Any, key: Any = None) -> list[Any]:
        dispatched.append((name, len(items)))
        return [fn(i) for i in items]

    def fold(snapshot: Any, u: Any, view: Any, strict: bool = True) -> Any:  # every module adds a line to one header
        seen.folded.append(u.path)
        shared = (snapshot.peek("include/common.h") or b"") + f"void from_{u.group}(void);\n".encode()
        return {u.path: snapshot.read(u.path), "include/common.h": shared}, ()

    monkeypatch.setattr(move.pool, "map", inline)
    monkeypatch.setattr(move.headers, "fold", fold)
    moves = [("source", g, f"src/x/{g}.c") for g in ("ga", "gb", "gc")]
    stdin(monkeypatch, "\n".join(json.dumps({"op": o, "subject": s, "target": t}) for o, s, t in moves))
    result = move.run(world(tmp_path), {"op": "-", "unproved": True})
    assert result["refused"] == [] and [n for d, n in dispatched if d == "move.fold"] == [3, 1]  # not 3, 2, 1
    (plan,) = seen.plans
    assert text(plan, "include/common.h").count("void from_") == 3  # each writer folded on the others' writes


# ------------------------------------------------------------------ type


def _typed(tmp_path: Path, old: str, new: str = "", use: str = "Old *p;\n") -> Any:
    """ga.h defines record Old, gb.h (when given) record New; f1.c uses Old; types.toml rows both."""
    heads = {
        "main/ga.h": "#ifndef GA\n#define GA\n"
        + f"struct Old {{ {old} }};\ntypedef struct Old Old;\nvoid f1(void);\n#endif\n",
        "main/gb.h": "#ifndef GB\n#define GB\n"
        + (f"struct New {{ {new} }};\ntypedef struct New New;\n" if new else "")
        + "void f2(void);\n#endif\n",
        "main/gc.h": "#ifndef GC\n#define GC\nvoid f3(void);\n#endif\n",
    }
    sources = {f"{n}.c": body(n) for n in ("f1", "f2", "f3")}
    sources["f1.c"] = body("f1", '#include "main/gb.h"\n' + use + "s32 g(struct Old *q) { return 0; }\n")
    cfg = world(tmp_path, headers=heads, sources=sources)
    rows = [f'"Old" = {{ declaration = "struct Old {{ {old} }};" }}']
    rows += [f'"New" = {{ declaration = "struct New {{ {new} }};" }}'] if new else []
    (cfg.project.root / "types.toml").write_text(
        "schema = 1\n\n[function]\n\n[global]\n\n[struct]\n" + "\n".join(rows) + "\n"
    )
    return cfg


def test_a_type_move_to_a_free_name_renames_the_tag_in_every_source_header_and_type_row(
    tmp_path: Path, seen: Any
) -> None:
    cfg = _typed(tmp_path, "s32 a; f32 b;")
    result = go(cfg, "type", "Old", "Fresh")
    assert result["refused"] == [] and result["applied"]["type"] == 1
    plan = seen.plans[0]
    assert "struct Fresh { s32 a; f32 b; };\ntypedef struct Fresh Fresh;" in text(plan, "include/main/ga.h")
    assert "Fresh *p;" in text(plan, "src/f1.c") and "struct Fresh *q" in text(plan, "src/f1.c")
    assert '"Fresh" = { declaration = "struct Fresh {' in text(plan, "types.toml")
    assert not any(b"Old" in (data or b"") for data in plan.writes.values()) and "src/f1.c" in seen.measured[0]


def test_a_type_move_onto_a_record_of_the_same_layout_folds_it_and_drops_the_old_body(
    tmp_path: Path, seen: Any
) -> None:
    cfg = _typed(tmp_path, "s32 a; f32 b[0x2];", "s32 a; f32 b[2];")
    result = go(cfg, "type", "Old", "New")
    assert result["refused"] == [] and result["applied"]["type"] == 1
    plan = seen.plans[0]
    ga = text(plan, "include/main/ga.h")
    assert "{" not in ga and "Old" not in ga  # the body is gone, the typedef names New
    assert "New *p;" in text(plan, "src/f1.c") and "include/main/gb.h" not in plan.writes
    types = text(plan, "types.toml")
    assert types.count('"New" =') == 1 and "Old" not in types


def test_a_type_folded_into_a_record_its_header_includes_keeps_one_typedef(tmp_path: Path, seen: Any) -> None:
    cfg = _typed(tmp_path, "s32 a;", "s32 a;")
    gb = cfg.project.root / "include/main/gb.h"
    gb.write_text(gb.read_text().replace("#define GB\n", '#define GB\n#include "main/ga.h"\n'))
    go(cfg, "type", "Old", "New")
    plan = seen.plans[0]
    spelled = {
        p: text(plan, p) if p in plan.writes else (cfg.project.root / p).read_text()
        for p in ("include/main/ga.h", "include/main/gb.h")
    }
    assert sum(t.count("typedef struct New New;") for t in spelled.values()) == 1  # gcc 2 refuses a repeated typedef


def _viewed(tmp_path: Path, old: str, new: str, use: str) -> Any:
    """_typed with the base typedefs a header of the project declares and Old's users reading members."""
    cfg = _typed(tmp_path, old, new, use="Old *p;\nvoid h(void) { " + use + " }\n")
    words = "typedef int s32;\ntypedef short s16;\ntypedef float f32;\ntypedef unsigned char u8;\n"
    (cfg.project.root / "include/ultra.h").write_text(words)
    return cfg


@pytest.mark.parametrize(
    ("old", "new", "use", "words"),
    [
        ("s32 a; f32 b;", "s32 a; s32 b;", "p->b = 0;", ("its b (float)", "b (int)", "differ in type")),
        ("s32 *a;", "u8 *a;", "p->a = 0;", ("differ in pointee",)),
        (
            "s32 a; s32 b;",
            "s32 a; s16 c; s16 d;",
            "p->b = 0;",
            ("its b at 0x4 (width 4) overlaps c (short), d (short)",),
        ),
        ("s32 w[2];", "s32 a; f32 b;", "p->w[1] = 0;", ("its w[1] (int) and b (float) differ in type",)),
        ("s32 w[2];", "s32 a; s32 b;", "p->w[k] = 0;", ("its w at 0x0 (width 8) overlaps a (int), b (int)",)),
        ("s32 a;", "char x[2]; s32 a;", "p->a = 0;", ("its a at 0x0 is not New's a",)),
    ],
)
def test_a_type_fold_by_offset_is_refused_for_a_member_whose_type_differs_or_overlaps(
    tmp_path: Path, seen: Any, old: str, new: str, use: str, words: tuple[str, ...]
) -> None:
    cfg = _viewed(tmp_path, old, new, use)
    found = go(cfg, "type", "Old", "New")["refused"][0]["findings"][0]
    assert found["key"] == "move.type_layout" and all(w in found["reason"] for w in words), found
    assert seen.plans == []


def test_a_partial_view_folds_by_offset_renaming_accesses_filling_padding_and_extending_the_record(
    tmp_path: Path, seen: Any
) -> None:
    old, new = "s32 a; s32 b; s32 c; s32 d;", "s32 x; char pad4[0x8];"
    cfg = _viewed(tmp_path, old, new, "p->a = p->c + p->d;")
    f3 = cfg.project.root / "src/f3.c"
    f3.write_text('#include "main/ga.h"\n' + f3.read_text())  # never spells Old: it newly reaches gb.h through ga.h
    result = go(cfg, "type", "Old", "New")
    assert result["refused"] == [] and result["applied"]["type"] == 1
    plan = seen.plans[0]
    gb, f1 = text(plan, "include/main/gb.h"), text(plan, "src/f1.c")
    assert "s32 x; char pad4[0x4]; s32 c; s32 d;" in " ".join(gb.split())  # b is never read: its space stays
    assert "p->x = p->c + p->d;" in f1 and "New *p;" in f1  # the access by offset takes New's name
    ga = text(plan, "include/main/ga.h")
    assert "{" not in ga and '#include "main/gb.h"' in ga  # the body went: the header that defines New comes in
    assert {"src/f1.c", "src/f3.c"} <= set(seen.measured[0])  # every unit reaching a header whose includes changed


def test_a_real_member_name_is_never_traded_for_a_placeholder_the_placeholder_takes_it(
    tmp_path: Path, seen: Any
) -> None:
    cfg = _viewed(tmp_path, "s32 keys; s32 stride;", "s32 unk0; s32 unk4;", "p->stride = p->keys;")
    f3 = cfg.project.root / "src/f3.c"
    f3.write_text('#include "main/gb.h"\n' + f3.read_text() + "void k(New *n) { n->unk4 = 1; }\n")
    assert go(cfg, "type", "Old", "New")["refused"] == []
    plan = seen.plans[0]
    assert "s32 keys; s32 stride;" in " ".join(text(plan, "include/main/gb.h").split())
    assert "p->stride = p->keys;" in text(plan, "src/f1.c") and "n->stride = 1;" in text(plan, "src/f3.c")


def _landed(seen: Any, cfg: Any, path: str) -> str:
    plan = seen.plans[0]
    return " ".join((text(plan, path) if path in plan.writes else (cfg.project.root / path).read_text()).split())


@pytest.mark.parametrize(
    ("old", "new", "use", "body", "access"),
    [
        # a member New lacks goes into unknown bytes: a placeholder byte array no user of New reads
        ("s32 a; s32 b;", "s32 a; u8 unk4[8];", "p->b = 1;", "s32 a; s32 b; char pad8[0x4];", "p->b = 1;"),
        # an array read only at constant indexes folds element by element: a match, then an insert into padding
        (
            "s32 w[3];",
            "s32 a; s32 b; char pad8[4];",
            "p->w[1] = p->w[0x2];",
            "s32 a; s32 b; s32 unk8;",
            "p->b = p->unk8;",
        ),
        # a member at an element of an array of its type indexes it
        ("s32 a; s32 b;", "s32 arr[2];", "p->b = 1;", "s32 arr[2];", "p->arr[1] = 1;"),
        # an array lying in an array of its element type takes its name, its indexes shifted by its place
        (
            "char pad[8]; s32 slots[1];",
            "s32 x; s32 y; s32 places[4];",
            "p->slots[i] = 1;",
            "s32 places[4];",
            "p->places[i] = 1;",
        ),
        (
            "char pad[0xC]; s32 slots[1];",
            "s32 x; s32 y; s32 places[4];",
            "p->slots[i] = 1;",
            "s32 places[4];",
            "p->places[1 + i] = 1;",
        ),
        # a void * of New takes the pointee its view reads (and a placeholder the real name)
        ("s32 *keys;", "void *unk0;", "p->keys = 0;", "s32 *keys;", "p->keys = 0;"),
        # a void * of Old reads New's typed pointer as it is
        ("void *keys;", "s32 *keys;", "p->keys = 0;", "s32 *keys;", "p->keys = 0;"),
    ],
)
def test_a_partial_view_fold_inserts_missing_members_into_unknown_bytes_and_maps_arrays_by_element(
    tmp_path: Path, seen: Any, old: str, new: str, use: str, body: str, access: str
) -> None:
    cfg = _viewed(tmp_path, old, new, "s32 i;\n" + use)
    result = go(cfg, "type", "Old", "New")
    assert result["refused"] == [], result["refused"]
    gb = _landed(seen, cfg, "include/main/gb.h")
    assert body in gb, gb
    assert access in _landed(seen, cfg, "src/f1.c")


def test_a_member_whose_name_another_member_of_new_holds_takes_the_placeholder_name(tmp_path: Path, seen: Any) -> None:
    cfg = _viewed(tmp_path, "s32 a; s32 width;", "s32 a; s32 unk4; s32 width;", "p->width = 1;")
    assert go(cfg, "type", "Old", "New")["refused"] == []
    assert "s32 a; s32 unk4; s32 width;" in _landed(seen, cfg, "include/main/gb.h")
    assert "p->unk4 = 1;" in _landed(seen, cfg, "src/f1.c")


def test_access_renames_reach_a_unit_a_relayout_composed_from_a_user_text() -> None:
    edited = SimpleNamespace(
        shaped={}, includes={}, accesses=[(("src/a.c",), {"slots": "places"}, frozenset({"Table"}))]
    )
    snapshot = SimpleNamespace(peek=lambda path: None)
    files = {
        "src/a.c": None,
        "src/merged.c": "Table *t;\nvoid f(void) { t->slots[1].x = 0; }\n",
        "src/b.c": "x->slots;",
    }
    out = move._shape(snapshot, files, edited)
    assert "t->places[1].x" in out["src/merged.c"] and "src/b.c" not in out  # b.c spells no Table


def test_unknown_bytes_a_user_of_new_reads_are_never_split(tmp_path: Path, seen: Any) -> None:
    cfg = _viewed(tmp_path, "s32 a; s32 b;", "s32 a; u8 unk4[8];", "p->b = 1;")
    f3 = cfg.project.root / "src/f3.c"
    f3.write_text('#include "main/gb.h"\n' + f3.read_text() + "void k(New *n) { n->unk4[0] = 1; }\n")
    found = go(cfg, "type", "Old", "New")["refused"][0]["findings"][0]
    assert found["key"] == "move.type_layout" and "its b at 0x4 (width 4) overlaps unk4" in found["reason"], found
    assert seen.plans == []


def test_a_fold_renames_every_access_once_even_when_one_member_takes_another_ones_old_name(
    tmp_path: Path, seen: Any
) -> None:
    cfg = _viewed(tmp_path, "s32 b; s32 c;", "s32 c; s32 d;", "p->b = p->c;")
    assert go(cfg, "type", "Old", "New")["refused"] == []
    assert "p->c = p->d;" in _landed(seen, cfg, "src/f1.c")  # b -> c and c -> d, never b -> c -> d


def test_a_fold_proves_only_the_units_that_spell_a_changed_name_of_a_shared_header(tmp_path: Path, seen: Any) -> None:
    cfg = _viewed(tmp_path, "s32 a; s32 b;", "s32 a; char pad4[4];", "p->b = 1;")
    f3 = cfg.project.root / "src/f3.c"
    f3.write_text('#include "main/gb.h"\n' + f3.read_text())  # includes New's header, spells nothing of it
    f2 = cfg.project.root / "src/f2.c"
    f2.write_text('#include "main/gb.h"\n' + f2.read_text() + "void k(New *n) { }\n")
    assert go(cfg, "type", "Old", "New")["refused"] == []
    assert "src/f2.c" in seen.measured[0] and "src/f3.c" not in seen.measured[0]


@pytest.mark.parametrize(
    ("missing", "changed"),
    [
        (("version a: 4 bytes differ at +0x12 .text",), True),
        (("version a: .text size 2136 != 2132",), True),
        (("version a: compile.error: conflicting types for `f'",), False),
        (("version a: unresolved D_1",), False),
        ((), False),
    ],
)
def test_only_a_proof_that_built_and_changed_bytes_is_evidence_of_a_file_boundary(missing, changed) -> None:
    assert move._bytes_changed(Finding("native.gap", "x", missing=missing)) is changed


def test_the_boundaries_of_a_merged_unit_are_where_its_old_files_began(monkeypatch: Patch) -> None:
    olds = {"fa": "src/a.c", "fb": "src/a.c", "fc": "src/c.c", "fd": "src/d.c"}
    monkeypatch.setattr(move.layout, "unit_of", lambda s, n: SimpleNamespace(path=olds[n]))
    merged = SimpleNamespace(members=("fa", "fb", "fc", "fd"))
    assert move._boundaries(None, merged) == {"fc", "fd"}  # type: ignore[arg-type]


def test_a_round_whose_proofs_find_file_boundaries_applies_the_same_moves_again_with_them(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    calls: list[list[str]] = []

    def apply(config: Any, moves: Any, unproved: bool = False, boundaries: Any = (), keep: Any = (), *_: Any) -> Any:
        calls.append(list(boundaries))
        found = {"refused": [], "moves": len(moves), "message": "x", "files": []}
        return {**found, "boundaries": ["fc"], "refused": [{"index": 0}]} if not boundaries else found

    monkeypatch.setattr(move, "_apply", apply)
    result = go(world(tmp_path), "relayout")
    assert calls == [[], ["fc"]] and result["boundaries"] == ["fc"] and result["refused"] == []


def test_proof_found_boundaries_reapply_a_batch_a_bounded_number_of_times_then_its_refusals_stand(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    calls: list[int] = []

    def apply(config: Any, moves: Any, unproved: bool = False, boundaries: Any = (), keep: Any = (), *_: Any) -> Any:
        calls.append(len(moves))
        row = {"index": 0, "move": {"op": "file", "subject": "s0", "target": "t0"}, "findings": [{"key": "k"}]}
        if len(moves) == 1:  # the move every proof blames, each time with a boundary no earlier round found
            return {"refused": [row], "moves": 1, "message": "x", "boundaries": [f"b{len(calls)}"]}
        return {"refused": [], "moves": len(moves), "message": "x", "files": ["src/t1.c"]}  # the rest lands

    monkeypatch.setattr(move, "_apply", apply)
    stdin(monkeypatch, json.dumps([{"op": "file", "subject": f"s{i}", "target": f"t{i}"} for i in range(1)]))
    result = move.run(world(tmp_path), {"op": "-", "partial": "1"})
    assert calls == [1] * (move._REAPPLY + 1) and [r["index"] for r in result["refused"]] == [0]
    assert len(result["boundaries"]) == move._REAPPLY + 1  # every boundary is kept for the map though


def test_a_stream_whose_proof_cascades_unit_by_unit_keeps_each_failing_unit_and_lands_the_rest(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    # the cutover trace (inv 000789ea): 8156 moves; each proof found one more not_exact unit once its neighbour was
    # dropped, and every chained round re-proved the whole rest, so after 254 rounds nothing had landed
    cascade = ["src/u1.c", "src/u2.c", "src/u3.c"]
    calls: list[tuple[bool, list[str]]] = []

    def apply(
        config: Any, moves: Any, unproved: bool = False, boundaries: Any = (), keep: Any = (), per_unit: Any = False
    ) -> Any:
        calls.append((per_unit, sorted(keep)))
        if failing := [u for u in cascade if u not in keep][:1]:
            found = {"key": "move.not_exact", "reason": "1 bytes differ at +0 .text", "unit": failing[0]}
            return {"keep": {failing[0]: [found]}, "bytes": {failing[0]: 8}, "boundaries": []}
        return {"refused": [], "moves": len(moves), "message": "x", "files": ["src/a.c"], "deleted": []}

    monkeypatch.setattr(move, "_apply", apply)
    stdin(monkeypatch, json.dumps([{"op": "member", "subject": f"f{i}", "target": "ga"} for i in range(8)]))
    cfg = world(tmp_path)
    result = move.run(cfg, {"op": "-", "partial": "1"})
    assert [u for u, _ in calls] == [True] * 4 and calls[-1][1] == cascade  # one unit kept per pass, then it lands
    assert result["refused"] == [] and result["files"] == ["src/a.c"] and sorted(result["kept"]) == cascade
    assert result["census"][0]["count"] == 3 and result["converged"]
    rows = [json.loads(x) for x in (cfg.project.root / ".unbake/move/rounds.jsonl").read_text().splitlines()]
    assert [list(r["kept"]) for r in rows] == [["src/u1.c"], ["src/u2.c"], ["src/u3.c"], []]  # live, per round


def test_a_kept_unit_keeps_every_unit_it_shares_a_new_unit_with_in_one_pass() -> None:
    # the cutover trace (inv 80f702e0): claims mismatches kept 12, 90, 7, 7, 7 new units pass after pass, because
    # keeping a unit changed each new unit its members joined; the chain a-b-c is kept at once, d and e land
    from types import SimpleNamespace as N

    def units(**spec: str) -> dict[str, Any]:  # name="members/group"
        return {
            f"src/{k}.c": N(path=f"src/{k}.c", members=tuple(v.split("/")[0].split()), group=v.split("/")[-1])
            for k, v in spec.items()
        }

    old = N(layout=N(units=units(a="a1 a2/g1", b="b1 b2/g2", c="c1/g3", d="d1/g4", e="e1/g5", f="f1/g1", y="y1/g2")))
    new = N(units=units(n1="a1 b1/g1", n2="b2 c1/g2", n3="d1/g4", n4="a2/g1", n5="e1/g5", n6="f1/g1", n7="y1/g2"))
    why: Any = move.defaultdict(list, {"src/a.c": [move.Finding("move.not_exact", "claims")]})
    move._keep_together(old, new, why)  # type: ignore[arg-type]
    assert (
        sorted(why) == ["src/a.c", "src/b.c", "src/c.c", "src/f.c", "src/y.c"]
        and "share src/n2.c" in why["src/c.c"][0].reason
    )
    # f shares no new unit with a, but a kept a reflows its module g1 (the eu_padding F8CAC, FAB6C ring of round 8)
    assert "module g1" in why["src/f.c"][0].reason
    assert "module g2" in why["src/y.c"][0].reason  # b came in through n1, then its module came along (fixpoint)


def test_a_claim_shift_at_a_kept_units_seam_is_bookkeeping_and_any_other_change_still_fails() -> None:
    # round 8 of inv 80f702e0: each kept unit moved the claims seam onto its successor (a padding unit, a plain
    # function, a rodata unit), one ring per pass; the bytes proved exact on both sides of every seam
    from unbake.contracts import Claim

    pad, fn, ro = (
        "src/func_8040AB6C_eu_padding_FAB6C.c",
        "src/func_8043AB40_de.c",
        "src/rodata.func_8044B7C0_de.800C4F50.c",
    )
    kept = ["src/func_80408CAC_eu_padding_F8CAC.c", "src/func_8043A890_de.c", "src/rodata.func_8044B600_de.800C4F04.c"]
    before = [
        Claim(kept[0], "eu", ".data", 0x100, 0x110),
        Claim(pad, "eu", ".data", 0x110, 0x120),
        Claim(kept[1], "de", ".text", 0x200, 0x2B0),
        Claim(fn, "de", ".text", 0x2B0, 0x300),
        Claim(kept[2], "de", ".rodata", 0x400, 0x44C),
        Claim(ro, "de", ".rodata", 0x450, 0x460),
    ]
    after = [
        *before[:1],
        Claim(pad, "eu", ".data", 0x118, 0x120),
        *before[2:5],
        Claim(ro, "de", ".rodata", 0x44C, 0x460),
    ]
    assert move._claims(after) != move._claims(before)  # the union moved: the old rule kept pad and ro
    assert move._unseamed(before, after, kept) == set() and move._unseamed(before, after, []) == {
        kept[0],
        pad,
        kept[2],
        ro,
    }
    far = [*after, Claim(fn, "de", ".text", 0x900, 0x910)]  # bytes no kept unit borders: only fn still fails
    assert move._unseamed(before, far, kept) == {fn}


def test_a_batch_met_again_with_nothing_new_kept_is_refused_not_applied_again(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    calls: list[int] = []

    def apply(config: Any, moves: Any, unproved: bool = False, boundaries: Any = (), keep: Any = (), *_: Any) -> Any:
        calls.append(len(moves))  # keeps the same unit every time: the old loop applied this batch forever
        return {"refused": [], "moves": len(moves), "message": "x", "keep": {"src/a.c": [{"key": "k"}]}}

    monkeypatch.setattr(move, "_apply", apply)
    stdin(monkeypatch, json.dumps([{"op": "file", "subject": "s0", "target": "t0"}]))
    result = move.run(world(tmp_path), {"op": "-", "partial": "1"})
    assert calls == [1, 1] and result["refused"][0]["findings"][0]["key"] == "move.rounds"


@pytest.mark.parametrize("missing", ["size", "compile.error: x.c:3: conflicting types for `g'"])
def test_a_relayout_keeps_the_old_files_of_every_unit_whose_proof_fails_and_lists_them_by_class(
    tmp_path: Path, seen: Any, monkeypatch: Patch, missing: str
) -> None:
    proofs = move.publish._proofs

    def merged_fails(snapshot: Any, affected: Any) -> Any:
        merged = [p for p in affected if len(snapshot.layout.units[p].members) > 1]
        seen.exact = not merged  # whatever the stage, a merged unit that fails keeps its old files
        made, gaps = proofs(snapshot, affected)
        return made, tuple(dataclasses.replace(g, missing=(f"version a: {missing}",)) for g in gaps)

    monkeypatch.setattr(move.publish, "_proofs", merged_fails)
    cfg = world(
        tmp_path,
        groups=[group("ga", "f1", "f2", "f3", source="src/f1.c")],
        units=[unit("src/f1.c", "ga", "f1"), unit("src/f2.c", "ga", "f2"), unit("src/f3.c", "ga", "f3")],
    )
    result = go(cfg, "relayout")
    assert (
        result["refused"] == [] and result["converged"] and sorted(result["kept"]) == [f"src/f{i}.c" for i in (1, 2, 3)]
    )
    assert result["boundaries"] == ["f1", "f2", "f3"] and len(result["census"]) == 1
    assert result["census"][0]["count"] == 3 and result["census"][0]["class"].startswith("move.not_exact: ")
    doc = text(seen.plans[-1], "layout.toml")
    assert 'boundaries = ["f1", "f2", "f3"]' in doc and doc.count("[[unit]]") == 3  # later relayouts never merge them
    census = json.loads((cfg.project.root / ".unbake/move/census.json").read_text())
    assert census == result["census"]


def test_a_converged_move_exits_zero_with_its_leftovers_as_findings() -> None:
    result = {"refused": [{"findings": [asdict_finding()]}], "converged": True}
    assert move_cli_code(result) == 0 and move_cli_code({**result, "converged": False}) == 1


def asdict_finding() -> dict[str, Any]:
    return dataclasses.asdict(Finding("move.not_exact", "x", unit="src/f1.c"))


def move_cli_code(result: dict[str, Any]) -> int:
    from unbake import cli

    return cli._failed(result, cli._refused(result))


def _rounds(monkeypatch: Patch, refuse: Any) -> list[list[str]]:
    """_apply replaced: each round's subjects recorded; `refuse(subjects)` gives per subject its findings (and
    whether the plan refused them), the rest land."""
    applied: list[list[str]] = []

    def apply(config: Any, moves: Any, *_: Any) -> Any:
        subjects = [m.subject for m in moves]
        applied.append(subjects)
        bad, planned = refuse(subjects)
        rows = [
            {
                "index": m.index,
                "move": {"op": m.op, "subject": m.subject, "target": m.target},
                "findings": bad[m.subject],
            }
            for m in moves
            if m.subject in bad
        ]
        return {
            "refused": rows,
            "moves": len(moves),
            "message": "x",
            "files": [f"src/{s}.c" for s in subjects if s not in bad],
            **({"planned": True} if planned and rows else {}),
        }

    monkeypatch.setattr(move, "_apply", apply)
    return applied


def test_partial_refuses_every_move_a_round_blames_at_once_and_lands_the_rest_the_next_round(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    # the 2026-10-15 fakes move, rounds 9-21: one kept unit (code_802B7520_5.c) failing again blamed 134 moves; the
    # old loop bisected them 67, 33, 16, 8, 4, 2, 1 per round on 3-4 cores for 50 minutes
    broken = {"key": "move.not_exact", "unit": "src/sdk/libgultra/part_800C7F20/code_802B7520_5.c", "reason": "size"}
    blamed = {f"s{i}" for i in range(1, 12, 2)}

    def refuse(subjects: list[str]) -> Any:
        return ({s: [broken] for s in subjects if s in blamed} if blamed & set(subjects) else {}), False

    applied = _rounds(monkeypatch, refuse)
    stdin(monkeypatch, json.dumps([{"op": "symbol", "subject": f"s{i}", "target": f"t{i}"} for i in range(12)]))
    result = move.run(world(tmp_path), {"op": "-", "partial": "1"})
    assert [r["index"] for r in result["refused"]] == sorted(int(s[1:]) for s in blamed)
    assert applied == [[f"s{i}" for i in range(12)], [f"s{i}" for i in range(0, 12, 2)]] and result["rounds"] == 2
    assert sorted(result["files"]) == sorted(f"src/s{i}.c" for i in range(0, 12, 2))


def test_partial_retries_alone_a_move_the_plan_refused_beside_others(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    def refuse(subjects: list[str]) -> Any:  # s1 plans only on the tree the others leave
        clash = {"key": "move.type_layout", "reason": "renamed earlier in the batch"}
        return ({"s1": [clash]} if "s1" in subjects and len(subjects) > 1 else {}), True

    applied = _rounds(monkeypatch, refuse)
    stdin(monkeypatch, json.dumps([{"op": "symbol", "subject": f"s{i}", "target": f"t{i}"} for i in range(3)]))
    result = move.run(world(tmp_path), {"op": "-", "partial": "1"})
    assert result["refused"] == [] and applied == [["s0", "s1", "s2"], ["s0", "s2"], ["s1"]]


def test_a_type_move_of_an_unknown_type_or_onto_a_symbol_is_refused(tmp_path: Path, seen: Any) -> None:
    cfg = _typed(tmp_path, "s32 a;")
    assert go(cfg, "type", "Nope", "X")["refused"][0]["findings"][0]["key"] == "move.unknown"
    assert go(cfg, "type", "Old", "f2")["refused"][0]["findings"][0]["key"] == "move.exists"
    assert go(cfg, "type", "Old", "1bad")["refused"][0]["findings"][0]["key"] == "move.name"
    assert seen.plans == []


def test_a_type_folds_into_a_name_an_earlier_move_of_the_stream_gave(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    cfg = _typed(tmp_path, "s32 a;", "s32 a;")
    stdin(
        monkeypatch,
        '{"op": "type", "subject": "New", "target": "Real"}\n{"op": "type", "subject": "Old", "target": "Real"}\n',
    )
    result = go(cfg, "-")
    assert result["refused"] == [] and result["applied"]["type"] == 2
    plan = seen.plans[0]
    assert "struct Real { s32 a; };" in text(plan, "include/main/gb.h") and "{" not in text(plan, "include/main/ga.h")
    assert "Real *p;" in text(plan, "src/f1.c")


@pytest.mark.parametrize(
    ("statement", "rows"),
    [
        ("struct S { s32 a, *b; char pad[0x10]; };", [("a", "int@"), ("b", "int*@"), ("pad", "char@[16]")]),
        ("struct S { struct { s32 x; } in; u8 c; };", [("", "struct { int x; } in"), ("c", "unsignedchar@")]),
        ("typedef struct S S;", None),
    ],
)
def test_members_reads_each_member_of_a_record_body_in_order(statement: str, rows: Any) -> None:
    assert headers.members(statement) == rows


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("struct A;\nstruct A { s32 a; };\ntypedef struct A A;\n", "struct A;\ntypedef struct A A;\n"),
        ("typedef struct A { s32 a; } A;\nstruct B { s32 b; };\n", "typedef struct A A;\nstruct B { s32 b; };\n"),
    ],
)
def test_drop_record_removes_only_the_body_of_the_tag(text: str, kept: str) -> None:
    assert headers.drop_record(text, "A") == kept


def test_a_typedef_alias_folded_into_its_record_leaves_one_types_row(tmp_path: Path, seen: Any) -> None:
    cfg = _typed(tmp_path, "s32 a;", "s32 a;")
    ga = cfg.project.root / "include/main/ga.h"
    ga.write_text(ga.read_text().replace("void f1(void);", "typedef struct New Alias;\nvoid f1(void);"))
    types = cfg.project.root / "types.toml"
    types.write_text(types.read_text() + '"Alias" = { declaration = "typedef struct New Alias;" }\n')
    result = go(cfg, "type", "Alias", "New")
    assert result["refused"] == [] and text(seen.plans[0], "types.toml").count('"New" =') == 1


def test_a_type_move_blames_only_the_units_that_spell_the_type_not_every_includer_of_its_header(
    tmp_path: Path, seen: Any
) -> None:
    cfg = _typed(tmp_path, "s32 a;")
    f2 = cfg.project.root / "src/f2.c"
    f2.write_text('#include "main/ga.h"\n' + f2.read_text())  # includes the header that spells Old, never Old itself
    snapshot = layout.capture(cfg)
    users = move._type_users(snapshot, "Old")
    assert "f1" in users and "f2" not in users and "f3" not in users


def test_a_relocate_cycle_refusal_blames_the_moves_that_merge_its_header_not_the_stream():
    # queue 2026-10-15 07:15 (RELAYOUT4): one move.cycle bisected an 8,156-move stream for 30 min
    from types import SimpleNamespace as NS

    from unbake.contracts import Move

    groups = {"a": NS(name="a", header="include/a.h"), "b": NS(name="b", header="include/b.h")}
    edited = NS(layout=NS(groups=groups), touched={0: {"a"}, 1: {"b"}, 2: {"b"}})
    moves = [Move(0, "module", "a", "x"), Move(1, "module", "b", "y"), Move(2, "member", "f", "b")]
    assert move._cycle_moves(edited, moves, "include/a.h") == [moves[0]]
    assert move._cycle_moves(edited, moves, "include/none.h") == moves


# ------------------------------------------------------------------ fast, converging rounds (2026-10-15 fakes move)


def _renamed_in(snapshot: Any, header: str, name: str) -> bool:
    return name.encode() in (snapshot.peek(header) or b"")


def test_a_header_that_no_longer_compiles_alone_refuses_the_moves_that_changed_it_and_proves_no_includer(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    # the 2026-10-15 fakes move: one broken shared header failed ~13,000 includer proofs in round 1 (1.2 GB of log)
    seen.broken = lambda s, h: [f"{h}:3: parse error before `g1'"] if _renamed_in(s, h, "g1") else []
    cfg = world(tmp_path)
    splat_rows(cfg)
    stdin(monkeypatch, json.dumps([{"op": "symbol", "subject": f"f{i}", "target": f"g{i}"} for i in (1, 2)]))
    result = move.run(cfg, {"op": "-", "partial": "1"})
    (row,) = result["refused"]
    assert row["move"]["subject"] == "f1" and row["findings"][0]["key"] == "move.header"
    assert row["findings"][0]["path"] == "include/main/ga.h" and "parse error" in row["findings"][0]["reason"]
    # round 1 compiled each changed header once and proved nothing; round 2 landed f2 and proved only its units
    assert result["rounds"] == 2 and len(seen.measured) == 1 and len(seen.plans) == 1
    assert seen.probed.count("include/main/ga.h") == 2 and "src/f1.c" not in seen.measured[0]  # after, then before


def test_a_header_that_failed_alone_before_the_move_too_is_left_to_the_includer_proofs(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    seen.broken = lambda s, h: [f"{h}:1: storage size of `weaponModel' isn't known"]  # broken on both sides
    cfg = world(tmp_path)
    splat_rows(cfg)
    result = go(cfg, "symbol", "f1", "g1")
    assert result["refused"] == [] and len(seen.plans) == 1 and seen.measured


def test_a_header_failing_only_through_a_broken_changed_header_it_includes_blames_that_header(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    cfg = world(tmp_path)
    before = layout.capture(cfg)
    after = layout.overlay(before, {"include/main/ga.h": b"bad;\n", "include/main/gb.h": b'#include "main/ga.h"\n'})
    error = "build/views/0123456789abcdef/include/main/ga.h:1: parse error"
    seen.broken = lambda s, h: [error] if s.peek("include/main/ga.h") == b"bad;\n" else []
    broken = move._broken_headers(before, after, ["include/main/ga.h", "include/main/gb.h"])
    assert list(broken) == ["include/main/ga.h"] and error in broken["include/main/ga.h"].reason


def test_kept_units_that_still_fail_refuse_the_moves_that_changed_what_they_read_never_the_first_move(
    tmp_path: Path, seen: Any
) -> None:
    # the 2026-10-15 fakes move: rounds 2 and 3 (7 and 24 min) each refused the stream's first move, unrelated to
    # the kept unit src/code_802BAC64.c that failed again on the headers it reads
    snapshot = layout.capture(world(tmp_path))
    moves = (Move(0, "symbol", "f2", "g2"), Move(1, "symbol", "f3", "g3"))
    failing = {"src/f1.c": [Finding("move.not_exact", "x", unit="src/f1.c", missing=("e" * 90_000,))]}
    asked: list[list[str]] = []

    def stuck(units: dict[str, Any]) -> list[Move]:
        asked.append(sorted(units))
        return list(moves[1:])

    result = move._kept(snapshot, snapshot.layout, failing, ["src/f1.c"], moves, stuck)
    assert asked == [["src/f1.c"]] and result["unkept"] and [r["index"] for r in result["refused"]] == [1]
    assert len(result["refused"][0]["findings"][0]["missing"][0]) < move._SAID + 30  # capped, never 90 KB


def test_a_files_changed_lines_blame_the_moves_they_spell() -> None:
    before = SimpleNamespace(peek=lambda p: b"#include <x.h>\nvoid Old(Thing *t);\nint keep;\n")
    after = SimpleNamespace(peek=lambda p: b"#include <x.h>\nvoid New(Thing *t);\nint keep;\n")
    moves = [Move(0, "type", "Other", "Else"), Move(1, "symbol", "Old", "New"), Move(2, "type", "keep", "k")]
    edited = SimpleNamespace(layout=SimpleNamespace(groups={}), touched={})
    assert move._file_moves(before, after, edited, moves, "include/a.h") == [moves[1]]  # type: ignore[arg-type]


def test_a_kept_unit_failing_through_an_added_include_blames_the_moves_that_added_it_not_the_chain() -> None:
    # the 2026-10-15 leftover-rows rerun, round 3: kept src/code_802BAC64.c failed through the chain
    # model_instance.h -> types_*.h -> code_802BAC64.h and its module header's new include of a types header made a
    # cycle; blaming every move spelled in a chain header's changed lines refused 741 of 751 rows
    types = "".join(f"typedef struct Rec{i} Name{i};\n" for i in range(40))
    old = {
        "include/code_802BAC64.h": '#include "shared/model_instance.h"\nvoid f(void *p);\n',
        "include/common/types_1dc.h": "#ifndef T\n#define T\n" + types + "#endif\n",
    }
    new = {
        "include/code_802BAC64.h": '#include "shared/model_instance.h"\n#include "common/types_1dc.h"\n'
        "void f(Name7 *p);\n",
        "include/common/types_1dc.h": '#ifndef T\n#define T\n#include "code_802BAC64.h"\n'
        + types.replace("Rec", "Moved")
        + "#endif\n",
    }
    before = SimpleNamespace(peek=lambda p: old[p].encode() if p in old else None)
    after = SimpleNamespace(peek=lambda p: new[p].encode() if p in new else None)
    moves = [Move(i, "type", f"Rec{i}", f"Moved{i}") for i in range(40)] + [Move(40, "type", "Old7", "Name7")]
    said = (
        "version de: compile.error: In file included from build/views/a6/include/shared/model_instance.h:4,\n"
        "                 from /root/include/common/types_1dc.h:3,\n"
        "                 from include/code_802BAC64.h:2,\n"
        "                 from src/code_802BAC64.c:1,\n"
        "include/acmd.h:8: parse error before `Vector4f'\n"
    )
    changed = set(new)
    blamed = move._edge_moves(before, after, moves, said, "/root/", changed)  # type: ignore[arg-type]
    assert blamed == [moves[40]]  # the move that spelled Name7 into the module header, which added the include
    edited = SimpleNamespace(layout=SimpleNamespace(groups={}), touched={})
    chain = move._file_moves(before, after, edited, moves, "include/common/types_1dc.h")  # type: ignore[arg-type]
    assert len(chain) == 41  # what the chain blame refused before: every move


def _cascade(monkeypatch: Patch, stop_at: int = 0) -> list[list[str]]:
    """_apply replaced by the unit-by-unit cascade of the cutover trace; raises KeyboardInterrupt on call `stop_at`."""
    order, calls = ["src/u1.c", "src/u2.c", "src/u3.c"], []

    def apply(config: Any, moves: Any, unproved: bool = False, boundaries: Any = (), keep: Any = (), *_: Any) -> Any:
        calls.append(sorted(keep))
        if len(calls) == stop_at:
            raise KeyboardInterrupt  # the command is killed mid-round
        if failing := [u for u in order if u not in keep][:1]:
            found = {"key": "move.not_exact", "reason": "size", "unit": failing[0]}
            return {"keep": {failing[0]: [found]}, "bytes": {failing[0]: 8}, "boundaries": []}
        return {"refused": [], "moves": len(moves), "message": "x", "files": ["src/a.c"], "deleted": []}

    monkeypatch.setattr(move, "_apply", apply)
    return calls


def test_a_killed_move_resumes_after_its_last_round_and_proves_no_kept_unit_again(
    tmp_path: Path, seen: Any, monkeypatch: Patch
) -> None:
    cfg = world(tmp_path)
    stream = json.dumps([{"op": "member", "subject": f"f{i}", "target": "ga"} for i in (2, 3)])
    calls = _cascade(monkeypatch, stop_at=3)
    stdin(monkeypatch, stream)
    with pytest.raises(KeyboardInterrupt):
        move.run(cfg, {"op": "-", "partial": "1"})
    assert calls == [[], ["src/u1.c"], ["src/u1.c", "src/u2.c"]]
    calls = _cascade(monkeypatch)
    stdin(monkeypatch, stream)
    result = move.run(cfg, {"op": "-", "partial": "1"})
    # the rerun starts where the killed one was: both kept units held, round numbers go on, nothing found twice
    assert calls == [["src/u1.c", "src/u2.c"], ["src/u1.c", "src/u2.c", "src/u3.c"]] and result["rounds"] == 4
    assert sorted(result["kept"]) == ["src/u1.c", "src/u2.c", "src/u3.c"] and result["files"] == ["src/a.c"]
    assert not (cfg.project.root / move._RESUME).exists()  # done: the next move starts afresh


def test_another_stream_or_a_changed_tree_starts_afresh(tmp_path: Path, seen: Any, monkeypatch: Patch) -> None:
    cfg = world(tmp_path)

    def killed(subject: str, stop_at: int = 0) -> list[list[str]]:
        calls = _cascade(monkeypatch, stop_at)
        stdin(monkeypatch, json.dumps([{"op": "member", "subject": subject, "target": "ga"}]))
        if stop_at:
            with pytest.raises(KeyboardInterrupt):
                move.run(cfg, {"op": "-", "partial": "1"})
        else:
            move.run(cfg, {"op": "-", "partial": "1"})
        return calls

    assert killed("f2", stop_at=2) == [[], ["src/u1.c"]]  # journalled with u1 kept
    assert killed("f3", stop_at=2) == [[], ["src/u1.c"]]  # another stream: found u1 afresh
    (cfg.project.root / "src/f9.c").write_text("int f9;\n")  # the tree moved under the journal
    assert killed("f3")[0] == []
