"""View contract tests with an in-memory cache and canned preprocessor output."""

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import fixture
import pytest

from unbake import store, view
from unbake.contracts import (
    Config,
    Finding,
    Host,
    LayoutMap,
    Project,
    Recipe,
    Refusal,
    Snapshot,
    UnitSpec,
    digest,
    sha256,
)


_STORE = (store.get, store.put, store.put_many, store.content)  # the real cache, before a lane mocks it


def _sha(data):
    return sha256(data)


@pytest.fixture
def lane(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "src/unit.c").write_bytes(b'#include "active.h"\nint value;\n')
    (root / "include/active.h").write_bytes(b"typedef int active;\n")
    (root / "include/z.h").write_bytes(b"typedef int z;\n")
    versions = ("a", "b", "c", "d", "e")
    project = Project(
        root=root,
        id="fixture",
        name="fixture",
        title="Fixture",
        versions=versions,
        names_from="a",
        toolchain="fake",
        build={},
        version_files={},
        version_macros={v: (f"VERSION_{v.upper()}",) for v in versions},
        resident={},
        origins={},
        digest=digest("project"),
    )
    host = Host(
        cores=2,
        workers=2,
        memory_parent_bytes=1024,
        memory_worker_bytes=1024,
        cache_dir=tmp_path / "cache",
        cache_max_bytes=1 << 30,
        toolchain_root=tmp_path / "tools",
        tools={},
        sdk_catalog=None,
        serial_seconds=2,
        serial_cores=2,
        pool_fill=0.8,
        pool_fanout=2,
        author=("test", "test@example.invalid"),
        origins={},
        digest=digest("host"),
    )
    config = Config(project, host, digest("config"))
    unit = UnitSpec("src/unit.c", "c", "group", ("value",), "fake", {})
    layout = LayoutMap({}, {}, {unit.path: unit}, digest("layout"), (), {})
    snapshot = Snapshot(config, "0" * 40, layout, {}, {}, digest("snapshot"))
    recipe = Recipe("fake", (), (), (), digest("recipe"))
    state = SimpleNamespace(
        root=root,
        snapshot=snapshot,
        unit=unit,
        recipe=recipe,
        cache={},
        events=[],
        made={},
        homes=[],
        named=None,
        text=f'# 1 "{root / unit.path}"\nint value;\n',
        result=fixture.native_result(),
        findings=(),
    )

    def preprocess(source, recipe, version, include, *, out):
        home = out.parent.parent  # the directory the view is made in, gone when the view returns
        state.made = {
            p.relative_to(home).as_posix(): (p.read_bytes(), p.is_symlink())
            for p in sorted(home.rglob("*"))
            if p.is_file() and ".work" not in p.parts
        }
        state.homes.append(home)
        out.write_text(state.text.replace(str(state.named), str(home)) if state.named else state.text)
        return state.result

    state.preprocess = Mock(side_effect=preprocess)
    state.diagnose = Mock(side_effect=lambda result: state.findings)
    state.toolchain = Mock(return_value=SimpleNamespace(preprocess=state.preprocess, diagnose=state.diagnose))

    def get(config, kind, key):
        return state.cache.get((kind, key))

    def put(config, kind, key, value):
        state.cache[kind, key] = value

    def put_many(config, kind, rows):
        for key, value in rows:
            put(config, kind, key, value)

    state.put = Mock(side_effect=put)
    state.content = Mock(return_value=SimpleNamespace(count=lambda kind, hit: state.events.append((kind, hit))))
    state.pool = Mock(side_effect=lambda config, name, function, items, key=None: [function(item) for item in items])
    state.stage = Mock(side_effect=lambda name: nullcontext())
    monkeypatch.setattr(view.adapters, "toolchain", state.toolchain)
    monkeypatch.setattr(view, "_identity", lambda config, recipe: "tool-bytes")
    monkeypatch.setattr(view.store, "get", get)
    monkeypatch.setattr(view.store, "put", state.put)
    monkeypatch.setattr(view.store, "put_many", put_many)
    monkeypatch.setattr(view.store, "content", state.content)
    monkeypatch.setattr(view.pool, "map", state.pool)
    monkeypatch.setattr(view.effort, "stage", state.stage)
    return state


def _get(lane, snapshot=None, recipe=None):
    return view.get(snapshot or lane.snapshot, lane.unit, "a", recipe or lane.recipe)


def _overlay(snapshot, writes):
    return replace(snapshot, overlays=writes, digest=digest((snapshot.digest, writes)))


def _dir(lane, snapshot):
    """Where a view that reads an overlaid file is made: named by the view key, never by the snapshot."""
    reads = view.closure(snapshot, lane.unit, "a")
    key = digest(
        (
            lane.unit.path,
            _sha(snapshot.read(lane.unit.path)),
            lane.recipe.digest,
            "a",
            snapshot.config.project.version_macros["a"],
            view._identity(snapshot.config, lane.recipe),
            reads,
            True,
            snapshot.tree,
        )
    )
    return lane.root / "build/views" / key[:16]


def test_cardinality_equals_request(lane):
    requested = ("e", "c", "a", "d", "b")
    result = view.all_versions(lane.snapshot, lane.unit, lane.recipe, requested)
    assert list(result) == list(requested)
    assert len(result) == 5
    assert len({item.text for item in result.values()}) == 1
    assert [item.version for item in result.values()] == list(requested)
    assert lane.preprocess.call_count == 5
    config, name, function, items = lane.pool.call_args.args
    assert config is lane.snapshot.config
    assert name == "view.all_versions"
    assert function.__module__ == "unbake.view"
    assert function.__qualname__ == "_one"
    assert items == [(lane.snapshot, lane.unit, version, lane.recipe) for version in requested]
    assert lane.stage.call_args_list[0].args == ("view.all_versions",)


def test_all_versions_empty(lane):
    assert view.all_versions(lane.snapshot, lane.unit, lane.recipe, ()) == {}
    lane.preprocess.assert_not_called()


def test_cache_hit_second_call(lane):
    first = _get(lane)
    second = _get(lane)
    assert first == second
    lane.preprocess.assert_called_once()
    assert lane.events == [("view", False), ("view", True)]
    lane.toolchain.assert_called_once_with(lane.snapshot.config, lane.recipe.toolchain)
    assert lane.stage.call_count == 2
    assert all(call.args == ("view.get",) for call in lane.stage.call_args_list)
    args, kwargs = lane.preprocess.call_args
    assert args == (lane.root / lane.unit.path, lane.recipe, "a", [lane.root / "include", lane.root / "src"])
    assert kwargs["out"].name == "view.i"


@pytest.mark.parametrize("overlay", [False, True], ids=["disk-header", "overlay-header"])
def test_header_change_invalidates(lane, overlay):
    header = "include/active.h"
    lane.text = f'# 1 "{header}"\ntypedef int active;\n# 1 "src/unit.c"\nint value;\n'
    first = _get(lane)
    changed = b"typedef long active;\n"
    snapshot = lane.snapshot
    if overlay:
        snapshot = _overlay(snapshot, {header: changed})
    else:
        (lane.root / header).write_bytes(changed)
        view.effort._memo.clear()  # the next command reads the file again
    lane.text = lane.text.replace("typedef int active;", "typedef long active;")
    second = _get(lane, snapshot)
    assert lane.preprocess.call_count == 2
    assert second.key != first.key
    assert dict(second.dependencies)[header] == _sha(changed)
    assert _get(lane, snapshot) == second
    assert lane.preprocess.call_count == 2
    assert json.loads(lane.cache["view", second.key])["deps"] == [list(row) for row in second.dependencies]


@pytest.mark.parametrize("source_overlaid", [False, True])
def test_overlay_paths_mapped_back(lane, source_overlaid):
    writes = {"include/active.h": b"typedef long active;\n", "include/deleted.h": None}
    if source_overlaid:
        writes[lane.unit.path] = b'#include "active.h"\nint changed;\n'
    snapshot = _overlay(lane.snapshot, writes)
    overlay = _dir(lane, snapshot)
    source = (overlay if source_overlaid else lane.root) / lane.unit.path
    outside = lane.root.parent / "system.h"
    lane.text = (
        f'# 7 "{overlay / "include/active.h"}" 1\ntypedef long active;\n'
        f'# 3 "{outside}"\nint external;\n# 0 "<built-in>"\nint builtin;\n'
        f'# 9 "{source}" 2\nint changed;\n'
    )
    result = _get(lane, snapshot)
    assert result.lines == (
        (2, "include/active.h", 7),
        (4, str(outside), 3),
        (6, "<built-in>", 0),
        (8, lane.unit.path, 9),
    )
    assert result.dependencies == (
        (lane.unit.path, _sha(snapshot.read(lane.unit.path))),
        ("include/active.h", _sha(writes["include/active.h"])),
    )
    home = lane.homes[0]
    assert lane.made["include/active.h"][0] == writes["include/active.h"]
    assert "include/deleted.h" not in lane.made
    assert lane.preprocess.call_args.args == (
        (home if source_overlaid else lane.root) / lane.unit.path,
        lane.recipe,
        "a",
        [home / "include", lane.root / "include", home / "src", lane.root / "src"],
    )


def test_an_overlaid_file_finds_the_files_it_names_relative_to_itself(lane):
    (lane.root / "include/sub").mkdir()
    (lane.root / "include/base.h").write_bytes(b'#include "sub/deep.h"\ntypedef int base;\n')
    (lane.root / "include/sub/deep.h").write_bytes(b"typedef int deep;\n")
    (lane.root / lane.unit.path).write_bytes(b'#include "sub/new.h"\n')  # the unit reaches the overlaid file
    snapshot = _overlay(lane.snapshot, {"include/sub/new.h": b'#include "../base.h"\n'})
    _get(lane, snapshot)
    assert lane.made["include/base.h"][0] == (lane.root / "include/base.h").read_bytes()
    assert lane.made["include/sub/deep.h"][0] == b"typedef int deep;\n"  # and what that file names
    assert "include/z.h" not in lane.made  # nothing else is copied


def test_view_does_not_copy_unrelated_publication_payloads(lane):
    snapshot = _overlay(
        lane.snapshot,
        {
            lane.unit.path: b'#include "active.h"\nint changed;\n',
            "include/active.h": b"typedef long active;\n",
            "Makefile": b"generated build rules\n" * 1000,
            "versions/a/report.json": b'{"large": "report"}',
            "src/other.c": b"int unrelated;\n",
        },
    )
    _get(lane, snapshot)
    assert lane.made[lane.unit.path][0] == snapshot.read(lane.unit.path)
    assert lane.made["include/active.h"][0] == snapshot.read("include/active.h")
    for path in ("Makefile", "versions/a/report.json", "src/other.c"):
        assert path not in lane.made


def test_relative_include_uses_the_overlaid_header_without_changing_the_real_header(lane):
    (lane.root / lane.unit.path).write_bytes(b'#include "../include/active.h"\n')
    real_header = (lane.root / "include/active.h").read_bytes()
    snapshot = _overlay(
        lane.snapshot,
        {
            lane.unit.path: b'#include "../include/active.h"\nint changed;\n',
            "include/active.h": b"typedef long active;\n",
        },
    )
    _get(lane, snapshot)
    assert lane.made["include/active.h"] == (b"typedef long active;\n", False)
    assert (lane.root / "include/active.h").read_bytes() == real_header


def test_new_include_refreshes_closure_without_changing_live_source(lane):
    source = b'#if 0\n#include "inactive.h"\n#endif\nint value;\n'
    (lane.root / lane.unit.path).write_bytes(source)
    lane.text = '# 1 "src/unit.c"\nint value;\n'
    first = _get(lane)
    assert first.dependencies == ((lane.unit.path, _sha(source)),)
    (lane.root / "include/inactive.h").write_bytes(b"changed but inactive")
    second = _get(lane)
    assert second.text == first.text
    assert view.live_digest(lane.snapshot.config, second) == view.live_digest(lane.snapshot.config, first)
    assert lane.preprocess.call_count == 2


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/unit.c", {10: "int value;", 11: "", 20: "int other;"}),
        ("include/active.h", {4: "typedef int active;"}),
        ("unknown.h", {}),
    ],
)
def test_active_lines_by_file(lane, path, expected):
    lane.text = (
        '# 4 "include/active.h"\ntypedef int active;\n# 10 "src/unit.c"\nint value;\n\n# 20 "src/unit.c"\nint other;\n'
    )
    result = _get(lane)
    assert view.active_lines(result, path) == expected
    assert lane.stage.call_args.args == ("view.active_lines",)


def test_dependency_order_and_cache_keys(lane):
    lane.text = (
        '# 1 "include/z.h"\nint z;\n# 2 "include/active.h"\nint active;\n'
        '# 1 "src/unit.c"\nint value;\n# 5 "include/z.h"\nint z_again;\n'
    )
    result = _get(lane)
    deps = [(path, _sha(lane.snapshot.read(path))) for path in (lane.unit.path, "include/active.h", "include/z.h")]
    reads = view.closure(lane.snapshot, lane.unit, "a")  # what the build can reach, pinned: the one key
    key = digest(
        (
            lane.unit.path,
            _sha(lane.snapshot.read(lane.unit.path)),
            lane.recipe.digest,
            "a",
            lane.snapshot.config.project.version_macros["a"],
            view._identity(lane.snapshot.config, lane.recipe),
            reads,
            False,
            False,
        )
    )
    assert result.dependencies == tuple(deps)
    assert result.key == key
    assert view._kept(lane.snapshot.config, key) == (lane.text, [list(row) for row in deps])


@pytest.mark.parametrize(
    "stderr", [b"fatal error: absent.h: No such file or directory\n", b'Cannot open include file "absent.h"\n']
)
def test_missing_include_refuses(lane, stderr):
    finding = Finding("preprocess.error", stderr.decode())
    lane.result = fixture.native_result(exit=1, stderr=stderr)
    lane.findings = (finding,)
    with pytest.raises(Refusal) as raised:
        _get(lane)
    assert raised.value.findings[0] == finding
    assert raised.value.findings[1].key == "view.include"
    assert raised.value.findings[1].missing == ("absent.h",)
    assert not lane.cache


def test_diagnose_findings_refuse_even_on_zero_exit(lane):
    lane.findings = (Finding("preprocess.error", "Rejected preprocessing."),)
    with pytest.raises(Refusal) as raised:
        _get(lane)
    assert raised.value.findings == lane.findings
    lane.diagnose.assert_called_once_with(lane.result)


@pytest.mark.parametrize("deleted", [False, True])
def test_missing_source_refuses(lane, deleted):
    snapshot = lane.snapshot
    if deleted:
        snapshot = _overlay(snapshot, {lane.unit.path: None})
    else:
        (lane.root / lane.unit.path).unlink()
    with pytest.raises(Refusal) as raised:
        _get(lane, snapshot)
    assert raised.value.findings[0].key == "preprocess.error"
    assert raised.value.findings[0].path == lane.unit.path
    lane.preprocess.assert_not_called()


def test_header_change_invalidates_when_the_preprocessor_prints_no_line_markers(lane):
    lane.text = "int value;\n"  # a -P preprocessor: no markers say which headers were read
    first = _get(lane)
    assert [path for path, _ in first.dependencies] == ["src/unit.c", "include/active.h"]
    (lane.root / "include/active.h").write_bytes(b"typedef long active;\n")
    view.effort._memo.clear()  # the next command reads the file again
    assert _get(lane).key != first.key
    assert lane.preprocess.call_count == 2


def test_the_view_is_preprocessed_with_line_markers_even_when_the_recipe_asks_for_p(lane):
    recipe = Recipe("fake", ("-P", "-undef"), (), (), digest("recipe-p"))
    _get(lane, recipe=recipe)
    assert lane.preprocess.call_args.args[1].cppflags == ("-undef",)
    assert lane.preprocess.call_args.args[1].digest == recipe.digest


def test_a_relative_include_beside_an_overlaid_header_is_linked_from_the_real_tree(lane):
    (lane.root / "include/sub").mkdir()
    (lane.root / "include/types.h").write_bytes(b'#include "z.h"\ntypedef int t;\n')
    (lane.root / lane.unit.path).write_bytes(b'#include "sub/a.h"\n')  # the unit reaches the overlaid file
    writes = {"include/sub/a.h": b'#include "../types.h"\n'}
    snapshot = _overlay(lane.snapshot, writes)
    _get(lane, snapshot)
    assert lane.made["include/types.h"][0] == (lane.root / "include/types.h").read_bytes()
    assert lane.made["include/z.h"][1]  # a link into the real tree


def test_two_creators_of_the_same_overlay_link_both_succeed(tmp_path, monkeypatch):
    root, overlay = tmp_path / "root", tmp_path / "overlay"
    (root / "include").mkdir(parents=True)
    (root / "include/a.h").write_text("int a;\n")
    (overlay / "include").mkdir(parents=True)
    (overlay / "include/a.h").symlink_to(root / "include/a.h")  # the other job's link, made after our check
    with monkeypatch.context() as stale:  # our check ran before it existed
        stale.setattr(Path, "exists", lambda self: False)
        stale.setattr(Path, "is_symlink", lambda self: False)
        view._link_relative(overlay, root, "include/b.h", b'#include "a.h"\n')
    link = overlay / "include/a.h"
    assert link.is_symlink() and link.resolve() == (root / "include/a.h").resolve()
    assert [p.name for p in (overlay / "include").iterdir()] == ["a.h"]


def test_diagnostic_maps_view_and_out_of_file_lines_through_markers(lane):
    header = lane.root / "include/type.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("int first;\nint previous;\n")
    lane.text = (
        f'# 1 "{header}"\nint first;\n# 1 "{lane.root / lane.unit.path}"\nint value;\n# 2 "{header}"\nint previous;\n'
    )
    (lane.root / "unit.i").write_text("\n" * 10)
    text = "span_1000/type.h:6: previous declaration; unit.i:4: conflict; include/type.h:1: real"
    assert view.diagnostic(lane.snapshot, lane.unit, "a", lane.recipe, text) == (
        f"include/type.h:2: previous declaration; {lane.unit.path}:1: conflict; include/type.h:1: real"
    )


def test_snapshots_giving_the_unit_the_same_files_share_one_view(lane):
    (lane.root / "include/sub").mkdir()
    (lane.root / lane.unit.path).write_bytes(b'#include "sub/a.h"\n')
    one = _overlay(lane.snapshot, {"include/sub/a.h": b"typedef int a;\n"})
    two = _overlay(lane.snapshot, {"include/sub/a.h": b"typedef int a;\n", "include/unrelated.h": b"x\n"})
    assert one.digest != two.digest
    first, second = _get(lane, one), _get(lane, two)
    assert second == first and lane.preprocess.call_count == 1  # one preprocessing, however many snapshots ask
    _get(lane, _overlay(lane.snapshot, {"include/sub/a.h": b"typedef long a;\n"}))
    assert lane.preprocess.call_count == 2  # other bytes of a reached file are another view


def test_an_overlay_the_unit_does_not_reach_shares_the_disk_view(lane):
    first = _get(lane)
    second = _get(lane, _overlay(lane.snapshot, {"include/unrelated.h": b"typedef int u;\n"}))
    assert second == first and lane.preprocess.call_count == 1
    assert lane.preprocess.call_args.args[3] == [lane.root / "include", lane.root / "src"]  # no overlay directory


def test_the_closure_pins_overlaid_files_by_content_like_files_on_disk(lane):
    (lane.root / "include/sub").mkdir()
    (lane.root / lane.unit.path).write_bytes(b'#include "sub/a.h"\n')
    (lane.root / "include/sub/a.h").write_bytes(b"typedef int a;\n")
    on_disk = view.closure(lane.snapshot, lane.unit, "a")
    same = _overlay(lane.snapshot, {"include/sub/a.h": b"typedef int a;\n"})
    changed = _overlay(lane.snapshot, {"include/sub/a.h": b"typedef long a;\n"})
    assert view.closure(same, lane.unit, "a") == on_disk
    assert view.closure(changed, lane.unit, "a") != on_disk


@pytest.mark.parametrize("overlaid", [False, True])
@pytest.mark.parametrize("refused", [False, True])
def test_no_view_directory_outlives_its_preprocess(lane, overlaid, refused):
    """The text and dependencies live in the content cache: the directory the view is made in is removed at once, on
    success and on refusal, and nothing but the cache is read to answer a warm view."""
    snapshot = _overlay(lane.snapshot, {"include/active.h": b"typedef long active;\n"}) if overlaid else lane.snapshot
    lane.findings = (Finding("preprocess.error", "no", path=lane.unit.path),) if refused else ()
    if refused:
        with pytest.raises(Refusal):
            _get(lane, snapshot)
    else:
        _get(lane, snapshot)
        _get(lane, snapshot)  # warm: no second directory is made
    assert len(lane.homes) == 1 and not lane.homes[0].exists()
    assert not any((lane.root / "build/views").iterdir())


def test_overlay_pin_hashes_once_per_stat_and_never_reuses_changed_source(lane, monkeypatch):
    import os

    path = lane.snapshot.config.project.root / lane.unit.path
    old = path.stat()
    read = Path.read_bytes
    reads = []

    def counted(p):
        if p == path:
            reads.append(p)
        return read(p)

    monkeypatch.setattr(Path, "read_bytes", counted)
    first = view._pin(lane.snapshot, lane.unit.path)
    assert view._pin(lane.snapshot, lane.unit.path) == first
    assert len(reads) == 1
    path.write_bytes(b"int changed(void) { return 2; }\n")
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1))
    assert view._pin(lane.snapshot, lane.unit.path) != first
    assert len(reads) == 2


def test_git_tree_view_materializes_only_captured_headers(lane):
    snapshot, unit, recipe, state = lane.snapshot, lane.unit, lane.recipe, lane
    frozen = replace(
        snapshot,
        tree=True,
        overlays={unit.path: b'#include "active.h"\nint value;\n', "include/active.h": b"typedef float active;\n"},
    )
    view.get(frozen, unit, "a", recipe)
    assert state.made["include/active.h"] == (b"typedef float active;\n", False)
    assert "include/z.h" not in state.made
    assert frozen.peek("include/z.h") is None
    assert view._pin(frozen, "include/z.h") == "missing"
    include = state.preprocess.call_args.args[3]
    assert all(not p.is_relative_to(snapshot.config.project.root / "include") for p in include)


def test_line_markers_name_files_from_the_project_root_so_every_checkout_shares_the_text(lane):
    seen = _get(lane)
    assert str(lane.root) not in seen.text
    assert seen.text.startswith('# 1 "src/unit.c"\n')


def _count(monkeypatch):
    seen = {"walks": 0, "parsed": []}
    walk, parse = view._walk, view.directives

    def walking(*args):
        seen["walks"] += 1
        return walk(*args)

    def parsing(content):
        seen["parsed"].append(content)
        return parse(content)

    monkeypatch.setattr(view, "_walk", walking)
    monkeypatch.setattr(view, "directives", parsing)
    return seen


def test_closure_is_walked_once_then_only_revalidated(lane, monkeypatch):
    seen = _count(monkeypatch)
    first = view.closure(lane.snapshot, lane.unit, "a")
    parsed = len(seen["parsed"])
    assert (seen["walks"], parsed) == (1, 2)  # the unit and its one header, each parsed once
    assert view.closure(lane.snapshot, lane.unit, "a") == first
    assert (seen["walks"], len(seen["parsed"])) == (1, parsed)


def test_edited_reached_header_recomputes_the_closure(lane, monkeypatch):
    seen = _count(monkeypatch)
    first = view.closure(lane.snapshot, lane.unit, "a")
    (lane.root / "include/active.h").write_bytes(b'#include "z.h"\ntypedef int active;\n')
    second = view.closure(lane.snapshot, lane.unit, "a")
    assert seen["walks"] == 2
    assert dict(second)["include/active.h"] != dict(first)["include/active.h"]
    assert "include/z.h" in dict(second)


def test_file_created_at_a_probed_missing_path_recomputes_the_closure(lane, monkeypatch):
    (lane.root / "src/unit.c").write_bytes(b'#include "late.h"\nint value;\n')
    seen = _count(monkeypatch)
    assert [n for n, _ in view.closure(lane.snapshot, lane.unit, "a")] == [lane.unit.path]
    (lane.root / "include/late.h").write_bytes(b"typedef int late;\n")
    assert "include/late.h" in dict(view.closure(lane.snapshot, lane.unit, "a"))
    assert seen["walks"] == 2


def test_overlay_files_are_hashed_once_across_closure_calls(lane, monkeypatch):
    snapshot = _overlay(lane.snapshot, {"include/active.h": b"typedef long active;\n"})
    hashed = []
    real = view._sha
    monkeypatch.setattr(view, "_sha", lambda value: hashed.append(value) or real(value))
    for _ in range(4):
        view.closure(snapshot, lane.unit, "a")
    assert hashed.count(b"typedef long active;\n") == 1


def test_build_headers_share_the_closure_walk(lane, monkeypatch):
    seen = _count(monkeypatch)
    argv = view._search(lane.snapshot, "a")
    view.headers(lane.snapshot, lane.unit, argv)
    view.closure(lane.snapshot, lane.unit, "a")
    assert seen["walks"] == 1


def test_the_view_text_construction_is_pinned_so_a_change_forces_the_format_bump():
    """The stored-view key names tool bytes, not code: edit how a view's text is cut or kept and bump view._FORMAT."""
    pinned = {
        "chunks": view._chunks("\n".join(f"line {i}" for i in range(2000))),
        "lines": view._lines('# 3 "include/a.h"\nx\n# 9 "/r/src/u.c"\ny\n', Path("/r"), None, "src/u.c"),
    }
    assert digest(pinned) == "360f345542cd4997da950d3a602327cfe7dbedca5e611c74bf99d8420793c450"
    assert view._FORMAT == 1


def test_a_warm_view_key_names_its_digests_without_loading_or_parsing_the_view(lane, monkeypatch):
    """A later command keys an unchanged unit from the kept digests: no view is loaded, no declaration is parsed."""
    kept = {}

    def cached_object(config, kind, key, produce):
        if (kind, key) not in kept:
            kept[kind, key] = produce()
        return kept[kind, key]

    monkeypatch.setattr(view.store, "cached_object", cached_object)
    loads = Mock(side_effect=view.get)
    monkeypatch.setattr(view, "get", loads)
    live = Mock(side_effect=view.declarations.live)
    monkeypatch.setattr(view.declarations, "live", live)

    def keyed():
        view.effort._memo.clear()  # each key is made by a command of its own
        return view.reads(lane.snapshot, lane.unit, "a", lane.recipe, whole=True)

    first = keyed()
    assert [keyed() for _ in range(3)] == [first] * 3
    assert (loads.call_count, live.call_count, lane.preprocess.call_count) == (1, 1, 1)
    (lane.root / "include/active.h").write_bytes(b"typedef long active;\n")  # a reached file moves the key
    keyed()
    assert (loads.call_count, live.call_count) == (2, 2)


@pytest.mark.parametrize("overlaid", [False, True])
def test_a_view_stored_by_one_install_and_root_hits_in_another_with_the_same_content(lane, monkeypatch, overlaid):
    """Map's cold ownership pass missed 0/14,140 views on every new install while units hit: the view kinds were keyed
    by the tool code. Same files, recipe and tool bytes in another checkout under another install share the stored view."""
    for name, real in zip(("get", "put", "put_many", "content"), _STORE, strict=True):
        monkeypatch.setattr(store, name, real)
    writes = {"include/active.h": b"typedef long active;\n"} if overlaid else {}
    monkeypatch.setattr(store, "code", lambda: "install-one")
    first = _get(lane, _overlay(lane.snapshot, writes))
    other = lane.root.parent / "other"
    import shutil
    shutil.copytree(lane.root, other)
    moved = replace(lane.snapshot.config, project=replace(lane.snapshot.config.project, root=other))
    lane.text = lane.text.replace(str(lane.root), str(other))
    monkeypatch.setattr(store, "code", lambda: "install-two")
    second = _get(lane, _overlay(replace(lane.snapshot, config=moved), writes))
    assert lane.preprocess.call_count == 1  # no second preprocess: the second run hit the first run's view
    assert (second.key, second.text, second.dependencies) == (first.key, first.text, first.dependencies)


def test_a_new_install_with_the_same_view_code_keeps_the_view_identity_and_a_view_code_edit_does_not(monkeypatch):
    """Only view.py's bytes, the tool bytes and the preprocess command name a view: another install hits."""
    monkeypatch.setattr(view.adapters, "tool_identity", lambda config, toolchain: "tool-bytes")
    monkeypatch.setattr(view.adapters, "row", lambda toolchain: {"preprocess": ["cpp"]})
    recipe = Recipe("fake", (), (), (), digest("recipe"))
    identities = {}
    runs = (("one", "install-one", "same"), ("two", "install-two", "same"), ("edit", "install-two", "edited"))
    for name, code, source in runs:
        view.effort._memo.clear()
        monkeypatch.setattr(store, "code", lambda code=code: code)
        monkeypatch.setattr(view, "_code", lambda source=source: source)
        identities[name] = view._identity(None, recipe)
    assert identities["one"] == identities["two"] != identities["edit"]
