"""Recorded change-set deltas follow moves without replacing a merged file wholesale."""

import shutil
import subprocess
from dataclasses import replace

import fixture
import pytest

from unbake import headers, moved
from unbake.contracts import LayoutMap, Refusal, Snapshot, UnitSpec


@pytest.fixture
def history(tmp_path, toolchains):
    cfg = fixture.config(
        tmp_path,
        headers={"old.h": "typedef int Count;\n", "other.h": "typedef long Other;\n"},
        sources={"old.c": '#include "old.h"\nint f(void) { return 1; }\n'},
        host={"tools.git": shutil.which("git")},
    )
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cfg.project.root).decode().strip()
    snapshot = Snapshot(cfg, base, LayoutMap({}, {}, {}, "map", (), {}), {}, {}, "snapshot")
    return snapshot, base


def merged(history):
    snapshot, base = history
    root = snapshot.config.project.root
    for name in ("old.h", "other.h"):
        (root / "include" / name).unlink()
    text = headers.wrapped("include/NEW.H", "typedef int Count;\n\ntypedef long Other;\n")
    (root / "include/NEW.H").write_text(text)
    moved.record(
        snapshot.config,
        [("path", "include/old.h", "include/NEW.H"), ("path", "include/other.h", "include/NEW.H")],
        "after",
    )
    return snapshot, base, text.encode()


def test_recorded_header_delta_preserves_the_other_merged_header(history):
    snapshot, base, current = merged(history)
    result = moved.extras(snapshot, {"include/old.h": b"typedef signed int Count;\n"}, base)
    assert set(result) == {"include/NEW.H"}
    assert b"typedef signed int Count;" in result["include/NEW.H"]
    assert b"typedef long Other;" in result["include/NEW.H"]
    assert snapshot.peek("include/old.h") is None and snapshot.read("include/NEW.H") == current


def test_two_extras_to_one_target_merge_both_deltas(history):
    snapshot, _ = history
    root = snapshot.config.project.root
    body = "typedef int Count;\nextern int anchor_one;\nextern int anchor_two;\ntypedef long Other;\n"
    for name in ("old.h", "other.h"):
        (root / "include" / name).write_text(body)
    subprocess.run(["git", "add", "include"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "Record shared declarations"], cwd=root, check=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    snapshot, _, _ = merged((snapshot, base))
    current = headers.wrapped("include/NEW.H", body).encode()
    (root / "include/NEW.H").write_bytes(current)
    result = moved.extras(
        snapshot,
        {
            "include/old.h": body.replace("typedef int Count;", "typedef signed int Count;").encode(),
            "include/other.h": body.replace("typedef long Other;", "typedef signed long Other;").encode(),
        },
        base,
    )
    assert b"typedef signed int Count;" in result["include/NEW.H"]
    assert b"typedef signed long Other;" in result["include/NEW.H"]
    assert snapshot.read("include/NEW.H") == current


def test_conflicting_old_and_current_targets_refuse_without_a_write(history):
    snapshot, base, current = merged(history)
    # The second extra names the same final destination as a newly-created file at the recorded base.
    with pytest.raises(Refusal) as caught:
        moved.extras(
            snapshot,
            {"include/old.h": b"typedef signed int Count;\n", "include/NEW.H": b"typedef unsigned int Count;\n"},
            base,
        )
    assert caught.value.findings[0].key == "land.rebase"
    assert snapshot.read("include/NEW.H") == current and snapshot.peek("include/old.h") is None


@pytest.mark.parametrize("base", [None, "missing-commit"])
def test_stale_extra_without_available_base_refuses_before_overlay(history, base):
    snapshot, _, current = merged(history)
    with pytest.raises(Refusal) as caught:
        moved.extras(snapshot, {"include/old.h": b"typedef signed int Count;\n"}, base)
    finding = caught.value.findings[0]
    assert finding.key == "land.rebase" and "submit" in finding.action
    assert finding.path == "include/NEW.H"
    assert snapshot.read("include/NEW.H") == current and snapshot.peek("include/old.h") is None


def test_an_unmoved_recorded_extra_keeps_concurrent_changes(history):
    snapshot, _ = history
    root = snapshot.config.project.root
    original = b"typedef int Count;\nextern int anchor_one;\nextern int anchor_two;\n"
    (root / "include/old.h").write_bytes(original)
    subprocess.run(["git", "add", "include/old.h"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "Record header context"], cwd=root, check=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    current = original + b"extern int concurrent;\n"
    (root / "include/old.h").write_bytes(current)
    result = moved.extras(snapshot, {"include/old.h": original.replace(b"int Count", b"signed int Count")}, base)
    assert b"typedef signed int Count;" in result["include/old.h"]
    assert b"extern int concurrent;" in result["include/old.h"]
    assert snapshot.read("include/old.h") == current


def test_an_existing_destination_edit_conflicts_instead_of_being_overwritten(history):
    snapshot, base = history
    current = b"typedef unsigned int Count;\n"
    (snapshot.config.project.root / "include/old.h").write_bytes(current)
    with pytest.raises(Refusal) as caught:
        moved.extras(snapshot, {"include/old.h": b"typedef long Count;\n"}, base)
    assert caught.value.findings[0].key == "land.rebase"
    assert snapshot.read("include/old.h") == current


def test_moved_source_extra_has_current_path_and_include_spelling(history):
    snapshot, base = history
    root = snapshot.config.project.root
    (root / "src/old.c").unlink()
    (root / "include/old.h").rename(root / "include/NEW.H")
    (root / "src/NEW.CPP").write_bytes(b'#include "NEW.H"\nint f(void) { return 1; }\n')
    unit = UnitSpec("src/NEW.CPP", "c", "group", ("f",), "gcc-test", {})
    snapshot = replace(snapshot, layout=replace(snapshot.layout, units={unit.path: unit}))
    moved.record(
        snapshot.config, [("path", "src/old.c", unit.path), ("path", "include/old.h", "include/NEW.H")], "after"
    )
    result = moved.extras(snapshot, {"src/old.c": b'#include "old.h"\nint f(void) { return 2; }\n'}, base)
    assert result == {unit.path: b'#include "NEW.H"\nint f(void) { return 2; }\n'}
    assert snapshot.peek("src/old.c") is None


def test_a_retired_source_without_a_proven_destination_cannot_be_recreated(history):
    snapshot, base = history
    (snapshot.config.project.root / "src/old.c").unlink()
    with pytest.raises(Refusal) as caught:
        moved.extras(snapshot, {"src/old.c": b"int f(void) { return 2; }\n"}, base)
    assert caught.value.findings[0].key == "land.rebase"
    assert snapshot.peek("src/old.c") is None


def test_fresh_current_path_extras_keep_the_existing_whole_file_contract(history):
    snapshot, _ = history
    changes = {"include/old.h": b"typedef long Count;\n", "include/new.h": b"extern int new_name;\n"}
    assert moved.extras(snapshot, changes) == changes


def test_primary_cannot_silently_discard_an_extra_at_the_same_moved_source(history):
    snapshot, _ = history
    pending = {"src/old.c": b"int f(void) { return 2; }\n"}
    with pytest.raises(Refusal) as caught:
        moved.check_primary(snapshot, "src/old.c", pending)
    assert caught.value.findings[0].key == "land.rebase"
    assert "combine" in caught.value.findings[0].reason
    moved.check_primary(snapshot, "src/old.c", {"src/old.c": snapshot.read("src/old.c")})
