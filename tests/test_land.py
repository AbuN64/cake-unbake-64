"""The submission intake and the drain around the landing; make, tree and measure are stubbed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_publish import CANDIDATE, OWNER, World  # noqa: F401

from unbake import land, publish
from unbake.contracts import Refusal


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch, versions=("us", "eu", "jp"), held=("us", "eu"))


def test_land_builds_affected_only(world):
    publish.land(world.config, world.submission())
    assert world.makes == [["build/eu/game.z64", "build/us/game.z64"]]  # jp does not hold the name: one make, no jp


def test_a_refused_landing_builds_each_affected_version_once(world):
    world.broken = {"us"}
    with pytest.raises(Refusal):
        publish.land(world.config, world.submission())
    assert sum(len(m) for m in world.makes) == 2 and world.owner == OWNER


def _proof(exact, score=1.0, symptoms=None):
    return SimpleNamespace(exact=exact, score=score, symptoms=symptoms or {})


def _intake(world, monkeypatch, proofs):
    monkeypatch.setattr(land, "process", SimpleNamespace(git=lambda *a: SimpleNamespace(stdout=b"abc\n")))
    request = {"function": "func_80000010", "file": str(world.root / "cand.c"), "overrides": {}, "note": ""}
    return land._intake(world.config, request, proofs, None)


def test_intake_exact_is_a_publish(world, monkeypatch):
    assert _intake(world, monkeypatch, [_proof(True), _proof(True)]).operation == "publish"


def test_intake_fuzzy_needs_the_gain(world, monkeypatch):
    assert _intake(world, monkeypatch, [_proof(False, 0.505)]) is None  # 0.5 -> 0.505 is under min_gain
    assert _intake(world, monkeypatch, [_proof(False, 0.7)]).operation == "fuzzy"


def test_intake_unmeasured_is_not_submitted(world, monkeypatch):
    assert _intake(world, monkeypatch, [_proof(False, 0.9, {"compile_failed": True})]) is None


def test_submit_many_keeps_each_verdict(world, monkeypatch):
    monkeypatch.setattr(land, "process", SimpleNamespace(git=lambda *a: SimpleNamespace(stdout=b"abc\n")))
    good = {"function": "func_80000010", "file": str(world.root / "cand.c"), "overrides": {}, "note": ""}
    bad = {**good, "function": "other_missing"}
    verdicts = publish.accept_each(world.config, [(good, {}, False), (bad, {}, False)])
    assert verdicts[0] == () and verdicts[1] and verdicts[1][0].key == "land.request"


def test_the_request_compare_builds_passes_the_submission_schema(world, monkeypatch):
    from unbake import compare

    monkeypatch.setattr(land, "process", SimpleNamespace(git=lambda *a: SimpleNamespace(stdout=b"abc\n")))
    monkeypatch.setattr(land.effort, "invocation", lambda: "unbake submit")
    request = compare._request("func_80000010", world.root / "cand.c", None)
    intake = land._intake(world.config, request, [_proof(True), _proof(True)], None)
    entry = land._entered(world.config, request, b"s32 f(void) {}\n", (), "compare", intake)  # validates "submission"
    assert entry.overrides == {"add": [], "omit": []}


def test_a_pre_provenance_inbox_entry_is_replaced_by_the_resubmission_not_refused_again(world, monkeypatch):
    import json

    from unbake import compare

    monkeypatch.setattr(land, "process", SimpleNamespace(git=lambda *a: SimpleNamespace(stdout=b"abc\n")))
    monkeypatch.setattr(land.effort, "invocation", lambda: "unbake submit")
    request = compare._request("func_80000010", world.root / "cand.c", None)
    intake = land._intake(world.config, request, [_proof(True), _proof(True)], None)
    entry = land._entered(world.config, request, b"s32 f(void) {}\n", (), "compare", intake)
    path = world.root / ".unbake/inbox" / f"{entry.id}.json"
    old = json.loads(path.read_text())
    old.pop("extra_bases")
    old["member"] = "func_80000010"
    path.write_text(json.dumps(old))
    with pytest.raises(Refusal) as stale:
        land._read(path)
    assert stale.value.findings[0].action == "recipe try func_80000010"
    again = land._entered(world.config, request, b"s32 f(void) {}\n", (), "compare", intake)
    assert again.id == entry.id and "extra_bases" in json.loads(path.read_text())
    assert (world.root / ".unbake/inbox/stale" / f"{entry.id}.json").is_file()


def test_drain_lands_only_the_named_entries_and_abandoned_ones_go_to_the_attic(world, monkeypatch):
    import json
    import os
    import time

    monkeypatch.setattr(land, "process", SimpleNamespace(git=lambda *a: SimpleNamespace(stdout=b"abc\n")))
    monkeypatch.setattr(land.effort, "invocation", lambda: "unbake submit")
    monkeypatch.setenv("RECIPE_SESSION", "LANE-A")
    from unbake import compare

    request = compare._request("func_80000010", world.root / "cand.c", None)
    intake = land._intake(world.config, request, [_proof(True)], None)
    mine = land._entered(world.config, request, b"s32 f(void) {}\n", (), "compare", intake)
    other = land._entered(world.config, request, b"s32 g(void) {}\n", (), "compare", intake)
    assert mine.session == "LANE-A" and mine.entered > 0
    assert [e.id for e in land.inbox(world.config, {mine.id})] == [mine.id]
    inbox = world.root / ".unbake/inbox"
    old = json.loads((inbox / f"{other.id}.json").read_text())
    old["entered"] = time.time() - 4 * 3600
    (inbox / f"{other.id}.json").write_text(json.dumps(old))
    dead = {**json.loads((inbox / f"{mine.id}.json").read_text()), "session": "999999999"}
    assert land._abandoned(dead["session"], time.time()) and not land._abandoned(str(os.getpid()), time.time())
    assert land.expire(world.config) == [other.id]
    assert (inbox / "attic" / f"{other.id}.json").is_file() and (inbox / f"{mine.id}.json").is_file()
