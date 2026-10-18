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
