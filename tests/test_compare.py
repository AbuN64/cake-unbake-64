"""compare: a candidate is spliced into the file that owns its stand-in, measured per holding version and entered."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import fixture
import pytest

from unbake import compare, effort, store
from unbake import config as configuration
from unbake.contracts import Config, Finding, Proof, Refusal, sha256

SOURCES = {
    "src/main/c_func.c": "s32 c_func(void) {\n    return 1;\n}\n",
    "src/main/stand_in.c": '#include "include_asm.h"\n\nINCLUDE_ASM("main", stand_in);\n',
    "src/ovl/fuzzy.c": (
        '#include "include_asm.h"\n\n#ifdef NON_MATCHING\ns32 fuzzy(s32 a) {\n    return a + 1;\n}\n#else\n'
        'INCLUDE_ASM("ovl", fuzzy);\n#endif\n'
    ),
}
CANDIDATE = "s32 stand_in(void) {\n    return 2;\n}\n"


@pytest.fixture(autouse=True)
def invocation():
    with effort.command("test-compare", ()):
        yield


class World:
    """The fixture project with the measuring and the inbox replaced: what compare asks of them is counted."""

    def __init__(self, cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        self.cfg, self.tmp = cfg, tmp_path
        self.measured: list[tuple[str, str, str, str]] = []
        self.exact = {"a": True, "b": True}
        self.submitted: list[dict] = []
        self.drains = 0
        self.refuse: Refusal | None = None
        (tmp_path / "stand_in.c").write_text(CANDIDATE)
        monkeypatch.setattr(compare.native, "measure", self.measure)
        monkeypatch.setattr(compare.pool, "map", lambda config, name, fn, items, key=None: [fn(i) for i in items])
        monkeypatch.setattr(compare.land, "submit", self.submit)
        monkeypatch.setattr(compare.land, "submit_many", self.submit_many)
        monkeypatch.setattr(compare.land, "drain", self.drain)

    def measure(self, config, version, name, path, text, overrides=None, overlays=None) -> Proof:
        self.measured.append((version, name, path, text))
        ok = self.exact[version]
        return fixture.proof(
            path, name, version, ok, () if ok else (f"version {version}: 4 bytes differ at +0x0 .text",)
        )

    def submit(self, config, request, proofs, source, origin, extras=None):
        if self.refuse:
            raise self.refuse
        self.submitted.append(
            {"request": request, "proofs": proofs, "source": source, "origin": origin, "extras": extras}
        )
        exact = all(p.exact for p in proofs)
        return None if not exact else fixture.submission(member=request["function"])

    def submit_many(self, config, calls):
        return [
            self.submit(config, c["request"], c["proofs"], c["source"], c["origin"], c.get("extras")) for c in calls
        ]

    def drain(self, config):
        self.drains += 1
        return {"running": False, "landed": [], "refused": []}


@pytest.fixture
def world(tmp_path, monkeypatch, toolchains):
    cfg = fixture.config(tmp_path, sources=SOURCES)
    return World(cfg, monkeypatch, tmp_path)


def _params(world, **given):
    return {"file": str(world.tmp / "stand_in.c"), "function": None, "out": None, "note": None, **given}


# ---------------------------------------------------------------- gaps
def test_gaps_name_every_version_that_does_not_match():
    proofs = [
        fixture.proof("src/x.c", "f", "a", True),
        fixture.proof("src/x.c", "f", "b", False, ("version b: 4 bytes differ at +0x0 .text",)),
    ]
    (gap,) = compare.gaps("f", ("a", "b", "c"), proofs)
    assert (gap.key, gap.unit, gap.versions) == ("land.not_exact", "f", ("b", "c"))
    assert gap.missing == ("version b: 4 bytes differ at +0x0 .text", "version c: no measurement")
    assert gap.symptoms["bytes_differ"] is True
    assert compare.gaps("f", ("a",), proofs[:1]) == ()


# ---------------------------------------------------------------- the candidate
def test_the_member_is_the_function_the_candidate_defines_that_a_file_owns(world):
    result = compare.run(world.cfg, _params(world))
    assert result["members"] == ["stand_in"] and result["unit"] == "src/main/stand_in.c"
    assert {v for v, *_ in world.measured} == {"a", "b"}  # both versions hold it
    _, name, path, text = world.measured[0]
    assert (name, path) == ("stand_in", "src/main/stand_in.c")
    assert "INCLUDE_ASM" not in text and "return 2;" in text and text.startswith('#include "include_asm.h"')


def test_function_names_the_member_when_the_source_does_not(world):
    (world.tmp / "stand_in.c").write_text("s32 helper(void) {\n    return 2;\n}\n")
    with pytest.raises(Refusal) as refused:
        compare.run(world.cfg, _params(world))
    assert refused.value.findings[0].key == "land.request" and "defines no function" in refused.value.findings[0].reason
    with pytest.raises(Refusal):  # helper is no stand-in of any file
        compare.run(world.cfg, _params(world, function="helper"))


def test_a_missing_source_is_refused_with_the_path(world):
    with pytest.raises(Refusal) as refused:
        compare.run(world.cfg, _params(world, file=str(world.tmp / "absent.c")))
    assert refused.value.findings[0].path.endswith("absent.c")


def test_a_landed_function_is_replaced_in_place(world):
    path = world.cfg.project.root / "src/main/c_func.c"
    (world.tmp / "stand_in.c").write_text("s32 c_func(void) {\n    return 7;\n}\n")
    compare.run(world.cfg, _params(world, function="c_func"))
    text = world.measured[0][3]
    assert "return 7;" in text and "return 1;" not in text and text.count("c_func") == 1
    assert path.read_text().count("return 1;") == 1  # the file on disk is for land to write


def test_a_fuzzy_block_is_replaced_by_the_new_body_over_the_same_stand_in(world):
    (world.tmp / "stand_in.c").write_text("s32 fuzzy(s32 a) {\n    return a + 3;\n}\n")
    compare.run(world.cfg, _params(world, function="fuzzy"))
    text = world.measured[0][3]
    assert "a + 3" in text and "a + 1" not in text and "NON_MATCHING" not in text


# ---------------------------------------------------------------- compare
def test_an_exact_candidate_is_submitted_and_the_inbox_drained(world):
    result = compare.run(world.cfg, _params(world, note="why"))
    assert result["exact"] and result["gaps"] == [] and result["submitted"]
    assert result["score"] == {"a": 1.0, "b": 1.0} and result["drain"]["running"] is False
    (call,) = world.submitted
    assert call["request"] == {
        "file": str(world.tmp / "stand_in.c"),
        "function": "stand_in",
        "overrides": {},
        "note": "why",
    }
    assert call["source"] == CANDIDATE.encode() and call["origin"] == "compare" and call["extras"] == {}
    assert world.drains == 1
    configuration.validate("result.compare", json.loads(json.dumps(result, default=str)), "compare")


def test_an_inexact_candidate_reports_the_gap_and_still_offers_the_fuzzy_gain(world):
    world.exact["b"] = False
    result = compare.run(world.cfg, _params(world))
    assert not result["exact"] and result["submitted"] is None and world.drains == 0
    (gap,) = result["gaps"]
    assert gap["key"] == "land.not_exact" and gap["versions"] == ("b",)
    assert len(world.submitted) == 1  # land decides whether a fuzzy gain is worth entering


def test_a_refusal_of_the_entry_is_a_finding_of_the_result(world):
    world.refuse = Refusal(Finding("source.unclean", "bad", path="src/main/stand_in.c"))
    result = compare.run(world.cfg, _params(world))
    assert not result["exact"] and [g["key"] for g in result["gaps"]] == ["source.unclean"]


def test_change_set_files_travel_as_extras(world):
    header = world.tmp / "new.h"
    header.write_text("typedef int T;\n")
    files = [world.tmp / "stand_in.c", Path(f"include/new.h={header}")]
    compare._attempt(world.cfg, files, _params(world), "submit")
    assert world.submitted[0]["extras"] == {"include/new.h": b"typedef int T;\n"}
    assert world.measured and world.submitted[0]["origin"] == "submit"


@pytest.mark.parametrize("spec", ["nope", "src/main/c_func.c=missing.txt", "../outside.h=x"])
def test_malformed_change_set_entries_are_refused(world, spec):
    with pytest.raises(Refusal) as refused:
        compare._extras(world.cfg, [Path(spec)])
    assert refused.value.findings[0].key == "land.request"


def test_a_new_source_cannot_be_a_change_set_entry(world):
    new = world.tmp / "x.c"
    new.write_text("int x;\n")
    with pytest.raises(Refusal) as refused:
        compare._extras(world.cfg, [Path(f"src/main/new.c={new}")])
    assert "would be a new source" in refused.value.findings[0].reason


# ---------------------------------------------------------------- submit
def test_submit_returns_one_row_per_candidate(world):
    params = {
        "files": [str(world.tmp / "stand_in.c")],
        "batch_data": None,
        "bulk": False,
        "function": None,
        "note": None,
    }
    result = compare.submit_command(world.cfg, params)
    (row,) = result["submissions"]
    assert (row["member"], row["operation"], row["exact"], row["score"]) == ("stand_in", "publish", True, 1.0)
    assert row["id"] and row["findings"] == [] and result["drain"]["landed"] == []
    configuration.validate("result.submit", json.loads(json.dumps(result, default=str)), "submit")


def test_submit_without_sources_is_refused(world):
    with pytest.raises(Refusal) as refused:
        compare.submit_command(world.cfg, {"files": [], "batch_data": None, "bulk": False})
    assert "Pass source files" in refused.value.findings[0].reason


def test_batch_data_is_the_data_modules_job(world, monkeypatch):
    from unbake import batch_data

    run = Mock(return_value={"submissions": [], "drain": None})
    monkeypatch.setattr(batch_data, "run", run)
    params = {"files": [], "batch_data": "m.json", "bulk": False}
    assert compare.submit_command(world.cfg, params) == {"submissions": [], "drain": None}
    run.assert_called_once()


def test_bulk_measures_all_candidates_together_and_a_refusal_is_its_row(world):
    good, bad = world.tmp / "stand_in.c", world.tmp / "orphan.c"
    bad.write_text("s32 orphan(void) {\n    return 0;\n}\n")
    params = {"files": [str(good), str(bad)], "batch_data": None, "bulk": True, "note": None}
    result = compare.submit_command(world.cfg, params)
    first, second = result["submissions"]
    assert first["id"] and first["exact"] and first["member"] == "stand_in"
    assert second["id"] is None and second["refused"] and second["findings"][0]["key"] == "land.request"
    assert result["drain"] is not None and world.drains == 1
    assert {n for _, n, *_ in world.measured} == {"stand_in"}  # the orphan was never measured


def test_bulk_attaches_following_change_set_files_to_their_source(world):
    header = world.tmp / "h.h"
    header.write_text("typedef int T;\n")
    params = {"files": [str(world.tmp / "stand_in.c"), f"include/h.h={header}"], "bulk": True, "batch_data": None}
    compare.submit_command(world.cfg, params)
    assert world.submitted[0]["extras"] == {"include/h.h": b"typedef int T;\n"}


# ---------------------------------------------------------------- score
def test_score_function_is_the_score_of_the_working_tree_text(world):
    path = "src/ovl/fuzzy.c"
    assert compare.score_function(world.cfg, path, "fuzzy", "a") == 1.0
    version, name, measured_path, text = world.measured[0]
    assert (version, name, measured_path) == ("a", "fuzzy", path)
    assert text == (world.cfg.project.root / path).read_text()


def _built(world, data):
    store.put(world.cfg, "bytes", sha256(data), data)


def test_score_shows_the_diff_against_the_rom(world, monkeypatch):
    built = bytes(0x10)
    _built(world, built)
    proof = fixture.proof("src/main/stand_in.c", "stand_in", "a", False, ("version a: 1 bytes differ at +0x0 .text",))
    proof = replace(proof, built_sha256=sha256(built))
    monkeypatch.setattr(compare.native, "measure", lambda config, v, *a, **k: replace(proof, version=v))
    monkeypatch.setattr(compare.versions, "rom_bytes", lambda record, start, end: b"\x01" + bytes(end - start - 1))
    monkeypatch.setattr(compare.native, "candidate", lambda *a, **k: (Mock(), Mock(), Mock(), []))
    monkeypatch.setattr(compare.policy, "evaluate", lambda *a: ())
    params = {"file": str(world.tmp / "stand_in.c"), "function": None, "version": "a", "lines": None, "short": False}
    result = compare._score_one(world.cfg, params)
    assert result["version"] == "a" and result["versions"] == ["a", "b"] and result["exact"] is False
    assert result["words"] == {"built": 4, "target": 4, "matched": 3}
    assert result["diff"] and result["source"] == [] and result["rewritten"] == []
    assert [p["version"] for p in result["proofs"]] == ["a", "b"]


def test_score_refuses_a_version_that_does_not_hold_the_function(world):
    params = {"file": str(world.tmp / "stand_in.c"), "function": "fuzzy", "version": "zz", "lines": None}
    (world.tmp / "stand_in.c").write_text("s32 fuzzy(s32 a) {\n    return a;\n}\n")
    with pytest.raises(Refusal) as refused:
        compare.score_command(world.cfg, params)
    assert "not held by version zz" in refused.value.findings[0].reason


def test_score_needs_a_source_or_variants(world):
    with pytest.raises(Refusal) as refused:
        compare.score_command(world.cfg, {"file": None, "variants": None})
    assert refused.value.findings[0].action == "pass FILE or --variants"


def test_score_overrides_are_flags_to_add_and_omit_and_a_toolchain(world):
    params = {"flag": ["-O1"], "omit_flag": ["-O2"], "toolchain": "ido-7.1"}
    assert compare._overrides(params) == {"add": ["-O1"], "omit": ["-O2"], "toolchain": "ido-7.1"}
    assert compare._overrides({}) == {"add": [], "omit": []}


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        ('{"file": "a.c"}', "must hold a JSON list"),
        ("[1]", "rows must be objects with a nonempty file path"),
        ('[{"file": "a.c", "bogus": 1}]', "take file, overlays, flags, omit_flags, function: not bogus"),
        ('[{"file": "a.c", "flags": "-O1"}]', "flags must be a list"),
        ('[{"file": "a.c", "overlays": [1]}]', "overlays must be a list of nonempty path strings"),
        ("not json", "is not JSON"),
    ],
)
def test_a_variants_manifest_is_validated_before_anything_is_scored(world, rows, reason):
    manifest = world.tmp / "variants.json"
    manifest.write_text(rows)
    with pytest.raises(Refusal) as refused:
        compare.score_command(world.cfg, {"file": None, "variants": str(manifest)})
    assert reason in refused.value.findings[0].reason and refused.value.findings[0].path == str(manifest)


def test_variants_are_scored_side_by_side_and_each_keeps_its_own_verdict(world, monkeypatch):
    manifest = world.tmp / "variants.json"
    manifest.write_text(json.dumps([{"file": "ok.c", "flags": ["-O1"]}, {"file": "bad.c", "omit_flags": ["-O2"]}]))

    def one(config, params):
        if params["file"].name == "bad.c":
            raise Refusal(Finding("score.compile", "parse error"))
        return {"score": 0.5, "proofs": [], "flags": params["flag"]}

    monkeypatch.setattr(compare, "_score_one", one)
    result = compare.score_command(world.cfg, {"file": None, "variants": str(manifest)})
    good, bad = result["variants"]
    assert good["ok"] and good["flags"] == ["-O1"] and good["result"] == {"score": 0.5, "flags": ["-O1"]}
    assert not bad["ok"] and bad["omit_flags"] == ["-O2"] and bad["findings"][0]["key"] == "score.compile"


def test_compact_keeps_hunk_headers_and_changed_lines():
    diff = ["--- a", "+++ b", "@@ -1 +1 @@", " same", "-old", "+new"]
    assert compare._compact(diff, 2) == ["@@ -1 +1 @@", "-old"]
    assert SimpleNamespace  # the module under test imports nothing else of the test helpers
