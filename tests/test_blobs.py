"""Blobs: blobs.toml facts, the exact byte partition, the import plan and the report fields, on mocked inputs."""

import json
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from unbake import blobs, report
from unbake.contracts import Refusal, sha1, sha256

PLATFORM = [(0, 0x1000)]
ROM = 0x10000


def _config(root: Path, versions=("v",), rom=ROM, segments=None) -> N:
    rows = segments or [
        ("retained", 0, "bin"),
        ("span_1000", 0x1000, "code"),
        ("rsp", 0x8000, "bin"),
        ("span_8100", 0x8100, "code"),
    ]
    version = {
        v: N(
            sha1=sha1(bytes(rom)),
            size=rom,
            segments=tuple(N(name=n, rom=r, vram=0x80000000 + r, kind=k) for n, r, k in rows),
            meta={},
        )
        for v in versions
    }
    (root / "roms").mkdir(exist_ok=True)
    for v in versions:
        (root / "roms" / f"baserom.{v}.z64").write_bytes(bytes(rom))
    return N(project=N(root=root, name="t", version=version))


# ---------------------------------------------------------------- the sweep
def sweep(code=(), data=(), platform=PLATFORM, decoded=(), identified=(), size=ROM):
    return blobs.sweep(size, list(code), list(data), list(platform), list(decoded), list(identified))


def test_every_byte_is_counted_once_in_the_order_code_data_platform_blobs_unknown():
    out = sweep(
        code=[(0x1000, 0x2000)],
        data=[(0x1800, 0x2800)],  # overlaps code: code wins, and the overlap is reported
        decoded=[(0x2000, 0x3000), (0x2800, 0x3800)],  # overlapping blobs are merged
        identified=[
            (0x3000, 0x4000),
            (0x1000, 0x1400),
        ],  # identified under code is shadowed, over decoded is not doubled
    )
    assert (out["code"], out["data"], out["platform"]) == (0x1000, 0x800, 0x1000)
    assert (out["blobs_decoded"], out["blobs_identified"]) == (0x1000, 0x800)
    parts = ("code", "data", "platform", "blobs", "unknown")
    assert sum(out[k] for k in parts) == ROM == out["rom_size"]
    assert out["overlap"] == 0x800 and out["outside"] == 0 and out["shadowed"] == 0xC00


def test_a_blob_over_a_credited_symbol_is_shadowed_and_stays_data():
    out = sweep(data=[(0x2000, 0x3000)], decoded=[(0x2800, 0x3800)])
    assert (out["data"], out["blobs_decoded"], out["shadowed"]) == (0x1000, 0x800, 0x800)


def test_platform_is_identified_unless_credit_covers_it():
    assert sweep()["platform"] == 0x1000 and sweep(code=[(0xC00, 0x1400)])["platform"] == 0xC00


def test_ranges_beyond_the_rom_are_reported_and_clamped():
    out = sweep(code=[(ROM - 0x10, ROM + 0x30)])
    assert out["code"] == 0x10 and out["outside"] == 0x30
    assert sum(out[k] for k in ("code", "data", "platform", "blobs", "unknown")) == ROM


def test_partition_sums_to_the_rom_on_arbitrary_input():
    out = sweep(
        code=[(0x1000, 0x1234), (0x5000, 0x5010)],
        data=[(0x7000, 0x7100)],
        decoded=[(0x2000, 0x2300)],
        identified=[(0x9000, 0xA000)],
    )
    assert sum(out[k] for k in ("code", "data", "platform", "blobs", "unknown")) == ROM and out["overlap"] == 0


# ---------------------------------------------------------------- the file
DOC = {
    "schema": 1,
    "blob": {
        "level.0": {
            "name": "PROVING GROUNDS",
            "type": "world",
            "format": "arena bundle",
            "codec": "none",
            "state": "decoded",
            "stored_size": 0x1000,
            "rom": {"v": [[0x2000, 0x3000]]},
            "evidence": [{"artifact": "arenas.json", "sha256": "ab" * 32, "pointer": "/0"}],
        },
        "sprite.1": {"type": "graphics", "state": "identified", "codec": "rnc1", "rom": {"v": [[0x4000, 0x4400]]}},
        "sprite.1.0": {
            "type": "graphics",
            "state": "decoded",
            "parent": "sprite.1",
            "decoded": [520, 600],
            "aliases": [{"label": "model_11", "semantic": False}],
        },
        "seen.0": {"type": "audio", "state": "observed", "rom": {"v": [[0x5000, 0x5100]]}},
    },
}


def test_emit_then_parse_round_trips_and_is_deterministic(tmp_path):
    config = _config(tmp_path)
    data = blobs.emit(DOC)
    assert blobs.parse(config, data) == DOC and blobs.emit(blobs.parse(config, data)) == data
    assert b'rom."v" = [[0x2000, 0x3000]]' in data


def test_children_and_observed_blobs_add_no_bytes_but_are_counted(tmp_path):
    decoded, identified = blobs.ranges(DOC, "v")
    assert decoded == [(0x2000, 0x3000)] and identified == [(0x4000, 0x4400)]
    assert blobs.census(DOC) == {
        "world": {"decoded": 1},
        "audio": {"observed": 1},
        "graphics": {"identified": 1, "decoded": 1},
    }
    assert blobs.census(DOC, "other") == {}
    assert blobs.census(DOC, "v")["graphics"] == {
        "decoded": 1,
        "identified": 1,
    }  # a child belongs to its parent's version


@pytest.mark.parametrize(
    ("edit", "key"),
    [
        (lambda d: d["blob"]["sprite.1.0"].update(parent="nope"), "blobs.parent"),
        (lambda d: d["blob"]["sprite.1"].update(parent="sprite.1.0"), "blobs.parent"),
        (lambda d: d["blob"]["level.0"]["rom"].update(v=[[0x2000, ROM + 1]]), "blobs.range"),
        (lambda d: d["blob"]["level.0"]["rom"].update(v=[[0x2000, 0x2000]]), "blobs.range"),
        (lambda d: d["blob"]["level.0"]["rom"].update(zz=[[0, 1]]), "blobs.range"),
        (lambda d: d["blob"]["sprite.1.0"].update(rom={"v": [[0, 1]]}), "blobs.schema"),
        (lambda d: d["blob"]["level.0"].update(state="bogus"), "blobs.schema"),
    ],
)
def test_bad_files_refuse_with_their_key(tmp_path, edit, key):
    document = json.loads(json.dumps(DOC))
    edit(document)
    with pytest.raises(Refusal) as error:
        blobs.parse(_config(tmp_path), blobs.emit(document))
    assert key in {f.key for f in error.value.findings}


def test_a_project_without_blobs_toml_has_no_blobs(tmp_path):
    assert blobs.load(_config(tmp_path)) == {"schema": 1, "blob": {}}
    assert blobs.sha(_config(tmp_path)) == ""


# ---------------------------------------------------------------- the report
def _table(**rows):
    return {n: (0x80000000 + rom, rom, size, kind) for n, (rom, size, kind) in rows.items()}


def test_explicit_sizes_are_clamped_and_a_bin_segment_earns_nothing(tmp_path):
    config = _config(tmp_path)
    table = _table(
        ucode=(0x8000, 0x100, "data"), big=(0x1000, 0x9000, "data"), ok=(0x8100, 0x40, "func"), gap=(0x20, 8, "func")
    )
    facts = report._sized(config, "v", table)
    assert facts["ucode"][2] == 0  # in the bin segment rsp
    assert facts["big"][2] == 0x7000  # cut at the segment end 0x8000
    assert facts["ok"][2] == 0x40 and facts["gap"][2] == 0  # retained is bin


def test_credit_never_exceeds_the_non_bin_bytes(tmp_path):
    config = _config(tmp_path)
    table = _table(
        a=(0x1000, None, "data"), rsp=(0x8000, 0x100, "data"), b=(0x8100, None, "func"), c=(0x1800, 0x8000, "func")
    )
    facts = report._sized(config, "v", table)
    assert sum(x[2] for x in facts.values()) <= (0x8000 - 0x1000) + (ROM - 0x8100)


def test_partition_of_a_version_sums_to_the_rom(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "blobs.toml").write_bytes(
        blobs.emit(
            {"schema": 1, "blob": {"b": {"type": "world", "state": "decoded", "rom": {"v": [[0x9000, 0xA000]]}}}}
        )
    )
    table = _table(f=(0x1000, 0x100, "func"), d=(0x1100, 0x200, "data"))
    out = report.partition(config, "v", report._sized(config, "v", table), ROM)
    assert (out["code"], out["data"], out["platform"], out["blobs"], out["blobs_decoded"]) == (
        0x100,
        0x200,
        0x1000,
        0x1000,
        0x1000,
    )
    assert out["unknown"] == ROM - 0x100 - 0x200 - 0x1000 - 0x1000


def _tree(digest):
    return N(digest=digest)


def _credits(monkeypatch, **rows):
    def row(name, rom, size, kind):
        member = report.Symbol("function" if kind == "func" else "data", "", "unowned", rom)
        code, data = (size, 0) if kind == "func" else (0, size)
        return report.Credit(name, member, None, "unknown", "v", code, data, 0, False, 0.0)

    monkeypatch.setattr(report, "credits", lambda config, tree: tuple(row(n, *r) for n, r in rows.items()))


def test_report_carries_the_blob_fields_per_version(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _credits(monkeypatch, f=(0x1000, 0x100, "func"))
    held = report.accounts(config, _tree("acc"))["v"]
    assert held == {
        "rom_size": ROM,
        "platform_total": 0x1000,
        "blobs_total": 0,
        "blobs_decoded": 0,
        "blobs_identified": 0,
        "unknown_total": ROM - 0x1100,
        "blobs": {},
    }
    assert report.layers(config, _tree("acc"), "v")["code"] == [(0x1000, 0x1100)]


def test_a_byte_counted_twice_is_the_blocking_finding(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _credits(monkeypatch, a=(0x1000, 0x100, "func"), b=(0x1080, 0x100, "data"))
    with pytest.raises(Refusal) as error:
        report.partitions(config, _tree("twice"))
    (finding,) = error.value.findings
    assert finding.key == "report.partition" and finding.blocking


def test_a_rom_that_is_absent_gives_no_partition(tmp_path, monkeypatch):
    _credits(monkeypatch)
    config = _config(tmp_path)
    (tmp_path / "roms" / "baserom.v.z64").unlink()
    assert report.partitions(config, _tree("none")) == {}


# ---------------------------------------------------------------- import
def _packet(config, **row):
    rom = sha256(bytes(ROM))
    artifact = "ab" * 32
    blob = {
        "type": "world",
        "state": "decoded",
        "rom": {"v": [[0x2000, 0x3000]]},
        "evidence": [{"artifact": "a.json", "sha256": artifact}],
        **row,
    }
    return {"schema": 1, "rom_sha256": {"v": rom}, "artifacts": {"a.json": artifact}, "blob": {"level.0": blob}}


def test_import_plan_merges_a_pinned_packet_and_applies_it(tmp_path):
    config = _config(tmp_path)
    plan = blobs.import_plan(config, _packet(config))
    assert not plan.blocking and set(plan.writes) == {"blobs.toml"} and plan.message == "Add 1 blobs to blobs.toml"
    done = blobs.apply_import(config, plan)
    assert done["add"] == "git add blobs.toml" and "git commit -m" in done["commit"]
    assert [r["id"] for r in blobs.rows(config)] == ["level.0"]
    assert blobs.import_plan(config, _packet(config)).writes == plan.writes  # idempotent


def test_import_refuses_a_rom_or_evidence_sha_that_does_not_match(tmp_path):
    config = _config(tmp_path)
    packet = _packet(config)
    packet["rom_sha256"]["v"] = "00" * 32
    packet["artifacts"]["a.json"] = "cd" * 32
    plan = blobs.import_plan(config, packet)
    assert {f.key for f in plan.blocking} == {"blobs.evidence"} and len(plan.blocking) == 2
    with pytest.raises(ValueError):
        blobs.apply_import(config, plan)


def test_import_checks_the_merged_file(tmp_path):
    config = _config(tmp_path)
    plan = blobs.import_plan(config, _packet(config, rom={"v": [[0x2000, ROM + 1]]}))
    assert {f.key for f in plan.blocking} == {"blobs.range"}


def test_shadowed_blobs_are_non_blocking_findings(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "blobs.toml").write_bytes(blobs.emit(DOC))
    (found,) = blobs.shadows(config, "v", [(0x2800, 0x2900)])
    assert found.key == "blobs.shadowed" and not found.blocking and "256 bytes" in found.reason
