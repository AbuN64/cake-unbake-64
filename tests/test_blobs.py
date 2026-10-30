"""Blobs: blobs.toml facts, the exact byte partition, the import plan and the report fields, on mocked inputs."""

import json
from pathlib import Path
from types import SimpleNamespace as N

import fixture
import pytest

from unbake import blobs, cli, human, report
from unbake.contracts import Refusal, sha1, sha256

PLATFORM = [(0, 0x1000)]
ROM = 0x10000


def image(size: int) -> bytes:
    """ROM bytes with no run of one value, so no gap is proven fill unless a test writes one."""
    return bytes(i & 0xFF for i in range(size))


def _config(root: Path, versions=("v",), rom=ROM, segments=None) -> N:
    (root / "config").mkdir(exist_ok=True)
    rows = segments or [
        ("retained", 0, "bin"),
        ("span_1000", 0x1000, "code"),
        ("rsp", 0x8000, "bin"),
        ("span_8100", 0x8100, "code"),
    ]
    version = {
        v: N(
            sha1=sha1(image(rom)),
            size=rom,
            segments=tuple(N(name=n, rom=r, vram=0x80000000 + r, kind=k) for n, r, k in rows),
            meta={},
        )
        for v in versions
    }
    (root / "roms").mkdir(exist_ok=True)
    for v in versions:
        (root / "roms" / f"baserom.{v}.z64").write_bytes(image(rom))
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


def test_proven_fill_is_matched_blob_bytes_and_unproven_runs_stay_unknown():
    rom = bytearray(range(256)) * (ROM // 256)
    rom[0x2010:0x2020] = bytes(0x10)  # gap 0x2010..0x2020 below the alignment of its end: fill
    rom[0x3004:0x3010] = b"\xaa" * 0xC  # 0xC below the 16-byte alignment of 0x3010: fill
    rom[0x5000:0x5800] = bytes(0x800)  # whole gap of zeros ending at 0x5800 (align 0x800): not below, unknown
    rom[ROM - 0x123 :] = b"\xff" * 0x123  # cart padding to the ROM end, counted from the 16-byte boundary
    decoded = [(0x1000, 0x2010), (0x2020, 0x3004), (0x3010, 0x5000), (0x5800, ROM - 0x200)]
    out = blobs.sweep(ROM, [], [], list(PLATFORM), decoded, [], bytes(rom))
    tail = ROM - 0x120
    assert out["spans"]["fill"] == [(0x2010, 0x2020), (0x3004, 0x3010), (tail, ROM)]
    assert out["blobs_fill"] == 0x10 + 0xC + 0x120
    assert out["blobs_decoded"] == blobs.length(decoded) + out["blobs_fill"]
    assert out["unknown"] == 0x800 + (tail - (ROM - 0x200))
    assert sum(out[k] for k in ("code", "data", "platform", "blobs", "unknown")) == ROM
    assert blobs.sweep(ROM, [], [], list(PLATFORM), decoded, [])["blobs_fill"] == 0  # no ROM bytes, no proof


def test_a_gap_splits_into_one_value_runs_each_proven_by_its_end():
    rom = bytearray(range(256)) * (ROM // 256)
    rom[0x6004:0x7000] = bytes(0xFFC)  # zeros up to 0x7000 (align 0x1000), then the cart tail of 0xFF
    rom[0x7000:] = b"\xff" * (ROM - 0x7000)
    decoded = [(0x1000, 0x6004)]
    out = blobs.sweep(ROM, [], [], list(PLATFORM), decoded, [], bytes(rom))
    assert out["spans"]["fill"] == [(0x6004, 0x7000), (0x7000, ROM)]
    assert out["blobs_fill"] == ROM - 0x6004 and out["unknown"] == 0
    assert blobs.fill(bytes(rom), [(0x6004, ROM)]) == [(0x6004, 0x7000), (0x7000, ROM)]
    # a run that is not at the gap's end, or whose end is not aligned past its length, stays unknown
    mixed = bytearray(range(256)) * (ROM // 256)
    mixed[0x2000:0x2010] = bytes(0x10)  # ends at 0x2010 but the gap goes on with other bytes
    assert blobs.fill(bytes(mixed), [(0x1F00, 0x2100)]) == []
    mixed[0x2100 - 8 : 0x2100] = bytes(8)  # a run shorter than RUN_MIN inside a longer gap proves nothing
    assert blobs.fill(bytes(mixed), [(0x1F00, 0x2100)]) == []
    mixed[0x2090:0x2100] = bytes(0x70)  # 0x70 < 0x100 (align of 0x2100): padding before the next item
    assert blobs.fill(bytes(mixed), [(0x1F00, 0x2100)]) == [(0x2090, 0x2100)]


def test_a_gap_of_two_values_is_never_fill():
    rom = bytearray(ROM)
    rom[0x2008] = 1
    out = blobs.sweep(ROM, [], [], [], [(0, 0x2000), (0x2010, ROM)], [], bytes(rom))
    assert out["blobs_fill"] == 0 and out["unknown"] == 0x10


def test_a_blob_over_a_credited_symbol_is_shadowed_and_stays_data():
    out = sweep(data=[(0x2000, 0x3000)], decoded=[(0x2800, 0x3800)])
    assert (out["data"], out["blobs_decoded"], out["shadowed"]) == (0x1000, 0x800, 0x800)


def test_platform_bytes_stay_platform_whatever_symbol_covers_them():
    assert sweep()["platform"] == 0x1000
    out = sweep(code=[(0xC00, 0x1400)], data=[(0x800, 0x900)])
    assert out["platform"] == 0x1000 and out["code"] == 0x400 and out["data"] == 0  # data 0x800.. lies in platform
    assert out["platform_overlap"] == 0x400 + 0x100  # code 0xC00..0x1000 and all of the data
    assert out["overlap"] == 0 and sum(out[k] for k in ("code", "data", "platform", "blobs", "unknown")) == ROM


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
    (tmp_path / "config" / "blobs.toml").write_bytes(
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

    monkeypatch.setattr(report, "_rows", lambda config, tree, exact, scored: tuple(row(n, *r) for n, r in rows.items()))


def test_scan_proves_catalogued_streams_and_lists_streams_in_unknown_bytes(tmp_path, monkeypatch):
    from test_rnc import _abcabc

    from unbake import tree

    config = _config(tmp_path)
    stream = _abcabc()
    rom = bytearray(image(ROM))
    rom[0x2000 : 0x2000 + len(stream)] = stream
    rom[0x3000 : 0x3000 + len(stream)] = stream
    rom[0x9000 : 0x9000 + len(stream)] = stream
    (tmp_path / "roms" / "baserom.v.z64").write_bytes(bytes(rom))
    config.project.version["v"].sha1 = sha1(bytes(rom))
    row = {"type": "graphics", "format": "f", "codec": "rnc1", "state": "decoded", "decoded_size": 9}
    document = {
        "schema": 1,
        "blob": {
            "good": {**row, "rom": {"v": [[0x2000, 0x2000 + len(stream)]]}},
            "short": {**row, "decoded_size": 8, "rom": {"v": [[0x3000, 0x3000 + len(stream)]]}},
        },
    }
    (tmp_path / "config" / "blobs.toml").write_bytes(blobs.emit(document))
    _credits(monkeypatch)
    monkeypatch.setattr(tree, "capture", lambda config: _tree("scan"))
    out = blobs.scan(config)["v"]
    assert out["rnc"]["checked"] == 2
    assert out["rnc"]["failed"] == [{"blob": "short", "reason": "decodes to 9 bytes, the row says 8"}]
    assert out["candidates"] == [
        {
            "rom": [0x9000, 0x9000 + len(stream)],
            "stored_size": len(stream),
            "codec": "rnc1",
            "decoded_size": 9,
            "sha256": sha256(b"abcabcabc"),
        }
    ]
    assert out["candidate_bytes"] == len(stream) and out["fill"] == 0


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
        "blobs_fill": 0,
        "blobs_no_row": ROM - 0x1100,
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
    rom = sha256(image(ROM))
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
    assert (
        not plan.blocking and set(plan.writes) == {"config/blobs.toml"} and plan.message == "Add 1 blobs to blobs.toml"
    )
    done = blobs.apply_import(config, plan)
    assert done["add"] == "git add config/blobs.toml" and "git commit -m" in done["commit"]
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
    (tmp_path / "config" / "blobs.toml").write_bytes(blobs.emit(DOC))
    (found,) = blobs.shadows(config, "v", [(0x2800, 0x2900)])
    assert found.key == "blobs.shadowed" and not found.blocking and "256 bytes" in found.reason


# ---------------------------------------------------------------- land
def _clean_project(config, monkeypatch, tmp_path):
    from unbake import policy, tree

    monkeypatch.setattr(tree, "capture", lambda c: _tree("land"))
    monkeypatch.setattr(policy, "audit_command", lambda params: {"paths": []})
    _credits(monkeypatch, f=(0x1000, 0x100, "func"))


def test_land_imports_a_packet_and_reports_the_credit_it_gains(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _clean_project(config, monkeypatch, tmp_path)
    out = blobs.land(config, _packet(config))
    assert (out["added"], out["unchanged"], out["files"]) == (1, 0, ["config/blobs.toml"])
    assert out["message"] == "Add 1 blobs to blobs.toml" and "landed" not in out
    assert out["delta"]["v"] == {"blobs_decoded": 0x1000, "blobs_identified": 0, "unknown": -0x1000}
    assert [r["id"] for r in blobs.rows(config)] == ["level.0"]
    again = blobs.land(config, _packet(config))  # idempotent: nothing written, no credit moves
    assert (again["added"], again["unchanged"], again["files"]) == (0, 1, [])
    assert human._written("blobs", out)[-1].startswith("  git commit -m ") and human._written("blobs", again) == []
    assert again["delta"]["v"] == {"blobs_decoded": 0, "blobs_identified": 0, "unknown": 0}


def test_land_puts_the_old_file_back_when_a_check_after_the_write_fails(tmp_path, monkeypatch):
    from unbake import policy

    config = _config(tmp_path)
    _clean_project(config, monkeypatch, tmp_path)
    (tmp_path / "config" / "blobs.toml").write_bytes(blobs.emit(DOC))
    kept = (tmp_path / "config" / "blobs.toml").read_bytes()

    def refuse(params):
        raise Refusal(blobs.Finding("audit.credit_drop", "lost"))

    monkeypatch.setattr(policy, "audit_command", refuse)
    with pytest.raises(Refusal):
        blobs.land(config, _packet(config))
    assert (tmp_path / "config" / "blobs.toml").read_bytes() == kept


def test_land_refuses_a_packet_the_import_refuses_and_writes_nothing(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _clean_project(config, monkeypatch, tmp_path)
    packet = _packet(config)
    packet["rom_sha256"]["v"] = "00" * 32
    with pytest.raises(Refusal):
        blobs.land(config, packet)
    assert not (tmp_path / "config" / "blobs.toml").exists()


def test_blobs_toml_is_no_build_or_score_input():
    """Facts-only landing skips the ROM build, and a blobs.toml edit rescores no code: no build, tree, rescore or
    setup module reads the blobs file."""
    import ast

    root = Path(blobs.__file__).parent
    for name in ("build", "tree", "rescore", "repo", "pool", "publish"):
        imported = {
            alias.name
            for node in ast.walk(ast.parse((root / f"{name}.py").read_text()))
            if isinstance(node, ast.ImportFrom) and node.module == "unbake"
            for alias in node.names
        }
        assert "blobs" not in imported, name


# ---------------------------------------------------------------- supersedes
def _observed(config, spans=((0x2400, 0x2800),), state="observed"):
    row = {"type": "sn-debug", "state": state, "format": "accounting", "rom": {"v": [list(s) for s in spans]}}
    (config.project.root / "config" / "blobs.toml").write_bytes(
        blobs.emit({"schema": 1, "blob": {"accounting.sn-debug.v": row}})
    )


def test_a_packet_supersedes_the_observed_rows_its_decoded_rows_cover(tmp_path):
    config = _config(tmp_path)
    _observed(config)
    packet = _packet(config, type="leftover", format="sn-debug")
    packet["supersedes"] = ["accounting.sn-debug.v", "accounting.gone.v"]  # an id already gone is no error
    plan = blobs.import_plan(config, packet)
    assert not plan.blocking and plan.message == "Add 1 blobs to blobs.toml, superseding 1 observed rows"
    blobs.apply_import(config, plan)
    assert [r["id"] for r in blobs.rows(config)] == ["level.0"]
    again = blobs.import_plan(config, packet)  # re-import: nothing left to supersede, same file
    assert not again.blocking and again.writes == plan.writes


@pytest.mark.parametrize(
    "spans, state, extra, why",
    [
        (((0x2400, 0x3400),), "observed", {}, "bytes are not covered"),  # past the packet's row
        (((0x2400, 0x2800),), "decoded", {}, "only an observed row"),
        (((0x2400, 0x2800),), "observed", {"supersedes": ["level.0"]}, "both superseded and written"),
    ],
)
def test_supersede_refuses_a_loose_decoded_or_self_written_row(tmp_path, spans, state, extra, why):
    config = _config(tmp_path)
    _observed(config, spans, state)
    packet = _packet(config)
    packet["supersedes"] = extra.get("supersedes", ["accounting.sn-debug.v"])
    plan = blobs.import_plan(config, packet)
    assert plan.blocking and {f.key for f in plan.blocking} == {"blobs.supersede"}
    assert any(why in f.reason for f in plan.blocking)


def test_land_supersedes_and_reports_the_count(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _clean_project(config, monkeypatch, tmp_path)
    _observed(config)
    packet = _packet(config, type="leftover", format="sn-debug")
    packet["supersedes"] = ["accounting.sn-debug.v"]
    out = blobs.land(config, packet)
    assert (out["added"], out["superseded"]) == (1, 1)
    assert [r["id"] for r in blobs.rows(config)] == ["level.0"]
    assert blobs.land(config, packet)["superseded"] == 0


def test_a_leftover_row_is_sn_debug_or_earlier_link_only(tmp_path):
    config = _config(tmp_path)
    for fmt in ("sn-debug", "earlier-link"):
        assert not blobs.import_plan(config, _packet(config, type="leftover", format=fmt)).blocking
    plan = blobs.import_plan(config, _packet(config, type="leftover", format="stale copy"))
    assert {f.key for f in plan.blocking} == {"blobs.schema"}


def test_the_land_command_returns_the_land_record_not_nested(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _clean_project(config, monkeypatch, tmp_path)
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps(_packet(config)))
    out = blobs.run(config, {"action": ["land", str(packet)]})
    assert out["files"] == ["config/blobs.toml"] and out["message"] and "landed" not in out
    assert human._written("blobs", out)[0] == "Written, not committed. To commit:"


def _good():
    document = json.loads(json.dumps(DOC))
    document["blob"]["sprite.1"]["evidence"] = [{"artifact": "s.json", "sha256": "cd" * 32}]
    return document


def _verify(tmp_path, monkeypatch, document, partition=None, text=None):
    """blobs.verify on a written blobs.toml, the tree and the partition mocked."""
    config = _config(tmp_path)
    (tmp_path / "config" / "blobs.toml").write_bytes(text if text is not None else blobs.emit(document))
    monkeypatch.setattr("unbake.tree.capture", lambda cfg, *a, **k: None)
    sums = {
        "rom_size": ROM,
        "code": 0,
        "data": 0,
        "platform": 0,
        "blobs": 0,
        "unknown": ROM,
        "overlap": 0,
        "outside": 0,
        "shadowed": 0,
    }
    monkeypatch.setattr(report, "partitions", lambda cfg, tree: {"v": {**sums, **(partition or {})}})
    return {f.key: f.reason for f in blobs.verify(config)}


def test_verify_passes_the_facts_land_would_accept(tmp_path, monkeypatch):
    assert _verify(tmp_path, monkeypatch, _good()) == {}


def test_verify_refuses_a_malformed_or_missing_evidence_sha256(tmp_path, monkeypatch):
    bad = _good()
    bad["blob"]["level.0"]["evidence"] = [{"artifact": "a.json", "sha256": "not-a-sha"}]
    assert "blobs.schema" in _verify(tmp_path, monkeypatch, bad)  # the schema pins the shape of a sha256
    bare = _good()
    bare["blob"]["level.0"].pop("evidence", None)
    assert "blobs.evidence" in _verify(tmp_path, monkeypatch, bare)


def test_verify_refuses_two_credited_rows_over_the_same_bytes(tmp_path, monkeypatch):
    twin = _good()
    twin["blob"]["level.1"] = json.loads(json.dumps(twin["blob"]["level.0"]))
    assert "blobs.range" in _verify(tmp_path, monkeypatch, twin)


def test_verify_accepts_an_overlap_the_base_already_had_and_refuses_a_new_one(tmp_path, monkeypatch):
    nested = _good()
    nested["blob"]["level.1"] = json.loads(json.dumps(nested["blob"]["level.0"]))
    assert "blobs.range" in _verify(tmp_path, monkeypatch, nested)
    config = _config(tmp_path)
    assert {f.key for f in blobs.verify(config, blobs.emit(nested))} == set()  # the base had the pair
    assert {f.key for f in blobs.verify(config, blobs.emit(_good()))} == {"blobs.range"}


def test_verify_refuses_a_file_that_is_not_the_canonical_emit(tmp_path, monkeypatch):
    assert "blobs.land" in _verify(tmp_path, monkeypatch, _good(), text=blobs.emit(_good()) + b"# hand edit\n")


def test_verify_refuses_a_partition_that_does_not_sum_or_overlaps(tmp_path, monkeypatch):
    assert "blobs.land" in _verify(tmp_path, monkeypatch, _good(), partition={"unknown": 5})
    assert "blobs.land" in _verify(tmp_path, monkeypatch, _good(), partition={"shadowed": 4})


def test_land_refuses_new_duplicate_claims_before_capture_or_write(tmp_path, monkeypatch):
    config = _config(tmp_path)
    initial = blobs.emit(_good())
    path = tmp_path / "config" / "blobs.toml"
    path.write_bytes(initial)
    packet = _packet(config)
    packet["blob"]["duplicate"] = packet["blob"].pop("level.0")
    monkeypatch.setattr(blobs, "_partitions", lambda c: pytest.fail("capture before duplicate refusal"))
    with pytest.raises(Refusal) as error:
        blobs.land(config, packet)
    assert [(f.key, f.versions) for f in error.value.findings] == [("blobs.range", ("v",))]
    assert "duplicate" in error.value.findings[0].reason
    assert path.read_bytes() == initial


def test_import_preserves_existing_duplicate_pairs_but_refuses_a_new_claim(tmp_path):
    config = _config(tmp_path)
    document = _good()
    document["blob"]["prior"] = json.loads(json.dumps(document["blob"]["level.0"]))
    (tmp_path / "config" / "blobs.toml").write_bytes(blobs.emit(document))
    packet = _packet(config)
    assert not blobs.import_plan(config, packet).blocking
    packet["blob"]["new"] = packet["blob"].pop("level.0")
    findings = blobs.import_plan(config, packet).blocking
    assert len(findings) == 2 and all(f.key == "blobs.range" for f in findings)


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
def test_partition_preserves_scored_extents_without_measuring_bodies(tmp_path, monkeypatch, score):
    config = _config(tmp_path)
    module = N(
        path="src/game.c",
        segment="main",
        owns={"v": ("boot", "exact", "draft", "data")},
        stand_ins={"v": ("draft",)},
        fuzzy=("draft",),
    )
    tree = N(
        digest="partition-unscored",
        modules=(module,),
        owner=lambda version, name: module.path if name in module.owns[version] else None,
    )
    facts = {
        "v": {
            "boot": (0x80000FF0, 0xFF0, 0x10, "func"),
            "exact": (0x80001000, 0x1000, 0x20, "func"),
            "draft": (0x80001020, 0x1020, 0x40, "func"),
            "data": (0x80001060, 0x1060, 0x20, "data"),
            "label": (0x80001070, 0x1070, 0x10, "data"),
        }
    }
    monkeypatch.setattr(report, "facts", lambda config: facts)
    monkeypatch.setattr(report, "_extents", lambda config, tree: {"v": {"data": 0x20}})
    monkeypatch.setattr(report, "_broken", lambda config, tree: frozenset())
    scored = report._rows(config, tree, frozenset({"v"}), {"draft": {"v": score}})
    table = {r.name: (0, r.member.rom, r.code + r.data, "func" if r.code else "data") for r in scored}
    expected = report.partition(config, "v", table, ROM)
    monkeypatch.setattr(report.rescore, "rows", lambda *a: pytest.fail("partition requested fuzzy compilation"))
    monkeypatch.setattr(report, "credits", lambda *a: pytest.fail("partition requested scored credits"))
    actual = report.partitions(config, tree)["v"]
    assert actual == expected
    assert actual["code"] == 0x60 and actual["data"] == 0x20
    assert actual["platform"] == 0x1000 and actual["unknown"] == ROM - 0x1080
    assert actual["overlap"] == actual["outside"] == 0


@pytest.mark.parametrize(
    "kind", ["texture", "sprite", "font", "layout", "sound", "sequence", "level", "microcode", "blob"]
)
def test_reader_rejects_retired_types(tmp_path, kind):
    config = _config(tmp_path)
    data = blobs.emit({"schema": 1, "blob": {"x": {"type": kind, "state": "identified"}}})
    with pytest.raises(Refusal, match=r"blobs\.schema"):
        blobs.parse(config, data)


def test_migrate_is_not_a_blobs_action(tmp_path, capsys, toolchains):
    root = fixture.project(tmp_path, sources={}, symbols={"a": ""})
    args = ["--json", "--host", str(fixture.host(tmp_path)), "--project", str(root)]
    assert cli.main(["blobs", "migrate", *args]) == 1
    output = json.loads(capsys.readouterr().out)
    assert not output["ok"]
    assert "unknown migrate" in output["findings"][0]["Message"]
    assert cli.main(["blobs", *args]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["count"] == 0
