"""Proof rejection boundaries: aliases, nulls, uncovered bytes, compression tails and observed-only inputs."""

import json
from types import SimpleNamespace as N

import pytest

from unbake import blob_decode, blob_proof, blobs, config
from unbake.contracts import Refusal, sha256


def words(*values):
    return b"".join(v.to_bytes(4, "big") for v in values)


def layout():
    return {
        "root": "Root",
        "null_values": [],
        "types": {
            "Root": {"directory": [None, "Refs"]},
            "Refs": {"leaf": "Ref"},
            "Ref": {"fields": "@0:Word @0:Halves"},
            "Word": {"fields": "4"},
            "Halves": {"fields": "22"},
        },
    }


def alias_body():
    return words(2, 16, 20, 36) + bytes.fromhex("12345678") + words(8, 1, 0, 0)


def test_aliased_widths_and_zero_offset_are_independent_lossless_views():
    body = alias_body()
    proof = blob_decode.Graph(body, layout()).finish()
    assert proof["typed_bytes"] == len(body)
    assert proof["sha256"] == proof["reemit_sha256"]
    views = {r["type"]: r for r in proof["records"]}
    assert views["Word"]["fields"] == [[4, 0x12345678]]
    assert views["Halves"]["fields"] == [[2, 0x1234], [2, 0x5678]]


def test_proof_refusals_have_a_public_diagnostic():
    assert "coverage" in config.sentence("blobs.proof")


def test_declared_null_zero_does_not_hide_untyped_pool():
    spec = layout()
    spec["null_values"] = [0]
    with pytest.raises(ValueError, match="unproved padding"):
        blob_decode.Graph(alias_body(), spec).finish()


@pytest.mark.parametrize("offset", [0xFFFFFFFF, 1, 5])
def test_references_must_fit_their_own_section(offset):
    body = alias_body()[:-8] + words(offset, offset)
    with pytest.raises(ValueError, match="outside extent"):
        blob_decode.Graph(body, layout()).finish()


def test_zero_content_without_alignment_is_not_typed():
    spec = {"root": "Root", "null_values": [], "types": {"Root": {"directory": [None]}}}
    body = words(1, 16, 24, 0) + bytes(8)
    with pytest.raises(ValueError, match="unproved padding"):
        blob_decode.Graph(body, spec).finish()


def test_one_byte_alignment_after_odd_width_component_is_proven():
    spec = layout()
    spec["types"]["Word"] = {"fields": "111"}
    spec["types"]["Halves"] = {"fields": "111"}
    body = words(2, 16, 20, 36) + b"abc\0" + words(8, 1, 0, 0)
    proof = blob_decode.Graph(body, spec).finish()
    assert proof["zero_alignment"] == [[19, 20]]
    with pytest.raises(ValueError, match="unproved padding"):
        blob_decode.Graph(body[:19] + b"x" + body[20:], spec).finish()


def test_directory_header_checks_count_offsets_and_parent_extent():
    header = words(2, 16, 24, 32)
    assert blob_decode.directory(header, header_only=True, extent=32)["count"] == 2
    for bad in (words(2, 16, 40, 32), words(2, 12, 24, 32), words(2, 16, 24, 40)):
        with pytest.raises(ValueError):
            blob_decode.directory(bad, header_only=True, extent=32)
    assert blob_decode.directory(bytes(8), header_only=True, extent=8)["count"] == 0
    with pytest.raises(ValueError):
        blob_decode.directory(words(1, 0, 0), header_only=True, extent=12)


def test_leaf_stride_and_head_tail_are_checked():
    spec = {"root": "Root", "null_values": [], "types": {"Root": {"leaf": "Value"}, "Value": {"fields": "4"}}}
    with pytest.raises(ValueError, match="stride"):
        blob_decode.Graph(words(8, 1, 1, 2), spec).finish()
    spec["types"]["Root"] = {"head_then": ["Value", "Value"]}
    proof = blob_decode.Graph(words(1, 2), spec).finish()
    assert proof["typed_bytes"] == 8


def fixture(root):
    rom = bytes(range(64)) + bytes(64) + b"\xff" * 128
    (root / "roms").mkdir()
    (root / "roms/baserom.v.z64").write_bytes(rom)
    cfg = N(
        project=N(
            root=root,
            version={
                "v": N(
                    size=len(rom),
                    segments=(
                        N(name="code", rom=0, vram=0x80000000, kind="code"),
                        N(name="archive", rom=64, vram=None, kind="bin"),
                    ),
                )
            },
        )
    )
    return cfg, rom


def test_observation_cannot_grant_data_credit_or_decode_zeros(tmp_path):
    cfg, rom = fixture(tmp_path)
    doc = {"blob": {}}
    row = {"id": 1, "region": [64, 68], "version": "v", "reader": "observe", "limit": "no reader"}
    proof = blob_proof.prove_one(cfg, row, rom, doc, [])
    assert proof["status"] == "observed" and proof["observation"]["all_zero"]
    assert not proof["context"]["loaded_data_candidate"]
    assert not proof["automatic_fill"]
    row["reader"] = "fill"
    with pytest.raises(ValueError, match="not fully covered"):
        blob_proof.prove_one(cfg, row, rom, doc, [])


def test_public_proof_dispatch_pins_rom_and_preserves_project(tmp_path, monkeypatch):
    cfg, rom = fixture(tmp_path)
    monkeypatch.setattr(blobs, "load", lambda _: {"schema": 1, "blob": {}})
    request = {
        "schema": 1,
        "rom_sha256": {"v": sha256(rom)},
        "rows": [{"id": "tail", "version": "v", "region": [64, 128], "reader": "fill"}],
    }
    source, target = tmp_path / "request.json", tmp_path / "receipt.json"
    source.write_text(json.dumps(request))
    result = blobs.run(cfg, {"action": ["prove", str(source), str(target)]})
    config.validate("result.blobs", result, "proof result")
    assert result["proved"]["rows"] == 1
    assert json.loads(target.read_text())["rows"][0]["status"] == "decoded"
    assert (tmp_path / "roms/baserom.v.z64").read_bytes() == rom
    with pytest.raises(Refusal, match="receipt exists"):
        blob_proof.run(cfg, str(source), str(target))
    request["rom_sha256"]["v"] = "0" * 64
    source.write_text(json.dumps(request))
    with pytest.raises(Refusal, match="ROM sha256 mismatch"):
        blob_proof.run(cfg, str(source), str(tmp_path / "no.json"))
    assert not (tmp_path / "no.json").exists()


def test_pinned_layout_refuses_changed_input(tmp_path):
    cfg, rom = fixture(tmp_path)
    path = tmp_path / "layout.json"
    path.write_text("{}")
    row = {
        "id": 1,
        "version": "v",
        "region": [0, 4],
        "reader": "graph",
        "evidence": "reader",
        "layout_file": str(path),
        "layout_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="layout file sha256"):
        blob_proof.prove_one(cfg, row, rom, {"blob": {}}, [])


def test_bank_pairs_are_typed_and_bounded(tmp_path):
    cfg, rom = fixture(tmp_path)
    row = {"id": 1, "version": "v", "region": [0, 16], "reader": "bank-pairs", "evidence": "loader"}
    rom = words(16, 32, 32, 64) + rom[16:]
    proof = blob_proof.prove_one(cfg, row, rom, {"blob": {}}, [])
    assert proof["decoded"]["banks"] == [(16, 32), (32, 64)]
    with pytest.raises(ValueError, match="tile"):
        blob_proof.prove_one(cfg, row, words(16, 32, 40, 64) + rom[16:], {"blob": {}}, [])


def test_packed_padding_is_relative_to_the_stored_record(tmp_path, monkeypatch):
    cfg, rom = fixture(tmp_path)
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps(
            {"blob_layouts": [{"blob": "x", "root": "Word", "null_values": [], "types": {"Word": {"fields": "4"}}}]}
        )
    )
    row = {
        "id": 1,
        "version": "v",
        "region": [4, 12],
        "reader": "graph",
        "evidence": "reader",
        "layout_file": str(path),
        "layout_sha256": sha256(path.read_bytes()),
        "layout_id": "x",
        "codec": "rnc1",
        "stored_alignment": 8,
    }
    monkeypatch.setattr(blob_proof.rnc, "header", lambda _: N(stored=7, unpacked=4, method=1))
    monkeypatch.setattr(blob_proof.rnc, "unpack", lambda _: words(123))
    rom = rom[:11] + b"\0" + rom[12:]
    assert blob_proof.prove_one(cfg, row, rom, {"blob": {}}, [])["status"] == "decoded"
