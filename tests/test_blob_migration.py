"""The explicit migration preserves type detail and exact per-version coverage; normal readers reject old types."""

import json
from contextlib import contextmanager
from copy import deepcopy

import fixture
import pytest
from test_blobs import _config

from unbake import blob_migration, blobs, cli, journal, platform
from unbake.contracts import Refusal, sha256

MAPPING = {
    "texture": "graphics",
    "sprite": "graphics",
    "font": "graphics",
    "layout": "graphics",
    "model": "model",
    "sound": "audio",
    "sequence": "audio",
    "level": "world",
    "strings": "text",
    "messages": "text",
}


def legacy():
    rows = {
        kind: {
            "type": kind,
            "format": "rgba16" if kind == "texture" else kind.upper() + " details",
            "state": "identified",
            "rom": {v: [[0x1000 + i * 32, 0x1010 + i * 32]] for v in ("v", "w")},
        }
        for i, kind in enumerate(MAPPING)
    }
    rows["sdk"] = {
        "type": "microcode",
        "format": "microcode SDK",
        "state": "identified",
        "name": "known SDK",
        "codec": "none",
        "stored_size": 256,
        "rom": {v: [[0x8000, 0x8100]] for v in ("v", "w")},
        "evidence": [{"artifact": "sdk.bin", "sha256": sha256(bytes(256))}],
    }
    return {"schema": 1, "blob": rows}


def test_migration_preserves_metadata_and_coverage_with_verified_readback(tmp_path):
    config = _config(tmp_path, versions=("v", "w"))
    old = legacy()
    path = tmp_path / "blobs.toml"
    path.write_bytes(b"# keep this review comment\n" + blobs.emit(old))
    result = blob_migration.run(config)
    assert result["before"] == result["after"] == {"v": 4512, "w": 4512}
    assert result["readback"] and result["types"] == 9 and result["platform"] == 1
    now = blobs.load(config)
    assert {row["type"] for row in now["blob"].values()} == {"graphics", "model", "audio", "world", "text"}
    assert now["blob"]["texture"]["format"] == "texture: rgba16"
    assert now["blob"]["sound"]["format"] == "SOUND details"
    for name in MAPPING:
        assert now["blob"][name]["rom"] == old["blob"][name]["rom"]
    assert b"# keep this review comment" in path.read_bytes()
    sdk = platform.load(config)["platform"]["rsp"]["sdk"]
    assert sdk["name"] == "known SDK" and sdk["evidence"] == old["blob"]["sdk"]["evidence"]
    assert sdk["format"] == "microcode SDK" and sdk["sha256"] == {v: [sha256(bytes(256))] for v in ("v", "w")}
    assert "type" not in sdk
    again = blob_migration.run(config)
    assert again["files"] == [] and again["types"] == again["platform"] == 0
    assert again["before"] == again["after"] == result["before"]


@pytest.mark.parametrize("kind", [*sorted(set(MAPPING) - {"model"}), "microcode", "blob"])
def test_normal_reader_rejects_all_retired_types(tmp_path, kind):
    config = _config(tmp_path)
    data = blobs.emit({"schema": 1, "blob": {"x": {"type": kind, "state": "identified"}}})
    with pytest.raises(Refusal, match=r"blobs\.schema"):
        blobs.parse(config, data)


@pytest.mark.parametrize(
    "row",
    [
        {"type": "blob", "state": "observed", "rom": {"v": [[0x8000, 0x8100]]}},
        {"type": "microcode", "state": "observed", "rom": {"v": [[0x8000, 0x8100]]}},
    ],
)
def test_unreviewed_content_refuses_without_writing(tmp_path, row):
    config = _config(tmp_path)
    path = tmp_path / "blobs.toml"
    before = blobs.emit({"schema": 1, "blob": {"review_me": row}})
    path.write_bytes(before)
    with pytest.raises(Refusal, match=r"blobs\.migrate"):
        blob_migration.run(config)
    assert path.read_bytes() == before and not (tmp_path / "rom.toml").exists()


def test_bad_readback_cannot_report_success(tmp_path, monkeypatch):
    config = _config(tmp_path, versions=("v", "w"))
    (tmp_path / "blobs.toml").write_bytes(blobs.emit(legacy()))
    real = blobs.load

    def lose(config):
        document = deepcopy(real(config))
        document["blob"].pop("texture")
        return document

    monkeypatch.setattr(blobs, "load", lose)
    with pytest.raises(Refusal, match="readback byte coverage"):
        blob_migration.run(config)


def test_changed_platform_input_refuses_even_when_migration_only_writes_blobs(tmp_path, monkeypatch):
    config = _config(tmp_path, versions=("v", "w"))
    old = legacy()
    old["blob"].pop("sdk")
    path = tmp_path / "blobs.toml"
    before = blobs.emit(old)
    path.write_bytes(before)
    held = journal.held

    @contextmanager
    def changed(config):
        with held(config):
            (tmp_path / "rom.toml").write_text("# concurrently added platform facts\n")
            yield

    monkeypatch.setattr(journal, "held", changed)
    with pytest.raises(Refusal, match=r"journal\.changed"):
        blob_migration.run(config)
    assert path.read_bytes() == before


def test_public_migration_command_reads_back_two_versions(tmp_path, capsys, toolchains):
    old = legacy()
    for row in old["blob"].values():
        row["rom"] = {v: spans for v, spans in zip(("a", "b"), row["rom"].values(), strict=True)}
    old["blob"]["sdk"]["rom"] = {v: [[0x3000, 0x3100]] for v in ("a", "b")}
    root = fixture.project(tmp_path, sources={}, symbols={"a": "", "b": ""}, extra={"blobs.toml": blobs.emit(old)})
    host = fixture.host(tmp_path)
    args = ["blobs", "migrate", "--json", "--host", str(host), "--project", str(root)]
    assert cli.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"]
    result = output["result"]
    assert result["migrated"]["before"] == result["migrated"]["after"] == {"a": 4512, "b": 4512}
    assert result["migrated"]["readback"]
    assert result["count"] == 10
    assert {row["type"] for row in result["rows"]} == {"graphics", "model", "audio", "world", "text"}
    for partition in result["partition"].values():
        assert partition["platform"] == 4352
        assert sum(partition[k] for k in ("code", "data", "platform", "blobs", "unknown")) == 0x4000
        assert "boot" not in partition
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["result"]["migrated"]["files"] == []
