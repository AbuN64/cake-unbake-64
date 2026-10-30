"""Platform claims require exact ROM hashes and share one configuration with the fixed boot range."""

import os

import pytest
import tomlkit
from test_blobs import _config, _credits, _tree

from unbake import blobs, effort, platform, report
from unbake import config as configuration
from unbake.contracts import Refusal, sha1, sha256


def document():
    return {
        "schema": 1,
        "platform": {
            "boot": [[0, 4096]],
            "rsp": {
                "sdk": {
                    "format": "microcode: verified fixture",
                    "state": "identified",
                    "rom": {"v": [[0x8000, 0x8100]]},
                    "sha256": {"v": [sha256(bytes(range(256)))]},
                }
            },
        },
    }


def test_platform_is_boot_plus_exact_hash_rsp_and_has_no_boot_output(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "config" / "platform.toml").write_text(tomlkit.dumps(document()))
    assert platform.ranges(config, "v") == [(0, 4096), (0x8000, 0x8100)]
    p = report.partition(config, "v", {"f": (0x80001000, 0x1000, 16, "func")}, 0x10000)
    assert p["platform"] == 4352 and p["code"] == 16
    assert "boot" not in p and "boot" not in p["spans"]
    assert sum(p[k] for k in ("code", "data", "platform", "blobs", "unknown")) == 0x10000


def test_platform_and_blobs_files_are_read_from_the_configured_paths(tmp_path):
    config = _config(tmp_path)
    config.project.paths = {"platform": "config/platform.toml", "blobs": "config/blobs.toml"}
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "platform.toml").write_text(tomlkit.dumps(document()))
    (tmp_path / "rom.toml").write_text("not the platform file")  # the default name is no longer read
    assert platform.ranges(config, "v") == [(0, 4096), (0x8000, 0x8100)]
    assert blobs.sha(config) == "" and blobs.path_of(config) == "config/blobs.toml"
    (tmp_path / "config" / "blobs.toml").write_text("schema = 1\n")
    assert blobs.sha(config) != ""


@pytest.mark.parametrize(
    "change,key",
    [
        (lambda d: d["platform"]["rsp"]["sdk"]["sha256"]["v"].__setitem__(0, "0" * 64), "platform.hash"),
        (lambda d: d["platform"]["rsp"]["sdk"]["rom"]["v"].__setitem__(0, [0, 0x20000]), "platform.range"),
        (lambda d: d["platform"].update(boot=[[0, 0x9000]]), "platform.schema"),
    ],
)
def test_unverified_or_out_of_range_platform_claim_refuses(tmp_path, change, key):
    config = _config(tmp_path)
    data = document()
    change(data)
    (tmp_path / "config" / "platform.toml").write_text(tomlkit.dumps(data))
    with pytest.raises(Refusal) as error:
        platform.load(config)
    assert error.value.findings[0].key == key


def test_changed_rom_invalidates_partition_before_claiming_platform(tmp_path, monkeypatch):
    config = _config(tmp_path)
    (tmp_path / "config" / "platform.toml").write_text(tomlkit.dumps(document()))
    _credits(monkeypatch, f=(0x1000, 16, "func"))
    tree = _tree("platform-change")
    assert report.partitions(config, tree)["v"]["platform"] == 4352
    path = tmp_path / "roms/baserom.v.z64"
    data = bytearray(path.read_bytes())
    data[0x8000] = 1
    path.write_bytes(data)
    config.project.version["v"].sha1 = sha1(data)
    with pytest.raises(Refusal, match=r"platform\.hash"):
        report.partitions(config, tree)


def test_sha_ignores_rom_mtime_and_follows_content_at_same_size_and_mtime(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / "roms/baserom.v.z64"
    first = platform.sha(config)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert platform.sha(config) == first
    data = bytearray(path.read_bytes())
    data[0x10] ^= 1
    path.write_bytes(data)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    config.project.version["v"].sha1 = sha1(data)
    configuration._VERIFIED.clear()
    assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns
    assert platform.sha(config) != first


def test_valid_document_is_read_from_the_rom_once_and_ranges_is_pure(tmp_path, monkeypatch):
    config = _config(tmp_path)
    (tmp_path / "config" / "platform.toml").write_text(tomlkit.dumps(document()))
    reads = []
    real = platform.read
    monkeypatch.setattr(platform, "read", lambda *args: reads.append(args) or real(*args))
    effort._memo.clear()
    loaded = platform.load(config)
    platform.load(config)
    assert len(reads) == 1
    platform.ranges(config, "v", loaded)
    platform.ranges(config, "v")
    assert len(reads) == 1


def test_rom_read_is_exact_or_refuses(tmp_path):
    config = _config(tmp_path)
    assert len(platform.read(config, "v", 0x8000, 0x8100)) == 256
    with pytest.raises(Refusal, match=r"platform\.hash"):
        platform.read(config, "v", 0xFFFF0, 0x200000)
