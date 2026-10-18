"""Platform ROM content: the fixed boot range and exact-hash RSP binaries declared in project rom.toml."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, Refusal, digest, sha256

FILE = "rom.toml"


def load(config: Config) -> Json:
    """Project platform facts, or the packaged boot facts when no project rom.toml exists."""
    path = config.project.root / FILE
    document = (
        configuration.toml("platform", path.read_bytes(), FILE, "platform.schema")
        if path.is_file()
        else {"schema": 1, "platform": deepcopy(configuration.load_resource(FILE)["platform"])}
    )
    validate(config, document)
    return document


def validate(config: Config, document: Json) -> None:
    """Every RSP range names a project version and matches its declared SHA-256 exactly."""
    configuration.validate("platform", document, FILE)
    block = document["platform"]
    if block["boot"] != configuration.load_resource(FILE)["platform"]["boot"]:
        raise Refusal(Finding("platform.schema", "boot must retain the packaged platform range; put RSP spans in rsp"))
    for name, row in block.get("rsp", {}).items():
        if set(row["rom"]) != set(row["sha256"]):
            raise Refusal(Finding("platform.hash", f"{name}: ROM versions and hash versions differ", path=FILE))
        for version, spans in row["rom"].items():
            if version not in config.project.version or len(spans) != len(row["sha256"][version]):
                raise Refusal(
                    Finding("platform.range", f"{name}: unknown version or one hash per span missing", path=FILE)
                )
            for (start, end), wanted in zip(spans, row["sha256"][version], strict=True):
                if not 0 <= start < end <= config.project.version[version].size:
                    raise Refusal(Finding("platform.range", f"{name}: range outside {version}", path=FILE))
                path = configuration.rom(config.project, version)
                try:
                    with path.open("rb") as handle:
                        handle.seek(start)
                        raw = handle.read(end - start)
                except OSError as error:
                    raise Refusal(Finding("platform.hash", str(error), path=str(path))) from error
                if len(raw) != end - start or sha256(raw) != wanted:
                    raise Refusal(
                        Finding("platform.hash", f"{name}: {version} range does not match its exact SHA-256", path=FILE)
                    )


def ranges(config: Config, version: str, document: Json | None = None) -> list[tuple[int, int]]:
    """Boot and verified RSP spans in one version; a proposed document is validated before use too."""
    document = load(config) if document is None else document
    validate(config, document)
    block = document["platform"]
    return [tuple(span) for span in block["boot"]] + [
        tuple(span) for row in block.get("rsp", {}).values() for span in row["rom"].get(version, ())
    ]


def sha(config: Config) -> str:
    """The declaration and ROM file identities that invalidate a cached byte partition."""
    path = config.project.root / FILE
    data = path.read_bytes() if path.is_file() else repr(configuration.load_resource(FILE)["platform"]).encode()
    roms = []
    for version in config.project.version:
        rom = Path(config.project.root) / "roms" / f"baserom.{version}.z64"
        stat = rom.stat() if rom.is_file() else None
        roms.append((version, stat.st_size if stat else 0, stat.st_mtime_ns if stat else 0))
    return digest((sha256(data), roms))
