"""Platform ROM content: the fixed boot range and exact-hash RSP binaries declared in project rom.toml."""

from __future__ import annotations

from copy import deepcopy

from unbake import config as configuration
from unbake import effort
from unbake.contracts import Config, Finding, Json, Refusal, digest, sha256

RESOURCE = "rom.toml"  # the packaged platform facts a project without its own file gets


def file(config: Config) -> str:
    """The repo-relative path of the project platform file: config.toml [paths] platform over repo.toml's."""
    return str(configuration.paths(config.project)["platform"])


def load(config: Config) -> Json:
    """Project platform facts, or the packaged boot facts when the project has no platform file."""
    name = file(config)
    path = config.project.root / name
    document = (
        configuration.toml("platform", path.read_bytes(), name, "platform.schema")
        if path.is_file()
        else {"schema": 1, "platform": deepcopy(configuration.load_resource(RESOURCE)["platform"])}
    )
    validate(config, document)
    return document


def read(config: Config, version: str, start: int, end: int) -> bytes:
    """The bytes [start, end) of one version's verified ROM; an unreadable or short ROM refuses."""
    path = configuration.rom(config.project, version)
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            raw = handle.read(end - start)
    except OSError as error:
        raise Refusal(Finding("platform.hash", str(error), path=str(path))) from error
    if len(raw) != end - start:
        raise Refusal(Finding("platform.hash", f"{version}: ROM ends before byte {end}", path=str(path)))
    return raw


def validate(config: Config, document: Json) -> None:
    """Every RSP range names a project version and matches its declared SHA-256 exactly; checked once per document
    content and configured ROM identity."""
    identities = tuple((v, row.sha1) for v, row in sorted(config.project.version.items()))
    key = ("platform-valid", str(config.project.root), digest(document), identities)
    effort.memo(key, lambda: _validate(config, document))


def _validate(config: Config, document: Json) -> bool:
    name = file(config)
    configuration.validate("platform", document, name)
    block = document["platform"]
    if block["boot"] != configuration.load_resource(RESOURCE)["platform"]["boot"]:
        raise Refusal(Finding("platform.schema", "boot must retain the packaged platform range; put RSP spans in rsp"))
    for name, row in block.get("rsp", {}).items():
        if set(row["rom"]) != set(row["sha256"]):
            raise Refusal(Finding("platform.hash", f"{name}: ROM versions and hash versions differ", path=name))
        for version, spans in row["rom"].items():
            if version not in config.project.version or len(spans) != len(row["sha256"][version]):
                raise Refusal(
                    Finding("platform.range", f"{name}: unknown version or one hash per span missing", path=name)
                )
            for (start, end), wanted in zip(spans, row["sha256"][version], strict=True):
                if not 0 <= start < end <= config.project.version[version].size:
                    raise Refusal(Finding("platform.range", f"{name}: range outside {version}", path=name))
                raw = read(config, version, start, end)
                if sha256(raw) != wanted:
                    raise Refusal(
                        Finding("platform.hash", f"{name}: {version} range does not match its exact SHA-256", path=name)
                    )
    return True


def ranges(config: Config, version: str, document: Json | None = None) -> list[tuple[int, int]]:
    """Boot and RSP spans in one version of a validated document (load's, or one passed through validate)."""
    document = load(config) if document is None else document
    block = document["platform"]
    return [tuple(span) for span in block["boot"]] + [
        tuple(span) for row in block.get("rsp", {}).values() for span in row["rom"].get(version, ())
    ]


def sha(config: Config) -> str:
    """The declaration and ROM content identities that invalidate a cached byte partition."""
    path = config.project.root / file(config)
    data = path.read_bytes() if path.is_file() else repr(configuration.load_resource(RESOURCE)["platform"]).encode()
    roms = []
    for version, row in config.project.version.items():
        try:
            configuration.rom(config.project, version)
        except Refusal as refusal:
            if "does not exist" not in refusal.findings[0].reason:
                raise
            roms.append((version, ""))
        else:
            roms.append((version, row.sha1))
    return digest((sha256(data), roms))
