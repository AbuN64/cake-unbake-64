"""Where a link goes wrong: the first object whose bytes the linker starts off the ROM address its first name has."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from unbake import process, store, symbols, tree
from unbake.contracts import Config, Finding, digest

_ENTRY = re.compile(r"^ (\.\S+)(?:[ \t]+|\n[ \t]+)0x([0-9a-f]+)[ \t]+0x([0-9a-f]+)[ \t]+(\S+\.o)$", re.M)
_KIND = {".rdata": ".rodata"}
_LINKED = re.compile(r"^\s*(\S+\.o)\(", re.M)  # an object link.ld takes sections of
_UNDEFINED = re.compile(r"undefined reference to [`'\u2018]([^'\u2019]+)['\u2019]")
_SHOWN = 8


def parse(map_text: str) -> list[tuple[str, str, int]]:
    """(object, section, address) of each nonempty input section of an ld map, in link order; .rdata is .rodata."""
    return [
        (m[4], _KIND.get(m[1], m[1]), int(m[2], 16))
        for m in _ENTRY.finditer(map_text)
        if int(m[3], 16) and m[1] != ".bss"
    ]


def walk(
    placed: Sequence[tuple[str, str, int]], expected: Mapping[tuple[str, str], int]
) -> tuple[str, int, int] | None:
    """The first (object, address its first name has, address the link gave it): an object that starts with bytes no
    name owns is off alone, so a shift is real when the next object with a name is off by the same amount."""
    seen, got = [], set()
    for obj, section, address in placed:
        if (obj, section) in expected and (obj, section) not in got:
            got.add((obj, section))
            seen.append((obj, expected[obj, section], address))
    for k, (obj, want, have) in enumerate(seen):
        later = seen[k + 1 :]
        if want != have and (not later or later[0][1] - later[0][2] == want - have):
            return obj, want, have
    return None


def first_off(config: Config, version: str) -> str:
    """One sentence naming the first object of `version`'s link that the linker places off the address of the first
    name it owns (an object before it too long or too short), or "" when every object is where its names say or when
    it cannot be told: a diagnostic never replaces the refusal it explains. The link runs again with a map."""
    try:
        found = _find(config, version)
    except Exception:
        return ""
    if found is None:
        return ""
    obj, want, have = found
    return (
        f"{version}: the first object off its ROM address is {obj}: its first name is at 0x{want:X} but the link "
        f"placed the object at 0x{have:X} ({have - want:+d} bytes): an object before it is {abs(have - want)} bytes "
        f"{'too long' if have > want else 'too short'}"
    )


def _find(config: Config, version: str) -> tuple[str, int, int] | None:
    root, held = config.project.root, tree.capture(config)
    table = symbols.load(config, version)
    expected: dict[tuple[str, str], int] = {}
    for module in held.modules:
        for name in module.owns.get(version, ()):
            if name in table:
                section = module.sections.get(version, {}).get(name) or ".text"
                key = (f"{module.path}.o", _KIND.get(section, section))
                expected[key] = min(table[name][0], expected.get(key, table[name][0]))
    scratch = process.scratch(root)
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / f"linkdiag-{version}.map"
    elf = scratch / f"linkdiag-{version}.elf"
    argv: list[str | Path] = [process.tool(config, "mips_ld"), "-T", "link.ld", f"-Map={out}", "-o", str(elf)]
    process.run("linkdiag", argv, root / "build" / version, tmp=scratch)
    text = out.read_text(errors="replace") if out.is_file() else ""
    out.unlink(missing_ok=True)
    elf.unlink(missing_ok=True)
    return walk(parse(text), expected)


def _linked_objects(config: Config, version: str) -> tuple[Path, list[str]] | None:
    """(build/<v>, the objects link.ld takes from sources: not the generated .bin.o) once the version has been built
    (some object link.ld names exists); None before, when there is nothing for a source to be stale against."""
    build = config.project.root / "build" / version
    try:
        script = (build / "link.ld").read_text(errors="replace")
    except OSError:
        return None
    names = [n for n in dict.fromkeys(_LINKED.findall(script)) if not n.endswith(".bin.o")]
    return (build, names) if any((build / n).is_file() for n in names) else None


def stale(config: Config, version: str) -> list[str]:
    """The objects `version`'s link takes whose build the tree has left behind, each as "OBJECT: why": the source is
    newer than the object, the source is gone (renamed or deleted) or the object was never made. Empty when the
    version was never built."""
    root = config.project.root
    built = _linked_objects(config, version)
    if built is None:
        return []
    build, names = built
    found = []
    for name in names:
        source, obj = name.removesuffix(".o"), build / name
        if not obj.is_file():
            found.append(f"{name}: not built")
        elif not (root / source).is_file():
            found.append(f"{name}: its source {source} is gone")
        elif (root / source).stat().st_mtime_ns > obj.stat().st_mtime_ns:
            found.append(f"{name}: older than its source {source}")
    return found


def unlinked(config: Config, version: str) -> tuple[list[str], str]:
    """(the symbols, the linker's last line) of `make`'s link of `version` (`ld -T link.ld` in build/<v>) when it fails:
    the undefined names, or the linker's own complaint when none is undefined; ([], "") when it links or the version
    was never built. The answer is kept per state of the link script and the objects."""
    built = _linked_objects(config, version)
    if built is None:
        return [], ""
    build, names = built
    state = [(n, (s := (build / n).stat()).st_mtime_ns, s.st_size) for n in names if (build / n).is_file()]
    script = (build / "link.ld").stat()
    key = digest(("linked", version, script.st_mtime_ns, script.st_size, state))
    kept = store.cached(config, "linked", key, lambda: json.dumps(_link(config, version)).encode())
    names_, error = json.loads(kept)
    return names_, error


def _link(config: Config, version: str) -> tuple[list[str], str]:
    root = config.project.root
    scratch = process.scratch(root)
    scratch.mkdir(parents=True, exist_ok=True)
    elf = scratch / f"linkcheck-{version}.elf"
    argv: list[str | Path] = [process.tool(config, "mips_ld"), "-T", "link.ld", "-o", str(elf)]
    done = process.run("linkcheck", argv, root / "build" / version, tmp=scratch)
    elf.unlink(missing_ok=True)
    if done.exit == 0:
        return [], ""
    text = done.stderr.decode(errors="replace")
    undefined = sorted(dict.fromkeys(_UNDEFINED.findall(text)))
    tail = next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), f"exit {done.exit}")
    return undefined, "" if undefined else tail


def unsound(config: Config, version: str) -> Finding | None:
    """The refusal a measure in `version` earns when the tree's build there cannot be trusted: stale objects first
    (the link of stale objects says nothing), then a link that fails; None when the tree stands behind the measure."""
    if objects := stale(config, version):
        shown = objects[:_SHOWN] + ([f"... {len(objects) - _SHOWN} more"] if len(objects) > _SHOWN else [])
        return Finding(
            "score.stale",
            f"{version}: {len(objects)} objects are stale against their sources, so a verdict here proves nothing "
            f"about the tree: {'; '.join(shown)}",
            versions=(version,),
            missing=tuple(objects),
            action="rebuild the tree (make check), then measure again",
        )
    undefined, error = unlinked(config, version)
    if undefined or error:
        shown = undefined[:_SHOWN] + ([f"... {len(undefined) - _SHOWN} more"] if len(undefined) > _SHOWN else [])
        what = f"undefined {', '.join(shown)}" if undefined else error
        return Finding(
            "score.link",
            f"{version} does not link: {what}, so a verdict here proves nothing about the tree",
            versions=(version,),
            missing=tuple(undefined) or (error,),
            action="make the version link (make check names the undefined symbols), then measure again",
        )
    return None
