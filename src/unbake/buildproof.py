"""`unbake buildproof`: the build proof of a change that must not alter any object, the one path every cleanup uses.

The tree sits clean at REV. The proof builds the objects the PATCHes can reach (every touched unit and every unit that
includes a touched header, in every version), keeps them, applies the patches, rebuilds, and compares object by object
(section bytes and relocation offsets and kinds; names are free, a rename changes them). It fails, never passes
quietly, when:
- a make run ends with an error other than a unit the version has no rule for (`buildproof.make`);
- an object built before is gone after the change, or one appears that no patch added (`buildproof.set`);
- an object differs (`buildproof.differs`);
- nothing at all could be compared (`buildproof.empty`): a proof of zero objects proves nothing.
The objects to build are named from the files, never read from earlier build output, so a file never built before is
built (and counted) like any other. A unit of any kind is covered (.c, .CPP, .s); a unit a patch adds is built and
reported as added. Anything else (a moved function, a changed type) differs and fails: only a build can say so."""

from __future__ import annotations

import contextlib
import re
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection

from unbake import build, hdrnarrow, process, store, symbols
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, NativeResult, Refusal

_INCLUDE = re.compile(rb'^[ \t]*#[ \t]*include[ \t]*["<]([^">]+)[">]', re.M)
_NO_RULE = re.compile(r"No rule to make target '([^']+)'(?!, needed by)")
_FREE = (".mdebug", ".comment", ".symtab", ".strtab", ".shstrtab")  # sections whose bytes are names
SHOWN = 8


def _changes(text: str) -> list[tuple[str | None, str | None]]:
    """(old path, new path) per file a git patch touches: old None = added, new None = deleted."""
    rows: list[tuple[str | None, str | None]] = []
    old: str | None = None
    new: str | None = None
    head = ""
    for line in text.splitlines():
        if line.startswith("diff --git "):
            if head:
                rows.append((old, new))
            head = line
            match = re.match(r"diff --git a/(.+) b/(.+)$", line)
            old, new = (match.group(1), match.group(2)) if match else (None, None)
        elif line.startswith("new file mode"):
            old = None
        elif line.startswith("deleted file mode"):
            new = None
    if head:
        rows.append((old, new))
    return rows


def _compose(patches: Sequence[Path]) -> tuple[dict[str, str | None], set[str]]:
    """Over the whole stack: original path -> final path (None: deleted), and the paths the stack adds."""
    final: dict[str, str | None] = {}  # original -> current
    added: set[str] = set()
    for patch in patches:
        for old, new in _changes(patch.read_text(errors="replace")):
            if old is None:
                if new is not None:
                    added.add(new)
                continue
            if old in added:  # a file an earlier patch added: it moves or goes with the stack
                added.discard(old)
                if new is not None:
                    added.add(new)
                continue
            origin = next((o for o, c in final.items() if c == old), old)
            final[origin] = new
    return final, added


def _graph(root: Path, tracked: Sequence[str], roots: Sequence[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(headers by every path suffix they answer to, file -> headers it includes), over the tracked source tree."""
    texts = [p for p in tracked if p.split("/", 1)[0] in roots and configuration.is_text(p)]
    answer: dict[str, set[str]] = {}
    for path in texts:
        if configuration.is_header(path):
            parts = path.lower().split("/")
            for i in range(len(parts)):
                answer.setdefault("/".join(parts[i:]), set()).add(path)
    includes: dict[str, set[str]] = {}
    for path in texts:
        try:
            body = (root / path).read_bytes()
        except OSError:
            continue
        names = {m.decode(errors="replace").lower().removeprefix("./") for m in _INCLUDE.findall(body)}
        includes[path] = {h for n in names for h in answer.get(n, ())}
    return answer, includes


def _consumers(headers: set[str], includes: dict[str, set[str]]) -> set[str]:
    """Every file that includes one of the headers, through any chain of headers."""
    reached, todo = set(headers), list(headers)
    users: dict[str, set[str]] = {}
    for path, held in includes.items():
        for h in held:
            users.setdefault(h, set()).add(path)
    while todo:
        for user in users.get(todo.pop(), ()):
            if user not in reached:
                reached.add(user)
                todo.append(user)
    return reached - headers


def _shape(path: Path) -> list[tuple[object, ...]]:
    """What an object is, names aside: each section's kind, flags and bytes, each relocation's offset and kind."""
    rows: list[tuple[object, ...]] = []
    with path.open("rb") as handle:
        elf = ELFFile(handle)
        for section in elf.iter_sections():
            name = section.name
            if name in _FREE or name.startswith((".debug", ".rel.debug", ".rela.debug")) or not name:
                continue
            if isinstance(section, RelocationSection):
                target = elf.get_section(section["sh_info"]).name
                rows.append(
                    ("rel", target, tuple((r["r_offset"], r["r_info"] & 0xFF) for r in section.iter_relocations()))
                )
            elif section["sh_type"] == "SHT_NOBITS":
                rows.append(("bss", name, section["sh_size"], section["sh_flags"]))
            else:
                rows.append(("sec", name, section["sh_type"], section["sh_flags"], section.data()))
    return rows


def _faults(ran: NativeResult, targets: Sequence[str]) -> tuple[set[str], list[str]]:
    """(targets make has no rule for, the other failures): an exit other than 0 with no named cause is a failure."""
    text = ran.stderr.decode(errors="replace")
    wanted = set(targets)
    absent = {m for m in _NO_RULE.findall(text) if m in wanted}
    bad = [
        line.strip()
        for line in text.splitlines()
        if "***" in line and not ((m := _NO_RULE.search(line)) and m.group(1) in absent)
    ]
    if ran.exit != 0 and not bad and not absent:
        bad = [f"make exit {ran.exit}"]
    return absent, bad


def _build(config: Config, targets: Sequence[str], jobs: int) -> tuple[set[str], list[str]]:
    store.write(config.project.root / "local.mk", build.local_mk(config))
    ran = process.make(config, list(targets), jobs)
    return _faults(ran, targets)


def _targets(versions: Sequence[str], units: Iterable[str]) -> list[str]:
    return [f"build/{v}/{u}.o" for v in versions for u in sorted(units)]


def _narrow(
    config: Config, patches: Sequence[Path], headers: set[str], units: set[str], versions: Sequence[str], holder: Path
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    """({(version, unit)} the header edits of the patches cannot change the object of, what it did). The edit's
    text comes
    from a scratch index with the patches applied, so the work tree is not touched. See `hdrnarrow` for the proof; any
    doubt keeps the unit."""
    texts = process.patched(config.project.root, patches, sorted(headers), holder / "index")
    if texts is None:
        return set(), {"headers": len(headers), "includers": len(units), "skipped": 0, "why": "a patch does not apply"}
    skipped: set[tuple[str, str]] = set()
    ordered = sorted(units)
    stats: dict[str, Any] = {"headers": len(headers), "includers": len(units), "skipped": 0, "why": ""}
    for v in versions:
        keep, got = hdrnarrow.narrow(config, v, sorted(headers), ordered, "build/{v}/{unit}.o", texts.get)
        stats["why"] = stats["why"] or got["why"]
        kept = set(keep)
        skipped |= {(v, u) for u in ordered if u not in kept}
    stats["skipped"] = len(skipped)
    return skipped, stats


def _pair(target: str) -> tuple[str, str]:
    """(version, unit) of a build/<version>/<unit>.o target."""
    _, v, unit = target.split("/", 2)
    return v, unit.removesuffix(".o")


def _restore(config: Config, applied: Sequence[Path], made: Sequence[str]) -> None:
    """Take the patches off again, last first: the proof leaves the tree as it found it (10-24 REV1-ONE-16C: it left
    its patch applied). Intent-to-add entries go first, so the reverse apply removes the files they named."""
    root = config.project.root
    if made:
        process.git(config, "reset", "-q", "--", *made)
    for patch in reversed(applied):
        process.git(config, "apply", "-R", str(patch))
    for path in made:  # an added file a failed reverse left behind
        with contextlib.suppress(OSError):
            (root / path).unlink()


def buildproof_command(config: Config, params: Json) -> Json:
    root = config.project.root
    patches = [Path(p).resolve() for p in params["patch"]]
    versions = list(params.get("version") or config.project.versions)
    jobs = int(params.get("jobs") or 0)
    against = params["against"]
    rules = configuration.load_resource("repo.toml")["cheat"]
    if process.git(config, "status", "--porcelain").stdout.strip():
        raise Refusal(
            Finding("buildproof.tree", "the tree must be clean: the proof builds REV, then applies the patches")
        )
    head = process.git(config, "rev-parse", "HEAD^{tree}").stdout.strip()
    if head != process.git(config, "rev-parse", f"{against}^{{tree}}").stdout.strip():
        raise Refusal(Finding("buildproof.tree", f"the tree is not at {against}: check it out first"))
    for patch in patches:
        process.git(config, "apply", "--check", str(patch))
    final, added = _compose(patches)
    tracked = sorted(process.tracked(root))
    _, includes = _graph(root, tracked, rules["source"])
    touched = set(final)
    units = {p for p in tracked if configuration.is_source(p) and p.split("/", 1)[0] in rules["source"]}
    base_units = (touched & units) | (_consumers({p for p in touched if configuration.is_header(p)}, includes) & units)
    new_of = {u: final.get(u, u) for u in base_units}
    post_units = {n for n in new_of.values() if n} | {a for a in added if configuration.is_source(a)}
    holder = process.state(root) / "buildproof"
    shutil.rmtree(holder, ignore_errors=True)
    skipped: set[tuple[str, str]] = set()
    narrowing: dict[str, Any] = {}
    edited = {p for p in touched if configuration.is_header(p)}
    includers = base_units - touched
    if edited and includers and all(final.get(p) == p for p in edited):
        skipped, narrowing = _narrow(config, patches, edited, includers, versions, holder)
    out: dict[str, Any] = {"against": against, "patches": [str(p) for p in patches], "versions": versions}
    findings: list[Finding] = []

    old_targets = [t for t in _targets(versions, base_units) if _pair(t) not in skipped]
    absent_old, bad = _build(config, old_targets, jobs)
    if bad:
        raise Refusal(
            Finding(
                "buildproof.make", "the baseline build failed: " + "; ".join(bad[:SHOWN]), missing=tuple(bad[:SHOWN])
            )
        )
    kept: dict[str, Path] = {}
    for v in versions:
        for u in base_units:
            if (v, u) in skipped:
                continue
            src = root / f"build/{v}/{u}.o"
            if f"build/{v}/{u}.o" not in absent_old and src.is_file():
                dest = holder / v / f"{new_of[u] or u}.o"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                kept[f"build/{v}/{new_of[u] or u}.o"] = dest

    applied: list[Path] = []
    extracted: list[str] = []
    new_targets: list[str] = []
    absent_new: set[str] = set()
    after: dict[str, Path] = {}
    made = sorted(a for a in added)
    try:
        for patch in patches:
            process.git(config, "apply", str(patch))
            applied.append(patch)
        if made:
            process.git(config, "add", "--intent-to-add", "--", *made)
        for v in versions:
            if any(p in (symbols.path(v), symbols.aliases_path(v)) for p in touched | added):
                extracted.append(v)
                build.extract(config, v)

        new_targets = [t for t in _targets(versions, post_units) if _pair(t) not in skipped]
        absent_new, bad = _build(config, new_targets, jobs)
        # Keep what the change built before the tree goes back: restoring REV's extraction runs make, whose stamp
        # sweep deletes every object assembled from the patched extraction (10-25: a rename of an INCLUDE_ASM function
        # reported its unit's object "gone" in all five versions).
        after: dict[str, Path] = {}
        for target in new_targets:
            if (root / target).is_file():
                dest = holder / "after" / target
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / target, dest)
                after[target] = dest
        if bad:
            findings.append(
                Finding(
                    "buildproof.make",
                    "the build after the change failed: " + "; ".join(bad[:SHOWN]),
                    missing=tuple(bad[:SHOWN]),
                )
            )
    finally:  # every exit path (a raise, a Refusal) leaves the tree clean at REV
        _restore(config, applied, made)
        for v in extracted:  # the asm of those versions followed the patched symbol rows: back to REV's
            build.extract(config, v)
    compared, differ, missing, extra, built_added = 0, [], [], [], []
    for target in new_targets:
        obj = after.get(target, root / "build" / "no-such-object")
        was = kept.get(target)
        unit = target.split("/", 2)[2].removesuffix(".o")
        if unit in added:
            if obj.is_file():
                built_added.append(target)
            continue
        if was is None:
            if obj.is_file() and target not in absent_new:
                extra.append(target)
            continue
        if not obj.is_file():
            missing.append(target)
            continue
        compared += 1
        if _shape(was) != _shape(obj):
            differ.append(target)
    out.update(
        compared=compared,
        identical=compared - len(differ),
        differ=differ[:50],
        missing=missing[:50],
        extra=extra[:50],
        added=len(built_added),
        absent=len(absent_new),
        **({"narrowed": narrowing} if narrowing else {}),
    )
    if differ:
        findings.append(Finding("buildproof.differs", f"{len(differ)} objects differ", missing=tuple(differ[:SHOWN])))
    if missing or extra:
        findings.append(
            Finding(
                "buildproof.set",
                f"{len(missing)} objects built before are gone, {len(extra)} appeared that no patch added",
                missing=tuple((missing + extra)[:SHOWN]),
            )
        )
    if not compared and not findings and not skipped:
        findings.append(Finding("buildproof.empty", "no object could be compared: the change reaches no built unit"))
    if findings:
        raise Refusal(*findings)
    return out
