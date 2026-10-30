"""ROM validation, repository bootstrap, sync and the one-pass setup."""

from __future__ import annotations

import bisect
import io
import itertools
import json
import os
import re
import shutil
import tarfile
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict
from functools import cache
from pathlib import Path
from string import Template
from typing import Any

import rabbitizer
import yaml

from unbake import (
    adapters,
    api,
    build,
    ci,
    effort,
    journal,
    libasm,
    linkfill,
    place,
    pool,
    process,
    report,
    store,
    symbols,
    tree,
    treelock,
)
from unbake import config as configuration
from unbake.contracts import Config, Finding, Json, NativeResult, Refusal, digest, sha1


def _mask(word: int) -> int:
    opcode = rabbitizer.Instruction(word).getOpcodeName()
    if opcode in {"lui", "addiu", "ori", "lw", "sw", "lh", "sh", "lb", "sb", "lwc1", "swc1"}:
        return word & 0xFFFF0000
    return word & 0xFC000000 if opcode == "jal" else word


def _windows(data: bytes, row: Json) -> set[tuple[int, ...]]:
    start, length, window = row["start"], row["length"], row["window"]
    return {
        tuple(_mask(int.from_bytes(chunk[j : j + 4], "big")) for j in range(0, len(chunk), 4))
        for i in range(start, min(start + length, len(data)), window)
        if len(chunk := data[i : min(i + window, start + length)]) == window
    }


def validate_roms(roms: Mapping[str, Path], names_from: str) -> Json:
    with effort.stage("repo.validate_roms"):
        rules = configuration.load_resource("rom.toml")
        data: dict[str, bytes] = {}
        result: dict[str, dict[str, Any]] = {}
        orders: list[Finding] = []
        duplicates: list[Finding] = []
        mismatches: list[Finding] = []
        hashes: dict[str, Any] = {}
        for v, path in sorted(roms.items()):
            try:
                data[v] = path.read_bytes()
            except OSError as error:
                raise Refusal(Finding("init.request", str(error), path=str(path))) from error
            word = int.from_bytes(data[v][:4], "big")
            if word != rules["order"]["z64"]:
                name = next((r["name"] for r in rules["order"]["refuse"] if r["word"] == word), "an unknown order")
                orders.append(
                    Finding(
                        "init.rom_order",
                        f"{v}: first word 0x{word:08X} is {name}; convert the dump to big-endian .z64",
                        path=str(path),
                    )
                )
            rom_sha = sha1(data[v])
            if rom_sha in hashes:
                duplicates.append(
                    Finding(
                        "init.rom_duplicate",
                        f"{hashes[rom_sha]} and {v} have identical ROM bytes",
                        versions=(hashes[rom_sha], v),
                    )
                )
            hashes[rom_sha] = v
            fields = {}
            for row in rules["field"]:
                raw = data[v][row["offset"] : row["offset"] + row["size"]]
                fields[row["name"]] = (
                    raw.decode("ascii", "replace").rstrip("\x00 ") if row["encoding"] == "ascii" else raw.hex()
                )
            result[v] = {"sha1": rom_sha, "size": len(data[v]), "fields": fields, "similarity": 1.0}
        for findings in (orders, duplicates):
            if findings:
                raise Refusal(*findings)
        if names_from not in result:
            raise Refusal(Finding("init.request", f"{names_from} is not one of the roms"))
        reference = _windows(data[names_from], rules["similarity"])
        for v in result:
            if v == names_from:
                continue
            for row in rules["field"]:
                field = row["name"]
                value, ref = result[v]["fields"][field], result[names_from]["fields"][field]
                if row["equal"] and value != ref:
                    mismatches.append(
                        Finding(
                            "init.rom_mismatch",
                            f"{v}: {field} {value!r} differs from {names_from} {ref!r}",
                            versions=(v,),
                        )
                    )
            similarity = len(reference & _windows(data[v], rules["similarity"])) / len(reference) if reference else 0.0
            result[v]["similarity"] = similarity
            if similarity < rules["similarity"]["min"]:
                mismatches.append(
                    Finding(
                        "init.rom_mismatch",
                        f"{v}: similarity {similarity:.6f} to {names_from} is below {rules['similarity']['min']}",
                        versions=(v,),
                    )
                )
        if mismatches:
            raise Refusal(*mismatches)
        return result


def _render(template_name: str, **values: object) -> bytes:
    return Template(configuration.template(template_name)).substitute(values).encode()


def _checked(result: NativeResult, key: str, name: str) -> None:
    if result.exit != 0 or result.signal is not None:
        tail = result.stderr.decode(errors="replace").splitlines()
        raise Refusal(Finding(key, f"{name}: {tail[-1] if tail else f'exit {result.exit}, signal {result.signal}'}"))


HEADER = "include/include_asm.h"


_CODE_SEGMENTS = frozenset({"code", "c", "asm"})  # splat segment types that hold code


def _segment_rows(document: Json) -> list[Json]:
    """The segment rows splat create_config detected, as C1 rows: list rows [start, type, name] and dict rows alike."""
    rows: list[Json] = []
    for item in document.get("segments") or ():
        if isinstance(item, dict):
            start, kind, name, vram = item.get("start"), item.get("type"), item.get("name"), item.get("vram")
        elif len(item) >= 2:
            start, kind, name, vram = item[0], item[1], item[2] if len(item) > 2 else None, None
        else:
            continue  # the closing [end] row
        if start is None or name is None:
            continue
        rows.append({"name": str(name), "rom": int(start), "vram": vram if kind in _CODE_SEGMENTS else None})
    return rows


def _toml_rows(rows: Iterable[Json]) -> str:
    out = []
    for row in rows:
        cells = [f"name = {json.dumps(row['name'])}", f"rom = 0x{row['rom']:X}"]
        cells.append(f"vram = 0x{int(row['vram']):08X}" if row["vram"] is not None else 'kind = "bin"')
        out.append("  { " + ", ".join(cells) + " },")
    return "segments = [\n" + "\n".join(out) + "\n]\n"


def init(params: Json) -> Json:
    with effort.stage("repo.init"):
        root, roms = Path(params["dir"]).absolute(), {}
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise Refusal(Finding("init.request", f"{root} is not an empty directory", path=str(root)))
        for item in params["rom"]:
            v, sep, path = item.partition("=")
            if not sep or not v or not path or v in roms or not Path(path).is_file():
                raise Refusal(Finding("init.request", f"--rom {item} is not a unique VERSION=PATH naming a file"))
            roms[v] = Path(path)
        toolchains = configuration.toolchains()
        if params["names_from"] not in roms or params["toolchain"] not in toolchains:
            raise Refusal(Finding("init.request", "names_from must name a ROM and toolchain must name a toolchain"))
        host_path = Path(params["host"])
        host = configuration.load_host(host_path)
        versions, rules = validate_roms(roms, params["names_from"]), configuration.load_resource("repo.toml")
        created, blocks = [], []

        def write(path: str, value: bytes) -> None:
            store.write(root / path, value)
            created.append(path)

        write(HEADER, configuration.template("include_asm.h.in").encode())
        for v, source in sorted(roms.items()):
            path = f"roms/baserom.{v}.z64"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, root / path)
            created.append(path)
        for v in sorted(roms):
            rom = f"roms/baserom.{v}.z64"
            before = set(root.glob("*.yaml"))
            result = process.run(
                "splat create_config", [str(host.tools["splat"]), "create_config", rom], root, tmp=process.scratch(root)
            )
            _checked(result, "setup.splat", "splat create_config")
            made = set(root.glob("*.yaml")) - before
            if len(made) != 1:
                raise Refusal(Finding("setup.splat", f"expected one new yaml, found {len(made)}", versions=(v,)))
            detected = made.pop()
            rows = _segment_rows(yaml.safe_load(detected.read_bytes()))
            detected.unlink()  # splat's scratch output: the generated versions/<v>/splat.yaml replaces it
            write(f"versions/{v}/symbol_addrs.txt", b"")
            blocks.append(
                f'\n[version.{v}]\nsha1 = "{versions[v]["sha1"]}"\nsize = {versions[v]["size"]}\n{_toml_rows(rows)}'
            )
        values = {key: json.dumps(params[key]) for key in ("name", "title", "names_from", "toolchain")}
        write("config.toml", _render("config.toml", **values, version_blocks="".join(blocks)))
        write(
            "README.md",
            _render("README.md.in", title=params["title"], begin=rules["readme"]["begin"], end=rules["readme"]["end"]),
        )
        roms_list = "\n".join(f"- `roms/baserom.{v}.z64`" for v in sorted(roms))
        write("CONTRIBUTING.md", _render("CONTRIBUTING.md.in", title=params["title"], roms=roms_list))
        _checked(
            process.run("git init repository", [str(host.tools["git"]), "init"], root, tmp=process.scratch(root)),
            "native.exit",
            "git init",
        )
        synced = sync(configuration.load(root, host_path), {"check": False})
        return {"dir": str(root), "created": sorted({*created, *synced["written"]}), "versions": versions}


def _macro(version: str) -> str:
    return "VERSION_" + version.upper().replace("-", "_")


def _segment_of(config: Config, version: str, rom: int) -> str | None:
    rows = sorted(config.project.version[version].segments, key=lambda s: s.rom)
    return next(
        (
            s.name
            for s, after in zip(rows, [*rows[1:], None], strict=True)
            if rom >= s.rom and (after is None or rom < after.rom)
        ),
        None,
    )


def _extension(modules: Iterable[Any]) -> str:
    """The extension most source files of the tree use: where a new stand-in file is written."""
    found = Counter(Path(m.path).suffix for m in modules if not configuration.is_assembly(m.path))
    return found.most_common(1)[0][0] if found else ".c"


_MACRO = {"func": "INCLUDE_ASM", ".rodata": "INCLUDE_RODATA"}  # by kind, then by section; INCLUDE_DATA otherwise
_DATA_SECTIONS = frozenset({".data", ".rodata", ".bss"})
_CONDITIONAL = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|endif)\b")


def _stand_in(macro: str, segment: str, name: str) -> str:
    return f'{macro}("{segment}", {name});'


def _line_of(name: str) -> re.Pattern[str]:
    return re.compile(rf'^[ \t]*INCLUDE_(?:ASM|DATA|RODATA)\(\s*"[^"]*"\s*,\s*{re.escape(name)}\s*\)\s*;[ \t]*$')


def _definition_end(lines: Sequence[str], name: str) -> int | None:
    """The index of the line that closes the C definition of `name` (a `}` in column 0), if the file has one."""
    start = re.compile(rf"^[A-Za-z_][^;{{}}#]*\b{re.escape(name)}\s*\(")
    first = next((i for i, text in enumerate(lines) if start.match(text) and not text.startswith("INCLUDE_")), None)
    if first is None:
        return None
    return next((j for j in range(first, len(lines)) if lines[j].rstrip() == "}"), None)


def _depths(lines: Sequence[str]) -> list[int]:
    """The `#if` nesting depth before each line."""
    out, depth = [], 0
    for line in lines:
        out.append(depth)
        found = _CONDITIONAL.match(line)
        if found:
            depth = max(0, depth + (-1 if found[1] == "endif" else 1))
    return out


def _after(lines: Sequence[str], at: int) -> int:
    """The first index after line `at` at top level: past the `#if` block that holds the line, if any."""
    depths = [*_depths(lines), 0]
    return next(j for j in range(at + 1, len(lines) + 1) if depths[j] == 0)


def _before(lines: Sequence[str], at: int) -> int:
    """The index of line `at` at top level: the start of the `#if` block that holds the line, if any."""
    depths = _depths(lines)
    return next(j for j in range(at, -1, -1) if depths[j] == 0)


def _head(lines: Sequence[str]) -> int:
    """The first index after the file's leading `#include` lines and blank lines."""
    at = 0
    while at < len(lines) and (not lines[at].strip() or lines[at].lstrip().startswith("#include")):
        at += 1
    return at


def _array_of(lines: Sequence[str], name: str) -> tuple[int, int] | None:
    """(first, last) line index of the word array `name` as the extractor writes it (`const unsigned int NAME[N] = {`
    to the closing `};`; without `const` for a .data stand-in), if the file defines it so."""
    head = re.compile(rf"^\s*(?:const\s+)?(?:unsigned\s+int|u32|s32|int)\s+{re.escape(name)}\s*\[\s*\d*\s*\]\s*=\s*\{{")
    first = next((i for i, line in enumerate(lines) if head.match(line)), None)
    if first is None:
        return None
    last = next((j for j in range(first, len(lines)) if lines[j].rstrip().endswith("};")), None)
    return None if last is None else (first, last)


def _drop(lines: list[str], names: Iterable[str]) -> None:
    """Remove the stand-in of each name (its INCLUDE line, or the word array the extractor wrote), and an `#if` around
    it that holds nothing else."""
    for name in names:
        pattern = _line_of(name)
        at = next((i for i, line in enumerate(lines) if pattern.match(line)), None)
        end = at
        if at is None:
            found = _array_of(lines, name)
            if found is None:
                continue
            at, end = found
        assert end is not None
        opened = _CONDITIONAL.match(lines[at - 1]) if at > 0 else None
        closed = _CONDITIONAL.match(lines[end + 1]) if end + 1 < len(lines) else None
        guarded = bool(opened and opened[1] != "endif" and closed and closed[1] == "endif")
        del lines[at - 1 if guarded else at : end + 2 if guarded else end + 1]


def _exclude(lines: list[str], name: str, versions: Sequence[str]) -> bool:
    """Keep the stand-in of `name` out of `versions` only (a literals row supersedes it there, other versions still
    own it): its INCLUDE line or word array gets an `#if !(defined(...))` of its own. False when it is not found or
    already fenced exactly so (a second sync changes nothing); a fence naming other versions is rewritten in place."""
    pattern = _line_of(name)
    at = next((i for i, line in enumerate(lines) if pattern.match(line)), None)
    end = at
    if at is None:
        found = _array_of(lines, name)
        if found is None:
            return False
        at, end = found
    assert end is not None
    spelled = " || ".join(f"defined({_macro(v)})" for v in versions)
    wanted = f"#if !({spelled})"
    if at > 0 and lines[at - 1].startswith("#if !(") and end + 1 < len(lines) and lines[end + 1] == "#endif":
        # already fenced by an earlier sync: a second run must not stack another wrapper (it was not idempotent)
        if lines[at - 1] == wanted:
            return False
        lines[at - 1] = wanted
        return True
    lines[end + 1 : end + 1] = ["#endif"]
    lines[at:at] = [wanted]
    return True


_WORDS = re.compile(r"0[xX][0-9A-Fa-f]+")


def _shrink(lines: list[str], name: str, words: int) -> bool:
    """Cut the word array `name` to its first `words` words, written as the extractor writes one."""
    found = _array_of(lines, name)
    if found is None:
        return False
    first, last = found
    values = _WORDS.findall("\n".join(lines[first + 1 : last + 1]))
    if words >= len(values) or words < 1:
        return False
    kept = values[:words]
    body = ["    " + " ".join(f"0x{int(v, 16):08X}," for v in kept[i : i + 8]) for i in range(0, len(kept), 8)]
    head = re.sub(r"\[\s*\d*\s*\]", f"[{words}]", lines[first], count=1)
    lines[first : last + 1] = [head, *body, "};"]
    return True


def _hoist(lines: list[str], name: str, function: str) -> bool:
    """Move the word array `name`, with an `#if` around it that holds nothing else, in front of the definition of
    `function` (and of any `#if` that definition sits in): the object's .rodata holds a module's data in source order
    and the function's own literals (a jump table) where it is defined, so an array the ROM has before them must be
    defined before it. False when the array is already first or either is not found."""
    found = _array_of(lines, name)
    definition = re.compile(rf"^[A-Za-z_][\w \t\*]*\b{re.escape(function)}\s*\([^;]*$")
    at = next((i for i, line in enumerate(lines) if definition.match(line)), None)
    if found is None or at is None or found[0] < at:
        return False
    first, last = found
    opened = _CONDITIONAL.match(lines[first - 1]) if first > 0 else None
    closed = _CONDITIONAL.match(lines[last + 1]) if last + 1 < len(lines) else None
    guarded = bool(opened and opened[1] != "endif" and closed and closed[1] == "endif")
    low, high = (first - 1, last + 2) if guarded else (first, last + 1)
    open_ifs: list[int] = []
    for i, line in enumerate(lines[:at]):
        if (kind := _CONDITIONAL.match(line)) is not None:
            if kind[1] != "endif":
                open_ifs.append(i)
            elif open_ifs:
                open_ifs.pop()
    at = open_ifs[0] if open_ifs else at
    block = lines[low:high]
    del lines[low:high]
    lines[at:at] = block
    return True


def _held_back(lines: Sequence[str]) -> bool:
    """A file with no code, data or stand-in left: only blank lines, comments and preprocessor lines."""
    return all(not line.strip() or line.lstrip().startswith(("#", "//", "/*", "*")) for line in lines)


def _narrowed(config: Config, current: Any, lines: Callable[[str], list[str]]) -> set[str]:
    """Shrink each word array that runs into the read-only bytes a `literals_` row hands to a function, to the words
    before the row: the array names the bytes up to the next name, and the row now names the rest. The paths touched.
    Versions that would cut one definition differently leave it as it is."""
    cuts: dict[tuple[str, str], set[int]] = {}
    owners: dict[tuple[str, str], set[str]] = {}
    for v in config.project.versions:
        facts = api.symbols(config, v)
        starts = sorted(
            (f[1], n[len(tree.LITERALS) :])
            for n, f in facts.items()
            if n.startswith(tree.LITERALS) and f[1] is not None
        )
        if not starts:
            continue
        for module in current.modules:
            if configuration.is_assembly(module.path):
                continue
            for name in module.owns.get(v, ()):
                rom = facts[name][1] if name in facts else None
                if (
                    rom is None
                    or name.startswith(tree.LITERALS)
                    or not module.sections.get(v, {}).get(name, "").startswith(".rodata")
                ):
                    continue
                after = next((s for s in starts if s[0] > rom), None)
                if after is not None and (after[0] - rom) % 4 == 0:
                    cuts.setdefault((module.path, name), set()).add((after[0] - rom) // 4)
                    owners.setdefault((module.path, name), set()).add(after[1])
    touched: set[str] = set()
    for (path, name), counts in sorted(cuts.items()):
        if len(counts) != 1:
            continue
        shrunk = _shrink(lines(path), name, next(iter(counts)))
        functions = owners[(path, name)]
        if (len(functions) == 1 and _hoist(lines(path), name, next(iter(functions)))) or shrunk:
            touched.add(path)
    return touched


def stand_ins(config: Config, current: Any) -> dict[str, bytes | None]:
    """The files to write so that a name has a stand-in exactly when no file defines it in C: a stand-in a C body
    supersedes is removed, and every name of tree.unowned(v) gets a line, under the `#if` of the versions it is
    unowned in when that is not every version (another version may know the name only as a second name of owned
    bytes, or not at all: it has no extracted file of that name), in the file owning the previous address of its
    segment, right after that name's own stand-in (else at the end); when nothing precedes it, in the file owning the
    next address, right before that name's stand-in (else after the leading includes); a new src/<segment>/<ADDR>.<ext>
    only when the segment has no owner at all. A library routine (foreign code, see libasm.homes) goes at the end
    of its library file instead, never into a game file. A data name takes the section of the data next to it: its
    bytes were part of that name's until now."""
    root, ext = config.project.root, _extension(current.modules)
    texts: dict[str, list[str]] = {}
    touched: set[str] = set()
    data_dropped: set[str] = set()  # files that lost a data stand-in: one left with nothing goes
    placed: set[str] = set()

    def lines(path: str) -> list[str]:
        if path not in texts:
            file = root / path
            texts[path] = file.read_text().splitlines() if file.is_file() else []
        return texts[path]

    for module in current.modules:
        headless = "include_asm.h" not in "\n".join(lines(module.path))
        if headless and not configuration.is_assembly(module.path) and any(module.stand_ins.values()):
            touched.add(module.path)  # its stand-ins were read with the header, which the write below puts on top
        every = {n for names in module.stale.values() for n in names}
        kept = {  # data stand-ins stale in some versions only: another version still owns the bytes through them
            n
            for v, names in module.owns.items()
            for n in names
            if n in every and n not in module.stale.get(v, ()) and module.sections.get(v, {}).get(n, ".text") != ".text"
        }
        if every - kept:
            _drop(lines(module.path), sorted(every - kept))
            touched.add(module.path)
            if any(
                n in api.symbols(config, v) and api.symbols(config, v)[n][3] != "func"
                for v, names in module.stale.items()
                for n in names
                if n not in kept
            ):
                data_dropped.add(module.path)
        for name in sorted(kept):
            # live only where this file owns the name and no literals row supersedes it: a version that knows the bytes
            # under another name (or not at all) must not emit the pair a second time (the link overlapped in .data)
            gone = [
                v
                for v in config.project.versions
                if name in module.stale.get(v, ()) or name not in module.owns.get(v, ())
            ]
            if _exclude(lines(module.path), name, gone):
                touched.add(module.path)
    touched |= _narrowed(config, current, lines)
    library = libasm.homes(config, current)  # library routines kept out of game files
    sections = {(v, n): s for m in current.modules for v, names in m.sections.items() for n, s in names.items()}
    missing = {v: set(current.unowned(v)) for v in config.project.versions}
    for v in config.project.versions:
        unowned = current.unowned(v)
        if not unowned:
            continue
        facts = api.symbols(config, v)
        owners: dict[str, list[tuple[int, str, str]]] = {}
        for module in current.modules:
            if configuration.is_assembly(module.path):
                continue
            for owned in module.owns.get(v, ()):
                if owned in facts and facts[owned][1] is not None and not owned.startswith(tree.LITERALS):
                    segment = _segment_of(config, v, facts[owned][1] or 0)
                    if segment is not None:
                        owners.setdefault(segment, []).append((facts[owned][1] or 0, module.path, owned))
        for rows in owners.values():
            rows.sort()
        for name in sorted(
            (n for n in unowned if facts.get(n, (0, None))[1] is not None), key=lambda n: facts[n][1] or 0
        ):
            if name in placed:
                continue
            vram, rom, _size, kind = facts[name]
            rom = rom or 0
            segment = _segment_of(config, v, rom)
            if segment is None:
                raise Refusal(Finding("sync.segment", f"{v}: {name} at rom 0x{rom:X} is in no segment", versions=(v,)))
            rows = owners.setdefault(segment, [])
            elsewhere = next((current.owner(o, name) for o in config.project.versions if current.owner(o, name)), None)
            before = [(path, anchor) for at, path, anchor in rows if at < rom]
            after = [(path, anchor) for at, path, anchor in rows if at > rom]
            new = (f"src/{segment}/{vram:08X}{ext}", None)
            ahead = not elsewhere and not before and bool(after)
            path, anchor = (elsewhere, None) if elsewhere else before[-1] if before else after[0] if after else new
            if not elsewhere and name in library:
                path, anchor, ahead = library[name].path, None, False
            before_it = sections.get((v, anchor or ""), ".data")
            section = ".text" if kind == "func" else before_it if before_it in _DATA_SECTIONS else ".data"
            line = _stand_in(_MACRO.get(kind, _MACRO.get(section, "INCLUDE_DATA")), segment, name)
            held = [v] if elsewhere else [o for o in config.project.versions if name in missing[o]]
            guard = " || ".join(_macro(o) for o in held) if len(held) < len(config.project.versions) else None
            block = [f"#if {guard}", line, "#endif"] if guard else [line]  # only where the name lacks a file
            if not elsewhere:
                placed.add(name)
            body = lines(path)
            pattern = _line_of(anchor) if anchor else None
            module = next((m for m in current.modules if m.path == path), None)
            c_body = bool(
                anchor and module is not None and module.stand_ins.get(v) and anchor not in module.stand_ins[v]
            )
            at = _definition_end(body, anchor) if c_body and anchor else None  # C here: an `#if` stand-in is elsewhere
            if at is None:
                at = next((i for i, text in enumerate(body) if pattern and pattern.match(text)), None)
            if ahead:
                where = _before(body, at) if at is not None else _head(body)
            else:
                where = _after(body, at) if at is not None else len(body)
            body[where:where] = block
            touched.add(path)
            rows.append((rom, path, name))
            rows.sort()
            sections[v, name] = section
    writes: dict[str, bytes | None] = {}
    for path in sorted(touched):
        if path in data_dropped and (root / path).is_file() and _held_back(texts[path]):
            writes[path] = None  # every stand-in it held was superseded: nothing left to assemble
            continue
        joined = "\n".join(texts[path])
        if "include_asm.h" not in joined and "INCLUDE_" in joined:
            joined = '#include "include_asm.h"\n\n' + joined
        writes[path] = (joined.rstrip("\n") + "\n").encode()
    return writes


def _lines(rows: Iterable[str]) -> bytes:
    return ("\n".join(rows) + "\n").encode()


def _unignored(ignore: Sequence[str], command: str) -> list[str]:
    """The ignore list with the one path `command` kept: an ignored folder above it lists its entries (`/tools/*`)
    instead of itself, and each folder between is let through the same way, so git sees the command and nothing else."""
    parts = command.split("/")
    above = ["/" + "/".join(parts[:n]) + "/" for n in range(1, len(parts))]
    first = next((n for n, folder in enumerate(above) if folder in ignore), None)
    if first is None:
        return list(ignore)  # no ignored folder above the command: nothing to let through
    kept = [f"{row}*" if row == above[first] else row for row in ignore]
    for folder in above[first + 1 :]:
        kept += [f"!{folder}", f"{folder}*"]
    return [*kept, f"!/{command}"]


def scaffold(declared: Mapping[str, Any] | None = None) -> dict[str, bytes]:
    """The repository files every game repository shares, from repo.toml: what git ignores and merges, the splat pins
    `make extract` installs, and the licence text. `declared` is the project's config.toml [repo]: each `generated` glob
    is written `<glob> generated=<regenerate>` (the paths a host regenerates by that command, never merges), and the
    command's own path stays out of the ignore list."""
    rules, declared = configuration.load_resource("repo.toml"), declared or {}
    lines = [f"{glob} linguist-language=C" for glob in rules["linguist"]]
    lines += [f"{glob} merge=union" for glob in rules["union"]]
    ignore = rules["ignore"]
    if "regenerate" in declared:
        lines += [f"{glob} generated={declared['regenerate']}" for glob in declared["generated"]]
        ignore = _unignored(ignore, declared["regenerate"])
    return {
        ".gitignore": _lines(ignore),
        ".gitattributes": _lines(lines),
        "requirements.txt": _lines(rules["splat_pip"]),
        "LICENSE": configuration.template("LICENSE.in").encode(),
    }


def _generated(config: Config, current: Any) -> dict[str, bytes]:
    project = config.project
    wanted = {
        "Makefile": build.makefile(config),
        f"{project.name}.ld": build.linker(config, current),
        **{f"versions/{v}/splat.yaml": build.splat_yaml(config, v, current) for v in project.versions},
        **{f"versions/{v}/text_sizes.mk": build.text_sizes(config, current, v) for v in project.versions},
        **ci.files(),
        **scaffold(configuration.repo(project)),
    }
    if not (project.root / HEADER).is_file():
        wanted[HEADER] = configuration.template("include_asm.h.in").encode()
    return wanted


def generated(config: Config, current: Any) -> dict[str, bytes]:
    """The files sync generates from the tree (Makefile, link script, splat.yaml, text sizes, CI, scaffold), by path."""
    return _generated(config, current)


def jump_table_of(config: Config, version: str, rom: int) -> str | None:
    """The function whose jump table the read-only piece at `rom` is, word for word up to the next row, else None."""
    mask = configuration.load_resource("rom.toml").get("text_mask", 0xFFFFFFFF)
    try:
        image = configuration.rom(config.project, version).read_bytes()
    except (OSError, Refusal):
        return None
    table = api.symbols(config, version)
    roms = sorted({f[1] for f in table.values() if f[1] is not None})
    ends = dict(itertools.pairwise(roms))
    if rom not in ends or ends[rom] - rom > _TABLE_MAX:
        return None
    code = sorted(
        (f[0] & mask, (f[0] & mask) + size, n)
        for n, f in table.items()
        if f[3] == "func" and f[1] is not None and (size := f[2] or (ends.get(f[1], f[1]) - f[1]))
    )
    data = image[rom : ends[rom]]
    return _jump_target([int.from_bytes(data[i : i + 4], "big") for i in range(0, len(data) - 3, 4)], code)


def _sync_placing(config: Config, params: Json, refresh: bool = False) -> Json:
    """`sync` after the rows that place the read-only data of the functions named by `place` are added to each
    version's aliases.txt (see place.py): the plan is refused whole before anything is written, and a sync that refuses
    afterwards puts the aliases back. With check: the rows it would add, nothing written."""
    root = config.project.root
    found = place.plan(config, tree.capture(config), list(params["place"]))
    wanted = {*(p.file for p in found), *(p.after for p in found if p.after), *(p.split_from for p in found if p.split)}
    wanted |= {f for p in found if p.junk for f in p.junk_files}
    texts = {path: (_read(root / path) or b"").decode() for path in wanted}
    writes = {**place.rows(found, texts), **place.sources(found, texts)}
    shown = [
        {
            "version": p.version,
            "function": p.function,
            "row": p.row,
            "file": p.file,
            "note": p.note + (f"; {', '.join(p.split)} move from {p.split_from} to {p.after}" if p.split else ""),
            **({"removed": [f"{n} (rom 0x{r:X})" for n, r in p.junk]} if p.junk else {}),
        }
        for p in found
    ]
    if params.get("check"):
        return {"written": [], "stand_ins": {}, "placed": shown}
    with store.journal() as undone:
        try:
            for path, data in writes.items():
                store.write(root / path, data)
            result = sync(config, {k: v for k, v in params.items() if k != "place"}, refresh)
        except BaseException:
            store.undo(undone)
            raise
    out = {**result, "placed": shown, "written": sorted({*result["written"], *writes})}
    if any(p.leftover for p in found):
        out["next"] = "run `make extract`: the leftover bytes have a new stand-in whose assembly is not extracted yet"
    return out


_PASSES = 8  # most passes one sync makes before it stops at a state it keeps rewriting


def sync_command(config: Config, params: Json) -> Json:
    """`unbake sync`: `sync` that brings the check verdict up to date before it writes what depends on it."""
    return sync(config, params, refresh=True)


def sync(config: Config, params: Json, refresh: bool = False) -> Json:
    """Write every generated file and the stand-ins for unowned names; with check, write nothing and refuse naming the
    files that would change. The README block, its image, progress.json and the docs maps follow the check verdict:
    with none for this tree (or one older than its build inputs) they are left as they are and a note says to check
    first, unless `refresh` (the `unbake sync` command; a caller inside another transaction does not build): then the
    check runs here, on the tree the generated files just left, so the outputs are never written from a stale verdict
    and are never skipped for want of one (a build that fails refuses `setup.make`). With settle, write and then
    refuse as check would unless a second sync would write nothing: when it wrote generated or stand-in files the
    check runs again in this process on the tree it left; the README files it writes are not read by the capture, so
    their fixed point is the README block's own (one `sync --settle` stands for `sync` then `sync --check`)."""
    if params.get("place"):
        return _sync_placing(config, params, refresh)
    if params.get("makefile") and not params.get("check"):
        # the Makefile and local.mk are one contract (the recipe names the variables local.mk sets): regenerate both,
        # local.mk first so a tree whose content refuses the sync still has the machine half of the current tool
        store.write(config.project.root / "local.mk", build.local_mk(config))
    result, current = _sync(config, params)
    root = config.project.root
    for _ in range(0 if params.get("check") else _PASSES):
        # a pass that deleted a file (an owned unit dropped one version at a time) moves what the next pass
        # reads: sync runs to its own fixed point, so one sync settles every version
        if not result["written"] or not (
            params.get("settle") or any(not (root / w).exists() for w in result["written"])
        ):
            break
        again, current = _sync(config, params)
        result = {
            "written": sorted({*result["written"], *again["written"]}),
            "stand_ins": {**result["stand_ins"], **again["stand_ins"]},
        }
        if not again["written"]:
            break
    if params.get("settle") and result["written"]:
        _, current = _sync(config, {"check": True})
    if not (config.project.root / "README.md").is_file():
        return result
    hold = report.publish_hold(config, current)
    if hold and hold.key == "sync.unchecked" and refresh and not params.get("check"):
        with effort.stage("repo.sync.check"):
            build.make_check(config, current)
        hold = report.publish_hold(config, current)
    if hold:
        return {**result, "notes": [asdict(hold)]}
    settle = bool(params.get("settle")) and not params.get("check")
    wanted = report.files(config, current, settle=True) if settle else report.files(config, current)
    if params.get("check"):
        differ = sorted(p for p, v in wanted.items() if _read(config.project.root / p) != v)
        if differ:
            raise Refusal(
                Finding(
                    "sync.changed",
                    f"{len(differ)} generated or stand-in files differ from `unbake sync`",
                    missing=tuple(differ),
                    action="run unbake sync and commit the result",
                )
            )
        return result
    for path, content in wanted.items():
        if store.write(config.project.root / path, content):
            result["written"].append(path)
    return result


def _read(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


_TABLE_MAX = place.TABLE_MAX


_jump_target = place.jump_target


def _unplaced_tables(config: Config, current: Any) -> list[Finding]:
    """The jump tables of C functions whose file links no read-only data. A function's table is its file's compiled
    .rodata once its body is C, and only a name the file owns in .rodata places that (a `literals_<function>` fact,
    `tree.LITERALS`); when the table's old stand-in goes and no such row comes, the table's name is owned by no file,
    so the link script and splat.yaml leave the object's .rodata out and the ROM comes out short by the table. Each
    finding names the row that places it. Needs the ROMs: without them there is no check."""
    mask = configuration.load_resource("rom.toml").get("text_mask", 0xFFFFFFFF)
    findings: list[Finding] = []
    seen: set[tuple[str, int]] = set()
    for v in config.project.versions:
        try:
            image = configuration.rom(config.project, v).read_bytes()
        except (OSError, Refusal):
            continue  # no ROM (a CI check): nothing to read the tables from
        table = api.symbols(config, v)
        c_code: dict[str, str] = {}
        with_rodata: set[str] = set()
        for m in current.modules:
            if configuration.is_assembly(m.path):
                continue
            for n in m.owns.get(v, ()):
                section = m.sections.get(v, {}).get(n, ".text")
                if section.startswith(".rodata"):
                    with_rodata.add(m.path)
                elif section == ".text" and n not in m.stand_ins.get(v, ()):
                    c_code[n] = m.path
        roms = sorted({f[1] for f in table.values() if f[1] is not None})

        def after(rom: int, roms: Sequence[int] = roms) -> int | None:
            at = bisect.bisect_right(roms, rom)
            return roms[at] if at < len(roms) else None

        spans = sorted(
            (f[0] & mask, (f[0] & mask) + size, n)  # code addresses as the ROM spells them (text_mask)
            for n, f in table.items()
            if f[3] == "func" and f[1] is not None and (size := f[2] or ((after(f[1]) or f[1]) - f[1]))
        )
        for n, fact in sorted(table.items(), key=lambda kv: (kv[1][1] or 0, kv[0])):
            rom = fact[1]
            if rom is None or fact[3] in ("func", "label") or n.startswith(tree.LITERALS):
                continue
            end = after(rom)
            if end is None or not 8 <= end - rom <= _TABLE_MAX or (end - rom) % 4 or current.owner(v, n) is not None:
                continue
            data = image[rom:end]
            function = _jump_target([int.from_bytes(data[i : i + 4], "big") for i in range(0, len(data), 4)], spans)
            path = c_code.get(function or "")
            if function is None or path is None or path in with_rodata or (v, rom) in seen:
                continue  # a file that links read-only data places the table with it; one finding per table
            seen.add((v, rom))
            findings.append(
                Finding(
                    "sync.literals",
                    f"{v}: {n} at 0x{fact[0]:08X} is the jump table of {function}, which {path} defines in C, but "
                    f"{path} owns no read-only data: the link would leave its compiled .rodata out of the ROM",
                    path=f"versions/{v}/aliases.txt",
                    unit=function,
                    versions=(v,),
                    action=f"add `{tree.LITERALS}{function} = 0x{fact[0]:08X}; // rom:0x{rom:X}` to "
                    f"versions/{v}/aliases.txt",
                )
            )
    return findings


def _runs(words: Sequence[int], code: Sequence[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """The (first index, count, function) of each run of two or more consecutive words inside one function: a jump
    table's cases."""
    found: list[tuple[int, int, str]] = []
    for i, word in enumerate(words):
        function = _jump_target([word, word], code)
        if function is not None and found and found[-1][2] == function and sum(found[-1][:2]) == i:
            found[-1] = (found[-1][0], found[-1][1] + 1, function)
        elif function is not None:
            found.append((i, 1, function))
    return [r for r in found if r[1] >= 2]


def _swallowed_tables(config: Config, current: Any) -> list[Finding]:
    """A `literals_<function>` row places its file's compiled .rodata, and splat sizes that piece up to the next row:
    when the bytes before the next row also hold the jump table of a function the same object does not compile (one
    of another file, or behind a stand-in), the object's .rodata ends before that table and the ROM comes out short.
    Each finding names the row that splits the table off. Needs the ROMs: without them there is no check."""
    mask = configuration.load_resource("rom.toml").get("text_mask", 0xFFFFFFFF)
    findings: list[Finding] = []
    for v in config.project.versions:
        try:
            image = configuration.rom(config.project, v).read_bytes()
        except (OSError, Refusal):
            continue
        table = api.symbols(config, v)
        files = {n: m.path for m in current.modules for n in m.owns.get(v, ())}  # in C or behind a stand-in
        compiled = {
            n: m.path
            for m in current.modules
            if not configuration.is_assembly(m.path)
            for n in m.owns.get(v, ())
            if m.sections.get(v, {}).get(n, ".text") == ".text" and n not in m.stand_ins.get(v, ())
        }
        # a fill row cuts its own piece, so it ends the row before it like a symbol would
        roms = sorted(
            {f[1] for f in table.values() if f[1] is not None} | {f.rom for f in linkfill.read(config.project.root, v)}
        )
        ends = dict(itertools.pairwise(roms))
        code = sorted(
            (f[0] & mask, (f[0] & mask) + size, n)
            for n, f in table.items()
            if f[3] == "func" and f[1] is not None and (size := f[2] or (ends.get(f[1], f[1]) - f[1]))
        )
        for name, fact in sorted(table.items()):
            rom = fact[1]
            if not name.startswith(tree.LITERALS) or rom is None or rom not in ends:
                continue
            owner = files.get(name[len(tree.LITERALS) :])  # one object: its functions' tables are its .rodata
            data = image[rom : ends[rom]]
            words = [int.from_bytes(data[i : i + 4], "big") for i in range(0, len(data) - 3, 4)]
            for first, count, function in _runs(words, code):
                if function == name[len(tree.LITERALS) :] or function not in files or compiled.get(function) == owner:
                    continue  # its own table, a label no file holds, or a table the same object compiles
                vram, at = fact[0] + 4 * first, rom + 4 * first
                findings.append(
                    Finding(
                        "sync.literals-span",
                        f"{v}: {name} at 0x{fact[0]:08X} runs 0x{ends[rom] - rom:X} bytes to the next row, but from "
                        f"0x{vram:08X} on it holds the {count}-entry jump table of {function}, which "
                        f"{files.get(function, 'no file')} holds: the piece would end before it and leave the "
                        f"ROM short by 0x{ends[rom] - at:X} bytes",
                        path=f"versions/{v}/symbol_addrs.txt",
                        unit=name,
                        versions=(v,),
                        action=f"add `jtbl_{vram:08X} = 0x{vram:08X}; // rom:0x{at:X}` to versions/{v}/symbol_addrs.txt"
                        f" (or `{tree.LITERALS}{function}` to aliases.txt when its file compiles it)",
                    )
                )
                break  # one finding per row: the split moves the next row
    return findings


def _sync(config: Config, params: Json) -> tuple[Json, Any]:
    """`sync`, and the tree it left (the one capture a setup shares with the report that follows)."""
    with effort.stage("repo.sync"):
        root = config.project.root
        current = tree.capture(config)
        assembly = libasm.plan(config, current)
        if assembly:
            current = tree.capture(config, assembly)
        with effort.stage("repo.stand_ins"):
            appended = {**assembly, **stand_ins(config, current)}
            if appended:
                # This capture sees the proposed stand-ins, a different input from the on-disk tree above.
                current = tree.capture(config, appended)
        if unplaced := [*_unplaced_tables(config, current), *_swallowed_tables(config, current)]:
            raise Refusal(*unplaced)
        changed = {
            path: value
            for path, value in {**appended, **_generated(config, current)}.items()
            # an existing Makefile is the project's own until `sync --makefile` regenerates it: a tool newer than the
            # tree's main would otherwise rewrite it into something main does not carry (every lane then restored it)
            if not (path == "Makefile" and (root / path).is_file() and not params.get("makefile"))
            if (root / path).is_file() != (value is not None)
            or (value is not None and (root / path).read_bytes() != value)
        }
        if params.get("check"):
            if changed:
                raise Refusal(
                    Finding(
                        "sync.changed",
                        f"{len(changed)} generated or stand-in files differ from `unbake sync`",
                        missing=tuple(sorted(changed)),
                        action="run unbake sync and commit the result",
                    )
                )
            return {"written": [], "stand_ins": {}}, current
        for path, value in changed.items():
            if value is None:
                (root / path).unlink()
                effort.forget("pin", "includes", "closure", "listing")
            else:
                if path == "Makefile":  # a rule make cannot see changing: its objects go before the next build
                    build.drop_reflagged(config, value)
                store.write(root / path, value)
        effort.count("sync", not changed)
        return {"written": sorted(changed), "stand_ins": {p: [] for p in sorted(appended)}}, current


def symbols_command(config: Config, params: Json) -> Json:
    if params.get("missing") or params.get("write"):
        from unbake import missing

        return missing.command(config, {**params, "write": params.get("write") and params.get("missing")})
    wanted = [params["version"]] if params.get("version") else list(config.project.versions)
    return {"symbols": {v: {n: list(row) for n, row in api.symbols(config, v).items()} for v in wanted}}


def _extract_stamp(config: Config, v: str) -> Path:
    return config.project.root / "asm" / v / ".unbake-extract"


def _extracted_key(config: Config, v: str) -> str | None:
    """The key of the extraction this tree's asm/<v>/ holds (written beside it), None when there is none."""
    stamp = _extract_stamp(config, v)
    return stamp.read_text() if stamp.is_file() else None


def _facts_hash(root: Path, v: str) -> str:
    """The hash of what an extraction reads (symbol_addrs.txt then splat.yaml, missing ones skipped): the same bytes the
    Makefile hashes with `cat | sha1sum`, so the mark and the build agree on which facts made asm/<v>/."""
    data = b"".join(
        (root / f"versions/{v}/{name}").read_bytes()
        for name in ("symbol_addrs.txt", "splat.yaml")
        if (root / f"versions/{v}/{name}").is_file()
    )
    return sha1(data)


def _mark_facts(config: Config, v: str) -> None:
    """Record the facts hash in asm/<v>/.extracted (content only: the mark keeps its mtime, which dates the extraction).
    make refuses to compile against a mark whose hash differs from the tree's facts."""
    mark = config.project.root / "asm" / v / ".extracted"
    if not mark.is_file():
        return
    want = (_facts_hash(config.project.root, v) + "\n").encode()
    if mark.read_bytes() != want:
        info = mark.stat()
        mark.write_bytes(want)
        os.utime(mark, ns=(info.st_atime_ns, info.st_mtime_ns))


def _why_path(config: Config, v: str) -> Path:
    return config.project.root / ".unbake" / f"extract-{v}.why"


def _asm_manifest_path(config: Config, v: str) -> Path:
    return config.project.root / "asm" / v / ".unbake-asm.json"


def _asm_files(config: Config, v: str) -> dict[str, Path]:
    """What one extraction generates, by project path: asm/<v>/ and the splat files build/<v>/*_auto.txt."""
    root = config.project.root
    found = [p for p in sorted((root / "asm" / v).rglob("*")) if p.is_file() and not p.name.startswith(".unbake")]
    found += sorted((root / "build" / v).glob("*_auto.txt"))
    return {str(p.relative_to(root)): p for p in found}


def _write_asm_manifest(config: Config, v: str) -> None:
    """Size, mtime and sha1 of every file the extraction wrote, so a later run can tell a file that changed under it."""
    rows = {}
    for rel, p in _asm_files(config, v).items():
        info = p.stat()
        rows[rel] = [info.st_size, info.st_mtime_ns, sha1(p.read_bytes())]
    store.write(_asm_manifest_path(config, v), json.dumps(rows, sort_keys=True).encode())


def _asm_damaged(config: Config, v: str) -> list[str]:
    """Files of asm/<v>/ that are missing or whose bytes differ from what the extraction recorded (a write through a
    shared link, an editor). Size and mtime clear a file at once; only a changed one is hashed. [] when no record."""
    try:
        rows = json.loads(_asm_manifest_path(config, v).read_text())
    except (FileNotFoundError, ValueError):
        return []
    base = config.project.root
    damaged = []
    for rel, (size, mtime, digest_) in sorted(rows.items()):
        if rel == f"asm/{v}/.extracted":
            continue  # the mark's content is the facts hash, rewritten without a new extraction
        try:
            info = (base / rel).stat()
            if (info.st_size, info.st_mtime_ns) != (size, mtime) and sha1((base / rel).read_bytes()) != digest_:
                damaged.append(rel)
        except FileNotFoundError:
            damaged.append(rel)
    return damaged


_FACT_ROW = re.compile(r"^(\S+)\s*=\s*(0x[0-9A-Fa-f]+)\s*;(.*)$")
_AUTO_NAME = re.compile(r"^[A-Za-z]+_?[0-9A-Fa-f]{8}$")  # what splat names an address no fact names
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TEXT = (".s", ".inc", ".h", ".txt", ".ld")


def _facts_snapshot(config: Config, v: str) -> Path:
    return config.project.root / "asm" / v / ".unbake-facts"


def _yaml_snapshot(config: Config, v: str) -> Path:
    return config.project.root / "asm" / v / ".unbake-yaml"


def _snapshot_inputs(config: Config, v: str) -> None:
    """The facts and piece list this extraction was made from: the next setup tells what changed in them."""
    root = config.project.root
    for snap, source in ((_facts_snapshot(config, v), "symbol_addrs.txt"), (_yaml_snapshot(config, v), "splat.yaml")):
        try:
            store.write(snap, (root / f"versions/{v}/{source}").read_bytes())
        except OSError:
            snap.unlink(missing_ok=True)


def _read_facts(text: str) -> tuple[dict[str, tuple[str, str]], list[str]] | None:
    rows: dict[str, tuple[str, str]] = {}
    other = []
    for line in text.splitlines():
        m = _FACT_ROW.match(line.strip())
        if m is None:
            other.append(line)
        elif m.group(1) in rows:
            return None
        else:
            rows[m.group(1)] = (m.group(2).lower(), m.group(3).strip())
    return rows, sorted(other)


def _auto_name_safe(name: str, before: Mapping[str, tuple[str, str]], after: Mapping[str, tuple[str, str]]) -> bool:
    """Whether a fact named like splat's own name can be renamed in the text: its suffix is its own address, or the
    address the suffix spells is a fact's address in both fact lists (splat prints that fact's name there, never the
    auto name, so the token cannot stand for another symbol: D_800CB7F0 at 0x800CB700 beside D_800CB7F0_us at
    0x800CB7F0)."""
    if name[-8:].upper() == before[name][0][2:].upper().zfill(8):
        return True
    want = int(name[-8:], 16)
    return all(any(int(a, 16) == want for a, _ in rows.values()) for rows in (before, after))


def _fact_delta(old: str, new: str) -> tuple[dict[str, str], set[str]] | None:
    """What turned the facts `old` into `new` when that is only names and unused rows: (renames, drops). A rename is an
    old name that keeps its address and every attribute under a new name, {} when only the order of rows moved; a drop
    is a row `new` no longer has (the caller proves its name never reached the extraction's text). None for any other
    difference: an added fact, a changed address or attribute moves what splat prints beyond a name."""
    before, after = _read_facts(old), _read_facts(new)
    if before is None or after is None or before[1] != after[1]:
        return None
    gone = {n: f for n, f in before[0].items() if after[0].get(n) != f}
    came = {n: f for n, f in after[0].items() if before[0].get(n) != f}
    by_fact = {f: n for n, f in came.items()}
    if len(by_fact) != len(came) or len(set(gone.values())) != len(gone):
        return None
    if any(n in after[0] for n in gone):  # a name that stays a fact with other attributes
        return None
    renames = {n: by_fact[f] for n, f in gone.items() if f in by_fact}
    if len(renames) != len(came):  # a fact with no old row is an added fact
        return None
    for old_name, new_name in renames.items():
        if not _IDENT.match(new_name) or _AUTO_NAME.match(new_name):
            return None
        # An old name that looks like splat's own name for another address cannot be told from that symbol's text.
        if _AUTO_NAME.match(old_name) and not _auto_name_safe(old_name, before[0], after[0]):
            return None
    return renames, {n for n in gone if n not in renames}


def _rename_delta(old: str, new: str) -> dict[str, str] | None:
    """The renames of `_fact_delta`, None when it has a drop or is not one."""
    delta = _fact_delta(old, new)
    return None if delta is None or delta[1] else delta[0]


_YAML_ROW = re.compile(r"^(\s+- \[0x[0-9A-Fa-f]+, \w+, )(\S+?)(\]\s*)$")


def _yaml_delta(old: str, new: str) -> dict[str, str] | None:
    """Old file name -> new file name when `new` differs from `old` only by the names of pieces (the part of a row after
    its folder), {} when it does not differ; None for any other difference (a row added, removed or moved)."""
    a, b = old.splitlines(), new.splitlines()
    if len(a) != len(b):
        return None
    renames: dict[str, str] = {}
    kept = set()
    for x, y in zip(a, b, strict=True):
        mx, my = _YAML_ROW.match(x), _YAML_ROW.match(y)
        if x == y:
            if mx:
                kept.add(mx.group(2).rpartition("/")[2])
            continue
        if not mx or not my or mx.group(1) != my.group(1) or mx.group(3) != my.group(3):
            return None
        (folder, _, was), (other, _, now) = (m.group(2).rpartition("/") for m in (mx, my))
        if folder != other or not _IDENT.match(now) or renames.setdefault(was, now) != now:
            return None
    if kept & (renames.keys() | set(renames.values())):  # a name another row still has
        return None
    return renames


def _renames(config: Config, v: str) -> tuple[dict[str, str], set[str]] | None:
    """The renames and dropped fact names that turn the facts and piece names of the extraction asm/<v>/ holds into
    today's, when that is all that moved: the snapshots, the stamp and every other input (splat, extract recipe, ROM)
    must agree, else None. An extraction made before the yaml was snapshotted counts the yaml as unmoved."""
    root = config.project.root
    try:
        snapshot = _facts_snapshot(config, v).read_bytes()
        pieces = _yaml_snapshot(config, v)
        was_yaml = pieces.read_bytes() if pieces.is_file() else None
        stamp = _extracted_key(config, v)
        if stamp is None or _extract_key(config, v, snapshot, was_yaml) != stamp:
            return None
        delta = _fact_delta(snapshot.decode(), (root / f"versions/{v}/symbol_addrs.txt").read_text())
        names = (
            {} if was_yaml is None else _yaml_delta(was_yaml.decode(), (root / f"versions/{v}/splat.yaml").read_text())
        )
        if delta is None or names is None:
            return None
        renames, drops = delta
        if any(renames.setdefault(n, m) != m for n, m in names.items()) or drops & (renames.keys() | names.keys()):
            return None
        return renames, drops
    except (OSError, UnicodeDecodeError):
        return None


def _fact_why(old: str, new: str) -> str:
    """Why `_fact_delta` refuses to treat old -> new as renames: the first reason, with counts and a few names."""
    before, after = _read_facts(old), _read_facts(new)
    if before is None or after is None:
        return "a fact name appears twice"
    if before[1] != after[1]:
        return "a line that is not a fact row changed"
    gone = {n: f for n, f in before[0].items() if after[0].get(n) != f}
    came = {n: f for n, f in after[0].items() if before[0].get(n) != f}
    by_fact = {f for f in came.values()}
    stay = sorted(n for n in gone if n in after[0])
    if stay:
        return f"{len(stay)} names stay with other attributes, e.g. {', '.join(stay[:3])}"
    added = sorted(n for n, f in came.items() if f not in set(gone.values()))
    if added:
        return f"{len(added)} added facts (a fact with no old row), e.g. {', '.join(added[:3])}"
    if len(by_fact) != len(came) or len(set(gone.values())) != len(gone):
        return "two names share one fact"
    ren = {n: next(k for k, f in came.items() if f == gone[n]) for n in gone}
    bad_new = sorted(m for m in ren.values() if not _IDENT.match(m) or _AUTO_NAME.match(m))
    if bad_new:
        return (
            f"{len(bad_new)} new names are not identifiers or look like splat's own names, "
            f"e.g. {', '.join(bad_new[:3])}"
        )
    mis = sorted(n for n in ren if _AUTO_NAME.match(n) and not _auto_name_safe(n, before[0], after[0]))
    return f"{len(mis)} old auto-looking names are misaddressed, e.g. {', '.join(mis[:3])}"


def _read_why(config: Config, v: str) -> str:
    try:
        return _why_path(config, v).read_text()
    except OSError:
        return "no record"


def _extract_why(config: Config, v: str) -> str:
    """Why the next setup extracts version v again, and by which path: '' when it will not, else 'full: <cause>' (splat
    runs and every file is written) or 'rename: N renames M drops' (the cheap path). Setup reports this beside
    `extracted`, so a round that rewrites all assembly names the input that forced it."""
    root = config.project.root
    asm = root / "asm" / v
    if _extracted_key(config, v) == _extract_key(config, v) and asm.is_dir():
        damaged = _asm_damaged(config, v)
        if not damaged:
            return ""
        return f"full: {len(damaged)} files damaged since the extraction, e.g. {', '.join(damaged[:3])}"
    if not asm.is_dir():
        return "full: no asm directory"
    damaged = _asm_damaged(config, v)
    if damaged:
        return f"full: {len(damaged)} files damaged since the extraction, e.g. {', '.join(damaged[:3])}"
    try:
        snapshot = _facts_snapshot(config, v).read_bytes()
    except OSError:
        return "full: no facts snapshot of the last extraction"
    pieces = _yaml_snapshot(config, v)
    was_yaml = pieces.read_bytes() if pieces.is_file() else None
    stamp = _extracted_key(config, v)
    if stamp is None:
        return "full: no extraction stamp"
    if _extract_key(config, v, snapshot, was_yaml) != stamp:
        return (
            "full: an input other than the facts and piece names moved (splat, extract recipe, ROM) "
            "or the snapshots disagree with the stamp"
        )
    now = (root / f"versions/{v}/symbol_addrs.txt").read_text()
    delta = _fact_delta(snapshot.decode(), now)
    if delta is None:
        return "full: " + _fact_why(snapshot.decode(), now)
    if was_yaml is not None and _yaml_delta(was_yaml.decode(), (root / f"versions/{v}/splat.yaml").read_text()) is None:
        return "full: splat.yaml changed beyond piece names"
    return f"rename: {len(delta[0])} renames {len(delta[1])} drops"


def _rename_extraction(
    config: Config, v: str, aside: Path, renames: Mapping[str, str], drops: Collection[str] = ()
) -> bool:
    """The extraction of today's facts made from the last one (`aside`, the old asm/<v>/) by renaming symbols in its
    text and in the files named for them: splat prints a name wherever a symbol is used and nothing else moves when the
    address and attributes stay (byte for byte against a cold splat of 3e15b80c1's rename, all of asm/de and build/de).
    A dropped fact (`drops`) whose name no text and no file name holds changed nothing splat printed but its row in the
    undefined-symbol lists, so the row goes. False when a new name is already in the text, a file that is not text holds
    a name, or a dropped name is used: splat runs instead."""
    root = config.project.root
    new = root / "asm" / v
    new.mkdir(parents=True, exist_ok=True)
    mapping = {n.encode(): m.encode() for n, m in renames.items()}
    if mapping:
        old_re = re.compile(rb"\b(" + b"|".join(re.escape(n) for n in mapping) + rb")\b")
        new_re = re.compile(rb"\b(" + b"|".join(re.escape(n) for n in mapping.values()) + rb")\b")
    else:
        old_re = new_re = re.compile(rb"(?!)")
    dropped = {n.encode() for n in drops}
    drop_re = (
        re.compile(rb"\b(" + b"|".join(re.escape(n) for n in dropped) + rb")\b") if dropped else re.compile(rb"(?!)")
    )

    def rewrite(data: bytes) -> bytes:
        return old_re.sub(lambda m: mapping[m.group(1)], data)

    for src in sorted(aside.rglob("*")):
        dst = new / src.relative_to(aside)
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if src.name.startswith(".unbake") or src.name == ".extracted":
            continue
        if src.suffix == ".s" and src.stem in renames:
            dst = dst.with_name(renames[src.stem] + ".s")
        data = src.read_bytes()
        if new_re.search(data) or drop_re.search(data) or drop_re.fullmatch(src.stem.encode()):
            return False
        if old_re.search(data):
            if src.suffix not in _TEXT:
                return False
            store.write(dst, rewrite(data))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)  # unchanged bytes keep their inode, so their mtime and the objects built on them
            except OSError:
                shutil.copy2(src, dst)
    store.write(new / ".extracted", b"")  # a cold extraction writes a fresh mark
    for path in [*sorted((root / "build" / v).glob("*_auto.txt")), root / "build" / v / "splat.ld"]:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            continue
        if new_re.search(data):
            return False
        text = rewrite(data)
        if dropped and path.name.endswith("_auto.txt"):
            text = b"".join(ln for ln in text.splitlines(keepends=True) if not _is_drop_row(dropped, ln))
        if drop_re.search(text):
            return False
        if text != data:
            store.write(path, text)
    return True


def _is_drop_row(names: Collection[bytes], line: bytes) -> bool:
    """Whether `line` is a splat undefined-symbol row `name = 0xADDR;` of one of `names`."""
    head, eq, _ = line.partition(b"=")
    return bool(eq) and head.strip() in names


def _prune_superseded(config: Config, v: str) -> bool:
    """Remove the extracted assembly an earlier name of a function left behind (asm/<v>/nonmatchings/<seg>/<old>.s
    when the facts call that address by another name): readers of the assembly (sync --place) took it for a second
    function reading the same tables. True when a file went."""
    root = config.project.root
    if not (root / f"versions/{v}/symbol_addrs.txt").is_file() or not (root / "asm" / v / "nonmatchings").is_dir():
        return False
    gone = symbols.superseded_asm(root, v, api.symbols(config, v))
    for file in gone:
        file.unlink(missing_ok=True)
    rows = _manifest_rows(config, v)
    if gone and rows is not None:  # the record no longer lists them (not damage); other differences still count
        for file in gone:
            rows.pop(str(file.relative_to(root)), None)
        store.write(_asm_manifest_path(config, v), json.dumps(rows, sort_keys=True).encode())
    return bool(gone)


def extract_job(job: tuple[Config, str]) -> bool:
    """One splat run unless asm/<v>/ already holds the extraction of this input; true when splat ran.

    The proof is a stamp written beside the extracted files: the tree's own asm/<v>/ was made from the input named
    there. A host-wide cache of "this input was extracted somewhere" cannot say what this tree holds, so a tree whose
    facts moved on and came back (or whose neighbour extracted the same input) kept stale assembly.

    An extraction again writes into a fresh asm/<v>/ (the old one moved aside, so no file a cold extraction would not
    write survives, and no link into a shared tree is written through), then hands the build only what changed: a file
    whose bytes equal the last extraction gets its old mtime back, and when no shared include changed the
    `.extracted` mark keeps its old time too and only the objects that include a changed file are rebuilt
    (`build.drop_including`). A rename touches a few callers, not every object of the version."""
    config, v = job
    key = _extract_key(config, v)
    asm = config.project.root / "asm" / v
    damaged = _asm_damaged(config, v) if asm.is_dir() else []  # what the last extraction wrote, still as written?
    if asm.is_dir() and _extracted_key(config, v) == key and not damaged:
        _prune_superseded(config, v)  # facts the stamp names cannot leave a file of an earlier name: one put there does
        if not (_facts_snapshot(config, v).exists() and _yaml_snapshot(config, v).exists()):
            _snapshot_inputs(config, v)  # the key says these facts and pieces made this extraction
        if not (config.project.root / "build" / f".extract-key-{v}").is_file():
            _stamp_make(config, v)  # a tree made before the Makefile skipped unchanged versions
        _mark_facts(config, v)  # the key says these facts made it: an older mark (empty) learns the hash
        return False
    # "Present" is not "correct": what the extraction generated was missing or changed, so it is made again, and the
    # linker scripts built from it (build/<v>/*.ld) go too, for make to render them from the fresh files.
    renames = None if damaged or not asm.is_dir() else _renames(config, v)
    for old in (config.project.root / "build" / v).glob("*.ld"):
        if renames is None or old.name != "splat.ld":  # a rename extraction renames inside splat.ld, splat does not run
            old.unlink(missing_ok=True)
    before = None if damaged or not asm.is_dir() else _manifest_rows(config, v)
    stamp = _extract_stamp(config, v)
    stamp.unlink(missing_ok=True)  # an extraction that dies half-way is not the one the old stamp named
    aside = asm.with_name(f".{v}.{os.getpid()}.old")
    shutil.rmtree(aside, ignore_errors=True)
    if asm.is_dir():
        asm.rename(aside)
    try:
        if renames is None or not _rename_extraction(config, v, aside, *renames):
            shutil.rmtree(asm, ignore_errors=True)  # a refused rename extraction leaves a part-made directory
            if not _restore_extraction(config, v, key):
                _checked(build.extract(config, v), "setup.splat", f"extract {v}")
                _save_extraction(config, v, key)
    finally:
        shutil.rmtree(aside, ignore_errors=True)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    kept = _keep_unchanged(config, v, before)
    _prune_superseded(config, v)  # a rename extraction only renames what the old tree held; files the facts left go
    _mark_facts(config, v)
    _write_asm_manifest(config, v)
    _snapshot_inputs(config, v)
    store.write(stamp, key.encode())
    store.write(_why_path(config, v), kept.encode())
    _stamp_make(config, v)
    return True


_SPLIT_KIND = "split"  # host cache kind: a version's whole extraction, keyed by _extract_key


def _split_files(config: Config, v: str) -> dict[str, Path]:
    """What `make extract-<v>` writes: asm/<v>/ (not unbake records), the splat link script and the _auto lists."""
    found = dict(_asm_files(config, v))
    ld = config.project.root / "build" / v / "splat.ld"
    if ld.is_file():
        found[str(ld.relative_to(config.project.root))] = ld
    return found


def _save_extraction(config: Config, v: str, key: str) -> None:
    """Keep this splat run in the host content cache under the extraction key, so another slot, the train tree or a pool
    refresh with the same ROM, yaml, facts, splat and recipe copies it instead of splitting again (stamp, check and
    pool re-extractions of one diff each paid ~40 CPU s per version). Best effort."""
    try:
        files = _split_files(config, v)
        if not (config.project.root / "asm" / v / ".extracted").is_file():
            return  # a run that did not finish its recipe is no extraction
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for rel, path in files.items():
                info = tar.gettarinfo(str(path), arcname=rel)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
        store.put(config, _SPLIT_KIND, f"extract-v1:{key}", buf.getvalue(), named=True)
    except (OSError, tarfile.TarError, Refusal, AttributeError):  # AttributeError: a config with no host cache
        pass


def _restore_extraction(config: Config, v: str, key: str) -> bool:
    """Write the cached extraction of `key` into the tree (private copies, fresh mtimes); false on a miss or bad entry,
    with nothing left behind so splat can run."""
    try:
        blob = store.get(config, _SPLIT_KIND, f"extract-v1:{key}", named=True)
    except (OSError, Refusal, AttributeError):
        return False
    effort.count("split-cache", blob is not None)
    if blob is None:
        return False
    root = config.project.root
    written: list[Path] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r") as tar:
            members = [m for m in tar.getmembers() if m.isreg()]
            if not any(m.name == f"asm/{v}/.extracted" for m in members):
                return False
            for m in members:
                if ".." in Path(m.name).parts or not m.name.startswith((f"asm/{v}/", f"build/{v}/")):
                    raise tarfile.TarError(m.name)
                target = root / m.name
                target.parent.mkdir(parents=True, exist_ok=True)
                written.append(target)
                target.unlink(missing_ok=True)
                data = tar.extractfile(m)
                if data is None:
                    raise tarfile.TarError(m.name)
                with target.open("wb") as out:
                    shutil.copyfileobj(data, out)
                target.chmod(m.mode & 0o7777 | 0o200)
    except (OSError, tarfile.TarError):
        for path in written:
            path.unlink(missing_ok=True)
        shutil.rmtree(root / "asm" / v, ignore_errors=True)
        return False
    return True


def _stamp_make(config: Config, v: str) -> None:
    """Tell the Makefile that asm/<v>/ holds the extraction of today's inputs, so `make extract-<v>` (a lane's build
    script) does not split it again. Nothing to do for a Makefile without the skip."""
    makefile = config.project.root / "Makefile"
    if makefile.is_file() and build.rule_text(makefile.read_text(errors="replace"), "extract-stamp-%") is not None:
        _checked(build.make(config, [f"extract-stamp-{v}"]), "setup.splat", f"stamp {v}")


_INCLUDE = re.compile(rb'^\s*\.include\s+"([^"]+)"', re.M)


def _manifest_rows(config: Config, v: str) -> dict[str, list[Any]] | None:
    try:
        rows = json.loads(_asm_manifest_path(config, v).read_text())
    except (FileNotFoundError, ValueError):
        return None
    return rows if isinstance(rows, dict) else None


def _keep_unchanged(config: Config, v: str, before: Mapping[str, list[Any]] | None) -> str:
    """After an extraction again: every file whose bytes equal the last extraction's gets that one's mtime back. When
    every changed file is one only C objects include (not a top-level file, not a file other assembly includes), the
    `.extracted` mark gets its old mtime too and the objects including a changed file are removed for make to rebuild;
    otherwise the fresh mark rebuilds the whole version, as after a first extraction. Returns what it decided, in words:
    setup reports it (`why`) so a round that rebuilds every object names the file that forced it."""
    if before is None:
        return "whole version rebuilds: no record of the last extraction to compare"
    root, prefix = config.project.root, f"asm/{v}/"
    mark = f"{prefix}.extracted"
    changed, includes = [rel for rel in before if not (root / rel).exists()], {}
    for rel, path in _asm_files(config, v).items():
        data = path.read_bytes()
        includes[rel[len(prefix) :]] = {name.decode(errors="replace") for name in _INCLUDE.findall(data)}
        old = before.get(rel)
        if rel == mark:
            continue
        if old and old[0] == len(data) and old[2] == sha1(data):
            os.utime(path, ns=(old[1], old[1]))
        else:
            changed.append(rel)
    names = {rel[len(prefix) :] for rel in changed if rel.startswith(prefix)}
    top = sorted(name for name in names if "/" not in name)
    if mark not in before or top:  # a top-level file (a macro set) reaches every object
        if top:
            return f"whole version rebuilds: {len(top)} top-level files changed, e.g. {', '.join(top[:3])}"
        return "whole version rebuilds: the last extraction recorded no mark"
    reach = _including(names, includes)
    effort.count("extract-reach", len(reach) <= len(names))
    os.utime(root / mark, ns=(before[mark][1], before[mark][1]))
    build.drop_including(config, v, reach)
    return f"{len(names)} files changed, {len(reach)} reached by their includers"


def _including(names: set[str], includes: Mapping[str, set[str]]) -> set[str]:
    """`names` and every file of the extraction that includes one of them, directly or through others: what an object
    that includes any of them was built from. An include is matched by its path or its tail (`-I` spells it shorter),
    and each name also stands for its tails, which is how the compiler output of a C unit may spell it."""
    reach = set(names)
    todo = list(names)
    while todo:
        name = todo.pop()
        tails = {name, *(name.split("/", i)[-1] for i in range(1, name.count("/") + 1))}
        for file, spelled in includes.items():
            if file not in reach and spelled & tails:
                reach.add(file)
                todo.append(file)
    return reach | {tail for name in reach for tail in (name.split("/", i)[-1] for i in range(1, name.count("/") + 1))}


@cache
def _splat_version(tool: Path, root: Path) -> bytes:
    result = process.run("splat version", [str(tool), "--version"], tool.parent, tmp=process.scratch(root))
    if result.exit:
        raise Refusal(Finding("setup.splat", result.stderr.decode(errors="replace")))
    return result.stdout


def _extract_key(config: Config, v: str, facts: bytes | None = None, yaml: bytes | None = None) -> str:
    """splat version, the Makefile's extract recipe with the variables it uses (it rewrites what splat writes; a
    change anywhere else in the Makefile extracts nothing again), yaml, symbol_addrs and ROM: nothing else decides
    what extraction writes. A Makefile without that rule counts whole."""
    root = config.project.root
    makefile = root / "Makefile"
    text = makefile.read_bytes() if makefile.is_file() else None
    recipe = None if text is None else build.rule_text(text.decode(errors="replace"), "extract-%")
    return digest(
        (
            _splat_version(config.host.tools["splat"], root),
            (sha1(recipe.encode()) if recipe is not None else sha1(text)) if text is not None else None,
            sha1((root / f"versions/{v}/splat.yaml").read_bytes() if yaml is None else yaml),
            sha1((root / f"versions/{v}/symbol_addrs.txt").read_bytes() if facts is None else facts),
            sha1(configuration.rom(config.project, v).read_bytes()),
        )
    )


def _extractable(config: Config) -> list[str]:
    root = config.project.root
    return [v for v in config.project.versions if (root / f"versions/{v}/splat.yaml").is_file()]


def _state(config: Config) -> str:
    """Everything setup reads or writes: the tool, the commit, the working tree and the extraction inputs. Unchanged
    means done."""
    root = config.project.root
    head = process.git(config, "rev-parse", "HEAD").stdout.decode().strip()
    dirty = process.git(config, "status", "--porcelain", "--untracked-files=all").stdout
    tool = digest(
        [p.read_bytes() for p in sorted(Path(__file__).parent.rglob("*")) if p.is_file() and p.suffix != ".pyc"]
    )  # the tool's content, not when it was touched
    facts = [sha1((root / f"versions/{v}/symbol_addrs.txt").read_bytes()) for v in config.project.versions]
    stamps = [(_extract_key(config, v), _extracted_key(config, v)) for v in _extractable(config)]
    return digest((head, dirty, config.digest, tool, facts, stamps))


def _extract(config: Config, versions: Sequence[str]) -> list[str]:
    ran = pool.map(config, "repo.extract", extract_job, [(config, v) for v in versions])
    for did in ran:
        effort.count("extract", not did)
    return [v for v, did in zip(versions, ran, strict=True) if did]


def _extracting(config: Config) -> bool:
    """Whether a setup now would extract some version again (its asm and splat files rewritten)."""
    try:
        return any(_extract_why(config, v) for v in _extractable(config))
    except (OSError, ValueError):
        return True


def setup(config: Config, params: Json) -> Json:
    """Extract what moved and sync the tree. An extraction rewrites asm/<v>/ and build/<v>/*_auto.txt, which make never
    makes again, so it runs under the tree lock held exclusively: a tree copied from this one (a stamp or lane seed
    holds the lock shared) never gets a half-written extraction (10-25: a seed copied during a pool refresh lacked
    build/de/undefined_syms_auto.txt and every link of the stamp failed). The tree lock comes before the store lock."""
    if not _extracting(config):
        return _setup(config, params)
    with treelock.hold(config.project.root, exclusive=True, who="unbake setup (extracting)"):
        return _setup(config, params)


def _setup(config: Config, params: Json) -> Json:
    wait_s = config.host.move_wait_s
    with effort.stage("repo.setup"), store.exclusive(config, "structure", wait=True, wait_s=wait_s) as held:
        adapters.check_abucache()
        if not held:
            raise Refusal(Finding("setup.busy", configuration.sentence("setup.busy")))
        journal.recover(config)
        project = config.project
        # the machine paths the Makefile runs (compiler fronts among them) follow the installed tool on every setup:
        # the build that decomp land runs is a plain `make check`, which never writes local.mk itself
        store.write(project.root / "local.mk", build.local_mk(config))
        done = project.root / ".unbake/setup.json"
        kept = json.loads(done.read_bytes()) if done.exists() else None
        build_too = not params.get("no_make")
        if kept and kept["state"] == _state(config) and (kept["result"].get("make") or not build_too):
            effort.count("setup", True)
            return {**kept["result"], "extracted": [], "files": [], "message": "setup"}
        effort.count("setup", False)
        with effort.stage("repo.setup.toolchains"):
            toolchains = sorted(
                {project.toolchain, *(row.toolchain for row in project.files.values() if row.toolchain)}
            )
            build.install_toolchains(config, toolchains)
        with effort.stage("repo.setup.extract"):
            why = {v: w for v in _extractable(config) if (w := _extract_why(config, v))}
            extracted = _extract(config, _extractable(config))
        synced, current = _sync(config, {"check": False})
        written = synced["written"]
        # sync moved an extraction input (the yaml, the facts, the extract recipe): that output is the old one's
        again = [v for v in _extractable(config) if _extract_key(config, v) != _extracted_key(config, v)]
        if again:
            extracted = sorted({*extracted, *_extract(config, again)})
        make = None  # --no-make: the caller builds and records the verdict itself (decomp land's check gate)
        if build_too:
            with effort.stage("repo.setup.make"):
                make = build.make_check(config, current)
            report.progress(config, current)
        result = {
            "toolchains": toolchains,
            "extracted": sorted(extracted),
            "why": {v: f"{w}; {_read_why(config, v)}" for v, w in sorted(why.items()) if v in extracted},
            "files": written,
            "message": "setup",
            "make": make,
        }
        store.write(done, json.dumps({"state": _state(config), "result": result}).encode())
        return result
