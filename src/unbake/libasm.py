"""Foreign routines: code the project's toolchain never emits (coprocessor, cache and sync instructions, FPU control
moves outside compiler conversions). They are library assembly: original source in a configured library directory, never
game assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import rabbitizer

from unbake import adapters, config, effort, layout, versions
from unbake.contracts import Finding, Snapshot, UnitSpec

_LOW_USERS = frozenset({0x08, 0x09, 0x0C, 0x0D, 0x20, 0x21, 0x23, 0x24, 0x25, 0x28, 0x29, 0x2B, 0x31, 0x39})
_CONTROL = frozenset({"cfc1", "ctc1"})
_HAND = "hasm"  # the split row splat gives hand-written assembly: never compiler output, library assembly as it is
_FIX = "place the module in a library directory with `recipe structure` or `unbake move`"


def _signed(value: int) -> int:
    return (value & 0xFFFF) - ((value & 0x8000) << 1)


def foreign(words: Sequence[int], vram: int, toolchain: Mapping[str, Any]) -> bool:
    """Whether the code is not compiler output: it holds an instruction the toolchain never emits, or FPU control moves
    without one of the conversions the toolchain pairs them with."""
    names = {rabbitizer.Instruction(w, vram + 4 * i).getOpcodeName() for i, w in enumerate(words)}
    return bool(names & set(toolchain["never_emits"])) or (
        bool(names & _CONTROL) and not names & set(toolchain["fpu_control_with"])
    )


def _words(snapshot: Snapshot, member: str, version: str) -> tuple[list[int], int] | None:
    place = next(
        (p for p in snapshot.layout.members[member].placements if p.version == version and p.section == ".text"), None
    )
    if place is None:
        return None
    blob = versions.rom_bytes(snapshot.versions[version], place.rom_start, place.rom_end)
    return [int.from_bytes(blob[i : i + 4], "big") for i in range(0, len(blob) - 3, 4)], place.vram


def recognised(snapshot: Snapshot, member: str) -> bool:
    """Whether the function is foreign in every version that holds it."""
    item = snapshot.layout.members[member]
    if item.kind != "function":
        return False
    row = adapters.row(snapshot.config.project.toolchain)
    rows = [_words(snapshot, member, v) for v in item.holders()]
    return bool(rows) and all(r is not None and foreign(*r, row) for r in rows)


def _operands(instruction: rabbitizer.Instruction, word: int) -> str:
    """The instruction as GNU as reads it: coprocessor registers by number, immediates plain."""
    text = instruction.disassemble()
    if instruction.getOpcodeName() in ("mfc0", "mtc0", "dmfc0", "dmtc0", "cfc1", "ctc1"):
        return f"{text.rsplit(',', 1)[0]}, ${word >> 11 & 31}"
    return text


def _target(ins: rabbitizer.Instruction, at: int) -> int | None:
    if ins.isBranch():
        return at + int(ins.getGenericBranchOffset(at))
    return int(ins.getInstrIndexAsVram()) if ins.getOpcodeName() in ("j", "jal") else None


def render(name: str, words: Sequence[int], vram: int, symbols: Mapping[int, str]) -> str | None:
    """Readable assembly of the words, one instruction per line, or None when a target outside the routine has no
    symbol. Internal targets get labels, external ones read as their symbol; a `lui` and the low-half instruction that
    completes a symbol's address read `%hi(symbol)` and `%lo(symbol)`."""
    code = [rabbitizer.Instruction(w, vram + 4 * i) for i, w in enumerate(words)]
    end = vram + 4 * len(words)
    targets = {i: t for i, ins in enumerate(code) if (t := _target(ins, vram + 4 * i)) is not None}
    if any(not vram <= t < end and t not in symbols for t in targets.values()):
        return None
    labels = {(t - vram) // 4 for t in targets.values() if vram <= t < end}
    override: dict[int, str] = {}
    for i, word in enumerate(words):
        if word >> 26 != 0x0F or (word >> 16 & 31) == 0:
            continue
        register = word >> 16 & 31
        for j in range(i + 1, min(i + 9, len(words))):
            low = words[j]
            if low >> 26 in _LOW_USERS and low >> 21 & 31 == register:
                address = ((word & 0xFFFF) << 16) + _signed(low) & 0xFFFFFFFF
                if address in symbols:
                    override[i], override[j] = f"%hi({symbols[address]})", f"%lo({symbols[address]})"
                break
            if _writes(words[j], register) or (words[j] >> 26 in (2, 3)):
                break
    lines = [".set noreorder", ".set noat", ".section .text", f".globl {name}", f"{name}:"]
    for i, ins in enumerate(code):
        if i in labels:
            lines.append(f".L{4 * i:03X}:")
        if i in override:
            text = ins.disassemble(immOverride=override[i])
        elif i in targets:
            t = targets[i]
            text = ins.disassemble(immOverride=f".L{t - vram:03X}" if vram <= t < end else symbols[t])
        else:
            text = _operands(ins, words[i])
        lines.append("    " + " ".join(text.split()))
    return "\n".join(lines) + "\n"


def _writes(word: int, register: int) -> bool:
    """Whether the word writes the register (a destination in rt for immediates and loads, rd for register forms)."""
    op = word >> 26
    if op == 0:
        return word >> 11 & 31 == register
    return (
        0x08 <= op <= 0x0F or 0x20 <= op <= 0x27 or (op == 0x10 and word >> 21 & 31 == 0)
    ) and word >> 16 & 31 == register


def _text(snapshot: Snapshot, member: str) -> str | Finding:
    """The assembly of the member, or the finding saying why its versions cannot be rendered."""
    texts = set()
    for version in snapshot.layout.members[member].holders():
        read = _words(snapshot, member, version)
        names = {a: n for n, a in sorted(snapshot.versions[version].symbols.items(), reverse=True)}
        text = None if read is None else render(member, read[0], read[1], names)
        if text is None:
            return _finding("library-asm.unnamed-target", member, "")
        texts.add(text)
    return texts.pop() if len(texts) == 1 else _finding("library-asm.version-split", member, "")


def _finding(key: str, member: str, path: str) -> Finding:
    return Finding(
        key,
        config.sentence(key),
        path=path,
        unit=member,
        blocking=False,
        action=_FIX if key.endswith(("unplaced", "mixed")) else "",
    )


def _scattered(snapshot: Snapshot, group: str, names: Sequence[str]) -> bool:
    """Whether another module's code lies between the routines in some version: one `.s` links one contiguous run."""
    members = snapshot.layout.members
    for version in {v for n in names for v in members[n].holders()}:
        runs = [p for n in names for p in members[n].placements if p.version == version and p.section == ".text"]
        low, high = min(p.rom_start for p in runs), max(p.rom_end for p in runs)
        if any(
            m.group != group and p.version == version and p.section == ".text" and low <= p.rom_start < high
            for m in members.values()
            for p in m.placements
        ):
            return True
    return False


def plan(snapshot: Snapshot) -> tuple[dict[str, bytes | None], tuple[Finding, ...]]:
    """The writes that turn each unlanded foreign or hand-written (splat `hasm`) assembly function into library
    assembly, one `.s` per module at the module's own path: its split rows turned to compiled and its unit in the
    layout map; and the findings for the functions that cannot be."""
    project = snapshot.config.project
    with effort.stage("libasm.plan"):
        compiled, assembly, _ = layout._kinds()
        writes: dict[str, bytes | None] = {}
        findings: list[Finding] = []
        units = dict(snapshot.layout.units)
        pending: dict[str, list[str]] = {}
        for name, member in snapshot.layout.members.items():
            if (
                member.kind == "function"
                and member.state in (assembly, _HAND)
                and member.group
                and layout.unit_of(snapshot, name) is None
                and (member.state == _HAND or recognised(snapshot, name))
            ):
                pending.setdefault(member.group, []).append(name)
        for group, names in pending.items():
            source = snapshot.layout.groups[group].source
            path = str(Path(source).with_suffix(".s"))
            members = snapshot.layout.members
            if any(members[m].kind == "function" and m not in names for m in snapshot.layout.groups[group].members):
                findings += [_finding("library-asm.mixed", n, source) for n in names]
                continue
            if not config.library(project, path):
                findings += [_finding("library-asm.unplaced", n, source) for n in names]
                continue
            if _scattered(snapshot, group, names):
                findings += [_finding("library-asm.scattered", n, source) for n in names]
                continue
            done = []
            for name in names:
                text = _text(snapshot, name)
                if isinstance(text, Finding):
                    findings.append(replace(text, path=source))
                    continue
                done.append(name)
                writes[path] = (writes.get(path) or b"") + text.encode()
                for holder in members[name].holders():
                    split = snapshot.versions[holder].split
                    current = writes.get(split) or snapshot.read(split)
                    writes[split] = layout.edit_split(current, name, state=compiled, types=frozenset({assembly, _HAND}))
            if done:
                units[path] = UnitSpec(path, assembly, group, tuple(done), project.toolchain, {"add": [], "omit": []})
        if writes:
            writes["layout.toml"] = layout.dump_map(replace(snapshot.layout, units=units))
        return writes, tuple(findings)


def outside(snapshot: Snapshot) -> tuple[str, ...]:
    """The assembly units that are not in a library directory: game assembly, which the policy refuses."""
    kinds = config.kinds()
    return tuple(
        sorted(
            u.path
            for u in snapshot.layout.units.values()
            if kinds[u.kind]["credit"] == "code"
            and "compile" not in kinds[u.kind]["phases"]
            and not config.library(snapshot.config.project, u.path)
        )
    )
