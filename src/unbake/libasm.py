"""Vendored library assembly, extracted from verified ROM words; restored from the pre-layout-cutover renderer."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import rabbitizer

from unbake import adapters, config, recipes
from unbake.contracts import Config, Finding, Refusal

_LOW_USERS = frozenset({0x08, 0x09, 0x0C, 0x0D, 0x20, 0x21, 0x23, 0x24, 0x25, 0x28, 0x29, 0x2B, 0x31, 0x39})
_CONTROL = frozenset({"cfc1", "ctc1"})


def _signed(value: int) -> int:
    return (value & 0xFFFF) - ((value & 0x8000) << 1)


def foreign(words: Sequence[int], vram: int, toolchain: Mapping[str, Any]) -> bool:
    """Whether the code is not compiler output: it holds an instruction the toolchain never emits, or FPU control moves
    without one of the conversions the toolchain pairs them with."""
    names = {rabbitizer.Instruction(w, vram + 4 * i).getOpcodeName() for i, w in enumerate(words)}
    return bool(names & set(toolchain["never_emits"])) or (
        bool(names & _CONTROL) and not names & set(toolchain["fpu_control_with"])
    )


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
    # o32 assembles a doubleword move as two words and an odd double FP register as a pair: code the ROM holds
    # as 64-bit instructions (an exception handler saving the whole context) is assembled with 64-bit registers
    wide = [".set gp=64", ".set fp=64"] if any(_wide(w) for w in words) else []
    lines = [".set noreorder", ".set noat", *wide, ".section .text", f".globl {name}", f"{name}:"]
    for i, ins in enumerate(code):
        if i in labels:
            lines.append(f".L{name}_{4 * i:03X}:")
        if i in override:
            text = ins.disassemble(immOverride=override[i])
        elif i in targets:
            t = targets[i]
            text = ins.disassemble(immOverride=f".L{name}_{t - vram:03X}" if vram <= t < end else symbols[t])
        else:
            text = _operands(ins, words[i])
        lines.append("    " + " ".join(text.split()))
    return "\n".join(lines) + "\n"


_DOUBLE_OPS = frozenset(
    (0x18, 0x19, 0x1A, 0x1B, 0x2C, 0x2D, 0x34, 0x37, 0x3C, 0x3F)
)  # daddi(u) ldl ldr sdl sdr lld ld scd sd
_DOUBLE_SPECIAL = frozenset((0x14, 0x16, 0x17, 0x1C, 0x1D, 0x1E, 0x1F, *range(0x2C, 0x30), *range(0x38, 0x40)))


def _wide(word: int) -> bool:
    """A word o32 cannot assemble as itself: a doubleword GPR operation, or a double FP move of an odd register."""
    op = word >> 26
    if op in _DOUBLE_OPS or (op == 0 and word & 0x3F in _DOUBLE_SPECIAL):
        return True
    if op in (0x35, 0x3D):  # ldc1, sdc1
        return bool(word >> 16 & 1)
    if op == 0x11 and word >> 21 & 31 in (1, 5):  # dmfc1, dmtc1
        return bool(word >> 11 & 1)
    return op == 0x11 and word >> 21 & 31 == 17 and bool((word >> 16 | word >> 11 | word >> 6) & 1)  # .d operands


def _writes(word: int, register: int) -> bool:
    """Whether the word writes the register (a destination in rt for immediates and loads, rd for register forms)."""
    op = word >> 26
    if op == 0:
        return word >> 11 & 31 == register
    return (
        0x08 <= op <= 0x0F or 0x20 <= op <= 0x27 or (op == 0x10 and word >> 21 & 31 == 0)
    ) and word >> 16 & 31 == register


def scan(text: str) -> dict[str, str]:
    """Global definitions and sections in ordinary assembler source; local labels never become units."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    globals_ = set(re.findall(r"(?m)^\s*\.glob(?:l|al)\s+([\w.$]+)", text))
    found: dict[str, str] = {}
    section = ".text"
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if match := re.match(r"\.section\s+([.\w]+)|\.(text|data|rodata|bss)\b", line):
            section = match[1] or "." + match[2]
        if match := re.match(r"([\w.$]+):", line):
            if match[1] in globals_:
                found[match[1]] = section
        elif match := re.match(r"[gd]label\s+([\w.$]+)", line):
            found[match[1]] = section
    return found


def plan(cfg: Config, held: Any) -> dict[str, bytes]:
    """Extract foreign library stand-ins as source, one function per file; existing source is never overwritten."""
    if not getattr(cfg.project, "libraries", ()):
        return {}
    writes: dict[str, bytes] = {}
    roms: dict[str, bytes] = {}
    by_path = {m.path: m for m in held.modules}

    def stand_in(version: str, name: str) -> bool:
        """The version's owner of the name is a library stand-in: a `.s` defines the name in every version whose facts
        hold it, so every one of them must give up a stand-in to it."""
        owner = by_path.get(held.owner(version, name) or "")
        return (
            owner is not None and config.library(cfg.project, owner.path) and name in owner.stand_ins.get(version, ())
        )

    for module in held.modules:
        if not config.library(cfg.project, module.path):
            continue
        toolchain = adapters.row(recipes.of(cfg, module.path).toolchain)
        for name in sorted({n for names in module.stand_ins.values() for n in names}):
            texts: set[str] = set()
            holders = [v for v in cfg.project.versions if name in held.symbols[v]]
            for version in holders:
                vram, rom, size, kind = held.symbols[version][name]
                if kind != "func" or rom is None or not stand_in(version, name):
                    break
                table = held.symbols[version]
                next_rom = min(
                    (r[1] for r in table.values() if r[1] is not None and r[1] > rom),
                    default=cfg.project.version[version].size,
                )
                size = min(size or next_rom - rom, next_rom - rom)
                if size <= 0 or size % 4:
                    break
                if version not in roms:
                    roms[version] = config.rom(cfg.project, version).read_bytes()
                raw = roms[version][rom : rom + size]
                words = [int.from_bytes(raw[i : i + 4], "big") for i in range(0, len(raw), 4)]
                if len(raw) != size or not foreign(words, vram, toolchain):
                    break
                names = {r[0]: n for n, r in sorted(table.items(), reverse=True)}
                text = render(name, words, vram, names)
                if text is None:
                    break
                texts.add(text)
            else:
                if len(texts) == 1:
                    path = str(Path(module.path).with_name(name + ".s"))
                    raw = texts.pop().encode()
                    existing = cfg.project.root / path
                    if existing.exists() and existing.read_bytes() != raw:
                        raise Refusal(
                            Finding("library-asm.source", "would overwrite existing assembly source", path=path)
                        )
                    writes[path] = raw
    return writes
