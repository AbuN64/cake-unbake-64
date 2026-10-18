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


def _rom_words(
    cfg: Config, held: Any, roms: dict[str, bytes], version: str, name: str, *, needed: bool = False
) -> tuple[list[int], int] | None:
    """The words and address of a function name in a version, or None when its extent is unknown. A ROM that is not
    there yields None too (a tree is checked without ROMs: a stand-in nothing can show foreign stays what it is, which
    is what a ROM-present run decides for a tree already extracted); a placement that cannot be decided without
    those words (`needed`) refuses naming the routine. A ROM that is there but not the configured one refuses."""
    table = held.symbols[version]
    vram, rom, size, kind = table[name]
    if kind != "func" or rom is None:
        return None
    next_rom = min(
        (r[1] for r in table.values() if r[1] is not None and r[1] > rom), default=cfg.project.version[version].size
    )
    size = min(size or next_rom - rom, next_rom - rom)
    if size <= 0 or size % 4:
        return None
    if version not in roms:
        if not (cfg.project.root / "roms" / f"baserom.{version}.z64").is_file():
            if needed:
                raise Refusal(
                    Finding(
                        "library-asm.rom",
                        f"`{name}` is a library routine nothing owns and only the {version} ROM shows its code",
                        path=f"roms/baserom.{version}.z64",
                        action=f"run unbake sync with roms/baserom.{version}.z64 present and commit the result",
                    )
                )
            return None
        roms[version] = config.rom(cfg.project, version).read_bytes()
    raw = roms[version][rom : rom + size]
    if len(raw) != size:
        return None
    return [int.from_bytes(raw[i : i + 4], "big") for i in range(0, len(raw), 4)], vram


def _home(cfg: Config, held: Any, version: str, name: str) -> Any:
    """The library source file that keeps a library routine nobody owns: the C file of a library directory owning the
    nearest address of the same mapping (ROM and RAM offset), the one before the name first."""
    table = held.symbols[version]
    vram, rom = table[name][0], table[name][1]
    best: tuple[bool, int, Any] | None = None
    for module in held.modules:
        if config.is_assembly(module.path) or not config.library(cfg.project, module.path):
            continue
        for owned in module.owns.get(version, ()):
            row = table.get(owned)
            if row is None or row[1] is None or row[0] - row[1] != vram - rom:
                continue
            key = (row[1] > rom, abs(row[1] - rom), module)
            if best is None or key[:2] < best[:2]:
                best = key
    return best[2] if best else None


def homes(cfg: Config, held: Any) -> dict[str, Any]:
    """Library routines no file owns, by name, with the library source file that must keep them: the code is not
    compiler output in any version that has it, so it is library assembly and never belongs in a game file."""
    if not getattr(cfg.project, "libraries", ()):
        return {}
    out: dict[str, Any] = {}
    roms: dict[str, bytes] = {}
    seen: set[str] = set()
    for version in cfg.project.versions:
        for name in held.unowned(version):
            if name in seen:
                continue
            seen.add(name)
            module = _home(cfg, held, version, name)
            if module is None:
                continue
            toolchain = adapters.row(recipes.of(cfg, module.path).toolchain)
            for holder in (v for v in cfg.project.versions if name in held.symbols[v]):
                found = _rom_words(cfg, held, roms, holder, name, needed=True)
                if found is None or not foreign(found[0], found[1], toolchain):
                    break
            else:
                out[name] = module
    return out


def plan(cfg: Config, held: Any) -> dict[str, bytes]:
    """Extract foreign library routines as source, one function per file; existing source is never overwritten. A
    routine is a library stand-in or has lost it (no file owns it); it is rendered only when every version holding it
    renders the same assembly, otherwise it keeps (or gets back) its stand-in in a library file."""
    if not getattr(cfg.project, "libraries", ()):
        return {}
    writes: dict[str, bytes] = {}
    roms: dict[str, bytes] = {}
    by_path = {m.path: m for m in held.modules}
    lost = homes(cfg, held)

    def free(version: str, name: str) -> bool:
        """The version's owner of the name is a library stand-in, or nothing owns it: a `.s` defines the name in every
        version whose facts hold it, so every one of them must give up a stand-in to it."""
        owner = by_path.get(held.owner(version, name) or "")
        if owner is None:
            return name in lost
        return config.library(cfg.project, owner.path) and name in owner.stand_ins.get(version, ())

    candidates = [(m, n) for m in held.modules if config.library(cfg.project, m.path) for n in _stand_ins(m)]
    candidates += [(m, n) for n, m in lost.items()]
    done: set[str] = set()
    for module, name in candidates:
        if name in done:
            continue
        done.add(name)
        toolchain = adapters.row(recipes.of(cfg, module.path).toolchain)
        texts: set[str] = set()
        for version in (v for v in cfg.project.versions if name in held.symbols[v]):
            found = _rom_words(cfg, held, roms, version, name) if free(version, name) else None
            if found is None or not foreign(found[0], found[1], toolchain):
                break
            names = {r[0]: n for n, r in sorted(held.symbols[version].items(), reverse=True)}
            text = render(name, found[0], found[1], names)
            if text is None:
                break
            texts.add(text)
        else:
            if len(texts) == 1:
                path = str(Path(module.path).with_name(name + ".s"))
                raw = texts.pop().encode()
                existing = cfg.project.root / path
                if existing.exists() and existing.read_bytes() != raw:
                    raise Refusal(Finding("library-asm.source", "would overwrite existing assembly source", path=path))
                writes[path] = raw
    return writes


def _stand_ins(module: Any) -> list[str]:
    return sorted({n for names in module.stand_ins.values() for n in names})
