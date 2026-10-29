"""Vendored library assembly, extracted from verified ROM words; restored from the pre-layout-cutover renderer."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import rabbitizer

from unbake import adapters, config, recipes
from unbake.contracts import Config, Finding, Refusal, macro

_LOW_USERS = frozenset({0x08, 0x09, 0x0C, 0x0D, 0x20, 0x21, 0x23, 0x24, 0x25, 0x28, 0x29, 0x2B, 0x31, 0x39})
_CONTROL = frozenset({"cfc1", "ctc1"})


def _signed(value: int) -> int:
    return (value & 0xFFFF) - ((value & 0x8000) << 1)


_LOADS = frozenset({"lb", "lbu", "lh", "lhu", "lw", "lwl", "lwr", "lwu", "ld", "ldl", "ldr"})
_HILO_USERS = frozenset({"mult", "multu", "div", "divu", "dmult", "dmultu", "ddiv", "ddivu", "mthi", "mtlo"})


def _load_delay(words: Sequence[int], vram: int) -> bool:
    """A nop between a GPR load and the next instruction, which reads the loaded register: MIPS I's load-delay slot,
    filled the way an assembler in MIPS I mode fills it. A nop a HI/LO hazard explains (mfhi/mflo just before, a
    multiply, divide or HI/LO move just after) is not one."""
    ins = [rabbitizer.Instruction(w, vram + 4 * i) for i, w in enumerate(words)]
    for i in range(len(ins) - 2):
        if ins[i].getOpcodeName() not in _LOADS or words[i + 1] != 0:
            continue
        if i and (ins[i - 1].isBranch() or ins[i - 1].isJump()):
            continue
        loaded, use, word = words[i] >> 16 & 31, ins[i + 2], words[i + 2]
        fields = {"Rs": word >> 21 & 31, "Rt": word >> 16 & 31, "Rd": word >> 11 & 31}
        if loaded and any(getattr(use, "reads" + f)() and r == loaded for f, r in fields.items()):
            hilo = use.getOpcodeName() in _HILO_USERS and ins[i - 1].getOpcodeName() in ("mfhi", "mflo")
            if not (i and hilo):
                return True
    return False


def foreign(words: Sequence[int], vram: int, toolchain: Mapping[str, Any]) -> bool:
    """Whether the code is not compiler output: it holds an instruction the toolchain never emits, FPU control moves
    without one of the conversions the toolchain pairs them with, or a filled load-delay slot when every ISA the
    toolchain targets interlocks loads (MIPS II and later: compiler and assembler leave the slot to the hardware)."""
    names = {rabbitizer.Instruction(w, vram + 4 * i).getOpcodeName() for i, w in enumerate(words)}
    return (
        bool(names & set(toolchain["never_emits"]))
        or (bool(names & _CONTROL) and not names & set(toolchain["fpu_control_with"]))
        or (min(toolchain.get("isa") or [1]) >= 2 and _load_delay(words, vram))
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


def _pairs(words: Sequence[int], symbols: Mapping[int, str] | None) -> dict[int, tuple[int, bool]]:
    """The `lui` and the low-half instruction that completes the address of a symbol: instruction index -> (address,
    whether it is the `lui`), also through an index add on the same base (a table access). With no symbols every
    address counts."""
    found: dict[int, tuple[int, bool]] = {}
    for i, word in enumerate(words):
        if word >> 26 != 0x0F or (word >> 16 & 31) == 0:
            continue
        register = word >> 16 & 31
        for j in range(i + 1, min(i + 9, len(words))):
            low = words[j]
            if low >> 26 in _LOW_USERS and low >> 21 & 31 == register:
                address = ((word & 0xFFFF) << 16) + _signed(low) & 0xFFFFFFFF
                if symbols is None or address in symbols:
                    found[i], found[j] = (address, True), (address, False)
                break
            if (_writes(words[j], register) and not _indexes(words[j], register)) or (words[j] >> 26 in (2, 3)):
                break
    return found


def _shape(words: Sequence[int], vram: int) -> tuple[int, ...]:
    """The words with every field that depends on where the routine and its neighbours sit masked: jump targets,
    branch offsets leaving the routine, and the halves of a symbol address. Two versions' copies of one library routine
    have the same shape though their addresses and symbol names differ."""
    pairs = _pairs(words, None)
    end = vram + 4 * len(words)
    out = []
    for i, word in enumerate(words):
        ins = rabbitizer.Instruction(word, vram + 4 * i)
        target = _target(ins, vram + 4 * i)
        if i in pairs or (target is not None and not vram <= target < end):
            word &= 0xFFFF0000 if ins.isBranch() or i in pairs else 0xFC000000
        out.append(word)
    return tuple(out)


def render(
    name: str,
    words: Sequence[int],
    vram: int,
    symbols: Mapping[int, str],
    functions: Mapping[int, str] = {},
    label: str | None = None,
) -> str | None:
    """Readable assembly of the words, one instruction per line, or None when a target outside the routine has no
    symbol. Internal targets get labels, external ones read as their symbol, or as `function+0xOFF` of the nearest
    of `functions` below it when it lands inside another routine (hand code shares tails between entry points);
    a `lui` and the low-half instruction that completes a symbol's address read `%hi(symbol)` and `%lo(symbol)`,
    also through an index add on the same base (`addu $at, $at, $t0`, a table access). Local labels are named after
    `label` (the routine's name unless given), so the versions of one routine named differently share them."""
    label = label or name
    code = [rabbitizer.Instruction(w, vram + 4 * i) for i, w in enumerate(words)]
    end = vram + 4 * len(words)
    targets = {i: t for i, ins in enumerate(code) if (t := _target(ins, vram + 4 * i)) is not None}
    outside = {t: symbols.get(t) or _inside(functions, t) for t in targets.values() if not vram <= t < end}
    if not all(outside.values()):
        return None
    labels = {(t - vram) // 4 for t in targets.values() if vram <= t < end}
    override = {
        i: text
        for i, address in _pairs(words, symbols).items()
        for text in (f"%hi({symbols[address[0]]})" if address[1] else f"%lo({symbols[address[0]]})",)
    }
    # o32 assembles a doubleword move as two words and an odd double FP register as a pair: code the ROM holds
    # as 64-bit instructions (an exception handler saving the whole context) is assembled with 64-bit registers
    wide = [".set gp=64", ".set fp=64"] if any(_wide(w) for w in words) else []
    lines = [".set noreorder", ".set noat", *wide, ".section .text", f".globl {name}", f"{name}:"]
    for i, ins in enumerate(code):
        if i in labels:
            lines.append(f".L{label}_{4 * i:03X}:")
        if i in override:
            text = ins.disassemble(immOverride=override[i])
        elif i in targets:
            t = targets[i]
            text = ins.disassemble(immOverride=f".L{label}_{t - vram:03X}" if vram <= t < end else outside[t])
        else:
            text = _operands(ins, words[i])
        lines.append("    " + " ".join(text.split()))
    return "\n".join(lines) + "\n"


def _inside(functions: Mapping[int, str], target: int) -> str | None:
    """`function+0xOFF` for a target inside the nearest function at or below it, or None when no function is."""
    below = [a for a in functions if a <= target]
    if not below:
        return None
    start = max(below)
    return functions[start] if start == target else f"{functions[start]}+0x{target - start:X}"


def _indexes(word: int, register: int) -> bool:
    """Whether the word is `addu register, register, x` (either operand order): an index added to a `%hi` base,
    which leaves the symbol's `%lo` still completing the address."""
    return (
        word >> 26 == 0
        and word & 0x3F == 0x21
        and word >> 11 & 31 == register
        and register in (word >> 21 & 31, word >> 16 & 31)
    )


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


_ENTRY = re.compile(r"(?:X?LEAF|NESTED)\(\s*([\w.$]+)\s*[,)]")


def scan(text: str) -> dict[str, str]:
    """Global definitions and sections in ordinary assembler source; local labels never become units. A function
    entry macro of sys/asm.h (`LEAF(name)`, `XLEAF(name)`, `NESTED(name, ...)`) defines its name as `.globl` does."""
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
        elif match := _ENTRY.match(line):
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


def _alike(a: Sequence[int], b: Sequence[int]) -> bool:
    """Whether two shapes read as one routine: the symbol files of the versions do not always cut a routine at the same
    word, so a few words at either end may differ."""
    from difflib import SequenceMatcher

    if a == b:
        return True
    if min(len(a), len(b)) < 16:
        return False
    match = SequenceMatcher(None, a, b, autojunk=False)
    return match.real_quick_ratio() >= 0.9 and match.quick_ratio() >= 0.9 and match.ratio() >= 0.9


def _stem(name: str, versions: Sequence[str]) -> str:
    """The routine's file stem: its name without the version suffix the symbol files add (`func_80200400_eu_x`)."""
    suffixes = sorted((("_" + v.replace("-", "_")) for v in versions), key=len, reverse=True)
    return next((name[: -len(x)] for x in suffixes if name.endswith(x) and len(name) > len(x)), name)


def plan(cfg: Config, held: Any) -> dict[str, bytes]:
    """Extract foreign library routines as source, one routine per file; existing source is never overwritten. A
    routine is a library stand-in or has lost it (no file owns it); it is rendered when every version holding it is
    foreign and renders. The copies of one routine that the versions name apart (`func_80200400_de`, `..._us`: the
    symbol files add a version suffix where the address differs) have the same shape, so they share one file named
    without the suffix, the lines the versions disagree on (their names, addresses, symbols) kept in version
    conditionals (`merge`). A routine that cannot render keeps (or gets back) its stand-in in a library file."""
    if not getattr(cfg.project, "libraries", ()):
        return {}
    writes: dict[str, bytes] = {}
    roms: dict[str, bytes] = {}
    by_path = {m.path: m for m in held.modules}
    lost = homes(cfg, held)

    def free(version: str, name: str) -> bool:
        """The version's owner of the name is a library stand-in, or nothing owns it: a `.s` defines the name in every
        version whose facts hold it, so every one of them must give up a stand-in to it."""
        owner = by_path.get(held.owned(version, name) or "")
        if owner is None:
            return name in lost
        return config.library(cfg.project, owner.path) and name in owner.stand_ins.get(version, ())

    candidates = [(m, n) for m in held.modules if config.library(cfg.project, m.path) for n in _stand_ins(m)]
    candidates += [(m, n) for n, m in lost.items()]
    versions = list(cfg.project.versions)
    # name -> (module, version -> (words, vram, symbol names, functions), version -> shape)
    routines: dict[str, tuple[Any, dict[str, tuple[Any, ...]], dict[str, tuple[int, ...]]]] = {}
    for module, name in candidates:
        if name in routines:
            continue
        toolchain = adapters.row(recipes.of(cfg, module.path).toolchain)
        facts: dict[str, tuple[Any, ...]] = {}
        shapes: dict[str, tuple[int, ...]] = {}
        for version in (v for v in versions if name in held.symbols[v]):
            found = _rom_words(cfg, held, roms, version, name) if free(version, name) else None
            if found is None or not foreign(found[0], found[1], toolchain):
                break
            table = sorted(held.symbols[version].items(), reverse=True)
            names = {r[0]: n for n, r in table}
            functions = {r[0]: n for n, r in table if r[3] == "func"}
            if render(name, found[0], found[1], names, functions) is None:
                break
            facts[version] = (found[0], found[1], names, functions)
            shapes[version] = _shape(found[0], found[1])
        else:
            if facts:
                routines[name] = (module, facts, shapes)

    def first(n: str) -> tuple[int, int, str]:
        return min(versions.index(v) for v in routines[n][1]), min(r[1] for r in routines[n][1].values()), n

    order = sorted(routines, key=first)
    grouped: set[str] = set()
    for name in order:
        if name in grouped:
            continue
        grouped.add(name)
        members = [name]
        taken = set(routines[name][1])
        own = set(routines[name][2].values())
        for other in order:
            if other in grouped or taken & set(routines[other][1]):
                continue
            if any(_alike(a, b) for a in own for b in routines[other][2].values()):
                grouped.add(other)
                members.append(other)
                taken |= set(routines[other][1])
        # a routine with a real name keeps it: the file is named after the first member that is not `func_<address>`
        lead = next((m for m in members if not m.startswith("func_")), name)
        stem = _stem(lead, versions) if len(members) > 1 else name
        module = routines[name][0]
        texts: dict[str, str] = {}
        for member in members:
            for version, (words, vram, names, functions) in routines[member][1].items():
                text = render(member, words, vram, names, functions, stem)
                assert text is not None
                texts[version] = text
        path = str(Path(module.path).with_name(stem + ".s"))
        raw = merge({v: texts[v] for v in versions if v in texts}).encode()
        existing = cfg.project.root / path
        if existing.exists() and existing.read_bytes() != raw:
            raise Refusal(Finding("library-asm.source", "would overwrite existing assembly source", path=path))
        writes[path] = raw
    return writes


def merge(texts: Mapping[str, str]) -> str:
    """One assembly source for the renders of a routine by version (project order): the lines every version shares
    as they are, each run the versions disagree on as `.if VERSION_A || VERSION_B` / `.elseif ...` / `.else` /
    `.endif` (the generated Makefile defines every version macro for the assembler as 1 or 0), the run most versions
    share under `.else`. `select` of the result gives every version its own render back, or this refuses."""
    if len(set(texts.values())) == 1:
        return next(iter(texts.values()))
    from difflib import SequenceMatcher

    lines = {v: t.splitlines() for v, t in texts.items()}
    groups: dict[str, list[str]] = {}
    for v, t in texts.items():
        groups.setdefault(t, []).append(v)
    base = lines[max(groups.values(), key=len)[0]]
    # the lines every version starts and ends with anchor first, so a difference reads as the lines it replaces
    # rather than as a line moved past an equal neighbour
    shortest = min(len(t) for t in lines.values())
    head = next((k for k in range(shortest) if len({t[k] for t in lines.values()}) > 1), shortest)
    tail = next((k for k in range(shortest - head) if len({t[-1 - k] for t in lines.values()}) > 1), shortest - head)
    maps: dict[str, dict[int, int]] = {}
    for v, other in lines.items():
        found = {k: k for k in range(head)} | {len(base) - 1 - k: len(other) - 1 - k for k in range(tail)}
        middle = SequenceMatcher(None, base[head : len(base) - tail], other[head : len(other) - tail], autojunk=False)
        for a, b, size in middle.get_matching_blocks():
            found.update((head + a + k, head + b + k) for k in range(size))
        maps[v] = found
    anchors = [i for i in range(len(base)) if all(i in m for m in maps.values())]
    order = list(lines)
    out: list[str] = []
    previous = {v: -1 for v in lines}
    for anchor in [*anchors, None]:
        runs = {
            v: lines[v][previous[v] + 1 : (maps[v][anchor] if anchor is not None else len(lines[v]))] for v in lines
        }
        if len({tuple(r) for r in runs.values()}) > 1:
            by_run: dict[tuple[str, ...], list[str]] = {}
            for v, own in runs.items():
                by_run.setdefault(tuple(own), []).append(v)
            # the versions with no lines here take the `.else` (none is written), else the run most versions share
            rest = by_run.get(()) or max(by_run.values(), key=len)
            rest_run = next(run for run, held in by_run.items() if held is rest)
            branches = sorted(
                ((run, held) for run, held in by_run.items() if held is not rest), key=lambda i: order.index(i[1][0])
            )
            for k, (run, held) in enumerate(branches):
                out.append((".if " if k == 0 else ".elseif ") + " || ".join(macro(v) for v in held))
                out.extend(run)
            if rest_run:
                out.append(".else")
                out.extend(rest_run)
            out.append(".endif")
        else:
            out.extend(next(iter(runs.values())))
        if anchor is not None:
            out.append(base[anchor])
            previous = {v: maps[v][anchor] for v in lines}
    text = "\n".join(out) + "\n"
    for v, wanted in texts.items():
        if select(text, v, list(texts)) != wanted:
            raise Refusal(Finding("library-asm.merge", f"{config.sentence('library-asm.merge')}: {v}"))
    return text


def defsyms(versions: Sequence[str], version: str) -> tuple[str, ...]:
    """The assembler flags defining every version macro, 1 for the version and 0 for the others (what the generated
    Makefile's ASDEFS passes), so `merge` conditionals assemble outside make too."""
    return tuple(arg for v in versions for arg in ("--defsym", f"{macro(v)}={int(v == version)}"))


def select(text: str, version: str, versions: Sequence[str]) -> str:
    """The lines the assembler reads for one version of a `merge` result: its version macros 1 for the version, 0
    for the others; one level of `.if` / `.elseif` / `.else` / `.endif` over `||` of macros."""
    value = {macro(v): v == version for v in versions}
    out: list[str] = []
    taking, taken, inside = True, False, False
    for line in text.splitlines():
        word = line.split(" ", 1)
        if word[0] in (".if", ".elseif"):
            hit = not taken and any(value[m.strip()] for m in word[1].split("||"))
            inside, taking, taken = True, hit, taken or hit
        elif word[0] == ".else" and inside:
            taking, taken = not taken, True
        elif word[0] == ".endif" and inside:
            inside, taking, taken = False, True, False
        elif taking:
            out.append(line)
    return "\n".join(out) + "\n"


def _stand_ins(module: Any) -> list[str]:
    return sorted({n for names in module.stand_ins.values() for n in names})
